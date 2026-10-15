"""LLM documentation check: redact, prompt, validate, ground, cache.

The checker only reports which documentation items are present, missing or unclear and
drafts a request for missing items. It never approves or denies a claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from claimguard import config
from claimguard.data.notes import load_checklists
from claimguard.llm.provider import LLMError, LLMProvider, QuotaExhausted
from claimguard.llm.redact import redact

DEFAULT_PROMPT_VERSION = "v3"
_DECISION_WORDS = re.compile(
    r"\b(approve|approved|approves|deny|denied|denies|reject|rejected|rejects)\b", re.IGNORECASE
)
_CORRECTION = (
    "\n\nYour previous answer was invalid: {errors}. Return only the corrected JSON object."
)


class DocItem(BaseModel):
    """One checklist item as judged by the LLM."""

    item_id: str
    status: Literal["present", "missing", "unclear"]
    evidence: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class DocCheckOutput(BaseModel):
    """The JSON schema the LLM must return."""

    items: list[DocItem]
    summary: str
    draft_request: str = ""


@lru_cache(maxsize=8)
def load_prompt(version: str) -> dict[str, Any]:
    """Load a versioned prompt from `prompts/doc_check_<version>.yaml`."""
    path = config.PROMPTS_DIR / f"doc_check_{version}.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def render_prompt(version: str, checklist: list[dict[str, str]], note: str) -> tuple[str, str]:
    """System and user messages for a prompt version, checklist and (redacted) note."""
    prompt = load_prompt(version)
    lines = "\n".join(prompt["checklist_format"].format(**item) for item in checklist)
    examples = ""
    if prompt.get("examples"):
        blocks = [
            f"Example {i}:\nChecklist:\n{ex['checklist'].strip()}\n"
            f"<note>\n{ex['note']}\n</note>\nOutput:\n{ex['output']}"
            for i, ex in enumerate(prompt["examples"], start=1)
        ]
        examples = "Worked examples (not the note to check):\n\n" + "\n\n".join(blocks)
    user = prompt["user_template"].format(checklist=lines, note=note, examples=examples)
    return prompt["system"].strip(), user


def parse_json(text: str) -> dict[str, Any]:
    """Parse model output defensively: strip code fences and surrounding prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip("\"'").strip().rstrip(".").lower()


def is_grounded(evidence: str | None, note: str) -> bool:
    """Evidence is a (whitespace- and case-normalised) substring of the redacted note."""
    if not evidence or not evidence.strip():
        return False
    return _normalize(evidence) in _normalize(note)


def validate_output(
    raw: dict[str, Any], checklist: list[dict[str, str]], note: str
) -> tuple[DocCheckOutput | None, list[str], dict[str, int]]:
    """Schema + semantic checks. Returns (output or None, errors, grounding counts)."""
    grounding = {"present": 0, "grounded": 0}
    try:
        out = DocCheckOutput.model_validate(raw)
    except ValidationError as exc:
        return None, [f"schema: {exc.error_count()} error(s)"], grounding
    allowed = {i["id"] for i in checklist}
    errors = []
    seen = set()
    for item in out.items:
        if item.item_id not in allowed:
            errors.append(f"unknown item_id {item.item_id!r}")
        if item.item_id in seen:
            errors.append(f"duplicate item_id {item.item_id!r}")
        seen.add(item.item_id)
        if item.status == "present":
            grounding["present"] += 1
            if is_grounded(item.evidence, note):
                grounding["grounded"] += 1
            else:
                errors.append(f"evidence for {item.item_id!r} is not a quote from the note")
    missing_ids = allowed - seen
    if missing_ids:
        errors.append(f"items not answered: {sorted(missing_ids)}")
    if _DECISION_WORDS.search(out.summary) or _DECISION_WORDS.search(out.draft_request):
        errors.append("approve/deny language is not allowed")
    return (out if not errors else None), errors, grounding


def note_hash(text: str) -> str:
    """Stable hash of the redacted note (used for storage and the cache key)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ResponseCache:
    """On-disk cache of raw LLM responses keyed by (note hash, prompt version, model)."""

    def __init__(self, directory: Path = config.LLM_CACHE_DIR) -> None:
        self.directory = directory

    def _path(self, key: tuple[str, str, str, int]) -> Path:
        digest = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: tuple[str, str, str, int]) -> dict[str, Any] | None:
        path = self._path(key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def put(self, key: tuple[str, str, str, int], value: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps(value), encoding="utf-8")


def check_documentation(
    note: str,
    category: str,
    provider: LLMProvider,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    cache: ResponseCache | None = None,
) -> dict[str, Any]:
    """Run the documentation check for one note. Never raises for LLM failures.

    Returns a dict with `llm_status` in {ok, invalid_output, unavailable, quota_exhausted}.
    """
    checklist = load_checklists()[category]
    redacted = redact(note)
    nhash = note_hash(redacted)
    system, user = render_prompt(prompt_version, checklist, redacted)
    result: dict[str, Any] = {
        "llm_status": "ok",
        "model": provider.model,
        "prompt_version": prompt_version,
        "note_hash": nhash,
        "items": [],
        "summary": "",
        "draft_request": "",
        "needs_review": [],
        "latency_s": 0.0,
        "attempts": 0,
        "errors": [],
        "grounding": {"present": 0, "grounded": 0},
    }
    for attempt in range(2):
        key = (nhash, prompt_version, provider.model, attempt)
        cached = cache.get(key) if cache else None
        if cached is None:
            prompt_user = (
                user
                if attempt == 0
                else user + _CORRECTION.format(errors="; ".join(result["errors"][-3:]))
            )
            try:
                resp = provider.complete(system, prompt_user)
            except QuotaExhausted as exc:
                result.update(llm_status="quota_exhausted", errors=[str(exc)])
                return result
            except LLMError as exc:
                result.update(llm_status="unavailable", errors=[str(exc)])
                return result
            cached = {"text": resp.text, "latency_s": resp.latency_s}
            if cache:
                cache.put(key, cached)
        result["attempts"] = attempt + 1
        result["latency_s"] += float(cached["latency_s"])
        try:
            raw = parse_json(cached["text"])
        except (json.JSONDecodeError, TypeError):
            result["errors"].append("output is not valid JSON")
            continue
        out, errors, grounding = validate_output(raw, checklist, redacted)
        if attempt == 0:
            result["grounding"] = grounding
        if out is None:
            result["errors"].extend(errors)
            continue
        items = [i.model_dump() for i in out.items]
        result.update(
            items=items,
            summary=out.summary,
            draft_request=out.draft_request,
            needs_review=[
                i["item_id"]
                for i in items
                if i["status"] == "unclear" or i["confidence"] < config.LLM_LOW_CONFIDENCE
            ],
        )
        return result
    result["llm_status"] = "invalid_output"
    return result
