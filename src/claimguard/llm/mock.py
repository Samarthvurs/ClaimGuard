"""Deterministic mock LLM provider for tests and for running without a Gemini key.

By default it reads the checklist ids and the note from the prompt and marks an item
present when a sentence contains one of the item's keywords, quoting that sentence.
Tests can instead queue canned raw responses (fixtures) or errors.
"""

from __future__ import annotations

import json
import re
from collections import deque

from claimguard.llm.provider import LLMResponse

KEYWORDS: dict[str, list[str]] = {
    "clinical_indication": ["indicat", "ordered to evaluate", "reason for study", "workup"],
    "prior_conservative_treatment": [
        "previously tried",
        "conservative management",
        "prior treatment",
        "trial of",
        "managed with",
    ],
    "symptom_duration": [
        "symptoms have been present",
        "onset of symptoms",
        "duration of complaint",
        "symptoms lasting",
        "ongoing symptoms",
    ],
    "physical_exam_findings": ["on exam", "physical exam", "exam findings", "examination reveals"],
    "diagnosis_confirmation": ["confirmed", "established the diagnosis"],
    "failed_conservative_treatment": [
        "failed conservative",
        "did not resolve",
        "has not improved",
        "exhausted",
        "did not respond",
    ],
    "surgical_risk_assessment": [
        "risk assessment",
        "cleared for surgery",
        "surgical risk",
        "pre-op evaluation",
    ],
    "informed_consent": ["consent"],
    "functional_limitation": [
        "limits walking",
        "unable to perform",
        "functional limitation",
        "interfere with",
        "limitation in activities",
    ],
    "treatment_plan": [
        "plan:",
        "treatment plan",
        "will continue",
        "planned course",
        "therapy plan",
    ],
    "functional_goals": ["goal", "aim to improve", "target outcome"],
    "progress_measurement": ["baseline", "measured", "progress measured", "recorded"],
    "relevant_history": [
        "history notable",
        "relevant history",
        "past medical history",
        "history support",
        "documented history",
    ],
    "ordering_rationale": [
        "chosen because",
        "ordered to decide",
        "result will determine",
        "ordered to rule out",
        "testing is needed",
    ],
    "visit_reason": [
        "visit for",
        "presents today",
        "reason for visit",
        "seen today for",
        "encounter purpose",
    ],
    "risk_factors": ["risk factor", "eligible for screening", "risk profile"],
    "follow_up_plan": ["follow up", "follow-up", "return visit", "reassess"],
}

_INJECTION_HINTS = ("ignore", "disregard", "override", "mark every", "mark all", "test mode")


def _sentences(note: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", note) if s.strip()]


def heuristic_response(user_prompt: str) -> str:
    """Build a deterministic JSON answer from the prompt's checklist and note."""
    checklist_part = user_prompt.split("Checklist:")[-1].split("<note>")[0]
    item_ids = re.findall(r"^- ([a-z_]+):", checklist_part, flags=re.MULTILINE)
    notes = re.findall(r"<note>\s*(.*?)\s*</note>", user_prompt, flags=re.DOTALL)
    note = notes[-1] if notes else ""  # the last note block; earlier ones are examples
    sentences = [s for s in _sentences(note) if not any(h in s.lower() for h in _INJECTION_HINTS)]
    items = []
    for item_id in item_ids:
        hit = next(
            (s for s in sentences if any(k in s.lower() for k in KEYWORDS.get(item_id, []))),
            None,
        )
        items.append(
            {
                "item_id": item_id,
                "status": "present" if hit else "missing",
                "evidence": hit,
                "confidence": 0.9 if hit else 0.8,
            }
        )
    missing = [i["item_id"] for i in items if i["status"] == "missing"]
    return json.dumps(
        {
            "items": items,
            "summary": f"{len(items) - len(missing)} of {len(items)} items documented.",
            "draft_request": (
                "Please provide documentation for: " + ", ".join(missing) + "." if missing else ""
            ),
        }
    )


class MockProvider:
    """Deterministic provider. Queue `responses` (str or Exception) to script behaviour."""

    name = "mock"

    def __init__(self, responses: list[str | Exception] | None = None, model: str = "mock-v1"):
        self.model = model
        self.queue: deque[str | Exception] = deque(responses or [])
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        if self.queue:
            item = self.queue.popleft()
            if isinstance(item, Exception):
                raise item
            return LLMResponse(text=item, latency_s=0.0)
        return LLMResponse(text=heuristic_response(user), latency_s=0.0)
