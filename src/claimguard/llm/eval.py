"""Evaluate prompt versions on labelled synthetic notes.

CLI: python -m claimguard.llm.eval [--n 100] [--versions v1 v2 v3]

Positive class = "item needs attention": ground truth `missing`; a prediction of
`missing` or `unclear` counts as flagging the item. Results are written to
`reports/prompt_eval.json` and `.md`. Runs are resumable through the on-disk cache.
Mock runs are for tests only and are never written to `reports/`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from claimguard import config
from claimguard.data.notes import load_notes
from claimguard.llm.checker import ResponseCache, check_documentation
from claimguard.llm.provider import GeminiProvider, LLMError, LLMProvider, get_provider

VERSIONS = ["v1", "v2", "v3"]
FLAGGED = {"missing", "unclear"}


def select_notes(notes: list[dict[str, Any]], n: int = 100, seed: int = config.SEED) -> list:
    """All adversarial notes plus a category-stratified sample of the others.

    Seeded sampling for reproducibility, not security.
    """
    rng = random.Random(seed)  # nosec B311
    adversarial = [r for r in notes if r["adversarial"]]
    others = [r for r in notes if not r["adversarial"]]
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for rec in others:
        by_cat.setdefault(rec["category"], []).append(rec)
    remaining = max(0, n - len(adversarial))
    cats = sorted(by_cat)
    chosen: list[dict[str, Any]] = []
    for i, cat in enumerate(cats):
        quota = remaining // len(cats) + (1 if i < remaining % len(cats) else 0)
        chosen.extend(rng.sample(by_cat[cat], min(quota, len(by_cat[cat]))))
    return adversarial + chosen


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {
        "precision": None if p is None else round(p, 4),
        "recall": None if r is None else round(r, 4),
        "f1": None if f is None else round(f, 4),
        "support_missing": tp + fn,
    }


def score_version(records: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics for one prompt version."""
    counts: dict[str, list[int]] = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    invalid = present = grounded = 0
    adv_total = adv_success = 0
    latencies = []
    for rec, res in zip(records, results, strict=True):
        latencies.append(res["latency_s"])
        present += res["grounding"]["present"]
        grounded += res["grounding"]["grounded"]
        if res["llm_status"] != "ok":
            invalid += 1
            if rec["adversarial"]:
                adv_total += 1
            continue
        pred = {i["item_id"]: i["status"] for i in res["items"]}
        if rec["adversarial"]:
            adv_total += 1
            if any(
                truth == "missing" and pred.get(item) == "present"
                for item, truth in rec["ground_truth"].items()
            ):
                adv_success += 1
        for item, truth in rec["ground_truth"].items():
            c = counts.setdefault(item, [0, 0, 0])
            flagged = pred.get(item, "unclear") in FLAGGED
            if truth == "missing" and flagged:
                c[0] += 1
            elif truth == "present" and flagged:
                c[1] += 1
            elif truth == "missing" and not flagged:
                c[2] += 1
    for c in counts.values():
        total["tp"] += c[0]
        total["fp"] += c[1]
        total["fn"] += c[2]
    n = len(records)
    return {
        "n_notes": n,
        "n_adversarial": adv_total,
        "per_item": {k: _prf(*v) for k, v in sorted(counts.items())},
        "micro": _prf(total["tp"], total["fp"], total["fn"]),
        "invalid_output_rate": round(invalid / n, 4) if n else None,
        "grounding_rate": round(grounded / present, 4) if present else None,
        "injection_success_rate": round(adv_success / adv_total, 4) if adv_total else None,
        "avg_latency_s": round(sum(latencies) / n, 3) if n else None,
    }


def run_eval(
    provider: LLMProvider,
    records: list[dict[str, Any]],
    versions: list[str],
    cache: ResponseCache,
) -> dict[str, Any]:
    """Run each version over the records; stop cleanly if the quota is exhausted."""
    out: dict[str, Any] = {"model": provider.model, "versions": {}}
    for version in versions:
        results = []
        for rec in records:
            res = check_documentation(rec["note"], rec["category"], provider, version, cache)
            if res["llm_status"] == "quota_exhausted":
                out["versions"][version] = {
                    "status": "TBD (quota exhausted)",
                    "completed_notes": len(results),
                }
                out["stopped_early"] = True
                return out
            results.append(res)
        out["versions"][version] = {"status": "complete", **score_version(records, results)}
    return out


def to_markdown(report: dict[str, Any]) -> str:
    """Comparison table for `reports/prompt_eval.md`."""
    lines = [
        "# Prompt evaluation",
        "",
        f"Model: `{report['model']}`. Notes: {report['n_notes']} synthetic notes "
        f"({report['n_adversarial']} adversarial) with template ground truth. "
        "Synthetic data only; positive class = item missing.",
        "",
        "| Version | Status | Micro P | Micro R | Micro F1 | Invalid output | Grounding | "
        "Injection success | Avg latency (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for version, m in report["versions"].items():
        if m["status"] != "complete":
            lines.append(f"| {version} | {m['status']} | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")
            continue
        micro = m["micro"]
        lines.append(
            f"| {version} | complete | {micro['precision']} | {micro['recall']} | {micro['f1']} | "
            f"{m['invalid_output_rate']} | {m['grounding_rate']} | "
            f"{m['injection_success_rate']} | {m['avg_latency_s']} |"
        )
    return "\n".join(lines) + "\n"


def resolve_model(settings: config.Settings) -> str:
    """Verify the key with one tiny call; if the model name is rejected, pick another."""
    provider = GeminiProvider(settings.gemini_api_key, settings.llm_model, max_calls=3)
    try:
        provider.complete("Reply with the single word OK.", "OK?")
        return settings.llm_model
    except LLMError as exc:
        status = getattr(exc.__cause__, "status_code", None)
        if status not in (400, 404):
            raise
    names = [m.id.split("/")[-1] for m in provider.client.models.list()]
    for pattern in ("flash-lite", "flash"):
        candidates = sorted(n for n in names if pattern in n and "gemini" in n)
        if candidates:
            print(f"LLM_MODEL {settings.llm_model!r} rejected; using {candidates[-1]!r}")
            return candidates[-1]
    raise RuntimeError("no suitable Gemini model found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate documentation-check prompts")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--versions", nargs="+", default=VERSIONS)
    parser.add_argument("--out-dir", type=Path, default=config.REPORTS_DIR)
    args = parser.parse_args()

    settings = config.get_settings()
    print(f"Gemini key configured: {'yes' if settings.llm_configured else 'no'}")
    if not settings.llm_configured:
        print("LLM not configured; nothing evaluated. Add GEMINI_API_KEY to .env and rerun.")
        return
    model = resolve_model(settings)
    settings = config.Settings(**{**settings.__dict__, "llm_model": model})
    provider, _ = get_provider(settings)
    records = select_notes(load_notes(), n=args.n)
    started = time.time()
    report = run_eval(provider, records, args.versions, ResponseCache())
    report.update(
        n_notes=len(records),
        n_adversarial=sum(r["adversarial"] for r in records),
        wall_time_s=round(time.time() - started, 1),
        data_note="Synthetic notes generated from templates; ground truth is known by "
        "construction. Not real clinical documentation.",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prompt_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "prompt_eval.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
