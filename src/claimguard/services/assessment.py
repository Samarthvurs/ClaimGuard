"""The single place that combines ML score, SHAP factors, redaction, LLM check and risk status.

The API and the dashboard both call these functions. The system only flags risk and
suggests missing documentation; it never approves or denies a claim. Decisions are made
by human reviewers.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from claimguard import config
from claimguard.api.db import Assessment, Claim, ClaimNote, DocCheck, Review
from claimguard.llm.checker import check_documentation, note_hash
from claimguard.llm.provider import LLMProvider, default_provider
from claimguard.llm.redact import redact
from claimguard.ml.features import RAW_FIELDS
from claimguard.ml.serve import get_scorer

log = logging.getLogger(__name__)

RISK_STATUSES = ("LOW_RISK", "NEEDS_REVIEW", "HIGH_RISK")
FLAGGED_STATUSES = ("NEEDS_REVIEW", "HIGH_RISK")


class NotFound(LookupError):
    """Requested record does not exist."""


def risk_status(probability: float, doc_items_need_attention: bool = False) -> str:
    """Map a denial probability to a review band (config thresholds).

    Missing or unclear documentation lifts a LOW_RISK claim to NEEDS_REVIEW.
    """
    if probability >= config.RISK_HIGH_MIN:
        return "HIGH_RISK"
    if probability >= config.RISK_LOW_MAX or doc_items_need_attention:
        return "NEEDS_REVIEW"
    return "LOW_RISK"


def create_claim(session: Session, data: dict[str, Any]) -> Claim:
    """Store a claim and its redacted note. The raw note is never stored."""
    data = dict(data)
    note = data.pop("note")
    external_id = data.pop("external_id", None) or f"API-{_next_id(session):06d}"
    if session.scalar(select(Claim).where(Claim.external_id == external_id)):
        raise ValueError(f"claim {external_id} already exists")
    service_date = str(data.pop("service_date"))[:10]
    claim = Claim(
        external_id=external_id,
        payer=data["payer"],
        procedure_category=data["procedure_category"],
        procedure_description=data.pop("procedure_description"),
        condition_description=data.pop("condition_description"),
        service_date=service_date,
        fields={k: data[k] for k in RAW_FIELDS},
        status="NEW",
    )
    redacted = redact(note)
    claim.note = ClaimNote(redacted_text=redacted, note_hash=note_hash(redacted))
    session.add(claim)
    session.commit()
    log.info("created claim %s", external_id)
    return claim


def _next_id(session: Session) -> int:
    return int(session.scalar(select(func.count(Claim.id))) or 0) + 1


def get_claim(session: Session, claim_id: int) -> Claim:
    """Fetch a claim or raise NotFound."""
    claim = session.get(Claim, claim_id)
    if claim is None:
        raise NotFound(f"claim {claim_id} not found")
    return claim


def assess_claim(
    session: Session,
    claim_id: int,
    use_llm: bool = True,
    provider: LLMProvider | None = None,
) -> Assessment:
    """Score the claim, run the documentation check if configured, store the result."""
    claim = get_claim(session, claim_id)
    scorer = get_scorer()
    ml = scorer.score(claim.fields)

    llm: dict[str, Any] | None = None
    if not use_llm:
        llm_status = "skipped"
    else:
        if provider is None:
            provider, configured = default_provider()
        else:
            configured = "ok"
        if configured == "not_configured":
            llm_status = "not_configured"
        else:
            llm = check_documentation(
                claim.note.redacted_text if claim.note else "",
                claim.procedure_category,
                provider,
            )
            llm_status = llm["llm_status"]
            if llm_status == "quota_exhausted":
                llm_status = "unavailable"

    attention = bool(
        llm is not None
        and llm_status == "ok"
        and (any(i["status"] != "present" for i in llm["items"]) or llm["needs_review"])
    )
    status = risk_status(ml["denial_probability"], attention)
    assessment = Assessment(
        claim_id=claim.id,
        denial_probability=ml["denial_probability"],
        likely_denied=ml["likely_denied"],
        risk_status=status,
        predicted_reason=ml["predicted_reason"],
        factors=ml["factors"],
        llm_status=llm_status,
        model_info={
            "ml_model": scorer.meta.get("model_name"),
            "ml_trained_at": scorer.meta.get("trained_at"),
            "threshold": ml["threshold"],
            "reason_probabilities": ml["reason_probabilities"],
        },
    )
    if llm is not None and llm_status in ("ok", "invalid_output"):
        assessment.doc_check = DocCheck(
            llm_status=llm["llm_status"],
            model=llm["model"],
            prompt_version=llm["prompt_version"],
            items=llm["items"],
            summary=llm["summary"],
            draft_request=llm["draft_request"],
            needs_review=llm["needs_review"],
        )
    claim.status = status
    session.add(assessment)
    session.commit()
    log.info("assessed claim %s: %s (llm=%s)", claim.external_id, status, llm_status)
    return assessment


def review_assessment(session: Session, assessment_id: int, decision: str, comment: str) -> Review:
    """Record a human reviewer's decision (accept_flag, dismiss or escalate)."""
    assessment = session.get(Assessment, assessment_id)
    if assessment is None:
        raise NotFound(f"assessment {assessment_id} not found")
    review = Review(assessment_id=assessment_id, decision=decision, comment=comment)
    session.add(review)
    session.commit()
    return review


def list_claims(
    session: Session,
    payer: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Claim], int]:
    """Claims filtered by payer and status, newest first, with the total count."""
    query = select(Claim)
    if payer:
        query = query.where(Claim.payer == payer)
    if status:
        query = query.where(Claim.status == status)
    total = int(session.scalar(select(func.count()).select_from(query.subquery())) or 0)
    rows = session.scalars(query.order_by(Claim.id.desc()).limit(limit).offset(offset)).all()
    return list(rows), total


def latest_assessment(claim: Claim) -> Assessment | None:
    """Most recent assessment for a claim, if any."""
    return claim.assessments[-1] if claim.assessments else None


def claim_summary(claim: Claim) -> dict[str, Any]:
    """Compact claim view for lists."""
    a = latest_assessment(claim)
    return {
        "id": claim.id,
        "external_id": claim.external_id,
        "payer": claim.payer,
        "procedure_category": claim.procedure_category,
        "procedure_description": claim.procedure_description,
        "service_date": claim.service_date,
        "status": claim.status,
        "denial_probability": a.denial_probability if a else None,
        "predicted_reason": a.predicted_reason if a else None,
        "llm_status": a.llm_status if a else None,
        "latest_assessment_id": a.id if a else None,
    }


def assessment_detail(a: Assessment) -> dict[str, Any]:
    """Full assessment view including the doc check and reviews."""
    doc = a.doc_check
    return {
        "id": a.id,
        "claim_id": a.claim_id,
        "denial_probability": a.denial_probability,
        "likely_denied": a.likely_denied,
        "risk_status": a.risk_status,
        "predicted_reason": a.predicted_reason,
        "factors": a.factors,
        "llm_status": a.llm_status,
        "model_info": a.model_info,
        "doc_check": None
        if doc is None
        else {
            "llm_status": doc.llm_status,
            "model": doc.model,
            "prompt_version": doc.prompt_version,
            "items": doc.items,
            "summary": doc.summary,
            "draft_request": doc.draft_request,
            "needs_review": doc.needs_review,
        },
        "reviews": [
            {
                "id": r.id,
                "decision": r.decision,
                "comment": r.comment,
                "created_at": r.created_at.isoformat(),
            }
            for r in a.reviews
        ],
        "created_at": a.created_at.isoformat(),
        "disclaimer": "Risk flag only. The system does not approve or deny claims; "
        "a human reviewer decides.",
    }


def claim_detail(claim: Claim) -> dict[str, Any]:
    """Claim with its fields, redacted note and latest assessment."""
    a = latest_assessment(claim)
    return {
        **claim_summary(claim),
        "condition_description": claim.condition_description,
        "fields": claim.fields,
        "note_redacted": claim.note.redacted_text if claim.note else None,
        "latest_assessment": assessment_detail(a) if a else None,
        "n_assessments": len(claim.assessments),
    }


def summary(session: Session) -> dict[str, Any]:
    """Aggregates over each claim's latest assessment, plus reviewer decisions."""
    claims = session.scalars(select(Claim)).all()
    latest = [a for c in claims if (a := latest_assessment(c)) is not None]
    n = len(latest)
    flagged = [a for a in latest if a.risk_status in FLAGGED_STATUSES]
    by_payer: dict[str, list[float]] = {}
    by_reason: dict[str, int] = {}
    for a in latest:
        by_payer.setdefault(a.claim.payer, []).append(a.denial_probability)
        if a.likely_denied:
            by_reason[a.predicted_reason] = by_reason.get(a.predicted_reason, 0) + 1
    reviews = session.scalars(select(Review)).all()
    decisions: dict[str, int] = {}
    for r in reviews:
        decisions[r.decision] = decisions.get(r.decision, 0) + 1
    n_reviews = len(reviews)
    return {
        "claims_total": len(claims),
        "claims_assessed": n,
        "flagged": len(flagged),
        "pct_flagged": round(len(flagged) / n, 4) if n else None,
        "avg_risk": round(sum(a.denial_probability for a in latest) / n, 4) if n else None,
        "status_counts": {s: sum(a.risk_status == s for a in latest) for s in RISK_STATUSES},
        "risk_by_payer": {
            p: {"n": len(v), "avg_risk": round(sum(v) / len(v), 4)}
            for p, v in sorted(by_payer.items())
        },
        "predicted_reason_counts": by_reason,
        "llm_status_counts": _count(a.llm_status for a in latest),
        "reviews_total": n_reviews,
        "review_decisions": decisions,
        "reviewer_acceptance_rate": round(decisions.get("accept_flag", 0) / n_reviews, 4)
        if n_reviews
        else None,
        "data_note": "Synthetic data only. Decisions are made by humans.",
    }


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out
