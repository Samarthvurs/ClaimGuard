"""API routes. Every route except /health requires the `X-API-Key` header."""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.orm import Session

from claimguard import config
from claimguard.api.db import session_factory
from claimguard.api.schemas import AssessOptions, ClaimCreate, ReviewCreate
from claimguard.services import assessment as svc

router = APIRouter()


def get_session() -> Iterator[Session]:
    """One database session per request."""
    session = session_factory()()
    try:
        yield session
    finally:
        session.close()


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """Demo protection only: compare the header to `API_KEY` from the environment."""
    expected = config.get_settings().api_key
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")


SessionDep = Annotated[Session, Depends(get_session)]
protected = [Depends(require_api_key)]


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@router.post("/claims", status_code=201, dependencies=protected)
def create_claim(body: ClaimCreate, session: SessionDep) -> dict[str, Any]:
    """Create a claim with its note (the note is redacted before storage)."""
    try:
        claim = svc.create_claim(session, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return svc.claim_detail(claim)


@router.get("/claims", dependencies=protected)
def list_claims(
    session: SessionDep,
    payer: str | None = None,
    status: str | None = Query(default=None, pattern="^(NEW|LOW_RISK|NEEDS_REVIEW|HIGH_RISK)$"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List claims with optional payer and status filters and pagination."""
    rows, total = svc.list_claims(session, payer, status, limit, offset)
    return {
        "items": [svc.claim_summary(c) for c in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/claims/{claim_id}", dependencies=protected)
def get_claim(claim_id: int, session: SessionDep) -> dict[str, Any]:
    """Claim detail with the latest assessment."""
    try:
        return svc.claim_detail(svc.get_claim(session, claim_id))
    except svc.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/claims/{claim_id}/assess", dependencies=protected)
def assess_claim(
    claim_id: int, session: SessionDep, options: AssessOptions | None = None
) -> dict[str, Any]:
    """Run the assessment: ML risk + SHAP factors + documentation check if configured."""
    use_llm = options.use_llm if options else True
    try:
        result = svc.assess_claim(session, claim_id, use_llm=use_llm)
    except svc.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return svc.assessment_detail(result)


@router.post("/assessments/{assessment_id}/review", status_code=201, dependencies=protected)
def review(assessment_id: int, body: ReviewCreate, session: SessionDep) -> dict[str, Any]:
    """Record a reviewer's decision: accept_flag, dismiss or escalate."""
    try:
        r = svc.review_assessment(session, assessment_id, body.decision, body.comment)
    except svc.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "id": r.id,
        "assessment_id": r.assessment_id,
        "decision": r.decision,
        "comment": r.comment,
        "created_at": r.created_at.isoformat(),
    }


@router.get("/summary", dependencies=protected)
def summary(session: SessionDep) -> dict[str, Any]:
    """Aggregates across claims, assessments and reviews."""
    return svc.summary(session)
