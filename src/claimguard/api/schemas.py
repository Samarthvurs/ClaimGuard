"""Pydantic request and response models for the API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

Category = Literal["imaging", "surgery", "therapy", "diagnostic", "routine"]
Decision = Literal["accept_flag", "dismiss", "escalate"]


class ClaimCreate(BaseModel):
    """A claim and its clinical note. All data must be synthetic."""

    external_id: str | None = Field(default=None, max_length=64)
    age: int = Field(ge=0, le=120)
    sex: Literal["M", "F", "U"]
    payer: str = Field(min_length=1, max_length=64)
    encounter_class: str = Field(min_length=1, max_length=32)
    procedure_category: Category
    provider_specialty: str = Field(default="UNKNOWN", max_length=64)
    procedure_code: str = Field(min_length=1, max_length=32)
    procedure_description: str = Field(min_length=1, max_length=256)
    condition_code: str = Field(default="NONE", max_length=32)
    condition_description: str = Field(default="No linked condition", max_length=256)
    service_date: date
    claim_cost: float = Field(ge=0)
    prior_auth_required: bool
    prior_auth_on_file: bool
    documents_attached_count: int = Field(ge=0, le=50)
    min_documents: int = Field(ge=0, le=20)
    days_to_submission: int = Field(ge=0, le=3650)
    provider_prior_denial_rate: float = Field(default=0.2, ge=0, le=1)
    patient_prior_denials: int = Field(default=0, ge=0)
    note: str = Field(min_length=1, max_length=20_000)


class ReviewCreate(BaseModel):
    """A human reviewer's decision on an assessment's flags."""

    decision: Decision
    comment: str = Field(default="", max_length=2000)


class AssessOptions(BaseModel):
    """Options for running an assessment."""

    use_llm: bool = True


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
