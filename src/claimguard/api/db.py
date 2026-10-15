"""SQLite storage via SQLAlchemy 2.x. Tables are created at startup.

Only redacted note text and its hash are stored; raw notes are never persisted.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from claimguard import config


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Claim(Base):
    __tablename__ = "claims"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    payer: Mapped[str] = mapped_column(String(64), index=True)
    procedure_category: Mapped[str] = mapped_column(String(32))
    procedure_description: Mapped[str] = mapped_column(String(256))
    condition_description: Mapped[str] = mapped_column(String(256))
    service_date: Mapped[str] = mapped_column(String(10))
    fields: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="NEW", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    note: Mapped[ClaimNote | None] = relationship(back_populates="claim", uselist=False)
    assessments: Mapped[list[Assessment]] = relationship(
        back_populates="claim", order_by="Assessment.id"
    )


class ClaimNote(Base):
    __tablename__ = "claim_notes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), unique=True)
    redacted_text: Mapped[str] = mapped_column(Text)
    note_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    claim: Mapped[Claim] = relationship(back_populates="note")


class Assessment(Base):
    __tablename__ = "assessments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), index=True)
    denial_probability: Mapped[float] = mapped_column(Float)
    likely_denied: Mapped[bool] = mapped_column(Boolean)
    risk_status: Mapped[str] = mapped_column(String(16), index=True)
    predicted_reason: Mapped[str] = mapped_column(String(32))
    factors: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    llm_status: Mapped[str] = mapped_column(String(32))
    model_info: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    claim: Mapped[Claim] = relationship(back_populates="assessments")
    doc_check: Mapped[DocCheck | None] = relationship(back_populates="assessment", uselist=False)
    reviews: Mapped[list[Review]] = relationship(back_populates="assessment", order_by="Review.id")


class DocCheck(Base):
    __tablename__ = "doc_checks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id"), unique=True)
    llm_status: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(8))
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    summary: Mapped[str] = mapped_column(Text, default="")
    draft_request: Mapped[str] = mapped_column(Text, default="")
    needs_review: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    assessment: Mapped[Assessment] = relationship(back_populates="doc_check")


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id"), index=True)
    decision: Mapped[str] = mapped_column(String(16))
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    assessment: Mapped[Assessment] = relationship(back_populates="reviews")


def database_url() -> str:
    """`CLAIMGUARD_DB_URL` if set, else the local SQLite file."""
    return os.getenv("CLAIMGUARD_DB_URL", f"sqlite:///{config.DB_PATH.as_posix()}")


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    """Create the engine once per URL and make sure all tables exist."""
    engine = create_engine(url or database_url(), connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def session_factory(url: str | None = None) -> sessionmaker[Session]:
    """Session factory bound to the engine for `url`."""
    return sessionmaker(bind=get_engine(url or database_url()), expire_on_commit=False)
