"""Regex redaction applied to every note before it reaches an LLM.

This is a simple learning-project redactor, not production-grade PII protection.
"""

from __future__ import annotations

import re

from claimguard.data.notes import FAKE_FIRST_NAMES, FAKE_LAST_NAMES

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ID]"),
    (re.compile(r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"), "[DATE]"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATE]"),
    (re.compile(r"\b\d{6,}\b"), "[ID]"),
]
_NAMES = re.compile(r"\b(?:" + "|".join(FAKE_FIRST_NAMES + FAKE_LAST_NAMES) + r")\b")
_LABELLED_NAME = re.compile(r"(Patient(?: name)?:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})")


def redact(text: str) -> str:
    """Replace emails, phone numbers, ID-like numbers, dates and known names."""
    out = _LABELLED_NAME.sub(r"\1[NAME]", text)
    for pattern, token in _PATTERNS:
        out = pattern.sub(token, out)
    return _NAMES.sub("[NAME]", out)
