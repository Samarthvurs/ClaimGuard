"""Shared fixtures. Tests never call a real LLM."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

# Force mock mode for every test, regardless of the developer's .env.
os.environ["GEMINI_API_KEY"] = "PASTE_YOUR_GEMINI_KEY_HERE"
os.environ["API_KEY"] = "test-key"

from claimguard.data import build_claims, fallback_generate  # noqa: E402


@pytest.fixture(scope="session")
def fallback_build(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """A small claims build from the seeded fallback generator (no Synthea needed)."""
    out_dir: Path = tmp_path_factory.mktemp("fallback_csv")
    fallback_generate.generate(n_patients=300, n_claims=1500, out_dir=out_dir)
    return build_claims.build(out_dir)


@pytest.fixture(scope="session")
def claims(fallback_build: dict[str, object]) -> pd.DataFrame:
    return fallback_build["claims"]  # type: ignore[return-value]
