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


@pytest.fixture(scope="session")
def tiny_model_path(claims: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small model trained on fallback claims (CI has no saved model)."""
    from claimguard.ml import train

    reports = tmp_path_factory.mktemp("tiny_reports")
    models = tmp_path_factory.mktemp("tiny_models")
    grid = {"num_leaves": [7], "learning_rate": [0.1], "n_estimators": [50]}
    train.run(claims, reports_dir=reports, models_dir=models, lgbm_grid=grid, shap_sample=100)
    return models / "denial_model.joblib"


@pytest.fixture()
def app_env(tiny_model_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fresh SQLite DB and the tiny model wired into the service layer."""
    import joblib

    from claimguard.ml.serve import Scorer
    from claimguard.services import assessment

    monkeypatch.setenv("CLAIMGUARD_DB_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    scorer = Scorer(joblib.load(tiny_model_path))
    monkeypatch.setattr(assessment, "get_scorer", lambda: scorer)
    return tmp_path
