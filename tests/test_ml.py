from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from claimguard.ml import evaluate, train
from claimguard.ml.features import FEATURES, RAW_FIELDS, add_derived
from claimguard.ml.serve import Scorer

SMALL_GRID = {"num_leaves": [7], "learning_rate": [0.1], "n_estimators": [50]}


@pytest.fixture(scope="module")
def trained(claims: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    reports = tmp_path_factory.mktemp("reports")
    models = tmp_path_factory.mktemp("models")
    train.run(claims, reports_dir=reports, models_dir=models, lgbm_grid=SMALL_GRID, shap_sample=200)
    return {"reports": reports, "models": models}


def test_time_split_has_no_date_overlap(claims: pd.DataFrame) -> None:
    tr, te = train.time_split(add_derived(claims))
    assert pd.to_datetime(tr["service_date"]).max() < pd.to_datetime(te["service_date"]).min()
    assert len(tr) + len(te) == len(claims)


def test_training_writes_metrics_and_artifact(trained: dict[str, Path]) -> None:
    metrics = json.loads((trained["reports"] / "metrics.json").read_text())
    test_metrics = metrics["lightgbm"]["test"]
    for key in ("pr_auc", "roc_auc", "f1", "precision", "recall", "brier", "confusion_matrix"):
        assert key in test_metrics
    assert "majority_baseline" in metrics and "slices" in metrics
    split = metrics["split"]
    assert split["train"][1] < split["validation"][0] <= split["validation"][1] < split["test"][0]
    for plot in ("calibration.png", "shap_summary.png", "pr_curve.png"):
        assert (trained["reports"] / plot).exists()
    meta = json.loads((trained["models"] / "denial_model_meta.json").read_text())
    assert meta["features"] == FEATURES and meta["seed"] == 42


def test_scorer_returns_probability_and_five_factors(
    trained: dict[str, Path], claims: pd.DataFrame
) -> None:
    scorer = Scorer(joblib.load(trained["models"] / "denial_model.joblib"))
    result = scorer.score(claims.iloc[-1].to_dict())
    assert 0.0 <= result["denial_probability"] <= 1.0
    assert len(result["factors"]) == 5
    for factor in result["factors"]:
        assert factor["direction"] in {"increases risk", "decreases risk"}
        assert isinstance(factor["description"], str) and factor["description"]
    if not result["likely_denied"]:
        assert result["predicted_reason"] == "NONE"


def test_scorer_rejects_missing_fields(trained: dict[str, Path]) -> None:
    scorer = Scorer(joblib.load(trained["models"] / "denial_model.joblib"))
    with pytest.raises(ValueError):
        scorer.score({"age": 40})


def test_shap_output_shape(trained: dict[str, Path], claims: pd.DataFrame) -> None:
    scorer = Scorer(joblib.load(trained["models"] / "denial_model.joblib"))
    X = add_derived(claims.head(20)[RAW_FIELDS])
    values, Xt = scorer.explainer.shap_values(X)
    assert values.shape == Xt.shape
    grouped = scorer.explainer.grouped(X)
    assert grouped.shape[0] == 20
    assert "payer" in grouped.columns and "procedure_condition_pair" in grouped.columns


def test_threshold_rule_maximises_f1() -> None:
    import numpy as np

    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.75])
    t = evaluate.best_f1_threshold(y, p)
    assert t == pytest.approx(0.7)
