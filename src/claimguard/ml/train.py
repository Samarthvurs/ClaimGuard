"""Train the denial models with a time-based split and save artifacts and reports.

CLI: python -m claimguard.ml.train
"""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.pipeline import Pipeline

from claimguard import config
from claimguard.data.labeling import REASONS
from claimguard.ml import evaluate
from claimguard.ml.explain import Explainer
from claimguard.ml.features import FEATURES, add_derived, make_preprocessor

TEST_FRACTION = 0.2
VAL_FRACTION = 0.2
LGBM_GRID = {
    "num_leaves": [15, 31],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
}
LR_GRID = {"C": [0.1, 1.0, 10.0]}
CALIBRATION_MIN_GAIN = 0.05  # adopt isotonic calibration only if Brier improves >= 5%


def time_split(
    df: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by service date: the latest `test_fraction` of dates go to test."""
    dates = pd.to_datetime(df["service_date"])
    cutoff = dates.quantile(1 - test_fraction)
    return df[dates < cutoff].copy(), df[dates >= cutoff].copy()


def make_lgbm(seed: int = config.SEED, **params: Any) -> Pipeline:
    """LightGBM pipeline (preprocessing + model)."""
    model = LGBMClassifier(
        random_state=seed,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        verbose=-1,
        **params,
    )
    return Pipeline([("pre", make_preprocessor()), ("model", model)])


def make_logreg(seed: int = config.SEED, **params: Any) -> Pipeline:
    """Logistic regression baseline with the same preprocessing (plus scaling)."""
    model = LogisticRegression(max_iter=2000, random_state=seed, **params)
    return Pipeline([("pre", make_preprocessor(scale=True)), ("model", model)])


def grid_search(
    factory: Any, grid: dict[str, list[Any]], train: pd.DataFrame, val: pd.DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick the parameters with the best validation PR-AUC."""
    results = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values, strict=True))
        pipe = factory(**params).fit(train[FEATURES], train["denied"])
        proba = pipe.predict_proba(val[FEATURES])[:, 1]
        pr_auc = round(float(average_precision_score(val["denied"], proba)), 4)
        results.append({"params": params, "val_pr_auc": pr_auc})
    best = max(results, key=lambda r: r["val_pr_auc"])
    return best["params"], results


def maybe_calibrate(
    pipe: Pipeline, val: pd.DataFrame
) -> tuple[IsotonicRegression | None, dict[str, Any]]:
    """Fit isotonic calibration on the first half of validation, judge on the second."""
    half = len(val) // 2
    fit_part, judge_part = val.iloc[:half], val.iloc[half:]
    raw_fit = pipe.predict_proba(fit_part[FEATURES])[:, 1]
    raw_judge = pipe.predict_proba(judge_part[FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_fit, fit_part["denied"])
    brier_raw = brier_score_loss(judge_part["denied"], raw_judge)
    brier_cal = brier_score_loss(judge_part["denied"], iso.predict(raw_judge))
    gain = (brier_raw - brier_cal) / brier_raw
    info = {
        "method": "isotonic",
        "brier_uncalibrated": round(float(brier_raw), 4),
        "brier_calibrated": round(float(brier_cal), 4),
        "relative_gain": round(float(gain), 4),
        "adopted": bool(gain >= CALIBRATION_MIN_GAIN),
    }
    return (iso if info["adopted"] else None), info


def train_reason_model(train: pd.DataFrame, seed: int = config.SEED) -> Pipeline:
    """Multiclass model over denied training claims: which rule category is most likely."""
    denied = train[train["denied"] == 1]
    model = LGBMClassifier(
        random_state=seed, n_estimators=200, num_leaves=15, min_child_samples=20, verbose=-1
    )
    pipe = Pipeline([("pre", make_preprocessor()), ("model", model)])
    return pipe.fit(denied[FEATURES], denied["denial_reason"])


def _date_range(df: pd.DataFrame) -> list[str]:
    return [str(df["service_date"].min())[:10], str(df["service_date"].max())[:10]]


def predict_proba(artifact: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Denial probability from a saved artifact (applies calibration if adopted)."""
    raw = artifact["pipeline"].predict_proba(X[FEATURES])[:, 1]
    calibrator = artifact.get("calibrator")
    return calibrator.predict(raw) if calibrator is not None else raw


def run(
    claims: pd.DataFrame,
    reports_dir: Path = config.REPORTS_DIR,
    models_dir: Path = config.MODELS_DIR,
    lgbm_grid: dict[str, list[Any]] | None = None,
    shap_sample: int = 2000,
) -> dict[str, Any]:
    """Train, evaluate and save everything. Returns the metrics dict."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    df = add_derived(claims)
    train_all, test = time_split(df)
    train, val = time_split(train_all, VAL_FRACTION)

    lr_params, lr_search = grid_search(make_logreg, LR_GRID, train, val)
    lgbm_params, lgbm_search = grid_search(make_lgbm, lgbm_grid or LGBM_GRID, train, val)
    logreg = make_logreg(**lr_params).fit(train[FEATURES], train["denied"])
    lgbm = make_lgbm(**lgbm_params).fit(train[FEATURES], train["denied"])

    calibrator, calibration = maybe_calibrate(lgbm, val)
    artifact: dict[str, Any] = {"pipeline": lgbm, "calibrator": calibrator}
    val_proba = predict_proba(artifact, val)
    threshold = evaluate.best_f1_threshold(val["denied"].to_numpy(), val_proba)
    artifact["threshold"] = threshold

    y_test = test["denied"].to_numpy()
    lgbm_proba = predict_proba(artifact, test)
    lr_proba = logreg.predict_proba(test[FEATURES])[:, 1]
    lr_val = logreg.predict_proba(val[FEATURES])[:, 1]
    lr_threshold = evaluate.best_f1_threshold(val["denied"].to_numpy(), lr_val)

    reason_model = train_reason_model(train_all)
    artifact["reason_model"] = reason_model
    denied_test = test[test["denied"] == 1]
    reason_pred = reason_model.predict(denied_test[FEATURES])
    reason_metrics = {
        "approach": "multiclass LightGBM trained on denied claims only; reported on denied "
        "test claims",
        "n": int(len(denied_test)),
        "accuracy": round(float(np.mean(reason_pred == denied_test["denial_reason"])), 4),
        "macro_f1": round(
            float(
                f1_score(denied_test["denial_reason"], reason_pred, average="macro", labels=REASONS)
            ),
            4,
        ),
    }

    overall = evaluate.binary_metrics(y_test, lgbm_proba, threshold)
    test = test.assign(age_band=test["age"].map(evaluate.age_band))
    by_payer = evaluate.slice_metrics(test, lgbm_proba, threshold, "payer")
    by_age = evaluate.slice_metrics(test, lgbm_proba, threshold, "age_band")

    metrics: dict[str, Any] = {
        "data_note": "Synthetic data (Synthea + simulated admin fields) with rule-derived "
        "labels and 5% label noise. High scores show the pipeline works, not real-world "
        "performance.",
        "split": {
            "method": "time-based by service_date",
            "train": _date_range(train),
            "validation": _date_range(val),
            "test": _date_range(test),
            "n_train": int(len(train)),
            "n_validation": int(len(val)),
            "n_test": int(len(test)),
        },
        "threshold_rule": "maximise F1 on the validation slice",
        "lightgbm": {"params": lgbm_params, "test": overall, "search": lgbm_search},
        "logistic_regression": {
            "params": lr_params,
            "test": evaluate.binary_metrics(y_test, lr_proba, lr_threshold),
            "search": lr_search,
        },
        "majority_baseline": evaluate.majority_baseline(train["denied"].to_numpy(), y_test),
        "calibration": calibration,
        "reason_model": reason_metrics,
        "slices": {
            "by_payer": by_payer,
            "by_age_band": by_age,
            "large_gaps": evaluate.slice_gaps(by_payer, overall, "payer")
            + evaluate.slice_gaps(by_age, overall, "age_band"),
        },
    }

    raw_proba = lgbm.predict_proba(test[FEATURES])[:, 1]
    curves = {"LightGBM (raw)": raw_proba, "Logistic regression": lr_proba}
    if calibrator is not None:
        curves["LightGBM (isotonic)"] = lgbm_proba
    evaluate.plot_calibration(y_test, curves, reports_dir / "calibration.png")
    pr_curves = {"LightGBM": lgbm_proba, "Logistic regression": lr_proba}
    evaluate.plot_pr_curves(y_test, pr_curves, reports_dir / "pr_curve.png")
    explainer = Explainer(lgbm)
    sample = test.sample(n=min(shap_sample, len(test)), random_state=config.SEED)
    explainer.summary_plot(sample, reports_dir / "shap_summary.png")

    meta = {
        "model_name": "LightGBM denial classifier",
        "seed": config.SEED,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "features": FEATURES,
        "params": lgbm_params,
        "threshold": round(threshold, 4),
        "calibrated": calibrator is not None,
        "test_metrics": overall,
        "n_train": int(len(train)),
    }
    artifact["meta"] = meta
    joblib.dump(artifact, models_dir / config.MODEL_PATH.name)
    (models_dir / config.MODEL_META_PATH.name).write_text(json.dumps(meta, indent=2))
    (reports_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    """Train on the processed claims table."""
    claims = pd.read_csv(config.CLAIMS_PATH)
    metrics = run(claims)
    m = metrics["lightgbm"]["test"]
    print(
        f"LightGBM test PR-AUC={m['pr_auc']} ROC-AUC={m['roc_auc']} F1={m['f1']} "
        f"threshold={m['threshold']} (synthetic data, rule-derived labels)"
    )


if __name__ == "__main__":
    main()
