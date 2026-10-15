"""Metrics, slices and plots for the denial model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

AGE_BANDS = [(0, 17, "0-17"), (18, 44, "18-44"), (45, 64, "45-64"), (65, 200, "65+")]


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Threshold that maximises F1 (the documented threshold rule)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    """PR-AUC, ROC-AUC, F1, precision, recall, Brier and confusion matrix."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4) if both_classes else None,
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if both_classes else None,
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
        "threshold": round(float(threshold), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Always predict the training majority class with the training base rate."""
    rate = float(np.mean(y_train))
    majority = int(rate >= 0.5)
    pred = np.full(len(y_test), majority)
    return {
        "predicts": majority,
        "accuracy": round(float(np.mean(pred == y_test)), 4),
        "pr_auc": round(float(np.mean(y_test)), 4),
        "roc_auc": 0.5,
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_test, np.full(len(y_test), rate))), 4),
    }


def age_band(age: int) -> str:
    """Age band label for slice metrics."""
    for lo, hi, label in AGE_BANDS:
        if lo <= age <= hi:
            return label
    return "unknown"


def slice_metrics(
    df: pd.DataFrame, proba: np.ndarray, threshold: float, column: str
) -> list[dict[str, Any]]:
    """Metrics per value of `column` (payer or age band)."""
    rows = []
    y = df["denied"].to_numpy()
    for value in sorted(df[column].unique()):
        mask = (df[column] == value).to_numpy()
        m = binary_metrics(y[mask], proba[mask], threshold)
        rows.append(
            {
                column: value,
                "n": m["n"],
                "positive_rate": m["positive_rate"],
                "pr_auc": m["pr_auc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )
    return rows


def slice_gaps(rows: list[dict[str, Any]], overall: dict[str, Any], key: str) -> list[str]:
    """Plain-language notes for slices whose recall or F1 differ by more than 0.10."""
    notes = []
    for row in rows:
        for metric in ("recall", "f1"):
            gap = row[metric] - overall[metric]
            if abs(gap) > 0.10 and row["n"] >= 50:
                notes.append(
                    f"{key}={row[key]}: {metric} {row[metric]} vs overall {overall[metric]}"
                )
    return notes


def plot_calibration(
    y_true: np.ndarray, curves: dict[str, np.ndarray], path: Path, n_bins: int = 10
) -> None:
    """Reliability diagram for one or more probability vectors."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    for label, proba in curves.items():
        frac, mean = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=label)
    ax.set_xlabel("Predicted denial probability")
    ax.set_ylabel("Observed denial rate")
    ax.set_title("Calibration (test set, synthetic data)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pr_curves(y_true: np.ndarray, curves: dict[str, np.ndarray], path: Path) -> None:
    """Precision-recall curves for the compared models."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, proba in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    ax.axhline(float(np.mean(y_true)), color="grey", linestyle="--", label="base rate")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall (test set, synthetic data)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
