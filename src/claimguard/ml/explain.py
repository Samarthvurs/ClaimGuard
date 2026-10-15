"""SHAP explanations grouped back to raw features and phrased in plain language."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from claimguard.ml.features import source_feature  # noqa: E402

LABELS = {
    "age": "Patient age",
    "log_claim_cost": "Claim cost",
    "prior_auth_required": "Prior authorization requirement",
    "prior_auth_on_file": "Prior authorization on file",
    "documents_attached_count": "Documents attached",
    "min_documents": "Documents required",
    "docs_shortfall": "Missing documents",
    "days_to_submission": "Days to submission",
    "filing_limit_days": "Payer filing limit",
    "days_over_limit": "Days past filing limit",
    "provider_prior_denial_rate": "Provider's past denial rate",
    "patient_prior_denials": "Patient's past denials",
    "sex": "Patient sex",
    "payer": "Payer",
    "encounter_class": "Encounter type",
    "procedure_category": "Procedure category",
    "provider_specialty": "Provider specialty",
    "procedure_condition_pair": "Procedure-condition pairing",
}


def describe(feature: str, row: pd.Series) -> str:
    """Plain-language statement of a feature's value for one claim."""
    if feature == "prior_auth_on_file":
        if row["prior_auth_required"] and not row["prior_auth_on_file"]:
            return "No prior authorization on file (required for this procedure)"
        return "Prior authorization on file" if row[feature] else "No prior authorization on file"
    if feature == "prior_auth_required":
        req = "requires" if row[feature] else "does not require"
        return f"Procedure {req} prior authorization"
    if feature == "docs_shortfall":
        if row[feature] > 0:
            return f"{int(row[feature])} fewer document(s) than required"
        return "All required documents attached"
    if feature == "documents_attached_count":
        return f"{int(row[feature])} document(s) attached (minimum {int(row['min_documents'])})"
    if feature == "min_documents":
        return f"Procedure needs at least {int(row[feature])} document(s)"
    if feature == "days_to_submission":
        return (
            f"Submitted {int(row[feature])} days after service "
            f"(payer limit {int(row['filing_limit_days'])})"
        )
    if feature == "days_over_limit":
        if row[feature] > 0:
            return f"Submitted {int(row[feature])} days past the payer's filing limit"
        return "Submitted within the payer's filing limit"
    if feature == "filing_limit_days":
        return f"Payer filing limit is {int(row[feature])} days"
    if feature == "provider_prior_denial_rate":
        return f"Provider's past denial rate is {row[feature]:.0%}"
    if feature == "patient_prior_denials":
        return f"Patient has {int(row[feature])} past denied claim(s)"
    if feature == "procedure_condition_pair":
        return f"Procedure-condition pairing seen {int(row['pair_count'])} time(s) in training data"
    if feature == "log_claim_cost":
        return f"Claim cost ${np.expm1(row[feature]):,.0f}"
    if feature == "age":
        return f"Patient age {int(row[feature])}"
    return f"{LABELS.get(feature, feature)}: {row[feature]}"


class Explainer:
    """TreeSHAP explainer for the LightGBM step of a fitted pipeline."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pre = pipeline.named_steps["pre"]
        self.model = pipeline.named_steps["model"]
        self.names = list(self.pre.get_feature_names_out())
        self.groups = [source_feature(n) for n in self.names]
        self.explainer = shap.TreeExplainer(self.model)

    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """SHAP values (log-odds) on transformed columns, plus the transformed matrix."""
        Xt = self.pre.transform(X)
        Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
        values = self.explainer.shap_values(Xt)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        return values, Xt

    def grouped(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values summed per raw feature (one-hot columns added back together)."""
        values, _ = self.shap_values(X)
        frame = pd.DataFrame(values, columns=self.names)
        return frame.T.groupby(self.groups, sort=False).sum().T

    def top_factors(self, X: pd.DataFrame, k: int = 5) -> list[list[dict[str, Any]]]:
        """Top-k factors per row with direction, size and a plain-language description."""
        grouped = self.grouped(X)
        pair_counts = self._pair_counts(X)
        results = []
        for i in range(len(X)):
            row = X.iloc[i].copy()
            row["pair_count"] = pair_counts[i]
            contrib = grouped.iloc[i]
            order = contrib.abs().sort_values(ascending=False).index[:k]
            results.append(
                [
                    {
                        "feature": feat,
                        "label": LABELS.get(feat, feat),
                        "description": describe(feat, row),
                        "shap_value": round(float(contrib[feat]), 4),
                        "direction": "increases risk" if contrib[feat] > 0 else "decreases risk",
                    }
                    for feat in order
                ]
            )
        return results

    def _pair_counts(self, X: pd.DataFrame) -> list[int]:
        enc = self.pre.named_transformers_["pair"]
        keys = X["procedure_code"].astype(str) + "|" + X["condition_code"].astype(str)
        return [int(enc.counts_.get(k, 0)) for k in keys]

    def summary_plot(self, X: pd.DataFrame, path: Path) -> None:
        """Global mean |SHAP| bar chart per raw feature."""
        grouped = self.grouped(X)
        importance = grouped.abs().mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([LABELS.get(f, f) for f in importance.index], importance.values, color="#3b6ea5")
        ax.set_xlabel("Mean |SHAP value| (log-odds of denial)")
        ax.set_title("Global feature importance (test sample, synthetic data)")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
