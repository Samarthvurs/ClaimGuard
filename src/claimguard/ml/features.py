"""Feature definitions and the preprocessing used by every model.

All preprocessing lives inside one scikit-learn `ColumnTransformer` so the saved
`Pipeline` turns a raw claim dict into model inputs at serving time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from claimguard.data import labeling

NUMERIC = [
    "age",
    "log_claim_cost",
    "prior_auth_required",
    "prior_auth_on_file",
    "documents_attached_count",
    "min_documents",
    "docs_shortfall",
    "days_to_submission",
    "filing_limit_days",
    "days_over_limit",
    "provider_prior_denial_rate",
    "patient_prior_denials",
]
CATEGORICAL = ["sex", "payer", "encounter_class", "procedure_category", "provider_specialty"]
PAIR = ["procedure_code", "condition_code"]
FEATURES = NUMERIC + CATEGORICAL + PAIR

# Raw input fields a caller must supply to score a claim.
RAW_FIELDS = [
    "age", "sex", "payer", "encounter_class", "procedure_category", "provider_specialty",
    "procedure_code", "condition_code", "claim_cost", "prior_auth_required",
    "prior_auth_on_file", "documents_attached_count", "min_documents", "days_to_submission",
    "provider_prior_denial_rate", "patient_prior_denials",
]  # fmt: skip


class PairFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode a (procedure, condition) pair by how often it appeared in training data.

    Rare pairings are a signal for coding mismatches. Fitted on training data only.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> PairFrequencyEncoder:
        keys = X.iloc[:, 0].astype(str) + "|" + X.iloc[:, 1].astype(str)
        self.counts_ = keys.value_counts().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        keys = X.iloc[:, 0].astype(str) + "|" + X.iloc[:, 1].astype(str)
        counts = keys.map(self.counts_).fillna(0).to_numpy(dtype=float)
        return np.log1p(counts).reshape(-1, 1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(["pair_log_frequency"])


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns from raw claim fields (same code at training and serving)."""
    out = df.copy()
    rules = labeling.load_rules()
    out["log_claim_cost"] = np.log1p(pd.to_numeric(out["claim_cost"], errors="coerce").fillna(0))
    for col in ("prior_auth_required", "prior_auth_on_file"):
        out[col] = out[col].astype(str).str.lower().isin(["true", "1"]).astype(int)
    out["docs_shortfall"] = (out["min_documents"] - out["documents_attached_count"]).clip(lower=0)
    out["filing_limit_days"] = out["payer"].map(lambda p: labeling.filing_limit(p, rules))
    out["days_over_limit"] = (out["days_to_submission"] - out["filing_limit_days"]).clip(lower=0)
    for col in CATEGORICAL + PAIR:
        out[col] = out[col].astype(str)
    return out


def make_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Column transformer shared by the baseline and LightGBM models."""
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
                CATEGORICAL,
            ),
            ("pair", PairFrequencyEncoder(), PAIR),
        ],
        verbose_feature_names_out=True,
    )


def source_feature(transformed_name: str) -> str:
    """Map a transformed column name back to the raw feature it came from."""
    prefix, _, rest = transformed_name.partition("__")
    if prefix == "num":
        return rest
    if prefix == "pair":
        return "procedure_condition_pair"
    for cat in CATEGORICAL:
        if rest.startswith(cat + "_"):
            return cat
    return rest
