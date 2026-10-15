"""Load the trained artifact once and score single claims."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from claimguard import config
from claimguard.ml.explain import Explainer
from claimguard.ml.features import FEATURES, RAW_FIELDS, add_derived
from claimguard.ml.train import predict_proba


class Scorer:
    """Wraps the saved artifact: probability, flag, predicted reason and SHAP factors."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.threshold = float(artifact["threshold"])
        self.meta = artifact.get("meta", {})
        self.explainer = Explainer(artifact["pipeline"])

    def score(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Score one claim given its raw fields (see `RAW_FIELDS`)."""
        missing = [f for f in RAW_FIELDS if f not in claim]
        if missing:
            raise ValueError(f"claim is missing fields: {missing}")
        X = add_derived(pd.DataFrame([{f: claim[f] for f in RAW_FIELDS}]))
        proba = float(predict_proba(self.artifact, X)[0])
        reason_model = self.artifact["reason_model"]
        reason_proba = reason_model.predict_proba(X[FEATURES])[0]
        reasons = {
            str(c): round(float(p), 4)
            for c, p in zip(reason_model.classes_, reason_proba, strict=True)
        }
        likely = proba >= self.threshold
        return {
            "denial_probability": round(proba, 4),
            "likely_denied": bool(likely),
            "threshold": round(self.threshold, 4),
            "predicted_reason": max(reasons, key=reasons.get) if likely else "NONE",
            "reason_probabilities": reasons,
            "factors": self.explainer.top_factors(X, k=5)[0],
        }


@lru_cache(maxsize=1)
def get_scorer(path: Path = config.MODEL_PATH) -> Scorer:
    """Load the artifact once per process."""
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `python -m claimguard.ml.train`.")
    return Scorer(joblib.load(path))


def score(claim: dict[str, Any]) -> dict[str, Any]:
    """Score one claim with the default saved model."""
    return get_scorer().score(claim)
