"""Rule-derived denial labels with seeded label noise.

The labels are SIMULATED: a claim is "denied" when one of the rules in
`data/reference/denial_rules.yaml` fires, then a fixed share of labels is flipped at random.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from claimguard import config

REASONS = ["MISSING_PRIOR_AUTH", "CODING_MISMATCH", "MISSING_DOCUMENTATION", "TIMELY_FILING"]


def load_rules(path: Path | None = None) -> dict[str, Any]:
    """Load the denial rule configuration."""
    path = path or config.REFERENCE_DIR / "denial_rules.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def filing_limit(payer: str, rules: dict[str, Any]) -> int:
    """Timely filing limit in days for a payer."""
    limits = rules["timely_filing_days"]
    return int(limits.get(payer, limits["default"]))


def rule_reason(df: pd.DataFrame, compat: pd.DataFrame, rules: dict[str, Any]) -> pd.Series:
    """First rule that fires per claim, or NONE."""
    pairs = set(zip(compat["procedure_code"], compat["condition_code"], strict=True))
    mismatch = [
        (p, c) not in pairs for p, c in zip(df["procedure_code"], df["condition_code"], strict=True)
    ]
    limits = df["payer"].map(lambda p: filing_limit(p, rules))
    conditions = [
        df["prior_auth_required"] & ~df["prior_auth_on_file"],
        pd.Series(mismatch, index=df.index),
        df["documents_attached_count"] < df["min_documents"],
        df["days_to_submission"] > limits,
    ]
    return pd.Series(np.select(conditions, REASONS, default="NONE"), index=df.index)


def apply_labels(
    df: pd.DataFrame, compat: pd.DataFrame, rules: dict[str, Any], seed: int = config.SEED
) -> pd.DataFrame:
    """Add `rule_reason`, `denied`, `denial_reason` and `label_flipped` columns."""
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    out["rule_reason"] = rule_reason(out, compat, rules)
    denied = (out["rule_reason"] != "NONE").to_numpy()
    flip = rng.random(len(out)) < rules["label_noise_rate"]
    noisy = np.where(flip, ~denied, denied)
    reason = out["rule_reason"].to_numpy().copy()
    random_reasons = rng.choice(REASONS, size=len(out))
    reason = np.where(flip & noisy, random_reasons, reason)
    reason = np.where(flip & ~noisy, "NONE", reason)
    out["denied"] = noisy.astype(int)
    out["denial_reason"] = reason
    out["label_flipped"] = flip
    return out


def label_distribution(df: pd.DataFrame, rules: dict[str, Any]) -> dict[str, Any]:
    """Summary of the label distribution for `reports/label_distribution.json`."""
    return {
        "n_claims": int(len(df)),
        "denial_rate": round(float(df["denied"].mean()), 4),
        "noise_rate_configured": rules["label_noise_rate"],
        "noise_rate_observed": round(float(df["label_flipped"].mean()), 4),
        "rule_reason_counts": df["rule_reason"].value_counts().to_dict(),
        "final_reason_counts": df["denial_reason"].value_counts().to_dict(),
        "note": "Synthetic data with rule-derived labels; not real payer outcomes.",
    }


def save_distribution(dist: dict[str, Any], path: Path | None = None) -> None:
    """Write the label distribution report."""
    path = path or config.REPORTS_DIR / "label_distribution.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dist, indent=2), encoding="utf-8")
