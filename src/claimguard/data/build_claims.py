"""Build the claims table from Synthea-shaped tables.

Real-structure fields come from Synthea (age, sex, payer, provider, procedure, condition,
service date, claim cost). Administrative fields are SIMULATED with a fixed seed:
prior authorization, attached documents, submission delay, and a small rate of
condition-code swaps that create coding mismatches. History features
(`provider_prior_denial_rate`, `patient_prior_denials`) use only strictly earlier claims.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from claimguard import config
from claimguard.data import labeling

START_DATE = "2016-01-01"
MAX_CLAIMS = 30_000
POLICY_TOP_N = 150
CODING_SWAP_RATE = 0.04
MIN_DOCS = {"routine": 0, "diagnostic": 1, "imaging": 2, "therapy": 2, "surgery": 3}
HISTORY_PRIOR_RATE = 0.2
HISTORY_PRIOR_WEIGHT = 5

_ROUTINE = (
    r"screening|assessment|education|plaque|fluoride|reconciliation|referral|guidance|"
    r"consultation|examination|evaluation|counsel|vaccin|immuniz|dental care|surveillance"
)
_IMAGING = r"x-ray|radiograph|computed tomography|magnetic resonance|\bmri\b|ultrasound|mammogra|imaging|\bscan\b"  # noqa: E501
_THERAPY = r"therap|dialysis|rehabilitation|radiation|hospice|injection|administration|nursing"
_SURGERY = (
    r"ectomy|otomy|ostomy|plasty|surgical|excision|transplant|insertion|replacement|repair|"
    r"amputation|biopsy|implant|restoration|filling|extraction|fixation|catheter"
)
_ADVANCED_IMAGING = r"computed tomography|magnetic resonance|\bmri\b"
_PA_THERAPY = r"chemotherapy|radiation|rehabilitation|physical therapy|occupational|immunotherapy"


def categorize(description: str) -> str:
    """Assign a procedure category by keyword rules on its description."""
    d = description.lower()
    if re.search(_IMAGING, d):
        return "imaging"
    if re.search(_ROUTINE, d):
        return "routine"
    if re.search(_THERAPY, d):
        return "therapy"
    if re.search(_SURGERY, d):
        return "surgery"
    return "diagnostic"


def requires_prior_auth(description: str, category: str) -> bool:
    """Simulated prior-authorization policy by category and keywords."""
    d = description.lower()
    if category == "surgery":
        return True
    if category == "imaging":
        return bool(re.search(_ADVANCED_IMAGING, d))
    if category == "therapy":
        return bool(re.search(_PA_THERAPY, d))
    return False


def build_policy(claims: pd.DataFrame, top_n: int = POLICY_TOP_N) -> pd.DataFrame:
    """Build `procedure_policy` for the most frequent procedures."""
    top = (
        claims.groupby(["procedure_code", "procedure_description"])
        .size()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index(name="count")
    )
    top["category"] = top["procedure_description"].map(categorize)
    top["prior_auth_required"] = [
        requires_prior_auth(d, c)
        for d, c in zip(top["procedure_description"], top["category"], strict=True)
    ]
    top["min_documents"] = top["category"].map(MIN_DOCS)
    return top.drop(columns="count")


def base_claims(tables: dict[str, pd.DataFrame], seed: int = config.SEED) -> pd.DataFrame:
    """One row per encounter with a procedure (the encounter's most costly procedure)."""
    enc = tables["encounters"].copy()
    enc["service_dt"] = pd.to_datetime(enc["START"], utc=True).dt.tz_localize(None)
    enc = enc[enc["service_dt"] >= START_DATE]

    proc = tables["procedures"].copy()
    proc["BASE_COST"] = pd.to_numeric(proc["BASE_COST"], errors="coerce").fillna(0.0)
    proc = proc.sort_values(["ENCOUNTER", "BASE_COST"], ascending=[True, False])
    proc = proc.drop_duplicates("ENCOUNTER")

    cond = tables["conditions"].drop_duplicates("ENCOUNTER")[["ENCOUNTER", "CODE", "DESCRIPTION"]]
    cond = cond.rename(columns={"CODE": "COND_CODE", "DESCRIPTION": "COND_DESC"})

    payers = tables["payers"].rename(columns={"Id": "PAYER", "NAME": "payer"})
    providers = tables["providers"].rename(columns={"Id": "PROVIDER", "SPECIALITY": "specialty"})
    patients = tables["patients"].rename(columns={"Id": "PATIENT"})

    df = proc.merge(enc, left_on="ENCOUNTER", right_on="Id", suffixes=("_proc", "_enc"))
    df = df.merge(cond, on="ENCOUNTER", how="left")
    df = df.merge(payers, on="PAYER", how="left")
    df = df.merge(providers, on="PROVIDER", how="left")
    df = df.merge(patients, left_on="PATIENT_enc", right_on="PATIENT", how="left")
    df = df[df["payer"].notna() & (df["payer"] != "NO_INSURANCE")]

    # Linked condition: procedure reason, else encounter reason, else a condition
    # recorded at the encounter, else none.
    df["condition_code"] = (
        df["REASONCODE_proc"].fillna(df["REASONCODE_enc"]).fillna(df["COND_CODE"]).fillna("NONE")
    )
    df["condition_description"] = (
        df["REASONDESCRIPTION_proc"]
        .fillna(df["REASONDESCRIPTION_enc"])
        .fillna(df["COND_DESC"])
        .fillna("No linked condition")
    )
    birth = pd.to_datetime(df["BIRTHDATE"], errors="coerce")
    df["age"] = ((df["service_dt"] - birth).dt.days // 365).clip(lower=0).astype(int)

    out = pd.DataFrame(
        {
            "encounter_id": df["ENCOUNTER"],
            "patient_id": df["PATIENT_enc"],
            "provider_id": df["PROVIDER"],
            "provider_specialty": df["specialty"].fillna("UNKNOWN"),
            "payer": df["payer"],
            "encounter_class": df["ENCOUNTERCLASS"],
            "service_date": df["service_dt"].dt.normalize(),
            "age": df["age"],
            "sex": df["GENDER"].fillna("U"),
            "procedure_code": df["CODE"].astype(str),
            "procedure_description": df["DESCRIPTION"],
            "condition_code": df["condition_code"].astype(str),
            "condition_description": df["condition_description"],
            "claim_cost": pd.to_numeric(df["TOTAL_CLAIM_COST"], errors="coerce").fillna(0.0),
        }
    )
    if len(out) > MAX_CLAIMS:
        out = out.sample(n=MAX_CLAIMS, random_state=seed)
    out = out.sort_values(["service_date", "encounter_id"]).reset_index(drop=True)
    out.insert(0, "claim_id", [f"CLM-{i:06d}" for i in range(1, len(out) + 1)])
    return out


def build_compat_table(claims: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Procedure-condition pairs seen at least `min_count` times (before simulated swaps)."""
    pairs = claims.groupby(["procedure_code", "condition_code"]).size().reset_index(name="count")
    return pairs[pairs["count"] >= min_count].reset_index(drop=True)


def add_simulated_fields(
    claims: pd.DataFrame, policy: pd.DataFrame, seed: int = config.SEED
) -> pd.DataFrame:
    """Add SIMULATED administrative fields (seeded)."""
    rng = np.random.default_rng(seed)
    df = claims.copy()
    n = len(df)

    pol = policy.set_index("procedure_code")
    in_policy = df["procedure_code"].isin(pol.index)
    df["procedure_category"] = np.where(
        in_policy,
        df["procedure_code"].map(pol["category"]),
        df["procedure_description"].map(categorize),
    )
    df["prior_auth_required"] = np.where(
        in_policy, df["procedure_code"].map(pol["prior_auth_required"]), False
    ).astype(bool)
    df["min_documents"] = df["procedure_category"].map(MIN_DOCS).astype(int)

    # Simulated provider "documentation quality": some providers are sloppier.
    providers = df["provider_id"].unique()
    sloppiness = dict(zip(providers, rng.beta(1.2, 6.0, size=len(providers)), strict=True))
    slop = df["provider_id"].map(sloppiness).to_numpy()

    pa_on_file_p = np.where(df["prior_auth_required"], 0.92 - 0.6 * slop, 0.03)
    df["prior_auth_on_file"] = rng.random(n) < pa_on_file_p

    delta = rng.choice([-2, -1, 0, 1, 2], size=n, p=[0.04, 0.08, 0.45, 0.31, 0.12])
    extra_short = rng.random(n) < slop * 0.5
    delta = np.where(extra_short, delta - 1, delta)
    df["documents_attached_count"] = np.clip(df["min_documents"] + delta, 0, None).astype(int)

    days = np.round(rng.gamma(1.5, 12.0, size=n)).astype(int) + 1
    late = rng.random(n) < 0.04 + 0.15 * slop
    days = np.where(late, rng.integers(60, 420, size=n), days)
    df["days_to_submission"] = days
    df["submission_date"] = df["service_date"] + pd.to_timedelta(days, unit="D")

    # Simulated coding errors: swap the condition with one from a random other claim.
    swap = rng.random(n) < CODING_SWAP_RATE
    donors = rng.integers(0, n, size=n)
    for col in ("condition_code", "condition_description"):
        values = df[col].to_numpy().copy()
        values[swap] = df[col].to_numpy()[donors[swap]]
        df[col] = values
    return df


def _prior_counts(df: pd.DataFrame, key: str) -> tuple[pd.Series, pd.Series]:
    """Counts of strictly earlier claims and denials per `key` (by service date)."""
    daily = df.groupby([key, "service_date"])["denied"].agg(["size", "sum"]).reset_index()
    daily = daily.sort_values([key, "service_date"])
    grp = daily.groupby(key)
    daily["prior_n"] = grp["size"].cumsum() - daily["size"]
    daily["prior_denied"] = grp["sum"].cumsum() - daily["sum"]
    merged = df[[key, "service_date"]].merge(daily, on=[key, "service_date"], how="left")
    return merged["prior_n"].to_numpy(), merged["prior_denied"].to_numpy()


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-safe history features computed only from earlier claims."""
    out = df.copy()
    n_prov, d_prov = _prior_counts(out, "provider_id")
    out["provider_prior_denial_rate"] = (
        d_prov + HISTORY_PRIOR_RATE * HISTORY_PRIOR_WEIGHT
    ) / (n_prov + HISTORY_PRIOR_WEIGHT)
    _, d_pat = _prior_counts(out, "patient_id")
    out["patient_prior_denials"] = d_pat.astype(int)
    return out


def build(csv_dir: Path, seed: int = config.SEED) -> dict[str, object]:
    """Run the full build: base table, policy, simulation, labels, history features."""
    from claimguard.data.synthea import load_tables

    rules = labeling.load_rules()
    tables = load_tables(csv_dir)
    claims = base_claims(tables, seed=seed)
    policy = build_policy(claims)
    compat = build_compat_table(claims, rules["compat_min_count"])
    claims = add_simulated_fields(claims, policy, seed=seed)
    claims = labeling.apply_labels(claims, compat, rules, seed=seed)
    claims = add_history_features(claims)
    return {"claims": claims, "policy": policy, "compat": compat}


def resolve_source() -> tuple[str, Path]:
    """Pick the raw data source: Synthea output if present, else the fallback output."""
    from claimguard.data.fallback_generate import FALLBACK_DIR

    if (config.SYNTHEA_CSV_DIR / "encounters.csv").exists():
        return "synthea", config.SYNTHEA_CSV_DIR
    if (FALLBACK_DIR / "encounters.csv").exists():
        return "fallback", FALLBACK_DIR
    raise FileNotFoundError("No raw data found. Run `python scripts/download_data.py` first.")


def main() -> None:
    """Build and save the claims table, reference files, notes, samples and label report."""
    from claimguard.data import notes as notes_mod

    source, csv_dir = resolve_source()
    result = build(csv_dir)
    claims: pd.DataFrame = result["claims"]  # type: ignore[assignment]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    claims.to_csv(config.CLAIMS_PATH, index=False)
    result["policy"].to_csv(config.REFERENCE_DIR / "procedure_policy.csv", index=False)  # type: ignore[union-attr]
    result["compat"].to_csv(  # type: ignore[union-attr]
        config.REFERENCE_DIR / "procedure_condition_compat.csv", index=False
    )
    records = notes_mod.build_notes(claims)
    notes_mod.save_notes(records)

    dist = labeling.label_distribution(claims, labeling.load_rules())
    dist["data_source"] = source
    labeling.save_distribution(dist)

    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    sample = claims.iloc[:: max(1, len(claims) // 600)].head(600)
    sample.to_csv(config.SAMPLES_DIR / "claims_sample.csv", index=False)
    print(f"source={source} claims={len(claims)} denial_rate={dist['denial_rate']}")


if __name__ == "__main__":
    main()
