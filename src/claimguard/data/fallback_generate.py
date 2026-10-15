"""Seeded pure-Python fallback that writes Synthea-shaped CSV tables.

Used only when Java or the Synthea download is unavailable. It writes the same tables and
columns that `synthea.load_tables` reads, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from claimguard import config

FALLBACK_DIR = config.RAW_DIR / "fallback_output" / "csv"

PAYERS = [
    "Medicare", "Medicaid", "Dual Eligible", "Humana", "Blue Cross Blue Shield",
    "UnitedHealthcare", "Aetna", "Cigna Health", "Anthem",
]  # fmt: skip
PAYER_WEIGHTS = [0.3, 0.14, 0.05, 0.14, 0.07, 0.07, 0.08, 0.07, 0.08]
SPECIALTIES = ["GENERAL PRACTICE", "INTERNAL MEDICINE", "ORTHOPEDICS", "RADIOLOGY", "ONCOLOGY"]
CLASSES = ["ambulatory", "wellness", "outpatient", "urgentcare", "emergency", "inpatient"]

# (code, description, condition code, condition description)
PROCEDURES = [
    ("710824005", "Assessment of health and social care needs (procedure)", None, None),
    ("171207006", "Depression screening (procedure)", None, None),
    ("430193006", "Medication reconciliation (procedure)", None, None),
    ("399208008", "Plain X-ray of chest (procedure)", "233604007", "Pneumonia (disorder)"),
    ("71651007", "Mammography (procedure)", None, None),
    ("418891003", "Computed tomography of chest and abdomen (procedure)",
     "254637007", "Non-small cell lung cancer (disorder)"),
    ("91251008", "Physical therapy procedure (regime/therapy)",
     "239873007", "Osteoarthritis of knee (disorder)"),
    ("52052004", "Rehabilitation therapy (regime/therapy)", "230690007", "Stroke (disorder)"),
    ("265764009", "Renal dialysis (procedure)", "433144002", "Chronic kidney disease stage 3"),
    ("703423002", "Combined chemotherapy and radiation therapy (procedure)",
     "254637007", "Non-small cell lung cancer (disorder)"),
    ("609588000", "Total knee replacement (procedure)",
     "239873007", "Osteoarthritis of knee (disorder)"),
    ("80146002", "Appendectomy (procedure)", "74400008", "Appendicitis (disorder)"),
    ("73761001", "Colonoscopy (procedure)", None, None),
    ("127783003", "Spirometry (procedure)", "195967001", "Asthma (disorder)"),
    ("117015009", "Throat culture (procedure)", "43878008", "Streptococcal sore throat"),
]  # fmt: skip
PROC_WEIGHTS = np.array([12, 10, 8, 5, 4, 2, 6, 4, 6, 3, 2, 1, 3, 4, 4], dtype=float)


def generate(
    n_patients: int = 900,
    n_claims: int = 5000,
    seed: int = config.SEED,
    out_dir: Path = FALLBACK_DIR,
) -> Path:
    """Write Synthea-shaped CSVs (default `data/raw/fallback_output/csv`); return the dir."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    payer_ids = [f"payer-{i}" for i in range(len(PAYERS))]
    pd.DataFrame({"Id": payer_ids, "NAME": PAYERS}).to_csv(out_dir / "payers.csv", index=False)

    provider_ids = [f"prov-{i:03d}" for i in range(120)]
    pd.DataFrame(
        {"Id": provider_ids, "SPECIALITY": rng.choice(SPECIALTIES, size=len(provider_ids))}
    ).to_csv(out_dir / "providers.csv", index=False)

    patient_ids = [f"pat-{i:05d}" for i in range(n_patients)]
    births = pd.to_datetime("1935-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 85, size=n_patients), unit="D"
    )
    pd.DataFrame(
        {
            "Id": patient_ids,
            "BIRTHDATE": births.strftime("%Y-%m-%d"),
            "GENDER": rng.choice(["M", "F"], size=n_patients),
        }
    ).to_csv(out_dir / "patients.csv", index=False)

    enc_ids = [f"enc-{i:06d}" for i in range(n_claims)]
    starts = pd.to_datetime("2016-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 10, size=n_claims), unit="D"
    )
    pidx = rng.choice(len(PROCEDURES), size=n_claims, p=PROC_WEIGHTS / PROC_WEIGHTS.sum())
    patients = rng.choice(patient_ids, size=n_claims)
    pd.DataFrame(
        {
            "Id": enc_ids,
            "START": starts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "PATIENT": patients,
            "PROVIDER": rng.choice(provider_ids, size=n_claims),
            "PAYER": rng.choice(payer_ids, size=n_claims, p=PAYER_WEIGHTS),
            "ENCOUNTERCLASS": rng.choice(CLASSES, size=n_claims),
            "TOTAL_CLAIM_COST": np.round(rng.lognormal(7.0, 1.0, size=n_claims), 2),
            "REASONCODE": None,
            "REASONDESCRIPTION": None,
        }
    ).to_csv(out_dir / "encounters.csv", index=False)

    procs = [PROCEDURES[i] for i in pidx]
    pd.DataFrame(
        {
            "START": starts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "PATIENT": patients,
            "ENCOUNTER": enc_ids,
            "CODE": [p[0] for p in procs],
            "DESCRIPTION": [p[1] for p in procs],
            "BASE_COST": np.round(rng.lognormal(6.0, 0.8, size=n_claims), 2),
            "REASONCODE": [p[2] for p in procs],
            "REASONDESCRIPTION": [p[3] for p in procs],
        }
    ).to_csv(out_dir / "procedures.csv", index=False)

    pd.DataFrame(
        columns=["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"]
    ).to_csv(out_dir / "conditions.csv", index=False)
    return out_dir


if __name__ == "__main__":
    print(f"Fallback data written to {generate()}")
