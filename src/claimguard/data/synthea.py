"""Download, run and load Synthea CSV output (open-source synthetic patient generator)."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - runs the local Synthea jar with fixed arguments
import time
import urllib.request
from pathlib import Path

import pandas as pd

from claimguard import config

JAR_URL = (
    "https://github.com/synthetichealth/synthea/releases/download/"
    "master-branch-latest/synthea-with-dependencies.jar"
)
JAR_PATH = config.RAW_DIR / "synthea-with-dependencies.jar"
OUTPUT_DIR = config.RAW_DIR / "synthea_output"

# Only the columns the claims builder uses (checked against real Synthea output).
COLUMNS = {
    "patients": ["Id", "BIRTHDATE", "GENDER"],
    "encounters": [
        "Id", "START", "PATIENT", "PROVIDER", "PAYER", "ENCOUNTERCLASS",
        "TOTAL_CLAIM_COST", "REASONCODE", "REASONDESCRIPTION",
    ],
    "procedures": [
        "START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "BASE_COST",
        "REASONCODE", "REASONDESCRIPTION",
    ],
    "conditions": ["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"],
    "payers": ["Id", "NAME"],
    "providers": ["Id", "SPECIALITY"],
}  # fmt: skip


def java_available() -> bool:
    """Return True if a `java` executable is on PATH."""
    return shutil.which("java") is not None


def download_jar(retries: int = 2) -> Path:
    """Download the Synthea jar (with retries) unless it already exists."""
    if JAR_PATH.exists() and JAR_PATH.stat().st_size > 1_000_000:
        return JAR_PATH
    JAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            urllib.request.urlretrieve(JAR_URL, JAR_PATH)  # nosec B310 - fixed https URL
            return JAR_PATH
        except OSError as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Synthea download failed: {last_error}")


def run_synthea(population: int = 3000, seed: int = config.SEED) -> Path:
    """Run Synthea with a fixed seed and CSV export; return the CSV directory."""
    cmd = [
        "java", "-Xmx4g", "-jar", str(JAR_PATH), "-p", str(population),
        "-s", str(seed), "-cs", str(seed),
        "--exporter.csv.export=true", "--exporter.fhir.export=false",
        "--exporter.hospital.fhir.export=false", "--exporter.practitioner.fhir.export=false",
        f"--exporter.baseDirectory={OUTPUT_DIR}",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=config.RAW_DIR, capture_output=True)  # nosec B603
    return OUTPUT_DIR / "csv"


def load_tables(csv_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the Synthea-shaped CSV tables used by the claims builder."""
    tables = {}
    for name, cols in COLUMNS.items():
        tables[name] = pd.read_csv(csv_dir / f"{name}.csv", usecols=cols, dtype=str)
    return tables
