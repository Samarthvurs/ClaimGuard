"""Central configuration: paths, environment settings and decision thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHEA_CSV_DIR = RAW_DIR / "synthea_output" / "csv"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
SAMPLES_DIR = DATA_DIR / "samples"
PROMPTS_DIR = ROOT / "prompts"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
LLM_CACHE_DIR = ROOT / ".llm_cache"

CLAIMS_PATH = PROCESSED_DIR / "claims.csv"
NOTES_PATH = PROCESSED_DIR / "notes.jsonl"
MODEL_PATH = MODELS_DIR / "denial_model.joblib"
MODEL_META_PATH = MODELS_DIR / "denial_model_meta.json"
REASON_MODEL_PATH = MODELS_DIR / "reason_model.joblib"
DB_PATH = ROOT / "claimguard.db"

SEED = 42
LABEL_NOISE_RATE = 0.05

# Decision threshold for the binary "likely denied" flag. Rule: the threshold that
# maximises F1 on the validation slice (last 20% of the training period by date).
# The trained value is written to the model metadata; this is only the fallback.
DEFAULT_DECISION_THRESHOLD = 0.5

# Risk status bands on the predicted denial probability. These only route claims
# for human review; the system never approves or denies a claim.
RISK_LOW_MAX = 0.30
RISK_HIGH_MIN = 0.60

# Doc-check items below this confidence are routed to human review.
LLM_LOW_CONFIDENCE = 0.6

PLACEHOLDER_KEY = "PASTE_YOUR_GEMINI_KEY_HERE"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

load_dotenv(ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    """Runtime settings read from the environment (and `.env`)."""

    gemini_api_key: str
    llm_model: str
    llm_max_rpm: int
    llm_max_calls_per_run: int
    api_key: str

    @property
    def llm_configured(self) -> bool:
        """True when a real (non-placeholder, non-empty) Gemini key is present."""
        key = self.gemini_api_key.strip()
        return bool(key) and key != PLACEHOLDER_KEY


def get_settings() -> Settings:
    """Read settings from the current environment."""
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"),
        llm_max_rpm=int(os.getenv("LLM_MAX_RPM", "10")),
        llm_max_calls_per_run=int(os.getenv("LLM_MAX_CALLS_PER_RUN", "600")),
        api_key=os.getenv("API_KEY", "dev-key-change-me"),
    )
