"""End-to-end run for a fresh clone: data, training, demo data.

The LLM prompt evaluation is a separate command: python -m claimguard.llm.eval

Usage: python scripts/run_all.py [--skip-download]
"""

from __future__ import annotations

import argparse

# Runs this project's own scripts with the current interpreter (no shell).
import subprocess  # nosec B404
import sys
from pathlib import Path

from claimguard import config

STEPS = [
    ["scripts/download_data.py"],
    ["-m", "claimguard.data.build_claims"],
    ["-m", "claimguard.ml.train"],
    ["scripts/load_demo_data.py"],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-download", action="store_true", help="reuse existing raw data if present"
    )
    args = parser.parse_args()
    raw_present = (config.RAW_DIR / "source.json").exists()
    for step in STEPS:
        if step[0].endswith("download_data.py") and args.skip_download and raw_present:
            print("skipping download (raw data present)")
            continue
        print(f"$ python {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, *step], check=True, cwd=Path(config.ROOT))  # nosec B603


if __name__ == "__main__":
    main()
