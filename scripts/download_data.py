"""Download and run Synthea; fall back to the seeded pure-Python generator on failure.

Usage: python scripts/download_data.py [--population 3000] [--force-fallback]
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from claimguard import config
from claimguard.data import fallback_generate, synthea


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=3000)
    parser.add_argument("--force-fallback", action="store_true")
    args = parser.parse_args()

    info: dict[str, object] = {"created": datetime.now(UTC).isoformat(), "seed": config.SEED}
    source = "fallback"
    reason = "forced" if args.force_fallback else ""
    if not args.force_fallback:
        if not synthea.java_available():
            reason = "java not found"
        else:
            try:
                synthea.download_jar()
                for attempt in range(2):
                    try:
                        synthea.run_synthea(population=args.population)
                        source = "synthea"
                        break
                    except Exception as exc:  # noqa: BLE001 - any failure triggers retry
                        reason = f"synthea run failed (attempt {attempt + 1}): {exc}"
            except RuntimeError as exc:
                reason = str(exc)
    if source == "fallback":
        fallback_generate.generate()
        info["fallback_reason"] = reason
    else:
        info["population"] = args.population
        info["jar_url"] = synthea.JAR_URL
    info["source"] = source
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_DIR / "source.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"data source: {source}" + (f" ({reason})" if reason else ""))


if __name__ == "__main__":
    main()
