"""Load a few hundred synthetic claims into SQLite and assess them for the dashboard.

ML-only when no Gemini key is configured (assessments show `llm_status = not_configured`).
With a key, the first `--llm-claims` claims also get the documentation check, within
the per-run call cap.

Usage: python scripts/load_demo_data.py [--n 300] [--llm-claims 20]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from claimguard import config
from claimguard.api.db import Base, get_engine, session_factory
from claimguard.data.notes import load_notes
from claimguard.llm.provider import get_provider
from claimguard.ml.train import time_split
from claimguard.services import assessment as svc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--llm-claims", type=int, default=20)
    args = parser.parse_args()

    claims = pd.read_csv(config.CLAIMS_PATH)
    _, test = time_split(claims)  # demo claims come from the held-out test period
    demo = test.sample(n=min(args.n, len(test)), random_state=config.SEED)
    notes = {r["claim_id"]: r["note"] for r in load_notes()}

    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = session_factory()()

    provider, llm_state = get_provider()
    statuses: Counter[str] = Counter()
    llm_statuses: Counter[str] = Counter()
    for i, row in enumerate(demo.to_dict(orient="records")):
        payload = {
            **row,
            "external_id": row["claim_id"],
            "service_date": str(row["service_date"])[:10],
            "note": notes[row["claim_id"]],
        }
        claim = svc.create_claim(session, payload)
        use_llm = llm_state == "not_configured" or i < args.llm_claims
        assessment = svc.assess_claim(
            session,
            claim.id,
            use_llm=use_llm,
            provider=None if llm_state == "not_configured" else provider,
        )
        statuses[assessment.risk_status] += 1
        llm_statuses[assessment.llm_status] += 1
    summary = svc.summary(session)
    session.close()
    summary["llm_configured"] = llm_state != "not_configured"
    (config.REPORTS_DIR / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"loaded {len(demo)} claims; risk status {dict(statuses)}; llm {dict(llm_statuses)}")


if __name__ == "__main__":
    main()
