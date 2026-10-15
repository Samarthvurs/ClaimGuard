# Decisions log

One line per decision: what, and why.

- Repo root is the project folder itself (not a nested `claimguard/` dir): simpler paths, same layout otherwise.
- Python 3.11 venv (3.14 is the system default): best wheel availability for LightGBM, SHAP and Streamlit.
- Default `LLM_MODEL` is `gemini-2.5-flash-lite`: cheapest instruction-following Flash-Lite model on the OpenAI-compatible endpoint; overridable via `.env`.
- Data source: Synthea `master-branch-latest` jar (build d9d07a6, built 2026-08-18), downloaded 2026-10-15, run with `-p 3000 -s 42 -cs 42`, CSV export only. Java 25 was available, so no fallback was needed.
- Synthea output row counts: 3,433 patients (3,000 alive + 433 deceased), 202,990 encounters, 548,888 procedures, 121,980 conditions, 10 payers, 1,014 providers.
- Claims = one row per encounter with a procedure, using the encounter's highest-cost procedure as the principal one; encounters from 2016-01-01 onward only (earlier years are sparse).
- Encounters with payer `NO_INSURANCE` dropped: no claim would be sent to a payer.
- Claims capped at 30,000 by a seeded random sample, to keep training and SHAP fast on a laptop.
- Linked condition = procedure reason, else encounter reason, else first condition recorded at the encounter, else `NONE`.
- `procedure_policy.csv` covers the 150 most frequent procedures; other procedures get a category from the same keyword rules, no prior-auth requirement, and the category's minimum documents.
- Simulated provider "documentation sloppiness" (Beta(1.2, 6) per provider) raises the chance of missing prior auth, missing documents and late filing, so provider history is a real (simulated) signal.
- Simulated coding errors: 4% of claims get the condition of a random other claim; compatibility table = pairs seen at least 5 times before the swaps (rare natural pairs also count as mismatches).
- Label noise flips 5% of labels; a label flipped to denied gets a random reason, one flipped to not denied gets `NONE`.
- History features use a smoothed rate (prior 0.2, weight 5) over strictly earlier service dates; same-day claims do not see each other.
- Every claim gets a note (routine claims too) so the API and dashboard always have a note to show; 12 notes are adversarial.
- Fallback generator writes Synthea-shaped CSVs (same columns) so the builder code is identical for both sources.
