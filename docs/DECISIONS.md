# Decisions log

One line per decision: what, and why.

- Repo root is the project folder itself (not a nested `claimguard/` dir): simpler paths, same layout otherwise.
- Python 3.11 venv (3.14 is the system default): best wheel availability for LightGBM, SHAP and Streamlit.
- Default `LLM_MODEL` is `gemini-2.5-flash-lite`: cheapest instruction-following Flash-Lite model on the OpenAI-compatible endpoint; overridable via `.env`.
