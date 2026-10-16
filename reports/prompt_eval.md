# Prompt evaluation

Model: `gemini-flash-lite-latest`. Notes: 100 synthetic notes (12 adversarial) with template ground truth. Synthetic data only; positive class = item missing.

| Version | Status | Micro P | Micro R | Micro F1 | Invalid output | Grounding | Injection success | Avg latency (s) |
|---|---|---|---|---|---|---|---|---|
| v1 | complete | 0.84 | 0.7778 | 0.8077 | 0.0 | 0.9893 | 0.3333 | 1.703 |
| v2 | complete | 0.8837 | 0.962 | 0.9212 | 0.01 | 1.0 | 0.0833 | 1.399 |
| v3 | complete | 0.8438 | 1.0 | 0.9153 | 0.0 | 1.0 | 0.0 | 1.603 |
