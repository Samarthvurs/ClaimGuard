# Limitations

ClaimGuard is a learning project. Read every result with these limits in mind.

- **Simulated labels.** Denials come from four rules plus 5% random noise, not from payers. The
  model mostly rediscovers the rules; its scores show the pipeline works, not that it predicts real
  denials.
- **Simulated administrative fields.** Prior authorization, attached documents, submission delay
  and coding errors are random draws with a fixed seed. Real claims have many more fields,
  missing values and payer-specific behaviour.
- **Simulated notes.** Notes are built from templates with known ground truth. Real notes are
  longer, messier, use abbreviations, and rarely map cleanly to a checklist. Template notes can
  also be clinically odd (for example a pain score in a behavioural therapy note). LLM metrics
  on these notes will overstate real-world accuracy.
- **Simple redaction.** `llm/redact.py` uses regular expressions for emails, phone numbers,
  ID-like numbers, dates and the generator's fake names. It is **not production-grade PII
  protection** and would miss many real identifiers (addresses, unusual names, free-text
  mentions).
- **No real payer rules.** Prior-auth lists, document minimums and filing limits are invented for
  the demo. Codes are Synthea's SNOMED-CT codes; no CPT or ICD accuracy is claimed.
- **Small scale.** 30,000 claims from 3,000 synthetic patients; 300 claims in the demo database;
  the prompt evaluation uses about 100 notes.
- **LLM results are on template notes.** The prompt evaluation used 100 synthetic notes with
  known answers. Real notes are messier, so real-world accuracy would be lower. The deterministic
  mock used in tests is keyed to the templates, so its outputs are never reported as results.
- **Demo security only.** One shared API key in `.env`, SQLite, no user accounts, no audit trail
  beyond review records. Not for public deployment. Not HIPAA compliant.
- **Human decision required.** The system never approves or denies a claim; it only flags risk and
  suggests missing documentation.
