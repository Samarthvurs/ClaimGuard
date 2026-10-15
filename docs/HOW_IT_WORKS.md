# How it works: one claim, start to finish

This follows one real synthetic claim from the demo database, `CLM-028392` (saved in
`reports/example_assessment.json`). Everything about it is synthetic.

## 1. The claim arrives

A 13-year-old boy (Synthea patient) has an ambulatory visit on 2026-04-13 for **cognitive and
behavioural therapy**, linked to *child attention deficit disorder*. The payer is Blue Cross Blue
Shield and the claim cost is $208.38.

The simulated administrative fields say: no prior authorization is required for this procedure;
the therapy category needs **at least 2 documents but only 1 is attached**; the claim is being
submitted 33 days after the visit (payer limit 90); the provider's past denial rate is about 17%.

Through the API this is `POST /claims`. Before anything is stored, the clinical note is
**redacted**: the name, date of birth, record number, phone number and dates become `[NAME]`,
`[DATE]`, `[ID]`, `[PHONE]`. Only the redacted text and its hash are saved in SQLite.

## 2. The ML model scores it

`POST /claims/{id}/assess` calls the service layer, which loads the saved scikit-learn pipeline
once and turns the raw fields into features (for example "documents short = 1"). LightGBM returns a
denial probability of **0.94**, above the decision threshold of 0.6355, so the claim is "likely
denied". The reason model says the most likely reason is **MISSING_DOCUMENTATION** (0.9992).

## 3. SHAP explains why

TreeSHAP splits the score into contributions per feature (in log-odds), summed back to the raw
fields and written in plain language. The top five for this claim:

| Factor | Effect |
|---|---|
| 1 fewer document(s) than required | increases risk (+4.65) |
| Procedure needs at least 2 document(s) | increases risk (+0.21) |
| Payer: Blue Cross Blue Shield | increases risk (+0.20) |
| Submitted 33 days after service (payer limit 90) | decreases risk (-0.17) |
| 1 document(s) attached (minimum 2) | increases risk (+0.16) |

The reviewer can see at a glance that the problem is paperwork, not coding or timing.

## 4. The LLM checks the note (when a key is configured)

The therapy checklist has three items: treatment plan, functional goals and progress measurement.
The redacted note reads, in part: "Therapy plan documented: cognitive and behavioral therapy for 12
visits. Medication list reviewed. Baseline pain score 7/10."

With a Gemini key, the checker sends the checklist and the note (inside `<note>` tags, treated as
untrusted data) using prompt v3, at temperature 0. It expects JSON with a status, a verbatim quote
and a confidence per item. The code then checks every answer: item ids must come from the
checklist, every "present" quote must actually appear in the redacted note (the **grounding
check**), confidences must be between 0 and 1, and the text must not say "approve" or "deny". A bad
answer is retried once; a second bad answer becomes `invalid_output`. Unclear or low-confidence
items are routed to the reviewer, and the model drafts a short request to the provider for what is
missing. In this note the ground truth is: treatment plan present, progress measurement
present, functional goals missing, so a correct check would ask the provider for functional goals.

Tonight no key was configured, so this claim's assessment shows `llm_status = "not_configured"`
and the dashboard says "LLM not configured" while still showing the ML result.

## 5. A risk status is set

A probability of 0.60 or more gives **HIGH_RISK** (0.30 to 0.60 is NEEDS_REVIEW; below 0.30 is
LOW_RISK, lifted to NEEDS_REVIEW if the LLM finds missing or unclear documentation). Everything is
stored in the `assessments` and `doc_checks` tables.

## 6. A human decides

The claim appears in the dashboard's **Review queue**. The reviewer reads the factors, the redacted
note and (when available) the documentation table, then clicks **Accept flag**, **Dismiss** or
**Escalate** with a comment. That decision is stored in `reviews`. ClaimGuard never approves or
denies the claim itself.
