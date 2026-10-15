# Model card: ClaimGuard denial risk model

All numbers come from `reports/metrics.json` and `reports/model_meta.json`, produced by
`python -m claimguard.ml.train`. **They come from synthetic data with rule-derived labels.**

## Intended use

Flag claims that look likely to be denied **before submission**, and explain why, so a human
reviewer can fix the claim. It does not approve, deny or pay anything. Learning and portfolio use
only; not validated on real claims.

## Data

30,000 synthetic claims (Synthea structure plus simulated administrative fields). Labels come from
four documented rules plus 5% random label noise. See [DATA_CARD.md](DATA_CARD.md).

## Features

Numeric: age, log claim cost, prior-auth required, prior-auth on file, documents attached,
documents required, documents short, days to submission, payer filing limit, days past the limit,
provider's past denial rate, patient's past denials. Categorical (one-hot, rare levels grouped):
sex, payer, encounter class, procedure category, provider specialty. Plus the log frequency of the
procedure-condition pair in the training data (a proxy for coding mismatches, fitted inside the
pipeline so test pairs are never counted). All preprocessing lives in one saved scikit-learn
`Pipeline`.

## Split and why

Time-based by service date:

| Part | Dates | Claims |
|---|---|---|
| Train | 2016-01-02 to 2023-05-12 | 19,194 |
| Validation | 2023-05-13 to 2024-11-21 | 4,801 |
| Test | 2024-11-22 to 2026-10-15 | 6,005 |

A random split would let the model learn from claims that happen after the ones it is tested on.
In real use the model always scores future claims, and the history features (provider and patient
past denials) only make sense looking backwards. Train dates end before validation dates begin,
and validation ends before test begins (checked in tests).

## Models

- **Baseline:** logistic regression (scaled numeric features), C chosen from {0.1, 1, 10}; best
  C = 1.0.
- **Served model:** LightGBM, chosen from a small seeded grid (num_leaves {15, 31}, learning rate
  {0.05, 0.1}, trees {200, 400}) by validation PR-AUC; best: 31 leaves, learning rate 0.05,
  200 trees. Seed 42.
- **Predicted reason:** a multiclass LightGBM trained on denied training claims only; shown only
  when the claim is above the threshold.

## Metrics (test set, positive rate 0.1962)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Brier |
|---|---|---|---|---|---|---|
| LightGBM | 0.8157 | 0.8913 | 0.8351 | 0.9221 | 0.7632 | 0.0546 |
| Logistic regression | 0.7928 | 0.8776 | 0.7843 | 0.8625 | 0.7190 | 0.0683 |
| Majority class (always "not denied") | 0.1962 | 0.5 | 0.0 | n/a | n/a | 0.1577 |

LightGBM confusion matrix at the threshold: TN 4,751, FP 76, FN 279, TP 899.

Predicted reason on 1,178 denied test claims: accuracy 0.8192, macro F1 0.7972.

**Ceiling note:** 5% of all labels are flipped at random, so about one in five "denied" labels in
the test set is pure noise that no model can predict. That is consistent with recall stopping
around 0.76.

## Threshold

Rule: the probability that maximises F1 on the validation slice. Value: **0.6355**, stored in the
model artifact and `reports/model_meta.json`. Separately, review bands in `config.py` route claims:
LOW_RISK below 0.30, NEEDS_REVIEW from 0.30, HIGH_RISK from 0.60.

## Calibration

Isotonic calibration was fitted on the first half of validation and judged on the second half.
Brier improved only from 0.0501 to 0.0498 (relative gain 0.0056, below the 5% rule), so it was
**not adopted**. The reliability plot is in `reports/calibration.png`.

## Slices (test set, same threshold)

No payer or age band differs from the overall recall or F1 by more than 0.10 (the rule used to
flag gaps; slices with at least 50 claims). F1 by payer ranges from 0.7778 (Anthem, n = 306) to
0.8947 (Cigna Health, n = 419). F1 by age band ranges from 0.8197 (65+) to 0.8427 (45-64). Full
tables are in `reports/metrics.json` and the dashboard's Evaluation page.

## Explanations

TreeSHAP values for the LightGBM model, summed per raw feature (one-hot columns added back), top
five per claim with direction and size, described in plain language. Global importance:
`reports/shap_summary.png`. The largest global factors are missing documents, the
procedure-condition pairing and days past the filing limit, which match the rules that generated
the labels.

One honest oddity: for some patients with many past claims (for example frequent dialysis), a
high count of past denials pushes risk **down**. It is a pattern of the simulation (frequent,
routine claims), not a real-world insight, and shows why SHAP explains the model rather than the
world.

## Limitations

- Labels come from rules, so the model mostly rediscovers those rules. High scores show the
  pipeline works; they say nothing about real denial prediction.
- Administrative fields are simulated; real data would bring messy codes, payer-specific policies
  and missing values.
- No fairness claims: slices are reported to show the method, on synthetic people.
- See [LIMITATIONS.md](LIMITATIONS.md).
