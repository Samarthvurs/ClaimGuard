from __future__ import annotations

import numpy as np
import pandas as pd

from claimguard.data import build_claims, fallback_generate, labeling, notes


def _claim(**overrides: object) -> pd.DataFrame:
    row = {
        "procedure_code": "P1",
        "condition_code": "C1",
        "payer": "Aetna",
        "prior_auth_required": False,
        "prior_auth_on_file": False,
        "documents_attached_count": 2,
        "min_documents": 2,
        "days_to_submission": 10,
    }
    row.update(overrides)
    return pd.DataFrame([row])


COMPAT = pd.DataFrame({"procedure_code": ["P1"], "condition_code": ["C1"]})
RULES = labeling.load_rules()


def test_rules_fire_in_order() -> None:
    assert labeling.rule_reason(_claim(), COMPAT, RULES).iloc[0] == "NONE"
    assert (
        labeling.rule_reason(_claim(prior_auth_required=True), COMPAT, RULES).iloc[0]
        == "MISSING_PRIOR_AUTH"
    )
    assert labeling.rule_reason(_claim(condition_code="X"), COMPAT, RULES).iloc[0] == (
        "CODING_MISMATCH"
    )
    assert (
        labeling.rule_reason(_claim(documents_attached_count=1), COMPAT, RULES).iloc[0]
        == "MISSING_DOCUMENTATION"
    )
    assert labeling.rule_reason(_claim(days_to_submission=91), COMPAT, RULES).iloc[0] == (
        "TIMELY_FILING"
    )
    # Medicare has a longer filing limit.
    late_medicare = _claim(days_to_submission=200, payer="Medicare")
    assert labeling.rule_reason(late_medicare, COMPAT, RULES).iloc[0] == "NONE"
    # Prior auth outranks every other rule.
    many = _claim(prior_auth_required=True, condition_code="X", days_to_submission=500)
    assert labeling.rule_reason(many, COMPAT, RULES).iloc[0] == "MISSING_PRIOR_AUTH"


def test_label_noise_rate_close_to_config(claims: pd.DataFrame) -> None:
    rate = claims["label_flipped"].mean()
    assert 0.02 < rate < 0.08
    clean = claims[~claims["label_flipped"]]
    assert ((clean["rule_reason"] != "NONE").astype(int) == clean["denied"]).all()


def test_build_is_deterministic(tmp_path) -> None:
    a = fallback_generate.generate(n_patients=100, n_claims=400, out_dir=tmp_path / "a")
    b = fallback_generate.generate(n_patients=100, n_claims=400, out_dir=tmp_path / "b")
    ca = build_claims.build(a)["claims"]
    cb = build_claims.build(b)["claims"]
    pd.testing.assert_frame_equal(ca, cb)


def test_history_features_use_only_earlier_claims() -> None:
    df = pd.DataFrame(
        {
            "provider_id": ["A", "A", "A", "A"],
            "patient_id": ["p", "p", "p", "q"],
            "service_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-02-01", "2020-03-01"]
            ),
            "denied": [1, 1, 0, 1],
        }
    )
    out = build_claims.add_history_features(df)
    # Same-day claims do not see each other.
    assert out["patient_prior_denials"].tolist() == [0, 0, 2, 0]
    prior = build_claims.HISTORY_PRIOR_RATE * build_claims.HISTORY_PRIOR_WEIGHT
    w = build_claims.HISTORY_PRIOR_WEIGHT
    assert np.isclose(out["provider_prior_denial_rate"].iloc[0], prior / w)
    assert np.isclose(out["provider_prior_denial_rate"].iloc[2], (2 + prior) / (2 + w))


def test_claims_table_has_no_label_leakage_columns(claims: pd.DataFrame) -> None:
    from claimguard.ml.features import FEATURES

    for leak in ("denied", "denial_reason", "rule_reason", "label_flipped"):
        assert leak not in FEATURES
    assert claims["claim_id"].is_unique


def test_categorize_keywords() -> None:
    assert build_claims.categorize("Plain X-ray of chest (procedure)") == "imaging"
    assert build_claims.categorize("Total knee replacement (procedure)") == "surgery"
    assert build_claims.categorize("Physical therapy procedure") == "therapy"
    assert build_claims.categorize("Depression screening (procedure)") == "routine"
    assert build_claims.categorize("Spirometry (procedure)") == "diagnostic"


def test_notes_ground_truth_matches_text(claims: pd.DataFrame) -> None:
    records = notes.build_notes(claims.head(200), n_adversarial=5)
    checklists = notes.load_checklists()
    assert sum(r["adversarial"] for r in records) == 5
    for rec in records:
        ids = {i["id"] for i in checklists[rec["category"]]}
        assert set(rec["ground_truth"]) == ids
        if rec["adversarial"]:
            assert list(rec["ground_truth"].values()).count("missing") >= 2


def test_every_item_has_five_phrasings() -> None:
    checklists = notes.load_checklists()
    for items in checklists.values():
        assert 3 <= len(items) <= 5
        for item in items:
            assert len(notes.PHRASINGS[item["id"]]) >= 5


def test_missing_items_have_no_phrasing_in_note() -> None:
    rng = np.random.default_rng(0)
    checklist = notes.load_checklists()["surgery"]
    claim = {"condition_description": "Gout", "procedure_description": "Repair", "age": 50}
    for _ in range(30):
        text, truth = notes.generate_note(claim, checklist, rng)
        consent_phrases = [p.split(";")[0][:20] for p in notes.PHRASINGS["informed_consent"]]
        if truth["informed_consent"] == "missing":
            assert not any(p in text for p in consent_phrases)
