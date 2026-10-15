from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from claimguard.api.main import app

HEADERS = {"X-API-Key": "test-key"}

CLAIM = {
    "external_id": "TEST-1",
    "age": 58,
    "sex": "F",
    "payer": "Aetna",
    "encounter_class": "outpatient",
    "procedure_category": "imaging",
    "provider_specialty": "GENERAL PRACTICE",
    "procedure_code": "418891003",
    "procedure_description": "Computed tomography of chest and abdomen (procedure)",
    "condition_code": "254637007",
    "condition_description": "Non-small cell lung cancer (disorder)",
    "service_date": "2025-06-01",
    "claim_cost": 2400.0,
    "prior_auth_required": True,
    "prior_auth_on_file": False,
    "documents_attached_count": 1,
    "min_documents": 2,
    "days_to_submission": 20,
    "provider_prior_denial_rate": 0.3,
    "patient_prior_denials": 1,
    "note": "Patient: Avery Ashdown. Phone: 555-123-4567. Reason for study: suspected "
    "lung cancer. Symptoms have been present for 6 weeks.",
}


@pytest.fixture()
def client(app_env) -> TestClient:
    with TestClient(app) as c:
        yield c


def test_health_is_open(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_api_key_required(client: TestClient) -> None:
    r = client.get("/claims")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"
    assert client.get("/claims", headers={"X-API-Key": "wrong"}).status_code == 401


def test_create_assess_review_flow(client: TestClient) -> None:
    r = client.post("/claims", json=CLAIM, headers=HEADERS)
    assert r.status_code == 201, r.text
    created = r.json()
    assert "Avery" not in created["note_redacted"]
    assert "555-123-4567" not in created["note_redacted"]
    claim_id = created["id"]

    r = client.post(f"/claims/{claim_id}/assess", headers=HEADERS)
    assert r.status_code == 200, r.text
    a = r.json()
    assert 0 <= a["denial_probability"] <= 1
    assert a["risk_status"] in {"LOW_RISK", "NEEDS_REVIEW", "HIGH_RISK"}
    assert a["llm_status"] == "not_configured"
    assert a["doc_check"] is None
    assert len(a["factors"]) == 5
    assert "does not approve or deny" in a["disclaimer"]

    r = client.post(
        f"/assessments/{a['id']}/review",
        json={"decision": "escalate", "comment": "check prior auth"},
        headers=HEADERS,
    )
    assert r.status_code == 201
    assert r.json()["decision"] == "escalate"

    detail = client.get(f"/claims/{claim_id}", headers=HEADERS).json()
    assert detail["latest_assessment"]["reviews"][0]["decision"] == "escalate"

    listing = client.get("/claims", params={"payer": "Aetna"}, headers=HEADERS).json()
    assert listing["total"] == 1 and listing["items"][0]["id"] == claim_id
    assert client.get("/claims", params={"payer": "Nobody"}, headers=HEADERS).json()["total"] == 0

    s = client.get("/summary", headers=HEADERS).json()
    assert s["claims_assessed"] == 1 and s["reviews_total"] == 1
    assert s["reviewer_acceptance_rate"] == 0.0


def test_assess_with_mock_llm_stores_doc_check(app_env) -> None:
    from claimguard.api.db import session_factory
    from claimguard.llm.mock import MockProvider
    from claimguard.services import assessment as svc

    session = session_factory()()
    claim = svc.create_claim(session, {**CLAIM, "external_id": "TEST-LLM"})
    a = svc.assess_claim(session, claim.id, provider=MockProvider())
    assert a.llm_status == "ok"
    detail = svc.assessment_detail(a)
    statuses = {i["item_id"]: i["status"] for i in detail["doc_check"]["items"]}
    assert statuses["clinical_indication"] == "present"
    assert statuses["prior_conservative_treatment"] == "missing"
    assert a.risk_status in {"NEEDS_REVIEW", "HIGH_RISK"}
    session.close()


def test_llm_failure_still_returns_ml_result(app_env) -> None:
    from claimguard.api.db import session_factory
    from claimguard.llm.mock import MockProvider
    from claimguard.llm.provider import LLMError
    from claimguard.services import assessment as svc

    session = session_factory()()
    claim = svc.create_claim(session, {**CLAIM, "external_id": "TEST-FAIL"})
    a = svc.assess_claim(session, claim.id, provider=MockProvider([LLMError("down")]))
    assert a.llm_status == "unavailable"
    assert 0 <= a.denial_probability <= 1
    session.close()


def test_validation_errors(client: TestClient) -> None:
    bad = {**CLAIM, "age": -3, "procedure_category": "magic"}
    r = client.post("/claims", json=bad, headers=HEADERS)
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "validation_error"
    fields = {d["loc"][-1] for d in body["details"]}
    assert {"age", "procedure_category"} <= fields
    r = client.post("/assessments/1/review", json={"decision": "approve"}, headers=HEADERS)
    assert r.status_code == 422
    assert client.get("/claims", params={"limit": 500}, headers=HEADERS).status_code == 422


def test_not_found_and_duplicate(client: TestClient) -> None:
    assert client.get("/claims/999", headers=HEADERS).status_code == 404
    assert client.post("/claims/999/assess", headers=HEADERS).status_code == 404
    r = client.post("/assessments/999/review", json={"decision": "dismiss"}, headers=HEADERS)
    assert r.status_code == 404
    assert client.post("/claims", json=CLAIM, headers=HEADERS).status_code == 201
    dup = client.post("/claims", json=CLAIM, headers=HEADERS)
    assert dup.status_code == 409 and dup.json()["error"]["code"] == "conflict"


def test_risk_status_bands() -> None:
    from claimguard.services.assessment import risk_status

    assert risk_status(0.1) == "LOW_RISK"
    assert risk_status(0.1, doc_items_need_attention=True) == "NEEDS_REVIEW"
    assert risk_status(0.45) == "NEEDS_REVIEW"
    assert risk_status(0.9) == "HIGH_RISK"
