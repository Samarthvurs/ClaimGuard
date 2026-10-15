from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from claimguard.api.db import session_factory
from claimguard.services import assessment as svc
from tests.test_api import CLAIM

APP = str(Path(__file__).resolve().parents[1] / "src" / "claimguard" / "dashboard" / "app.py")


@pytest.fixture()
def seeded(app_env) -> None:
    session = session_factory()()
    for i, (prior_auth, docs) in enumerate([(False, 0), (True, 3), (False, 1)]):
        claim = svc.create_claim(
            session,
            {
                **CLAIM,
                "external_id": f"DASH-{i}",
                "prior_auth_on_file": prior_auth,
                "documents_attached_count": docs,
                "days_to_submission": 200 if i == 0 else 10,
            },
        )
        svc.assess_claim(session, claim.id)
    session.close()


def _run(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", ["Overview", "Review queue", "Evaluation"])
def test_page_renders_without_exceptions(seeded, page: str) -> None:
    at = _run(page)
    assert not at.exception, at.exception
    assert any("Synthetic data only" in w.value for w in at.warning)


def test_review_queue_shows_llm_not_configured(seeded) -> None:
    at = _run("Review queue")
    infos = " ".join(i.value for i in at.info)
    assert "No flagged claims" not in infos
    assert "LLM not configured" in infos


def test_review_button_records_decision(seeded) -> None:
    at = _run("Review queue")
    at.text_input[0].set_value("needs prior auth")
    next(b for b in at.button if b.label == "Escalate").click().run()
    assert not at.exception
    session = session_factory()()
    summary = svc.summary(session)
    session.close()
    assert summary["review_decisions"] == {"escalate": 1}


def test_empty_database_overview(app_env) -> None:
    at = _run("Overview")
    assert not at.exception
    assert any("No assessed claims" in i.value for i in at.info)
