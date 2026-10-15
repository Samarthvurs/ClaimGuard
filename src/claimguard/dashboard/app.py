"""ClaimGuard reviewer dashboard (Streamlit). Uses the service layer directly, no HTTP hop.

Run: streamlit run src/claimguard/dashboard/app.py
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from claimguard import config  # noqa: E402
from claimguard.api.db import session_factory  # noqa: E402
from claimguard.services import assessment as svc  # noqa: E402

PAGES = ["Overview", "Review queue", "Evaluation"]
BANNER = "Synthetic data only. Decisions are made by humans."
DECISION_LABELS = {
    "accept_flag": "Accept flag",
    "dismiss": "Dismiss",
    "escalate": "Escalate",
}


def _read_json(name: str) -> dict[str, Any] | None:
    path = config.REPORTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def page_overview(session: Any) -> None:
    st.header("Overview")
    s = svc.summary(session)
    if not s["claims_assessed"]:
        st.info("No assessed claims yet. Run `python scripts/load_demo_data.py`.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claims assessed", s["claims_assessed"])
    c2.metric("Flagged for review", _pct(s["pct_flagged"]))
    c3.metric("Average risk", _pct(s["avg_risk"]))
    c4.metric("Reviewer acceptance rate", _pct(s["reviewer_acceptance_rate"]))
    st.caption(
        "Risk = model-predicted denial probability on synthetic claims with rule-derived "
        "labels. Flagged = NEEDS_REVIEW or HIGH_RISK."
    )

    left, right = st.columns(2)
    payer = pd.DataFrame(
        [{"payer": p, "average risk": v["avg_risk"]} for p, v in s["risk_by_payer"].items()]
    )
    left.subheader("Average risk by payer")
    left.bar_chart(payer, x="payer", y="average risk", horizontal=True)
    right.subheader("Likely-denied claims by predicted reason")
    reasons = pd.DataFrame(
        [{"reason": k, "claims": v} for k, v in s["predicted_reason_counts"].items()]
    )
    if reasons.empty:
        right.write("No claims above the decision threshold.")
    else:
        right.bar_chart(reasons, x="reason", y="claims", horizontal=True)
    st.subheader("Status counts")
    st.dataframe(pd.DataFrame([s["status_counts"]]), hide_index=True)
    llm_counts = s["llm_status_counts"]
    if llm_counts.get("not_configured"):
        st.info("LLM not configured: assessments show the ML result only.")


def _doc_check_section(detail: dict[str, Any]) -> None:
    st.subheader("Documentation check")
    llm_status = detail["llm_status"]
    doc = detail["doc_check"]
    if llm_status == "not_configured":
        st.info(
            "LLM not configured. Add `GEMINI_API_KEY` to `.env` to enable the documentation "
            "check. The ML risk result above is still valid."
        )
        return
    if llm_status in ("unavailable", "skipped") or doc is None:
        st.warning(f"Documentation check not available for this claim ({llm_status}).")
        return
    if llm_status == "invalid_output":
        st.warning("The LLM returned invalid output twice; please check the note manually.")
        return
    st.caption(f"Model `{doc['model']}`, prompt {doc['prompt_version']}. Evidence is quoted.")
    items = pd.DataFrame(doc["items"])
    items["needs review"] = items["item_id"].isin(doc["needs_review"])
    st.dataframe(items, hide_index=True, width="stretch")
    st.write(doc["summary"])
    if doc["draft_request"]:
        st.text_area("Draft request for missing documentation", doc["draft_request"], height=100)


def _review_section(session: Any, detail: dict[str, Any]) -> None:
    st.subheader("Reviewer decision")
    for r in detail["reviews"]:
        st.write(f"- {DECISION_LABELS[r['decision']]} ({r['created_at'][:16]}): {r['comment']}")
    comment = st.text_input("Comment", key=f"comment-{detail['id']}")
    cols = st.columns(3)
    for col, (decision, label) in zip(cols, DECISION_LABELS.items(), strict=True):
        if col.button(label, key=f"{decision}-{detail['id']}"):
            svc.review_assessment(session, detail["id"], decision, comment)
            st.success(f"Recorded: {label}")
            st.rerun()


def _factor_chart(factors: pd.DataFrame) -> plt.Figure:
    """Horizontal SHAP bars, largest at the top; red raises risk, blue lowers it."""
    data = factors.iloc[::-1]
    colors = ["#c0392b" if v > 0 else "#2e6da4" for v in data["shap_value"]]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(data) + 0.8))
    ax.barh(data["description"], data["shap_value"], color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("SHAP value (log-odds); positive raises denial risk")
    fig.tight_layout()
    return fig


def page_review_queue(session: Any) -> None:
    st.header("Review queue")
    rows: list[Any] = []
    for status in svc.FLAGGED_STATUSES:
        claims, _ = svc.list_claims(session, status=status, limit=500)
        rows.extend(claims)
    if not rows:
        st.info("No flagged claims.")
        return
    queue = pd.DataFrame([svc.claim_summary(c) for c in rows]).sort_values(
        "denial_probability", ascending=False
    )
    st.dataframe(
        queue[
            [
                "external_id",
                "status",
                "denial_probability",
                "predicted_reason",
                "payer",
                "procedure_category",
                "service_date",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    options = queue["id"].tolist()
    labels = dict(zip(queue["id"], queue["external_id"], strict=True))
    claim_id = st.selectbox("Open claim", options, format_func=lambda i: labels[i])
    claim = svc.get_claim(session, int(claim_id))
    detail = svc.claim_detail(claim)
    a = detail["latest_assessment"]

    st.markdown(f"### {claim.external_id}: {claim.procedure_description}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Denial risk", f"{a['denial_probability']:.1%}")
    c2.metric("Risk status", a["risk_status"])
    c3.metric("Predicted reason", a["predicted_reason"])
    st.caption(a["disclaimer"])

    st.subheader("Top factors (SHAP)")
    factors = pd.DataFrame(a["factors"])
    fig = _factor_chart(factors)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(
        factors[["description", "direction", "shap_value"]],
        hide_index=True,
        width="stretch",
    )
    with st.expander("Redacted clinical note"):
        st.text(detail["note_redacted"] or "")
    _doc_check_section(a)
    _review_section(session, a)


def page_evaluation() -> None:
    st.header("Evaluation")
    st.caption("All results come from synthetic data with rule-derived labels (`reports/`).")
    metrics = _read_json("metrics.json")
    if metrics is None:
        st.info("TBD (not yet measured): run `python -m claimguard.ml.train`.")
    else:
        lgbm = metrics["lightgbm"]["test"]
        lr = metrics["logistic_regression"]["test"]
        base = metrics["majority_baseline"]
        table = pd.DataFrame(
            [
                {"model": "LightGBM", **{k: lgbm[k] for k in ("pr_auc", "roc_auc", "f1", "brier")}},
                {
                    "model": "Logistic regression",
                    **{k: lr[k] for k in ("pr_auc", "roc_auc", "f1", "brier")},
                },
                {
                    "model": "Majority class",
                    **{k: base[k] for k in ("pr_auc", "roc_auc", "f1", "brier")},
                },
            ]
        )
        st.subheader("Model metrics (time-based test set)")
        st.dataframe(table, hide_index=True)
        st.write(
            f"Threshold {lgbm['threshold']} ({metrics['threshold_rule']}): precision "
            f"{lgbm['precision']}, recall {lgbm['recall']}. {metrics['data_note']}"
        )
        cols = st.columns(3)
        for col, (name, caption) in zip(
            cols,
            [
                ("calibration.png", "Calibration"),
                ("pr_curve.png", "Precision-recall"),
                ("shap_summary.png", "SHAP summary"),
            ],
            strict=True,
        ):
            path = config.REPORTS_DIR / name
            if path.exists():
                col.image(str(path), caption=caption)
        st.subheader("Slices")
        st.dataframe(pd.DataFrame(metrics["slices"]["by_payer"]), hide_index=True)
        st.dataframe(pd.DataFrame(metrics["slices"]["by_age_band"]), hide_index=True)

    st.subheader("Prompt comparison (LLM documentation check)")
    prompt_eval = _read_json("prompt_eval.json")
    if prompt_eval is None:
        st.info("TBD (not yet measured): add a Gemini key and run `python -m claimguard.llm.eval`.")
        return
    rows = []
    for version, m in prompt_eval["versions"].items():
        if m["status"] != "complete":
            rows.append({"version": version, "status": m["status"]})
            continue
        rows.append(
            {
                "version": version,
                "status": "complete",
                "micro_f1": m["micro"]["f1"],
                "precision": m["micro"]["precision"],
                "recall": m["micro"]["recall"],
                "invalid_output_rate": m["invalid_output_rate"],
                "grounding_rate": m["grounding_rate"],
                "injection_success_rate": m["injection_success_rate"],
                "avg_latency_s": m["avg_latency_s"],
            }
        )
    st.caption(f"Model: {prompt_eval['model']}. {prompt_eval['n_notes']} synthetic notes.")
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="ClaimGuard", layout="wide")
    st.title("ClaimGuard")
    st.warning(BANNER)
    page = st.sidebar.radio("Page", PAGES)
    st.sidebar.caption("Flags risk and missing documentation. Never approves or denies a claim.")
    session = session_factory()()
    try:
        if page == "Overview":
            page_overview(session)
        elif page == "Review queue":
            page_review_queue(session)
        else:
            page_evaluation()
    finally:
        session.close()


main()
