"""
Demo samples and result formatting shared by the dashboard and the
script that regenerates demo_samples.json from the real pipeline.

No Gradio import here, so scripts can use it without building the UI.
"""

from __future__ import annotations

import json
from pathlib import Path

DEMO_SAMPLES_PATH = Path(__file__).parent / "demo_samples.json"

# (income, loan, annuity, years_employed, ext_score, overdue, refused, late_rate, debt_ratio)
SAMPLES: dict[str, tuple] = {
    "High Risk — High debt, late payments": (24000, 380000, 19000, 0.5, 0.12, 4, 3, 0.6, 0.92),
    "Medium Risk — Average borrower": (55000, 200000, 10000, 4.0, 0.42, 1, 1, 0.1, 0.4),
    "Low Risk — Stable borrower": (120000, 180000, 9000, 12.0, 0.74, 0, 0, 0.0, 0.15),
}


# The form has sliders for overdue accounts, refused applications and the
# late-payment rate, but not for how many records those come from. The demo
# assumes this credit history; it is shown on the page.
DEMO_HISTORY = {"bureau_loan_count": 5, "prev_app_count": 3, "instalment_count": 24}
HISTORY_NOTE = (
    "Assumed history for every form input: 5 credit bureau loans, "
    "3 previous applications, 24 past instalments."
)


def build_payload(income, loan, annuity, emp, ext, overdue, refused, late, debt) -> dict:
    """Dashboard inputs -> API request body (a consistent credit history)."""
    return {
        "bureau_loan_count": max(DEMO_HISTORY["bureau_loan_count"], int(overdue)),
        "prev_app_count": max(DEMO_HISTORY["prev_app_count"], int(refused)),
        "instalment_count": DEMO_HISTORY["instalment_count"],
        "amt_income_total": income,
        "amt_credit": loan,
        "amt_annuity": annuity,
        "employment_years": emp,
        "ext_source_2": ext,
        "bureau_overdue_count": int(overdue),
        "prev_refused_count": int(refused),
        "late_payment_rate": late,
        "bureau_debt_to_credit": debt,
    }


def load_demo_samples() -> dict:
    if DEMO_SAMPLES_PATH.exists():
        return json.loads(DEMO_SAMPLES_PATH.read_text())
    return {}


def format_result(result: dict, footer: str = "") -> tuple[str, str, str, str, str, str]:
    """API /explain response -> the six dashboard text boxes."""
    decision = result["decision"]
    pct = result["probability"] * 100
    label = {
        "decline": "🔴 DECLINED",
        "review": "🟡 SENT TO HUMAN REVIEW",
        "approve": "🟢 APPROVED",
    }[decision]
    decision_text = f"{label} — Estimated probability of default: {pct:.1f}%"

    lines = []
    for f in result.get("shap_factors", []):
        arrow = "raises risk" if f["direction"] == "increases_risk" else "lowers risk"
        code = f["cfpb_code"] or "not a citable reason"
        lines.append(f"{f['rank']}. {f['label']}\n   SHAP {f['shap_value']:+.3f} ({arrow}) | {code}")
    shap_text = "\n\n".join(lines) or "N/A"

    if decision == "decline":
        notice = result.get("adverse_action_notice") or "N/A"
    elif decision == "review":
        notice = "No adverse action notice: the application goes to a human underwriter."
    else:
        notice = "No adverse action notice required for approvals."

    passages = result.get("retrieved_passages", [])
    citations = "\n\n".join(
        f"📄 {p['source']} (p.{p['page']}) — similarity {p['similarity_score']:.2f}\n{p['text'][:200]}..."
        for p in passages
    ) or "N/A"

    meta = [f"Reason codes cited: {', '.join(result.get('adverse_action_codes') or []) or 'none'}"]
    if result.get("generation_time_ms"):
        meta.append(f"Notice generation time: {result['generation_time_ms']} ms")
    if result.get("regulatory_basis"):
        meta.append(f"Regulatory basis: {result['regulatory_basis']}")
    if footer:
        meta.append(footer)
    rights = result.get("applicant_rights") or "N/A"
    return decision_text, shap_text, notice, citations, "\n".join(meta), rights
