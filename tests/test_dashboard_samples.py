"""Tests for demo sample formatting (no Gradio needed)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loanlens.dashboard.samples import SAMPLES, build_payload, format_result  # noqa: E402

BASE = {
    "probability": 0.42,
    "shap_factors": [
        {"rank": 1, "label": "History of late installment payments", "shap_value": 0.8,
         "direction": "increases_risk", "cfpb_code": "A1 - Delinquent past or present credit obligations"},
        {"rank": 2, "label": "Other model factor (not a citable reason)", "shap_value": 0.1,
         "direction": "increases_risk", "cfpb_code": None},
    ],
    "adverse_action_codes": ["A1 - Delinquent past or present credit obligations"],
    "retrieved_passages": [{"source": "manual.pdf", "page": 3, "similarity_score": 0.71, "text": "x" * 300}],
    "adverse_action_notice": "Your application was declined.",
    "applicant_rights": "You may request a free credit report.",
    "generation_time_ms": 1200,
}


def test_decline_shows_notice_and_codes():
    decision, shap, notice, citations, meta, rights = format_result({**BASE, "decision": "decline"})
    assert decision.startswith("🔴 DECLINED") and "42.0%" in decision
    assert notice == "Your application was declined."
    assert "raises risk" in shap and "not a citable reason" in shap
    assert "A1 - Delinquent" in meta
    assert "similarity 0.71" in citations


def test_review_never_shows_a_notice():
    _, _, notice, _, _, _ = format_result({**BASE, "decision": "review"})
    assert "human underwriter" in notice


def test_approve_never_shows_a_notice():
    _, _, notice, _, _, _ = format_result({**BASE, "decision": "approve", "adverse_action_notice": None})
    assert "No adverse action notice" in notice


def test_payload_never_sends_prohibited_fields():
    for inputs in SAMPLES.values():
        payload = build_payload(*inputs)
        for field in ("age_years", "code_gender", "name_family_status"):
            assert field not in payload


def test_stored_demo_samples_match_sample_inputs():
    from loanlens.dashboard.samples import DEMO_SAMPLES_PATH
    if not DEMO_SAMPLES_PATH.exists():
        return
    stored = json.loads(DEMO_SAMPLES_PATH.read_text())
    assert set(stored) == set(SAMPLES)
    for name, entry in stored.items():
        assert entry["inputs"] == list(SAMPLES[name])
        if entry["result"]["decision"] != "decline":
            assert entry["result"]["adverse_action_notice"] is None
