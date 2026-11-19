"""Edge cases: implausible input, LLM outage during a decline, notice reasons."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loanlens.api.schemas import ScoreRequest  # noqa: E402
from loanlens.rag.citations import check_reasons  # noqa: E402

BASE = {"amt_income_total": 50000, "amt_credit": 100000}


# ── 1. Implausible input ──────────────────────────────────────

class TestPlausibleInput:
    @pytest.mark.parametrize("field, value", [
        ("amt_income_total", 1e13), ("amt_credit", 5e8), ("employment_years", 120),
        ("bureau_loan_count", 10_000), ("instalment_count", 1_000_000), ("debt_to_income", 1e6),
        ("bureau_oldest_credit_days", -1e7),
    ])
    def test_absurd_values_rejected(self, field, value):
        with pytest.raises(ValidationError, match=field):
            ScoreRequest(**{**BASE, field: value})

    @pytest.mark.parametrize("extra, message", [
        ({"amt_annuity": 200000}, "amt_annuity"),
        ({"instalment_count": 10, "late_payment_count": 11}, "late_payment_count cannot exceed"),
        ({"bureau_loan_count": 2, "bureau_closed_loans": 3}, "bureau_closed_loans cannot exceed"),
        ({"bureau_oldest_credit_days": -100, "bureau_newest_credit_days": -2000}, "must be earlier"),
    ])
    def test_impossible_combinations_rejected(self, extra, message):
        with pytest.raises(ValidationError, match=message):
            ScoreRequest(**BASE, **extra)

    def test_largest_realistic_application_accepted(self):
        ScoreRequest(amt_income_total=1.17e8, amt_credit=4.05e6, amt_annuity=2.58e5,
                     employment_years=49, bureau_loan_count=116, instalment_count=372)


# ── 3. Notice states the model's reasons, and only those ──────

REASONS = ["External credit score assessment", "History of late installment payments"]
OTHERS = REASONS + ["Debt-to-income ratio too high", "Insufficient length of employment"]


class TestCheckReasons:
    def test_all_reasons_stated(self):
        notice = "Declined. The principal reasons are: external credit score assessment; history of late installment payments."
        r = check_reasons(notice, REASONS, OTHERS)
        assert r["ok"] and r["missing"] == [] and r["extra"] == []

    def test_missing_reason_flagged(self):
        r = check_reasons("Declined because of an external credit score assessment.", REASONS, OTHERS)
        assert not r["ok"] and r["missing"] == ["History of late installment payments"]

    def test_added_reason_flagged(self):
        notice = ("External credit score assessment; history of late installment payments; "
                  "debt-to-income ratio too high.")
        r = check_reasons(notice, REASONS, OTHERS)
        assert not r["ok"] and r["extra"] == ["Debt-to-income ratio too high"]

    def test_committed_demo_notice_passes(self):
        """The real gpt-6-luna notice shown on the demo states exactly the model's reasons."""
        from loanlens.dashboard.samples import DEMO_SAMPLES_PATH
        from loanlens.model.explain import FEATURE_TO_REGULATORY
        from loanlens.rag.generator import principal_reasons
        if not DEMO_SAMPLES_PATH.exists():
            pytest.skip("no demo samples")
        r = json.loads(DEMO_SAMPLES_PATH.read_text())["High Risk — High debt, late payments"]["result"]
        labels = sorted({v["label"] for v in FEATURE_TO_REGULATORY.values()})
        assert check_reasons(r["adverse_action_notice"], principal_reasons(r["shap_factors"]), labels)["ok"]


FACTORS = [
    {"label": "History of late installment payments", "direction": "increases_risk", "feature": "late_payment_rate",
     "cfpb_code": "A1 - Delinquent past or present credit obligations", "shap_value": 0.9},
]
PASSAGES = [{"text": "Under 12 CFR 1002.9 a creditor must state reasons.", "source": "m.pdf", "page": 1}]


def _reply(payload):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(payload)), finish_reason="stop")])


class TestNoticeStatus:
    @patch("loanlens.rag.generator.OpenAI")
    def test_drafted_when_reasons_and_citations_check_out(self, mock_openai):
        mock_openai.return_value.chat.completions.create.return_value = _reply({
            "adverse_action_notice": "Declined per 12 CFR 1002.9: history of late installment payments.",
            "regulatory_basis": "12 CFR 1002.9", "applicant_rights": "r"})
        from loanlens.rag.generator import generate_explanation
        assert generate_explanation(FACTORS, PASSAGES, 0.5, "decline")["notice_status"] == "drafted"

    @patch("loanlens.rag.generator.OpenAI")
    def test_needs_review_when_a_reason_is_missing(self, mock_openai):
        mock_openai.return_value.chat.completions.create.return_value = _reply({
            "adverse_action_notice": "Your application was declined.", "regulatory_basis": "", "applicant_rights": "r"})
        from loanlens.rag.generator import generate_explanation
        result = generate_explanation(FACTORS, PASSAGES, 0.5, "decline")
        assert result["notice_status"] == "needs_review"
        assert result["reasons_check"]["missing"] == ["History of late installment payments"]


# ── 2. LLM outage during a decline ─────────────────────────────

def _declining_pipeline():
    with patch("loanlens.rag.pipeline.load_production_model"), \
         patch("loanlens.rag.pipeline.CreditExplainer"), \
         patch("loanlens.rag.pipeline.RegulatoryRetriever"), \
         patch("loanlens.rag.pipeline.load_preprocessing_artifacts", return_value=(["x"], {}, 4)):
        from loanlens.rag.pipeline import LoanLensPipeline
        p = LoanLensPipeline()
    p.model = MagicMock()
    p.model.predict_proba.return_value = np.array([[0.4, 0.6]])
    p.explainer = MagicMock()
    p.explainer.get_shap_factors.return_value = FACTORS
    p.explainer.get_adverse_action_codes.return_value = ["A1 - Delinquent past or present credit obligations"]
    p.retriever = MagicMock()
    p.retriever.retrieve_for_explanation.return_value = PASSAGES
    return p


@patch("loanlens.rag.pipeline.generate_explanation", side_effect=TimeoutError("OpenAI timed out"))
def test_decline_survives_llm_outage(_):
    result = _declining_pipeline().explain(pd.DataFrame({"x": [1.0]}))
    assert result["decision"] == "decline" and result["probability"] == 0.6
    assert result["adverse_action_codes"] == ["A1 - Delinquent past or present credit obligations"]
    assert result["primary_reasons"] == ["History of late installment payments"]
    assert result["notice_status"] == "unavailable"
    assert result["adverse_action_notice"] is None


def test_api_returns_200_with_unavailable_notice():
    from fastapi.testclient import TestClient

    import loanlens.api.main as main_module
    pipeline = MagicMock()
    pipeline.feature_names, pipeline.category_maps = ["amt_credit"], {}
    pipeline.explain.return_value = {
        "risk_score": 60.0, "probability": 0.6, "decision": "decline", "shap_factors": [],
        "adverse_action_codes": ["A1 - Delinquent past or present credit obligations"], "retrieved_passages": [],
        "adverse_action_notice": None, "primary_reasons": ["History of late installment payments"],
        "regulatory_basis": None, "applicant_rights": None, "grounding_score": None, "citation_check": None,
        "reasons_check": None, "notice_status": "unavailable", "generation_time_ms": 0,
    }
    main_module._pipeline = pipeline
    response = TestClient(main_module.app).post("/explain", json=BASE)
    assert response.status_code == 200
    assert response.json()["notice_status"] == "unavailable"
    assert response.json()["decision"] == "decline"


def test_dashboard_explains_unavailable_notice():
    from loanlens.dashboard.samples import format_result
    out = format_result({"decision": "decline", "probability": 0.6, "notice_status": "unavailable",
                         "adverse_action_codes": ["A1"], "shap_factors": []})
    assert "could not be drafted" in out[2] and "still stand" in out[2]
