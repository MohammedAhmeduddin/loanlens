"""
Unit and integration tests for LoanLens FastAPI endpoints.
OpenAI and MLflow calls are mocked — no real API costs in tests.
"""

import pytest
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_pipeline():
    """Mock pipeline that avoids loading real ML models in tests."""
    pipeline = MagicMock()

    pipeline.score.return_value = {
        "risk_score": 72.5,
        "probability": 0.725,
        "decision": "decline",
    }

    pipeline.explain.return_value = {
        "risk_score": 72.5,
        "probability": 0.725,
        "decision": "decline",
        "shap_factors": [
            {
                "rank": 1,
                "feature": "ext_source_mean",
                "feature_value": 0.25,
                "shap_value": -0.45,
                "direction": "increases_risk",
                "label": "External credit score assessment",
                "cfpb_code": "A9 - Credit score",
                "regulation": "FCRA",
            }
        ],
        "adverse_action_codes": ["A9 - Credit score"],
        "retrieved_passages": [
            {
                "text": "Adverse action notice requirements under ECOA.",
                "source": "supervision_manual.pdf",
                "page": 376,
                "similarity_score": 0.75,
                "rank": 1,
            }
        ],
        "adverse_action_notice": "Your application has been declined due to credit score.",
        "primary_reasons": ["Low credit score"],
        "regulatory_basis": "ECOA Regulation B Section 202.9",
        "applicant_rights": "You have the right to obtain a free copy of your credit report.",
        "grounding_score": 0.95,
        "generation_time_ms": 1200,
    }

    pipeline.retriever.collection.count.return_value = 9977
    return pipeline


@pytest.fixture(scope="module")
def client(mock_pipeline):
    """TestClient with mocked pipeline."""
    import loanlens.api.main as main_module
    main_module._pipeline = mock_pipeline

    from loanlens.api.main import app
    return TestClient(app)


# Sample valid request body
VALID_REQUEST = {
    "amt_income_total": 75000,
    "amt_credit": 250000,
    "amt_annuity": 12000,
    "age_years": 35,
    "cnt_children": 2,
    "employment_years": 5.0,
    "ext_source_2": 0.45,
    "bureau_overdue_count": 0,
    "prev_refused_count": 0,
    "late_payment_rate": 0.05,
}

HIGH_RISK_REQUEST = {
    "amt_income_total": 20000,
    "amt_credit": 400000,
    "amt_annuity": 20000,
    "age_years": 22,
    "cnt_children": 5,
    "employment_years": 0.3,
    "ext_source_2": 0.10,
    "bureau_overdue_count": 5,
    "prev_refused_count": 4,
    "late_payment_rate": 0.8,
}


# ── Health Tests ───────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "chromadb_chunks" in data
        assert "version" in data

    def test_health_model_loaded(self, client):
        response = client.get("/health")
        assert response.json()["model_loaded"] is True

    def test_health_chromadb_count(self, client):
        response = client.get("/health")
        assert response.json()["chromadb_chunks"] == 9977


# ── Score Tests ────────────────────────────────────────────────

class TestScore:
    def test_score_returns_200(self, client):
        response = client.post("/score", json=VALID_REQUEST)
        assert response.status_code == 200

    def test_score_response_structure(self, client):
        response = client.post("/score", json=VALID_REQUEST)
        data = response.json()
        assert "risk_score" in data
        assert "probability" in data
        assert "decision" in data
        assert "model_version" in data

    def test_score_risk_score_range(self, client):
        response = client.post("/score", json=VALID_REQUEST)
        data = response.json()
        assert 0 <= data["risk_score"] <= 100

    def test_score_probability_range(self, client):
        response = client.post("/score", json=VALID_REQUEST)
        data = response.json()
        assert 0 <= data["probability"] <= 1

    def test_score_valid_decision(self, client):
        response = client.post("/score", json=VALID_REQUEST)
        data = response.json()
        assert data["decision"] in ["approve", "review", "decline"]

    def test_score_missing_required_field(self, client):
        bad_request = {"amt_credit": 250000}  # missing amt_income_total
        response = client.post("/score", json=bad_request)
        assert response.status_code == 422

    def test_score_negative_income_rejected(self, client):
        bad_request = {**VALID_REQUEST, "amt_income_total": -1000}
        response = client.post("/score", json=bad_request)
        assert response.status_code == 422

    def test_score_high_risk_request(self, client):
        response = client.post("/score", json=HIGH_RISK_REQUEST)
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "decline"


# ── Explain Tests ──────────────────────────────────────────────

class TestExplain:
    def test_explain_returns_200(self, client):
        response = client.post("/explain", json=VALID_REQUEST)
        assert response.status_code == 200

    def test_explain_response_structure(self, client):
        response = client.post("/explain", json=VALID_REQUEST)
        data = response.json()
        required_fields = [
            "risk_score", "probability", "decision",
            "shap_factors", "adverse_action_codes",
            "retrieved_passages", "adverse_action_notice",
            "primary_reasons", "regulatory_basis",
            "grounding_score", "generation_time_ms"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_explain_shap_factors_not_empty(self, client):
        response = client.post("/explain", json=HIGH_RISK_REQUEST)
        data = response.json()
        assert len(data["shap_factors"]) > 0

    def test_explain_shap_factor_structure(self, client):
        response = client.post("/explain", json=HIGH_RISK_REQUEST)
        factor = response.json()["shap_factors"][0]
        assert "rank" in factor
        assert "feature" in factor
        assert "shap_value" in factor
        assert "direction" in factor
        assert factor["direction"] in ["increases_risk", "decreases_risk"]

    def test_explain_retrieved_passages_not_empty(self, client):
        response = client.post("/explain", json=HIGH_RISK_REQUEST)
        data = response.json()
        assert len(data["retrieved_passages"]) > 0

    def test_explain_adverse_action_notice_present(self, client):
        response = client.post("/explain", json=HIGH_RISK_REQUEST)
        data = response.json()
        assert data["adverse_action_notice"] is not None
        assert len(data["adverse_action_notice"]) > 20

    def test_explain_grounding_score_range(self, client):
        response = client.post("/explain", json=HIGH_RISK_REQUEST)
        data = response.json()
        if data["grounding_score"] is not None:
            assert 0 <= data["grounding_score"] <= 1

    def test_explain_missing_required_field(self, client):
        response = client.post("/explain", json={"amt_credit": 250000})
        assert response.status_code == 422


# ── Schema Validation Tests ────────────────────────────────────

class TestSchemas:
    def test_age_below_18_rejected(self, client):
        bad = {**VALID_REQUEST, "age_years": 16}
        response = client.post("/score", json=bad)
        assert response.status_code == 422

    def test_late_payment_rate_above_1_rejected(self, client):
        bad = {**VALID_REQUEST, "late_payment_rate": 1.5}
        response = client.post("/score", json=bad)
        assert response.status_code == 422

    def test_ext_source_above_1_rejected(self, client):
        bad = {**VALID_REQUEST, "ext_source_2": 1.5}
        response = client.post("/score", json=bad)
        assert response.status_code == 422
