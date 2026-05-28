"""Tests for pipeline, registry, and monitoring modules."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── pipeline tests ─────────────────────────────────────────────

class TestLoanLensPipeline:
    @pytest.fixture
    def mock_pipeline(self):
        with patch("loanlens.rag.pipeline.load_production_model") as mock_model, \
             patch("loanlens.rag.pipeline.CreditExplainer") as mock_explainer, \
             patch("loanlens.rag.pipeline.RegulatoryRetriever") as mock_retriever:

            mock_model.return_value = MagicMock()
            mock_explainer.return_value = MagicMock()
            mock_retriever.return_value = MagicMock()
            mock_retriever.return_value.collection.count.return_value = 9977

            from loanlens.rag.pipeline import LoanLensPipeline
            pipeline = LoanLensPipeline()
            pipeline.model = MagicMock()
            pipeline.explainer = MagicMock()
            pipeline.retriever = MagicMock()
            yield pipeline

    def test_pipeline_initializes(self, mock_pipeline):
        assert mock_pipeline is not None

    def test_score_decline(self, mock_pipeline):
        mock_pipeline.model.predict_proba.return_value = np.array([[0.2, 0.8]])
        X = pd.DataFrame({"feature": [1.0]})
        result = mock_pipeline.score(X)
        assert result["decision"] == "decline"
        assert result["risk_score"] == 80.0
        assert result["probability"] == 0.8

    def test_score_approve(self, mock_pipeline):
        mock_pipeline.model.predict_proba.return_value = np.array([[0.9, 0.1]])
        X = pd.DataFrame({"feature": [1.0]})
        result = mock_pipeline.score(X)
        assert result["decision"] == "approve"

    def test_score_review(self, mock_pipeline):
        mock_pipeline.model.predict_proba.return_value = np.array([[0.65, 0.35]])
        X = pd.DataFrame({"feature": [1.0]})
        result = mock_pipeline.score(X)
        assert result["decision"] == "review"

    def test_score_returns_required_keys(self, mock_pipeline):
        mock_pipeline.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        X = pd.DataFrame({"feature": [1.0]})
        result = mock_pipeline.score(X)
        assert "risk_score" in result
        assert "probability" in result
        assert "decision" in result

    @patch("loanlens.rag.pipeline.generate_explanation")
    def test_explain_decline_calls_generation(self, mock_gen, mock_pipeline):
        mock_pipeline.model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_pipeline.explainer.get_shap_factors.return_value = [
            {"rank": 1, "feature": "ext_source_mean", "feature_value": 0.2,
             "shap_value": -0.5, "direction": "increases_risk",
             "label": "Credit score", "cfpb_code": "A9", "regulation": "FCRA"}
        ]
        mock_pipeline.explainer.build_rag_query.return_value = "query"
        mock_pipeline.explainer.get_adverse_action_codes.return_value = ["A9"]
        mock_pipeline.retriever.retrieve_for_explanation.return_value = [
            {"text": "reg text", "source": "manual.pdf",
             "page": 1, "similarity_score": 0.7, "rank": 1}
        ]
        mock_gen.return_value = {
            "adverse_action_notice": "Declined.",
            "primary_reasons": ["Credit score"],
            "regulatory_basis": "ECOA",
            "applicant_rights": "Rights.",
            "grounding_score": 0.9,
            "generation_time_ms": 1000,
        }

        X = pd.DataFrame({"feature": [1.0]})
        result = mock_pipeline.explain(X)
        mock_gen.assert_called_once()
        assert result["adverse_action_notice"] == "Declined."

    @patch("loanlens.rag.pipeline.generate_explanation")
    def test_explain_approve_skips_generation(self, mock_gen, mock_pipeline):
        mock_pipeline.model.predict_proba.return_value = np.array([[0.9, 0.1]])
        mock_pipeline.explainer.get_shap_factors.return_value = [
            {"rank": 1, "feature": "ext_source_mean", "feature_value": 0.8,
             "shap_value": 0.5, "direction": "decreases_risk",
             "label": "Credit score", "cfpb_code": "A9", "regulation": "FCRA"}
        ]
        mock_pipeline.explainer.build_rag_query.return_value = "query"
        mock_pipeline.explainer.get_adverse_action_codes.return_value = []
        mock_pipeline.retriever.retrieve_for_explanation.return_value = []

        X = pd.DataFrame({"feature": [1.0]})
        result = mock_pipeline.explain(X)
        mock_gen.assert_not_called()
        assert result["adverse_action_notice"] is None


# ── registry tests ─────────────────────────────────────────────

class TestRegistry:
    @patch("loanlens.model.registry.MlflowClient")
    @patch("loanlens.model.registry.mlflow")
    def test_list_model_versions_returns_list(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.run_id = "abc123def456"
        mock_client.search_model_versions.return_value = [mock_version]

        from loanlens.model.registry import list_model_versions
        versions = list_model_versions("TestModel")
        assert isinstance(versions, list)
        assert len(versions) == 1

    @patch("loanlens.model.registry.MlflowClient")
    @patch("loanlens.model.registry.mlflow")
    def test_promote_to_staging_calls_transition(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        from loanlens.model.registry import promote_to_staging
        promote_to_staging(version=1, model_name="TestModel")
        mock_client.transition_model_version_stage.assert_called_once()

    @patch("loanlens.model.registry.MlflowClient")
    @patch("loanlens.model.registry.mlflow")
    def test_promote_to_production_archives_existing(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        from loanlens.model.registry import promote_to_production
        promote_to_production(version=2, model_name="TestModel")

        call_kwargs = mock_client.transition_model_version_stage.call_args
        assert call_kwargs[1].get("archive_existing_versions") is True or \
               call_kwargs[0][3] is True

    @patch("loanlens.model.registry.MlflowClient")
    @patch("loanlens.model.registry.mlflow")
    def test_get_production_model_info_returns_dict(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.run_id = "abc123"
        mock_client.get_latest_versions.return_value = [mock_version]

        mock_run = MagicMock()
        mock_run.data.metrics = {"val_auc": 0.77, "test_auc": 0.77}
        mock_run.data.params = {"n_features": "75", "train_size": "215000"}
        mock_client.get_run.return_value = mock_run

        from loanlens.model.registry import get_production_model_info
        info = get_production_model_info("TestModel")
        assert "val_auc" in info
        assert "model_name" in info

    @patch("loanlens.model.registry.MlflowClient")
    @patch("loanlens.model.registry.mlflow")
    def test_get_production_model_info_no_model(self, mock_mlflow, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_latest_versions.return_value = []

        from loanlens.model.registry import get_production_model_info
        info = get_production_model_info("TestModel")
        assert info == {}


# ── logging utility tests ──────────────────────────────────────

class TestLogging:
    @patch("loanlens.utils.logging.logger")
    def test_setup_logging_does_not_raise(self, mock_logger):
        from loanlens.utils.logging import setup_logging
        try:
            setup_logging()
        except Exception as e:
            pytest.fail(f"setup_logging raised {e}")

    def test_logger_is_importable(self):
        from loanlens.utils.logging import logger
        assert logger is not None
