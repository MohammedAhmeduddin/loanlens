"""Tests for training pipeline and ingestion."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── train.py tests ─────────────────────────────────────────────

class TestTrain:
    @patch("loanlens.model.train.mlflow")
    @patch("loanlens.model.train.load_features")
    @patch("loanlens.model.train.preprocess_features")
    @patch("loanlens.model.train.compute_metrics")
    @patch("loanlens.model.train.xgb.XGBClassifier")
    def test_train_model_returns_run_id(
        self, mock_xgb, mock_metrics, mock_preprocess,
        mock_load, mock_mlflow
    ):
        # Setup data mocks
        mock_load.return_value = pd.DataFrame({"a": range(100)})
        X = pd.DataFrame(np.random.rand(100, 5),
                         columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        mock_preprocess.return_value = (X, y, list(X.columns))
        mock_metrics.side_effect = [
            {"train_auc": 0.87, "train_ks": 0.5, "train_gini": 0.74},
            {"val_auc": 0.77, "val_ks": 0.41, "val_gini": 0.54},
            {"test_auc": 0.77, "test_ks": 0.41, "test_gini": 0.54},
        ]

        # Setup MLflow mock
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Setup XGBoost mock
        mock_model = MagicMock()
        mock_model.best_iteration = 100
        mock_model.feature_importances_ = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mock_xgb.return_value = mock_model

        from loanlens.model.train import train_model
        run_id = train_model(n_estimators=10, limit=100)
        assert run_id == "test_run_id_123"

    @patch("loanlens.model.train.mlflow")
    @patch("loanlens.model.train.load_features")
    @patch("loanlens.model.train.preprocess_features")
    @patch("loanlens.model.train.compute_metrics")
    @patch("loanlens.model.train.xgb.XGBClassifier")
    def test_train_model_logs_params(
        self, mock_xgb, mock_metrics, mock_preprocess,
        mock_load, mock_mlflow
    ):
        mock_load.return_value = pd.DataFrame({"a": range(100)})
        X = pd.DataFrame(np.random.rand(100, 5),
                         columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        mock_preprocess.return_value = (X, y, list(X.columns))
        mock_metrics.side_effect = [
            {"train_auc": 0.87, "train_ks": 0.5, "train_gini": 0.74},
            {"val_auc": 0.77, "val_ks": 0.41, "val_gini": 0.54},
            {"test_auc": 0.77, "test_ks": 0.41, "test_gini": 0.54},
        ]

        mock_run = MagicMock()
        mock_run.info.run_id = "run_abc"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_model = MagicMock()
        mock_model.best_iteration = 50
        mock_model.feature_importances_ = np.ones(5) / 5
        mock_xgb.return_value = mock_model

        from loanlens.model.train import train_model
        train_model(n_estimators=10, limit=100)
        mock_mlflow.log_params.assert_called_once()

    @patch("loanlens.model.train.mlflow")
    @patch("loanlens.model.train.load_features")
    @patch("loanlens.model.train.preprocess_features")
    @patch("loanlens.model.train.compute_metrics")
    @patch("loanlens.model.train.xgb.XGBClassifier")
    def test_train_model_logs_metrics(
        self, mock_xgb, mock_metrics, mock_preprocess,
        mock_load, mock_mlflow
    ):
        mock_load.return_value = pd.DataFrame({"a": range(100)})
        X = pd.DataFrame(np.random.rand(100, 5),
                         columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        mock_preprocess.return_value = (X, y, list(X.columns))
        mock_metrics.side_effect = [
            {"train_auc": 0.87, "train_ks": 0.5, "train_gini": 0.74},
            {"val_auc": 0.77, "val_ks": 0.41, "val_gini": 0.54},
            {"test_auc": 0.77, "test_ks": 0.41, "test_gini": 0.54},
        ]

        mock_run = MagicMock()
        mock_run.info.run_id = "run_xyz"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_model = MagicMock()
        mock_model.best_iteration = 50
        mock_model.feature_importances_ = np.ones(5) / 5
        mock_xgb.return_value = mock_model

        from loanlens.model.train import train_model
        train_model(n_estimators=10, limit=100)
        mock_mlflow.log_metrics.assert_called_once()


# ── ingest.py tests ────────────────────────────────────────────

class TestIngestPipeline:
    @patch("loanlens.rag.ingest.chromadb.PersistentClient")
    def test_get_chroma_client_uses_settings(self, mock_chroma):
        from loanlens.rag.ingest import get_chroma_client
        get_chroma_client()
        mock_chroma.assert_called_once()

    @patch("loanlens.rag.ingest.chromadb.PersistentClient")
    def test_get_or_create_collection_returns_collection(self, mock_chroma):
        mock_client = MagicMock()
        mock_col = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col

        from loanlens.rag.ingest import get_or_create_collection
        result = get_or_create_collection(mock_client)
        assert result == mock_col

    @patch("loanlens.rag.ingest.HuggingFaceEmbeddings")
    def test_get_embedding_model_returns_embeddings(self, mock_emb):
        mock_instance = MagicMock()
        mock_emb.return_value = mock_instance

        from loanlens.rag.ingest import get_embedding_model
        result = get_embedding_model()
        assert result == mock_instance

    def test_load_and_chunk_raises_if_no_pdfs(self, tmp_path):
        from loanlens.rag.ingest import load_and_chunk_pdfs
        with pytest.raises(FileNotFoundError):
            load_and_chunk_pdfs(str(tmp_path))

    @patch("loanlens.rag.ingest.PyPDFLoader")
    def test_load_and_chunk_skips_short_chunks(self, mock_loader, tmp_path):
        # Create a dummy PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        mock_doc = MagicMock()
        mock_doc.page_content = "short"  # too short, should be filtered
        mock_doc.metadata = {"page": 0}

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance

        from loanlens.rag.ingest import load_and_chunk_pdfs
        chunks = load_and_chunk_pdfs(str(tmp_path))
        # "short" is < 50 chars so should be filtered out
        assert all(len(c["text"]) >= 50 for c in chunks)

    @patch("loanlens.rag.ingest.get_chroma_client")
    @patch("loanlens.rag.ingest.get_or_create_collection")
    def test_ingest_skips_if_already_indexed(
        self, mock_collection_fn, mock_client_fn
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 5000
        mock_collection_fn.return_value = mock_col

        from loanlens.rag.ingest import ingest_to_chromadb
        result = ingest_to_chromadb("data/regulations", force_reingest=False)
        assert result == 5000

    @patch("loanlens.rag.ingest.get_chroma_client")
    @patch("loanlens.rag.ingest.get_or_create_collection")
    @patch("loanlens.rag.ingest.load_and_chunk_pdfs")
    @patch("loanlens.rag.ingest.get_embedding_model")
    def test_ingest_force_reingest_deletes_collection(
        self, mock_emb, mock_chunks, mock_collection_fn, mock_client_fn
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 100
        mock_collection_fn.return_value = mock_col
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        mock_chunks.return_value = []
        mock_emb_instance = MagicMock()
        mock_emb_instance.embed_documents.return_value = []
        mock_emb.return_value = mock_emb_instance

        from loanlens.rag.ingest import ingest_to_chromadb
        ingest_to_chromadb("data/regulations", force_reingest=True)
        mock_client.delete_collection.assert_called_once()


# ── api/main.py additional tests ──────────────────────────────

class TestAPIMain:
    @pytest.fixture
    def mock_pipeline(self):
        pipeline = MagicMock()
        pipeline.score.return_value = {
            "risk_score": 45.0,
            "probability": 0.45,
            "decision": "review",
        }
        pipeline.explain.return_value = {
            "risk_score": 45.0,
            "probability": 0.45,
            "decision": "review",
            "shap_factors": [{
                "rank": 1, "feature": "ext_source_mean",
                "feature_value": 0.3, "shap_value": -0.3,
                "direction": "increases_risk",
                "label": "Credit score", "cfpb_code": "A9",
                "regulation": "FCRA"
            }],
            "adverse_action_codes": ["A9"],
            "retrieved_passages": [{
                "text": "text", "source": "manual.pdf",
                "page": 1, "similarity_score": 0.7, "rank": 1
            }],
            "adverse_action_notice": "Declined.",
            "primary_reasons": ["Credit score"],
            "regulatory_basis": "ECOA",
            "applicant_rights": "Rights.",
            "grounding_score": 0.9,
            "generation_time_ms": 1000,
        }
        pipeline.retriever.collection.count.return_value = 9977
        return pipeline

    @pytest.fixture
    def client(self, mock_pipeline):
        import loanlens.api.main as main_module
        main_module._pipeline = mock_pipeline
        from loanlens.api.main import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_model_info_endpoint(self, client):
        with patch("loanlens.model.registry.get_production_model_info") as mock_info:
            mock_info.return_value = {"val_auc": 0.77, "model_name": "CreditScoringModel"}
            response = client.get("/model/info")
            assert response.status_code == 200

    def test_score_pipeline_called(self, client, mock_pipeline):
        request = {
            "amt_income_total": 75000,
            "amt_credit": 250000,
            "amt_annuity": 12000,
            "age_years": 35,
        }
        client.post("/score", json=request)
        mock_pipeline.score.assert_called_once()

    def test_explain_pipeline_called(self, client, mock_pipeline):
        request = {
            "amt_income_total": 75000,
            "amt_credit": 250000,
            "amt_annuity": 12000,
            "age_years": 35,
        }
        client.post("/explain", json=request)
        mock_pipeline.explain.assert_called_once()

    def test_score_503_when_no_pipeline(self, client):
        import loanlens.api.main as main_module
        original = main_module._pipeline
        main_module._pipeline = None
        response = client.post("/score", json={
            "amt_income_total": 75000,
            "amt_credit": 250000,
            "age_years": 35,
        })
        assert response.status_code == 503
        main_module._pipeline = original

    def test_explain_503_when_no_pipeline(self, client):
        import loanlens.api.main as main_module
        original = main_module._pipeline
        main_module._pipeline = None
        response = client.post("/explain", json={
            "amt_income_total": 75000,
            "amt_credit": 250000,
            "age_years": 35,
        })
        assert response.status_code == 503
        main_module._pipeline = original


class TestIngestBatchInsert:
    @patch("loanlens.rag.ingest.get_chroma_client")
    @patch("loanlens.rag.ingest.get_or_create_collection")
    @patch("loanlens.rag.ingest.load_and_chunk_pdfs")
    @patch("loanlens.rag.ingest.get_embedding_model")
    def test_ingest_inserts_chunks_into_chromadb(
        self, mock_emb, mock_chunks, mock_col_fn, mock_client_fn
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_col_fn.return_value = mock_col
        mock_client_fn.return_value = MagicMock()

        mock_chunks.return_value = [
            {"text": "a" * 100, "source": "test.pdf", "page": 1},
            {"text": "b" * 100, "source": "test.pdf", "page": 2},
        ]
        mock_emb_inst = MagicMock()
        mock_emb_inst.embed_documents.return_value = [
            [0.1] * 384,
            [0.2] * 384,
        ]
        mock_emb.return_value = mock_emb_inst

        from loanlens.rag.ingest import ingest_to_chromadb
        result = ingest_to_chromadb("data/regulations", force_reingest=False)
        mock_col.add.assert_called()
        assert result == 2

    @patch("loanlens.rag.ingest.get_chroma_client")
    @patch("loanlens.rag.ingest.get_or_create_collection")
    @patch("loanlens.rag.ingest.load_and_chunk_pdfs")
    @patch("loanlens.rag.ingest.get_embedding_model")
    def test_ingest_returns_zero_for_empty_chunks(
        self, mock_emb, mock_chunks, mock_col_fn, mock_client_fn
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_col_fn.return_value = mock_col
        mock_client_fn.return_value = MagicMock()
        mock_chunks.return_value = []
        mock_emb_inst = MagicMock()
        mock_emb_inst.embed_documents.return_value = []
        mock_emb.return_value = mock_emb_inst

        from loanlens.rag.ingest import ingest_to_chromadb
        result = ingest_to_chromadb("data/regulations", force_reingest=False)
        assert result == 0


class TestIngestEdgeCases:
    @patch("loanlens.rag.ingest.PyPDFLoader")
    def test_load_and_chunk_handles_loader_exception(self, mock_loader, tmp_path):
        pdf_file = tmp_path / "bad.pdf"
        pdf_file.write_bytes(b"%PDF fake")
        mock_loader.side_effect = Exception("PDF corrupt")

        from loanlens.rag.ingest import load_and_chunk_pdfs
        chunks = load_and_chunk_pdfs(str(tmp_path))
        assert isinstance(chunks, list)

    @patch("loanlens.rag.ingest.PyPDFLoader")
    def test_load_and_chunk_filters_short_text(self, mock_loader, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF fake")

        mock_doc = MagicMock()
        mock_doc.page_content = "x" * 200
        mock_doc.metadata = {"page": 1}

        mock_loader_inst = MagicMock()
        mock_loader_inst.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_inst

        from loanlens.rag.ingest import load_and_chunk_pdfs
        chunks = load_and_chunk_pdfs(str(tmp_path))
        for c in chunks:
            assert "text" in c
            assert "source" in c
            assert "page" in c
