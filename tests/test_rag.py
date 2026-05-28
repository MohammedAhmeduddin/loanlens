"""Tests for RAG pipeline components."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── generator tests ────────────────────────────────────────────

class TestGenerator:
    @pytest.fixture
    def mock_factors(self):
        return [
            {
                "label": "Low external credit score",
                "cfpb_code": "A9 - Credit score",
                "shap_value": -0.45,
                "direction": "increases_risk",
            },
            {
                "label": "Debt-to-income ratio too high",
                "cfpb_code": "A6 - Debt-to-income ratio",
                "shap_value": -0.23,
                "direction": "increases_risk",
            },
        ]

    @pytest.fixture
    def mock_passages(self):
        return [
            {
                "source": "supervision_manual.pdf",
                "page": 376,
                "text": "Adverse action notices required under FCRA and ECOA.",
            }
        ]

    @patch("loanlens.rag.generator.OpenAI")
    def test_generate_explanation_returns_dict(
        self, mock_openai, mock_factors, mock_passages
    ):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"adverse_action_notice": "Declined.", "primary_reasons": ["Low credit score"], "regulatory_basis": "ECOA", "applicant_rights": "You may request a copy.", "grounding_score": 0.9}'))]
        )

        from loanlens.rag.generator import generate_explanation
        result = generate_explanation(mock_factors, mock_passages, 0.75, "decline")
        assert isinstance(result, dict)
        assert "adverse_action_notice" in result
        assert "primary_reasons" in result
        assert "grounding_score" in result

    @patch("loanlens.rag.generator.OpenAI")
    def test_generate_explanation_has_timing(
        self, mock_openai, mock_factors, mock_passages
    ):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"adverse_action_notice": "Declined.", "primary_reasons": [], "regulatory_basis": "ECOA", "applicant_rights": "Rights.", "grounding_score": 1.0}'))]
        )

        from loanlens.rag.generator import generate_explanation
        result = generate_explanation(mock_factors, mock_passages, 0.75, "decline")
        assert "generation_time_ms" in result
        assert result["generation_time_ms"] >= 0

    @patch("loanlens.rag.generator.OpenAI")
    def test_generate_handles_markdown_wrapped_json(
        self, mock_openai, mock_factors, mock_passages
    ):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='```json\n{"adverse_action_notice": "Declined.", "primary_reasons": [], "regulatory_basis": "ECOA", "applicant_rights": "Rights.", "grounding_score": 0.8}\n```'))]
        )

        from loanlens.rag.generator import generate_explanation
        result = generate_explanation(mock_factors, mock_passages, 0.75, "decline")
        assert "adverse_action_notice" in result


# ── retriever tests ────────────────────────────────────────────

class TestRetriever:
    @patch("loanlens.rag.retriever.get_chroma_client")
    @patch("loanlens.rag.retriever.get_or_create_collection")
    @patch("loanlens.rag.retriever.HuggingFaceEmbeddings")
    def test_retriever_initializes(
        self, mock_embeddings, mock_collection, mock_client
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 9977
        mock_collection.return_value = mock_col

        from loanlens.rag.retriever import RegulatoryRetriever
        retriever = RegulatoryRetriever()
        assert retriever.collection.count() == 9977

    @patch("loanlens.rag.retriever.get_chroma_client")
    @patch("loanlens.rag.retriever.get_or_create_collection")
    @patch("loanlens.rag.retriever.HuggingFaceEmbeddings")
    def test_retrieve_returns_list(
        self, mock_embeddings, mock_collection_fn, mock_client
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 9977
        mock_col.query.return_value = {
            "documents": [["passage 1", "passage 2"]],
            "metadatas": [[{"source": "test.pdf", "page": 1}, {"source": "test.pdf", "page": 2}]],
            "distances": [[0.3, 0.4]],
        }
        mock_collection_fn.return_value = mock_col

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1] * 384
        mock_embeddings.return_value = mock_emb

        from loanlens.rag.retriever import RegulatoryRetriever
        retriever = RegulatoryRetriever()
        passages = retriever.retrieve("test query", k=2)
        assert isinstance(passages, list)
        assert len(passages) == 2

    @patch("loanlens.rag.retriever.get_chroma_client")
    @patch("loanlens.rag.retriever.get_or_create_collection")
    @patch("loanlens.rag.retriever.HuggingFaceEmbeddings")
    def test_passage_has_required_fields(
        self, mock_embeddings, mock_collection_fn, mock_client
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 100
        mock_col.query.return_value = {
            "documents": [["Some regulatory text"]],
            "metadatas": [[{"source": "manual.pdf", "page": 5}]],
            "distances": [[0.25]],
        }
        mock_collection_fn.return_value = mock_col

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1] * 384
        mock_embeddings.return_value = mock_emb

        from loanlens.rag.retriever import RegulatoryRetriever
        retriever = RegulatoryRetriever()
        passages = retriever.retrieve("query", k=1)
        p = passages[0]
        assert "text" in p
        assert "source" in p
        assert "page" in p
        assert "similarity_score" in p
        assert "rank" in p

    @patch("loanlens.rag.retriever.get_chroma_client")
    @patch("loanlens.rag.retriever.get_or_create_collection")
    @patch("loanlens.rag.retriever.HuggingFaceEmbeddings")
    def test_similarity_score_computed_correctly(
        self, mock_embeddings, mock_collection_fn, mock_client
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 100
        mock_col.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[{"source": "x.pdf", "page": 1}]],
            "distances": [[0.3]],  # distance 0.3 → similarity 0.7
        }
        mock_collection_fn.return_value = mock_col

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1] * 384
        mock_embeddings.return_value = mock_emb

        from loanlens.rag.retriever import RegulatoryRetriever
        retriever = RegulatoryRetriever()
        passages = retriever.retrieve("query", k=1)
        assert abs(passages[0]["similarity_score"] - 0.7) < 0.01

    @patch("loanlens.rag.retriever.get_chroma_client")
    @patch("loanlens.rag.retriever.get_or_create_collection")
    @patch("loanlens.rag.retriever.HuggingFaceEmbeddings")
    def test_retrieve_for_explanation_filters_low_similarity(
        self, mock_embeddings, mock_collection_fn, mock_client
    ):
        mock_col = MagicMock()
        mock_col.count.return_value = 100
        mock_col.query.return_value = {
            "documents": [["good text", "bad text", "ok text"]],
            "metadatas": [[
                {"source": "a.pdf", "page": 1},
                {"source": "b.pdf", "page": 2},
                {"source": "c.pdf", "page": 3},
            ]],
            "distances": [[0.1, 0.8, 0.4]],
        }
        mock_collection_fn.return_value = mock_col

        mock_emb = MagicMock()
        mock_emb.embed_query.return_value = [0.1] * 384
        mock_embeddings.return_value = mock_emb

        from loanlens.rag.retriever import RegulatoryRetriever
        retriever = RegulatoryRetriever()
        passages = retriever.retrieve_for_explanation("query", top_n=3)
        # Only passages with similarity >= 0.3 should be kept
        for p in passages:
            assert p["similarity_score"] >= 0.3


# ── ingest tests ───────────────────────────────────────────────

class TestIngest:
    def test_generate_chunk_id_is_string(self):
        from loanlens.rag.ingest import generate_chunk_id
        chunk_id = generate_chunk_id("some text", "test.pdf", 1, 0)
        assert isinstance(chunk_id, str)
        assert len(chunk_id) == 32  # MD5 hex length

    def test_generate_chunk_id_different_for_different_chunks(self):
        from loanlens.rag.ingest import generate_chunk_id
        id1 = generate_chunk_id("text one", "test.pdf", 1, 0)
        id2 = generate_chunk_id("text two", "test.pdf", 1, 1)
        assert id1 != id2

    def test_generate_chunk_id_stable(self):
        from loanlens.rag.ingest import generate_chunk_id
        id1 = generate_chunk_id("same text", "doc.pdf", 5, 10)
        id2 = generate_chunk_id("same text", "doc.pdf", 5, 10)
        assert id1 == id2

    @patch("loanlens.rag.ingest.chromadb.PersistentClient")
    def test_get_chroma_client_returns_client(self, mock_chroma):
        from loanlens.rag.ingest import get_chroma_client
        client = get_chroma_client()
        assert client is not None
