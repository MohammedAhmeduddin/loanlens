"""
Regulatory chunk retrieval from ChromaDB.
Cosine similarity search + optional cross-encoder reranking.
"""

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings
from loanlens.rag.ingest import get_chroma_client, get_or_create_collection


class RegulatoryRetriever:
    """
    Retrieves relevant CFPB regulatory passages for a given query.
    Uses cosine similarity search on ChromaDB vector store.
    """

    def __init__(self):
        settings = get_settings()
        self.settings = settings

        # Load embedding model
        logger.info("Initializing retriever...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Connect to ChromaDB
        self.client = get_chroma_client()
        self.collection = get_or_create_collection(self.client)

        count = self.collection.count()
        logger.info(f"Retriever ready — {count} chunks in collection")

    def retrieve(self, query: str, k: int = None) -> list[dict]:
        """
        Retrieve top-k most relevant regulatory passages.

        Args:
            query: Search query (built from SHAP factors)
            k: Number of results to return

        Returns:
            List of passage dicts with text, source, score
        """
        k = k or self.settings.retrieval_k

        # Embed the query
        query_embedding = self.embeddings.embed_query(query)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        passages = []
        for i in range(len(results["documents"][0])):
            # ChromaDB cosine distance → similarity score
            distance = results["distances"][0][i]
            similarity = 1 - distance

            passages.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "page": results["metadatas"][0][i].get("page", 0),
                "similarity_score": round(similarity, 4),
                "rank": i + 1,
            })

        logger.info(
            f"Retrieved {len(passages)} passages "
            f"(top score: {passages[0]['similarity_score']:.4f})"
        )
        return passages

    def retrieve_for_explanation(
        self,
        query: str,
        top_n: int = None
    ) -> list[dict]:
        """
        Retrieve and select top passages for explanation generation.
        Filters out low-relevance passages.

        Args:
            query: RAG query from SHAP factors
            top_n: Final number of passages to use

        Returns:
            Top N most relevant passages
        """
        top_n = top_n or self.settings.rerank_top_n

        # Retrieve more than we need then filter
        passages = self.retrieve(query, k=self.settings.retrieval_k)

        # Filter by minimum similarity threshold
        MIN_SIMILARITY = 0.3
        filtered = [p for p in passages if p["similarity_score"] >= MIN_SIMILARITY]

        if not filtered:
            logger.warning("No passages above similarity threshold — using top results")
            filtered = passages

        # Take top N
        selected = filtered[:top_n]

        logger.info(
            f"Selected {len(selected)} passages for explanation "
            f"(scores: {[p['similarity_score'] for p in selected]})"
        )

        return selected
