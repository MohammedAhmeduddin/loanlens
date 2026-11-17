"""
LoanLens FastAPI application.
Serves credit scoring and RAG explanation endpoints.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings
from loanlens.model.features import encode_categoricals
from loanlens.model.serving import derive_features
from loanlens.api.schemas import (
    ScoreRequest, ScoreResponse,
    ExplainRequest, ExplainResponse,
    HealthResponse, ShapFactor, RetrievedPassage
)


# Global pipeline instance — loaded once at startup
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, cleanup on shutdown."""
    global _pipeline
    logger.info("Starting LoanLens API...")

    try:
        from loanlens.rag.pipeline import LoanLensPipeline
        _pipeline = LoanLensPipeline()
        logger.info("Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        _pipeline = None

    yield

    logger.info("Shutting down LoanLens API")


settings = get_settings()

app = FastAPI(
    title="LoanLens API",
    description="RAG-powered credit risk explainer over CFPB regulations",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def request_to_dataframe(
    request: ScoreRequest,
    feature_names: list[str],
    category_maps: dict[str, list[str]],
) -> pd.DataFrame:
    """Convert an API request into the exact feature row the model was trained on.

    - Column order and set come from the trained model (feature_names).
    - Categoricals use the vocabulary saved at training time.
    - Computed ratios, "no record" zeros and history flags follow the dbt
      feature SQL (see model/serving.py); anything dbt leaves NULL stays NaN.
    """
    data = derive_features(request.model_dump())

    row = {feat: data.get(feat) for feat in feature_names}
    X = pd.DataFrame([row], columns=feature_names)
    X = encode_categoricals(X, category_maps)
    return X.apply(pd.to_numeric, errors="coerce").astype(float)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    model_loaded = _pipeline is not None

    chromadb_chunks = 0
    if model_loaded:
        try:
            chromadb_chunks = _pipeline.retriever.collection.count()
        except Exception:
            pass

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        chromadb_chunks=chromadb_chunks,
        version=settings.app_version,
    )


@app.post("/score", response_model=ScoreResponse)
async def score(request: ScoreRequest):
    """
    Score a loan application.
    Returns risk score, probability, and decision.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = request_to_dataframe(request, _pipeline.feature_names, _pipeline.category_maps)
        result = _pipeline.score(X)

        return ScoreResponse(
            risk_score=result["risk_score"],
            probability=result["probability"],
            decision=result["decision"],
        )
    except Exception as e:
        logger.exception(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail="Scoring failed")


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Full explanation endpoint.
    Returns score + SHAP factors + RAG-generated adverse action notice.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = request_to_dataframe(request, _pipeline.feature_names, _pipeline.category_maps)
        result = _pipeline.explain(X)

        return ExplainResponse(
            risk_score=result["risk_score"],
            probability=result["probability"],
            decision=result["decision"],
            shap_factors=[ShapFactor(**f) for f in result["shap_factors"]],
            adverse_action_codes=result["adverse_action_codes"],
            retrieved_passages=[
                RetrievedPassage(**p) for p in result["retrieved_passages"]
            ],
            adverse_action_notice=result.get("adverse_action_notice"),
            primary_reasons=result.get("primary_reasons", []),
            regulatory_basis=result.get("regulatory_basis"),
            applicant_rights=result.get("applicant_rights"),
            grounding_score=result.get("grounding_score"),
            citation_check=result.get("citation_check"),
            generation_time_ms=result.get("generation_time_ms", 0),
        )
    except Exception as e:
        logger.exception(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail="Explanation failed")


@app.get("/model/info")
async def model_info():
    """Metadata for the exact model version this API is serving."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from loanlens.model.registry import get_model_version_info
        return get_model_version_info(_pipeline.model_version)
    except Exception as e:
        logger.exception(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail="Model info unavailable")
