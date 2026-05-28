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
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings
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


def request_to_dataframe(request: ScoreRequest) -> pd.DataFrame:
    """Convert API request to feature DataFrame."""
    data = request.model_dump()

    # Convert bool to int
    if isinstance(data.get("is_unemployed"), bool):
        data["is_unemployed"] = int(data["is_unemployed"])

    # Encode categoricals the same way as training
    categorical_map = {
        "name_contract_type": None,
        "name_family_status": None,
        "name_education_type": None,
        "name_income_type": None,
        "name_housing_type": None,
        "code_gender": None,
        "flag_own_car": None,
        "flag_own_realty": None,
        "occupation_type": None,
        "organization_type": None,
    }

    for col in categorical_map:
        if data.get(col) is not None:
            data[col] = hash(data[col]) % 100
        else:
            data[col] = -1

    # Fill computed fields
    if data.get("debt_to_income") is None and data["amt_income_total"] > 0:
        data["debt_to_income"] = round(data["amt_credit"] / data["amt_income_total"], 4)

    if data.get("ext_source_mean") is None:
        sources = [data.get(f"ext_source_{i}") for i in [1, 2, 3]]
        valid = [s for s in sources if s is not None]
        data["ext_source_mean"] = round(sum(valid) / len(valid), 6) if valid else None

    # All 75 features the model was trained on
    ALL_FEATURES = [
        'name_contract_type','amt_credit','amt_annuity','amt_income_total',
        'amt_goods_price','age_years','cnt_children','cnt_fam_members',
        'name_family_status','name_education_type','name_income_type',
        'name_housing_type','code_gender','employment_years','is_unemployed',
        'occupation_type','organization_type','ext_source_1','ext_source_2',
        'ext_source_3','ext_source_mean','debt_to_income','credit_to_annuity',
        'income_per_person','employed_to_age_ratio','flag_own_car','flag_own_realty',
        'flag_work_phone','flag_email','region_rating_client',
        'region_rating_client_w_city','reg_city_not_live_city','reg_city_not_work_city',
        'amt_req_credit_bureau_hour','amt_req_credit_bureau_day',
        'amt_req_credit_bureau_week','amt_req_credit_bureau_mon',
        'amt_req_credit_bureau_qrt','amt_req_credit_bureau_year',
        'bureau_loan_count','bureau_active_loans','bureau_closed_loans',
        'bureau_overdue_count','bureau_max_overdue_days','bureau_avg_overdue_days',
        'bureau_total_credit','bureau_total_debt','bureau_total_overdue_amt',
        'bureau_debt_to_credit','bureau_oldest_credit_days','bureau_newest_credit_days',
        'bureau_credit_type_count','bureau_total_prolongations','has_bureau_history',
        'prev_app_count','prev_approved_count','prev_refused_count','prev_canceled_count',
        'prev_approval_rate','prev_avg_application_amt','prev_max_credit','prev_avg_credit',
        'prev_last_decision_days','prev_first_decision_days','has_prev_application',
        'instalment_count','avg_payment_ratio','late_payment_count','late_payment_rate',
        'avg_payment_deficit','avg_days_late','has_instalment_history',
        'total_debt_burden','total_credit_applications','total_delinquency_count'
    ]

    # Fill missing features with 0, keep existing values
    for feat in ALL_FEATURES:
        if feat not in data or data[feat] is None:
            data[feat] = 0

    # Return in exact training order
    row = {feat: data.get(feat, 0) for feat in ALL_FEATURES}
    return pd.DataFrame([row])


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
        X = request_to_dataframe(request)
        result = _pipeline.score(X)

        return ScoreResponse(
            risk_score=result["risk_score"],
            probability=result["probability"],
            decision=result["decision"],
        )
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Full explanation endpoint.
    Returns score + SHAP factors + RAG-generated adverse action notice.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = request_to_dataframe(request)
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
            generation_time_ms=result.get("generation_time_ms", 0),
        )
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Return current production model metadata."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from loanlens.model.registry import get_production_model_info
        info = get_production_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
