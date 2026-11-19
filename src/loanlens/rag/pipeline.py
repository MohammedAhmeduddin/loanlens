import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

"""
End-to-end LoanLens pipeline.
Connects: XGBoost scoring → SHAP → ChromaDB retrieval → OpenAI notice drafting (declines only)
"""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings
from loanlens.model.explain import (
    CreditExplainer,
    load_preprocessing_artifacts,
    load_production_model,
)
from loanlens.model.features import assert_no_prohibited
from loanlens.rag.generator import generate_explanation, principal_reasons
from loanlens.rag.retriever import RegulatoryRetriever


class LoanLensPipeline:
    """
    Full LoanLens inference pipeline.
    Loads model and retriever once, reuses for each request.
    """

    def __init__(self):
        settings = get_settings()
        logger.info("Initializing LoanLens pipeline...")

        # Load credit scoring model
        # Resolve one model version and load both the model and its
        # preprocessing from it, so they can never come from different runs.
        self.feature_names, self.category_maps, self.model_version = load_preprocessing_artifacts()
        self.model = load_production_model(self.model_version)
        assert_no_prohibited(self.feature_names)
        self.explainer = CreditExplainer(self.model)

        # Load retriever
        self.retriever = RegulatoryRetriever()

        self.decline_threshold = settings.decline_threshold
        self.review_threshold = settings.review_threshold

        logger.info("LoanLens pipeline ready")

    def score(self, features: pd.DataFrame) -> dict:
        """
        Score a borrower and return risk assessment.

        Args:
            features: Preprocessed feature DataFrame (single row)

        Returns:
            Score result dict
        """
        prob = float(self.model.predict_proba(features)[0, 1])
        risk_score = round(prob * 100, 2)

        if prob >= self.decline_threshold:
            decision = "decline"
        elif prob >= self.review_threshold:
            decision = "review"
        else:
            decision = "approve"

        return {
            "risk_score": risk_score,
            "probability": round(prob, 4),
            "decision": decision,
        }

    def explain(self, features: pd.DataFrame) -> dict:
        """
        Full explanation pipeline for a single borrower.

        Args:
            features: Preprocessed feature DataFrame (single row)

        Returns:
            Complete explanation with score, SHAP, and adverse action notice
        """
        # 1. Score
        score_result = self.score(features)
        logger.info(
            f"Risk score: {score_result['risk_score']} | "
            f"Decision: {score_result['decision']}"
        )

        # 2. SHAP explanation (shown for every decision)
        shap_factors = self.explainer.get_shap_factors(features, n_top=5)
        logger.info(f"SHAP top factor: {shap_factors[0]['label']}")

        # 3-4. Only a decline is an adverse action: only then are reason codes
        #      stated, regulations retrieved and a notice drafted. "review" goes
        #      to a human underwriter; "approve" needs no notice.
        adverse_action_codes, passages = [], []
        explanation = {
            "adverse_action_notice": None,
            "primary_reasons": [],
            "regulatory_basis": None,
            "applicant_rights": None,
            "grounding_score": None,
            "citation_check": None,
            "reasons_check": None,
            "notice_status": "not_required",
            "generation_time_ms": 0,
        }
        if score_result["decision"] == "decline":
            adverse_action_codes = self.explainer.get_adverse_action_codes(shap_factors)
            rag_query = self.explainer.build_rag_query(shap_factors)
            passages = self.retriever.retrieve_for_explanation(rag_query)
            logger.info(f"Retrieved {len(passages)} regulatory passages")
            try:
                explanation = generate_explanation(
                    shap_factors=shap_factors,
                    retrieved_passages=passages,
                    risk_score=score_result["probability"],
                    decision=score_result["decision"],
                )
            except Exception as e:
                # The decision and its reasons come from the model, not the LLM, so an
                # LLM outage must not lose them. The notice is left for a human to issue.
                logger.exception(f"Notice drafting failed: {e}")
                explanation = {
                    **explanation,
                    "primary_reasons": principal_reasons(shap_factors),
                    "notice_status": "unavailable",
                }

        return {
            **score_result,
            "shap_factors": shap_factors,
            "adverse_action_codes": adverse_action_codes,
            "retrieved_passages": passages,
            "adverse_action_notice": explanation.get("adverse_action_notice"),
            "primary_reasons": explanation.get("primary_reasons", []),
            "regulatory_basis": explanation.get("regulatory_basis"),
            "applicant_rights": explanation.get("applicant_rights"),
            "grounding_score": explanation.get("grounding_score"),
            "citation_check": explanation.get("citation_check"),
            "reasons_check": explanation.get("reasons_check"),
            "notice_status": explanation.get("notice_status", "not_required"),
            "generation_time_ms": explanation.get("generation_time_ms", 0),
        }
