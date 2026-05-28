import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

"""
End-to-end LoanLens pipeline.
Connects: XGBoost scoring → SHAP → ChromaDB retrieval → GPT-4o-mini generation
"""

import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings
from loanlens.model.explain import CreditExplainer, load_production_model
from loanlens.model.data_loader import preprocess_features
from loanlens.rag.retriever import RegulatoryRetriever
from loanlens.rag.generator import generate_explanation


class LoanLensPipeline:
    """
    Full LoanLens inference pipeline.
    Loads model and retriever once, reuses for each request.
    """

    def __init__(self):
        settings = get_settings()
        logger.info("Initializing LoanLens pipeline...")

        # Load credit scoring model
        self.model = load_production_model()
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

        # 2. SHAP explanation
        shap_factors = self.explainer.get_shap_factors(features, n_top=5)
        rag_query = self.explainer.build_rag_query(shap_factors)
        adverse_action_codes = self.explainer.get_adverse_action_codes(shap_factors)

        logger.info(f"SHAP top factor: {shap_factors[0]['label']}")

        # 3. Retrieve regulatory passages
        passages = self.retriever.retrieve_for_explanation(rag_query)
        logger.info(f"Retrieved {len(passages)} regulatory passages")

        # 4. Generate explanation (only for declines and reviews)
        if score_result["decision"] in ["decline", "review"]:
            explanation = generate_explanation(
                shap_factors=shap_factors,
                retrieved_passages=passages,
                risk_score=score_result["probability"],
                decision=score_result["decision"],
            )
        else:
            explanation = {
                "adverse_action_notice": None,
                "primary_reasons": [],
                "regulatory_basis": None,
                "applicant_rights": None,
                "grounding_score": None,
                "generation_time_ms": 0,
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
            "generation_time_ms": explanation.get("generation_time_ms", 0),
        }
