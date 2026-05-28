"""
SHAP-based explanation for credit scoring model.
Extracts top risk factors per borrower.
These factors become the RAG retrieval query.
"""

import shap
import numpy as np
import pandas as pd
import mlflow.xgboost
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings


# Maps internal feature names to business language
# and CFPB adverse action reason codes
FEATURE_TO_REGULATORY = {
    "ext_source_mean": {
        "label": "External credit score assessment",
        "cfpb_code": "A9 - Credit score",
        "regulation": "FCRA"
    },
    "ext_source_1": {
        "label": "First external credit assessment",
        "cfpb_code": "A9 - Credit score",
        "regulation": "FCRA"
    },
    "ext_source_2": {
        "label": "Second external credit assessment",
        "cfpb_code": "A9 - Credit score",
        "regulation": "FCRA"
    },
    "ext_source_3": {
        "label": "Third external credit assessment",
        "cfpb_code": "A9 - Credit score",
        "regulation": "FCRA"
    },
    "debt_to_income": {
        "label": "Debt-to-income ratio too high",
        "cfpb_code": "A6 - Debt-to-income ratio",
        "regulation": "ECOA"
    },
    "credit_to_annuity": {
        "label": "Credit amount relative to payment capacity",
        "cfpb_code": "A6 - Debt-to-income ratio",
        "regulation": "ECOA"
    },
    "bureau_overdue_count": {
        "label": "Delinquent accounts in credit history",
        "cfpb_code": "A1 - Delinquent past or present credit obligations",
        "regulation": "FCRA"
    },
    "bureau_max_overdue_days": {
        "label": "Severity of past delinquency",
        "cfpb_code": "A1 - Delinquent past or present credit obligations",
        "regulation": "FCRA"
    },
    "bureau_debt_to_credit": {
        "label": "High credit utilization across accounts",
        "cfpb_code": "A3 - Too many accounts or too much credit",
        "regulation": "FCRA"
    },
    "employment_years": {
        "label": "Insufficient length of employment",
        "cfpb_code": "A13 - Length of employment",
        "regulation": "ECOA"
    },
    "prev_refused_count": {
        "label": "Previous credit applications declined",
        "cfpb_code": "A7 - Number of recent inquiries on credit bureau",
        "regulation": "ECOA"
    },
    "late_payment_rate": {
        "label": "History of late installment payments",
        "cfpb_code": "A1 - Delinquent past or present credit obligations",
        "regulation": "FCRA"
    },
    "bureau_total_debt": {
        "label": "Excessive outstanding debt obligations",
        "cfpb_code": "A6 - Debt-to-income ratio",
        "regulation": "ECOA"
    },
    "amt_req_credit_bureau_year": {
        "label": "Too many recent credit inquiries",
        "cfpb_code": "A7 - Number of recent inquiries on credit bureau",
        "regulation": "ECOA"
    },
    "income_per_person": {
        "label": "Insufficient income relative to household size",
        "cfpb_code": "A5 - Insufficient income",
        "regulation": "ECOA"
    },
    "age_years": {
        "label": "Insufficient length of credit history",
        "cfpb_code": "A8 - Length of credit history",
        "regulation": "FCRA"
    },
    "name_education_type": {
        "label": "Creditworthiness assessment based on financial profile",
        "cfpb_code": "A9 - Credit score",
        "regulation": "ECOA"
    },
    "organization_type": {
        "label": "Employment stability and income reliability",
        "cfpb_code": "A13 - Length of employment",
        "regulation": "ECOA"
    },
    "prev_max_credit": {
        "label": "Previous credit utilization patterns",
        "cfpb_code": "A3 - Too many accounts or too much credit",
        "regulation": "FCRA"
    },
    "bureau_active_loans": {
        "label": "Too many open credit accounts",
        "cfpb_code": "A3 - Too many accounts or too much credit",
        "regulation": "FCRA"
    },
    "total_delinquency_count": {
        "label": "Pattern of delinquency across accounts",
        "cfpb_code": "A1 - Delinquent past or present credit obligations",
        "regulation": "FCRA"
    },
}


class CreditExplainer:
    """
    SHAP-based explainer for XGBoost credit scoring model.
    Extracts per-borrower risk factors for RAG query construction.
    """

    def __init__(self, model):
        """
        Initialize with trained XGBoost model.

        Args:
            model: Trained XGBClassifier
        """
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        logger.info("CreditExplainer initialized with TreeExplainer")

    def get_shap_factors(
        self,
        X: pd.DataFrame,
        n_top: int = 5
    ) -> list[dict]:
        """
        Extract top N risk factors for a single borrower.

        Args:
            X: Single-row DataFrame with borrower features
            n_top: Number of top factors to return

        Returns:
            List of factor dicts with feature, value, shap, direction
        """
        shap_values = self.explainer.shap_values(X)

        # Handle both old and new SHAP output formats
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0] if shap_values.ndim > 1 else shap_values

        # Get top N by absolute SHAP value
        top_indices = np.argsort(np.abs(sv))[-n_top:][::-1]

        factors = []
        feature_names = X.columns.tolist()

        for rank, idx in enumerate(top_indices):
            feature = feature_names[idx]
            shap_val = float(sv[idx])
            feature_val = float(X.iloc[0, idx]) if not pd.isna(X.iloc[0, idx]) else None

            # Get regulatory mapping
            reg_info = FEATURE_TO_REGULATORY.get(feature, {
                "label": feature.replace("_", " ").title(),
                "cfpb_code": "General creditworthiness",
                "regulation": "ECOA"
            })

            factors.append({
                "rank": rank + 1,
                "feature": feature,
                "feature_value": feature_val,
                "shap_value": shap_val,
                "direction": "increases_risk" if shap_val < 0 else "decreases_risk",
                "label": reg_info["label"],
                "cfpb_code": reg_info["cfpb_code"],
                "regulation": reg_info["regulation"],
            })

        return factors

    def build_rag_query(self, factors: list[dict]) -> str:
        """
        Build regulatory search query from SHAP risk factors.
        This query is used to retrieve relevant CFPB passages.

        Args:
            factors: Output of get_shap_factors()

        Returns:
            Query string for ChromaDB retrieval
        """
        risk_factors = [
            f["label"]
            for f in factors
            if f["direction"] == "increases_risk"
        ]

        if not risk_factors:
            risk_factors = [factors[0]["label"]]

        query = (
            f"Adverse action notice requirements under ECOA Regulation B "
            f"and FCRA for credit application declined due to: "
            f"{', '.join(risk_factors)}. "
            f"Required disclosure language and prohibited bases for "
            f"adverse action notifications."
        )

        return query

    def get_adverse_action_codes(self, factors: list[dict]) -> list[str]:
        """
        Return official CFPB adverse action codes for decline letter.

        Args:
            factors: Output of get_shap_factors()

        Returns:
            Deduplicated list of CFPB reason codes
        """
        codes = []
        for f in factors:
            if f["direction"] == "increases_risk":
                codes.append(f["cfpb_code"])

        # Deduplicate preserving order
        seen = set()
        unique_codes = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)

        return unique_codes[:5]


def load_production_model():
    """
    Load the Production model from MLflow registry.

    Returns:
        Loaded XGBoost model
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    try:
        model = mlflow.xgboost.load_model(
            f"models:/{settings.mlflow_model_name}/latest"
        )
        logger.info(f"Loaded model: {settings.mlflow_model_name}/latest")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from registry: {e}")
        raise
