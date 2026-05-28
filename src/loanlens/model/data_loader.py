"""
Data loader for credit scoring model training.
Loads feat_master from PostgreSQL feature store.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings


# Categorical columns to encode
CATEGORICAL_COLS = [
    "name_contract_type",
    "name_family_status",
    "name_education_type",
    "name_income_type",
    "name_housing_type",
    "code_gender",
    "flag_own_car",
    "flag_own_realty",
    "occupation_type",
    "organization_type",
]

# Columns to drop before training
DROP_COLS = [
    "sk_id_curr",
    "target",
    "is_unemployed",         # bool — convert separately
]


def load_features(limit: int = None) -> pd.DataFrame:
    """
    Load feature table from PostgreSQL.

    Args:
        limit: Optional row limit for development/testing

    Returns:
        DataFrame with all features and target
    """
    settings = get_settings()
    engine = create_engine(settings.database_url_sync)

    query = "SELECT * FROM staging_features.feat_master"
    if limit:
        query += f" LIMIT {limit}"

    logger.info(f"Loading features from PostgreSQL{f' (limit={limit})' if limit else ''}...")
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    return df


def preprocess_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Preprocess features for XGBoost training.

    Args:
        df: Raw feature DataFrame from PostgreSQL

    Returns:
        X: Feature matrix
        y: Target series
        feature_names: List of feature names
    """
    logger.info("Preprocessing features...")

    # Separate target
    y = df["target"].astype(int)

    # Drop non-feature columns
    X = df.drop(columns=["sk_id_curr", "target"], errors="ignore")

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    # Label encode categoricals
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes
            X[col] = X[col].replace(-1, np.nan)  # -1 = was NaN

    # Verify no object columns remain
    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        logger.warning(f"Dropping remaining object columns: {obj_cols}")
        X = X.drop(columns=obj_cols)

    feature_names = X.columns.tolist()
    logger.info(f"Final feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    logger.info(f"Default rate: {y.mean():.4f} ({y.sum():,} defaults)")

    return X, y, feature_names
