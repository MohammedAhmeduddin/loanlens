"""
Data loader for credit scoring model training.
Loads feat_master from PostgreSQL feature store.
"""

import pandas as pd
from sqlalchemy import create_engine
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings


from loanlens.model.features import (  # noqa: E402
    CATEGORICAL_COLS,
    NON_FEATURE_COLS,
    build_category_maps,
    drop_prohibited,
    encode_categoricals,
)

__all__ = ["CATEGORICAL_COLS", "load_features", "preprocess_features"]


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


def preprocess_features(
    df: pd.DataFrame,
    category_maps: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Preprocess features for XGBoost training.

    Prohibited-basis columns (sex, marital status, age) are removed here,
    so they can never reach the model.

    Args:
        df: Raw feature DataFrame from PostgreSQL
        category_maps: Saved categorical vocabularies. Built from df if None.

    Returns:
        X: Feature matrix
        y: Target series
        feature_names: List of feature names
    """
    logger.info("Preprocessing features...")

    y = df["target"].astype(int)

    X = df.drop(columns=list(NON_FEATURE_COLS), errors="ignore")
    X = drop_prohibited(X)

    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X[col] = X[col].astype(int)

    if category_maps is None:
        category_maps = build_category_maps(X)
    X = encode_categoricals(X, category_maps)

    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        logger.warning(f"Dropping remaining object columns: {obj_cols}")
        X = X.drop(columns=obj_cols)

    feature_names = X.columns.tolist()
    logger.info(f"Final feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    logger.info(f"Default rate: {y.mean():.4f} ({y.sum():,} defaults)")

    return X, y, feature_names
