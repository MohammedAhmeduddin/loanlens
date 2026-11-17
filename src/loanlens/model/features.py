"""
Feature policy shared by training and serving.

Two rules live here so training and the API can never disagree:

1. Prohibited-basis features are never model inputs. ECOA / Regulation B
   forbid credit decisions on sex, marital status or age (among others).
   These columns stay in the feature store so we can run fairness audits
   against them, but they are dropped before training and before scoring.

2. Categorical encoding is a fixed, saved vocabulary per column. Training
   builds it, logs it to MLflow, and the API encodes requests with the same
   vocabulary. Unknown or missing values become NaN, which XGBoost treats as
   "missing" exactly as it did during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that encode (or are derived from) an ECOA prohibited basis.
# employed_to_age_ratio is employment_years / age_years, so it carries age.
PROHIBITED_BASIS_FEATURES: tuple[str, ...] = (
    "code_gender",          # sex
    "name_family_status",   # marital status
    "age_years",            # age
    "employed_to_age_ratio",  # derived from age
)

# Columns kept only for fairness auditing, never for modelling.
FAIRNESS_AUDIT_COLS: tuple[str, ...] = ("code_gender", "age_years")

CATEGORICAL_COLS: tuple[str, ...] = (
    "name_contract_type",
    "name_education_type",
    "name_income_type",
    "name_housing_type",
    "flag_own_car",
    "flag_own_realty",
    "occupation_type",
    "organization_type",
)

NON_FEATURE_COLS: tuple[str, ...] = ("sk_id_curr", "target")


def drop_prohibited(X: pd.DataFrame) -> pd.DataFrame:
    """Remove every prohibited-basis column that is present."""
    return X.drop(columns=[c for c in PROHIBITED_BASIS_FEATURES if c in X.columns])


def assert_no_prohibited(feature_names: list[str]) -> None:
    """Fail loudly if a model was trained on a prohibited basis."""
    leaked = sorted(set(feature_names) & set(PROHIBITED_BASIS_FEATURES))
    if leaked:
        raise ValueError(
            f"Model uses prohibited-basis features {leaked}. "
            "Retrain with the current data_loader before serving."
        )


def build_category_maps(df: pd.DataFrame) -> dict[str, list[str]]:
    """Sorted vocabulary of observed values for each categorical column."""
    maps: dict[str, list[str]] = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            values = df[col].dropna().astype(str).unique().tolist()
            maps[col] = sorted(values)
    return maps


def encode_categoricals(X: pd.DataFrame, category_maps: dict[str, list[str]]) -> pd.DataFrame:
    """Map each categorical value to its index in the saved vocabulary.

    Values not in the vocabulary (including None) become NaN.
    """
    X = X.copy()
    for col, vocab in category_maps.items():
        if col not in X.columns:
            continue
        index = {v: float(i) for i, v in enumerate(vocab)}
        X[col] = X[col].map(lambda v: index.get(str(v), np.nan) if pd.notna(v) else np.nan)
        X[col] = X[col].astype(float)
    return X
