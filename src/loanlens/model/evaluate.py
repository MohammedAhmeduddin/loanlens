"""
Model evaluation metrics for credit scoring.
Computes AUC, KS statistic, and Gini coefficient.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from loguru import logger


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute KS statistic — max separation between
    cumulative default and non-default distributions.
    Industry standard for credit model validation.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Gini = 2 * AUC - 1.
    Standard credit risk metric — ranges 0 to 1.
    """
    auc = roc_auc_score(y_true, y_prob)
    return float(2 * auc - 1)


def compute_metrics(model, X, y, prefix: str = "") -> dict:
    """
    Compute full set of credit model metrics.

    Args:
        model: Trained XGBoost model
        X: Feature matrix
        y: True labels
        prefix: Metric name prefix (train/val/test)

    Returns:
        Dictionary of metric_name: value
    """
    y_prob = model.predict_proba(X)[:, 1]

    auc   = roc_auc_score(y, y_prob)
    ks    = ks_statistic(y.values, y_prob)
    gini  = gini_coefficient(y.values, y_prob)

    p = f"{prefix}_" if prefix else ""

    metrics = {
        f"{p}auc":  round(auc,  4),
        f"{p}ks":   round(ks,   4),
        f"{p}gini": round(gini, 4),
    }

    return metrics
