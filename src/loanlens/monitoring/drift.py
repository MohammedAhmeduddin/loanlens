"""
Population Stability Index (PSI) drift detection.
Monitors feature distributions to detect model input drift.
PSI < 0.10: No drift
PSI 0.10-0.20: Moderate drift — monitor closely
PSI > 0.20: Severe drift — retrain model
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings


# Features to monitor for drift
MONITORED_FEATURES = [
    "ext_source_mean",
    "debt_to_income",
    "employment_years",
    "bureau_overdue_count",
    "bureau_debt_to_credit",
    "prev_refused_count",
    "late_payment_rate",
    "amt_income_total",
    "age_years",
    "bureau_active_loans",
]

BASELINE_PATH = Path("data/processed/feature_baseline.json")


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10
) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Args:
        expected: Baseline distribution (training data)
        actual: Current distribution (incoming data)
        buckets: Number of bins for discretization

    Returns:
        PSI value
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    # Compute bucket frequencies
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero and log(0)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    # PSI formula
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi, 6))


def compute_baseline(df: pd.DataFrame) -> dict:
    """
    Compute baseline statistics from training data.
    Saves to JSON file for future drift comparisons.

    Args:
        df: Training feature DataFrame

    Returns:
        Baseline statistics dict
    """
    baseline = {}

    for feature in MONITORED_FEATURES:
        if feature not in df.columns:
            continue

        values = df[feature].dropna().values
        if len(values) == 0:
            continue

        baseline[feature] = {
            "values": values.tolist()[:5000],  # Store sample for PSI
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "count": len(values),
        }

    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f)

    logger.info(f"Baseline saved to {BASELINE_PATH} ({len(baseline)} features)")
    return baseline


def load_baseline() -> dict:
    """Load baseline statistics from file."""
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline not found at {BASELINE_PATH}. "
            f"Run compute_baseline() first."
        )
    with open(BASELINE_PATH) as f:
        return json.load(f)


class DriftMonitor:
    """
    Monitors feature drift using PSI.
    Compares incoming data distribution against training baseline.
    """

    def __init__(self):
        self.settings = get_settings()
        self.baseline = None

    def load_baseline(self) -> None:
        """Load pre-computed baseline."""
        self.baseline = load_baseline()
        logger.info(f"Loaded baseline for {len(self.baseline)} features")

    def check_drift(self, current_df: pd.DataFrame) -> dict:
        """
        Check PSI drift for all monitored features.

        Args:
            current_df: Recent incoming data

        Returns:
            Drift report dict with PSI values and alerts
        """
        if self.baseline is None:
            self.load_baseline()

        results = {}
        alerts = []
        settings = get_settings()

        for feature in MONITORED_FEATURES:
            if feature not in self.baseline or feature not in current_df.columns:
                continue

            expected = np.array(self.baseline[feature]["values"])
            actual = current_df[feature].dropna().values

            if len(actual) < 10:
                continue

            psi = compute_psi(expected, actual)

            if psi >= settings.psi_severe_threshold:
                status = "SEVERE"
                alerts.append({
                    "feature": feature,
                    "psi": psi,
                    "status": status,
                    "message": f"PSI {psi:.4f} >= {settings.psi_severe_threshold} — retrain immediately"
                })
            elif psi >= settings.psi_moderate_threshold:
                status = "MODERATE"
                alerts.append({
                    "feature": feature,
                    "psi": psi,
                    "status": status,
                    "message": f"PSI {psi:.4f} >= {settings.psi_moderate_threshold} — monitor closely"
                })
            else:
                status = "OK"

            results[feature] = {
                "psi": psi,
                "status": status,
                "baseline_mean": self.baseline[feature]["mean"],
                "current_mean": float(np.mean(actual)) if len(actual) > 0 else None,
            }

            logger.info(f"  {feature}: PSI={psi:.4f} [{status}]")

        report = {
            "total_features_checked": len(results),
            "features_with_severe_drift": sum(1 for r in results.values() if r["status"] == "SEVERE"),
            "features_with_moderate_drift": sum(1 for r in results.values() if r["status"] == "MODERATE"),
            "alerts": alerts,
            "feature_results": results,
            "requires_retraining": any(r["status"] == "SEVERE" for r in results.values()),
        }

        logger.info(
            f"Drift check complete: "
            f"{report['features_with_severe_drift']} severe, "
            f"{report['features_with_moderate_drift']} moderate"
        )

        return report
