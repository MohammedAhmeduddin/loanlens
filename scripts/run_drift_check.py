"""
Standalone drift check script.
Run weekly via GitHub Actions or cron.

Usage:
    python scripts/run_drift_check.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from sqlalchemy import create_engine
import pandas as pd

from loanlens.config import get_settings
from loanlens.monitoring.drift import DriftMonitor, compute_baseline, MONITORED_FEATURES
from loanlens.monitoring.alerts import send_slack_alert


def main():
    settings = get_settings()
    logger.info("Starting drift check...")

    engine = create_engine(settings.database_url_sync)

    # Load recent data (last 10K predictions)
    logger.info("Loading recent feature data...")
    try:
        query = """
            SELECT *
            FROM staging_features.feat_master
            ORDER BY RANDOM()
            LIMIT 10000
        """
        current_df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(current_df):,} recent rows for drift check")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Compute baseline if it doesn't exist
    from pathlib import Path
    baseline_path = Path("data/processed/feature_baseline.json")
    if not baseline_path.exists():
        logger.info("No baseline found — computing from current data...")
        compute_baseline(current_df)
        logger.info("Baseline computed. Run again tomorrow to compare distributions.")
        return

    # Run drift check
    monitor = DriftMonitor()
    report = monitor.check_drift(current_df)

    # Print summary
    logger.info(f"Drift check results:")
    logger.info(f"  Features checked: {report['total_features_checked']}")
    logger.info(f"  Severe drift: {report['features_with_severe_drift']}")
    logger.info(f"  Moderate drift: {report['features_with_moderate_drift']}")
    logger.info(f"  Requires retraining: {report['requires_retraining']}")

    if report["alerts"]:
        for alert in report["alerts"]:
            logger.warning(f"  ALERT: {alert['message']}")
        send_slack_alert(report)
    else:
        logger.info("  No drift detected — model is stable")


if __name__ == "__main__":
    main()
