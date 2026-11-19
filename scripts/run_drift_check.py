"""
Compare recent applications against the training baseline with PSI.

The baseline is written by train.py (data/processed/feature_baseline.json)
from the training split. There is no production traffic in this project,
so the "current" data is a file you provide (CSV or Parquet with the
feature columns, e.g. a month of scored applications).

Usage:
    python scripts/run_drift_check.py --current recent_applications.csv

Exit code: 0 = no severe drift, 1 = severe drift (retrain), 2 = setup error.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from loguru import logger

from loanlens.monitoring.alerts import send_slack_alert
from loanlens.monitoring.drift import BASELINE_PATH, DriftMonitor


def load_current(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", type=Path, required=True, help="CSV/Parquet of recent applications")
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--slack", action="store_true", help="Send a Slack alert if drift is found")
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        logger.error(f"No baseline at {args.baseline}. Run train.py first; it writes the baseline.")
        return 2
    if not args.current.exists():
        logger.error(f"Current data file not found: {args.current}")
        return 2

    current = load_current(args.current)
    monitor = DriftMonitor()
    monitor.baseline = json.loads(args.baseline.read_text())
    report = monitor.check_drift(current)

    if report["total_features_checked"] == 0:
        logger.error("No monitored feature columns found in the current data")
        return 2

    logger.info(f"Checked {report['total_features_checked']} features on {len(current):,} rows")
    for feature, r in report["feature_results"].items():
        logger.info(f"  {feature:<24} PSI {r['psi']:.4f}  {r['status']}")
    for alert in report["alerts"]:
        logger.warning(f"  {alert['feature']}: {alert['message']}")
    if args.slack and report["alerts"]:
        send_slack_alert(report)

    return 1 if report["requires_retraining"] else 0


if __name__ == "__main__":
    sys.exit(main())
