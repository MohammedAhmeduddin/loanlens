"""
Slack alerting for drift detection.
Sends webhook notifications when PSI thresholds are exceeded.
"""

import httpx
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings


def send_slack_alert(report: dict) -> bool:
    """
    Send drift alert to Slack webhook.

    Args:
        report: Drift report from DriftMonitor.check_drift()

    Returns:
        True if alert sent successfully
    """
    settings = get_settings()

    if not settings.slack_webhook_url:
        logger.info("Slack webhook not configured — skipping alert")
        return False

    if not report.get("alerts"):
        logger.info("No alerts to send")
        return False

    # Build Slack message
    severe = report["features_with_severe_drift"]
    moderate = report["features_with_moderate_drift"]

    color = "#FF0000" if severe > 0 else "#FFA500"
    title = "🚨 LoanLens Drift Alert" if severe > 0 else "⚠️ LoanLens Drift Warning"

    alert_lines = []
    for alert in report["alerts"]:
        alert_lines.append(
            f"• *{alert['feature']}*: PSI={alert['psi']:.4f} [{alert['status']}]"
        )

    message = {
        "attachments": [
            {
                "color": color,
                "title": title,
                "text": "\n".join(alert_lines),
                "fields": [
                    {
                        "title": "Severe Features",
                        "value": str(severe),
                        "short": True
                    },
                    {
                        "title": "Moderate Features",
                        "value": str(moderate),
                        "short": True
                    },
                    {
                        "title": "Action Required",
                        "value": "Retrain model immediately" if severe > 0 else "Monitor closely",
                        "short": False
                    }
                ],
                "footer": "LoanLens Drift Monitor",
            }
        ]
    }

    try:
        response = httpx.post(
            settings.slack_webhook_url,
            json=message,
            timeout=10.0
        )
        response.raise_for_status()
        logger.info("Slack alert sent successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")
        return False
