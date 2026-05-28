"""Tests for drift monitoring."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestPSI:
    def test_psi_zero_for_identical_distributions(self):
        from loanlens.monitoring.drift import compute_psi
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data.copy())
        assert psi < 0.01

    def test_psi_high_for_different_distributions(self):
        from loanlens.monitoring.drift import compute_psi
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(5, 1, 1000)
        psi = compute_psi(expected, actual)
        assert psi > 0.2

    def test_psi_returns_float(self):
        from loanlens.monitoring.drift import compute_psi
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 100)
        actual = np.array([1.1, 2.1, 3.1, 4.1, 5.1] * 100)
        result = compute_psi(expected, actual)
        assert isinstance(result, float)

    def test_psi_handles_empty_arrays(self):
        from loanlens.monitoring.drift import compute_psi
        result = compute_psi(np.array([]), np.array([1.0, 2.0]))
        assert result == 0.0

    def test_psi_handles_nan_values(self):
        from loanlens.monitoring.drift import compute_psi
        expected = np.array([1.0, 2.0, np.nan, 3.0, 4.0] * 100)
        actual = np.array([1.0, 2.0, 3.0, np.nan, 4.0] * 100)
        result = compute_psi(expected, actual)
        assert isinstance(result, float)

    def test_psi_moderate_drift(self):
        from loanlens.monitoring.drift import compute_psi
        expected = np.random.normal(0, 1, 2000)
        actual = np.random.normal(0.8, 1, 2000)
        psi = compute_psi(expected, actual)
        assert psi >= 0.0


class TestDriftMonitor:
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "ext_source_mean": np.random.uniform(0.2, 0.8, 500),
            "debt_to_income": np.random.uniform(1, 8, 500),
            "employment_years": np.random.uniform(0, 20, 500),
            "bureau_overdue_count": np.random.randint(0, 5, 500).astype(float),
            "bureau_debt_to_credit": np.random.uniform(0, 1, 500),
            "prev_refused_count": np.random.randint(0, 3, 500).astype(float),
            "late_payment_rate": np.random.uniform(0, 0.5, 500),
            "amt_income_total": np.random.normal(75000, 20000, 500),
            "age_years": np.random.uniform(22, 65, 500),
            "bureau_active_loans": np.random.randint(0, 8, 500).astype(float),
        })

    @patch("loanlens.monitoring.drift.load_baseline")
    def test_check_drift_returns_report(self, mock_baseline, sample_df):
        mock_baseline.return_value = {
            feat: {
                "values": sample_df[feat].tolist(),
                "mean": float(sample_df[feat].mean()),
                "std": float(sample_df[feat].std()),
                "p25": float(sample_df[feat].quantile(0.25)),
                "p50": float(sample_df[feat].quantile(0.5)),
                "p75": float(sample_df[feat].quantile(0.75)),
                "count": len(sample_df),
            }
            for feat in sample_df.columns
        }

        from loanlens.monitoring.drift import DriftMonitor
        monitor = DriftMonitor()
        report = monitor.check_drift(sample_df)

        assert "total_features_checked" in report
        assert "features_with_severe_drift" in report
        assert "requires_retraining" in report
        assert "feature_results" in report
        assert "alerts" in report

    @patch("loanlens.monitoring.drift.load_baseline")
    def test_check_drift_detects_severe_drift(self, mock_baseline, sample_df):
        drifted_df = sample_df.copy()
        drifted_df["debt_to_income"] = np.random.normal(50, 5, 500)

        mock_baseline.return_value = {
            feat: {
                "values": sample_df[feat].tolist(),
                "mean": float(sample_df[feat].mean()),
                "std": float(sample_df[feat].std()),
                "p25": float(sample_df[feat].quantile(0.25)),
                "p50": float(sample_df[feat].quantile(0.5)),
                "p75": float(sample_df[feat].quantile(0.75)),
                "count": len(sample_df),
            }
            for feat in sample_df.columns
        }

        from loanlens.monitoring.drift import DriftMonitor
        monitor = DriftMonitor()
        report = monitor.check_drift(drifted_df)
        assert report["features_with_severe_drift"] >= 1
        assert report["requires_retraining"] is True

    @patch("loanlens.monitoring.drift.load_baseline")
    def test_check_drift_no_drift_stable_data(self, mock_baseline, sample_df):
        mock_baseline.return_value = {
            feat: {
                "values": sample_df[feat].tolist(),
                "mean": float(sample_df[feat].mean()),
                "std": float(sample_df[feat].std()),
                "p25": float(sample_df[feat].quantile(0.25)),
                "p50": float(sample_df[feat].quantile(0.5)),
                "p75": float(sample_df[feat].quantile(0.75)),
                "count": len(sample_df),
            }
            for feat in sample_df.columns
        }

        from loanlens.monitoring.drift import DriftMonitor
        monitor = DriftMonitor()
        report = monitor.check_drift(sample_df)
        assert report["requires_retraining"] is False


class TestAlerts:
    def test_send_slack_no_webhook_returns_false(self):
        from loanlens.monitoring.alerts import send_slack_alert
        with patch("loanlens.monitoring.alerts.get_settings") as mock_settings:
            mock_settings.return_value.slack_webhook_url = ""
            result = send_slack_alert({"alerts": [{"feature": "x", "psi": 0.3, "status": "SEVERE"}]})
            assert result is False

    def test_send_slack_no_alerts_returns_false(self):
        from loanlens.monitoring.alerts import send_slack_alert
        result = send_slack_alert({"alerts": []})
        assert result is False

    @patch("loanlens.monitoring.alerts.httpx.post")
    def test_send_slack_with_webhook_sends_request(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        from loanlens.monitoring.alerts import send_slack_alert
        with patch("loanlens.monitoring.alerts.get_settings") as mock_settings:
            mock_settings.return_value.slack_webhook_url = "https://hooks.slack.com/test"
            report = {
                "alerts": [{"feature": "debt_to_income", "psi": 0.35, "status": "SEVERE", "message": "test"}],
                "features_with_severe_drift": 1,
                "features_with_moderate_drift": 0,
            }
            result = send_slack_alert(report)
            assert result is True
            mock_post.assert_called_once()
