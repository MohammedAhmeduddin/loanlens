"""Tests for credit scoring model components."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── data_loader tests ──────────────────────────────────────────

class TestDataLoader:
    @patch("loanlens.model.data_loader.create_engine")
    @patch("pandas.read_sql")
    def test_load_features_returns_dataframe(self, mock_sql, mock_engine):
        mock_sql.return_value = pd.DataFrame({
            "sk_id_curr": [1, 2],
            "target": [0, 1],
            "amt_credit": [100000, 200000],
        })
        from loanlens.model.data_loader import load_features
        df = load_features(limit=2)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_preprocess_features_removes_target(self):
        from loanlens.model.data_loader import preprocess_features
        df = pd.DataFrame({
            "sk_id_curr": [1, 2, 3],
            "target": [0, 1, 0],
            "ext_source_2": [0.5, 0.3, 0.7],
            "debt_to_income": [2.0, 5.0, 1.5],
            "is_unemployed": [False, True, False],
        })
        X, y, features = preprocess_features(df)
        assert "target" not in X.columns
        assert "sk_id_curr" not in X.columns
        assert len(y) == 3

    def test_preprocess_features_bool_to_int(self):
        from loanlens.model.data_loader import preprocess_features
        df = pd.DataFrame({
            "sk_id_curr": [1],
            "target": [0],
            "is_unemployed": [True],
            "ext_source_2": [0.5],
        })
        X, y, _ = preprocess_features(df)
        assert X["is_unemployed"].dtype in [int, np.int64, np.int32]

    def test_preprocess_features_returns_feature_names(self):
        from loanlens.model.data_loader import preprocess_features
        df = pd.DataFrame({
            "sk_id_curr": [1],
            "target": [0],
            "ext_source_2": [0.5],
            "debt_to_income": [2.0],
        })
        X, y, features = preprocess_features(df)
        assert isinstance(features, list)
        assert len(features) == len(X.columns)


# ── evaluate tests ─────────────────────────────────────────────

class TestEvaluate:
    def test_ks_statistic_perfect_separation(self):
        from loanlens.model.evaluate import ks_statistic
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        ks = ks_statistic(y_true, y_prob)
        assert 0 < ks <= 1.0

    def test_ks_statistic_range(self):
        from loanlens.model.evaluate import ks_statistic
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.2, 0.7, 0.3, 0.8, 0.1, 0.9])
        ks = ks_statistic(y_true, y_prob)
        assert 0 <= ks <= 1

    def test_gini_coefficient_range(self):
        from loanlens.model.evaluate import gini_coefficient
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        gini = gini_coefficient(y_true, y_prob)
        assert 0 <= gini <= 1

    def test_gini_equals_2auc_minus_1(self):
        from loanlens.model.evaluate import gini_coefficient
        from sklearn.metrics import roc_auc_score
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        gini = gini_coefficient(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        assert abs(gini - (2 * auc - 1)) < 1e-6

    def test_compute_metrics_returns_dict(self):
        from loanlens.model.evaluate import compute_metrics
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        X = pd.DataFrame({"a": [1, 2]})
        y = pd.Series([0, 1])
        metrics = compute_metrics(model, X, y, prefix="test")
        assert "test_auc" in metrics
        assert "test_ks" in metrics
        assert "test_gini" in metrics

    def test_compute_metrics_prefix_empty(self):
        from loanlens.model.evaluate import compute_metrics
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        X = pd.DataFrame({"a": [1, 2]})
        y = pd.Series([0, 1])
        metrics = compute_metrics(model, X, y, prefix="")
        assert "auc" in metrics


# ── explain tests ──────────────────────────────────────────────

class TestCreditExplainer:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        return model

    @pytest.fixture
    def sample_features(self):
        return pd.DataFrame({
            "ext_source_mean": [0.35],
            "debt_to_income": [3.5],
            "bureau_overdue_count": [2],
            "employment_years": [1.5],
            "late_payment_rate": [0.3],
        })

    @patch("shap.TreeExplainer")
    def test_explainer_initializes(self, mock_shap, mock_model):
        from loanlens.model.explain import CreditExplainer
        explainer = CreditExplainer(mock_model)
        assert explainer.model == mock_model

    @patch("shap.TreeExplainer")
    def test_get_shap_factors_returns_list(self, mock_shap, mock_model, sample_features):
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = np.array([[-0.4, -0.2, -0.1, 0.1, 0.05]])
        mock_shap.return_value = mock_explainer_instance

        from loanlens.model.explain import CreditExplainer
        explainer = CreditExplainer(mock_model)
        factors = explainer.get_shap_factors(sample_features, n_top=3)
        assert isinstance(factors, list)
        assert len(factors) == 3

    @patch("shap.TreeExplainer")
    def test_get_shap_factors_has_required_keys(self, mock_shap, mock_model, sample_features):
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = np.array([[-0.4, -0.2, -0.1, 0.1, 0.05]])
        mock_shap.return_value = mock_explainer_instance

        from loanlens.model.explain import CreditExplainer
        explainer = CreditExplainer(mock_model)
        factors = explainer.get_shap_factors(sample_features, n_top=2)
        for f in factors:
            assert "rank" in f
            assert "feature" in f
            assert "shap_value" in f
            assert "direction" in f
            assert "label" in f
            assert "cfpb_code" in f

    @patch("shap.TreeExplainer")
    def test_build_rag_query_returns_string(self, mock_shap, mock_model, sample_features):
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = np.array([[-0.4, -0.2, -0.1, 0.1, 0.05]])
        mock_shap.return_value = mock_explainer_instance

        from loanlens.model.explain import CreditExplainer
        explainer = CreditExplainer(mock_model)
        factors = explainer.get_shap_factors(sample_features, n_top=3)
        query = explainer.build_rag_query(factors)
        assert isinstance(query, str)
        assert "ECOA" in query or "FCRA" in query

    @patch("shap.TreeExplainer")
    def test_direction_increases_risk_for_negative_shap(self, mock_shap, mock_model, sample_features):
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = np.array([[-0.9, -0.2, -0.1, 0.1, 0.05]])
        mock_shap.return_value = mock_explainer_instance

        from loanlens.model.explain import CreditExplainer
        explainer = CreditExplainer(mock_model)
        factors = explainer.get_shap_factors(sample_features, n_top=1)
        assert factors[0]["direction"] == "increases_risk"

    @patch("shap.TreeExplainer")
    def test_get_adverse_action_codes(self, mock_shap, mock_model, sample_features):
        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = np.array([[-0.4, -0.2, -0.1, 0.1, 0.05]])
        mock_shap.return_value = mock_explainer_instance

        from loanlens.model.explain import CreditExplainer
        explainer = CreditExplainer(mock_model)
        factors = explainer.get_shap_factors(sample_features, n_top=3)
        codes = explainer.get_adverse_action_codes(factors)
        assert isinstance(codes, list)
        assert len(codes) <= 5
