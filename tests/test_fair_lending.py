"""
Fair-lending and explanation-correctness tests.

These guard the three things a credit model must get right for ECOA /
Regulation B: no prohibited-basis inputs, adverse-action reasons that point
the right way, and reasons that are honest about what they mean.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loanlens.model.features import (  # noqa: E402
    PROHIBITED_BASIS_FEATURES,
    assert_no_prohibited,
    build_category_maps,
    encode_categoricals,
)


def _feature_store_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sk_id_curr": np.arange(n),
        "target": rng.integers(0, 2, n),
        "code_gender": rng.choice(["F", "M"], n),
        "name_family_status": rng.choice(["Married", "Single / not married"], n),
        "age_years": rng.uniform(21, 70, n),
        "employed_to_age_ratio": rng.uniform(0, 0.5, n),
        "cnt_children": rng.integers(0, 3, n),
        "name_housing_type": rng.choice(["House / apartment", "Rented apartment", None], n),
        "flag_own_car": rng.choice(["Y", "N"], n),
        "ext_source_mean": rng.uniform(0, 1, n),
        "late_payment_rate": rng.uniform(0, 1, n),
        "is_unemployed": rng.choice([True, False], n),
    })


# ── Prohibited bases never reach the model ─────────────────────

class TestProhibitedFeatures:
    def test_preprocess_drops_every_prohibited_feature(self):
        from loanlens.model.data_loader import preprocess_features
        X, y, names = preprocess_features(_feature_store_frame())
        assert not set(names) & set(PROHIBITED_BASIS_FEATURES)
        assert not set(X.columns) & set(PROHIBITED_BASIS_FEATURES)

    def test_preprocess_keeps_permitted_features(self):
        from loanlens.model.data_loader import preprocess_features
        _, _, names = preprocess_features(_feature_store_frame())
        for kept in ["ext_source_mean", "late_payment_rate", "cnt_children", "name_housing_type"]:
            assert kept in names

    def test_assert_no_prohibited_raises(self):
        with pytest.raises(ValueError, match="age_years"):
            assert_no_prohibited(["ext_source_mean", "age_years"])

    def test_assert_no_prohibited_passes_clean_list(self):
        assert_no_prohibited(["ext_source_mean", "late_payment_rate"])


# ── Categorical encoding is a saved, deterministic vocabulary ──

class TestCategoryEncoding:
    def test_vocabulary_is_sorted_and_ignores_missing(self):
        maps = build_category_maps(_feature_store_frame())
        assert maps["name_housing_type"] == ["House / apartment", "Rented apartment"]

    def test_training_and_serving_encode_identically(self):
        from loanlens.model.data_loader import preprocess_features
        df = _feature_store_frame()
        maps = build_category_maps(df)
        X_train, _, _ = preprocess_features(df, maps)
        served = encode_categoricals(df[["name_housing_type"]], maps)
        pd.testing.assert_series_equal(
            X_train["name_housing_type"].reset_index(drop=True),
            served["name_housing_type"].reset_index(drop=True),
        )

    def test_unknown_and_missing_become_nan(self):
        maps = {"name_housing_type": ["House / apartment"]}
        out = encode_categoricals(pd.DataFrame({"name_housing_type": ["Boat", None]}), maps)
        assert out["name_housing_type"].isna().all()


# ── Adverse-action direction (regression for the inverted-sign bug) ─

class TestRiskDirection:
    @pytest.fixture(scope="class")
    def trained(self):
        import xgboost as xgb
        rng = np.random.default_rng(0)
        n = 4000
        X = pd.DataFrame({
            "late_payment_rate": rng.uniform(0, 1, n),
            "ext_source_mean": rng.uniform(0, 1, n),
        })
        # 1 = default. Late payments raise default; good external score lowers it.
        p = 0.05 + 0.5 * X["late_payment_rate"] + 0.3 * (1 - X["ext_source_mean"])
        y = (rng.uniform(size=n) < p.clip(0, 1)).astype(int)
        model = xgb.XGBClassifier(n_estimators=60, max_depth=3, random_state=0).fit(X, y)
        return model

    def test_bad_behaviour_is_reported_as_increasing_risk(self, trained):
        from loanlens.model.explain import CreditExplainer
        risky = pd.DataFrame({"late_payment_rate": [0.95], "ext_source_mean": [0.5]})
        factors = CreditExplainer(trained).get_shap_factors(risky, n_top=2)
        late = next(f for f in factors if f["feature"] == "late_payment_rate")
        assert late["shap_value"] > 0
        assert late["direction"] == "increases_risk"

    def test_strength_is_never_cited_as_a_denial_reason(self, trained):
        from loanlens.model.explain import CreditExplainer
        # Excellent external score, many late payments
        applicant = pd.DataFrame({"late_payment_rate": [0.95], "ext_source_mean": [0.98]})
        explainer = CreditExplainer(trained)
        factors = explainer.get_shap_factors(applicant, n_top=2)
        codes = explainer.get_adverse_action_codes(factors)
        assert "A1 - Delinquent past or present credit obligations" in codes
        assert "A9 - Credit score" not in codes


# ── Reasons must be honest ─────────────────────────────────────

class TestHonestReasons:
    def test_no_mapping_for_prohibited_or_mislabeled_features(self):
        from loanlens.model.explain import FEATURE_TO_REGULATORY
        for feature in [*PROHIBITED_BASIS_FEATURES, "name_education_type", "organization_type"]:
            assert feature not in FEATURE_TO_REGULATORY

    def test_unmapped_feature_is_not_citable(self):
        from loanlens.model.explain import is_citable_reason
        factor = {"direction": "increases_risk", "cfpb_code": None}
        assert not is_citable_reason(factor)

    def test_unmapped_feature_never_shows_raw_column_name(self):
        from unittest.mock import MagicMock, patch
        from loanlens.model.explain import UNMAPPED_LABEL, CreditExplainer
        with patch("shap.TreeExplainer") as tree:
            tree.return_value.shap_values.return_value = np.array([[0.8, 0.1]])
            explainer = CreditExplainer(MagicMock())
            X = pd.DataFrame({"some_internal_column": [1.0], "ext_source_mean": [0.2]})
            factors = explainer.get_shap_factors(X, n_top=2)
        unmapped = factors[0]
        assert unmapped["label"] == UNMAPPED_LABEL
        assert unmapped["cfpb_code"] is None
        assert explainer.get_adverse_action_codes(factors) == ["A9 - Credit score"]
        assert "some_internal_column" not in explainer.build_rag_query(factors)


# ── Fairness audit ─────────────────────────────────────────────

class TestFairnessAudit:
    def test_equal_rates_have_ratio_one(self):
        from loanlens.model.fairness import adverse_impact
        approved = pd.Series([1, 0] * 100)
        group = pd.Series(["F", "F", "M", "M"] * 50)
        result = adverse_impact(approved, group)
        assert result["min_ratio"] == 1.0
        assert result["flagged"] == []

    def test_four_fifths_rule_flags_disadvantaged_group(self):
        from loanlens.model.fairness import adverse_impact
        approved = pd.Series([1] * 90 + [0] * 10 + [1] * 50 + [0] * 50)
        group = pd.Series(["A"] * 100 + ["B"] * 100)
        result = adverse_impact(approved, group)
        assert result["groups"]["B"]["impact_ratio"] == pytest.approx(0.5 / 0.9, abs=1e-3)
        assert result["flagged"] == ["B"]

    def test_small_groups_are_skipped(self):
        from loanlens.model.fairness import adverse_impact
        approved = pd.Series([1] * 100 + [0] * 5)
        group = pd.Series(["A"] * 100 + ["tiny"] * 5)
        assert "tiny" not in adverse_impact(approved, group)["groups"]

    def test_report_covers_sex_and_age_bands(self):
        from loanlens.model.fairness import fairness_report
        df = _feature_store_frame(n=400)
        probs = np.random.default_rng(1).uniform(0, 1, len(df))
        report = fairness_report(probs, df[["code_gender", "age_years"]], approve_below=0.35)
        assert set(report["sex"]["groups"]) == {"F", "M"}
        assert set(report["age_band"]["groups"]) <= {"under_25", "25_39", "40_61", "62_plus"}

    def test_no_approvals_gives_undefined_ratio_not_nan(self):
        from loanlens.model.fairness import adverse_impact
        result = adverse_impact(pd.Series([0] * 100), pd.Series(["F", "M"] * 50))
        assert result["min_ratio"] is None
        assert result["note"] == "no approvals"

    def test_age_band_boundaries(self):
        from loanlens.model.fairness import age_band
        bands = age_band(pd.Series([24.9, 25, 61.9, 62])).tolist()
        assert bands == ["under_25", "25_39", "40_61", "62_plus"]


# ── Calibration and cutoff trade-offs ──────────────────────────

class TestGroupOutcomes:
    def test_repayers_turned_away_and_calibration(self):
        from loanlens.model.fairness import group_outcomes
        # Group A: 40 repayers (10 declined), 10 defaulters (all declined)
        p = [0.9] * 10 + [0.1] * 30 + [0.9] * 10
        y = [0] * 40 + [1] * 10
        out = group_outcomes(p, y, pd.Series(["A"] * 50), approve_below=0.5)["A"]
        assert out["repayers_turned_away"] == 0.25
        assert out["defaulters_caught"] == 1.0
        assert out["actual_default"] == 0.2
        assert out["mean_score"] == pytest.approx((0.9 * 20 + 0.1 * 30) / 50, abs=1e-4)

    def test_small_groups_skipped(self):
        from loanlens.model.fairness import group_outcomes
        out = group_outcomes([0.1] * 5, [0] * 5, pd.Series(["tiny"] * 5), approve_below=0.5)
        assert out == {}


class TestThresholdTable:
    def test_higher_cutoff_approves_more_and_turns_away_fewer(self):
        from loanlens.model.fairness import threshold_table
        df = _feature_store_frame(n=600)
        rng = np.random.default_rng(2)
        probs = rng.uniform(0, 0.6, len(df))
        y = (rng.uniform(size=len(df)) < probs).astype(int)
        table = threshold_table(probs, y, df[["code_gender", "age_years"]], cutoffs=[0.1, 0.3, 0.5])
        assert list(table["cutoff"]) == [0.1, 0.3, 0.5]
        assert table["approval_rate"].is_monotonic_increasing
        assert table["repayers_turned_away"].is_monotonic_decreasing
        for col in ["default_rate_approved", "defaulters_caught", "min_ratio_sex",
                    "min_ratio_age", "turned_away_under_25", "turned_away_62_plus"]:
            assert col in table.columns


class TestCalibrationMetrics:
    def test_compute_metrics_reports_calibration(self):
        from unittest.mock import MagicMock
        from loanlens.model.evaluate import compute_metrics
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        m = compute_metrics(model, pd.DataFrame({"a": range(4)}), pd.Series([0, 0, 1, 1]), prefix="test")
        assert m["test_mean_predicted"] == pytest.approx(0.35)
        assert m["test_actual_default_rate"] == 0.5
        assert 0 <= m["test_brier"] <= 1


def test_class_weighting_is_off_by_default():
    import inspect
    from loanlens.model.train import train_model
    assert inspect.signature(train_model).parameters["use_class_weight"].default is False
