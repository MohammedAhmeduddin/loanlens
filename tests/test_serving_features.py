"""Serving builds features the same way the dbt feature SQL does."""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from loanlens.model.serving import COALESCE_ZERO, derive_features  # noqa: E402

FEAT_MASTER = (ROOT / "dbt/loanlens/models/features/feat_master.sql").read_text()


def test_coalesce_list_matches_feat_master_sql():
    in_sql = set(re.findall(r"COALESCE\(\w+\.\w+, 0\)\s+AS (\w+)", FEAT_MASTER))
    # total_* are sums of two COALESCEs, computed separately in derive_features
    in_sql = {f for f in in_sql if not f.startswith("total_")}
    assert in_sql and set(COALESCE_ZERO) == in_sql


def test_ratios_match_stg_application_sql():
    d = derive_features({"amt_income_total": 50000, "amt_credit": 200000,
                         "amt_annuity": 10000, "cnt_fam_members": 2})
    assert d["debt_to_income"] == 4.0
    assert d["credit_to_annuity"] == 20.0
    assert d["income_per_person"] == 25000.0


def test_income_per_person_without_family_size_is_income():
    assert derive_features({"amt_income_total": 50000})["income_per_person"] == 50000


def test_no_records_mean_zero_and_no_history():
    d = derive_features({"amt_income_total": 50000, "amt_credit": 100000})
    for f in COALESCE_ZERO:
        assert d[f] == 0
    assert d["has_bureau_history"] == 0 and d["has_prev_application"] == 0 and d["has_instalment_history"] == 0
    assert d["total_debt_burden"] == 100000


def test_counts_set_history_flags_and_totals():
    d = derive_features({"amt_credit": 100000, "bureau_loan_count": 4, "bureau_active_loans": 1,
                         "bureau_overdue_count": 2, "bureau_total_debt": 5000, "prev_app_count": 3,
                         "instalment_count": 20, "late_payment_rate": 0.25,
                         "amt_req_credit_bureau_year": 2})
    assert (d["has_bureau_history"], d["has_prev_application"], d["has_instalment_history"]) == (1, 1, 1)
    assert d["late_payment_count"] == 5
    assert d["bureau_closed_loans"] == 3
    assert d["total_debt_burden"] == 105000
    assert d["total_credit_applications"] == 5
    assert d["total_delinquency_count"] == 7


def test_provided_values_are_kept():
    d = derive_features({"amt_income_total": 1, "amt_credit": 1, "credit_to_annuity": 9.9, "has_bureau_history": 1})
    assert d["credit_to_annuity"] == 9.9 and d["has_bureau_history"] == 1


def test_null_when_no_record_stays_missing():
    d = derive_features({"amt_income_total": 1, "amt_credit": 1})
    assert d.get("bureau_oldest_credit_days") is None
    assert d.get("ext_source_mean") is None


class TestConsistencyChecks:
    BASE = {"amt_income_total": 50000, "amt_credit": 100000}

    @pytest.mark.parametrize("bad, message", [
        ({"bureau_overdue_count": 4}, "bureau_overdue_count cannot exceed"),
        ({"bureau_loan_count": 2, "bureau_active_loans": 3}, "bureau_active_loans cannot exceed"),
        ({"prev_refused_count": 2, "prev_app_count": 1}, "prev_refused_count cannot exceed"),
        ({"late_payment_rate": 0.5}, "requires instalment_count"),
        ({"bureau_loan_count": 2, "has_bureau_history": 0}, "contradicts"),
    ])
    def test_contradictory_history_rejected(self, bad, message):
        from pydantic import ValidationError
        from loanlens.api.schemas import ScoreRequest
        with pytest.raises(ValidationError, match=message):
            ScoreRequest(**self.BASE, **bad)

    def test_consistent_history_accepted(self):
        from loanlens.api.schemas import ScoreRequest
        ScoreRequest(**self.BASE, bureau_loan_count=5, bureau_overdue_count=4,
                     prev_app_count=3, prev_refused_count=3, instalment_count=24, late_payment_rate=0.6)


def test_api_row_has_no_missing_values_for_sql_zero_features():
    from loanlens.api.main import request_to_dataframe
    from loanlens.api.schemas import ScoreRequest
    names = list(COALESCE_ZERO) + ["credit_to_annuity", "income_per_person", "total_debt_burden",
                                   "has_bureau_history", "bureau_oldest_credit_days"]
    X = request_to_dataframe(ScoreRequest(amt_income_total=50000, amt_credit=100000, amt_annuity=8000), names, {})
    assert not X[[n for n in names if n != "bureau_oldest_credit_days"]].isna().any().any()
    assert np.isnan(X["bureau_oldest_credit_days"].iloc[0])   # dbt leaves this NULL too


def test_demo_payloads_are_consistent():
    from loanlens.api.schemas import ExplainRequest
    from loanlens.dashboard.samples import SAMPLES, build_payload
    for inputs in SAMPLES.values():
        ExplainRequest(**build_payload(*inputs))   # raises if contradictory


def test_demo_payload_leaves_only_commonly_missing_fields_empty():
    """Fields never missing in training must not be missing in the demo."""
    from loanlens.api.schemas import ExplainRequest
    from loanlens.dashboard.samples import SAMPLES, build_payload
    from loanlens.model.features import PROHIBITED_BASIS_FEATURES
    names = set(re.findall(r"AS (\w+)", FEAT_MASTER)) | set(re.findall(r"app\.(\w+),", FEAT_MASTER)) \
        | set(re.findall(r"(?:bureau|prev|instalment)\.(\w+),", FEAT_MASTER))
    names -= {"sk_id_curr", "target", *PROHIBITED_BASIS_FEATURES}
    # Often empty in the Home Credit training data too
    commonly_missing = {"ext_source_1", "ext_source_3", "occupation_type"}
    for inputs in SAMPLES.values():
        d = derive_features(ExplainRequest(**build_payload(*inputs)).model_dump())
        assert {n for n in names if d.get(n) is None} <= commonly_missing


def test_every_model_feature_can_be_sent_to_the_api():
    """Each feat_master column is either an API field or computed from them."""
    from loanlens.api.schemas import ScoreRequest
    from loanlens.model.features import PROHIBITED_BASIS_FEATURES
    names = set(re.findall(r"AS (\w+)", FEAT_MASTER)) | set(re.findall(r"app\.(\w+),", FEAT_MASTER)) \
        | set(re.findall(r"(?:bureau|prev|instalment)\.(\w+),", FEAT_MASTER))
    names -= {"sk_id_curr", "target", *PROHIBITED_BASIS_FEATURES}
    computed = {"total_debt_burden", "total_credit_applications", "total_delinquency_count"}
    assert names - computed - set(ScoreRequest.model_fields) == set()
