"""
Build a model feature row from an API request the same way dbt builds
staging_features.feat_master, so scoring sees what training saw.

Mirrors dbt/loanlens/models/staging/stg_application.sql and
dbt/loanlens/models/features/feat_master.sql. tests/test_serving_features.py
parses feat_master.sql and fails if COALESCE_ZERO drifts from it.
"""

from __future__ import annotations

# feat_master.sql: COALESCE(<agg>, 0). "No record" in the joined table means 0.
COALESCE_ZERO: tuple[str, ...] = (
    "bureau_loan_count", "bureau_active_loans", "bureau_closed_loans",
    "bureau_overdue_count", "bureau_max_overdue_days", "bureau_avg_overdue_days",
    "bureau_total_credit", "bureau_total_debt", "bureau_total_overdue_amt",
    "bureau_debt_to_credit", "bureau_credit_type_count", "bureau_total_prolongations",
    "prev_app_count", "prev_approved_count", "prev_refused_count", "prev_canceled_count",
    "prev_approval_rate", "prev_avg_application_amt", "prev_max_credit", "prev_avg_credit",
    "instalment_count", "late_payment_count", "late_payment_rate",
    "avg_payment_deficit", "avg_days_late",
)


def _num(v):
    return None if v is None else float(v)


def derive_features(data: dict) -> dict:
    """Fill computed and "no record" features exactly as the dbt models do.

    Values the caller provided are kept. Features dbt leaves NULL when there
    is no record (e.g. bureau_oldest_credit_days) stay missing.
    """
    d = dict(data)
    income, credit, annuity = _num(d.get("amt_income_total")), _num(d.get("amt_credit")), _num(d.get("amt_annuity"))

    # stg_application.sql
    if d.get("debt_to_income") is None and income and income > 0 and credit is not None:
        d["debt_to_income"] = round(credit / income, 4)
    if d.get("credit_to_annuity") is None and annuity and annuity > 0 and credit is not None:
        d["credit_to_annuity"] = round(credit / annuity, 4)
    if d.get("income_per_person") is None and income is not None:
        fam = _num(d.get("cnt_fam_members"))
        d["income_per_person"] = round(income / fam, 2) if fam and fam > 0 else income
    if d.get("ext_source_mean") is None:
        vals = [d.get(f"ext_source_{i}") for i in (1, 2, 3)]
        vals = [v for v in vals if v is not None]
        d["ext_source_mean"] = round(sum(vals) / len(vals), 6) if vals else None
    if isinstance(d.get("is_unemployed"), bool):
        d["is_unemployed"] = int(d["is_unemployed"])

    # Record-presence flags (feat_master.sql: CASE WHEN <table>.sk_id_curr IS NOT NULL)
    if d.get("has_bureau_history") is None:
        d["has_bureau_history"] = int((d.get("bureau_loan_count") or 0) > 0)
    if d.get("has_prev_application") is None:
        d["has_prev_application"] = int((d.get("prev_app_count") or 0) > 0)
    if d.get("has_instalment_history") is None:
        d["has_instalment_history"] = int((d.get("instalment_count") or 0) > 0)

    # Aggregates implied by what was provided
    if d.get("late_payment_count") is None and d.get("instalment_count") and d.get("late_payment_rate") is not None:
        d["late_payment_count"] = round(d["late_payment_rate"] * d["instalment_count"])
    loans, active = d.get("bureau_loan_count"), d.get("bureau_active_loans")
    if d.get("bureau_closed_loans") is None and loans is not None and active is not None:
        d["bureau_closed_loans"] = d["bureau_loan_count"] - d["bureau_active_loans"]

    for f in COALESCE_ZERO:
        if d.get(f) is None:
            d[f] = 0

    # feat_master.sql totals
    d.setdefault("total_debt_burden", None)
    if d["total_debt_burden"] is None:
        d["total_debt_burden"] = (credit or 0) + d["bureau_total_debt"]
    if d.get("total_credit_applications") is None:
        d["total_credit_applications"] = (d.get("amt_req_credit_bureau_year") or 0) + d["prev_app_count"]
    if d.get("total_delinquency_count") is None:
        d["total_delinquency_count"] = d["bureau_overdue_count"] + d["late_payment_count"]
    return d
