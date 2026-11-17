"""
Fairness audit for the credit model.

The model never sees sex or age. This audit checks whether its decisions
still differ by those groups (for example through proxies), using the
adverse impact ratio: each group's approval rate divided by the highest
group's approval rate. A ratio below 0.80 (the "four-fifths rule") is a
common screening flag for disparate impact and should be investigated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FOUR_FIFTHS = 0.80
MIN_GROUP_SIZE = 30

AGE_BANDS = [0, 25, 40, 62, 200]
AGE_LABELS = ["under_25", "25_39", "40_61", "62_plus"]


def age_band(age_years: pd.Series) -> pd.Series:
    """Bucket ages; 62+ matches Regulation B's definition of elderly."""
    return pd.cut(age_years, bins=AGE_BANDS, labels=AGE_LABELS, right=False).astype(object)


def adverse_impact(approved: pd.Series, group: pd.Series) -> dict:
    """Approval rate per group and the ratio to the most-approved group.

    Groups smaller than MIN_GROUP_SIZE or with a missing label are skipped,
    because their rates are too noisy to judge.
    """
    frame = pd.DataFrame({"approved": approved.astype(float).values, "group": group.values})
    frame = frame.dropna(subset=["group"])
    stats = frame.groupby("group")["approved"].agg(["mean", "size"])
    stats = stats[stats["size"] >= MIN_GROUP_SIZE]
    if stats.empty:
        return {"groups": {}, "min_ratio": None, "flagged": []}

    best = stats["mean"].max()
    if best <= 0:
        # Nobody approved: ratios are undefined, not "fair".
        groups = {
            str(name): {"n": int(row["size"]), "approval_rate": 0.0, "impact_ratio": None}
            for name, row in stats.iterrows()
        }
        return {"groups": groups, "min_ratio": None, "flagged": [], "note": "no approvals"}

    groups = {}
    for name, row in stats.iterrows():
        ratio = float(row["mean"] / best)
        groups[str(name)] = {
            "n": int(row["size"]),
            "approval_rate": round(float(row["mean"]), 4),
            "impact_ratio": round(ratio, 4),
        }
    ratios = [g["impact_ratio"] for g in groups.values()]
    return {
        "groups": groups,
        "min_ratio": round(min(ratios), 4),
        "flagged": sorted(k for k, g in groups.items() if g["impact_ratio"] < FOUR_FIFTHS),
    }


def fairness_report(
    probabilities: np.ndarray,
    audit_df: pd.DataFrame,
    approve_below: float,
) -> dict:
    """Run the audit for sex and age band.

    Args:
        probabilities: Predicted probability of default per applicant
        audit_df: Rows aligned with probabilities, holding code_gender and age_years
        approve_below: Probability under which the decision is "approve"
    """
    approved = pd.Series(np.asarray(probabilities) < approve_below)
    report = {}
    if "code_gender" in audit_df.columns:
        sex = audit_df["code_gender"].reset_index(drop=True).where(lambda s: s.isin(["F", "M"]))
        report["sex"] = adverse_impact(approved, sex)
    if "age_years" in audit_df.columns:
        report["age_band"] = adverse_impact(approved, age_band(audit_df["age_years"].reset_index(drop=True)))
    return report
