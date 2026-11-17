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


def group_outcomes(probabilities, y_true, group: pd.Series, approve_below: float) -> dict:
    """Per-group calibration and error rates at one cutoff.

    - actual_default vs mean_score: is the score a real probability for this group?
    - repayers_turned_away: share of people who repaid but were not approved.
      Large gaps between groups here mean good borrowers in one group pay
      a higher price for the model's mistakes.
    """
    t = pd.DataFrame({
        "p": np.asarray(probabilities, dtype=float),
        "y": np.asarray(y_true, dtype=int),
        "g": group.reset_index(drop=True).values,
    }).dropna(subset=["g"])
    t["approved"] = t["p"] < approve_below
    out = {}
    for name, part in t.groupby("g"):
        if len(part) < MIN_GROUP_SIZE:
            continue
        repayers = part[part["y"] == 0]
        defaulters = part[part["y"] == 1]
        out[str(name)] = {
            "n": int(len(part)),
            "actual_default": round(float(part["y"].mean()), 4),
            "mean_score": round(float(part["p"].mean()), 4),
            "approval_rate": round(float(part["approved"].mean()), 4),
            "repayers_turned_away": round(float((~repayers["approved"]).mean()), 4) if len(repayers) else None,
            "defaulters_caught": round(float((~defaulters["approved"]).mean()), 4) if len(defaulters) else None,
        }
    return out


def threshold_table(probabilities, y_true, audit_df: pd.DataFrame, cutoffs) -> pd.DataFrame:
    """Trade-offs of each candidate approval cutoff, so the threshold is a
    visible business choice: approval, losses, and fairness side by side."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    audit_df = audit_df.reset_index(drop=True)
    ages = age_band(audit_df["age_years"]) if "age_years" in audit_df else None
    rows = []
    for c in cutoffs:
        approved = p < c
        row = {
            "cutoff": c,
            "approval_rate": approved.mean(),
            "default_rate_approved": y[approved].mean() if approved.any() else np.nan,
            "defaulters_caught": (~approved[y == 1]).mean(),
            "repayers_turned_away": (~approved[y == 0]).mean(),
        }
        rep = fairness_report(p, audit_df, approve_below=c)
        row["min_ratio_sex"] = rep.get("sex", {}).get("min_ratio")
        row["min_ratio_age"] = rep.get("age_band", {}).get("min_ratio")
        if ages is not None:
            outcomes = group_outcomes(p, y, ages, approve_below=c)
            row["turned_away_under_25"] = outcomes.get("under_25", {}).get("repayers_turned_away")
            row["turned_away_62_plus"] = outcomes.get("62_plus", {}).get("repayers_turned_away")
        rows.append(row)
    return pd.DataFrame(rows).round(3)
