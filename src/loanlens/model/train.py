"""
XGBoost credit scoring model training with MLflow tracking.
Trains on Home Credit feat_master feature table.
Logs all metrics, parameters, and artifacts to MLflow.
"""

import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from loguru import logger
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings
from loanlens.model.data_loader import load_features, preprocess_features
from loanlens.model.features import FAIRNESS_AUDIT_COLS, build_category_maps
from loanlens.model.fairness import (
    age_band,
    fairness_report,
    group_outcomes,
    threshold_table,
)
from loanlens.model.evaluate import compute_metrics


def train_model(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    limit: int = None,
    use_class_weight: bool = False,
) -> str:
    """
    Train XGBoost credit scoring model with MLflow tracking.

    Args:
        All XGBoost hyperparameters
        limit: Row limit for dev runs (None = full dataset)
        use_class_weight: Reweight defaults by the class ratio. Off by default:
            it improves nothing for ranking here but inflates every score
            (~4.6x on Home Credit), so scores stop being real probabilities
            and fixed approval cutoffs decline far too many applicants.

    Returns:
        MLflow run_id
    """
    settings = get_settings()

    # Configure MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info("Starting XGBoost training run...")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # ── 1. Load data ──────────────────────────────────────────
        df = load_features(limit=limit)
        category_maps = build_category_maps(df)
        X, y, feature_names = preprocess_features(df, category_maps)
        audit_df = df[[c for c in FAIRNESS_AUDIT_COLS if c in df.columns]]

        # ── 2. Train/val/test split ───────────────────────────────
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.30,
            random_state=42,
            stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )

        logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

        # Class imbalance ratio
        class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
        scale_pos_weight = class_ratio if use_class_weight else 1.0
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

        # ── 3. Log parameters ─────────────────────────────────────
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "use_class_weight": use_class_weight,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_features": len(feature_names),
        }
        mlflow.log_params(params)

        # ── 4. Train model ────────────────────────────────────────
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=50,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        logger.info("Training XGBoost model...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        logger.info(f"Best iteration: {model.best_iteration}")

        # ── 5. Evaluate ───────────────────────────────────────────
        train_metrics = compute_metrics(model, X_train, y_train, prefix="train")
        val_metrics   = compute_metrics(model, X_val,   y_val,   prefix="val")
        test_metrics  = compute_metrics(model, X_test,  y_test,  prefix="test")

        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        all_metrics["best_iteration"] = model.best_iteration

        mlflow.log_metrics(all_metrics)

        logger.info(f"Train AUC: {train_metrics['train_auc']:.4f}")
        logger.info(f"Val AUC:   {val_metrics['val_auc']:.4f}")
        logger.info(f"Test AUC:  {test_metrics['test_auc']:.4f}")

        # ── 5b. Fairness audit (sex and age are not model inputs) ──
        p_test = model.predict_proba(X_test)[:, 1]
        logger.info(
            f"Calibration (test): mean score {float(np.mean(p_test)):.4f} "
            f"vs actual default rate {float(np.mean(y_test)):.4f}"
        )
        if audit_df.shape[1] == 0:
            logger.warning("No sex/age columns in feature store; fairness audit skipped")
        else:
            audit_test = audit_df.loc[X_test.index]
            report = fairness_report(p_test, audit_test, approve_below=settings.review_threshold)
            if "age_years" in audit_test:
                report["outcomes_by_age_band"] = group_outcomes(
                    p_test, y_test, age_band(audit_test["age_years"]), settings.review_threshold)
            if "code_gender" in audit_test:
                report["outcomes_by_sex"] = group_outcomes(
                    p_test, y_test, audit_test["code_gender"].where(audit_test["code_gender"].isin(["F", "M"])),
                    settings.review_threshold)
            mlflow.log_dict(report, "fairness/fairness_report.json")
            logger.info(f"Fairness at current review_threshold={settings.review_threshold}:")
            for attr in ("sex", "age_band"):
                result = report.get(attr)
                if not result:
                    continue
                if result["min_ratio"] is not None:
                    mlflow.log_metric(f"fairness_min_impact_ratio_{attr}", result["min_ratio"])
                logger.info(f"  [{attr}] min impact ratio: {result['min_ratio']} | flagged: {result['flagged']}")
            for key in ("outcomes_by_age_band", "outcomes_by_sex"):
                for group, g in report.get(key, {}).items():
                    logger.info(
                        f"  {group:10} n={g['n']:>6,}  actual={g['actual_default']:.3f}  "
                        f"score={g['mean_score']:.3f}  approved={g['approval_rate']:.3f}  "
                        f"repayers_turned_away={g['repayers_turned_away']}"
                    )

            # ── 5c. Cutoff trade-offs (pick decline/review thresholds from this) ──
            table = threshold_table(
                p_test, y_test, audit_test,
                cutoffs=[0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50],
            )
            table_path = "/tmp/threshold_table.csv"
            table.to_csv(table_path, index=False)
            mlflow.log_artifact(table_path, "fairness")
            logger.info("Cutoff trade-offs on the test set (approve if score < cutoff):\n" + table.to_string(index=False))

        # ── 6. Log feature importance ─────────────────────────────
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        importance_path = "/tmp/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, "feature_importance")

        # Log top 10 features
        top10 = importance_df.head(10).to_dict(orient="records")
        mlflow.log_dict({"top_features": top10}, "top_features.json")

        logger.info("Top 10 features:")
        for row in top10:
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # ── 7. Save feature names ─────────────────────────────────
        feature_names_path = "/tmp/feature_names.json"
        with open(feature_names_path, "w") as f:
            json.dump(feature_names, f)
        mlflow.log_artifact(feature_names_path, "feature_names")

        # Serving reads these to encode requests exactly like training
        mlflow.log_dict(feature_names, "preprocessing/feature_names.json")
        mlflow.log_dict(category_maps, "preprocessing/category_maps.json")

        # ── 8. Log model ──────────────────────────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=settings.mlflow_model_name,
        )

        logger.info(f"Model logged to MLflow registry as '{settings.mlflow_model_name}'")
        logger.info(f"Run ID: {run_id}")

        return run_id


if __name__ == "__main__":
    run_id = train_model(limit=None)
    logger.info(f"Training complete. Run ID: {run_id}")
