"""
MLflow model registry management.
Champion/challenger promotion workflow.
"""

import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loanlens.config import get_settings


def get_client() -> MlflowClient:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def list_model_versions(model_name: str = None) -> list:
    """List all versions of the credit scoring model."""
    settings = get_settings()
    client = get_client()
    name = model_name or settings.mlflow_model_name

    versions = client.search_model_versions(f"name='{name}'")
    logger.info(f"Found {len(versions)} version(s) of '{name}'")

    for v in versions:
        logger.info(
            f"  Version {v.version} | "
            f"Stage: {v.current_stage} | "
            f"Run: {v.run_id[:8]}..."
        )
    return versions


def promote_to_staging(version: int, model_name: str = None) -> None:
    """Promote a model version to Staging."""
    settings = get_settings()
    client = get_client()
    name = model_name or settings.mlflow_model_name

    client.transition_model_version_stage(
        name=name,
        version=version,
        stage="Staging",
        archive_existing_versions=False
    )
    logger.info(f"Model '{name}' version {version} promoted to Staging")


def promote_to_production(version: int, model_name: str = None) -> None:
    """
    Promote a model version to Production.
    Archives any existing Production version automatically.
    """
    settings = get_settings()
    client = get_client()
    name = model_name or settings.mlflow_model_name

    client.transition_model_version_stage(
        name=name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    logger.info(f"Model '{name}' version {version} promoted to Production")


def get_production_model_info(model_name: str = None) -> dict:
    """Get metadata about the current Production model."""
    settings = get_settings()
    client = get_client()
    name = model_name or settings.mlflow_model_name

    versions = client.get_latest_versions(name, stages=["Production"])

    if not versions:
        logger.warning(f"No Production model found for '{name}'")
        return {}

    v = versions[0]
    run = client.get_run(v.run_id)

    info = {
        "model_name": name,
        "version": v.version,
        "stage": v.current_stage,
        "run_id": v.run_id,
        "val_auc": run.data.metrics.get("val_auc"),
        "test_auc": run.data.metrics.get("test_auc"),
        "val_ks": run.data.metrics.get("val_ks"),
        "val_gini": run.data.metrics.get("val_gini"),
        "n_features": run.data.params.get("n_features"),
        "train_size": run.data.params.get("train_size"),
    }

    logger.info(f"Production model: {name} v{v.version}")
    logger.info(f"  Val AUC: {info['val_auc']} | Test AUC: {info['test_auc']}")

    return info


if __name__ == "__main__":
    # Promote version 1 to Production
    settings = get_settings()
    list_model_versions()
    promote_to_production(version=1)
    info = get_production_model_info()
    print(f"\nProduction model info: {info}")
