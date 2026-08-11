"""Centralized MLflow tracking utilities.

This module is the SINGLE AUTHORITATIVE LOCATION for all MLflow interactions.
No other module or script in this repository should directly call `mlflow.log_*`.
"""

import os
from typing import Any, Dict, Optional
import mlflow

from src.utils.logging_utils import get_logger
from src.utils.paths import get_mlruns_dir

logger = get_logger(__name__)


def init_mlflow(experiment_name: str = "qor-smoke-test") -> str:
    """Initializes MLflow tracking URI to local ./mlruns and sets experiment.

    Returns the experiment ID string.
    """
    mlruns_path = get_mlruns_dir().resolve()
    mlruns_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"file:///{mlruns_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI set to: {tracking_uri}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new MLflow experiment '{experiment_name}' (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{experiment_name}' (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return experiment_id


def start_run(
    run_name: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, Any]] = None,
) -> mlflow.ActiveRun:
    """Starts a new MLflow run context."""
    formatted_tags = {k: str(v) for k, v in (tags or {}).items()}
    return mlflow.start_run(run_name=run_name, nested=nested, tags=formatted_tags)


def log_params(params: Dict[str, Any]) -> None:
    """Logs parameter dictionary to current active MLflow run.

    Flattens non-primitive values to strings if necessary.
    """
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run found. Call start_run() first.")

    str_params = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            str_params[k] = v
        else:
            str_params[k] = str(v)

    mlflow.log_params(str_params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Logs metric dictionary to current active MLflow run."""
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run found. Call start_run() first.")

    clean_metrics = {}
    for k, v in metrics.items():
        if v is not None:
            clean_metrics[k] = float(v)

    mlflow.log_metrics(clean_metrics, step=step)


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """Logs a local file artifact to the current MLflow run."""
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run found. Call start_run() first.")

    mlflow.log_artifact(local_path, artifact_path=artifact_path)


def log_dict(dictionary: Dict[str, Any], artifact_file: str) -> None:
    """Logs a dictionary directly as a JSON/YAML artifact to MLflow."""
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run found. Call start_run() first.")

    mlflow.log_dict(dictionary, artifact_file)


def end_run(status: str = "FINISHED") -> None:
    """Ends the current active MLflow run."""
    if mlflow.active_run():
        mlflow.end_run(status=status)
