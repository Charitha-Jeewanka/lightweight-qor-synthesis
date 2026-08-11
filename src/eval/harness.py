"""Evaluation harness for running models across split protocols and logging to MLflow.

Strictly follows GEMINI.md §9, §10, and §11 conventions.
All MLflow logging is delegated to `src.tracking.mlflow_utils`.
"""

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config.schema import ExperimentConfig
from src.data.loaders import ModelingDataset
from src.data.splits import SplitResult, resolve_loco_folds, resolve_random_split
from src.eval.metrics import evaluate_all_metrics
from src.eval.profiling import PeakMemoryTracker, Timer, measure_inference_latency
from src.models.base import BaseQoRModel
from src.models.baselines import PerCircuitMeanBaseline
from src.tracking.mlflow_utils import (
    end_run,
    init_mlflow,
    log_artifact,
    log_metrics,
    log_params,
    start_run,
)
from src.utils.logging_utils import get_logger
from src.utils.paths import get_git_commit_hash, get_project_root

logger = get_logger(__name__)


def run_single_split(
    model: BaseQoRModel,
    dataset: ModelingDataset,
    split: SplitResult,
    config: ExperimentConfig,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Runs model fitting and evaluation on a single fold/split."""
    X = dataset.get_feature_matrix(
        encoding=config.encoding,
        use_circuit_features=(config.model_family != "seq_only"),
        use_structural_features=config.structural_features,
    )
    target_col = config.target if config.target in dataset.df.columns else dataset.target_name
    y = dataset.df[target_col].to_numpy(dtype=np.float32)

    X_train, y_train = X[split.train_indices], y[split.train_indices]
    X_val, y_val = X[split.val_indices], y[split.val_indices]
    X_test, y_test = X[split.test_indices], y[split.test_indices]

    val_data = (X_val, y_val) if len(X_val) > 0 else None

    train_circuits = dataset.df.iloc[split.train_indices]["circuit"].to_numpy()
    test_circuits_arr = dataset.df.iloc[split.test_indices]["circuit"].to_numpy()

    # Profile training
    with Timer() as timer, PeakMemoryTracker() as mem_tracker:
        if isinstance(model, PerCircuitMeanBaseline):
            model.fit(X_train, y_train, val_data=val_data, circuits_train=train_circuits)
        else:
            model.fit(X_train, y_train, val_data=val_data)

    train_time_s = timer.elapsed_s

    # Profile inference latency
    latency_sample = X_test[:1] if len(X_test) > 0 else X_train[:1]
    inference_latency_ms = measure_inference_latency(model, latency_sample, n_samples=1000)

    # Predict on test
    if isinstance(model, PerCircuitMeanBaseline):
        y_pred_test = model.predict(X_test, circuits_test=test_circuits_arr)
    else:
        y_pred_test = model.predict(X_test)

    # Compute metrics
    metrics = evaluate_all_metrics(y_test, y_pred_test, test_circuits_arr)

    # Log efficiency metrics
    metrics["train_time_s"] = train_time_s
    metrics["inference_latency_ms"] = inference_latency_ms
    metrics["peak_gpu_mem_mb"] = mem_tracker.peak_gpu_mem_mb
    metrics["peak_cpu_mem_mb"] = mem_tracker.peak_cpu_rss_mb
    metrics["n_parameters"] = float(model.param_count)

    # Build predictions dataframe
    test_df = dataset.df.iloc[split.test_indices]
    pred_df = pd.DataFrame(
        {
            "run_id": test_df["run_id"].values,
            "circuit": test_df["circuit"].values,
            "y_true": y_test,
            "y_pred": y_pred_test,
            "split": ["test"] * len(y_test),
            "fold": [split.fold_idx] * len(y_test),
        }
    )

    return metrics, pred_df


def run_experiment(
    model: BaseQoRModel,
    dataset: ModelingDataset,
    config: ExperimentConfig,
    temp_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executes a full evaluation protocol for a model and logs results to MLflow.

    Handles split resolution, fold execution, metric aggregation, MLflow logging,
    and Parquet artifact output per GEMINI.md §9.
    """
    config.validate()
    exp_id = init_mlflow(config.experiment_name)

    # Build mandatory parameters dict per §9
    params_to_log: Dict[str, Any] = {
        "model_family": config.model_family,
        "dataset_L": config.dataset_L,
        "target": config.target,
        "encoding": config.encoding,
        "structural_features": config.structural_features,
        "split_protocol": config.split_protocol,
        "exclude_hyp": config.exclude_hyp,
        "seed": config.seed,
        "config_hash": config.compute_config_hash(),
        "git_commit": get_git_commit_hash(),
    }
    params_to_log.update(model.get_params())
    params_to_log.update({f"config_{k}": v for k, v in config.model_params.items()})

    run_name = f"{model.name}_{config.split_protocol}_L{config.dataset_L}_seed{config.seed}"

    parent_run = start_run(
        run_name=run_name,
        tags={"rq": "eval_harness", "phase": "phase_2", "status": "running"},
    )

    try:
        log_params(params_to_log)

        all_fold_metrics: List[Dict[str, float]] = []
        all_pred_dfs: List[pd.DataFrame] = []

        if config.split_protocol == "random":
            splits = [resolve_random_split(dataset.df, exclude_hyp=config.exclude_hyp)]
        elif config.split_protocol == "loco":
            splits = resolve_loco_folds(dataset.df, exclude_hyp=config.exclude_hyp)
        else:
            raise ValueError(f"Unknown split protocol: {config.split_protocol}")

        for split in splits:
            logger.info(f"Executing split/fold {split.fold_idx}...")

            # Log fold sizes on first fold for parent run
            if split.fold_idx == 0:
                log_params(
                    {
                        "n_train_rows": len(split.train_indices),
                        "n_test_rows": len(split.test_indices),
                        "n_val_rows": len(split.val_indices),
                        "n_train_circuits": len(split.train_circuits),
                        "n_test_circuits": len(split.test_circuits),
                        "n_val_circuits": len(split.val_circuits),
                    }
                )

            # Optional child run for LOCO folds
            if config.split_protocol == "loco":
                test_circs_str = ",".join(sorted(split.test_circuits))
                train_circs_str = ",".join(sorted(split.train_circuits))

                with start_run(
                    run_name=f"{run_name}_fold_{split.fold_idx}",
                    nested=True,
                    tags={
                        "fold": str(split.fold_idx),
                        "test_circuits": test_circs_str,
                    },
                ):
                    log_params(
                        {
                            "n_train_rows": len(split.train_indices),
                            "n_test_rows": len(split.test_indices),
                            "n_val_rows": len(split.val_indices),
                            "n_train_circuits": len(split.train_circuits),
                            "n_test_circuits": len(split.test_circuits),
                            "n_val_circuits": len(split.val_circuits),
                            "test_circuits": test_circs_str,
                            "train_circuits": train_circs_str,
                        }
                    )
                    fold_metrics, pred_df = run_single_split(model, dataset, split, config)
                    log_metrics(fold_metrics)
            else:
                fold_metrics, pred_df = run_single_split(model, dataset, split, config)

            all_fold_metrics.append(fold_metrics)
            all_pred_dfs.append(pred_df)

        # Aggregate metrics across folds
        aggregated_metrics: Dict[str, float] = {}
        first_keys = all_fold_metrics[0].keys()
        for key in first_keys:
            vals = [m[key] for m in all_fold_metrics if key in m and not np.isnan(m[key])]
            if vals:
                aggregated_metrics[key] = float(np.mean(vals))
                if len(vals) > 1 and not key.startswith("spearman_per_circuit"):
                    aggregated_metrics[f"{key}_std"] = float(np.std(vals))

        log_metrics(aggregated_metrics)

        # Save prediction artifact as Parquet outside mlruns directory
        full_pred_df = pd.concat(all_pred_dfs, ignore_index=True)
        if temp_dir is None:
            temp_dir = get_project_root() / "tmp" / "artifacts"
        temp_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = temp_dir / f"predictions_{parent_run.info.run_id[:8]}.parquet"
        full_pred_df.to_parquet(parquet_path, index=False)
        log_artifact(str(parquet_path), artifact_path="predictions")

        end_run(status="FINISHED")
        logger.info(f"Experiment run '{run_name}' completed successfully.")

        return {
            "status": "complete",
            "metrics": aggregated_metrics,
            "predictions": full_pred_df,
            "run_id": parent_run.info.run_id,
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Experiment run failed with error: {e}\n{error_trace}")
        log_params({"failure_reason": str(e)[:250]})
        end_run(status="FAILED")
        raise e
