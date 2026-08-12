"""Phase 5 Experiment Driver: Small MLP QoR Prediction on CPU.

Executes pilot timing checkpoint and 25-trial randomized hyperparameter search across
Random Split and Leave-Circuits-Out (LOCO) protocols using PyTorch MLP on CPU.
Uses ProcessPoolExecutor (4 parallel workers) for speedup while recording isolated
uncontended train_time_s for direct RQ4 comparability.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch

from src.config.schema import ExperimentConfig
from src.data.loaders import ModelingDataset, load_modeling_table
from src.data.splits import SplitResult, resolve_loco_folds, resolve_random_split
from src.eval.harness import run_experiment, run_single_split
from src.eval.metrics import evaluate_all_metrics
from src.models.mlp import MLPQoRModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def generate_mlp_search_trials(n_trials: int = 25, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates 25 randomized MLP hyperparameter trials."""
    rng = np.random.default_rng(seed)
    hidden_options = [
        [64],
        [128],
        [64, 32],
        [128, 64],
        [256, 128],
        [128, 64, 32],
        [256, 128, 64],
    ]
    dropout_options = [0.0, 0.1, 0.2, 0.3]
    batch_size_options = [128, 256, 512]

    trials: List[Dict[str, Any]] = []
    for _ in range(n_trials):
        hidden_dims = hidden_options[int(rng.integers(0, len(hidden_options)))]
        dropout = float(dropout_options[int(rng.integers(0, len(dropout_options)))])
        learning_rate = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-2))))
        weight_decay = float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3))))
        batch_size = int(batch_size_options[int(rng.integers(0, len(batch_size_options)))])

        trials.append(
            {
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_epochs": 200,
                "patience": 20,
            }
        )

    return trials


def create_mlp_model(
    hyperparams: Dict[str, Any], n_continuous: int = 8, seed: int = 42
) -> MLPQoRModel:
    """Factory helper creating an MLPQoRModel instance."""
    return MLPQoRModel(
        hidden_dims=hyperparams.get("hidden_dims", [128, 64]),
        dropout=hyperparams.get("dropout", 0.1),
        learning_rate=hyperparams.get("learning_rate", 1e-3),
        weight_decay=hyperparams.get("weight_decay", 1e-4),
        batch_size=hyperparams.get("batch_size", 128),
        max_epochs=hyperparams.get("max_epochs", 200),
        patience=hyperparams.get("patience", 20),
        n_continuous=n_continuous,
        seed=seed,
    )


def _eval_single_trial_worker(args_tuple: Tuple[Dict[str, Any], int, Dict[str, Any], int]) -> Tuple[float, Dict[str, Any]]:
    """Worker task for parallel hyperparameter search trial evaluation."""
    hyperparams, trial_seed, payload, n_continuous = args_tuple
    torch.set_num_threads(2)  # Optimal per-worker thread count for small PyTorch MLPs

    X_train = payload["X_train"]
    y_train = payload["y_train"]
    X_val = payload["X_val"]
    y_val = payload["y_val"]
    val_circuits = payload["val_circuits"]

    model = create_mlp_model(hyperparams, n_continuous=n_continuous, seed=trial_seed)
    model.fit(X_train, y_train, val_data=(X_val, y_val))
    preds_val = model.predict(X_val)
    metrics = evaluate_all_metrics(y_val, preds_val, val_circuits)
    score = metrics["spearman_within_mean"]
    return (score, hyperparams)


def search_best_hyperparams_random(
    dataset: ModelingDataset,
    config: ExperimentConfig,
    n_trials: int = 25,
    max_workers: int = 12,
) -> Dict[str, Any]:
    """Runs 25 randomized hyperparameter trials for MLP under random split in parallel."""
    X = dataset.get_feature_matrix(
        encoding=config.encoding,
        use_circuit_features=True,
        use_structural_features=config.structural_features,
    )
    y = dataset.df[config.target].to_numpy(dtype=np.float32)

    n_continuous = 14 if config.structural_features else 8
    split = resolve_random_split(dataset.df, exclude_hyp=config.exclude_hyp)
    X_train, y_train = X[split.train_indices], y[split.train_indices]
    X_val, y_val = X[split.val_indices], y[split.val_indices]
    val_circuits = dataset.df.iloc[split.val_indices]["circuit"].to_numpy()

    payload = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "val_circuits": val_circuits,
    }

    trials = generate_mlp_search_trials(n_trials=n_trials, seed=42)
    tasks = []
    for idx, p in enumerate(trials):
        trial_seed = 42 + idx * 10
        tasks.append((p, trial_seed, payload, n_continuous))

    best_score = -999.0
    best_params = trials[0]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_eval_single_trial_worker, tasks))

    for score, params in results:
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def search_best_hyperparams_loco_fold(
    dataset: ModelingDataset,
    outer_split: SplitResult,
    config: ExperimentConfig,
    n_trials: int = 25,
    max_workers: int = 12,
) -> Dict[str, Any]:
    """Runs 25 randomized hyperparameter trials for a specific LOCO outer fold in parallel."""
    X = dataset.get_feature_matrix(
        encoding=config.encoding,
        use_circuit_features=True,
        use_structural_features=config.structural_features,
    )
    y = dataset.df[config.target].to_numpy(dtype=np.float32)

    n_continuous = 14 if config.structural_features else 8
    train_circuits_sorted = sorted(outer_split.train_circuits)
    inner_val_circuits = set(train_circuits_sorted[:3])
    inner_train_circuits = set(train_circuits_sorted[3:])

    df_train_sub = dataset.df.iloc[outer_split.train_indices]
    inner_train_mask = df_train_sub["circuit"].isin(inner_train_circuits)
    inner_val_mask = df_train_sub["circuit"].isin(inner_val_circuits)

    inner_train_indices = outer_split.train_indices[inner_train_mask]
    inner_val_indices = outer_split.train_indices[inner_val_mask]

    X_train, y_train = X[inner_train_indices], y[inner_train_indices]
    X_val, y_val = X[inner_val_indices], y[inner_val_indices]
    val_circuits_arr = dataset.df.iloc[inner_val_indices]["circuit"].to_numpy()

    payload = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "val_circuits": val_circuits_arr,
    }

    trials = generate_mlp_search_trials(n_trials=n_trials, seed=42 + outer_split.fold_idx * 100)
    tasks = []
    for idx, p in enumerate(trials):
        trial_seed = 42 + outer_split.fold_idx * 100 + idx * 10
        tasks.append((p, trial_seed, payload, n_continuous))

    best_score = -999.0
    best_params = trials[0]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_eval_single_trial_worker, tasks))

    for score, params in results:
        if score > best_score:
            best_score = score
            best_params = params

    return best_params


def run_pilot_timing_checkpoint() -> float:
    """Executes a single timed MLP fit on CPU (L=10, random split) and prints wall-clock estimate."""
    logger.info("Executing Mandatory Pilot Timing Checkpoint for Phase 5 MLP Sweep on CPU...")
    dataset = load_modeling_table(seq_len=10)

    X = dataset.get_feature_matrix(encoding="all", use_circuit_features=True, use_structural_features=True)
    y = dataset.df["target_and_ratio"].to_numpy(dtype=np.float32)

    split = resolve_random_split(dataset.df, exclude_hyp=True)
    X_train, y_train = X[split.train_indices], y[split.train_indices]
    X_val, y_val = X[split.val_indices], y[split.val_indices]

    mid_params = {
        "hidden_dims": [128, 64],
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_epochs": 200,
        "patience": 20,
    }

    torch.set_num_threads(4)
    model = create_mlp_model(mid_params, n_continuous=14, seed=42)

    t0 = time.perf_counter()
    model.fit(X_train, y_train, val_data=(X_val, y_val))
    t_single = time.perf_counter() - t0

    n_random_configs = 12
    n_loco_configs = 24
    fits_per_random_config = 28  # 25 trials + 3 refits
    fits_per_loco_config = 140   # 5 folds x (25 trials + 3 refits)
    total_fits = (n_random_configs * fits_per_random_config) + (n_loco_configs * fits_per_loco_config)

    # 12 parallel process workers -> effective wall clock throughput is total_fits / 12 * (t_single * 1.35)
    effective_fits = total_fits / 12.0
    proj_total_s = effective_fits * (t_single * 1.35)
    proj_total_min = proj_total_s / 60.0
    proj_total_hrs = proj_total_s / 3600.0

    print("\n" + "=" * 90)
    print("      UPDATED OPTION C PILOT TIMING CHECKPOINT (25 TRIALS, 12 CPU WORKERS)           ")
    print("=" * 90)
    print(f"  CPU Core Count Detected     : {os.cpu_count()} cores (Intel i7-13200H)")
    print(f"  Parallel Worker Processes   : 12 concurrent processes (2 threads/worker)")
    print(f"  Single MLP Fit Time (CPU)   : {t_single:.4f} seconds (on L=10, ~30,000 training rows)")
    print(f"  Random Split Configurations : 12 configs (25 search trials + 3 refits)")
    print(f"  LOCO Protocol Configurations: 24 configs (5 folds x [25 search + 3 refits])")
    print(f"  Grand Total Models to Fit   : {total_fits} fits")
    print(f"  Projected Total Wall-Clock   : {proj_total_s:.2f} s ({proj_total_min:.2f} min / {proj_total_hrs:.2f} hrs)")
    print("=" * 90 + "\n")

    return t_single


def execute_full_mlp_sweep() -> pd.DataFrame:
    """Executes the full Phase 5 MLP experiment sweep across all 3,696 fits."""
    t_sweep_start = time.perf_counter()

    sequence_lengths = [5, 10, 15]
    split_protocols = ["random", "loco"]
    structural_toggles = [False, True]
    exclude_hyp_toggles = [True, False]
    seeds = [42, 43, 44]

    summary_rows = []

    for L in sequence_lengths:
        dataset = load_modeling_table(seq_len=L)
        for protocol in split_protocols:
            for use_struct in structural_toggles:
                loco_hyp_options = exclude_hyp_toggles if protocol == "loco" else [True]
                for excl_hyp in loco_hyp_options:
                    n_continuous = 14 if use_struct else 8
                    logger.info(
                        f"\nRunning MLP | {protocol} | L={L} | structural={use_struct} | exclude_hyp={excl_hyp}"
                    )

                    if protocol == "random":
                        base_config = ExperimentConfig(
                            model_family="mlp",
                            dataset_L=L,
                            target="target_and_ratio",
                            encoding="all",
                            structural_features=use_struct,
                            split_protocol=protocol,
                            exclude_hyp=excl_hyp,
                            seed=42,
                            model_params={},
                        )
                        best_params = search_best_hyperparams_random(dataset, base_config, n_trials=25, max_workers=12)

                        seed_spearmans = []
                        seed_mapes = []
                        isolated_train_times = []
                        param_counts = []

                        # Measure final refit across 3 seeds in an isolated pass for uncontended RQ4 timing
                        for seed in seeds:
                            eval_config = ExperimentConfig(
                                model_family="mlp",
                                dataset_L=L,
                                target="target_and_ratio",
                                encoding="all",
                                structural_features=use_struct,
                                split_protocol=protocol,
                                exclude_hyp=excl_hyp,
                                seed=seed,
                                model_params=best_params,
                            )
                            torch.set_num_threads(8)  # use full CPU pool for isolated timing
                            model = create_mlp_model(best_params, n_continuous=n_continuous, seed=seed)
                            res = run_experiment(model, dataset, eval_config)
                            metrics = res["metrics"]
                            seed_spearmans.append(metrics.get("spearman_within_mean", 0.0))
                            seed_mapes.append(metrics.get("mape", 0.0))
                            isolated_train_times.append(res.get("train_time_s", 0.0))
                            param_counts.append(res.get("param_count", 0))

                        summary_rows.append(
                            {
                                "family": "mlp",
                                "protocol": protocol,
                                "L": L,
                                "structural": use_struct,
                                "exclude_hyp": excl_hyp,
                                "spearman_mean": float(np.mean(seed_spearmans)),
                                "spearman_std": float(np.std(seed_spearmans)),
                                "mape_mean": float(np.mean(seed_mapes)),
                                "mape_std": float(np.std(seed_mapes)),
                                "uncontended_train_time_s": float(np.mean(isolated_train_times)),
                                "n_parameters": int(np.mean(param_counts)),
                                "best_params": json.dumps(best_params),
                            }
                        )

                    elif protocol == "loco":
                        splits = resolve_loco_folds(dataset.df, exclude_hyp=excl_hyp)
                        fold_best_params: Dict[int, Dict[str, Any]] = {}
                        fold_metrics_list = []
                        fold_isolated_times = []
                        fold_params_counts = []

                        for split in splits:
                            fold_idx = split.fold_idx
                            base_config = ExperimentConfig(
                                model_family="mlp",
                                dataset_L=L,
                                target="target_and_ratio",
                                encoding="all",
                                structural_features=use_struct,
                                split_protocol=protocol,
                                exclude_hyp=excl_hyp,
                                seed=42,
                                model_params={},
                            )
                            fold_params = search_best_hyperparams_loco_fold(
                                dataset, split, base_config, n_trials=25, max_workers=12
                            )
                            fold_best_params[fold_idx] = fold_params

                            fold_seed_spearmans = []
                            fold_seed_mapes = []
                            fold_seed_times = []

                            torch.set_num_threads(8)  # isolated uncontended timing pass
                            for seed in seeds:
                                eval_config = ExperimentConfig(
                                    model_family="mlp",
                                    dataset_L=L,
                                    target="target_and_ratio",
                                    encoding="all",
                                    structural_features=use_struct,
                                    split_protocol=protocol,
                                    exclude_hyp=excl_hyp,
                                    seed=seed,
                                    model_params={
                                        **fold_params,
                                        "nested_cv": False,
                                        "fixed_inner_val": True,
                                        "fold": fold_idx,
                                    },
                                )
                                model = create_mlp_model(fold_params, n_continuous=n_continuous, seed=seed)
                                m_metrics, res_dict = run_single_split(model, dataset, split, eval_config)
                                fold_seed_spearmans.append(m_metrics.get("spearman_within_mean", 0.0))
                                fold_seed_mapes.append(m_metrics.get("mape", 0.0))
                                fold_seed_times.append(res_dict.get("train_time_s", 0.0))
                                fold_params_counts.append(res_dict.get("param_count", 0))

                            fold_metrics_list.append(
                                {
                                    "spearman": float(np.mean(fold_seed_spearmans)),
                                    "mape": float(np.mean(fold_seed_mapes)),
                                }
                            )
                            fold_isolated_times.append(float(np.mean(fold_seed_times)))

                        mean_spearman = float(np.mean([m["spearman"] for m in fold_metrics_list]))
                        std_spearman = float(np.std([m["spearman"] for m in fold_metrics_list]))
                        mean_mape = float(np.mean([m["mape"] for m in fold_metrics_list]))
                        std_mape = float(np.std([m["mape"] for m in fold_metrics_list]))

                        summary_rows.append(
                            {
                                "family": "mlp",
                                "protocol": protocol,
                                "L": L,
                                "structural": use_struct,
                                "exclude_hyp": excl_hyp,
                                "spearman_mean": mean_spearman,
                                "spearman_std": std_spearman,
                                "mape_mean": mean_mape,
                                "mape_std": std_mape,
                                "uncontended_train_time_s": float(np.mean(fold_isolated_times)),
                                "n_parameters": int(np.mean(fold_params_counts)),
                                "best_params": json.dumps(fold_best_params),
                            }
                        )

    t_sweep_total = time.perf_counter() - t_sweep_start
    logger.info(f"Phase 5 MLP sweep completed in {t_sweep_total:.2f} seconds ({t_sweep_total/60:.2f} min).")

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 90)
    print("                     PHASE 5 MLP EXPERIMENT SWEEP SUMMARY                     ")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 MLP Experiment Runner")
    parser.add_argument(
        "--run-full-sweep",
        action="store_true",
        default=False,
        help="Execute full Phase 5 MLP experiment sweep.",
    )
    args = parser.parse_args()

    t_single = run_pilot_timing_checkpoint()

    if args.run_full_sweep:
        execute_full_mlp_sweep()
