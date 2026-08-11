"""Phase 4 Gradient-Boosted Trees (XGBoost & LightGBM) Experiment Suite & Pilot Timing Checkpoint.

Strictly follows GEMINI.md §9, §11 Phase 4, §4.2 (exclude_hyp toggle), §5 (invariants), and user directives:
- Randomized search (25 trials per outer fold in LOCO, 25 trials per random config)
- Structural feature toggle (true/false)
- exclude_hyp toggle for LOCO (true/false per §4.2)
- Per-fold inner validation setup for LOCO
- Pilot timing checkpoint before full sweep
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config.schema import ExperimentConfig
from src.data.loaders import ModelingDataset, load_modeling_table
from src.data.splits import SplitResult, resolve_loco_folds, resolve_random_split
from src.eval.harness import run_experiment, run_single_split
from src.eval.metrics import evaluate_all_metrics
from src.models.base import BaseQoRModel
from src.models.gbt import LightGBMQoRModel, XGBoostQoRModel, get_cpu_count
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


def generate_search_trials(
    n_trials: int = 25, seed: int = 42
) -> List[Dict[str, Any]]:
    """Generates 25 randomized hyperparameter trials."""
    rng = np.random.default_rng(seed)
    trials: List[Dict[str, Any]] = []

    for _ in range(n_trials):
        max_depth = int(rng.integers(3, 16))
        learning_rate = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
        n_estimators = int(rng.integers(50, 501))
        subsample = float(rng.uniform(0.5, 1.0))
        colsample_bytree = float(rng.uniform(0.5, 1.0))
        min_child_weight = float(np.exp(rng.uniform(np.log(1.0), np.log(20.0))))
        reg_lambda = float(np.exp(rng.uniform(np.log(0.001), np.log(10.0))))

        trials.append(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "min_child_weight": min_child_weight,
                "reg_lambda": reg_lambda,
            }
        )

    return trials


def create_gbt_model(model_family: str, hyperparams: Dict[str, Any], seed: int = 42) -> BaseQoRModel:
    """Factory helper creating an XGBoost or LightGBM model instance with given hyperparams."""
    if model_family == "xgboost":
        return XGBoostQoRModel(
            max_depth=hyperparams["max_depth"],
            learning_rate=hyperparams["learning_rate"],
            n_estimators=hyperparams["n_estimators"],
            subsample=hyperparams["subsample"],
            colsample_bytree=hyperparams["colsample_bytree"],
            min_child_weight=hyperparams["min_child_weight"],
            reg_lambda=hyperparams["reg_lambda"],
            seed=seed,
        )
    elif model_family == "lightgbm":
        return LightGBMQoRModel(
            max_depth=hyperparams["max_depth"],
            learning_rate=hyperparams["learning_rate"],
            n_estimators=hyperparams["n_estimators"],
            subsample=hyperparams["subsample"],
            colsample_bytree=hyperparams["colsample_bytree"],
            min_child_weight=hyperparams["min_child_weight"],
            reg_lambda=hyperparams["reg_lambda"],
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown GBT model family: {model_family}")


def search_best_hyperparams_random(
    model_family: str,
    dataset: ModelingDataset,
    config: ExperimentConfig,
    n_trials: int = 25,
) -> Dict[str, Any]:
    """Runs 25 randomized hyperparameter trials for random split protocol."""
    X = dataset.get_feature_matrix(
        encoding=config.encoding,
        use_circuit_features=True,
        use_structural_features=config.structural_features,
    )
    y = dataset.df[config.target].to_numpy(dtype=np.float32)

    split = resolve_random_split(dataset.df, exclude_hyp=config.exclude_hyp)
    X_train, y_train = X[split.train_indices], y[split.train_indices]
    X_val, y_val = X[split.val_indices], y[split.val_indices]
    val_circuits = dataset.df.iloc[split.val_indices]["circuit"].to_numpy()

    trials = generate_search_trials(n_trials=n_trials, seed=42)
    best_score = -999.0
    best_params = trials[0]

    for hyperparams in trials:
        model = create_gbt_model(model_family, hyperparams, seed=42)
        model.fit(X_train, y_train, val_data=(X_val, y_val))
        preds_val = model.predict(X_val)
        metrics = evaluate_all_metrics(y_val, preds_val, val_circuits)
        score = metrics["spearman_within_mean"]
        if score > best_score:
            best_score = score
            best_params = hyperparams

    return best_params


def search_best_hyperparams_loco_fold(
    model_family: str,
    dataset: ModelingDataset,
    outer_split: SplitResult,
    config: ExperimentConfig,
    n_trials: int = 25,
) -> Dict[str, Any]:
    """Runs 25 randomized hyperparameter trials for a specific LOCO outer fold.

    Holds out a fixed 3-circuit inner validation set from fold k's outer training circuits.
    """
    X = dataset.get_feature_matrix(
        encoding=config.encoding,
        use_circuit_features=True,
        use_structural_features=config.structural_features,
    )
    y = dataset.df[config.target].to_numpy(dtype=np.float32)

    train_circuits_sorted = sorted(outer_split.train_circuits)
    # Hold out first 3 circuits of outer train set as inner val set
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

    trials = generate_search_trials(n_trials=n_trials, seed=42)
    best_score = -999.0
    best_params = trials[0]

    for hyperparams in trials:
        model = create_gbt_model(model_family, hyperparams, seed=42)
        model.fit(X_train, y_train, val_data=(X_val, y_val))
        preds_val = model.predict(X_val)
        metrics = evaluate_all_metrics(y_val, preds_val, val_circuits_arr)
        score = metrics["spearman_within_mean"]
        if score > best_score:
            best_score = score
            best_params = hyperparams

    return best_params


def run_pilot_timing_checkpoint() -> float:
    """Executes a single timed XGBoost fit on L=10 modeling table and prints updated wall-clock estimate."""
    logger.info("Executing Mandatory Pilot Timing Checkpoint for Phase 4 GBT Sweep...")
    dataset = load_modeling_table(seq_len=10)

    X = dataset.get_feature_matrix(encoding="all", use_circuit_features=True, use_structural_features=True)
    y = dataset.df["target_and_ratio"].to_numpy(dtype=np.float32)

    split = resolve_random_split(dataset.df, exclude_hyp=True)
    X_train, y_train = X[split.train_indices], y[split.train_indices]

    # Mid-range hyperparameter fit
    model = XGBoostQoRModel(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3.0,
        reg_lambda=1.0,
        seed=42,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_single = time.perf_counter() - t0

    # Total fits calculation per GEMINI.md §4.2 (with and without hyp for LOCO):
    # Random split protocol:
    #   2 families x 3 L x 2 structural = 12 search configs (exclude_hyp=True primary)
    #   12 configs x (25 search trials + 3 refits) = 336 fits
    # LOCO protocol (run both exclude_hyp=True and exclude_hyp=False per §4.2):
    #   2 families x 3 L x 2 structural x 2 exclude_hyp toggles = 24 search configs
    #   Per fold k (5 folds): 25 search trials + 3 final seed refits = 28 fits per fold
    #   24 LOCO configs x (5 folds x 28 fits) = 24 x 140 = 3,360 fits
    # Grand Total fits = 336 (random) + 3,360 (LOCO) = 3,696 fits
    n_random_configs = 12
    n_loco_configs = 24
    fits_per_random_config = 28
    fits_per_loco_config = 140
    total_fits = (n_random_configs * fits_per_random_config) + (n_loco_configs * fits_per_loco_config)

    proj_total_s = total_fits * t_single
    proj_total_min = proj_total_s / 60.0
    proj_total_hrs = proj_total_s / 3600.0

    print("\n" + "=" * 90)
    print("      UPDATED PILOT TIMING CHECKPOINT (WITH EXCLUDE_HYP LOCO TOGGLE)      ")
    print("=" * 90)
    print(f"  CPU Core Count Detected     : {get_cpu_count()} cores (Intel i7-13200H)")
    print(f"  Single XGBoost Fit Time     : {t_single:.4f} seconds (on L=10, ~30,000 training rows)")
    print(f"  Random Split Configurations : 12 configs (2 families x 3 L x 2 structural)")
    print(f"  LOCO Protocol Configurations: 24 configs (2 families x 3 L x 2 structural x 2 exclude_hyp)")
    print(f"  Random Split Fits           : 336 fits")
    print(f"  LOCO Protocol Fits (Per-Fold): 3,360 fits (24 configs x 5 folds x [25 search + 3 refits])")
    print(f"  Grand Total Models to Fit   : {total_fits} fits")
    print(f"  Projected Total Wall-Clock   : {proj_total_s:.2f} s ({proj_total_min:.2f} min / {proj_total_hrs:.2f} hrs)")
    print("=" * 90 + "\n")

    return t_single


def execute_full_gbt_sweep() -> pd.DataFrame:
    """Executes the full Phase 4 GBT experiment sweep across all 3,696 fits."""
    t_sweep_start = time.perf_counter()

    model_families = ["xgboost", "lightgbm"]
    sequence_lengths = [5, 10, 15]
    split_protocols = ["random", "loco"]
    structural_toggles = [False, True]
    exclude_hyp_toggles = [True, False]  # per GEMINI.md §4.2
    seeds = [42, 43, 44]

    summary_rows = []
    config_idx = 0

    # Total configs: 12 random + 24 loco = 36 config blocks
    for L in sequence_lengths:
        dataset = load_modeling_table(seq_len=L)
        for protocol in split_protocols:
            for use_struct in structural_toggles:
                loco_hyp_options = exclude_hyp_toggles if protocol == "loco" else [True]
                for excl_hyp in loco_hyp_options:
                    for family in model_families:
                        config_idx += 1
                        logger.info(
                            f"\nRunning {family} | {protocol} | L={L} | structural={use_struct} | exclude_hyp={excl_hyp}"
                        )

                        if protocol == "random":
                            base_config = ExperimentConfig(
                                model_family=family,
                                dataset_L=L,
                                target="target_and_ratio",
                                encoding="all",
                                structural_features=use_struct,
                                split_protocol=protocol,
                                exclude_hyp=excl_hyp,
                                seed=42,
                                model_params={},
                            )
                            best_hyperparams = search_best_hyperparams_random(family, dataset, base_config, n_trials=25)

                            seed_spearmans = []
                            seed_mapes = []

                            for seed in seeds:
                                eval_config = ExperimentConfig(
                                    model_family=family,
                                    dataset_L=L,
                                    target="target_and_ratio",
                                    encoding="all",
                                    structural_features=use_struct,
                                    split_protocol=protocol,
                                    exclude_hyp=excl_hyp,
                                    seed=seed,
                                    model_params=best_hyperparams,
                                )
                                model = create_gbt_model(family, best_hyperparams, seed=seed)
                                res = run_experiment(model, dataset, eval_config)
                                metrics = res["metrics"]
                                seed_spearmans.append(metrics.get("spearman_within_mean", 0.0))
                                seed_mapes.append(metrics.get("mape", 0.0))

                            summary_rows.append(
                                {
                                    "family": family,
                                    "protocol": protocol,
                                    "L": L,
                                    "structural": use_struct,
                                    "exclude_hyp": excl_hyp,
                                    "spearman_mean": float(np.mean(seed_spearmans)),
                                    "spearman_std": float(np.std(seed_spearmans)),
                                    "mape_mean": float(np.mean(seed_mapes)),
                                    "mape_std": float(np.std(seed_mapes)),
                                    "best_params": json.dumps(best_hyperparams),
                                }
                            )

                        elif protocol == "loco":
                            splits = resolve_loco_folds(dataset.df, exclude_hyp=excl_hyp)
                            fold_best_params: Dict[int, Dict[str, Any]] = {}
                            fold_metrics_list = []

                            for split in splits:
                                fold_idx = split.fold_idx
                                base_config = ExperimentConfig(
                                    model_family=family,
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
                                    family, dataset, split, base_config, n_trials=25
                                )
                                fold_best_params[fold_idx] = fold_params

                                fold_seed_spearmans = []
                                fold_seed_mapes = []
                                for seed in seeds:
                                    eval_config = ExperimentConfig(
                                        model_family=family,
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
                                    model = create_gbt_model(family, fold_params, seed=seed)
                                    m_metrics, _ = run_single_split(model, dataset, split, eval_config)
                                    fold_seed_spearmans.append(m_metrics.get("spearman_within_mean", 0.0))
                                    fold_seed_mapes.append(m_metrics.get("mape", 0.0))

                                fold_metrics_list.append(
                                    {
                                        "spearman": float(np.mean(fold_seed_spearmans)),
                                        "mape": float(np.mean(fold_seed_mapes)),
                                    }
                                )

                            mean_spearman = float(np.mean([m["spearman"] for m in fold_metrics_list]))
                            std_spearman = float(np.std([m["spearman"] for m in fold_metrics_list]))
                            mean_mape = float(np.mean([m["mape"] for m in fold_metrics_list]))
                            std_mape = float(np.std([m["mape"] for m in fold_metrics_list]))

                            summary_rows.append(
                                {
                                    "family": family,
                                    "protocol": protocol,
                                    "L": L,
                                    "structural": use_struct,
                                    "exclude_hyp": excl_hyp,
                                    "spearman_mean": mean_spearman,
                                    "spearman_std": std_spearman,
                                    "mape_mean": mean_mape,
                                    "mape_std": std_mape,
                                    "best_params": json.dumps(fold_best_params),
                                }
                            )

    t_sweep_total = time.perf_counter() - t_sweep_start
    logger.info(f"Phase 4 full sweep completed in {t_sweep_total:.2f} seconds ({t_sweep_total/60:.2f} min).")

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 90)
    print("                     PHASE 4 GBT EXPERIMENT SWEEP SUMMARY                     ")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 GBT Experiment Runner")
    parser.add_argument(
        "--run-full-sweep",
        action="store_true",
        default=False,
        help="Execute full Phase 4 experiment sweep.",
    )
    args = parser.parse_args()

    t_single = run_pilot_timing_checkpoint()

    if args.run_full_sweep:
        execute_full_gbt_sweep()
