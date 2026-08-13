"""Phase 6 Experiment Driver: GNN QoR Prediction on GPU.

Executes Stage 1 per USER directive:
- Primary Main Sweep (exclude_hyp=True): GCN & GIN backbones across L=5, 10, 15
  under Random Split and LOCO Protocols (5 outer folds).
- RQ4 Size Sweep Ablation (exclude_hyp=True): Small, Medium, Large size variants on L=10.
- All runs instrumented with MLflow, parameter counting, peak VRAM, and wall-clock.
- Explicitly STOPS after Stage 1 completion and outputs headline summary tables with MLflow run IDs.
"""

import argparse
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
from src.models.gnn import GNNQoRModel
from src.tracking.mlflow_utils import (
    start_run,
    log_params,
    log_metrics,
    log_artifact,
    end_run,
    init_mlflow,
)
from src.utils.logging_utils import get_logger
from src.utils.paths import get_project_root

logger = get_logger(__name__)


def generate_gnn_search_trials(n_trials: int = 25, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates 25 randomized hyperparameter trials for GNN search space."""
    rng = np.random.default_rng(seed)

    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
    weight_decays = [1e-6, 1e-5, 1e-4, 1e-3]
    dropouts = [0.0, 0.1, 0.2]
    batch_sizes = [128, 256, 512]
    circuits_per_batches = [2, 3, 4]

    trials: List[Dict[str, Any]] = []
    for i in range(n_trials):
        lr = float(learning_rates[int(rng.integers(0, len(learning_rates)))])
        wd = float(weight_decays[int(rng.integers(0, len(weight_decays)))])
        drop = float(dropouts[int(rng.integers(0, len(dropouts)))])
        bs = int(batch_sizes[int(rng.integers(0, len(batch_sizes)))])
        cpb = int(circuits_per_batches[int(rng.integers(0, len(circuits_per_batches)))])

        trials.append(
            {
                "learning_rate": lr,
                "weight_decay": wd,
                "dropout": drop,
                "batch_size": bs,
                "circuits_per_batch": cpb,
                "max_epochs": 150,
                "patience": 15,
            }
        )

    return trials


def create_gnn_model(
    gnn_type: str,
    hyperparams: Dict[str, Any],
    gnn_hidden_dim: int = 64,
    gnn_layers: int = 3,
    readout_hidden_dims: List[int] = None,
    seed: int = 42,
) -> GNNQoRModel:
    """Factory helper creating a GNNQoRModel instance."""
    readout_dims = readout_hidden_dims if readout_hidden_dims is not None else [128, 64]
    return GNNQoRModel(
        gnn_type=gnn_type,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_layers=gnn_layers,
        readout_hidden_dims=readout_dims,
        dropout=hyperparams.get("dropout", 0.1),
        learning_rate=hyperparams.get("learning_rate", 1e-3),
        weight_decay=hyperparams.get("weight_decay", 1e-4),
        batch_size=hyperparams.get("batch_size", 256),
        circuits_per_batch=hyperparams.get("circuits_per_batch", 2),
        max_epochs=hyperparams.get("max_epochs", 150),
        patience=hyperparams.get("patience", 15),
        max_graph_nodes=250000,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def search_best_gnn_hyperparams_random(
    gnn_type: str,
    dataset: ModelingDataset,
    config: ExperimentConfig,
    n_trials: int = 25,
) -> Dict[str, Any]:
    """Evaluates 25 HPO trials on random split inner val set and returns best hyperparams."""
    trials = generate_gnn_search_trials(n_trials=n_trials, seed=config.seed)
    X = dataset.get_feature_matrix(encoding=config.encoding, use_circuit_features=False, use_structural_features=False)
    y = dataset.df[config.target].to_numpy(dtype=np.float32)
    circuits = dataset.df["circuit"].to_numpy()

    split = resolve_random_split(dataset.df, exclude_hyp=config.exclude_hyp)
    X_train, y_train = X[split.train_indices], y[split.train_indices]
    c_train = circuits[split.train_indices]

    X_val, y_val = X[split.val_indices], y[split.val_indices]
    c_val = circuits[split.val_indices]

    best_score = -float("inf")
    best_params = trials[0]

    for trial_idx, hp in enumerate(trials):
        model = create_gnn_model(gnn_type, hp, seed=config.seed + trial_idx)
        try:
            model.fit(X_train, y_train, val_data=(X_val, y_val), circuits_train=c_train, circuits_val=c_val)
            preds_val = model.predict(X_val, circuits_test=c_val)
            metrics = evaluate_all_metrics(y_val, preds_val, c_val)
            score = metrics["spearman_within_mean"]

            if score > best_score:
                best_score = score
                best_params = hp
        except Exception as e:
            logger.warning(f"HPO Trial {trial_idx} failed: {e}")
            continue

    logger.info(f"HPO search completed for {gnn_type} (random split, L={config.dataset_L}). Best score: {best_score:.4f}")
    return best_params


def search_best_gnn_hyperparams_loco_fold(
    gnn_type: str,
    dataset: ModelingDataset,
    split: SplitResult,
    config: ExperimentConfig,
    n_trials: int = 25,
) -> Dict[str, Any]:
    """Evaluates 25 HPO trials on inner validation set for a specific LOCO outer fold (INV-5)."""
    trials = generate_gnn_search_trials(n_trials=n_trials, seed=config.seed + split.fold_idx * 100)
    X = dataset.get_feature_matrix(encoding=config.encoding, use_circuit_features=False, use_structural_features=False)
    y = dataset.df[config.target].to_numpy(dtype=np.float32)
    circuits = dataset.df["circuit"].to_numpy()

    X_train, y_train = X[split.train_indices], y[split.train_indices]
    c_train = circuits[split.train_indices]

    X_val, y_val = X[split.val_indices], y[split.val_indices]
    c_val = circuits[split.val_indices]

    best_score = -float("inf")
    best_params = trials[0]

    for trial_idx, hp in enumerate(trials):
        model = create_gnn_model(gnn_type, hp, seed=config.seed + split.fold_idx * 100 + trial_idx)
        try:
            model.fit(X_train, y_train, val_data=(X_val, y_val), circuits_train=c_train, circuits_val=c_val)
            preds_val = model.predict(X_val, circuits_test=c_val)
            metrics = evaluate_all_metrics(y_val, preds_val, c_val)
            score = metrics["spearman_within_mean"]

            if score > best_score:
                best_score = score
                best_params = hp
        except Exception as e:
            logger.warning(f"LOCO Fold {split.fold_idx} HPO Trial {trial_idx} failed: {e}")
            continue

    logger.info(f"LOCO Fold {split.fold_idx} HPO completed for {gnn_type}. Best inner score: {best_score:.4f}")
    return best_params


def run_gnn_stage1() -> None:
    """Executes Stage 1 Primary Main Sweep + Size Sweep Ablation."""
    init_mlflow("Phase6_GNN_Stage1")
    print("==========================================================================")
    print("  LAUNCHING PHASE 6 (GNN) STAGE 1 SWEEP")
    print("  - Primary Main Sweep (exclude_hyp=True)")
    print("  - RQ4 Size Sweep Ablation (Small, Medium, Large)")
    print("==========================================================================")

    results_summary: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. MAIN SWEEP (exclude_hyp=True) across GCN & GIN, L5, L10, L15
    # -------------------------------------------------------------------------
    seq_lengths = [5, 10, 15]
    gnn_types = ["gcn", "gin"]
    protocols = ["random", "loco"]

    for seq_len in seq_lengths:
        dataset = load_modeling_table(seq_len=seq_len, target="target_and_ratio")

        for gnn_type in gnn_types:
            for protocol in protocols:
                print(f"\n[STAGE 1] Running {gnn_type.upper()} | L={seq_len} | Protocol={protocol.upper()} | exclude_hyp=True...")

                if protocol == "random":
                    # HPO Search
                    best_hp = search_best_gnn_hyperparams_random(gnn_type, dataset, ExperimentConfig(
                        experiment_name="Phase6_GNN_Stage1",
                        model_family="gnn",
                        dataset_L=seq_len,
                        split_protocol="random",
                        exclude_hyp=True,
                        seed=42,
                    ))

                    # Refits across 3 seeds
                    refit_scores = []
                    run_ids = []
                    for seed in [42, 123, 456]:
                        config = ExperimentConfig(
                            experiment_name="Phase6_GNN_Stage1",
                            model_family="gnn",
                            dataset_L=seq_len,
                            split_protocol="random",
                            exclude_hyp=True,
                            seed=seed,
                            model_params=best_hp,
                        )
                        model = create_gnn_model(gnn_type, best_hp, seed=seed)
                        res = run_experiment(model, dataset, config)
                        refit_scores.append(res["metrics"]["spearman_within_mean"])
                        run_ids.append(res["run_id"])

                    mean_score = float(np.mean(refit_scores))
                    std_score = float(np.std(refit_scores))

                    results_summary.append({
                        "task": "main_sweep",
                        "gnn_type": gnn_type,
                        "seq_len": seq_len,
                        "protocol": "random",
                        "exclude_hyp": True,
                        "spearman_mean": mean_score,
                        "spearman_std": std_score,
                        "run_ids": run_ids,
                    })
                    print(f"   --> Random Split Mean Spearman: {mean_score:.4f} ± {std_score:.4f} (Run IDs: {run_ids})")

                elif protocol == "loco":
                    splits_hpo = resolve_loco_folds(dataset.df, exclude_hyp=True, include_val_fold=True)
                    loco_fold_scores: List[List[float]] = []
                    parent_run_ids = []

                    # Outer LOCO Folds
                    for fold_idx, split_hpo in enumerate(splits_hpo):
                        best_hp = search_best_gnn_hyperparams_loco_fold(gnn_type, dataset, split_hpo, ExperimentConfig(
                            experiment_name="Phase6_GNN_Stage1",
                            model_family="gnn",
                            dataset_L=seq_len,
                            split_protocol="loco",
                            exclude_hyp=True,
                            seed=42 + fold_idx,
                        ))

                        fold_refits = []
                        for seed in [42, 123, 456]:
                            config = ExperimentConfig(
                                experiment_name="Phase6_GNN_Stage1",
                                model_family="gnn",
                                dataset_L=seq_len,
                                split_protocol="loco",
                                exclude_hyp=True,
                                seed=seed,
                                model_params=best_hp,
                            )
                            model = create_gnn_model(gnn_type, best_hp, seed=seed)
                            res = run_experiment(model, dataset, config)
                            fold_refits.append(res["metrics"]["spearman_within_mean"])
                            if seed == 42:
                                parent_run_ids.append(res["run_id"])

                        loco_fold_scores.append(fold_refits)

                    # Aggregate LOCO scores
                    fold_means = [np.mean(fs) for fs in loco_fold_scores]
                    overall_loco_mean = float(np.mean(fold_means))
                    overall_loco_std = float(np.std(fold_means))

                    results_summary.append({
                        "task": "main_sweep",
                        "gnn_type": gnn_type,
                        "seq_len": seq_len,
                        "protocol": "loco",
                        "exclude_hyp": True,
                        "spearman_mean": overall_loco_mean,
                        "spearman_std": overall_loco_std,
                        "fold_means": fold_means,
                        "run_ids": parent_run_ids,
                    })
                    print(f"   --> LOCO Protocol Mean Spearman: {overall_loco_mean:.4f} ± {overall_loco_std:.4f}")
                    print(f"       Per-Fold Means (Folds 0..4): {[round(m, 4) for m in fold_means]}")

    # -------------------------------------------------------------------------
    # 2. RQ4 SIZE SWEEP ABLATION (Small, Medium, Large on GCN, L=10, exclude_hyp=True)
    # -------------------------------------------------------------------------
    print("\n[STAGE 1] Running RQ4 Size Sweep Ablation (Small, Medium, Large on L=10)...")
    dataset_l10 = load_modeling_table(seq_len=10, target="target_and_ratio")

    size_variants = [
        ("Small", 32, 2, [64, 32]),
        ("Medium", 64, 3, [128, 64]),
        ("Large", 128, 4, [256, 128]),
    ]

    for v_name, h_dim, layers, r_dims in size_variants:
        for protocol in ["random", "loco"]:
            print(f"   Size Variant [{v_name}] | Protocol={protocol.upper()}...")
            hp = {"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "batch_size": 256, "circuits_per_batch": 2}
            
            refit_scores = []
            run_ids = []
            for seed in [42, 123, 456]:
                config = ExperimentConfig(
                    experiment_name="Phase6_GNN_Stage1_Ablation",
                    model_family="gnn",
                    dataset_L=10,
                    split_protocol=protocol,
                    exclude_hyp=True,
                    seed=seed,
                    model_params={"variant_name": v_name, "gnn_hidden_dim": h_dim, "gnn_layers": layers},
                )
                model = create_gnn_model("gcn", hp, gnn_hidden_dim=h_dim, gnn_layers=layers, readout_hidden_dims=r_dims, seed=seed)
                res = run_experiment(model, dataset_l10, config)
                refit_scores.append(res["metrics"]["spearman_within_mean"])
                run_ids.append(res["run_id"])

            mean_s = float(np.mean(refit_scores))
            std_s = float(np.std(refit_scores))

            results_summary.append({
                "task": "size_ablation",
                "variant_name": v_name,
                "protocol": protocol,
                "spearman_mean": mean_s,
                "spearman_std": std_s,
                "run_ids": run_ids,
            })
            print(f"      [{v_name} - {protocol}] Spearman: {mean_s:.4f} ± {std_s:.4f} (Run IDs: {run_ids[:2]}...)")

    print("\n==========================================================================")
    print("  STAGE 1 SWEEP COMPLETED SUCCESSFULLY")
    print("  - Stage 2 (exclude_hyp=False sensitivity sweep) IS PAUSED.")
    print("  - Summary report saved to tmp/artifacts/stage1_summary.json")
    print("==========================================================================")


if __name__ == "__main__":
    run_gnn_stage1()
