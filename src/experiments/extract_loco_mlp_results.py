"""Extracts and computes LOCO fold results for Phase 5 MLP."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config.schema import ExperimentConfig
from src.data.loaders import load_modeling_table
from src.data.splits import resolve_loco_folds
from src.eval.harness import run_single_split
from src.models.mlp import MLPQoRModel


def compute_loco_results():
    print("=" * 80)
    print("COMPUTING LOCO RESULTS FOR PHASE 5 MLP (PATIENCE=20, MAX_EPOCHS=200)")
    print("=" * 80)

    # Key LOCO configuration: L=10, structural=True, exclude_hyp=True
    ds10 = load_modeling_table(seq_len=10)
    splits10 = resolve_loco_folds(ds10.df, exclude_hyp=True)

    # Best params for L=10 LOCO structural=True found during search
    best_params_l10_loco = {
        "hidden_dims": [256, 128],
        "dropout": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_epochs": 200,
        "patience": 20,
    }

    fold_rhos = []
    fold_mapes = []
    fold_times = []
    fold_params = []

    for split in splits10:
        fold_seed_rhos = []
        fold_seed_mapes = []
        fold_seed_times = []
        fold_seed_params = []

        for seed in [42, 43, 44]:
            config = ExperimentConfig(
                model_family="mlp",
                dataset_L=10,
                target="target_and_ratio",
                encoding="all",
                structural_features=True,
                split_protocol="loco",
                exclude_hyp=True,
                seed=seed,
                model_params=best_params_l10_loco,
            )
            model = MLPQoRModel(
                hidden_dims=best_params_l10_loco["hidden_dims"],
                dropout=best_params_l10_loco["dropout"],
                learning_rate=best_params_l10_loco["learning_rate"],
                weight_decay=best_params_l10_loco["weight_decay"],
                batch_size=best_params_l10_loco["batch_size"],
                max_epochs=200,
                patience=20,
                n_continuous=14,
                seed=seed,
            )
            m_metrics, _ = run_single_split(model, ds10, split, config)
            fold_seed_rhos.append(m_metrics["spearman_within_mean"])
            fold_seed_mapes.append(m_metrics["mape"])
            fold_seed_times.append(m_metrics["train_time_s"])
            fold_seed_params.append(m_metrics["n_parameters"])

        fold_rhos.append(np.mean(fold_seed_rhos))
        fold_mapes.append(np.mean(fold_seed_mapes))
        fold_times.append(np.mean(fold_seed_times))
        fold_params.append(np.mean(fold_seed_params))
        print(f"LOCO L=10 Fold {split.fold_idx} ({' / '.join(split.test_circuits)}): Spearman = {fold_rhos[-1]:.4f} | MAPE = {fold_mapes[-1]:.2f}%")

    print("\n" + "=" * 80)
    print("LOCO L=10 STRUCTURAL=TRUE HEADLINE RESULTS (PATIENCE=20, MAX_EPOCHS=200)")
    print("=" * 80)
    print(f"LOCO Spearman (Mean +/- Std) : {np.mean(fold_rhos):.4f} +/- {np.std(fold_rhos):.4f}")
    print(f"LOCO MAPE (Mean +/- Std)     : {np.mean(fold_mapes):.2f}% +/- {np.std(fold_mapes):.2f}%")
    print(f"Uncontended Train Time        : {np.mean(fold_times):.2f}s")
    print(f"Parameter Count               : {int(np.mean(fold_params))} parameters")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    compute_loco_results()
