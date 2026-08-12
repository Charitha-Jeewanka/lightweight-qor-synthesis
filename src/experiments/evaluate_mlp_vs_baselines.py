"""Evaluation script to compile Phase 5 MLP results, parameter counts, and per-fold LOCO breakdown."""

import json
import numpy as np
import pandas as pd

from src.data.loaders import load_modeling_table
from src.data.splits import resolve_loco_folds, resolve_random_split
from src.eval.metrics import evaluate_all_metrics
from src.models.mlp import MLPQoRModel


def run_full_mlp_analysis():
    # Read task log table or reconstruct best parameters from run_mlp summary
    # Let's evaluate MLP models with exact best hyperparams to get exact n_parameters, isolated train_time_s, and per-fold LOCO values
    
    # Load L=10 dataset for headline comparison
    ds10 = load_modeling_table(seq_len=10)
    
    # 1. Random Split L=10 Structural=True Exclude_hyp=True
    # Best params logged for Random L=10 Struct=True:
    # {"hidden_dims": [256, 128, 64], "dropout": 0.1, "learning_rate": 0.0005507, "weight_decay": 2.56e-5, "batch_size": 128}
    
    mlp_random_l10_struct = MLPQoRModel(
        hidden_dims=[256, 128, 64],
        dropout=0.1,
        learning_rate=0.0005507,
        weight_decay=2.56e-5,
        batch_size=128,
        n_continuous=14,
        seed=42
    )
    
    split_rand = resolve_random_split(ds10.df, exclude_hyp=True)
    X_rand = ds10.get_feature_matrix(encoding="all", use_circuit_features=True, use_structural_features=True)
    y_rand = ds10.df["target_and_ratio"].to_numpy(dtype=np.float32)
    
    mlp_random_l10_struct.fit(X_rand[split_rand.train_indices], y_rand[split_rand.train_indices], val_data=(X_rand[split_rand.val_indices], y_rand[split_rand.val_indices]))
    p_rand = mlp_random_l10_struct.predict(X_rand[split_rand.test_indices])
    m_rand = evaluate_all_metrics(y_rand[split_rand.test_indices], p_rand, ds10.df.iloc[split_rand.test_indices]["circuit"].to_numpy())
    
    print("=== RANDOM SPLIT L=10 STRUCTURAL=TRUE ===")
    print(f"Spearman: {m_rand['spearman_within_mean']:.4f}")
    print(f"MAPE: {m_rand['mape']:.4f}")
    print(f"Parameter Count: {mlp_random_l10_struct.param_count}")
    print(f"Single-Fit Train Time: {mlp_random_l10_struct.train_time_s:.4f} s")
    
    # 2. LOCO L=10 Fold-by-Fold Breakdown for MLP (Exclude_hyp=True, Structural=True)
    splits_loco = resolve_loco_folds(ds10.df, exclude_hyp=True)
    
    # Fold best params from sweep log:
    fold_params_l10_struct = {
        0: {"hidden_dims": [128, 64], "dropout": 0.0, "learning_rate": 0.002323, "weight_decay": 0.0001715, "batch_size": 256},
        1: {"hidden_dims": [256, 128, 64], "dropout": 0.0, "learning_rate": 0.006554, "weight_decay": 4.828e-6, "batch_size": 512},
        2: {"hidden_dims": [64], "dropout": 0.2, "learning_rate": 0.0001986, "weight_decay": 3.318e-6, "batch_size": 128},
        3: {"hidden_dims": [128], "dropout": 0.2, "learning_rate": 0.0031756, "weight_decay": 8.313e-6, "batch_size": 256},
        4: {"hidden_dims": [128, 64, 32], "dropout": 0.3, "learning_rate": 0.000252, "weight_decay": 4.76e-5, "batch_size": 128},
    }
    
    print("\n=== LOCO L=10 STRUCTURAL=TRUE (EXCLUDE_HYP=TRUE) FOLD BREAKDOWN ===")
    fold_spearmans = []
    fold_param_counts = []
    
    for split in splits_loco:
        f_idx = split.fold_idx
        params = fold_params_l10_struct[f_idx]
        
        # Inner train/val split inside train circuits
        train_circuits_sorted = sorted(split.train_circuits)
        inner_val_circuits = set(train_circuits_sorted[:3])
        inner_train_circuits = set(train_circuits_sorted[3:])
        
        df_tr_sub = ds10.df.iloc[split.train_indices]
        inner_tr_idx = split.train_indices[df_tr_sub["circuit"].isin(inner_train_circuits)]
        inner_val_idx = split.train_indices[df_tr_sub["circuit"].isin(inner_val_circuits)]
        
        m = MLPQoRModel(
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            n_continuous=14,
            seed=42
        )
        m.fit(X_rand[inner_tr_idx], y_rand[inner_tr_idx], val_data=(X_rand[inner_val_idx], y_rand[inner_val_idx]))
        
        preds_test = m.predict(X_rand[split.test_indices])
        test_circuits = ds10.df.iloc[split.test_indices]["circuit"].to_numpy()
        metrics = evaluate_all_metrics(y_rand[split.test_indices], preds_test, test_circuits)
        
        rho = metrics["spearman_within_mean"]
        fold_spearmans.append(rho)
        fold_param_counts.append(m.param_count)
        
        print(f"Fold {f_idx} (Test circuits: {split.test_circuits}): Spearman = {rho:.4f} | Param Count = {m.param_count} | Train Time = {m.train_time_s:.4f} s")
        
    print(f"Mean LOCO Spearman across 5 folds: {np.mean(fold_spearmans):.4f} +/- {np.std(fold_spearmans):.4f}")
    print(f"Average Parameter Count: {int(np.mean(fold_param_counts))}")

if __name__ == "__main__":
    run_full_mlp_analysis()
