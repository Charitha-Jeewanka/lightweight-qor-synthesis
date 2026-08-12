"""Discrepancy Resolution & Precise Patience Audit Script for Phase 5 MLP."""

import json
import numpy as np

from src.data.loaders import load_modeling_table
from src.data.splits import resolve_random_split
from src.eval.metrics import evaluate_all_metrics
from src.models.mlp import MLPQoRModel


def resolve_discrepancy():
    print("=" * 80)
    print("DISCREPANCY RESOLUTION & PRECISE PATIENCE AUDIT")
    print("=" * 80)

    ds = load_modeling_table(seq_len=10)
    split = resolve_random_split(ds.df, exclude_hyp=True)

    # 1. Structural=False Exact Best Hyperparameters
    best_params_struct_false = {
        "hidden_dims": [128, 64, 32],
        "dropout": 0.3,
        "learning_rate": 0.0001803961503006787,
        "weight_decay": 2.2446974525094107e-05,
        "batch_size": 256,
        "max_epochs": 100,
        "patience": 8,
    }

    X_false = ds.get_feature_matrix(
        encoding="all", use_circuit_features=True, use_structural_features=False
    )
    y = ds.df["target_and_ratio"].to_numpy(dtype=np.float32)

    X_tr, y_tr = X_false[split.train_indices], y[split.train_indices]
    X_val, y_val = X_false[split.val_indices], y[split.val_indices]
    X_te, y_te = X_false[split.test_indices], y[split.test_indices]
    test_circuits = ds.df.iloc[split.test_indices]["circuit"].to_numpy()

    spearmans_false = []
    mapes_false = []
    param_counts_false = []

    for seed in [42, 43, 44]:
        model = MLPQoRModel(
            hidden_dims=best_params_struct_false["hidden_dims"],
            dropout=best_params_struct_false["dropout"],
            learning_rate=best_params_struct_false["learning_rate"],
            weight_decay=best_params_struct_false["weight_decay"],
            batch_size=best_params_struct_false["batch_size"],
            max_epochs=100,
            patience=8,
            n_continuous=8,
            seed=seed,
        )
        model.fit(X_tr, y_tr, val_data=(X_val, y_val))
        preds = model.predict(X_te)
        metrics = evaluate_all_metrics(y_te, preds, test_circuits)
        spearmans_false.append(metrics["spearman_within_mean"])
        mapes_false.append(metrics["mape"])
        param_counts_false.append(model.param_count)

    mean_false = np.mean(spearmans_false)
    std_false = np.std(spearmans_false)
    print("\n--- Structural=False (Exact Best Hyperparams [128, 64, 32], lr=1.8e-4, bs=256) ---")
    print(f"3-Seed Spearman: {mean_false:.4f} +/- {std_false:.4f} (Seeds: {spearmans_false})")
    print(f"3-Seed MAPE: {np.mean(mapes_false):.4f}% +/- {np.std(mapes_false):.4f}%")
    print(f"Parameter Count: {param_counts_false[0]} parameters")

    # 2. Structural=True Exact Best Hyperparameters
    best_params_struct_true = {
        "hidden_dims": [256, 128, 64],
        "dropout": 0.1,
        "learning_rate": 0.0005507054976993257,
        "weight_decay": 2.562521004023822e-05,
        "batch_size": 128,
        "max_epochs": 100,
        "patience": 8,
    }

    X_true = ds.get_feature_matrix(
        encoding="all", use_circuit_features=True, use_structural_features=True
    )
    X_tr_t, X_val_t, X_te_t = (
        X_true[split.train_indices],
        X_true[split.val_indices],
        X_true[split.test_indices],
    )

    spearmans_true = []
    mapes_true = []
    param_counts_true = []

    for seed in [42, 43, 44]:
        model = MLPQoRModel(
            hidden_dims=best_params_struct_true["hidden_dims"],
            dropout=best_params_struct_true["dropout"],
            learning_rate=best_params_struct_true["learning_rate"],
            weight_decay=best_params_struct_true["weight_decay"],
            batch_size=best_params_struct_true["batch_size"],
            max_epochs=100,
            patience=8,
            n_continuous=14,
            seed=seed,
        )
        model.fit(X_tr_t, y_tr, val_data=(X_val_t, y_val))
        preds = model.predict(X_te_t)
        metrics = evaluate_all_metrics(y_te, preds, test_circuits)
        spearmans_true.append(metrics["spearman_within_mean"])
        mapes_true.append(metrics["mape"])
        param_counts_true.append(model.param_count)

    mean_true = np.mean(spearmans_true)
    std_true = np.std(spearmans_true)
    print("\n--- Structural=True (Exact Best Hyperparams [256, 128, 64], lr=5.5e-4, bs=128) ---")
    print(f"3-Seed Spearman: {mean_true:.4f} +/- {std_true:.4f} (Seeds: {spearmans_true})")
    print(f"3-Seed MAPE: {np.mean(mapes_true):.4f}% +/- {np.std(mapes_true):.4f}%")
    print(f"Parameter Count: {param_counts_true[0]} parameters")

    # 3. Precise Patience Audit on Exact Best Hyperparams
    print("\n--- Precise Patience Convergence Audit (Exact Best Hyperparams) ---")
    for pat in [8, 15, 25, 40]:
        m = MLPQoRModel(
            hidden_dims=best_params_struct_false["hidden_dims"],
            dropout=best_params_struct_false["dropout"],
            learning_rate=best_params_struct_false["learning_rate"],
            weight_decay=best_params_struct_false["weight_decay"],
            batch_size=best_params_struct_false["batch_size"],
            max_epochs=200,
            patience=pat,
            n_continuous=8,
            seed=42,
        )
        m.fit(X_tr, y_tr, val_data=(X_val, y_val))
        p = m.predict(X_te)
        met = evaluate_all_metrics(y_te, p, test_circuits)
        print(
            f"Structural=False | Patience={pat:2d} | Test Spearman = {met['spearman_within_mean']:.4f} | Train Time = {m.train_time_s:.2f}s"
        )


if __name__ == "__main__":
    resolve_discrepancy()
