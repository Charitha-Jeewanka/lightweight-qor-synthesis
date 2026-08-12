"""Mathematical and Code Audit of MAPE implementation."""

import numpy as np
import pandas as pd
from src.data.loaders import load_modeling_table
from src.eval.metrics import evaluate_all_metrics
from src.models.mlp import MLPQoRModel
from src.data.splits import resolve_loco_folds

def audit_mape():
    print("=" * 80)
    print("MATHEMATICAL & CODE AUDIT OF MAPE IMPLEMENTATION")
    print("=" * 80)

    # 1. Load div circuit rows from L=10 modeling table
    ds = load_modeling_table(seq_len=10)
    df_div = ds.df[ds.df["circuit"] == "div"].head(5)
    
    init_and = df_div["init_and"].values
    final_and = df_div["final_and"].values
    target_ratio = df_div["target_and_ratio"].values

    # Mock predictions: ratio pred ~ 0.6820
    pred_ratio = np.full_like(target_ratio, 0.6820)
    pred_raw = pred_ratio * init_and

    # Mathematical identity check
    mape_ratio = np.mean(np.abs((target_ratio - pred_ratio) / target_ratio)) * 100.0
    mape_raw = np.mean(np.abs((final_and - pred_raw) / final_and)) * 100.0

    print("5 Sample Rows from 'div' Circuit:")
    for i in range(5):
        print(f"  Row {i+1}: init_and={init_and[i]}, final_and={final_and[i]}, true_ratio={target_ratio[i]:.4f}, pred_ratio={pred_ratio[i]:.4f}")

    print(f"\nMAPE computed on ratio scale    : {mape_ratio:.4f}%")
    print(f"MAPE computed on raw count scale: {mape_raw:.4f}%")
    print(f"Mathematical Identity Confirmed: {np.isclose(mape_ratio, mape_raw)}")

    # 2. Check MAE vs MAPE on div rows
    mae_ratio = np.mean(np.abs(target_ratio - pred_ratio))
    print(f"\nMAE on ratio scale              : {mae_ratio:.4f}")
    print(f"MAE * 100                        : {mae_ratio * 100.0:.2f}%")

    # 3. Check actual metrics on L=10 Fold 0 using evaluate_all_metrics
    splits = resolve_loco_folds(ds.df, exclude_hyp=True)
    fold0 = splits[0] # test circuits: voter, div, max, i2c
    
    # Run Fold 0 evaluation
    best_params = {
        "hidden_dims": [256, 128],
        "dropout": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_epochs": 200,
        "patience": 20,
    }
    
    X = ds.get_feature_matrix(encoding="all", use_circuit_features=True, use_structural_features=True)
    y = ds.df["target_and_ratio"].to_numpy(dtype=np.float32)
    
    X_tr, y_tr = X[fold0.train_indices], y[fold0.train_indices]
    X_te, y_te = X[fold0.test_indices], y[fold0.test_indices]
    test_circs = ds.df.iloc[fold0.test_indices]["circuit"].to_numpy()
    
    model = MLPQoRModel(**best_params, n_continuous=14, seed=42)
    model.fit(X_tr, y_tr, val_data=(X[fold0.val_indices], y[fold0.val_indices]))
    preds_te = model.predict(X_te)
    
    metrics = evaluate_all_metrics(y_te, preds_te, test_circs)
    
    # Calculate MAE * 100 on fold 0 test set
    mae_test = np.mean(np.abs(y_te - preds_te))
    mape_test = np.mean(np.abs((y_te - preds_te) / y_te)) * 100.0
    
    print("\n--- Fold 0 Test Metrics Breakdown ---")
    print(f"evaluate_all_metrics['mape']   : {metrics['mape']:.2f}%")
    print(f"Direct formula MAPE            : {mape_test:.2f}%")
    print(f"Direct formula MAE             : {mae_test:.4f}")
    print(f"Direct formula MAE * 100       : {mae_test * 100.0:.2f}%")

if __name__ == "__main__":
    audit_mape()
