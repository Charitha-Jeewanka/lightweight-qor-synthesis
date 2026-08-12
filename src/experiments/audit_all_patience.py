"""Audit script to test patience=8 vs patience=20 across all 36 Phase 5 configuration blocks."""

import json
import numpy as np
import pandas as pd

from src.config.schema import ExperimentConfig
from src.data.loaders import load_modeling_table
from src.data.splits import resolve_loco_folds, resolve_random_split
from src.eval.metrics import evaluate_all_metrics
from src.models.mlp import MLPQoRModel
from src.experiments.run_mlp import generate_mlp_search_trials, create_mlp_model


def audit_all_configs_patience():
    print("=" * 90)
    print("      AUDITING PATIENCE=8 VS PATIENCE=20 ACROSS ALL PHASE 5 CONFIGURATIONS      ")
    print("=" * 90)

    sequence_lengths = [5, 10, 15]
    split_protocols = ["random", "loco"]
    structural_toggles = [False, True]
    exclude_hyp_toggles = [True, False]

    results = []

    for L in sequence_lengths:
        dataset = load_modeling_table(seq_len=L)
        for protocol in split_protocols:
            for use_struct in structural_toggles:
                loco_hyp_options = exclude_hyp_toggles if protocol == "loco" else [True]
                for excl_hyp in loco_hyp_options:
                    n_continuous = 14 if use_struct else 8
                    
                    if protocol == "random":
                        split = resolve_random_split(dataset.df, exclude_hyp=excl_hyp)
                        X = dataset.get_feature_matrix(encoding="all", use_circuit_features=True, use_structural_features=use_struct)
                        y = dataset.df["target_and_ratio"].to_numpy(dtype=np.float32)
                        
                        X_tr, y_tr = X[split.train_indices], y[split.train_indices]
                        X_val, y_val = X[split.val_indices], y[split.val_indices]
                        X_te, y_te = X[split.test_indices], y[split.test_indices]
                        val_circs = dataset.df.iloc[split.val_indices]["circuit"].to_numpy()
                        test_circs = dataset.df.iloc[split.test_indices]["circuit"].to_numpy()
                        
                        # Test best params found with patience=8 vs patience=20
                        trials = generate_mlp_search_trials(n_trials=25, seed=42)
                        
                        scores_pat8 = []
                        for t in trials:
                            m8 = create_mlp_model({**t, "patience": 8, "max_epochs": 200}, n_continuous=n_continuous, seed=42)
                            m8.fit(X_tr, y_tr, val_data=(X_val, y_val))
                            sc = evaluate_all_metrics(y_val, m8.predict(X_val), val_circs)["spearman_within_mean"]
                            scores_pat8.append(sc)
                            
                        best_idx8 = int(np.argmax(scores_pat8))
                        best_params8 = trials[best_idx8]
                        
                        # Refit test evaluation with patience=8
                        m8_refit = create_mlp_model({**best_params8, "patience": 8, "max_epochs": 200}, n_continuous=n_continuous, seed=42)
                        m8_refit.fit(X_tr, y_tr, val_data=(X_val, y_val))
                        test_rho8 = evaluate_all_metrics(y_te, m8_refit.predict(X_te), test_circs)["spearman_within_mean"]
                        
                        # Refit test evaluation with patience=20
                        m20_refit = create_mlp_model({**best_params8, "patience": 20, "max_epochs": 200}, n_continuous=n_continuous, seed=42)
                        m20_refit.fit(X_tr, y_tr, val_data=(X_val, y_val))
                        test_rho20 = evaluate_all_metrics(y_te, m20_refit.predict(X_te), test_circs)["spearman_within_mean"]
                        
                        diff = test_rho20 - test_rho8
                        results.append({
                            "protocol": protocol,
                            "L": L,
                            "structural": use_struct,
                            "exclude_hyp": excl_hyp,
                            "test_rho_pat8": test_rho8,
                            "test_rho_pat20": test_rho20,
                            "diff": diff,
                            "affected": abs(diff) > 0.005
                        })
                        print(f"[{protocol:<6} L={L:2d} struct={str(use_struct):<5} excl_hyp={str(excl_hyp):<5}] Pat8: {test_rho8:.4f} | Pat20: {test_rho20:.4f} | Diff: {diff:+.4f}")

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print(f"Summary: {df_res['affected'].sum()} of {len(df_res)} random configurations affected by patience parameter!")
    print("=" * 90)

if __name__ == "__main__":
    audit_all_configs_patience()
