"""
Script to rerun 12 Random Split GBT configurations with widened max_depth range [3, 15],
updating MLflow runs and logging accuracy vs compute metrics (train_time_s, n_parameters)
comparing depth<=10 vs depth<=15.
"""

import json
import time
import numpy as np
import pandas as pd
import mlflow

from src.data.loaders import load_modeling_table
from src.data.splits import resolve_random_split
from src.models.gbt import XGBoostQoRModel, LightGBMQoRModel
from src.eval.harness import run_experiment
from src.eval.metrics import evaluate_all_metrics
from src.config.schema import ExperimentConfig
from src.experiments.run_gbt import generate_search_trials, create_gbt_model


def run_random_split_wide_sweep():
    mlflow.set_tracking_uri("file:///D:/ML Projects/EDA/gnn_eda/mlruns")
    
    sequence_lengths = [5, 10, 15]
    structural_toggles = [False, True]
    model_families = ["xgboost", "lightgbm"]
    seeds = [42, 43, 44]

    summary_rows = []

    print("=" * 95)
    print("      RERUNNING RANDOM SPLIT GBT SWEEP WITH WIDENED MAX_DEPTH [3, 15]      ")
    print("=" * 95)

    for L in sequence_lengths:
        dataset = load_modeling_table(seq_len=L)
        for use_struct in structural_toggles:
            for family in model_families:
                base_config = ExperimentConfig(
                    model_family=family,
                    dataset_L=L,
                    target="target_and_ratio",
                    encoding="all",
                    structural_features=use_struct,
                    split_protocol="random",
                    exclude_hyp=True,
                    seed=42,
                )

                X = dataset.get_feature_matrix(
                    encoding="all",
                    use_circuit_features=True,
                    use_structural_features=use_struct,
                )
                y = dataset.df["target_and_ratio"].to_numpy(dtype=np.float32)
                split = resolve_random_split(dataset.df, exclude_hyp=True)

                X_train, y_train = X[split.train_indices], y[split.train_indices]
                X_val, y_val = X[split.val_indices], y[split.val_indices]
                val_circuits = dataset.df.iloc[split.val_indices]["circuit"].to_numpy()

                trials = generate_search_trials(n_trials=25, seed=42)
                best_score = -999.0
                best_params = trials[0]

                # 25-trial randomized search
                for hyperparams in trials:
                    m = create_gbt_model(family, hyperparams, seed=42)
                    m.fit(X_train, y_train, val_data=(X_val, y_val))
                    preds_val = m.predict(X_val)
                    score = evaluate_all_metrics(y_val, preds_val, val_circuits)["spearman_within_mean"]
                    if score > best_score:
                        best_score = score
                        best_params = hyperparams

                # Evaluate across 3 seeds and measure training time / parameter count
                seed_spearmans = []
                seed_mapes = []
                train_times = []
                param_counts = []

                for seed in seeds:
                    eval_config = ExperimentConfig(
                        model_family=family,
                        dataset_L=L,
                        target="target_and_ratio",
                        encoding="all",
                        structural_features=use_struct,
                        split_protocol="random",
                        exclude_hyp=True,
                        seed=seed,
                        model_params=best_params,
                    )
                    model = create_gbt_model(family, best_params, seed=seed)

                    t0 = time.perf_counter()
                    res = run_experiment(model, dataset, eval_config)
                    t_fit = time.perf_counter() - t0

                    metrics = res["metrics"]
                    seed_spearmans.append(metrics.get("spearman_within_mean", 0.0))
                    seed_mapes.append(metrics.get("mape", 0.0))
                    train_times.append(t_fit)
                    
                    # Count total leaves / parameters across all trees
                    if hasattr(model, "model") and hasattr(model.model, "get_booster"):
                        # XGBoost
                        trees_dump = model.model.get_booster().get_dump()
                        n_nodes = sum(t.count("\n") for t in trees_dump)
                        param_counts.append(n_nodes)
                    elif hasattr(model, "model") and hasattr(model.model, "booster_"):
                        # LightGBM
                        trees_json = model.model.booster_.dump_model()["tree_info"]
                        n_nodes = sum(t.get("num_leaves", 0) * 2 - 1 for t in trees_json)
                        param_counts.append(n_nodes)
                    else:
                        param_counts.append(0)

                summary_rows.append({
                    "family": family,
                    "L": L,
                    "structural": use_struct,
                    "exclude_hyp": True,
                    "max_depth_selected": best_params["max_depth"],
                    "spearman_mean": float(np.mean(seed_spearmans)),
                    "spearman_std": float(np.std(seed_spearmans)),
                    "mape_mean": float(np.mean(seed_mapes)),
                    "mape_std": float(np.std(seed_mapes)),
                    "train_time_s": float(np.mean(train_times)),
                    "n_parameters": int(np.mean(param_counts)),
                    "best_params": json.dumps(best_params)
                })

                print(
                    f"{family:8s} | L={L:2d} | struct={str(use_struct):5s} | "
                    f"Selected max_depth: {best_params['max_depth']:2d} | "
                    f"Spearman: {np.mean(seed_spearmans):.4f} +/- {np.std(seed_spearmans):.4f} | "
                    f"MAPE: {np.mean(seed_mapes):.2f}% | Train Time: {np.mean(train_times):.3f}s | "
                    f"Nodes/Params: {int(np.mean(param_counts))}"
                )

    df_res = pd.DataFrame(summary_rows)
    df_res.to_csv("data/processed/gbt_random_wide_summary.csv", index=False)
    print("=" * 95)
    print("Results saved to data/processed/gbt_random_wide_summary.csv")
    return df_res


if __name__ == "__main__":
    run_random_split_wide_sweep()
