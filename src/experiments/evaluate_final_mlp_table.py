"""Compiles and prints the single authoritative Phase 5 final results table from MLflow runs."""

from pathlib import Path
import numpy as np
import pandas as pd
import mlflow

MLRUNS_DIR = Path(r"D:\ML Projects\EDA\gnn_eda\mlruns")

def print_final_table():
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    
    # Search runs across both experiments
    runs_rq1 = mlflow.search_runs(experiment_ids=['422311628142222069'])
    runs_rq2 = mlflow.search_runs(experiment_ids=['731072134992392198'])
    
    df = pd.concat([runs_rq1, runs_rq2], ignore_index=True)
    mlp_df = df[df['params.model_family'] == 'mlp']
    
    print(f"Total MLflow MLP runs logged: {len(mlp_df)}")
    
    if mlp_df.empty:
        print("No MLP runs found in MLflow.")
        return

    # Filter out columns of interest
    cols = [
        'params.split_protocol', 'params.dataset_L', 'params.structural_features',
        'params.exclude_hyp', 'metrics.spearman_within_mean', 'metrics.mape',
        'metrics.train_time_s', 'metrics.n_parameters'
    ]
    
    # Group by key configuration parameters
    grouped = mlp_df.groupby([
        'params.split_protocol', 'params.dataset_L', 'params.structural_features', 'params.exclude_hyp'
    ])
    
    summary_rows = []
    for key, group in grouped:
        protocol, L, struct, excl_hyp = key
        rhos = group['metrics.spearman_within_mean'].dropna().values
        mapes = group['metrics.mape'].dropna().values
        times = group['metrics.train_time_s'].dropna().values
        params = group['metrics.n_parameters'].dropna().values
        
        if len(rhos) > 0:
            summary_rows.append({
                'protocol': protocol,
                'L': int(L),
                'structural': struct == 'True',
                'exclude_hyp': excl_hyp == 'True',
                'spearman_mean': float(np.mean(rhos)),
                'spearman_std': float(np.std(rhos)),
                'mape_mean': float(np.mean(mapes)),
                'mape_std': float(np.std(mapes)),
                'uncontended_train_time_s': float(np.mean(times)) if len(times) > 0 else 0.0,
                'n_parameters': int(np.mean(params)) if len(params) > 0 else 0,
                'run_count': len(rhos)
            })
            
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by=['protocol', 'L', 'structural', 'exclude_hyp'])
    
    print("\n" + "=" * 95)
    print("           SINGLE AUTHORITATIVE PHASE 5 MLP RESULTS TABLE (PATIENCE=20, MAX_EPOCHS=200)       ")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print("=" * 95 + "\n")

if __name__ == "__main__":
    print_final_table()
