"""Quick status inspector for Phase 5 MLP sweep."""

import os
from pathlib import Path
import pandas as pd
import mlflow

MLRUNS_DIR = Path(r"D:\ML Projects\EDA\gnn_eda\mlruns")

def inspect_progress():
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    
    # Search runs across both experiments
    runs_rq1 = mlflow.search_runs(experiment_ids=['422311628142222069'])
    runs_rq2 = mlflow.search_runs(experiment_ids=['731072134992392198'])
    
    df = pd.concat([runs_rq1, runs_rq2], ignore_index=True)
    mlp_df = df[df['params.model_family'] == 'mlp']
    
    print(f"Total MLflow MLP runs logged so far: {len(mlp_df)}")
    if not mlp_df.empty and 'start_time' in mlp_df.columns:
        mlp_df_sorted = mlp_df.sort_values(by='start_time', ascending=False)
        print("\nTop 5 Most Recent MLP Runs:")
        for idx, row in mlp_df_sorted.head(5).iterrows():
            run_id = row.get('run_id')
            L = row.get('params.dataset_L')
            proto = row.get('params.split_protocol')
            struct = row.get('params.structural_features')
            spearman = row.get('metrics.spearman_within_mean')
            mape = row.get('metrics.mape')
            print(f"  - Run {run_id[:8]} | Protocol={proto} | L={L} | Struct={struct} | Spearman={spearman} | MAPE={mape}")

if __name__ == "__main__":
    inspect_progress()
