"""Traceable MLflow Audit Script to resolve all 3 numerical items."""

from pathlib import Path
import pandas as pd
import mlflow

MLRUNS_DIR = Path(r"D:\ML Projects\EDA\gnn_eda\mlruns")

def audit_mlflow():
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    
    # 1. Inspect Phase 4 GBT Runs directly from MLflow
    gbt_runs = mlflow.search_runs(experiment_ids=['422311628142222069', '633556858958344200'])
    
    print("=" * 90)
    print("ITEM 2 AUDIT: PHASE 4 GBT RUNS IN MLFLOW (L=10, structural=True, exclude_hyp=True)")
    print("=" * 90)
    
    xgb_l10_struct = gbt_runs[
        (gbt_runs['params.model_family'] == 'xgboost') & 
        (gbt_runs['params.dataset_L'] == '10') & 
        (gbt_runs['params.structural_features'] == 'True')
    ]
    lgb_l10_struct = gbt_runs[
        (gbt_runs['params.model_family'] == 'lightgbm') & 
        (gbt_runs['params.dataset_L'] == '10') & 
        (gbt_runs['params.structural_features'] == 'True')
    ]
    
    print("XGBoost L=10 Structural=True Runs:")
    for idx, row in xgb_l10_struct.iterrows():
        print(f"  Run ID: {row['run_id']} | Exp ID: {row['experiment_id']} | Depth={row.get('params.config_max_depth')} | Spearman={row.get('metrics.spearman_within_mean')} | MAPE={row.get('metrics.mape')}")
        
    print("\nLightGBM L=10 Structural=True Runs:")
    for idx, row in lgb_l10_struct.iterrows():
        print(f"  Run ID: {row['run_id']} | Exp ID: {row['experiment_id']} | Depth={row.get('params.config_max_depth')} | Spearman={row.get('metrics.spearman_within_mean')} | MAPE={row.get('metrics.mape')}")

    # 2. Inspect Phase 5 MLP LOCO Runs directly from MLflow
    mlp_loco_runs = gbt_runs[
        (gbt_runs['params.model_family'] == 'mlp') & 
        (gbt_runs['params.split_protocol'] == 'loco') & 
        (gbt_runs['params.dataset_L'] == '10')
    ]
    
    print("\n" + "=" * 90)
    print("ITEM 1 AUDIT: PHASE 5 MLP LOCO RUNS IN MLFLOW (L=10, structural=True)")
    print("=" * 90)
    print(f"Total MLflow MLP LOCO Runs logged: {len(mlp_loco_runs)}")
    for idx, row in mlp_loco_runs.iterrows():
        run_id = row['run_id']
        patience = row.get('params.config_patience', row.get('params.patience', 'N/A'))
        struct = row.get('params.structural_features')
        spearman = row.get('metrics.spearman_within_mean')
        mape = row.get('metrics.mape')
        print(f"  Run ID: {run_id} | Struct={struct} | Patience={patience} | Spearman={spearman} | MAPE={mape}")

if __name__ == "__main__":
    audit_mlflow()
