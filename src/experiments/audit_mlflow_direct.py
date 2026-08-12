"""Fast MLflow auditor for Phase 4 GBT and Phase 5 MLP runs."""

from pathlib import Path
import mlflow

MLRUNS_DIR = Path(r"D:\ML Projects\EDA\gnn_eda\mlruns")

def audit_direct():
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    client = mlflow.tracking.MlflowClient()
    
    print("=" * 80)
    print("ITEM 2: PHASE 4 GBT RUNS IN MLFLOW (L=10, structural=True, exclude_hyp=True)")
    print("=" * 80)
    
    # Search runs in qor-rq1-random-split (422311628142222069) & qor-smoke-test (633556858958344200)
    runs_gbt = client.search_runs(
        experiment_ids=['422311628142222069', '633556858958344200'],
        filter_string="params.dataset_L = '10' and params.structural_features = 'True'"
    )
    
    for r in runs_gbt:
        family = r.data.params.get('model_family')
        depth = r.data.params.get('config_max_depth')
        spearman = r.data.metrics.get('spearman_within_mean')
        mape = r.data.metrics.get('mape')
        if family in ['xgboost', 'lightgbm']:
            print(f"Family: {family:<10} | Run ID: {r.info.run_id} | Depth: {str(depth):<4} | Spearman: {spearman} | MAPE: {mape}%")

    print("\n" + "=" * 80)
    print("ITEM 1: PHASE 5 MLP LOCO RUNS IN MLFLOW (L=10, structural=True)")
    print("=" * 80)
    
    runs_mlp_loco = client.search_runs(
        experiment_ids=['731072134992392198', '633556858958344200'],
        filter_string="params.model_family = 'mlp' and params.split_protocol = 'loco'"
    )
    
    for r in runs_mlp_loco:
        patience = r.data.params.get('config_patience', r.data.params.get('patience'))
        spearman = r.data.metrics.get('spearman_within_mean')
        mape = r.data.metrics.get('mape')
        fold = r.data.params.get('fold', 'all')
        print(f"Run ID: {r.info.run_id} | Fold: {fold} | Patience: {patience} | Spearman: {spearman} | MAPE: {mape}%")

if __name__ == "__main__":
    audit_direct()
