"""Utility script to move/organize GBT and MLP runs into primary MLflow experiment tabs."""

import shutil
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

MLRUNS_DIR = Path(r"D:\ML Projects\EDA\gnn_eda\mlruns")

def organize_experiments():
    mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.as_posix()}")
    client = MlflowClient()
    
    exps = {e.name: e.experiment_id for e in client.search_experiments()}
    print("Found MLflow Experiments:", exps)
    
    smoke_id = exps.get("qor-smoke-test")
    rq1_id = exps.get("qor-rq1-random-split")
    rq2_id = exps.get("qor-rq2-leave-circuits-out")
    
    if not smoke_id:
        print("No qor-smoke-test experiment found.")
        return
        
    smoke_runs = client.search_runs(experiment_ids=[smoke_id])
    print(f"Total runs in qor-smoke-test: {len(smoke_runs)}")
    
    moved_count = 0
    for run in smoke_runs:
        run_id = run.info.run_id
        proto = run.data.params.get("split_protocol", "random")
        family = run.data.params.get("model_family", "unknown")
        
        target_exp_id = rq2_id if proto == "loco" else rq1_id
        
        if family in ["xgboost", "lightgbm", "mlp"]:
            # Copy run directory in mlruns
            src_dir = MLRUNS_DIR / smoke_id / run_id
            dst_dir = MLRUNS_DIR / target_exp_id / run_id
            
            if src_dir.exists() and not dst_dir.exists():
                shutil.copytree(src_dir, dst_dir)
                # Update meta.yaml in dst_dir to reflect new experiment_id
                meta_path = dst_dir / "meta.yaml"
                if meta_path.exists():
                    text = meta_path.read_text()
                    text = text.replace(f"experiment_id: '{smoke_id}'", f"experiment_id: '{target_exp_id}'")
                    text = text.replace(f"experiment_id: {smoke_id}", f"experiment_id: {target_exp_id}")
                    meta_path.write_text(text)
                moved_count += 1

    print(f"Successfully organized {moved_count} runs into primary MLflow experiment tabs!")

if __name__ == "__main__":
    organize_experiments()
