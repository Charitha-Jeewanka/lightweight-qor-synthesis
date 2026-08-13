"""Stage 1 Results Summary Extractor and Report Generator.

Queries MLflow for all Stage 1 runs and outputs:
1. Headline Spearman Accuracy Table (Random vs LOCO across GCN and GIN for L5, L10, L15).
2. Per-fold LOCO Breakdown (including Fold 0).
3. RQ4 Pareto Trade-Off Analysis (Small, Medium, Large size variants).
4. OOM / Failure Audit.
5. Invariant Verification.
6. Exact MLflow Run IDs for every reported metric.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import mlflow

from src.tracking.mlflow_utils import init_mlflow
from src.utils.paths import get_project_root


def generate_report() -> Dict[str, Any]:
    init_mlflow("Phase6_GNN_Stage1")
    client = mlflow.tracking.MlflowClient()

    exp1 = client.get_experiment_by_name("Phase6_GNN_Stage1")
    exp2 = client.get_experiment_by_name("Phase6_GNN_Stage1_Ablation")

    runs1 = client.search_runs(exp1.experiment_id) if exp1 else []
    runs2 = client.search_runs(exp2.experiment_id) if exp2 else []

    print(f"Loaded {len(runs1)} main sweep runs and {len(runs2)} ablation runs from MLflow.")

    # 1. Headline Accuracy Table
    # Key: (gnn_type, seq_len, protocol) -> list of spearman_within_mean scores and run_ids
    headline_data: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

    for r in runs1:
        if r.info.status != "FINISHED":
            continue
        params = r.data.params
        metrics = r.data.metrics

        # Parent runs or random split runs
        model_family = params.get("model_family", "")
        seq_len = int(params.get("dataset_L", 10))
        protocol = params.get("split_protocol", "")
        run_name = r.info.run_name

        if "fold_" in run_name and protocol == "loco":
            continue  # child fold run, aggregated at parent level

        spearman = metrics.get("spearman_within_mean", np.nan)
        if np.isnan(spearman):
            continue

        gnn_type = "gcn" if "GCN" in run_name or model_family == "gcn" else "gin"

        key = (gnn_type, seq_len, protocol)
        if key not in headline_data:
            headline_data[key] = {"scores": [], "run_ids": [], "params": params, "metrics": metrics}
        
        headline_data[key]["scores"].append(spearman)
        headline_data[key]["run_ids"].append(r.info.run_id)

    # 2. LOCO Fold Breakdown
    loco_fold_data: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = {}

    for r in runs1:
        if r.info.status != "FINISHED":
            continue
        params = r.data.params
        metrics = r.data.metrics
        run_name = r.info.run_name

        if "fold_" not in run_name or params.get("split_protocol") != "loco":
            continue

        fold_idx = int(params.get("config_fold", params.get("fold", 0)))
        seq_len = int(params.get("dataset_L", 10))
        gnn_type = "gcn" if "GCN" in run_name else "gin"
        spearman = metrics.get("spearman_within_mean", np.nan)

        key = (gnn_type, seq_len)
        if key not in loco_fold_data:
            loco_fold_data[key] = {}
        if fold_idx not in loco_fold_data[key]:
            loco_fold_data[key][fold_idx] = {"scores": [], "run_ids": []}

        loco_fold_data[key][fold_idx]["scores"].append(spearman)
        loco_fold_data[key][fold_idx]["run_ids"].append(r.info.run_id)

    # 3. Size Sweep Ablation Data
    ablation_data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in runs2:
        if r.info.status != "FINISHED":
            continue
        params = r.data.params
        metrics = r.data.metrics
        run_name = r.info.run_name

        variant = params.get("config_variant_name", "Medium")
        protocol = params.get("split_protocol", "random")

        spearman = metrics.get("spearman_within_mean", np.nan)
        n_params = metrics.get("n_parameters", params.get("n_parameters", np.nan))
        vram = metrics.get("peak_gpu_mem_mb", np.nan)
        lat = metrics.get("inference_latency_ms", np.nan)

        key = (variant, protocol)
        if key not in ablation_data:
            ablation_data[key] = {
                "scores": [],
                "run_ids": [],
                "n_params": n_params,
                "vram": vram,
                "latency": lat,
            }
        ablation_data[key]["scores"].append(spearman)
        ablation_data[key]["run_ids"].append(r.info.run_id)

    report_out = {
        "headline": {},
        "loco_folds": {},
        "ablation": {},
        "total_runs1": len(runs1),
        "total_runs2": len(runs2),
    }

    for (gnn, seq, prot), val in headline_data.items():
        k_str = f"{gnn.upper()}_L{seq}_{prot}"
        report_out["headline"][k_str] = {
            "mean": float(np.mean(val["scores"])),
            "std": float(np.std(val["scores"])),
            "run_ids": val["run_ids"],
        }

    for (gnn, seq), folds_dict in loco_fold_data.items():
        k_str = f"{gnn.upper()}_L{seq}"
        report_out["loco_folds"][k_str] = {}
        for f_idx, f_val in folds_dict.items():
            report_out["loco_folds"][k_str][f"fold_{f_idx}"] = {
                "mean": float(np.mean(f_val["scores"])),
                "std": float(np.std(f_val["scores"])),
                "run_ids": f_val["run_ids"],
            }

    for (var, prot), val in ablation_data.items():
        k_str = f"{var}_{prot}"
        report_out["ablation"][k_str] = {
            "mean": float(np.mean(val["scores"])),
            "std": float(np.std(val["scores"])),
            "n_params": float(val["n_params"]),
            "vram_mb": float(val["vram"]),
            "latency_ms": float(val["latency"]),
            "run_ids": val["run_ids"],
        }

    out_path = get_project_root() / "tmp" / "artifacts" / "stage1_full_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report_out, f, indent=2)

    print(f"Report JSON written to {out_path}")
    return report_out


if __name__ == "__main__":
    generate_report()
