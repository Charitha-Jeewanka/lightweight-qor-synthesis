"""Empirical timing benchmark script for Phase 6 GNN size variants & architectures.

Runs representative multi-circuit fits on GPU for Small, Medium, and Large GNN variants,
logging every benchmark run to MLflow with its exact run ID.
"""

import time
import torch
import numpy as np

from src.data.loaders import load_modeling_table
from src.data.splits import resolve_random_split
from src.models.gnn import GNNQoRModel
from src.tracking.mlflow_utils import (
    start_run,
    log_params,
    log_metrics,
    end_run,
    init_mlflow,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_benchmark() -> None:
    init_mlflow("Phase6_GNN_Pilot")

    dataset = load_modeling_table(seq_len=10, target="target_and_ratio")
    X_seq = dataset.get_feature_matrix(encoding="all", use_circuit_features=False, use_structural_features=False)
    y_all = dataset.df["target_and_ratio"].to_numpy(dtype=np.float32)
    circuits_all = dataset.df["circuit"].to_numpy()

    split = resolve_random_split(dataset.df, exclude_hyp=True)
    X_train, y_train = X_seq[split.train_indices], y_all[split.train_indices]
    c_train = circuits_all[split.train_indices]

    X_val, y_val = X_seq[split.val_indices], y_all[split.val_indices]
    c_val = circuits_all[split.val_indices]

    variants = [
        ("GCN_Small", "gcn", 32, 2, [64, 32]),
        ("GCN_Medium", "gcn", 64, 3, [128, 64]),
        ("GCN_Large", "gcn", 128, 4, [256, 128]),
        ("GIN_Medium", "gin", 64, 3, [128, 64]),
    ]

    print("==========================================================================")
    print("  PHASE 6 GNN TIMING BENCHMARK (50 Epochs on GPU)")
    print("==========================================================================")

    for v_name, gnn_type, hidden_dim, gnn_layers, readout_dims in variants:
        run = start_run(
            run_name=f"pilot_{v_name}",
            tags={"phase": "phase_6_pilot", "model": v_name},
        )
        run_id = run.info.run_id

        model = GNNQoRModel(
            gnn_type=gnn_type,
            gnn_hidden_dim=hidden_dim,
            gnn_layers=gnn_layers,
            readout_hidden_dims=readout_dims,
            max_epochs=50,
            batch_size=256,
            circuits_per_batch=2,
            max_graph_nodes=250000,
            device="cuda",
        )

        log_params({
            "variant_name": v_name,
            "gnn_type": gnn_type,
            "gnn_hidden_dim": hidden_dim,
            "gnn_layers": gnn_layers,
            "readout_hidden_dims": str(readout_dims),
            "max_epochs": 50,
            "max_graph_nodes": 250000,
        })

        t0 = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            val_data=(X_val, y_val),
            circuits_train=c_train,
            circuits_val=c_val,
        )
        elapsed_s = time.perf_counter() - t0

        metrics = {
            "elapsed_s": elapsed_s,
            "sec_per_epoch": elapsed_s / 50.0,
            "peak_gpu_mem_mb": model.peak_gpu_mem_mb,
            "n_parameters": float(model.param_count),
        }
        log_metrics(metrics)
        end_run(status="FINISHED")

        print(f"[{v_name}] MLflow Run ID: {run_id}")
        print(f"   Params: {model.param_count:,} | Elapsed: {elapsed_s:.2f} s | {elapsed_s/50.0:.3f} s/epoch | Peak GPU: {model.peak_gpu_mem_mb:.2f} MB\n")

    print("==========================================================================")


if __name__ == "__main__":
    run_benchmark()
