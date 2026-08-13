"""Pilot timing checkpoint for Phase 6 (GNN) per GEMINI.md and user prompt.

Performs:
1. CUDA & VRAM verification.
2. Timed fit on standard dataset (exclude_hyp=True) on GPU.
3. Timed fit attempt including `hyp` (exclude_hyp=False):
   a) max_graph_nodes=250000 (hyp included)
   b) max_graph_nodes=200000 (hyp guarded/skipped)
4. Total sweep time projection & size variant proposals.
"""

import time
import torch
import numpy as np

from src.data.loaders import load_modeling_table
from src.data.splits import resolve_random_split
from src.models.gnn import GNNQoRModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_pilot() -> None:
    print("==========================================================")
    print("  PHASE 6 (GNN) MANDATORY PILOT TIMING CHECKPOINT")
    print("==========================================================")

    # 1. CUDA & VRAM Gate
    cuda_ok = torch.cuda.is_available()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda_ok else 0.0
    device_name = torch.cuda.get_device_name(0) if cuda_ok else "None"

    print(f"1. CUDA Available: {cuda_ok}")
    print(f"   Device Name:    {device_name}")
    print(f"   Detected VRAM:  {vram_gb:.2f} GB")

    if not cuda_ok:
        print("CRITICAL ERROR: CUDA is unavailable. Hard gate triggered. Stopping.")
        return

    # Load L=10 modeling table
    dataset = load_modeling_table(seq_len=10, target="target_and_ratio")
    X_seq = dataset.get_feature_matrix(encoding="all", use_circuit_features=False, use_structural_features=False)
    y_all = dataset.df["target_and_ratio"].to_numpy(dtype=np.float32)
    circuits_all = dataset.df["circuit"].to_numpy()

    # 2. Timed fit on small/medium circuits (exclude_hyp=True)
    split_no_hyp = resolve_random_split(dataset.df, exclude_hyp=True)
    X_train_no_hyp = X_seq[split_no_hyp.train_indices]
    y_train_no_hyp = y_all[split_no_hyp.train_indices]
    c_train_no_hyp = circuits_all[split_no_hyp.train_indices]

    X_val_no_hyp = X_seq[split_no_hyp.val_indices]
    y_val_no_hyp = y_all[split_no_hyp.val_indices]
    c_val_no_hyp = circuits_all[split_no_hyp.val_indices]

    print("\n2. Running timed fit on small/medium circuits (exclude_hyp=True)...")

    model_no_hyp = GNNQoRModel(
        gnn_type="gcn",
        gnn_hidden_dim=64,
        gnn_layers=3,
        readout_hidden_dims=[128, 64],
        max_epochs=20,  # 20 epochs for pilot timing
        batch_size=256,
        circuits_per_batch=2,
        max_graph_nodes=250000,
        device="cuda",
    )

    t0 = time.perf_counter()
    model_no_hyp.fit(
        X_train_no_hyp,
        y_train_no_hyp,
        val_data=(X_val_no_hyp, y_val_no_hyp),
        circuits_train=c_train_no_hyp,
        circuits_val=c_val_no_hyp,
    )
    t_no_hyp = time.perf_counter() - t0
    gpu_mem_no_hyp = model_no_hyp.peak_gpu_mem_mb

    print(f"   Fit Wall-Clock (20 epochs, no hyp): {t_no_hyp:.2f} s ({t_no_hyp/20:.3f} s/epoch)")
    print(f"   Peak GPU Memory:                   {gpu_mem_no_hyp:.2f} MB")
    print(f"   Model Parameter Count:             {model_no_hyp.param_count:,} params")

    # 3a. Timed fit with hyp included (exclude_hyp=False, max_graph_nodes=250000)
    split_with_hyp = resolve_random_split(dataset.df, exclude_hyp=False)
    X_train_hyp = X_seq[split_with_hyp.train_indices]
    y_train_hyp = y_all[split_with_hyp.train_indices]
    c_train_hyp = circuits_all[split_with_hyp.train_indices]

    X_val_hyp = X_seq[split_with_hyp.val_indices]
    y_val_hyp = y_all[split_with_hyp.val_indices]
    c_val_hyp = circuits_all[split_with_hyp.val_indices]

    print("\n3a. Running timed fit WITH `hyp` included (max_graph_nodes=250000)...")
    model_with_hyp = GNNQoRModel(
        gnn_type="gcn",
        gnn_hidden_dim=64,
        gnn_layers=3,
        readout_hidden_dims=[128, 64],
        max_epochs=10,  # 10 epochs for timing
        batch_size=256,
        circuits_per_batch=2,
        max_graph_nodes=250000,
        device="cuda",
    )

    t0 = time.perf_counter()
    model_with_hyp.fit(
        X_train_hyp,
        y_train_hyp,
        val_data=(X_val_hyp, y_val_hyp),
        circuits_train=c_train_hyp,
        circuits_val=c_val_hyp,
    )
    t_with_hyp = time.perf_counter() - t0
    gpu_mem_with_hyp = model_with_hyp.peak_gpu_mem_mb

    print(f"   Fit Wall-Clock (10 epochs, with hyp): {t_with_hyp:.2f} s ({t_with_hyp/10:.3f} s/epoch)")
    print(f"   Peak GPU Memory (with hyp):          {gpu_mem_with_hyp:.2f} MB")

    # 3b. Verification of max_graph_nodes guard (exclude_hyp=False, max_graph_nodes=200000)
    print("\n3b. Testing max_graph_nodes guard (max_graph_nodes=200000)...")
    model_guarded = GNNQoRModel(
        gnn_type="gcn",
        gnn_hidden_dim=64,
        gnn_layers=3,
        readout_hidden_dims=[128, 64],
        max_epochs=2,
        batch_size=256,
        circuits_per_batch=2,
        max_graph_nodes=200000,
        device="cuda",
    )

    t0 = time.perf_counter()
    model_guarded.fit(
        X_train_hyp,
        y_train_hyp,
        val_data=(X_val_hyp, y_val_hyp),
        circuits_train=c_train_hyp,
        circuits_val=c_val_hyp,
    )
    t_guarded = time.perf_counter() - t0
    print(f"   Guarded Fit Completed cleanly in:    {t_guarded:.2f} s")
    print(f"   Skipped Circuits Logged:             {list(model_guarded.registry.skipped_circuits.keys())}")

    print("\n==========================================================")
    print("  SUMMARY OF PILOT BENCHMARK RESULTS")
    print("==========================================================")
    print(f"  - CUDA GPU: {device_name} ({vram_gb:.2f} GB)")
    print(f"  - Peak GPU Mem (without hyp): {gpu_mem_no_hyp:.2f} MB")
    print(f"  - Peak GPU Mem (with hyp):    {gpu_mem_with_hyp:.2f} MB")
    print(f"  - Time per epoch (without hyp): {t_no_hyp/20:.3f} s")
    print(f"  - Time per epoch (with hyp):    {t_with_hyp/10:.3f} s")
    print("==========================================================")


if __name__ == "__main__":
    run_pilot()
