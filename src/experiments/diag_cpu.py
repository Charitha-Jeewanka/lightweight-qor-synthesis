"""Diagnostic script for Phase 5 MLP CPU utilization & multiprocessing tuning."""

import time
import os
from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np
import psutil

from src.data.loaders import load_modeling_table
from src.data.splits import resolve_random_split
from src.models.mlp import MLPQoRModel


def worker_diag_fit(task_tuple):
    task_idx, bs, threads_per_worker, payload = task_tuple
    
    # Set PyTorch & OpenMP thread count inside child process
    torch.set_num_threads(threads_per_worker)
    eff_threads = torch.get_num_threads()
    pid = os.getpid()
    
    X_train, y_train, X_val, y_val = payload
    
    t_start = time.perf_counter()
    model = MLPQoRModel(
        hidden_dims=[128, 64],
        dropout=0.1,
        learning_rate=1e-3,
        batch_size=bs,
        max_epochs=100,
        patience=8,
        n_continuous=14,
        seed=42 + task_idx,
    )
    model.fit(X_train, y_train, val_data=(X_val, y_val))
    t_end = time.perf_counter()
    
    return {
        "task_idx": task_idx,
        "pid": pid,
        "batch_size": bs,
        "threads": eff_threads,
        "t_start": t_start,
        "t_end": t_end,
        "duration": t_end - t_start,
        "train_time_s": model.train_time_s,
    }


def run_diagnostics():
    print("=" * 80)
    print("     PHASE 5 MLP CPU UTILIZATION & MULTIPROCESSING DIAGNOSTICS     ")
    print("=" * 80)
    
    cpu_count = os.cpu_count()
    print(f"Detected CPU Logical Cores: {cpu_count} (Intel i7-13200H)")
    
    dataset = load_modeling_table(seq_len=10)
    X = dataset.get_feature_matrix(encoding="all", use_circuit_features=True, use_structural_features=True)
    y = dataset.df["target_and_ratio"].to_numpy(dtype=np.float32)
    split = resolve_random_split(dataset.df, exclude_hyp=True)
    
    X_tr, y_tr = X[split.train_indices], y[split.train_indices]
    X_val, y_val = X[split.val_indices], y[split.val_indices]
    payload = (X_tr, y_tr, X_val, y_val)
    
    # Test different worker count and thread count combinations
    combos = [
        (4, 2),  # 4 workers x 2 threads = 8 total threads
        (4, 4),  # 4 workers x 4 threads = 16 total threads
        (6, 2),  # 6 workers x 2 threads = 12 total threads
        (8, 2),  # 8 workers x 2 threads = 16 total threads
    ]
    
    batch_sizes = [128, 256, 512]
    
    for max_workers, threads_per_worker in combos:
        print(f"\n--- Testing Pool: {max_workers} Workers x {threads_per_worker} Threads/Worker ---")
        tasks = []
        for i in range(max_workers * 2): # 2 tasks per worker
            bs = batch_sizes[i % len(batch_sizes)]
            tasks.append((i, bs, threads_per_worker, payload))
            
        t0 = time.perf_counter()
        psutil_proc = psutil.Process()
        cpu_percentages = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_diag_fit, t) for t in tasks]
            
            # Sample system CPU percent while running
            while not all(f.done() for f in futures):
                cpu_percentages.append(psutil.cpu_percent(interval=0.2))
                time.sleep(0.1)
                
            results = [f.result() for f in futures]
            
        t_total = time.perf_counter() - t0
        avg_cpu_pct = np.mean(cpu_percentages) if cpu_percentages else 0.0
        max_cpu_pct = np.max(cpu_percentages) if cpu_percentages else 0.0
        
        print(f"Total Wall-Clock Time  : {t_total:.2f} s")
        print(f"Average CPU Percent    : {avg_cpu_pct:.1f}% (Peak: {max_cpu_pct:.1f}%)")
        print(f"Throughput             : {len(tasks) / t_total:.3f} fits/sec")
        
        # Verify genuine concurrency by checking start/end overlap
        print("Worker Timestamps & Concurrency Check:")
        for r in results[:max_workers]:
            st = r["t_start"] - t0
            et = r["t_end"] - t0
            print(f"  Task {r['task_idx']:2d} (PID {r['pid']}): bs={r['batch_size']:3d} | torch_threads={r['threads']} | start={st:.2f}s -> end={et:.2f}s ({r['duration']:.2f}s)")

if __name__ == "__main__":
    run_diagnostics()
