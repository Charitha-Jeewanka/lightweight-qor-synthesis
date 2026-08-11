"""Profiling utilities for runtime, memory, parameters, and inference latency.

Instruments RQ4 efficiency metrics per GEMINI.md §6.3, §9, & §11.
"""

import time
import psutil
import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ProfileResult:
    """Container holding profiled execution resource metrics."""

    wall_clock_s: float = 0.0
    peak_gpu_mem_mb: float = 0.0
    peak_cpu_mem_mb: float = 0.0


class Timer:
    """Context manager for measuring wall-clock execution time."""

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        self.elapsed_s: float = 0.0
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed_s = time.perf_counter() - self.start_time


class PeakMemoryTracker:
    """Context manager tracking peak GPU and peak CPU RSS memory usage."""

    def __enter__(self) -> "PeakMemoryTracker":
        self.process = psutil.Process()
        self.initial_cpu_rss_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_cpu_rss_mb: float = self.initial_cpu_rss_mb
        self.peak_gpu_mem_mb: float = 0.0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # CPU Peak check
        current_rss_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_cpu_rss_mb = max(self.initial_cpu_rss_mb, current_rss_mb)

        # GPU Peak check
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            self.peak_gpu_mem_mb = peak_bytes / (1024 * 1024)
        else:
            self.peak_gpu_mem_mb = 0.0


def measure_inference_latency(
    model: Any,
    sample_feature: np.ndarray,
    n_samples: int = 1000,
) -> float:
    """Measures mean per-sample inference latency in milliseconds (batch_size = 1).

    Args:
        model: fitted model object implementing predict()
        sample_feature: feature array of shape (1, D) or (D,)
        n_samples: number of inference runs to average over (min 1000)

    Returns:
        float: mean per-sample inference latency in milliseconds
    """
    if sample_feature.ndim == 1:
        single_sample = sample_feature.reshape(1, -1)
    else:
        single_sample = sample_feature[:1]

    # Warmup
    for _ in range(20):
        _ = model.predict(single_sample)

    latencies_ms = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        _ = model.predict(single_sample)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies_ms))
