"""Evaluation metrics per GEMINI.md §10.

Includes MAPE, Within-Circuit Spearman (INV-4), RMSE, MAE, Regret@k, and Recovery@k.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE).

    Formula: mean(|y_true - y_pred| / |y_true|) * 100

    Raises ValueError if any y_true is near zero (< 1e-12).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)

    if np.any(np.abs(y_true_arr) < 1e-12):
        raise ValueError("Cannot compute MAPE: y_true contains zero or near-zero values.")

    return float(np.mean(np.abs((y_true_arr - y_pred_arr) / y_true_arr)) * 100.0)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_within_circuit_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    circuits: np.ndarray,
) -> Tuple[float, float, Dict[str, float], List[str]]:
    """Computes within-circuit Spearman rank correlation per GEMINI.md INV-4 & §10.

    NEVER pools across circuits. Computes rank correlation within each circuit
    independently, excludes circuits with near-zero target/prediction variance,
    and returns mean, std, per-circuit map, and list of excluded circuits.

    Returns:
        Tuple of (spearman_mean, spearman_std, spearman_per_circuit_dict, excluded_circuits_list)
    """
    unique_circuits = np.unique(circuits)
    per_circuit_rhos: Dict[str, float] = {}
    valid_rhos: List[float] = []
    excluded_circuits: List[str] = []

    for c in sorted(unique_circuits):
        mask = (circuits == c)
        c_true = y_true[mask]
        c_pred = y_pred[mask]

        if len(c_true) < 2:
            logger.info(f"Circuit '{c}' has fewer than 2 samples; excluding from Spearman.")
            excluded_circuits.append(c)
            per_circuit_rhos[c] = float("nan")
            continue

        std_true = float(np.std(c_true))
        std_pred = float(np.std(c_pred))

        if std_true < 1e-12:
            logger.info(
                f"Circuit '{c}' has near-zero true target variance (std={std_true:.2e}); "
                "excluding from Spearman aggregate."
            )
            excluded_circuits.append(c)
            per_circuit_rhos[c] = float("nan")
            continue

        if std_pred < 1e-12:
            logger.info(
                f"Circuit '{c}' has near-zero prediction variance (std={std_pred:.2e}); "
                "excluding from Spearman aggregate."
            )
            excluded_circuits.append(c)
            per_circuit_rhos[c] = float("nan")
            continue

        rho = spearmanr(c_true, c_pred).statistic
        if np.isnan(rho):
            logger.info(f"Circuit '{c}' Spearman evaluated to NaN; excluding.")
            excluded_circuits.append(c)
            per_circuit_rhos[c] = float("nan")
        else:
            rho_val = float(rho)
            per_circuit_rhos[c] = rho_val
            valid_rhos.append(rho_val)

    if not valid_rhos:
        logger.warning("All circuits were excluded from Spearman calculation!")
        return 0.0, 0.0, per_circuit_rhos, excluded_circuits

    spearman_mean = float(np.mean(valid_rhos))
    spearman_std = float(np.std(valid_rhos))

    return spearman_mean, spearman_std, per_circuit_rhos, excluded_circuits


def compute_regret_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    circuits: np.ndarray,
    k_frac: float = 0.10,
    q_baseline_dict: Optional[Dict[str, float]] = None,
) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    """Computes predictor-guided top-k Regret and Recovery per GEMINI.md §10.

    Target is assumed to be a QoR metric to be MINIMIZED (e.g. area ratio).

    Args:
        y_true: actual target values
        y_pred: predicted target values
        circuits: circuit names array
        k_frac: fraction of candidates to select (e.g., 0.10 for top 10%)
        q_baseline_dict: optional dict mapping circuit -> baseline QoR (e.g. resyn2 result)

    Returns:
        Tuple of (mean_regret, mean_recovery, per_circuit_regret, per_circuit_recovery)
    """
    unique_circuits = np.unique(circuits)
    regrets: Dict[str, float] = {}
    recoveries: Dict[str, float] = {}

    for c in unique_circuits:
        mask = (circuits == c)
        c_true = y_true[mask]
        c_pred = y_pred[mask]

        n_samples = len(c_true)
        k = max(1, int(np.ceil(k_frac * n_samples)))

        # Sort candidates ascending by predicted QoR (minimization)
        sort_order = np.argsort(c_pred)
        top_k_indices = sort_order[:k]

        q_k = float(np.min(c_true[top_k_indices]))
        q_star = float(np.min(c_true))

        regret = (q_k - q_star) / q_star if q_star != 0 else 0.0
        regrets[c] = regret

        if q_baseline_dict and c in q_baseline_dict:
            q_base = q_baseline_dict[c]
            denom = q_base - q_star
            if abs(denom) < 1e-12:
                recovery = 1.0
            else:
                recovery = (q_base - q_k) / denom
            recoveries[c] = float(recovery)
        else:
            recoveries[c] = float("nan")

    mean_regret = float(np.mean(list(regrets.values())))
    valid_recoveries = [r for r in recoveries.values() if not np.isnan(r)]
    mean_recovery = float(np.mean(valid_recoveries)) if valid_recoveries else float("nan")

    return mean_regret, mean_recovery, regrets, recoveries


def evaluate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    circuits: np.ndarray,
    q_baseline_dict: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Computes all summary metrics into a flat dictionary for MLflow logging."""
    mape = compute_mape(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)

    sp_mean, sp_std, sp_per_circuit, _ = compute_within_circuit_spearman(
        y_true, y_pred, circuits
    )

    regret_10, recovery_10, _, _ = compute_regret_at_k(
        y_true, y_pred, circuits, k_frac=0.10, q_baseline_dict=q_baseline_dict
    )

    metrics = {
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "spearman_within_mean": sp_mean,
        "spearman_within_std": sp_std,
        "regret_at_10pct": regret_10,
        "recovery_at_10pct": recovery_10 if not np.isnan(recovery_10) else -1.0,
    }

    # Add per-circuit spearman
    for c_name, rho in sp_per_circuit.items():
        metrics[f"spearman_per_circuit__{c_name}"] = rho if not np.isnan(rho) else 0.0

    return metrics
