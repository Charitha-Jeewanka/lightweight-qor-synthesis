"""Unit tests for metrics calculation with analytically known outputs."""

import pytest
import numpy as np

from src.eval.metrics import (
    compute_mape,
    compute_mae,
    compute_rmse,
    compute_within_circuit_spearman,
    compute_regret_at_k,
    evaluate_all_metrics,
)


def test_mape_known_values():
    y_true = np.array([100.0, 200.0], dtype=np.float32)
    y_pred = np.array([110.0, 190.0], dtype=np.float32)
    # (|10/100| + |-10/200|) / 2 * 100 = (0.10 + 0.05) / 2 * 100 = 7.5
    mape = compute_mape(y_true, y_pred)
    assert np.isclose(mape, 7.5)


def test_mape_zero_true_raises():
    y_true = np.array([0.0, 100.0], dtype=np.float32)
    y_pred = np.array([10.0, 90.0], dtype=np.float32)
    with pytest.raises(ValueError):
        compute_mape(y_true, y_pred)


def test_within_circuit_spearman_perfect():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    y_pred = np.array([10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    circuits = np.array(["c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2"])

    # c1: monotonically increasing -> rho = +1.0
    # c2: monotonically increasing -> rho = +1.0
    sp_mean, sp_std, sp_dict, excluded = compute_within_circuit_spearman(y_true, y_pred, circuits)
    assert np.isclose(sp_mean, 1.0)
    assert np.isclose(sp_std, 0.0)
    assert np.isclose(sp_dict["c1"], 1.0)
    assert np.isclose(sp_dict["c2"], 1.0)
    assert len(excluded) == 0


def test_within_circuit_spearman_zero_variance_exclusion():
    # c1 has constant y_true (zero variance), c2 has perfect correlation
    y_true = np.array([5.0, 5.0, 5.0, 5.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    circuits = np.array(["c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2"])

    sp_mean, sp_std, sp_dict, excluded = compute_within_circuit_spearman(y_true, y_pred, circuits)
    assert "c1" in excluded
    assert np.isnan(sp_dict["c1"])
    assert np.isclose(sp_dict["c2"], 1.0)
    assert np.isclose(sp_mean, 1.0)  # average over valid circuits only (c2)


def test_regret_at_k_known_values():
    y_true = np.array([0.8, 0.9, 0.7, 1.0], dtype=np.float32)  # q_star = 0.7
    y_pred = np.array([0.8, 0.9, 0.7, 1.0], dtype=np.float32)  # perfect ranking
    circuits = np.array(["c1", "c1", "c1", "c1"])

    mean_regret, mean_rec, regrets, _ = compute_regret_at_k(y_true, y_pred, circuits, k_frac=0.25)
    # top 25% of 4 samples = top 1 predicted -> index 2 (y_pred=0.7, y_true=0.7) -> q_k = 0.7
    # regret = (0.7 - 0.7) / 0.7 = 0.0
    assert np.isclose(regrets["c1"], 0.0)
    assert np.isclose(mean_regret, 0.0)
