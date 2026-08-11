"""Unit tests for Phase 2 baseline models."""

import pytest
import numpy as np

from src.models.baselines import (
    ConstantMeanBaseline,
    PerCircuitMeanBaseline,
    LinearRegressionBaseline,
    SequenceOnlyBaseline,
)


def test_constant_mean_baseline():
    X = np.ones((10, 5), dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)

    model = ConstantMeanBaseline()
    model.fit(X, y)

    assert model.name == "baseline_mean"
    assert model.param_count == 1
    assert np.isclose(model.global_mean, 5.5)

    preds = model.predict(X[:3])
    assert len(preds) == 3
    assert np.allclose(preds, 5.5)


def test_per_circuit_mean_baseline():
    X = np.ones((6, 2), dtype=np.float32)
    y = np.array([10.0, 10.0, 20.0, 20.0, 30.0, 30.0], dtype=np.float32)
    circuits_train = np.array(["c1", "c1", "c2", "c2", "c3", "c3"])

    model = PerCircuitMeanBaseline()
    model.fit(X, y, circuits_train=circuits_train)

    assert model.name == "baseline_percircuit"
    assert model.param_count == 4  # 3 circuit means + 1 global fallback mean
    assert np.isclose(model.circuit_means["c1"], 10.0)
    assert np.isclose(model.circuit_means["c2"], 20.0)

    # Predict on seen circuits (c1, c2) and unseen circuit (c4)
    circuits_test = np.array(["c1", "c2", "c4"])
    preds = model.predict(X[:3], circuits_test=circuits_test)

    assert np.isclose(preds[0], 10.0)
    assert np.isclose(preds[1], 20.0)
    assert np.isclose(preds[2], 20.0)  # global mean fallback (10+10+20+20+30+30)/6 = 20.0


def test_linear_regression_baseline():
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.5 * X[:, 2]

    model = LinearRegressionBaseline(alpha=0.01)
    model.fit(X, y)

    assert model.name == "linear"
    assert model.param_count == 6  # 5 weights + 1 intercept
    assert model.is_fitted

    preds = model.predict(X[:10])
    assert np.allclose(preds, y[:10], atol=1e-2)


def test_sequence_only_baseline():
    np.random.seed(42)
    X_seq = np.random.randn(100, 10).astype(np.float32)
    y = 1.5 * X_seq[:, 0] - 2.0 * X_seq[:, 3]

    model = SequenceOnlyBaseline(alpha=0.01)
    model.fit(X_seq, y)

    assert model.name == "seq_only"
    assert model.param_count == 11
    assert model.is_fitted

    preds = model.predict(X_seq[:5])
    assert np.allclose(preds, y[:5], atol=1e-2)
