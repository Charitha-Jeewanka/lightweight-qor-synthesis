"""Unit tests for MLPQoRModel implementation."""

import numpy as np
import pytest
from src.models.mlp import MLPQoRModel


def test_mlp_model_basic_fit_predict():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(200, 20)).astype(np.float32)
    y_train = (X_train[:, 0] * 2.0 + X_train[:, 1] * 0.5).astype(np.float32)

    X_val = rng.normal(size=(50, 20)).astype(np.float32)
    y_val = (X_val[:, 0] * 2.0 + X_val[:, 1] * 0.5).astype(np.float32)

    model = MLPQoRModel(
        hidden_dims=[32, 16],
        dropout=0.1,
        learning_rate=0.01,
        max_epochs=20,
        patience=5,
        n_continuous=8,
        seed=42,
    )

    model.fit(X_train, y_train, val_data=(X_val, y_val))
    preds = model.predict(X_val)

    assert preds.shape == (50,)
    assert preds.dtype == np.float32
    assert model.param_count > 0
    assert model.train_time_s > 0.0


def test_mlp_model_no_float64():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10)).astype(np.float32)
    y = rng.normal(size=(100,)).astype(np.float32)

    model = MLPQoRModel(hidden_dims=[16], max_epochs=5, n_continuous=5, seed=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.dtype != np.float64
    assert preds.dtype == np.float32
