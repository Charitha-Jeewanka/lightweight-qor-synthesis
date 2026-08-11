"""Unit tests for XGBoost and LightGBM model implementations."""

import numpy as np
import pytest
from src.models.gbt import LightGBMQoRModel, XGBoostQoRModel


def test_xgboost_fit_predict():
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = XGBoostQoRModel(max_depth=3, n_estimators=10, seed=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (100,)
    assert preds.dtype == np.float32
    assert "xgb_max_depth" in model.get_params()
    assert len(model.get_feature_importances()) == 10


def test_lightgbm_fit_predict():
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    model = LightGBMQoRModel(max_depth=3, n_estimators=10, seed=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (100,)
    assert preds.dtype == np.float32
    assert "lgb_max_depth" in model.get_params()
    assert len(model.get_feature_importances()) == 10
