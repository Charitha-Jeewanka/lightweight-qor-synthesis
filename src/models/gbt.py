"""XGBoost and LightGBM models for logic synthesis QoR prediction.

Strictly follows GEMINI.md §11 Phase 4.
CPU-bound training with explicit thread and core count logging.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import lightgbm as lgb
import numpy as np
import xgboost as xgb

from src.models.base import BaseQoRModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_cpu_count() -> int:
    """Returns physical/logical CPU core count for CPU thread configuration."""
    count = os.cpu_count()
    return count if count is not None else 4


class XGBoostQoRModel(BaseQoRModel):
    """XGBoost regressor model wrapped for QoR prediction evaluation."""

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        seed: int = 42,
        n_jobs: Optional[int] = None,
    ):
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.seed = int(seed)
        self.n_jobs = n_jobs if n_jobs is not None else get_cpu_count()

        self.model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            tree_method="hist",
            eval_metric="rmse",
        )
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "XGBoostQoRModel":
        eval_set = [val_data] if val_data is not None else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("XGBoostQoRModel is not fitted yet!")
        preds = self.model.predict(X)
        return np.asarray(preds, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "xgb_max_depth": self.max_depth,
            "xgb_learning_rate": self.learning_rate,
            "xgb_n_estimators": self.n_estimators,
            "xgb_subsample": self.subsample,
            "xgb_colsample_bytree": self.colsample_bytree,
            "xgb_min_child_weight": self.min_child_weight,
            "xgb_reg_lambda": self.reg_lambda,
            "xgb_seed": self.seed,
            "xgb_n_jobs": self.n_jobs,
            "cpu_core_count": get_cpu_count(),
        }

    @property
    def param_count(self) -> int:
        if not self._is_fitted:
            return 0
        try:
            booster = self.model.get_booster()
            dump = booster.get_dump()
            return sum(len(tree.split("\n")) for tree in dump)
        except Exception:
            return self.n_estimators * (2 ** self.max_depth)

    def get_feature_importances(self) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted!")
        return np.asarray(self.model.feature_importances_, dtype=np.float32)


class LightGBMQoRModel(BaseQoRModel):
    """LightGBM regressor model wrapped for QoR prediction evaluation."""

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        seed: int = 42,
        n_jobs: Optional[int] = None,
    ):
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.min_child_weight = float(min_child_weight)
        self.reg_lambda = float(reg_lambda)
        self.seed = int(seed)
        self.n_jobs = n_jobs if n_jobs is not None else get_cpu_count()

        self.model = lgb.LGBMRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            verbosity=-1,
        )
        self._is_fitted = False

    @property
    def name(self) -> str:
        return "lightgbm"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs: Any,
    ) -> "LightGBMQoRModel":
        eval_set = [val_data] if val_data is not None else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
        )
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("LightGBMQoRModel is not fitted yet!")
        preds = self.model.predict(X)
        return np.asarray(preds, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "lgb_max_depth": self.max_depth,
            "lgb_learning_rate": self.learning_rate,
            "lgb_n_estimators": self.n_estimators,
            "lgb_subsample": self.subsample,
            "lgb_colsample_bytree": self.colsample_bytree,
            "lgb_min_child_weight": self.min_child_weight,
            "lgb_reg_lambda": self.reg_lambda,
            "lgb_seed": self.seed,
            "lgb_n_jobs": self.n_jobs,
            "cpu_core_count": get_cpu_count(),
        }

    @property
    def param_count(self) -> int:
        if not self._is_fitted:
            return 0
        try:
            booster = self.model.booster_
            dump = booster.dump_model()
            num_nodes = 0
            for tree in dump.get("tree_info", []):
                num_nodes += len(tree.get("tree_structure", {}))
            return num_nodes
        except Exception:
            return self.n_estimators * (2 ** self.max_depth)

    def get_feature_importances(self) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted!")
        return np.asarray(self.model.feature_importances_, dtype=np.float32)
