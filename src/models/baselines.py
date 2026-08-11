"""Baseline models for Phase 2 comparisons.

Includes four baseline models of increasing strength per GEMINI.md §11 Phase 2:
1. Global mean (ConstantMeanBaseline)
2. Per-circuit mean (PerCircuitMeanBaseline)
3. Linear regression (LinearRegressionBaseline)
4. Sequence-only linear model (SequenceOnlyBaseline)
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseQoRModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ConstantMeanBaseline(BaseQoRModel):
    """Trivial baseline predicting the global mean of training target values."""

    def __init__(self) -> None:
        self.global_mean: float = 0.0
        self.is_fitted: bool = False

    @property
    def name(self) -> str:
        return "baseline_mean"

    @property
    def param_count(self) -> int:
        return 1  # single scalar mean value

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "ConstantMeanBaseline":
        """Computes global training target mean."""
        self.global_mean = float(np.mean(y))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns constant prediction array of training target mean."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n_samples, self.global_mean, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {"baseline_type": "global_mean"}


class PerCircuitMeanBaseline(BaseQoRModel):
    """Predicts each circuit's own training mean.

    Purpose:
        This baseline isolates how much QoR signal is attributable to circuit
        identity versus sequence structure.

    LOCO Fallback Behavior:
        Under leave-circuits-out (LOCO), test circuits are completely unseen
        during training. For unseen circuits, this model FALLS BACK to the
        global training target mean.
    """

    def __init__(self) -> None:
        self.circuit_means: Dict[str, float] = {}
        self.global_mean: float = 0.0
        self.is_fitted: bool = False

    @property
    def name(self) -> str:
        return "baseline_percircuit"

    @property
    def param_count(self) -> int:
        return len(self.circuit_means) + 1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        circuits_train: Optional[np.ndarray] = None,
    ) -> "PerCircuitMeanBaseline":
        """Computes per-circuit target means on training rows.

        Note: If `circuits_train` is not provided, assumes circuit identity can be
        inferred or falls back to global mean.
        """
        self.global_mean = float(np.mean(y))
        self.circuit_means.clear()

        if circuits_train is not None:
            unique_circuits = np.unique(circuits_train)
            for c in unique_circuits:
                c_mask = (circuits_train == c)
                self.circuit_means[str(c)] = float(np.mean(y[c_mask]))

        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        circuits_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predicts per-circuit mean for seen circuits, or falls back to global mean for unseen circuits."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        preds = np.full(n_samples, self.global_mean, dtype=np.float32)

        if circuits_test is not None:
            unseen_count = 0
            for i, c in enumerate(circuits_test):
                c_str = str(c)
                if c_str in self.circuit_means:
                    preds[i] = self.circuit_means[c_str]
                else:
                    unseen_count += 1

            if unseen_count > 0:
                logger.info(
                    f"PerCircuitMeanBaseline fallback: {unseen_count}/{n_samples} samples belong to "
                    f"unseen circuits under LOCO; assigned global training mean ({self.global_mean:.4f})."
                )

        return preds

    def get_params(self) -> Dict[str, Any]:
        return {
            "baseline_type": "per_circuit_mean",
            "n_seen_circuits": len(self.circuit_means),
        }


class LinearRegressionBaseline(BaseQoRModel):
    """Linear (Ridge) regression baseline trained on full available feature set.

    Invariants:
        StandardScaler is fit on training fold rows ONLY inside fit() to prevent
        data leakage under leave-circuits-out (INV-3).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.is_fitted: bool = False
        self.n_features_: int = 0

    @property
    def name(self) -> str:
        return "linear"

    @property
    def param_count(self) -> int:
        return self.n_features_ + 1 if self.is_fitted else 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "LinearRegressionBaseline":
        """Fits StandardScaler on training fold X only, then fits Ridge regression."""
        self.n_features_ = X.shape[1]
        # Fit scaler on training fold rows ONLY (INV-3)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transforms test X using fitted scaler and predicts target values."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "baseline_type": "ridge_linear",
            "ridge_alpha": self.alpha,
            "n_features": self.n_features_,
        }


class SequenceOnlyBaseline(BaseQoRModel):
    """Linear (Ridge) regression trained on sequence encodings ONLY (no circuit features).

    Purpose:
        Quantifies how much the sequence representation contributes to QoR prediction
        without any circuit context.

    Invariants:
        StandardScaler is fit on training fold rows ONLY inside fit() to prevent
        data leakage under leave-circuits-out (INV-3).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.is_fitted: bool = False
        self.n_features_: int = 0

    @property
    def name(self) -> str:
        return "seq_only"

    @property
    def param_count(self) -> int:
        return self.n_features_ + 1 if self.is_fitted else 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "SequenceOnlyBaseline":
        """Fits StandardScaler on sequence training fold X only, then fits Ridge regression."""
        self.n_features_ = X.shape[1]
        # Fit scaler on training fold rows ONLY (INV-3)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transforms sequence test X using fitted scaler and predicts target values."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "baseline_type": "ridge_seq_only",
            "ridge_alpha": self.alpha,
            "n_features": self.n_features_,
        }
