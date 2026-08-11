"""Baseline models for Phase 1 deliverable and Phase 2 comparisons.

Includes ConstantMeanBaseline (predicts training target mean).
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.models.base import BaseQoRModel


class ConstantMeanBaseline(BaseQoRModel):
    """Trivial baseline predicting the global mean of training target values."""

    def __init__(self) -> None:
        self.mean_val: float = 0.0
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
        self.mean_val = float(np.mean(y))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns constant prediction array of training target mean."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n_samples, self.mean_val, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {"baseline_type": "global_mean"}
