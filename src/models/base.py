"""Base abstract interface that all QoR prediction models must implement."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class BaseQoRModel(ABC):
    """Abstract Base Class for all QoR prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for the model architecture."""
        pass

    @property
    @abstractmethod
    def param_count(self) -> int:
        """Total trainable parameter count of the model."""
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "BaseQoRModel":
        """Fits model on training features and targets.

        Args:
            X: feature array of shape (N, D)
            y: target array of shape (N,)
            val_data: optional tuple of (X_val, y_val) for early stopping / validation

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates target predictions.

        Args:
            X: feature array of shape (N, D)

        Returns:
            np.ndarray of shape (N,), dtype float32
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Returns dictionary of model hyperparameter attributes."""
        return {}
