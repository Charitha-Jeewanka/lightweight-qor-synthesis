"""PyTorch Small MLP architecture and BaseQoRModel wrapper for QoR prediction."""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import psutil
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseQoRModel


class PyTorchMLPModule(nn.Module):
    """PyTorch MLP Module with configurable hidden layers and dropout."""

    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        curr_dim = in_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class MLPQoRModel(BaseQoRModel):
    """BaseQoRModel wrapper for PyTorch MLP trained strictly on CPU."""

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 200,
        patience: int = 15,
        n_continuous: int = 8,  # number of leading continuous features to scale
        seed: int = 42,
    ):
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.n_continuous = int(n_continuous)
        self.seed = int(seed)

        self.scaler = StandardScaler()
        self.device = torch.device("cpu")
        self.torch_model: Optional[PyTorchMLPModule] = None
        self._param_count: int = 0
        self.train_time_s: float = 0.0
        self.peak_cpu_mem_mb: float = 0.0
        self.inference_latency_ms: float = 0.0

    @property
    def name(self) -> str:
        h_str = "x".join(str(d) for d in self.hidden_dims)
        return f"MLP_{h_str}"

    @property
    def param_count(self) -> int:
        return self._param_count

    def _preprocess(self, X: np.ndarray, is_train: bool = False) -> np.ndarray:
        """Standardizes continuous features while keeping categorical/one-hot columns unscaled."""
        X_copy = X.copy().astype(np.float32)
        if self.n_continuous > 0 and X_copy.shape[1] >= self.n_continuous:
            if is_train:
                X_copy[:, : self.n_continuous] = self.scaler.fit_transform(X_copy[:, : self.n_continuous])
            else:
                X_copy[:, : self.n_continuous] = self.scaler.transform(X_copy[:, : self.n_continuous])
        return X_copy

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "MLPQoRModel":
        t0 = time.perf_counter()
        process = psutil.Process()
        mem_before = process.memory_info().rss

        # Set seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Scale continuous features strictly inside fit (INV-3)
        X_train_scaled = self._preprocess(X, is_train=True)
        y_train_arr = y.astype(np.float32)

        in_dim = X_train_scaled.shape[1]
        self.torch_model = PyTorchMLPModule(
            in_dim=in_dim, hidden_dims=self.hidden_dims, dropout=self.dropout
        ).to(self.device)

        self._param_count = sum(p.numel() for p in self.torch_model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            self.torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.from_numpy(X_train_scaled), torch.from_numpy(y_train_arr)
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        has_val = val_data is not None
        if has_val and val_data is not None:
            X_v, y_v = val_data
            X_val_scaled = self._preprocess(X_v, is_train=False)
            val_x_t = torch.from_numpy(X_val_scaled).to(self.device)
            val_y_t = torch.from_numpy(y_v.astype(np.float32)).to(self.device)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.torch_model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.torch_model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            if has_val:
                self.torch_model.eval()
                with torch.no_grad():
                    val_out = self.torch_model(val_x_t)
                    val_loss = criterion(val_out, val_y_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.torch_model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if has_val and best_state is not None:
            self.torch_model.load_state_dict(best_state)

        mem_after = process.memory_info().rss
        self.peak_cpu_mem_mb = max(mem_before, mem_after) / (1024.0 * 1024.0)
        self.train_time_s = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.torch_model is None:
            raise RuntimeError("MLPQoRModel must be fit before calling predict()")

        t0 = time.perf_counter()
        X_scaled = self._preprocess(X, is_train=False)
        x_t = torch.from_numpy(X_scaled).to(self.device)

        self.torch_model.eval()
        with torch.no_grad():
            preds = self.torch_model(x_t).cpu().numpy().astype(np.float32)

        self.inference_latency_ms = (time.perf_counter() - t0) * 1000.0
        return preds

    def get_params(self) -> Dict[str, Any]:
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "n_continuous": self.n_continuous,
            "seed": self.seed,
        }
