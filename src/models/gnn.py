"""PyTorch GCN and GIN architectures and BaseQoRModel wrapper for QoR prediction.

Strictly follows GEMINI.md §6 (OOM Defense) and §11 Phase 6.
- Architecture: GCN / GIN graph backbone + Readout MLP concatenating pooled graph embedding
  with sequence encoding.
- All models capped under 1M parameter budget.
- Automatic GPU OOM handler with batch-size backoff.
- Peak GPU memory tracked via torch.cuda.max_memory_allocated().
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import psutil
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

from src.data.graphs import CircuitGraphRegistry, CircuitGroupedDataLoader
from src.models.base import BaseQoRModel
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SafeBatchNorm1d(nn.Module):
    """Wrapper around nn.BatchNorm1d that safely bypasses normalization if batch size is 1 during training."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.size(0) == 1 and self.training:
            return x
        return self.bn(x)


class PyTorchGNNModule(nn.Module):
    """PyTorch GNN Module supporting GCN and GIN backbone variants."""

    def __init__(
        self,
        gnn_type: str,
        in_dim: int,
        gnn_hidden_dim: int,
        gnn_layers: int,
        seq_dim: int,
        readout_hidden_dims: List[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_layers_count = gnn_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        curr_dim = in_dim
        for _ in range(gnn_layers):
            if self.gnn_type == "gcn":
                conv = GCNConv(curr_dim, gnn_hidden_dim)
            elif self.gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(curr_dim, gnn_hidden_dim),
                    SafeBatchNorm1d(gnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
                )
                conv = GINConv(mlp)
            else:
                raise ValueError(f"Unknown GNN type: '{gnn_type}'. Must be 'gcn' or 'gin'")

            self.convs.append(conv)
            self.bns.append(SafeBatchNorm1d(gnn_hidden_dim))
            curr_dim = gnn_hidden_dim

        # Readout MLP taking concatenated [graph_embedding || sequence_encoding]
        readout_in_dim = gnn_hidden_dim + seq_dim
        readout_layers: List[nn.Module] = []
        curr_r_dim = readout_in_dim

        for r_dim in readout_hidden_dims:
            readout_layers.append(nn.Linear(curr_r_dim, r_dim))
            readout_layers.append(SafeBatchNorm1d(r_dim))
            readout_layers.append(nn.ReLU())
            if dropout > 0.0:
                readout_layers.append(nn.Dropout(dropout))
            curr_r_dim = r_dim

        readout_layers.append(nn.Linear(curr_r_dim, 1))
        self.readout_mlp = nn.Sequential(*readout_layers)

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encodes single circuit graph into global graph embedding of shape [1, gnn_hidden_dim]."""
        edge_index = edge_index.long()
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            if h.dim() == 2 and h.size(0) > 1:
                h = bn(h)
            h = torch.relu(h)
            if self.dropout > 0.0:
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        batch_idx = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        pooled = global_mean_pool(h, batch_idx)  # [1, gnn_hidden_dim]
        return pooled

    def forward(
        self,
        unique_graph_embeddings: torch.Tensor,
        circuit_indices: torch.Tensor,
        seq_features: torch.Tensor,
    ) -> torch.Tensor:
        row_graph_embeddings = unique_graph_embeddings[circuit_indices]
        combined = torch.cat([row_graph_embeddings, seq_features], dim=1)
        return self.readout_mlp(combined).squeeze(-1)


class GNNQoRModel(BaseQoRModel):
    """BaseQoRModel wrapper for GCN and GIN graph neural networks."""

    def __init__(
        self,
        gnn_type: str = "gcn",
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 3,
        readout_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        circuits_per_batch: int = 2,
        max_epochs: int = 150,
        patience: int = 15,
        max_graph_nodes: int = 250000,
        seed: int = 42,
        device: str = "auto",
    ):
        self.gnn_type = gnn_type.lower()
        self.gnn_hidden_dim = int(gnn_hidden_dim)
        self.gnn_layers = int(gnn_layers)
        self.readout_hidden_dims = readout_hidden_dims if readout_hidden_dims is not None else [128, 64]
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.circuits_per_batch = int(circuits_per_batch)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.max_graph_nodes = int(max_graph_nodes)
        self.seed = int(seed)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.registry = CircuitGraphRegistry(max_graph_nodes=self.max_graph_nodes)
        self.scaler = StandardScaler()
        self.torch_model: Optional[PyTorchGNNModule] = None

        self._param_count: int = 0
        self.train_time_s: float = 0.0
        self.peak_gpu_mem_mb: float = 0.0
        self.peak_cpu_mem_mb: float = 0.0
        self.inference_latency_ms: float = 0.0

    @property
    def name(self) -> str:
        r_str = "x".join(str(d) for d in self.readout_hidden_dims)
        return f"GNN_{self.gnn_type.upper()}_h{self.gnn_hidden_dim}_L{self.gnn_layers}_r{r_str}"

    @property
    def param_count(self) -> int:
        return self._param_count

    def _fit_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]],
        circuits_train: np.ndarray,
        circuits_val: Optional[np.ndarray],
        current_batch_size: int,
    ) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_train_scaled = self.scaler.fit_transform(X).astype(np.float32)
        seq_dim = X_train_scaled.shape[1]

        self.torch_model = PyTorchGNNModule(
            gnn_type=self.gnn_type,
            in_dim=4,  # 3 one-hot + 1 norm level
            gnn_hidden_dim=self.gnn_hidden_dim,
            gnn_layers=self.gnn_layers,
            seq_dim=seq_dim,
            readout_hidden_dims=self.readout_hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        self._param_count = sum(p.numel() for p in self.torch_model.parameters() if p.requires_grad)

        if self._param_count >= 1_000_000:
            raise ValueError(
                f"GNN parameter count ({self._param_count}) exceeds non-negotiable 1M parameter cap!"
            )

        optimizer = torch.optim.AdamW(
            self.torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        train_loader = CircuitGroupedDataLoader(
            circuits=circuits_train,
            seq_features=X_train_scaled,
            targets=y,
            registry=self.registry,
            batch_size=current_batch_size,
            circuits_per_batch=self.circuits_per_batch,
            shuffle=True,
            seed=self.seed,
        )

        has_val = val_data is not None and circuits_val is not None and len(val_data[0]) > 0
        val_loader = None
        if has_val and val_data is not None and circuits_val is not None:
            X_v, y_v = val_data
            X_val_scaled = self.scaler.transform(X_v).astype(np.float32)
            val_loader = CircuitGroupedDataLoader(
                circuits=circuits_val,
                seq_features=X_val_scaled,
                targets=y_v,
                registry=self.registry,
                batch_size=current_batch_size,
                circuits_per_batch=self.circuits_per_batch,
                shuffle=False,
                seed=self.seed,
            )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.torch_model.train()
            for batch in train_loader:
                circuits = batch["circuits"]
                unique_embs = []
                for c_name in circuits:
                    g_data = self.registry.get_graph(c_name)
                    if g_data is None:
                        continue
                    x_t = g_data.x.to(self.device)
                    e_t = g_data.edge_index.to(self.device)
                    h_c = self.torch_model.encode_graph(x_t, e_t)
                    unique_embs.append(h_c)

                if not unique_embs:
                    continue

                unique_emb_t = torch.cat(unique_embs, dim=0)
                c_indices = batch["circuit_indices"].to(self.device)
                seq_f = batch["seq_features"].to(self.device)
                targets = batch["targets"].to(self.device)

                optimizer.zero_grad()
                out = self.torch_model(unique_emb_t, c_indices, seq_f)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()

            # Validation step
            if has_val and val_loader is not None:
                self.torch_model.eval()
                val_losses: List[float] = []
                with torch.no_grad():
                    for batch in val_loader:
                        circuits = batch["circuits"]
                        unique_embs = []
                        for c_name in circuits:
                            g_data = self.registry.get_graph(c_name)
                            if g_data is None:
                                continue
                            x_t = g_data.x.to(self.device)
                            e_t = g_data.edge_index.to(self.device)
                            h_c = self.torch_model.encode_graph(x_t, e_t)
                            unique_embs.append(h_c)

                        if not unique_embs:
                            continue

                        unique_emb_t = torch.cat(unique_embs, dim=0)
                        c_indices = batch["circuit_indices"].to(self.device)
                        seq_f = batch["seq_features"].to(self.device)
                        targets = batch["targets"].to(self.device)

                        out = self.torch_model(unique_emb_t, c_indices, seq_f)
                        val_losses.append(criterion(out, targets).item())

                mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.torch_model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if has_val and best_state is not None:
            self.torch_model.load_state_dict(best_state)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        circuits_train: Optional[np.ndarray] = None,
        circuits_val: Optional[np.ndarray] = None,
    ) -> "GNNQoRModel":
        t0 = time.perf_counter()
        process = psutil.Process()
        mem_before_cpu = process.memory_info().rss

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        if circuits_train is None:
            # Fallback if circuits_train is omitted: single dummy circuit
            circuits_train = np.array(["arbiter"] * len(X))

        current_batch_size = self.batch_size
        max_retries = 2
        attempt = 0

        while attempt <= max_retries:
            try:
                self._fit_loop(X, y, val_data, circuits_train, circuits_val, current_batch_size)
                break
            except torch.cuda.OutOfMemoryError as oom_err:
                attempt += 1
                logger.warning(
                    f"GPU OOM caught during fit (attempt {attempt}/{max_retries+1}). "
                    f"Config: {self.get_params()}, batch_size={current_batch_size}. Halving batch_size."
                )

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if self.torch_model is not None:
                    del self.torch_model
                    self.torch_model = None

                current_batch_size = max(16, current_batch_size // 2)

                if attempt > max_retries:
                    error_msg = (
                        f"OOM error persisted after {max_retries} retries with batch_size={current_batch_size}. "
                        f"Full config: {self.get_params()}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from oom_err

        mem_after_cpu = process.memory_info().rss
        self.peak_cpu_mem_mb = max(mem_before_cpu, mem_after_cpu) / (1024.0 * 1024.0)

        if self.device.type == "cuda":
            self.peak_gpu_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
            torch.cuda.empty_cache()
        else:
            self.peak_gpu_mem_mb = 0.0

        self.train_time_s = time.perf_counter() - t0
        return self

    def predict(
        self,
        X: np.ndarray,
        circuits_test: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.torch_model is None:
            raise RuntimeError("GNNQoRModel must be fit before calling predict()")

        t0 = time.perf_counter()

        if circuits_test is None:
            # Fallback to single dummy circuit if omitted (e.g. latency sampling)
            circuits_test = np.array(["arbiter"] * len(X))

        X_scaled = self.scaler.transform(X).astype(np.float32)

        loader = CircuitGroupedDataLoader(
            circuits=circuits_test,
            seq_features=X_scaled,
            targets=None,
            registry=self.registry,
            batch_size=self.batch_size,
            circuits_per_batch=self.circuits_per_batch,
            shuffle=False,
            seed=self.seed,
        )

        preds_list: List[np.ndarray] = []
        self.torch_model.eval()

        with torch.no_grad():
            for batch in loader:
                circuits = batch["circuits"]
                unique_embs = []
                for c_name in circuits:
                    g_data = self.registry.get_graph(c_name)
                    if g_data is None:
                        # Fallback for skipped circuit if predict called on it: zero embedding
                        h_c = torch.zeros(1, self.gnn_hidden_dim, device=self.device)
                    else:
                        x_t = g_data.x.to(self.device)
                        e_t = g_data.edge_index.to(self.device)
                        h_c = self.torch_model.encode_graph(x_t, e_t)
                    unique_embs.append(h_c)

                unique_emb_t = torch.cat(unique_embs, dim=0)
                c_indices = batch["circuit_indices"].to(self.device)
                seq_f = batch["seq_features"].to(self.device)

                batch_preds = self.torch_model(unique_emb_t, c_indices, seq_f)
                preds_list.append(batch_preds.cpu().numpy().astype(np.float32))

        if preds_list:
            preds = np.concatenate(preds_list, axis=0)
        else:
            preds = np.zeros(len(X), dtype=np.float32)

        self.inference_latency_ms = (time.perf_counter() - t0) * 1000.0

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return preds

    def get_params(self) -> Dict[str, Any]:
        return {
            "gnn_type": self.gnn_type,
            "gnn_hidden_dim": self.gnn_hidden_dim,
            "gnn_layers": self.gnn_layers,
            "readout_hidden_dims": self.readout_hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "circuits_per_batch": self.circuits_per_batch,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "max_graph_nodes": self.max_graph_nodes,
            "seed": self.seed,
            "device": str(self.device),
        }
