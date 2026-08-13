"""Unit tests for Phase 6 GNN data pipeline, architecture, parameter count, and OOM handler."""

import pytest
import numpy as np
import torch

from src.data.graphs import CircuitGraphRegistry, CircuitGroupedDataLoader
from src.models.gnn import GNNQoRModel, PyTorchGNNModule


def test_circuit_graph_registry() -> None:
    """Verifies graph registry loading, feature dimension (4), and edge_index dtype."""
    registry = CircuitGraphRegistry(max_graph_nodes=250000)
    g_data = registry.get_graph("arbiter")

    assert g_data is not None
    assert g_data.x.shape[1] == 4  # 3 one-hot + 1 norm level
    assert g_data.x.dtype == torch.float32
    assert g_data.edge_index.dtype == torch.int32
    assert g_data.num_nodes > 0


def test_registry_max_graph_nodes_guard() -> None:
    """Verifies that max_graph_nodes threshold correctly skips oversized graphs."""
    registry = CircuitGraphRegistry(max_graph_nodes=100)  # artificially low threshold
    g_data = registry.get_graph("arbiter")

    assert g_data is None
    assert "arbiter" in registry.skipped_circuits


def test_circuit_grouped_dataloader() -> None:
    """Verifies circuit-grouped batching behavior."""
    registry = CircuitGraphRegistry(max_graph_nodes=250000)

    circuits = np.array(["arbiter"] * 50 + ["bar"] * 50 + ["cavlc"] * 50)
    seq_features = np.random.randn(150, 40).astype(np.float32)
    targets = np.random.randn(150).astype(np.float32)

    loader = CircuitGroupedDataLoader(
        circuits=circuits,
        seq_features=seq_features,
        targets=targets,
        registry=registry,
        batch_size=32,
        circuits_per_batch=2,
        shuffle=True,
        seed=42,
    )

    batch_count = 0
    for batch in loader:
        batch_count += 1
        assert len(batch["circuits"]) <= 2
        assert "circuit_indices" in batch
        assert "seq_features" in batch
        assert "targets" in batch
        assert batch["seq_features"].shape[1] == 40

    assert batch_count > 0


def test_gnn_param_count_and_fit_predict() -> None:
    """Verifies parameter count cap (<1M) and basic fit/predict execution."""
    model = GNNQoRModel(
        gnn_type="gcn",
        gnn_hidden_dim=32,
        gnn_layers=2,
        readout_hidden_dims=[64, 32],
        max_epochs=2,
        batch_size=32,
        device="cpu",
    )

    # Synthetic data
    X = np.random.randn(60, 40).astype(np.float32)
    y = np.random.randn(60).astype(np.float32)
    circuits = np.array(["arbiter"] * 30 + ["bar"] * 30)

    model.fit(X, y, circuits_train=circuits)

    assert model.param_count < 1_000_000
    assert model.param_count > 0

    preds = model.predict(X, circuits_test=circuits)
    assert len(preds) == 60
    assert not np.isnan(preds).any()


def test_gin_variant_smoke() -> None:
    """Verifies GIN variant instantiation and execution."""
    model = GNNQoRModel(
        gnn_type="gin",
        gnn_hidden_dim=16,
        gnn_layers=2,
        readout_hidden_dims=[32],
        max_epochs=1,
        batch_size=16,
        device="cpu",
    )

    X = np.random.randn(30, 20).astype(np.float32)
    y = np.random.randn(30).astype(np.float32)
    circuits = np.array(["bar"] * 30)

    model.fit(X, y, circuits_train=circuits)
    preds = model.predict(X, circuits_test=circuits)

    assert len(preds) == 30
    assert model.param_count < 1_000_000
