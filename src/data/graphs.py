"""Circuit-indexed graph registry and circuit-grouped batching for GNN models.

Strictly follows GEMINI.md §6 (OOM Defense) and §7.1.
- Graphs parsed from Phase 3 .npz cache into PyTorch Geometric Data objects once per circuit.
- Node features: 4-dim (3-dim PI/AND/PO one-hot + normalized depth).
- Edge index: int32 tensor on CPU registry, cast to int64 only during CUDA message passing.
- max_graph_nodes guard (default 250,000). Skips graphs exceeding limit.
- Circuit-grouped batching: batches formed by sampling 2-4 circuits and drawing rows from
  those circuits only. NEVER per-sample graph attachment or full uniform row sampling.
"""

from typing import Any, Dict, List, Optional, Tuple, Iterator
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

from src.data.aiger import load_or_parse_aig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CircuitGraphRegistry:
    """Registry holding parsed PyG graph objects for all EPFL circuits.

    Reads from cached Phase 3 `.npz` files.
    Constructs minimal 4-dim node features (PI/AND/PO one-hot + normalized depth).
    Guards large graphs with `max_graph_nodes`.
    """

    def __init__(self, max_graph_nodes: int = 250000) -> None:
        self.max_graph_nodes = int(max_graph_nodes)
        self.graphs: Dict[str, Data] = {}
        self.skipped_circuits: Dict[str, str] = {}

    def get_graph(self, circuit_name: str) -> Optional[Data]:
        """Returns PyG Data object for specified circuit, or None if skipped/too large."""
        c_str = str(circuit_name)
        if c_str in self.skipped_circuits:
            return None
        if c_str in self.graphs:
            return self.graphs[c_str]

        # Load from .npz cache or parse
        try:
            parsed = load_or_parse_aig(c_str)
        except Exception as e:
            msg = f"Failed to load/parse graph for '{c_str}': {e}"
            logger.warning(msg)
            self.skipped_circuits[c_str] = msg
            return None

        num_nodes = len(parsed.node_types)
        if num_nodes > self.max_graph_nodes:
            msg = (
                f"Circuit '{c_str}' has {num_nodes} nodes, which exceeds max_graph_nodes "
                f"guard threshold ({self.max_graph_nodes}). Skipping graph."
            )
            logger.warning(msg)
            self.skipped_circuits[c_str] = msg
            return None

        # Build 4-dim node feature matrix (float32)
        # 1. One-hot node type: 0=PI, 1=AND, 2=PO -> 3 dims
        node_types_t = torch.from_numpy(parsed.node_types).long()
        one_hot = torch.zeros(num_nodes, 3, dtype=torch.float32)
        one_hot.scatter_(1, node_types_t.unsqueeze(1), 1.0)

        # 2. Normalized level/depth -> 1 dim
        max_lev = float(parsed.max_level) if parsed.max_level > 0 else 1.0
        norm_level = torch.from_numpy(parsed.levels).float().unsqueeze(1) / max_lev

        # Concatenate features -> shape [N, 4]
        x = torch.cat([one_hot, norm_level], dim=1)

        # Edge index in int32 format per §6.2
        edge_index = torch.from_numpy(parsed.edge_index).int()

        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        self.graphs[c_str] = data
        return data

    def preload_all(self, circuit_names: List[str]) -> None:
        """Preloads graphs for a list of circuit names."""
        for name in circuit_names:
            self.get_graph(name)


class CircuitGroupedDataLoader:
    """Mini-batch data loader enforcing circuit-grouped batching per GEMINI.md §6.1.

    Groups samples by circuit so each mini-batch contains samples from only 2-4 circuits.
    Prevents materializing all circuit graphs on GPU per step and prevents OOM.
    """

    def __init__(
        self,
        circuits: np.ndarray,
        seq_features: np.ndarray,
        targets: Optional[np.ndarray],
        registry: CircuitGraphRegistry,
        batch_size: int = 256,
        circuits_per_batch: int = 2,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.circuits_arr = np.array([str(c) for c in circuits])
        self.seq_features = seq_features.astype(np.float32)
        self.targets = targets.astype(np.float32) if targets is not None else None
        self.registry = registry
        self.batch_size = int(batch_size)
        self.circuits_per_batch = max(1, min(4, int(circuits_per_batch)))
        self.shuffle = shuffle
        self.seed = int(seed)

        # Filter out rows belonging to skipped/too-large circuits
        self.valid_indices: List[int] = []
        for i, c in enumerate(self.circuits_arr):
            if self.registry.get_graph(c) is not None:
                self.valid_indices.append(i)

        if len(self.valid_indices) == 0:
            logger.warning("CircuitGroupedDataLoader initialized with 0 valid sample indices.")

    def __len__(self) -> int:
        return (len(self.valid_indices) + self.batch_size - 1) // self.batch_size if self.valid_indices else 0

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not self.valid_indices:
            return

        rng = np.random.default_rng(self.seed)

        if self.shuffle:
            # 1. Group valid indices by circuit
            c_to_indices: Dict[str, List[int]] = {}
            for idx in self.valid_indices:
                c = self.circuits_arr[idx]
                if c not in c_to_indices:
                    c_to_indices[c] = []
                c_to_indices[c].append(idx)

            # Shuffle row order within each circuit
            for c in c_to_indices:
                rng.shuffle(c_to_indices[c])

            # 2. Partition available circuits into groups of size circuits_per_batch
            unique_circuits = list(c_to_indices.keys())
            rng.shuffle(unique_circuits)

            circuit_groups: List[List[str]] = []
            for i in range(0, len(unique_circuits), self.circuits_per_batch):
                circuit_groups.append(unique_circuits[i : i + self.circuits_per_batch])

            # 3. For each group of circuits, form mini-batches
            for c_group in circuit_groups:
                group_row_indices: List[int] = []
                for c in c_group:
                    group_row_indices.extend(c_to_indices[c])

                rng.shuffle(group_row_indices)

                for b_start in range(0, len(group_row_indices), self.batch_size):
                    batch_row_indices = group_row_indices[b_start : b_start + self.batch_size]
                    yield self._build_batch_dict(batch_row_indices)
        else:
            # Sequential / non-shuffled evaluation
            for b_start in range(0, len(self.valid_indices), self.batch_size):
                batch_row_indices = self.valid_indices[b_start : b_start + self.batch_size]
                yield self._build_batch_dict(batch_row_indices)

    def _build_batch_dict(self, batch_indices: List[int]) -> Dict[str, Any]:
        batch_circuits = [self.circuits_arr[i] for i in batch_indices]
        unique_c_list = list(dict.fromkeys(batch_circuits))  # preserves insertion order
        c_to_idx = {c: i for i, c in enumerate(unique_c_list)}

        c_indices = torch.tensor([c_to_idx[c] for c in batch_circuits], dtype=torch.long)
        seq_feats = torch.from_numpy(self.seq_features[batch_indices])

        batch_dict: Dict[str, Any] = {
            "circuits": unique_c_list,
            "circuit_indices": c_indices,
            "seq_features": seq_feats,
        }

        if self.targets is not None:
            batch_dict["targets"] = torch.from_numpy(self.targets[batch_indices])

        return batch_dict
