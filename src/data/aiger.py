"""Binary AIGER (.aig) parser, graph structure extractor, and graph caching.

Strictly follows GEMINI.md §7.1.
All graph traversals are strictly ITERATIVE to support deep circuits like `hyp` (24,801 levels).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple
import numpy as np

from src.utils.logging_utils import get_logger
from src.utils.paths import get_processed_data_dir, get_project_root

logger = get_logger(__name__)


@dataclass
class ParsedAIG:
    """Container holding graph data structures parsed from binary AIGER file."""

    circuit_name: str
    num_inputs: int   # I (pi)
    num_outputs: int  # O (po)
    num_and: int      # A (and_count)
    max_var: int      # M
    max_level: int    # level_count
    node_types: np.ndarray   # int8: 0=PI, 1=AND, 2=PO
    levels: np.ndarray       # int32: level of each node
    edge_index: np.ndarray   # int32: shape [2, E], directed edges (fanin -> node)
    and_fanins: np.ndarray   # int32: shape [A, 2], fanin node indices for each AND node
    and_node_indices: np.ndarray # int32: node indices of AND gates (shape [A])
    po_driver_nodes: np.ndarray  # int32: node indices driving POs (shape [O])


def decode_uvarint(f: BinaryIO) -> int:
    """Decodes a 7-bit variable-length little-endian unsigned integer from binary stream."""
    val = 0
    shift = 0
    while True:
        byte = f.read(1)
        if not byte:
            raise EOFError("Unexpected end of binary AIGER file while decoding uvarint.")
        b = byte[0]
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return val


def parse_aiger_file(aig_path: Path) -> ParsedAIG:
    """Parses a binary AIGER (.aig) file into graph data structures.

    Args:
        aig_path: absolute path to .aig file

    Returns:
        ParsedAIG container with node_types, levels, edge_index, and fanins.
    """
    circuit_name = aig_path.stem

    with open(aig_path, "rb") as f:
        # 1. Parse header line: aig M I L O A
        header_line = f.readline().decode("ascii").strip()
        if not header_line.startswith("aig"):
            raise ValueError(f"Invalid AIGER file header: '{header_line}' in {aig_path}")

        parts = header_line.split()
        if len(parts) < 6:
            raise ValueError(f"Malformed AIGER header: '{header_line}' in {aig_path}")

        M = int(parts[1])
        I = int(parts[2])
        L = int(parts[3])
        O = int(parts[4])
        A = int(parts[5])

        if L != 0:
            raise ValueError(f"EPFL benchmark is expected to be combinational (L=0), got L={L} in {aig_path}")

        # 2. Parse Latches (L lines) - None expected
        for _ in range(L):
            f.readline()

        # 3. Parse Outputs (O lines, ASCII literals)
        po_literals: List[int] = []
        for _ in range(O):
            line = f.readline().decode("ascii").strip()
            po_literals.append(int(line))

        # 4. Node Indexing Mapping
        # Variable 0: Constant False (mapped to index -1 or special)
        # Variable 1..I: PIs -> node indices 0 .. I-1
        # Variable I+1..I+A: ANDs -> node indices I .. I+A-1
        # Virtual PO nodes (optional): indices I+A .. I+A+O-1
        num_nodes = I + A + O

        # Map variable index v (1..M) to node index (0..I+A-1)
        # Variable 0 (Constant False) is mapped to -1
        var_to_node = np.full(M + 1, -1, dtype=np.int32)
        for v in range(1, I + 1):
            var_to_node[v] = v - 1  # PI node indices 0..I-1

        for i in range(1, A + 1):
            v = I + i
            var_to_node[v] = I + (i - 1)  # AND node indices I..I+A-1

        node_types = np.zeros(num_nodes, dtype=np.int8)
        node_types[:I] = 0           # 0 = PI
        node_types[I : I + A] = 1    # 1 = AND
        node_types[I + A :] = 2      # 2 = PO

        levels = np.zeros(num_nodes, dtype=np.int32)  # PIs start at level 0

        and_fanins = np.zeros((A, 2), dtype=np.int32)
        and_node_indices = np.arange(I, I + A, dtype=np.int32)

        edge_sources: List[int] = []
        edge_targets: List[int] = []

        # 5. Parse Binary Delta-Encoded AND Gates (A gates)
        for i in range(1, A + 1):
            lhs_var = I + i
            lhs_lit = 2 * lhs_var
            and_node_idx = I + (i - 1)

            delta0 = decode_uvarint(f)
            delta1 = decode_uvarint(f)

            rhs0_lit = lhs_lit - delta0
            rhs1_lit = rhs0_lit - delta1

            var0 = rhs0_lit >> 1
            var1 = rhs1_lit >> 1

            node0 = int(var_to_node[var0]) if var0 > 0 else -1
            node1 = int(var_to_node[var1]) if var1 > 0 else -1

            and_fanins[i - 1, 0] = node0
            and_fanins[i - 1, 1] = node1

            # Compute level iteratively in topological order
            l0 = levels[node0] if node0 >= 0 else 0
            l1 = levels[node1] if node1 >= 0 else 0
            levels[and_node_idx] = max(l0, l1) + 1

            if node0 >= 0:
                edge_sources.append(node0)
                edge_targets.append(and_node_idx)
            if node1 >= 0:
                edge_sources.append(node1)
                edge_targets.append(and_node_idx)

        # 6. Process PO Nodes
        po_driver_nodes = np.zeros(O, dtype=np.int32)
        po_levels: List[int] = []

        for j, po_lit in enumerate(po_literals):
            po_var = po_lit >> 1
            driver_node = int(var_to_node[po_var]) if po_var > 0 else -1
            po_node_idx = I + A + j

            po_driver_nodes[j] = driver_node
            po_level = levels[driver_node] if driver_node >= 0 else 0
            levels[po_node_idx] = po_level
            po_levels.append(po_level)

            if driver_node >= 0:
                edge_sources.append(driver_node)
                edge_targets.append(po_node_idx)

        max_level = max(po_levels) if po_levels else (int(np.max(levels[I : I + A])) if A > 0 else 0)

        edge_index = np.array([edge_sources, edge_targets], dtype=np.int32)

        return ParsedAIG(
            circuit_name=circuit_name,
            num_inputs=I,
            num_outputs=O,
            num_and=A,
            max_var=M,
            max_level=max_level,
            node_types=node_types,
            levels=levels,
            edge_index=edge_index,
            and_fanins=and_fanins,
            and_node_indices=and_node_indices,
            po_driver_nodes=po_driver_nodes,
        )


def find_aig_path(circuit_name: str) -> Path:
    """Finds path to circuit's .aig benchmark file in EFPL/benchmarks."""
    root = get_project_root()
    # Search in arithmetic and random_control
    candidates = list(root.glob(f"**/benchmarks/*/{circuit_name}.aig"))
    if not candidates:
        raise FileNotFoundError(f"AIGER file for circuit '{circuit_name}' not found under EFPL/benchmarks/")
    return candidates[0]


def get_cached_graph_path(circuit_name: str) -> Path:
    """Returns destination .npz path under data/processed/graphs/."""
    graphs_dir = get_processed_data_dir() / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir / f"{circuit_name}.npz"


def load_or_parse_aig(circuit_name: str, force_reparse: bool = False) -> ParsedAIG:
    """Loads parsed graph from .npz cache, or parses .aig file and caches as .npz."""
    cache_path = get_cached_graph_path(circuit_name)

    if cache_path.exists() and not force_reparse:
        data = np.load(cache_path)
        return ParsedAIG(
            circuit_name=circuit_name,
            num_inputs=int(data["num_inputs"]),
            num_outputs=int(data["num_outputs"]),
            num_and=int(data["num_and"]),
            max_var=int(data["max_var"]),
            max_level=int(data["max_level"]),
            node_types=data["node_types"],
            levels=data["levels"],
            edge_index=data["edge_index"],
            and_fanins=data["and_fanins"],
            and_node_indices=data["and_node_indices"],
            po_driver_nodes=data["po_driver_nodes"],
        )

    aig_path = find_aig_path(circuit_name)
    parsed = parse_aiger_file(aig_path)

    # Save cache
    np.savez_compressed(
        cache_path,
        num_inputs=parsed.num_inputs,
        num_outputs=parsed.num_outputs,
        num_and=parsed.num_and,
        max_var=parsed.max_var,
        max_level=parsed.max_level,
        node_types=parsed.node_types,
        levels=parsed.levels,
        edge_index=parsed.edge_index,
        and_fanins=parsed.and_fanins,
        and_node_indices=parsed.and_node_indices,
        po_driver_nodes=parsed.po_driver_nodes,
    )
    logger.info(f"Parsed and cached graph for '{circuit_name}' to {cache_path}")
    return parsed
