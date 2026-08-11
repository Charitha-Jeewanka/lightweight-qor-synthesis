"""Extraction of six mechanism-aligned structural circuit features.

Enforces strict topological algorithms, budget guards, memory defenses, and range assertions
per GEMINI.md §7.2, §7.3, §7.4, and §7.5.
All traversals are strictly ITERATIVE with explicit stacks (no recursion).
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np

from src.data.aiger import ParsedAIG, load_or_parse_aig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class StructuralFeatures:
    """Dataclass holding the six mechanism-aligned structural features for a single circuit."""

    circuit: str
    mffc_mean: float
    mffc_large_frac: float
    multifanout_frac: float
    cut4_mean: float
    critical_cone_frac: float
    balance_ratio: float
    mffc_sampled: bool = False
    cut4_truncated: bool = False
    extraction_time_s: float = 0.0


def compute_fanouts(parsed: ParsedAIG) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """Computes fanout counts array and fanout adjacency lists.

    Returns:
        refs: int32 array of shape [num_nodes], containing fanout counts
        fanouts_dict: map of node_idx -> list of successor node_idxs
    """
    num_nodes = len(parsed.node_types)
    refs = np.zeros(num_nodes, dtype=np.int32)
    fanouts_dict: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}

    edge_src = parsed.edge_index[0]
    edge_dst = parsed.edge_index[1]

    for src, dst in zip(edge_src, edge_dst):
        refs[src] += 1
        fanouts_dict[src].append(dst)

    return refs, fanouts_dict


def compute_mffc_features(
    parsed: ParsedAIG,
    refs: np.ndarray,
    time_budget_s: float = 600.0,
    seed: int = 42,
) -> Tuple[float, float, bool]:
    """Computes mffc_mean and mffc_large_frac via iterative reference counting.

    Asserts bit-identical ref count restoration after processing each node.
    Enforces pure time budget guard (structural_time_budget_s) with seeded sampling fallback.
    """
    initial_refs = refs.copy()
    and_indices = parsed.and_node_indices
    num_and = len(and_indices)

    if num_and == 0:
        return 1.0, 0.0, False

    t0 = time.perf_counter()
    mffc_sampled = False
    sample_indices = and_indices

    mffc_sizes: List[int] = []
    and_fanins = parsed.and_fanins
    I = parsed.num_inputs

    # Execute MFFC reference-counting
    for i, node_n in enumerate(and_indices):
        # Time budget check per node batch
        if i % 5000 == 0 and (time.perf_counter() - t0 > time_budget_s) and not mffc_sampled:
            rng = np.random.default_rng(seed)
            sample_size = min(50000, num_and)
            sample_indices = rng.choice(and_indices, size=sample_size, replace=False)
            mffc_sampled = True
            logger.info(
                f"MFFC time budget guard triggered (> {time_budget_s}s) for '{parsed.circuit_name}'; "
                f"switching to seeded random sample of {sample_size} nodes (seed={seed})."
            )
            # Restart MFFC calculation on sample
            mffc_sizes.clear()
            np.copyto(refs, initial_refs)
            break

        # 1. Deref
        count = 1
        stack = [node_n]

        while stack:
            u = stack.pop()
            if u < I or u >= I + num_and:
                continue  # Only deref fanins of AND nodes
            u_and_idx = u - I
            f0 = int(and_fanins[u_and_idx, 0])
            f1 = int(and_fanins[u_and_idx, 1])

            for f in (f0, f1):
                if f >= 0:
                    refs[f] -= 1
                    if refs[f] == 0 and f >= I and f < I + num_and:
                        count += 1
                        stack.append(f)

        mffc_sizes.append(count)

        # 2. Reref (exact mirror image to restore refs)
        stack = [node_n]
        while stack:
            u = stack.pop()
            if u < I or u >= I + num_and:
                continue
            u_and_idx = u - I
            f0 = int(and_fanins[u_and_idx, 0])
            f1 = int(and_fanins[u_and_idx, 1])

            for f in (f0, f1):
                if f >= 0:
                    if refs[f] == 0 and f >= I and f < I + num_and:
                        stack.append(f)
                    refs[f] += 1

    # If fallback sample was triggered, execute on sampled indices
    if mffc_sampled:
        for node_n in sample_indices:
            count = 1
            stack = [node_n]
            while stack:
                u = stack.pop()
                if u < I or u >= I + num_and:
                    continue
                u_and_idx = u - I
                f0 = int(and_fanins[u_and_idx, 0])
                f1 = int(and_fanins[u_and_idx, 1])
                for f in (f0, f1):
                    if f >= 0:
                        refs[f] -= 1
                        if refs[f] == 0 and f >= I and f < I + num_and:
                            count += 1
                            stack.append(f)
            mffc_sizes.append(count)
            stack = [node_n]
            while stack:
                u = stack.pop()
                if u < I or u >= I + num_and:
                    continue
                u_and_idx = u - I
                f0 = int(and_fanins[u_and_idx, 0])
                f1 = int(and_fanins[u_and_idx, 1])
                for f in (f0, f1):
                    if f >= 0:
                        if refs[f] == 0 and f >= I and f < I + num_and:
                            stack.append(f)
                        refs[f] += 1

    # 3. Assert bit-identical ref count restoration
    if not np.array_equal(refs, initial_refs):
        raise RuntimeError(
            f"Deref/Reref asymmetry corruption detected in circuit '{parsed.circuit_name}'! "
            "Ref counts were not restored to bit-identical state."
        )

    mffc_mean = float(np.mean(mffc_sizes))
    large_count = sum(1 for sz in mffc_sizes if sz > 4)
    mffc_large_frac = float(large_count / len(mffc_sizes))

    return mffc_mean, mffc_large_frac, mffc_sampled


def filter_dominated_cuts(cuts: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Removes dominated cuts from candidate list.

    A cut c1 is dominated by c2 if c2 is a strict subset of c1.
    """
    if len(cuts) <= 1:
        return cuts

    sets = [set(c) for c in cuts]
    retained: List[Tuple[int, ...]] = []

    for i, (c1_tuple, s1) in enumerate(zip(cuts, sets)):
        dominated = False
        for j, (c2_tuple, s2) in enumerate(zip(cuts, sets)):
            if i != j and s2 < s1:
                dominated = True
                break
        if not dominated:
            retained.append(c1_tuple)

    return retained


def compute_cut4_features(
    parsed: ParsedAIG,
    refs: np.ndarray,
    time_budget_s: float = 600.0,
    cap_C: int = 6,
) -> Tuple[float, bool]:
    """Computes cut4_mean via bottom-up cut enumeration with cap C=6.

    Uses sorted int32 tuples, frees cuts in topological order once all fanouts
    are processed (memory guard for hyp), and enforces time budget guard.
    """
    t0 = time.perf_counter()
    num_and = parsed.num_and
    if num_and == 0:
        return 1.0, False

    cut4_truncated = False
    I = parsed.num_inputs

    cuts_dict: Dict[int, List[Tuple[int, ...]]] = {}
    remaining_fanouts = refs.copy()

    for p in range(I):
        cuts_dict[p] = [(p,)]

    total_cut_counts: List[int] = []
    and_fanins = parsed.and_fanins

    for node_idx in range(I, I + num_and):
        if (node_idx - I) % 5000 == 0 and (time.perf_counter() - t0 > time_budget_s):
            cut4_truncated = True
            logger.info(
                f"Cut4 enumeration budget guard triggered (> {time_budget_s}s) for '{parsed.circuit_name}'; "
                f"truncating cut computation at node {node_idx - I}/{num_and}."
            )
            break

        u_and_idx = node_idx - I
        f0 = int(and_fanins[u_and_idx, 0])
        f1 = int(and_fanins[u_and_idx, 1])

        c0_list = cuts_dict.get(f0, [(f0,)]) if f0 >= 0 else [()]
        c1_list = cuts_dict.get(f1, [(f1,)]) if f1 >= 0 else [()]

        node_cuts_set: Set[Tuple[int, ...]] = {(node_idx,)}

        for c0 in c0_list:
            for c1 in c1_list:
                combined = tuple(sorted(set(c0).union(c1)))
                if 1 <= len(combined) <= 4:
                    node_cuts_set.add(combined)

        retained = filter_dominated_cuts(list(node_cuts_set))

        retained.sort(key=lambda c: (len(c), c))
        capped_cuts = retained[:cap_C]

        cuts_dict[node_idx] = capped_cuts
        total_cut_counts.append(len(capped_cuts))

        for f in (f0, f1):
            if f >= 0:
                remaining_fanouts[f] -= 1
                if remaining_fanouts[f] == 0 and f in cuts_dict:
                    del cuts_dict[f]

    if not total_cut_counts:
        return 1.0, cut4_truncated

    cut4_mean = float(np.mean(total_cut_counts))
    return cut4_mean, cut4_truncated


def extract_circuit_structural_features(
    circuit_name: str,
    time_budget_s: float = 600.0,
    seed: int = 42,
) -> StructuralFeatures:
    """Extracts the six frozen structural features for a circuit.

    Args:
        circuit_name: circuit identifier
        time_budget_s: maximum time budget in seconds (default 600)
        seed: random seed for sampling fallback

    Returns:
        StructuralFeatures dataclass instance
    """
    t_start = time.perf_counter()
    parsed = load_or_parse_aig(circuit_name)

    refs, _ = compute_fanouts(parsed)
    and_indices = parsed.and_node_indices
    num_and = len(and_indices)

    if num_and == 0:
        return StructuralFeatures(
            circuit=circuit_name,
            mffc_mean=1.0,
            mffc_large_frac=0.0,
            multifanout_frac=0.0,
            cut4_mean=1.0,
            critical_cone_frac=0.0,
            balance_ratio=1.0,
            extraction_time_s=time.perf_counter() - t_start,
        )

    # 1. mffc_mean, mffc_large_frac
    mffc_mean, mffc_large_frac, mffc_sampled = compute_mffc_features(
        parsed, refs, time_budget_s=time_budget_s, seed=seed
    )

    # 2. multifanout_frac
    and_refs = refs[and_indices]
    multifanout_count = sum(1 for r in and_refs if r >= 2)
    multifanout_frac = float(multifanout_count / num_and)

    # 3. cut4_mean
    cut4_mean, cut4_truncated = compute_cut4_features(
        parsed, refs, time_budget_s=time_budget_s
    )

    # 4. critical_cone_frac
    max_level = parsed.max_level
    and_levels = parsed.levels[and_indices]
    if max_level > 0:
        crit_count = sum(1 for lvl in and_levels if lvl >= 0.9 * max_level)
        critical_cone_frac = float(crit_count / num_and)
    else:
        critical_cone_frac = 0.0

    # 5. balance_ratio
    pi = parsed.num_inputs
    min_possible_depth = math.ceil(math.log2(pi)) if pi > 1 else 1.0
    balance_ratio = float(max_level / min_possible_depth) if min_possible_depth > 0 else 1.0

    t_end = time.perf_counter()
    extraction_time_s = t_end - t_start

    features = StructuralFeatures(
        circuit=circuit_name,
        mffc_mean=mffc_mean,
        mffc_large_frac=mffc_large_frac,
        multifanout_frac=multifanout_frac,
        cut4_mean=cut4_mean,
        critical_cone_frac=critical_cone_frac,
        balance_ratio=balance_ratio,
        mffc_sampled=mffc_sampled,
        cut4_truncated=cut4_truncated,
        extraction_time_s=extraction_time_s,
    )

    validate_structural_features(features)

    return features


def validate_structural_features(f: StructuralFeatures) -> None:
    """Asserts all validation checks from §7.4 on extracted features."""
    if f.mffc_mean < 1.0:
        raise ValueError(f"Validation failure for '{f.circuit}': mffc_mean ({f.mffc_mean}) < 1.0")

    if f.mffc_mean > 50.0:
        raise ValueError(
            f"Validation failure for '{f.circuit}': mffc_mean ({f.mffc_mean}) is implausibly high (> 50.0)"
        )

    for name, val in [
        ("mffc_large_frac", f.mffc_large_frac),
        ("multifanout_frac", f.multifanout_frac),
        ("critical_cone_frac", f.critical_cone_frac),
    ]:
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Validation failure for '{f.circuit}': {name} ({val}) outside [0.0, 1.0]")

    if f.cut4_mean < 1.0:
        raise ValueError(f"Validation failure for '{f.circuit}': cut4_mean ({f.cut4_mean}) < 1.0")

    if f.balance_ratio < 1.0:
        raise ValueError(f"Validation failure for '{f.circuit}': balance_ratio ({f.balance_ratio}) < 1.0")

    vals = [
        f.mffc_mean,
        f.mffc_large_frac,
        f.multifanout_frac,
        f.cut4_mean,
        f.critical_cone_frac,
        f.balance_ratio,
    ]
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        raise ValueError(f"Validation failure for '{f.circuit}': Contains NaN or Inf values: {f}")
