"""Split resolution for random split and leave-circuits-out (LOCO) protocols.

Strictly deterministic resolution from frozen split columns per GEMINI.md INV-1 & INV-3.
This module contains NO randomness.
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SplitResult:
    """Container holding integer row index arrays for train, val, and test splits."""

    fold_idx: int  # 0 for random split, 0..4 for LOCO folds
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_circuits: Set[str]
    val_circuits: Set[str]
    test_circuits: Set[str]

    def assert_disjoint_circuits(self) -> None:
        """Asserts that test circuits are strictly disjoint from train and val circuits."""
        train_val = self.train_circuits.union(self.val_circuits)
        overlap = train_val.intersection(self.test_circuits)
        if overlap:
            raise RuntimeError(
                f"LEAKAGE DETECTED in fold {self.fold_idx}: "
                f"Circuits present in both train/val and test splits: {overlap}"
            )

    def assert_disjoint_row_indices(self) -> None:
        """Asserts that no row index appears in multiple splits."""
        set_tr = set(self.train_indices)
        set_va = set(self.val_indices)
        set_te = set(self.test_indices)

        tr_va = set_tr.intersection(set_va)
        tr_te = set_tr.intersection(set_te)
        va_te = set_va.intersection(set_te)

        if tr_va or tr_te or va_te:
            raise RuntimeError(
                f"LEAKAGE DETECTED: Row index overlap between splits! "
                f"tr_va={len(tr_va)}, tr_te={len(tr_te)}, va_te={len(va_te)}"
            )


def resolve_random_split(
    df: pd.DataFrame,
    exclude_hyp: bool = False,
) -> SplitResult:
    """Resolves the fixed random split (train/val/test) from the modeling table."""
    if "split_random" not in df.columns:
        raise ValueError("DataFrame missing required 'split_random' column.")

    mask = np.ones(len(df), dtype=bool)
    if exclude_hyp:
        mask = mask & (df["circuit"] != "hyp").to_numpy()

    indices = np.arange(len(df))

    train_mask = mask & (df["split_random"] == "train").to_numpy()
    val_mask = mask & (df["split_random"] == "val").to_numpy()
    test_mask = mask & (df["split_random"] == "test").to_numpy()

    train_idx = indices[train_mask]
    val_idx = indices[val_mask]
    test_idx = indices[test_mask]

    train_circuits = set(df.iloc[train_idx]["circuit"].unique())
    val_circuits = set(df.iloc[val_idx]["circuit"].unique())
    test_circuits = set(df.iloc[test_idx]["circuit"].unique())

    result = SplitResult(
        fold_idx=0,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        train_circuits=train_circuits,
        val_circuits=val_circuits,
        test_circuits=test_circuits,
    )
    result.assert_disjoint_row_indices()
    return result


def resolve_loco_folds(
    df: pd.DataFrame,
    exclude_hyp: bool = True,
    include_val_fold: bool = False,
) -> List[SplitResult]:
    """Resolves 5 leave-circuits-out (LOCO) cross-validation folds.

    For outer fold k:
      - Test circuits: fold == k
      - Train circuits: fold != k (all remaining 4 folds, 15-16 circuits)
      - Val circuits: empty (or fold == (k + 1) % 5 if include_val_fold is True)

    Args:
        df: cleaned modeling table DataFrame
        exclude_hyp: whether to exclude 'hyp' circuit (default True per §4.2)
        include_val_fold: whether to separate an inner validation fold for hyperparameter selection

    Returns:
        List of 5 SplitResult objects, one per outer fold
    """
    if "fold" not in df.columns:
        raise ValueError("DataFrame missing required 'fold' column.")

    folds_results: List[SplitResult] = []
    indices = np.arange(len(df))

    base_mask = np.ones(len(df), dtype=bool)
    if exclude_hyp:
        base_mask = base_mask & (df["circuit"] != "hyp").to_numpy()

    for fold_k in range(5):
        test_mask = base_mask & (df["fold"] == fold_k).to_numpy()

        if include_val_fold:
            val_fold_idx = (fold_k + 1) % 5
            val_mask = base_mask & (df["fold"] == val_fold_idx).to_numpy()
            train_mask = base_mask & (~(df["fold"].isin([fold_k, val_fold_idx]))).to_numpy()
        else:
            val_mask = np.zeros(len(df), dtype=bool)
            train_mask = base_mask & (df["fold"] != fold_k).to_numpy()

        train_idx = indices[train_mask]
        val_idx = indices[val_mask]
        test_idx = indices[test_mask]

        train_circuits = set(df.iloc[train_idx]["circuit"].unique())
        val_circuits = set(df.iloc[val_idx]["circuit"].unique()) if len(val_idx) > 0 else set()
        test_circuits = set(df.iloc[test_idx]["circuit"].unique())

        result = SplitResult(
            fold_idx=fold_k,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            train_circuits=train_circuits,
            val_circuits=val_circuits,
            test_circuits=test_circuits,
        )

        # Enforce leak prevention invariant (INV-3)
        result.assert_disjoint_circuits()
        result.assert_disjoint_row_indices()

        total_circuits = len(train_circuits) + len(val_circuits) + len(test_circuits)
        expected_total = 19 if exclude_hyp else 20
        if total_circuits != expected_total:
            raise RuntimeError(
                f"Circuit count discrepancy in fold {fold_k}: "
                f"Total circuits = {total_circuits} (expected {expected_total})"
            )

        folds_results.append(result)

    logger.info(
        f"Resolved {len(folds_results)} LOCO folds (exclude_hyp={exclude_hyp}, include_val_fold={include_val_fold}). "
        "All disjointness assertions passed."
    )
    return folds_results
