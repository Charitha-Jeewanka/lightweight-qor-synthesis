"""Unit tests for dataset split resolution and leakage prevention."""

import pytest
import numpy as np
import pandas as pd

from src.data.loaders import load_modeling_table
from src.data.splits import resolve_loco_folds, resolve_random_split, SplitResult


def test_random_split_no_leakage():
    dataset = load_modeling_table(seq_len=10)
    split = resolve_random_split(dataset.df, exclude_hyp=True)

    # Check row indices disjoint
    split.assert_disjoint_row_indices()

    # Check run_ids disjoint
    train_ids = set(dataset.df.iloc[split.train_indices]["run_id"])
    val_ids = set(dataset.df.iloc[split.val_indices]["run_id"])
    test_ids = set(dataset.df.iloc[split.test_indices]["run_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_loco_folds_circuit_disjointness():
    dataset = load_modeling_table(seq_len=10)
    folds = resolve_loco_folds(dataset.df, exclude_hyp=True)

    assert len(folds) == 5

    for fold in folds:
        fold.assert_disjoint_circuits()
        fold.assert_disjoint_row_indices()

        # Assert test circuits are strictly disjoint from train and val circuits
        train_val = fold.train_circuits.union(fold.val_circuits)
        assert train_val.isdisjoint(fold.test_circuits)

        # Assert hyp is excluded if requested
        assert "hyp" not in fold.test_circuits
        assert "hyp" not in fold.train_circuits
        assert "hyp" not in fold.val_circuits


def test_leakage_assertion_raises():
    result = SplitResult(
        fold_idx=0,
        train_indices=np.array([0, 1, 2]),
        val_indices=np.array([3, 4]),
        test_indices=np.array([2, 5]),  # 2 overlaps!
        train_circuits={"c1", "c2"},
        val_circuits={"c3"},
        test_circuits={"c4"},
    )

    with pytest.raises(RuntimeError):
        result.assert_disjoint_row_indices()

    result_circuit_leak = SplitResult(
        fold_idx=0,
        train_indices=np.array([0, 1]),
        val_indices=np.array([2]),
        test_indices=np.array([3]),
        train_circuits={"c1", "c2"},
        val_circuits={"c3"},
        test_circuits={"c2"},  # c2 overlaps!
    )

    with pytest.raises(RuntimeError):
        result_circuit_leak.assert_disjoint_circuits()
