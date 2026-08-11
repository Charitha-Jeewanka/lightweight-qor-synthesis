"""Unit tests for dataset loader schema validation and data integrity."""

import pytest
import numpy as np
import pandas as pd

from src.data.loaders import load_modeling_table, validate_schema, ModelingDataset, get_expected_dtypes
from src.utils.paths import get_processed_data_dir


def test_load_modeling_table_l10():
    dataset = load_modeling_table(seq_len=10)

    assert isinstance(dataset, ModelingDataset)
    assert len(dataset.df) > 30000
    assert dataset.target_name == "target_and_ratio"

    # Verify zero-variance fanin columns were dropped
    assert "fanin_max" not in dataset.df.columns
    assert "fanin_avg" not in dataset.df.columns

    # Verify feature blocks
    X_all = dataset.get_feature_matrix(encoding="all")
    assert X_all.dtype == np.float32
    assert X_all.ndim == 2
    assert X_all.shape[0] == len(dataset.df)


def test_schema_validation_corrupted_table():
    csv_path = get_processed_data_dir() / "modeling_table_L10.csv"
    raw_df = pd.read_csv(csv_path, dtype=get_expected_dtypes(10), nrows=100)
    bad_df = raw_df.drop(columns=["init_and"])

    with pytest.raises(ValueError, match="missing.*required columns"):
        validate_schema(bad_df, seq_len=10)


def test_schema_validation_float64_rejection():
    csv_path = get_processed_data_dir() / "modeling_table_L10.csv"
    raw_df = pd.read_csv(csv_path, dtype=get_expected_dtypes(10), nrows=100)
    bad_df = raw_df.copy()
    bad_df["init_and"] = bad_df["init_and"].astype(np.float64)  # Forbidden float64!

    with pytest.raises(ValueError, match="Forbidden float64 columns"):
        validate_schema(bad_df, seq_len=10)
