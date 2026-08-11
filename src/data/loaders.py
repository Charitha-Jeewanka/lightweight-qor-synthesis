"""Dataset loading, schema validation, and typed feature block extraction.

Enforces strict data contracts per GEMINI.md §4, §4.2, §6.3.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger
from src.utils.paths import get_processed_data_dir

logger = get_logger(__name__)

# Fixed command vocabulary order per GEMINI.md INV-2
VOCAB = [
    "balance",
    "rewrite",
    "rewrite -z",
    "refactor",
    "refactor -z",
    "resub",
    "resub -z",
    "dc2",
]

# Vocabulary names formatted for column headers (e.g., rewrite -z -> rewrite_z)
VOCAB_COL_NAMES = [cmd.replace(" -z", "_z").replace(" ", "_") for cmd in VOCAB]


@dataclass
class ModelingDataset:
    """Typed container holding cleaned modeling table and feature block column lists."""

    df: pd.DataFrame
    seq_len: int
    target_name: str
    circuit_feature_cols: List[str]
    structural_feature_cols: List[str]
    bag_feature_cols: List[str]
    positional_feature_cols: List[str]
    bigram_feature_cols: List[str]

    @property
    def all_feature_cols(self) -> List[str]:
        """Returns concatenated list of all feature columns."""
        return (
            self.circuit_feature_cols
            + self.structural_feature_cols
            + self.bag_feature_cols
            + self.positional_feature_cols
            + self.bigram_feature_cols
        )

    @property
    def sequence_only_feature_cols(self) -> List[str]:
        """Returns sequence encoding features only."""
        return (
            self.bag_feature_cols
            + self.positional_feature_cols
            + self.bigram_feature_cols
        )

    def get_feature_matrix(
        self,
        encoding: str = "all",
        use_circuit_features: bool = True,
        use_structural_features: bool = True,
    ) -> np.ndarray:
        """Slices feature matrix based on requested feature composition.

        Args:
            encoding: 'bag', 'positional', 'bigram', 'bag+positional', or 'all'
            use_circuit_features: whether to include baseline circuit features
            use_structural_features: whether to include extended structural features

        Returns:
            np.ndarray of shape (N, D), dtype float32
        """
        cols: List[str] = []

        if use_circuit_features:
            cols.extend(self.circuit_feature_cols)

        if use_structural_features and self.structural_feature_cols:
            cols.extend(self.structural_feature_cols)

        if encoding == "bag":
            cols.extend(self.bag_feature_cols)
        elif encoding == "positional":
            cols.extend(self.positional_feature_cols)
        elif encoding == "bigram":
            cols.extend(self.bigram_feature_cols)
        elif encoding == "bag+positional":
            cols.extend(self.bag_feature_cols + self.positional_feature_cols)
        elif encoding == "all":
            cols.extend(self.sequence_only_feature_cols)
        else:
            raise ValueError(f"Unknown encoding mode: '{encoding}'")

        if not cols:
            raise ValueError("Feature matrix resolution produced zero columns.")

        return self.df[cols].to_numpy(dtype=np.float32)


def get_expected_dtypes(seq_len: int) -> Dict[str, str]:
    """Generates expected explicit dtype mapping for modeling table loading."""
    dtypes: Dict[str, str] = {
        "run_id": "string",
        "category": "string",
        "circuit": "string",
        "seq_type": "string",
        "seq_len": "int32",
        "seq_str": "string",
        "init_and": "int32",
        "init_lev": "int32",
        "final_and": "int32",
        "final_lev": "int32",
        "runtime_s": "float32",
        "pi": "int32",
        "po": "int32",
        "latches": "int32",
        "and_count": "int32",
        "level_count": "int32",
        "fanin_max": "int32",
        "fanin_avg": "float32",
        "fanout_avg": "float32",
        "fanout_max": "int32",
        "edges": "int32",
        "fold": "int32",
        "split_random": "string",
    }

    # Bag-of-commands (8 columns)
    for cmd in VOCAB_COL_NAMES:
        dtypes[f"cnt_{cmd}"] = "int16"

    # Positional one-hot (seq_len * 8 columns)
    for pos in range(seq_len):
        for cmd in VOCAB_COL_NAMES:
            dtypes[f"pos{pos}_{cmd}"] = "int8"

    # Bigram counts (64 columns)
    for cmd1 in VOCAB_COL_NAMES:
        for cmd2 in VOCAB_COL_NAMES:
            dtypes[f"bg_{cmd1}__{cmd2}"] = "int16"

    # Targets
    target_cols = [
        "target_and_ratio",
        "target_lev_ratio",
        "target_and_raw",
        "target_lev_raw",
        "target_and_logratio",
        "target_lev_logratio",
        "target_and_delta",
        "target_lev_delta",
    ]
    for col in target_cols:
        dtypes[col] = "float32"

    return dtypes


def validate_schema(df: pd.DataFrame, seq_len: int) -> None:
    """Validates dataframe columns, widths, and types against §4.1 schema contracts."""
    expected_dtypes = get_expected_dtypes(seq_len)

    # 1. Missing columns check
    missing_cols = [col for col in expected_dtypes if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Schema validation failed: missing {len(missing_cols)} required columns: {missing_cols[:5]}..."
        )

    # 2. Sequence encoding width checks
    bag_cols = [c for c in df.columns if c.startswith("cnt_")]
    pos_cols = [c for c in df.columns if c.startswith("pos")]
    bg_cols = [c for c in df.columns if c.startswith("bg_")]

    if len(bag_cols) != 8:
        raise ValueError(f"Schema mismatch: expected 8 bag-of-commands columns, found {len(bag_cols)}")
    if len(pos_cols) != seq_len * 8:
        raise ValueError(
            f"Schema mismatch: expected {seq_len * 8} positional columns for L={seq_len}, found {len(pos_cols)}"
        )
    if len(bg_cols) != 64:
        raise ValueError(f"Schema mismatch: expected 64 bigram columns, found {len(bg_cols)}")

    # 3. Check for float64 columns (strictly forbidden per §6.3)
    float64_cols = df.select_dtypes(include=["float64"]).columns.tolist()
    if float64_cols:
        raise ValueError(f"Forbidden float64 columns found in modeling table: {float64_cols}")

    logger.info(f"Schema validation PASSED for modeling table (L={seq_len}, {len(df)} rows)")


def assert_data_quirks(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    """Asserts and handles known data quirks per §4.2.

    1. `hyp` row count check.
    2. L=5 circuit row count logging.
    3. `fanin_max == 2` and `fanin_avg == 2.0` zero-variance assertion, followed by dropping them.
    4. Verification of `final_and > init_and` cases.
    """
    # 1. hyp row count check
    if "hyp" in df["circuit"].values:
        hyp_rows = (df["circuit"] == "hyp").sum()
        expected_hyp = 55 if seq_len == 15 else 100
        if hyp_rows == expected_hyp:
            logger.info(f"Data quirk check: 'hyp' has expected row count of {hyp_rows} for L={seq_len}")
        else:
            logger.warning(
                f"Data quirk note: 'hyp' has {hyp_rows} rows (expected {expected_hyp} for L={seq_len})"
            )

    # 2. L=5 circuit row counts
    if seq_len == 5:
        row_counts = df.groupby("circuit").size()
        min_rows, max_rows = row_counts.min(), row_counts.max()
        logger.info(
            f"Data quirk check: L=5 circuit row counts span [{min_rows}, {max_rows}] "
            "(birthday-paradox dedup behavior per §4.2)"
        )

    # 3. Zero-variance fanin columns check & drop
    if "fanin_max" in df.columns and "fanin_avg" in df.columns:
        fanin_max_unique = df["fanin_max"].unique()
        fanin_avg_unique = df["fanin_avg"].unique()

        if not (len(fanin_max_unique) == 1 and fanin_max_unique[0] == 2):
            raise ValueError(f"Data integrity failure: fanin_max is not uniformly 2! Got: {fanin_max_unique}")
        if not (len(fanin_avg_unique) == 1 and np.isclose(fanin_avg_unique[0], 2.0)):
            raise ValueError(f"Data integrity failure: fanin_avg is not uniformly 2.0! Got: {fanin_avg_unique}")

        logger.info("Data quirk check: fanin_max==2 and fanin_avg==2.0 verified. Dropping both zero-variance columns.")
        df = df.drop(columns=["fanin_max", "fanin_avg"])

    # 4. Verify area expansion runs exist
    expanded_runs = (df["final_and"] > df["init_and"]).sum()
    logger.info(
        f"Data quirk check: verified {expanded_runs} runs ({expanded_runs/len(df):.2%}) "
        "where final_and > init_and (legitimate synthesis outcomes retained)"
    )

    return df


def load_modeling_table(
    seq_len: int = 10,
    target: str = "target_and_ratio",
    processed_dir: Optional[Path] = None,
) -> ModelingDataset:
    """Loads and validates a modeling table dataset for a specified sequence length L.

    Args:
        seq_len: 5, 10, or 15
        target: primary or secondary target column name
        processed_dir: path to processed data directory (defaults to data/processed/)

    Returns:
        ModelingDataset containing typed feature blocks and validated DataFrame
    """
    if seq_len not in (5, 10, 15):
        raise ValueError(f"Invalid sequence length L={seq_len}. Must be 5, 10, or 15.")

    if processed_dir is None:
        processed_dir = get_processed_data_dir()

    csv_path = processed_dir / f"modeling_table_L{seq_len}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Modeling table CSV not found: {csv_path}")

    logger.info(f"Loading modeling table from {csv_path}...")

    dtypes = get_expected_dtypes(seq_len)
    df = pd.read_csv(csv_path, dtype=dtypes)

    # Validate schema
    validate_schema(df, seq_len)

    # Check target existence
    if target not in df.columns:
        raise ValueError(f"Requested target '{target}' not found in modeling table columns.")

    # Process quirks and drop zero-variance fanin columns
    df = assert_data_quirks(df, seq_len)

    # Build feature column blocks
    circuit_feature_cols = [
        "pi",
        "po",
        "latches",
        "and_count",
        "level_count",
        "fanout_avg",
        "fanout_max",
        "edges",
    ]

    structural_feature_cols: List[str] = []
    ext_csv_path = processed_dir / "circuit_features_extended.csv"
    if ext_csv_path.exists():
        logger.info(f"Joining extended structural features from {ext_csv_path}")
        ext_df = pd.read_csv(ext_csv_path)
        ext_cols = [c for c in ext_df.columns if c != "circuit"]
        df = df.merge(ext_df, on="circuit", how="left")
        structural_feature_cols = ext_cols
    else:
        logger.info(
            "Phase 3 extended structural features (circuit_features_extended.csv) not found; "
            "structural_feature_cols will be empty."
        )

    bag_feature_cols = [f"cnt_{cmd}" for cmd in VOCAB_COL_NAMES]
    positional_feature_cols = [
        f"pos{p}_{cmd}" for p in range(seq_len) for cmd in VOCAB_COL_NAMES
    ]
    bigram_feature_cols = [
        f"bg_{c1}__{c2}" for c1 in VOCAB_COL_NAMES for c2 in VOCAB_COL_NAMES
    ]

    return ModelingDataset(
        df=df,
        seq_len=seq_len,
        target_name=target,
        circuit_feature_cols=circuit_feature_cols,
        structural_feature_cols=structural_feature_cols,
        bag_feature_cols=bag_feature_cols,
        positional_feature_cols=positional_feature_cols,
        bigram_feature_cols=bigram_feature_cols,
    )
