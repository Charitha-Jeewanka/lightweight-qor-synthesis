#!/usr/bin/env python3
"""
Dataset assembly for the modeling phase of "Lightweight Learning-Based QoR
Prediction for Logic Synthesis Optimization Sequences" (Sections 5.3-5.4).

Takes the raw QoR CSVs from generate_data.py plus circuit_features.csv from
extract_features.py, and produces:

  1. A single flat modeling table with all engineered features, all sequence
     encodings, and all target variants as columns.
  2. A leave-circuits-out fold assignment (stratified across arithmetic /
     random_control), written once and reused by every model family so the
     RQ1 comparison is controlled.
  3. A random-split assignment over (circuit, sequence) pairs.

Usage:
    python prepare_dataset.py --qor data/qor_dataset_L10.csv \
        --features data/circuit_features.csv \
        --seq-len 10 \
        --outdir data/processed

    # sensitivity datasets (reuse the SAME loco folds for comparability)
    python prepare_dataset.py --qor data/qor_dataset_L5.csv \
        --features data/circuit_features.csv --seq-len 5 \
        --outdir data/processed --folds data/processed/loco_folds.csv

Design notes:
- VOCAB order is copied verbatim from generate_data.py. Do not reorder: the
  positional one-hot column order depends on it, and reordering silently
  invalidates any previously trained model.
- Reference-recipe rows (resyn2, resyn2rs) have a different length than L, so
  they are excluded from the modeling table by default and written to a
  separate file. They are evaluation reference points, not training data.
- Targets are emitted in four forms (raw, ratio, log-ratio, delta) so the
  choice can be made empirically rather than assumed.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Command vocabulary -- MUST match generate_data.py exactly
# ---------------------------------------------------------------------------
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

VOCAB_INDEX = {cmd: i for i, cmd in enumerate(VOCAB)}


def slug(cmd: str) -> str:
    """'rewrite -z' -> 'rewrite_z' (safe as a column name)."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", cmd).strip("_")


SLUGS = [slug(c) for c in VOCAB]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def find_circuit_column(df: pd.DataFrame) -> str:
    """circuit_features.csv column naming isn't pinned down; detect it."""
    for candidate in ("circuit", "circuit_name", "name", "design"):
        if candidate in df.columns:
            return candidate
    sys.exit(
        f"ERROR: could not find a circuit-name column in the features file. "
        f"Columns present: {list(df.columns)}"
    )


def parse_sequence(seq_str: str):
    """'balance; rewrite -z; dc2' -> ['balance', 'rewrite -z', 'dc2']"""
    return [tok.strip() for tok in seq_str.split(";") if tok.strip()]


# ---------------------------------------------------------------------------
# Sequence encodings (Section 5.3)
# ---------------------------------------------------------------------------
def encode_bag(seqs, ):
    """Bag-of-commands counts: 8 columns."""
    out = np.zeros((len(seqs), len(VOCAB)), dtype=np.int16)
    for r, seq in enumerate(seqs):
        for cmd in seq:
            out[r, VOCAB_INDEX[cmd]] += 1
    cols = [f"cnt_{s}" for s in SLUGS]
    return pd.DataFrame(out, columns=cols)


def encode_positional(seqs, seq_len):
    """Positional one-hot: seq_len * 8 columns. Rows shorter than seq_len are
    zero-padded at the tail; rows longer are truncated (both flagged upstream)."""
    out = np.zeros((len(seqs), seq_len * len(VOCAB)), dtype=np.int8)
    for r, seq in enumerate(seqs):
        for p, cmd in enumerate(seq[:seq_len]):
            out[r, p * len(VOCAB) + VOCAB_INDEX[cmd]] = 1
    cols = [f"pos{p}_{s}" for p in range(seq_len) for s in SLUGS]
    return pd.DataFrame(out, columns=cols)


def encode_bigrams(seqs):
    """Adjacent-pair counts: 64 columns. Captures order effects that
    bag-of-commands discards (e.g. 'rewrite then balance' vs the reverse)."""
    n = len(VOCAB)
    out = np.zeros((len(seqs), n * n), dtype=np.int16)
    for r, seq in enumerate(seqs):
        for a, b in zip(seq[:-1], seq[1:]):
            out[r, VOCAB_INDEX[a] * n + VOCAB_INDEX[b]] += 1
    cols = [f"bg_{SLUGS[i]}__{SLUGS[j]}" for i in range(n) for j in range(n)]
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def build_loco_folds(circuit_category: pd.DataFrame, n_folds: int, seed: int):
    """Leave-circuits-out folds, stratified so each fold holds out an equal
    number of arithmetic and random_control circuits (Section 5.4a)."""
    rng = np.random.default_rng(seed)
    assignments = {}
    for category, group in circuit_category.groupby("category"):
        names = sorted(group["circuit"].tolist())
        rng.shuffle(names)
        for i, name in enumerate(names):
            assignments[name] = i % n_folds
    folds = pd.DataFrame(
        sorted(assignments.items()), columns=["circuit", "fold"]
    )
    return folds.merge(circuit_category, on="circuit", how="left")


def assign_random_split(n_rows, seed, val_frac=0.15, test_frac=0.15):
    """Random split over (circuit, sequence) pairs (Section 5.4b)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n_rows)
    split = np.full(n_rows, "train", dtype=object)
    split[u >= 1.0 - test_frac] = "test"
    split[(u >= 1.0 - test_frac - val_frac) & (u < 1.0 - test_frac)] = "val"
    return split


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qor", required=True, help="Path to qor_dataset_L*.csv")
    ap.add_argument("--features", required=True, help="Path to circuit_features.csv")
    ap.add_argument("--seq-len", type=int, required=True, help="Sequence length L")
    ap.add_argument("--outdir", default="data/processed", help="Output directory")
    ap.add_argument("--folds", default=None,
                    help="Reuse an existing loco_folds.csv instead of generating new folds")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-circuits", nargs="*", default=None,
                    help="Circuits to drop entirely, e.g. --exclude-circuits hyp")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.qor).stem.replace("qor_dataset_", "")

    # --- load -------------------------------------------------------------
    qor = pd.read_csv(args.qor)
    feats = pd.read_csv(args.features)
    print(f"Loaded {len(qor)} QoR rows and {len(feats)} circuit feature rows.")

    fcol = find_circuit_column(feats)
    feats = feats.rename(columns={fcol: "circuit"})
    # avoid duplicating columns that already exist in the QoR table
    drop_dupes = [c for c in feats.columns if c in qor.columns and c != "circuit"]
    if drop_dupes:
        print(f"  (dropping duplicated feature columns already in QoR table: {drop_dupes})")
        feats = feats.drop(columns=drop_dupes)

    # --- separate reference recipes --------------------------------------
    is_ref = qor["seq_type"] == "reference"
    ref_rows = qor[is_ref].copy()
    df = qor[~is_ref].copy().reset_index(drop=True)
    print(f"  {len(ref_rows)} reference rows held out; {len(df)} random-sequence rows retained.")

    if args.exclude_circuits:
        before = len(df)
        df = df[~df["circuit"].isin(args.exclude_circuits)].reset_index(drop=True)
        print(f"  Excluded {args.exclude_circuits}: {before} -> {len(df)} rows.")

    # --- parse and validate sequences ------------------------------------
    seqs = [parse_sequence(s) for s in df["seq_str"]]
    unknown = {c for seq in seqs for c in seq} - set(VOCAB)
    if unknown:
        sys.exit(f"ERROR: sequence tokens not in VOCAB: {sorted(unknown)}")
    lengths = np.array([len(s) for s in seqs])
    if (lengths != args.seq_len).any():
        n_bad = int((lengths != args.seq_len).sum())
        print(f"  WARNING: {n_bad} row(s) have length != {args.seq_len} "
              f"(range {lengths.min()}-{lengths.max()}); they will be padded/truncated.")

    # --- build feature blocks --------------------------------------------
    enc = pd.concat(
        [encode_bag(seqs), encode_positional(seqs, args.seq_len), encode_bigrams(seqs)],
        axis=1,
    )
    df = pd.concat([df, enc], axis=1)
    df = df.merge(feats, on="circuit", how="left")

    missing_feats = df[feats.columns.drop("circuit")].isna().any(axis=1).sum()
    if missing_feats:
        print(f"  WARNING: {missing_feats} row(s) failed to match a circuit in the features file.")

    # --- targets ----------------------------------------------------------
    # Raw: comparable to published OpenABC-D / LOSTIN numbers.
    # Ratio: the primary target -- removes between-circuit scale, so the model
    #        must learn the sequence effect rather than echoing circuit size.
    df["target_and_raw"] = df["final_and"]
    df["target_lev_raw"] = df["final_lev"]
    df["target_and_ratio"] = df["final_and"] / df["init_and"]
    df["target_lev_ratio"] = df["final_lev"] / df["init_lev"].replace(0, np.nan)
    df["target_and_logratio"] = np.log(df["target_and_ratio"])
    df["target_lev_logratio"] = np.log(df["target_lev_ratio"])
    df["target_and_delta"] = df["init_and"] - df["final_and"]
    df["target_lev_delta"] = df["init_lev"] - df["final_lev"]

    # --- splits -----------------------------------------------------------
    circuit_category = (
        df[["circuit", "category"]].drop_duplicates().sort_values("circuit").reset_index(drop=True)
    )
    if args.folds:
        folds = pd.read_csv(args.folds)[["circuit", "fold"]]
        print(f"  Reusing folds from {args.folds}")
    else:
        folds = build_loco_folds(circuit_category, args.n_folds, args.seed)
        fold_path = outdir / "loco_folds.csv"
        folds.to_csv(fold_path, index=False)
        print(f"  Wrote leave-circuits-out folds to {fold_path}")
    df = df.merge(folds[["circuit", "fold"]], on="circuit", how="left")
    df["split_random"] = assign_random_split(len(df), args.seed)

    # --- write ------------------------------------------------------------
    out_path = outdir / f"modeling_table_{tag}.csv"
    df.to_csv(out_path, index=False)
    ref_path = outdir / f"reference_recipes_{tag}.csv"
    ref_rows.to_csv(ref_path, index=False)

    # --- summary ----------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"  Rows: {len(df)}   Columns: {len(df.columns)}   Circuits: {df['circuit'].nunique()}")
    print(f"  Encoding widths: bag={len(VOCAB)}  positional={args.seq_len * len(VOCAB)}  bigram={len(VOCAB)**2}")
    print(f"  Random split: " + ", ".join(
        f"{k}={v}" for k, v in df['split_random'].value_counts().items()))
    print("  Leave-circuits-out folds:")
    for f, grp in folds.groupby("fold"):
        print(f"    fold {f}: {', '.join(sorted(grp['circuit']))}")
    print("\n  Target spread (area ratio, per circuit):")
    spread = df.groupby("circuit")["target_and_ratio"].agg(["min", "median", "max"])
    print(spread.round(4).to_string().replace("\n", "\n  "))
    print(f"\n  Wrote {out_path}")
    print(f"  Wrote {ref_path}")


if __name__ == "__main__":
    main()