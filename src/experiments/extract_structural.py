"""Driver script for extracting structural features for all 20 EPFL circuits.

Outputs data/processed/circuit_features_extended.csv.
Meets all requirements from GEMINI.md §7.
"""

import time
from pathlib import Path
import pandas as pd

from src.data.structural import extract_circuit_structural_features
from src.utils.logging_utils import get_logger
from src.utils.paths import get_processed_data_dir, get_project_root

logger = get_logger(__name__)


def run_extraction() -> pd.DataFrame:
    """Extracts structural features for all 20 circuits and saves to CSV."""
    circuit_features_path = get_project_root() / "data" / "circuit_features.csv"
    circuit_df = pd.read_csv(circuit_features_path)
    circuits = circuit_df["circuit"].tolist()

    logger.info(f"Starting structural feature extraction for {len(circuits)} circuits...")

    records = []
    total_t0 = time.perf_counter()

    for i, circuit in enumerate(circuits, 1):
        logger.info(f"[{i}/{len(circuits)}] Extracting features for '{circuit}'...")
        sf = extract_circuit_structural_features(circuit, time_budget_s=600.0, seed=42)

        records.append(
            {
                "circuit": sf.circuit,
                "mffc_mean": sf.mffc_mean,
                "mffc_large_frac": sf.mffc_large_frac,
                "multifanout_frac": sf.multifanout_frac,
                "cut4_mean": sf.cut4_mean,
                "critical_cone_frac": sf.critical_cone_frac,
                "balance_ratio": sf.balance_ratio,
                "mffc_sampled": sf.mffc_sampled,
                "cut4_truncated": sf.cut4_truncated,
                "extraction_time_s": sf.extraction_time_s,
            }
        )

        logger.info(
            f"   Done '{circuit}' in {sf.extraction_time_s:.2f}s | "
            f"mffc_mean={sf.mffc_mean:.2f}, cut4_mean={sf.cut4_mean:.2f}, "
            f"mffc_sampled={sf.mffc_sampled}, cut4_truncated={sf.cut4_truncated}"
        )

    total_time_s = time.perf_counter() - total_t0
    logger.info(f"Extraction completed for all {len(circuits)} circuits in {total_time_s:.2f}s.")

    res_df = pd.DataFrame(records)

    # Save to data/processed/circuit_features_extended.csv
    out_dir = get_processed_data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "circuit_features_extended.csv"

    # Select final columns for CSV (circuit + six feature columns)
    export_cols = [
        "circuit",
        "mffc_mean",
        "mffc_large_frac",
        "multifanout_frac",
        "cut4_mean",
        "critical_cone_frac",
        "balance_ratio",
    ]
    res_df[export_cols].to_csv(out_path, index=False)
    logger.info(f"Saved extended circuit features to {out_path}")

    # Also log full summary table
    print("\n" + "=" * 100)
    print("                      STRUCTURAL FEATURES EXTRACTION SUMMARY                      ")
    print("=" * 100)
    print(res_df.to_string(index=False))
    print("=" * 100 + "\n")

    return res_df


if __name__ == "__main__":
    run_extraction()
