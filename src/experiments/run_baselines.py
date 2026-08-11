"""Phase 2 Experiment Driver: Evaluates all four baselines across protocols and sequence lengths.

Scope:
- Models: baseline_mean, baseline_percircuit, linear, seq_only
- Protocols: random split, leave-circuits-out (LOCO)
- Sequence lengths: L=5, L=10, L=15
- Seeds: 42, 43, 44 (≥3 seeds per INV-6)
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config.schema import ExperimentConfig
from src.data.loaders import load_modeling_table, ModelingDataset
from src.eval.harness import run_experiment
from src.models.baselines import (
    ConstantMeanBaseline,
    LinearRegressionBaseline,
    PerCircuitMeanBaseline,
    SequenceOnlyBaseline,
)
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("src.experiments.run_baselines")


def instantiate_model(model_family: str):
    """Instantiates a baseline model based on family name."""
    if model_family == "baseline_mean":
        return ConstantMeanBaseline()
    elif model_family == "baseline_percircuit":
        return PerCircuitMeanBaseline()
    elif model_family == "linear":
        return LinearRegressionBaseline(alpha=1.0)
    elif model_family == "seq_only":
        return SequenceOnlyBaseline(alpha=1.0)
    else:
        raise ValueError(f"Unknown baseline family: {model_family}")


def main() -> None:
    """Executes full Phase 2 baseline evaluation matrix and checks interpretation checkpoint."""
    setup_logging()
    logger.info("Starting Phase 2 Baseline Evaluation...")

    seq_lengths = [5, 10, 15]
    protocols = ["random", "loco"]
    model_families = ["baseline_mean", "baseline_percircuit", "linear", "seq_only"]
    seeds = [42, 43, 44]

    # Pre-load datasets for L=5, L=10, L=15
    datasets: Dict[int, ModelingDataset] = {}
    for L in seq_lengths:
        logger.info(f"Loading modeling table for L={L}...")
        datasets[L] = load_modeling_table(seq_len=L, target="target_and_ratio")

    results_records: List[Dict[str, float]] = []

    total_runs = len(seq_lengths) * len(protocols) * len(model_families) * len(seeds)
    run_idx = 0

    for L in seq_lengths:
        dataset = datasets[L]
        for protocol in protocols:
            exp_name = (
                "qor-rq1-random-split" if protocol == "random" else "qor-rq2-leave-circuits-out"
            )

            for family in model_families:
                for seed in seeds:
                    run_idx += 1
                    logger.info(
                        f"[{run_idx}/{total_runs}] Running {family} | {protocol} | L={L} | seed={seed}"
                    )

                    config = ExperimentConfig(
                        model_family=family,
                        dataset_L=L,
                        target="target_and_ratio",
                        encoding="all",
                        structural_features=False,
                        split_protocol=protocol,
                        exclude_hyp=True,
                        seed=seed,
                        experiment_name=exp_name,
                    )

                    model = instantiate_model(family)
                    res = run_experiment(model, dataset, config)

                    m = res["metrics"]
                    record = {
                        "model": family,
                        "protocol": protocol,
                        "L": L,
                        "seed": seed,
                        "spearman": m.get("spearman_within_mean", 0.0),
                        "mape": m.get("mape", 0.0),
                        "mae": m.get("mae", 0.0),
                        "rmse": m.get("rmse", 0.0),
                        "regret_10": m.get("regret_at_10pct", 0.0),
                    }
                    results_records.append(record)

    df_res = pd.DataFrame(results_records)

    # Group by (model, protocol, L) and average over seeds
    grouped = (
        df_res.groupby(["model", "protocol", "L"])
        .agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            regret_mean=("regret_10", "mean"),
        )
        .reset_index()
    )

    print("\n" + "=" * 90)
    print("                      PHASE 2 BASELINES SUMMARY TABLE                       ")
    print("=" * 90)
    print(
        f"{'Model':<22} | {'Protocol':<8} | {'L':<3} | {'Spearman (mean±std)':<21} | {'MAPE (%) (mean±std)':<21}"
    )
    print("-" * 90)

    for _, row in grouped.iterrows():
        sp_str = f"{row['spearman_mean']:.4f} ± {row['spearman_std']:.4f}"
        mape_str = f"{row['mape_mean']:.2f}% ± {row['mape_std']:.2f}"
        print(
            f"{row['model']:<22} | {row['protocol']:<8} | {row['L']:<3} | {sp_str:<21} | {mape_str:<21}"
        )
    print("=" * 90 + "\n")

    # Interpretation Checkpoint Analysis (§11 Phase 2)
    # Check if per-circuit mean is close to best model (linear) on random split
    rand_l10 = grouped[(grouped["protocol"] == "random") & (grouped["L"] == 10)]

    per_circuit_row = rand_l10[rand_l10["model"] == "baseline_percircuit"]
    linear_row = rand_l10[rand_l10["model"] == "linear"]

    if not per_circuit_row.empty and not linear_row.empty:
        per_circuit_mape = per_circuit_row["mape_mean"].values[0]
        linear_mape = linear_row["mape_mean"].values[0]

        per_circuit_sp = per_circuit_row["spearman_mean"].values[0]
        linear_sp = linear_row["spearman_mean"].values[0]

        diff_mape = abs(per_circuit_mape - linear_mape)

        print("=" * 90)
        print("          INTERPRETATION CHECKPOINT (§11 Phase 2)           ")
        print("=" * 90)
        print(f"  Random Split (L=10) - Per-Circuit Mean MAPE : {per_circuit_mape:.2f}% (Spearman: {per_circuit_sp:.4f})")
        print(f"  Random Split (L=10) - Linear Model MAPE      : {linear_mape:.2f}% (Spearman: {linear_sp:.4f})")
        print(f"  MAPE Gap                                     : {diff_mape:.2f}%")

        if diff_mape < 2.0 or per_circuit_mape <= linear_mape + 1.0:
            print("\n  [FLAG TRIGGERED]: The per-circuit mean baseline is VERY CLOSE to the best model on the random split.")
            print("  This confirms that circuit identity dominates the random-split task, making it nearly trivial.")
            print("  Conclusion: The paper's primary evidentiary weight MUST shift toward Leave-Circuits-Out (LOCO).")
        else:
            print("\n  [FLAG NOT TRIGGERED]: Linear model maintains a distinct advantage over per-circuit mean on random split.")
        print("=" * 90 + "\n")

    logger.info("Phase 2 baseline evaluation complete!")


if __name__ == "__main__":
    main()
