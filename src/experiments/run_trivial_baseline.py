"""Phase 1 Deliverable: Driver script executing ConstantMeanBaseline across split protocols.

Verifies the complete pipeline end-to-end:
loaders -> splits -> harness -> metrics -> MLflow tracking artifact creation.
"""

from src.config.schema import ExperimentConfig
from src.data.loaders import load_modeling_table
from src.eval.harness import run_experiment
from src.models.baselines import ConstantMeanBaseline
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("src.experiments.run_trivial_baseline")


def main() -> None:
    """Executes ConstantMeanBaseline on random split and LOCO protocols."""
    setup_logging()
    logger.info("Executing Phase 1 Deliverable: Trivial Constant Predictor Baseline...")

    # Load dataset for L=10
    dataset = load_modeling_table(seq_len=10, target="target_and_ratio")

    # 1. Random Split Protocol
    cfg_random = ExperimentConfig(
        model_family="baseline_mean",
        dataset_L=10,
        target="target_and_ratio",
        encoding="all",
        structural_features=False,
        split_protocol="random",
        exclude_hyp=True,
        seed=42,
        experiment_name="qor-rq1-random-split",
    )

    model_random = ConstantMeanBaseline()
    logger.info("Running ConstantMeanBaseline on Random Split protocol...")
    res_random = run_experiment(model_random, dataset, cfg_random)

    print("\n==================================================")
    print("      RANDOM SPLIT - CONSTANT MEAN BASELINE       ")
    print("==================================================")
    print(f"  MLflow Run ID       : {res_random['run_id']}")
    print(f"  MAPE                : {res_random['metrics']['mape']:.4f}%")
    print(f"  MAE                 : {res_random['metrics']['mae']:.4f}")
    print(f"  RMSE                : {res_random['metrics']['rmse']:.4f}")
    print(f"  Spearman Within Mean: {res_random['metrics']['spearman_within_mean']:.4f}")
    print(f"  Spearman Within Std : {res_random['metrics']['spearman_within_std']:.4f}")
    print(f"  Regret@10%          : {res_random['metrics']['regret_at_10pct']:.4f}")
    print("==================================================\n")

    # 2. Leave-Circuits-Out (LOCO) Protocol
    cfg_loco = ExperimentConfig(
        model_family="baseline_mean",
        dataset_L=10,
        target="target_and_ratio",
        encoding="all",
        structural_features=False,
        split_protocol="loco",
        exclude_hyp=True,
        seed=42,
        experiment_name="qor-rq2-leave-circuits-out",
    )

    model_loco = ConstantMeanBaseline()
    logger.info("Running ConstantMeanBaseline on Leave-Circuits-Out (LOCO) protocol...")
    res_loco = run_experiment(model_loco, dataset, cfg_loco)

    print("==================================================")
    print("      LOCO SPLIT - CONSTANT MEAN BASELINE         ")
    print("==================================================")
    print(f"  MLflow Run ID       : {res_loco['run_id']}")
    print(f"  MAPE                : {res_loco['metrics']['mape']:.4f}%")
    print(f"  MAE                 : {res_loco['metrics']['mae']:.4f}")
    print(f"  RMSE                : {res_loco['metrics']['rmse']:.4f}")
    print(f"  Spearman Within Mean: {res_loco['metrics']['spearman_within_mean']:.4f}")
    print(f"  Spearman Within Std : {res_loco['metrics']['spearman_within_std']:.4f}")
    print(f"  Regret@10%          : {res_loco['metrics']['regret_at_10pct']:.4f}")
    print("==================================================\n")

    logger.info("Phase 1 Deliverable execution finished successfully!")


if __name__ == "__main__":
    main()
