"""Smoke test script for Phase 0 scaffolding and environment verification.

Verifies PyTorch CUDA availability, prints full environment summary,
and logs a trivial experiment run to MLflow via src.tracking.mlflow_utils.
"""

import sys
import platform
import psutil
import torch

from src.config.schema import ExperimentConfig
from src.tracking.mlflow_utils import init_mlflow, start_run, log_params, log_metrics, end_run
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.paths import get_project_root, get_data_dir, get_mlruns_dir

logger = get_logger("src.experiments.smoke_test")


def check_environment() -> dict:
    """Verifies CUDA availability and collects system info.

    Raises RuntimeError if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        error_msg = (
            "CRITICAL ENVELOPE VIOLATION: PyTorch CUDA is unavailable!\n"
            "Silently falling back to CPU is strictly prohibited by GEMINI.md hardware requirements."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)

    cpu_physical = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    ram_bytes = psutil.virtual_memory().total
    ram_gb = ram_bytes / (1024**3)

    env_info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "cpu_physical_cores": cpu_physical,
        "cpu_logical_cores": cpu_logical,
        "ram_gb": round(ram_gb, 2),
        "os_platform": platform.platform(),
    }

    return env_info


def main() -> None:
    """Executes the Phase 0 smoke test."""
    setup_logging()
    logger.info("Initializing Phase 0 Smoke Test...")

    # 1. Check environment and CUDA
    env_info = check_environment()

    # Print clean environment summary to stdout as required by deliverable
    summary_text = (
        "\n==================================================\n"
        "           ENVIRONMENT SUMMARY (Phase 0)          \n"
        "==================================================\n"
        f"  Python Version : {env_info['python_version']}\n"
        f"  PyTorch Version: {env_info['torch_version']}\n"
        f"  CUDA Version   : {env_info['cuda_version']}\n"
        f"  GPU Device Name: {env_info['gpu_name']}\n"
        f"  Detected VRAM  : {env_info['vram_gb']:.2f} GB\n"
        f"  CPU Cores      : {env_info['cpu_physical_cores']} physical / {env_info['cpu_logical_cores']} logical\n"
        f"  System RAM     : {env_info['ram_gb']:.2f} GB\n"
        f"  OS Platform    : {env_info['os_platform']}\n"
        "==================================================\n"
    )
    print(summary_text)

    # 2. Test Configuration loading & validation
    default_config_path = get_project_root() / "src" / "config" / "experiments" / "default_smoke_test.yaml"
    if default_config_path.exists():
        config = ExperimentConfig.from_yaml(default_config_path)
        logger.info(f"Loaded config from {default_config_path}")
    else:
        config = ExperimentConfig(model_family="dummy", experiment_name="qor-smoke-test")
        config.validate()
        logger.info("Loaded default ExperimentConfig")

    # 3. Test MLflow initialization and logging convention
    exp_id = init_mlflow(config.experiment_name)
    logger.info(f"MLflow initialized with experiment ID: {exp_id}")

    with start_run(
        run_name="smoke_test_run",
        tags={"phase": "phase_0", "rq": "smoke_test", "status": "complete"},
    ):
        params_to_log = config.to_dict()
        params_to_log.update({f"env_{k}": v for k, v in env_info.items()})
        params_to_log["config_hash"] = config.compute_config_hash()

        log_params(params_to_log)
        log_metrics({"smoke_test_status": 1.0, "dummy_metric": 0.99})

        logger.info("Successfully logged parameters and metrics to MLflow run.")

    end_run(status="FINISHED")
    logger.info("Smoke test completed successfully end-to-end!")


if __name__ == "__main__":
    main()
