"""Unit tests for Phase 0 scaffolding: paths, config schema, MLflow utils, and CUDA."""

import pytest
import torch
from pathlib import Path

from src.config.schema import ExperimentConfig
from src.utils.paths import get_project_root, get_data_dir, get_mlruns_dir
from src.tracking.mlflow_utils import init_mlflow, start_run, log_params, log_metrics, end_run


def test_paths_resolution():
    root = get_project_root()
    assert (root / "GEMINI.md").exists(), "Project root must contain GEMINI.md"
    assert get_data_dir() == root / "data"
    assert get_mlruns_dir() == root / "mlruns"


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA must be available in this environment"
    assert torch.cuda.device_count() >= 1
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    assert vram_gb >= 5.0, f"Detected VRAM {vram_gb} GB is below required ~6GB envelope"


def test_config_schema_validation():
    # Valid config
    cfg = ExperimentConfig(
        model_family="xgboost",
        dataset_L=10,
        target="target_and_ratio",
        encoding="all",
        split_protocol="random",
        seed=42,
    )
    cfg.validate()

    # Invalid dataset_L
    with pytest.raises(ValueError):
        invalid_cfg = ExperimentConfig(model_family="xgboost", dataset_L=7)
        invalid_cfg.validate()

    # Invalid model family
    with pytest.raises(ValueError):
        invalid_cfg = ExperimentConfig(model_family="invalid_family")
        invalid_cfg.validate()


def test_mlflow_utils_logging(tmp_path):
    exp_id = init_mlflow("qor-test-experiment")
    assert exp_id is not None

    with start_run(run_name="test_run", tags={"test": "true"}):
        log_params({"test_param": 123, "test_str": "abc"})
        log_metrics({"test_metric": 0.5})

    end_run(status="FINISHED")
