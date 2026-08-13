"""Dataclass configuration definitions and YAML validation for experiments."""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

VALID_MODEL_FAMILIES = {
    "baseline_mean",
    "baseline_percircuit",
    "linear",
    "seq_only",
    "xgboost",
    "lightgbm",
    "mlp",
    "gcn",
    "gin",
    "gnn",
    "dummy",
}

VALID_DATASET_L = {5, 10, 15}

VALID_TARGETS = {
    "target_and_ratio",
    "target_lev_ratio",
    "target_and_raw",
    "target_lev_raw",
    "target_and_logratio",
    "target_and_delta",
}

VALID_ENCODINGS = {"bag", "positional", "bigram", "bag+positional", "all"}

VALID_SPLIT_PROTOCOLS = {"random", "loco"}


@dataclass
class ExperimentConfig:
    """Strongly-typed experiment configuration container with validation."""

    model_family: str
    dataset_L: int = 10
    target: str = "target_and_ratio"
    encoding: str = "all"
    structural_features: bool = False
    split_protocol: str = "random"
    exclude_hyp: bool = True
    seed: int = 42
    max_graph_nodes: int = 250000
    experiment_name: str = "qor-smoke-test"
    model_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validates all config fields, raising ValueError on mismatch."""
        if self.model_family not in VALID_MODEL_FAMILIES:
            raise ValueError(
                f"Invalid model_family '{self.model_family}'. Must be one of {sorted(VALID_MODEL_FAMILIES)}"
            )

        if self.dataset_L not in VALID_DATASET_L:
            raise ValueError(
                f"Invalid dataset_L '{self.dataset_L}'. Must be one of {sorted(VALID_DATASET_L)}"
            )

        if self.target not in VALID_TARGETS:
            raise ValueError(
                f"Invalid target '{self.target}'. Must be one of {sorted(VALID_TARGETS)}"
            )

        if self.encoding not in VALID_ENCODINGS:
            raise ValueError(
                f"Invalid encoding '{self.encoding}'. Must be one of {sorted(VALID_ENCODINGS)}"
            )

        if self.split_protocol not in VALID_SPLIT_PROTOCOLS:
            raise ValueError(
                f"Invalid split_protocol '{self.split_protocol}'. Must be one of {sorted(VALID_SPLIT_PROTOCOLS)}"
            )

        if self.seed < 0:
            raise ValueError(f"Seed must be non-negative, got {self.seed}")

        if self.max_graph_nodes <= 0:
            raise ValueError(f"max_graph_nodes must be positive, got {self.max_graph_nodes}")

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to dictionary."""
        return asdict(self)

    def compute_config_hash(self) -> str:
        """Computes a stable SHA1 hash of the resolved configuration for resumability."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(config_str.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Constructs and validates ExperimentConfig from a dictionary."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        config_kwargs = {}
        model_params = data.get("model_params", {}).copy()

        for k, v in data.items():
            if k in known_keys and k != "model_params":
                config_kwargs[k] = v
            elif k != "model_params":
                # Extra parameters automatically go into model_params
                model_params[k] = v

        config_kwargs["model_params"] = model_params
        config = cls(**config_kwargs)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ExperimentConfig":
        """Loads configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config YAML file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Saves configuration to a YAML file."""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
