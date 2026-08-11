"""Project-root and directory path resolution utilities."""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Returns the absolute Path to the root directory of the project."""
    # This file is at src/utils/paths.py, so root is two levels up from src (or parent of src)
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / "GEMINI.md").exists():
        # Fallback search upwards for GEMINI.md
        current = Path(__file__).resolve().parent
        for parent in current.parents:
            if (parent / "GEMINI.md").exists():
                return parent
    return root


def get_data_dir() -> Path:
    """Returns the path to data/ directory."""
    return get_project_root() / "data"


def get_processed_data_dir() -> Path:
    """Returns the path to data/processed/ directory."""
    return get_data_dir() / "processed"


def get_mlruns_dir() -> Path:
    """Returns the path to mlruns/ directory."""
    return get_project_root() / "mlruns"
