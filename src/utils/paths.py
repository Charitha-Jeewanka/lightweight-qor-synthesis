"""Project-root and directory path resolution utilities."""

import os
import subprocess
from pathlib import Path


def get_project_root() -> Path:
    """Returns the absolute Path to the root directory of the project."""
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / "GEMINI.md").exists():
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


def get_git_commit_hash() -> str:
    """Retrieves short git commit hash for tracking reproducibility."""
    try:
        root = get_project_root()
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception as e:
        return f"uncommitted ({e.__class__.__name__})"
