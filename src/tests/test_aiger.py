"""Unit tests for binary AIGER parser and cross-validation against circuit_features.csv.

Strictly enforces GEMINI.md §7.1 mandatory cross-checks.
"""

import pandas as pd
import pytest
from src.data.aiger import load_or_parse_aig
from src.utils.paths import get_project_root


def get_all_circuits():
    circuit_csv = get_project_root() / "data" / "circuit_features.csv"
    df = pd.read_csv(circuit_csv)
    return df.to_dict(orient="records")


@pytest.mark.parametrize("circuit_info", get_all_circuits(), ids=lambda c: c["circuit"])
def test_aiger_parser_mandatory_cross_checks(circuit_info):
    """Mandatory cross-validation check for all 20 circuits per GEMINI.md §7.1.

    Checks:
    1. Parsed AND-node count == and_count in circuit_features.csv
    2. Parsed PI count == pi
    3. Parsed PO count == po
    4. Computed max level == level_count
    """
    circuit_name = circuit_info["circuit"]
    parsed = load_or_parse_aig(circuit_name, force_reparse=True)

    expected_and = int(circuit_info["and_count"])
    expected_pi = int(circuit_info["pi"])
    expected_po = int(circuit_info["po"])
    expected_level = int(circuit_info["level_count"])

    assert parsed.num_and == expected_and, (
        f"Circuit '{circuit_name}': AND count mismatch! Parsed {parsed.num_and} != Expected {expected_and}"
    )

    assert parsed.num_inputs == expected_pi, (
        f"Circuit '{circuit_name}': PI count mismatch! Parsed {parsed.num_inputs} != Expected {expected_pi}"
    )

    assert parsed.num_outputs == expected_po, (
        f"Circuit '{circuit_name}': PO count mismatch! Parsed {parsed.num_outputs} != Expected {expected_po}"
    )

    assert parsed.max_level == expected_level, (
        f"Circuit '{circuit_name}': Max level mismatch! Parsed {parsed.max_level} != Expected {expected_level}"
    )
