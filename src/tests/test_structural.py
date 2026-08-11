"""Unit tests for structural feature extraction algorithms, invariants, and determinism.

Strictly enforces GEMINI.md §7.3, §7.4.
"""

import numpy as np
import pytest

from src.data.aiger import load_or_parse_aig
from src.data.structural import (
    compute_fanouts,
    compute_mffc_features,
    extract_circuit_structural_features,
)


def test_deref_reref_symmetry():
    """Asserts that deref/reref reference counting restores refs to bit-identical state."""
    parsed = load_or_parse_aig("adder")
    refs, _ = compute_fanouts(parsed)
    initial_refs = refs.copy()

    # Run MFFC feature computation
    mffc_mean, _, _ = compute_mffc_features(parsed, refs, seed=42)

    assert np.array_equal(refs, initial_refs), "refs array was not restored to bit-identical state!"
    assert mffc_mean >= 1.0


def test_structural_feature_ranges():
    """Asserts feature range validity per GEMINI.md §7.4 on test circuits."""
    for circuit_name in ["ctrl", "cavlc", "adder"]:
        sf = extract_circuit_structural_features(circuit_name)

        assert sf.mffc_mean >= 1.0
        assert 1.0 <= sf.mffc_mean <= 50.0
        assert 0.0 <= sf.mffc_large_frac <= 1.0
        assert 0.0 <= sf.multifanout_frac <= 1.0
        assert sf.cut4_mean >= 1.0
        assert 0.0 <= sf.critical_cone_frac <= 1.0
        assert sf.balance_ratio >= 1.0

        for val in [
            sf.mffc_mean,
            sf.mffc_large_frac,
            sf.multifanout_frac,
            sf.cut4_mean,
            sf.critical_cone_frac,
            sf.balance_ratio,
        ]:
            assert not np.isnan(val)
            assert not np.isinf(val)


def test_extraction_determinism():
    """Asserts that two consecutive extractions produce byte-identical output."""
    sf1 = extract_circuit_structural_features("cavlc", seed=42)
    sf2 = extract_circuit_structural_features("cavlc", seed=42)

    assert sf1.mffc_mean == sf2.mffc_mean
    assert sf1.mffc_large_frac == sf2.mffc_large_frac
    assert sf1.multifanout_frac == sf2.multifanout_frac
    assert sf1.cut4_mean == sf2.cut4_mean
    assert sf1.critical_cone_frac == sf2.critical_cone_frac
    assert sf1.balance_ratio == sf2.balance_ratio
