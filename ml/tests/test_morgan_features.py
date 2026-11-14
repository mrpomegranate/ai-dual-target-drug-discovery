
"""Tests for molecular Morgan fingerprint generation."""

import numpy as np
import pytest
from dual_target.morgan_features import make_morgan_fingerprints


def test_fingerprint_shape_and_type():
    smiles = ["CCO", "CCN", "c1ccccc1"]

    X = make_morgan_fingerprints(smiles)

    assert X.shape == (3, 2048)
    assert X.dtype == np.uint8
    assert set(np.unique(X)).issubset({0, 1})


def test_deterministic_fingerprints():
    smiles = ["CCO", "CCN"]

    X1 = make_morgan_fingerprints(smiles)
    X2 = make_morgan_fingerprints(smiles)

    np.testing.assert_array_equal(X1, X2)


def test_equivalent_smiles():
    # Both strings represent ethanol.
    X = make_morgan_fingerprints(["CCO", "OCC"])

    np.testing.assert_array_equal(X[0], X[1])


def test_different_molecules():
    X = make_morgan_fingerprints(["CCO", "c1ccccc1"])

    assert not np.array_equal(X[0], X[1])


def test_chirality_is_preserved():
    # These are opposite stereochemical configurations.
    smiles = ["C[C@H](O)F", "C[C@@H](O)F"]

    with_chirality = make_morgan_fingerprints(
        smiles, include_chirality=True
    )
    without_chirality = make_morgan_fingerprints(
        smiles, include_chirality=False
    )

    assert not np.array_equal(
        with_chirality[0], with_chirality[1]
    )
    np.testing.assert_array_equal(
        without_chirality[0], without_chirality[1]
    )


def test_custom_fingerprint_size():
    X = make_morgan_fingerprints(["CCO"], n_bits=1024)

    assert X.shape == (1, 1024)


def test_empty_input():
    X = make_morgan_fingerprints([])

    assert X.shape == (0, 2048)


@pytest.mark.parametrize("smiles", ["", "C1CC", None])
def test_invalid_smiles_rejected(smiles):
    with pytest.raises(ValueError):
        make_morgan_fingerprints([smiles])


def test_invalid_parameters():
    with pytest.raises(ValueError):
        make_morgan_fingerprints(["CCO"], radius=-1)

    with pytest.raises(ValueError):
        make_morgan_fingerprints(["CCO"], n_bits=0)
