
"""Tests for cached Morgan fingerprints and molecule-to-feature alignment."""

from dataclasses import replace

import numpy as np
import pytest
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.features import morgan_pipeline
from dual_target.features.morgan_pipeline import build_morgan_feature_bank
from dual_target.morgan_features import make_morgan_fingerprints


@pytest.fixture
def records():
    """Tiny synthetic dataset with distinct molecule and scaffold identities."""
    cases = (
        ("a", "CCO", "scaffold_a", 7.1, None),
        ("b", "CCN", "scaffold_b", None, 6.8),
        ("c", "c1ccccc1", "scaffold_c", 7.5, 6.5),
        ("d", "CCCl", "scaffold_d", 5.9, 7.0),
    )
    return tuple(
        MoleculeRecord(
            structure_id=structure_id,
            smiles=smiles,
            scaffold_group_id=scaffold_id,
            egfr_mask=int(egfr is not None),
            her2_mask=int(her2 is not None),
            egfr_pic50=egfr,
            her2_pic50=her2,
        )
        for structure_id, smiles, scaffold_id, egfr, her2 in cases
    )


def test_bank_shape_type_and_metadata(records):
    bank = build_morgan_feature_bank(records)
    assert bank.features.shape == (4, 2048)
    assert bank.features.dtype == np.uint8
    assert bank.molecule_ids == ("a", "b", "c", "d")
    assert bank.smiles == tuple(row.smiles for row in records)
    assert bank.scaffold_ids == tuple(row.scaffold_group_id for row in records)
    assert (bank.radius, bank.n_bits, bank.include_chirality) == (2, 2048, True)
    assert set(np.unique(bank.features)).issubset({0, 1})


def test_bank_matches_original_fingerprint_generator(records):
    bank = build_morgan_feature_bank(records)
    expected = make_morgan_fingerprints([row.smiles for row in records])
    np.testing.assert_array_equal(bank.features, expected)


def test_select_rows_in_requested_order(records):
    bank = build_morgan_feature_bank(records)
    selected = bank.select_features((records[3], records[0], records[2]))
    np.testing.assert_array_equal(selected, bank.features[[3, 0, 2]])
    assert selected.shape == (3, 2048)


def test_select_single_row(records):
    bank = build_morgan_feature_bank(records)
    np.testing.assert_array_equal(
        bank.select_features((records[1],)), bank.features[[1]]
    )


def test_empty_selection_is_two_dimensional(records):
    bank = build_morgan_feature_bank(records)
    assert bank.select_features(()).shape == (0, 2048)


def test_cached_matrix_is_read_only(records):
    bank = build_morgan_feature_bank(records)
    assert not bank.features.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        bank.features[0, 0] = 1


def test_selected_features_are_independent_copies(records):
    bank = build_morgan_feature_bank(records)
    selection = bank.select_features((records[0],))
    original = int(bank.features[0, 0])
    selection[0, 0] = 1 - original
    assert int(bank.features[0, 0]) == original


def test_selection_does_not_regenerate_fingerprints(records, monkeypatch):
    bank = build_morgan_feature_bank(records)

    def forbidden_generator(*args, **kwargs):
        raise AssertionError("Fingerprint generation must not run on selection")

    monkeypatch.setattr(
        morgan_pipeline, "make_morgan_fingerprints", forbidden_generator
    )
    selected = bank.select_features((records[2], records[1]))
    assert selected.shape == (2, 2048)


def test_hash_is_deterministic(records):
    first = build_morgan_feature_bank(records)
    second = build_morgan_feature_bank(records)
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert len(first.fingerprint_sha256) == 64


def test_hash_changes_with_row_order(records):
    first = build_morgan_feature_bank(records)
    reordered = build_morgan_feature_bank(tuple(reversed(records)))
    assert first.fingerprint_sha256 != reordered.fingerprint_sha256


def test_hash_changes_with_configuration(records):
    first = build_morgan_feature_bank(records)
    changed = build_morgan_feature_bank(records, n_bits=1024)
    assert first.fingerprint_sha256 != changed.fingerprint_sha256
    assert changed.features.shape == (4, 1024)


def test_hash_changes_with_scaffold_identity(records):
    bank = build_morgan_feature_bank(records)
    changed = replace(records[0], scaffold_group_id="new_scaffold")
    other = build_morgan_feature_bank((changed, *records[1:]))
    assert bank.fingerprint_sha256 != other.fingerprint_sha256


def test_chirality_configuration_affects_features():
    stereo = (
        MoleculeRecord("left", "C[C@H](O)F", "s1", 1, 0, 7.0, None),
        MoleculeRecord("right", "C[C@@H](O)F", "s2", 1, 0, 7.0, None),
    )
    with_chirality = build_morgan_feature_bank(stereo, include_chirality=True)
    without_chirality = build_morgan_feature_bank(stereo, include_chirality=False)
    assert not np.array_equal(with_chirality.features[0], with_chirality.features[1])
    np.testing.assert_array_equal(
        without_chirality.features[0], without_chirality.features[1]
    )


def test_empty_bank_is_rejected():
    with pytest.raises(ValueError, match="no molecules"):
        build_morgan_feature_bank(())


def test_duplicate_bank_id_is_rejected(records):
    with pytest.raises(ValueError, match="Duplicate molecule IDs"):
        build_morgan_feature_bank((records[0], records[0]))


def test_missing_bank_id_is_rejected(records):
    modified = replace(records[0], structure_id="")
    with pytest.raises(ValueError, match="Missing molecule ID"):
        build_morgan_feature_bank((modified, *records[1:]))


def test_missing_scaffold_is_rejected(records):
    modified = replace(records[0], scaffold_group_id="")
    with pytest.raises(ValueError, match="Missing scaffold ID"):
        build_morgan_feature_bank((modified, *records[1:]))


@pytest.mark.parametrize("kwargs", [{"radius": -1}, {"n_bits": 0}])
def test_invalid_fingerprint_parameters_are_rejected(records, kwargs):
    with pytest.raises(ValueError, match="Invalid Morgan"):
        build_morgan_feature_bank(records, **kwargs)


def test_invalid_smiles_is_rejected(records):
    modified = replace(records[0], smiles="C1CC")
    with pytest.raises(ValueError, match="Invalid SMILES"):
        build_morgan_feature_bank((modified, *records[1:]))


def test_unknown_selected_id_is_rejected(records):
    bank = build_morgan_feature_bank(records)
    unknown = replace(records[0], structure_id="unknown")
    with pytest.raises(ValueError, match="not in fingerprint bank"):
        bank.select_features((unknown,))


def test_duplicate_selected_id_is_rejected(records):
    bank = build_morgan_feature_bank(records)
    with pytest.raises(ValueError, match="Duplicate requested molecule ID"):
        bank.select_features((records[0], records[0]))


def test_changed_smiles_is_rejected(records):
    bank = build_morgan_feature_bank(records)
    wrong = replace(records[0], smiles="CCCC")
    with pytest.raises(ValueError, match="SMILES mismatch"):
        bank.select_features((wrong,))


def test_changed_scaffold_is_rejected(records):
    bank = build_morgan_feature_bank(records)
    wrong = replace(records[0], scaffold_group_id="other_scaffold")
    with pytest.raises(ValueError, match="Scaffold mismatch"):
        bank.select_features((wrong,))
