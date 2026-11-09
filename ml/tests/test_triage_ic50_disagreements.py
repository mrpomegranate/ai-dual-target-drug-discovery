"""Synthetic regression tests for M1.9b; no real BindingDB labels are invented."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from triage_ic50_disagreements import classify_case, triage, validate


def make_tables():
    mol = pd.DataFrame([
        {"target": "EGFR", "m18c_structure_id": "a", "n_observations": 2,
         "n_provenance_groups": 2, "n_source_categories": 2,
         "min_pic50": 5.5, "max_pic50": 7.5, "range_pic50": 2.0,
         "minimum_ic50_nm": 31.6227766, "maximum_ic50_nm": 3162.27766,
         "n_active": 1, "n_inactive": 1, "crosses_activity_cutoff": True,
         "n_study_groups": 2, "study_groups_within_range_gt1": 0,
         "study_median_range_pic50": 2.0},
        {"target": "HER2", "m18c_structure_id": "b", "n_observations": 2,
         "n_provenance_groups": 1, "n_source_categories": 1,
         "min_pic50": 6.2, "max_pic50": 6.7, "range_pic50": 0.5,
         "minimum_ic50_nm": 199.5262315, "maximum_ic50_nm": 630.9573445,
         "n_active": 2, "n_inactive": 0, "crosses_activity_cutoff": False,
         "n_study_groups": 1, "study_groups_within_range_gt1": 0,
         "study_median_range_pic50": 0.0},
        {"target": "HER2", "m18c_structure_id": "c", "n_observations": 1,
         "n_provenance_groups": 1, "n_source_categories": 1,
         "min_pic50": 8.0, "max_pic50": 8.0, "range_pic50": 0.0,
         "minimum_ic50_nm": 10.0, "maximum_ic50_nm": 10.0,
         "n_active": 1, "n_inactive": 0, "crosses_activity_cutoff": False,
         "n_study_groups": 1, "study_groups_within_range_gt1": 0,
         "study_median_range_pic50": 0.0},
    ])
    study = pd.DataFrame([
        {"target": "EGFR", "m18c_structure_id": "a", "provenance_key": "pmid:1",
         "n_observations": 1, "median_pic50": 5.5, "min_pic50": 5.5,
         "max_pic50": 5.5, "within_study_range_pic50": 0., "n_exact_duplicate_values": 0},
        {"target": "EGFR", "m18c_structure_id": "a", "provenance_key": "patent:2",
         "n_observations": 1, "median_pic50": 7.5, "min_pic50": 7.5,
         "max_pic50": 7.5, "within_study_range_pic50": 0., "n_exact_duplicate_values": 0},
        {"target": "HER2", "m18c_structure_id": "b", "provenance_key": "pmid:3",
         "n_observations": 2, "median_pic50": 6.45, "min_pic50": 6.2,
         "max_pic50": 6.7, "within_study_range_pic50": 0.5, "n_exact_duplicate_values": 0},
        {"target": "HER2", "m18c_structure_id": "c", "provenance_key": "pmid:4",
         "n_observations": 1, "median_pic50": 8.0, "min_pic50": 8.0,
         "max_pic50": 8.0, "within_study_range_pic50": 0., "n_exact_duplicate_values": 0},
    ])
    return mol, study


def test_validate_synthetic():
    validate(*make_tables())


def test_triage_counts():
    cases, studies, summary = triage(*make_tables())
    assert len(cases) == 2
    assert len(studies) == 2
    assert summary["priority_pairs"] == 1
    assert summary["threshold_crossing_pairs"] == 1
    assert summary["repeated_pairs_gt10fold"] == 1
    assert summary["tiers"]["T0_crossing_gt10fold"] == 1
    assert summary["tiers"]["T3_other_repeated"] == 1
    assert cases.iloc[0]["review_rank"] == 1
    assert cases.iloc[0]["m18c_structure_id"] == "a"


@pytest.mark.parametrize(
    ("n", "cross", "range_value", "tier"),
    [
        (1, False, 0, "not_repeated"),
        (2, True, 1.1, "T0_crossing_gt10fold"),
        (2, True, 0.2, "T1_crossing_up_to_10fold"),
        (2, False, 1.1, "T2_gt10fold_non_crossing"),
        (2, False, 0.2, "T3_other_repeated"),
        (2, False, 1.0, "T3_other_repeated"),
    ],
)
def test_tier_boundary(n, cross, range_value, tier):
    row = pd.Series({
        "n_observations": n,
        "crosses_activity_cutoff": cross,
        "range_pic50": range_value,
    })
    assert classify_case(row) == tier


def test_reject_missing_columns():
    mol, study = make_tables()
    with pytest.raises(ValueError, match="missing columns"):
        validate(mol.drop(columns=["range_pic50"]), study)


def test_reject_duplicate_keys():
    mol, study = make_tables()
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pd.concat([mol, mol.iloc[[0]]], ignore_index=True), study)


def test_reject_provenance_observation_mismatch():
    mol, study = make_tables()
    study.loc[0, "n_observations"] = 2
    with pytest.raises(ValueError, match="observation totals differ"):
        validate(mol, study)


def test_reject_potency_range_inconsistent():
    mol, study = make_tables()
    mol.loc[0, "range_pic50"] = 1.9
    with pytest.raises(ValueError, match="pIC50 ranges"):
        validate(mol, study)


def test_reject_threshold_crossing_flag():
    mol, study = make_tables()
    mol.loc[0, "crosses_activity_cutoff"] = False
    with pytest.raises(ValueError, match="threshold crossing"):
        validate(mol, study)


def test_does_not_modify_input():
    mol, study = make_tables()
    original_m, original_s = mol.copy(deep=True), study.copy(deep=True)
    triage(mol, study)
    pd.testing.assert_frame_equal(mol, original_m)
    pd.testing.assert_frame_equal(study, original_s)
