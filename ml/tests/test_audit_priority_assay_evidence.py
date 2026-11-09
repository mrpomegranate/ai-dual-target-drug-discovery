"""Synthetic M1.9c regression tests; no real binding labels are invented."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from audit_priority_assay_evidence import (
    build_summary,
    choose_cases,
    make_figure,
    reconstruct,
)


def sample_tables():
    def observation(record_id, target, structure, ic50, provenance, partition="primary_exact_candidate"):
        return {
            "reactant_set_id": str(record_id),
            "target": target,
            "m18c_structure_id": structure,
            "m17_partition": partition,
            "m18c_status": "provisional_exact_structure",
            "measurement_status": "exact",
            "ic50_relation": "=",
            "ic50_nm": float(ic50),
            "pic50": 9 - np.log10(ic50),
            "ic50_raw": str(ic50),
            "activity_at_1000_nm": "active" if ic50 < 1000 else "inactive",
            "provenance_key": provenance,
            "target_name": "Epidermal growth factor receptor" if target == "EGFR" else "Receptor tyrosine-protein kinase erbB-2",
            "source": "ChEMBL",
            "article_doi": "10.test/abc",
            "pmid": "123456",
            "patent_number": None,
            "pubchem_aid": None,
            "ph": None,
            "temperature_c": None,
        }

    obs = pd.DataFrame([
        observation(1, "EGFR", "s1", 100, "pmid:1"),
        observation(2, "EGFR", "s1", 2000, "pmid:2"),
        observation(3, "HER2", "s2", 10, "pmid:3"),
        observation(4, "HER2", "s2", 200, "pmid:3"),
        # Same molecule on an annotated construct must not join into primary evidence.
        observation(5, "EGFR", "s1", 500, "pmid:4", partition="secondary_construct_reserve"),
    ])
    cases = pd.DataFrame([
        {"review_rank": 1, "target": "EGFR", "m18c_structure_id": "s1",
         "triage_tier": "T0_crossing_gt10fold", "n_observations": 2,
         "n_provenance_groups": 2, "minimum_ic50_nm": 100,
         "maximum_ic50_nm": 2000, "range_pic50": np.log10(20),
         "crosses_activity_cutoff": True},
        {"review_rank": 2, "target": "HER2", "m18c_structure_id": "s2",
         "triage_tier": "T2_gt10fold_non_crossing", "n_observations": 2,
         "n_provenance_groups": 1, "minimum_ic50_nm": 10,
         "maximum_ic50_nm": 200, "range_pic50": np.log10(20),
         "crosses_activity_cutoff": False},
    ])
    return obs, cases


def test_reconstruction_preserves_primary_exact_only():
    obs, cases = sample_tables()
    _, evidence, studies = reconstruct(obs, cases, per_target=10)
    assert len(evidence) == 4
    assert set(evidence["reactant_set_id"]) == {"1", "2", "3", "4"}
    assert len(studies) == 3
    assert evidence.loc[evidence["target"] == "HER2", "provenance_key"].nunique() == 1


def test_balanced_selection_across_targets():
    _, cases = sample_tables()
    extra = pd.concat([
        cases,
        pd.DataFrame([
            {**cases.iloc[0].to_dict(), "review_rank": 3, "m18c_structure_id": "s3"},
        ]),
    ], ignore_index=True)
    selected = choose_cases(extra, 1)
    assert len(selected) == 2
    assert set(selected["target"]) == {"EGFR", "HER2"}


def test_reject_wrong_repeated_case_counts():
    obs, cases = sample_tables()
    cases.loc[0, "n_observations"] = 3
    with pytest.raises(ValueError, match="Observation counts"):
        reconstruct(obs, cases)


def test_reject_wrong_provenance_counts():
    obs, cases = sample_tables()
    cases.loc[1, "n_provenance_groups"] = 2
    with pytest.raises(ValueError, match="Provenance counts"):
        reconstruct(obs, cases)


def test_reject_inconsistent_ic50_and_pic50():
    obs, cases = sample_tables()
    obs.loc[0, "pic50"] = 6.0
    with pytest.raises(ValueError, match="IC50/pIC50"):
        reconstruct(obs, cases)


def test_reject_inconsistent_activity_flag():
    obs, cases = sample_tables()
    obs.loc[0, "activity_at_1000_nm"] = "inactive"
    with pytest.raises(ValueError, match="Activity label"):
        reconstruct(obs, cases)


def test_reject_repeated_source_record():
    obs, cases = sample_tables()
    obs.loc[1, "reactant_set_id"] = "1"
    with pytest.raises(ValueError, match="Duplicate BindingDB"):
        reconstruct(obs, cases)


def test_reject_missing_columns():
    obs, cases = sample_tables()
    with pytest.raises(ValueError, match="missing columns"):
        reconstruct(obs.drop(columns=["provenance_key"]), cases)


def test_reject_duplicate_case_ids():
    _, cases = sample_tables()
    copies = pd.concat([cases, cases.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        choose_cases(copies, per_target=10)


def test_summary_and_non_destructive(tmp_path):
    obs, cases = sample_tables()
    before = obs.copy(deep=True)
    selected, evidence, study = reconstruct(obs, cases)
    summary = build_summary(selected, evidence, study)
    assert summary["selected_pairs"] == 2
    assert summary["exact_observation_rows"] == 4
    assert summary["provenance_groups"] == 3
    assert summary["metadata_present_rows"]["ph"] == 0
    pd.testing.assert_frame_equal(obs, before)
    figure = tmp_path / "figure.png"
    make_figure(selected, evidence, figure)
    assert figure.exists() and figure.stat().st_size > 1000


def test_reject_invalid_case_limit():
    _, cases = sample_tables()
    with pytest.raises(ValueError, match="per_target"):
        choose_cases(cases, 0)
