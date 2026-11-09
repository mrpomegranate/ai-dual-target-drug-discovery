"""Tests for M1.7 provisional policy; no real BindingDB data required."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from apply_construct_inclusion_policy import decide, norm_human, run


@pytest.fixture
def base_row():
    return {
        "target": "EGFR", "target_name": "Epidermal growth factor receptor",
        "construct_category": "plain_name", "organism": "Homo sapiens",
        "smiles": "CCO", "measurement_status": "exact", "ic50_nm": "100",
        "pic50": "7", "activity_at_1000_nm": "active",
    }


@pytest.mark.parametrize("organism", ["Human", "human", "Homo sapiens", "  HOMO SAPIENS "])
def test_normalize_human(organism):
    assert norm_human(organism)


@pytest.mark.parametrize("organism", ["", "Mus musculus", "Unknown"])
def test_nonhuman_or_missing_not_imputed(organism):
    assert not norm_human(organism)


def test_plain_exact(base_row):
    assert decide(base_row) == ("primary_exact_candidate", "exact_candidate", "provisional_plain_target")


def test_exact_requires_consistent_pic50(base_row):
    base_row["pic50"] = "10"
    assert decide(base_row)[0] == "primary_hold_measurement"


@pytest.mark.parametrize("activity", ["active", "inactive"])
def test_determinate_censored_classification(base_row, activity):
    base_row["measurement_status"] = "censored"
    base_row["pic50"] = ""
    base_row["activity_at_1000_nm"] = activity
    assert decide(base_row)[0] == "primary_classification_candidate"


def test_indeterminate_censored_held(base_row):
    base_row["measurement_status"] = "censored"
    base_row["pic50"] = ""
    base_row["activity_at_1000_nm"] = "indeterminate"
    assert decide(base_row)[0] == "primary_hold_measurement"


def test_missing_organism_held(base_row):
    base_row["organism"] = ""
    assert decide(base_row)[0] == "primary_hold_organism"


def test_missing_structure_held(base_row):
    base_row["smiles"] = ""
    assert decide(base_row)[0] == "primary_hold_smiles"


@pytest.mark.parametrize("category", ["substitutions_only", "mixed_construct", "other_construct"])
def test_annotated_reserved(base_row, category):
    base_row["construct_category"] = category
    assert decide(base_row) == ("secondary_construct_reserve", "exact_candidate", "annotated_target_construct")


def test_annotated_invalid_still_reserved_but_not_ready(base_row):
    base_row["construct_category"] = "substitutions_only"
    base_row["organism"] = ""
    assert decide(base_row) == ("secondary_construct_reserve", "not_ready", "organism_unconfirmed")


def test_unclassified_target_name_requires_manual_review(base_row):
    base_row["construct_category"] = "unclassified_nonbase"
    assert decide(base_row)[0] == "manual_construct_review"


def test_full_synthetic_audit(tmp_path):
    observations = [
        {"reactant_set_id": "1", "target": "EGFR", "target_name": "Epidermal growth factor receptor", "source": "ChEMBL", "inchi_key": "A", "organism": "Human", "organism_flag": "nonhuman_or_unexpected", "smiles": "CCO", "measurement_status": "exact", "ic50_nm": "100", "pic50": "7", "activity_at_1000_nm": "active", "article_doi": "", "pmid": "1", "patent_number": ""},
        {"reactant_set_id": "2", "target": "EGFR", "target_name": "Epidermal growth factor receptor [T790M]", "source": "US Patent", "inchi_key": "A", "organism": "Human", "organism_flag": "human", "smiles": "CCO", "measurement_status": "exact", "ic50_nm": "1000", "pic50": "6", "activity_at_1000_nm": "inactive", "article_doi": "", "pmid": "", "patent_number": "US222"},
        {"reactant_set_id": "3", "target": "HER2", "target_name": "Receptor tyrosine-protein kinase erbB-2", "source": "ChEMBL", "inchi_key": "A", "organism": "Homo sapiens", "organism_flag": "human", "smiles": "CCO", "measurement_status": "censored", "ic50_nm": "500", "pic50": "", "activity_at_1000_nm": "active", "article_doi": "", "pmid": "2", "patent_number": ""},
        {"reactant_set_id": "4", "target": "EGFR", "target_name": "Epidermal growth factor receptor", "source": "ChEMBL", "inchi_key": "B", "organism": "", "organism_flag": "missing", "smiles": "CCC", "measurement_status": "exact", "ic50_nm": "10", "pic50": "8", "activity_at_1000_nm": "active", "article_doi": "", "pmid": "1", "patent_number": ""},
    ]
    input_file = tmp_path / "observations.csv"
    pd.DataFrame(observations).to_csv(input_file, index=False)
    out_csv = tmp_path / "decisions.csv"
    out_json = tmp_path / "summary.json"
    figures = tmp_path / "figures"
    result = run(input_file, out_csv, out_json, figures)
    assert result["observations"] == 4
    assert result["source_organism_flag_disagreements"] == 1
    assert result["per_target"]["EGFR"]["partitions"]["primary_exact_candidate"] == 1
    assert result["per_target"]["EGFR"]["partitions"]["secondary_construct_reserve"] == 1
    assert result["per_target"]["HER2"]["partitions"]["primary_classification_candidate"] == 1
    assert len(pd.read_csv(out_csv)) == 4
    assert out_json.exists()
    assert (figures / "09_provisional_inclusion_decisions.png").exists()
    assert (figures / "10_source_composition_after_policy.png").exists()
