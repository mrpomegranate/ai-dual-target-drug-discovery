"""Tests for conservative original-source review casebook; no label editing."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_original_source_casebook.py"
spec = importlib.util.spec_from_file_location("build_original_source_casebook", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def example():
    evidence = pd.DataFrame([
        {
            "target": "EGFR", "m18c_structure_id": "s1", "provenance_key": "pmid:32931277",
            "reactant_set_id": "r1", "monomer_id": "m1", "ic50_nm": 10.0, "pic50": 8.0,
            "target_name": "Epidermal growth factor receptor",
            "m17_partition": "primary_exact_candidate",
            "m18c_status": "provisional_exact_structure", "source": "ChEMBL",
            "pmid": "32931277", "article_doi": "doi1",
        },
        {
            "target": "EGFR", "m18c_structure_id": "s1", "provenance_key": "pmid:32931277",
            "reactant_set_id": "r2", "monomer_id": "m1", "ic50_nm": 10000.0, "pic50": 5.0,
            "target_name": "Epidermal growth factor receptor",
            "m17_partition": "primary_exact_candidate",
            "m18c_status": "provisional_exact_structure", "source": "ChEMBL",
            "pmid": "32931277", "article_doi": "doi1",
        },
        {
            "target": "HER2", "m18c_structure_id": "s2", "provenance_key": "pmid:12",
            "reactant_set_id": "r3", "monomer_id": "m2", "ic50_nm": 100.0, "pic50": 7.0,
            "target_name": "Receptor tyrosine-protein kinase erbB-2",
            "m17_partition": "primary_exact_candidate",
            "m18c_status": "provisional_exact_structure", "source": "ChEMBL",
            "pmid": "12", "article_doi": "doi2",
        },
    ])
    queue = pd.DataFrame([
        {
            "target": "EGFR", "m18c_structure_id": "s1", "provenance_key": "pmid:32931277",
            "source_review_priority": "highest", "n_observations": 2, "min_ic50_nm": 10.0,
            "max_ic50_nm": 10000.0, "observed_pic50_range": 3.0, "review_rank": 1,
            "within_group_threshold_crossing": True, "article_doi_candidates": "doi1",
            "pmid_candidates": "32931277", "patent_candidates": "",
        },
        {
            "target": "HER2", "m18c_structure_id": "s2", "provenance_key": "pmid:12",
            "source_review_priority": "routine", "n_observations": 1, "min_ic50_nm": 100.0,
            "max_ic50_nm": 100.0, "observed_pic50_range": 0.0, "review_rank": 2,
            "within_group_threshold_crossing": False, "article_doi_candidates": "doi2",
            "pmid_candidates": "12", "patent_candidates": "",
        },
    ])
    return queue, evidence


def test_build_casebook_retains_exact_rows():
    q, e = example()
    w, raw, s = module.build_casebook(q, e, per_target=1)
    assert s["selected_source_groups"] == 2
    assert s["selected_observation_rows"] == 3
    assert set(raw.reactant_set_id) == {"r1", "r2", "r3"}
    assert w.loc[w.target == "EGFR", "decision"].iloc[0] == "defer_aggregation"


def test_publication_precheck_not_row_verified():
    q, e = example()
    w, _, s = module.build_casebook(q, e, 1)
    a = w.loc[w.provenance_key == "pmid:32931277"].iloc[0]
    assert a.publication_context_review == "pubmed_abstract_checked"
    assert a.exact_compound_matched_to_source_table == "not_checked"
    assert s["row_to_table_matches_confirmed"] == 0


def test_threshold_and_spread_check():
    q, e = example()
    module.validate_evidence(q, e)
    q.loc[0, "within_group_threshold_crossing"] = False
    with pytest.raises(ValueError, match="contradiction"):
        module.validate_evidence(q, e)


def test_wrong_rows_fail():
    q, e = example()
    e.loc[0, "ic50_nm"] = -1
    with pytest.raises(ValueError, match="IC50"):
        module.validate_evidence(q, e)


def test_duplicate_record_id_fails():
    q, e = example()
    e.loc[1, "reactant_set_id"] = "r1"
    with pytest.raises(ValueError, match="Nonunique"):
        module.validate_evidence(q, e)


def test_queue_count_mismatch_fails():
    q, e = example()
    q.loc[0, "n_observations"] = 3
    with pytest.raises(ValueError, match="observation count"):
        module.validate_evidence(q, e)


def test_non_primary_exact_fails():
    q, e = example()
    e.loc[0, "m17_partition"] = "secondary_construct_reserve"
    with pytest.raises(ValueError, match="Non-primary-exact"):
        module.validate_evidence(q, e)


def test_missing_group_fails():
    q, e = example()
    q = q.iloc[:1]
    with pytest.raises(ValueError, match="identical source groups"):
        module.validate_evidence(q, e)


def test_url_resolution_only_known_key_types():
    assert module.source_identifier_to_url("pmid:32931277").endswith("32931277/")
    assert module.source_identifier_to_url("doi:10.1000/xyz") == "https://doi.org/10.1000/xyz"
    assert module.source_identifier_to_url("patent:US12345") == ""


def test_deterministic_selection_and_focus_pmid():
    q, _ = example()
    selected = module.select_groups(q, per_target=1, focus_pmids=("32931277",))
    assert len(selected) == 2
    assert selected.duplicated(subset=module.KEYS).sum() == 0


def test_bad_per_target_fails():
    q, _ = example()
    with pytest.raises(ValueError, match="per_target"):
        module.select_groups(q, per_target=0)
