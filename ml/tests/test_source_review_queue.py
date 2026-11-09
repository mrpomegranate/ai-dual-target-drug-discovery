"""Unit tests for manual provenance review: no label reconciliation."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_source_review_queue.py"
spec = importlib.util.spec_from_file_location("build_source_review_queue", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def fixture():
    e = pd.DataFrame([
        {"target": "EGFR", "m18c_structure_id": "id1", "provenance_key": "pmid:1", "reactant_set_id": "r1", "review_rank": 1,
         "ic50_nm": 50.0, "pic50": 7.30103, "source": "ChEMBL", "m17_partition": "primary_exact_candidate",
         "m18c_status": "provisional_exact_structure", "article_doi": "doi1", "pmid": "1", "patent_number": "",
         "pubchem_aid": "", "ph": "", "temperature_c": "", "candidate_duplicate_signature": False, "target_name": "EGFR"},
        {"target": "EGFR", "m18c_structure_id": "id1", "provenance_key": "pmid:1", "reactant_set_id": "r2", "review_rank": 1,
         "ic50_nm": 5000.0, "pic50": 5.30103, "source": "ChEMBL", "m17_partition": "primary_exact_candidate",
         "m18c_status": "provisional_exact_structure", "article_doi": "doi1", "pmid": "1", "patent_number": "",
         "pubchem_aid": "", "ph": "", "temperature_c": "", "candidate_duplicate_signature": False, "target_name": "EGFR"},
    ])
    g = pd.DataFrame([{"target": "EGFR", "m18c_structure_id": "id1", "provenance_key": "pmid:1",
                       "n_observations": 2, "min_ic50_nm": 50.0, "max_ic50_nm": 5000.0,
                       "within_provenance_range_pic50": 2.0, "candidate_duplicate_rows": 0}])
    return e, g


def test_reconciles_and_marks_manual_review():
    e, g = fixture()
    q, s = module.check_and_build(e, g)
    assert s["observations"] == 2
    assert s["within_provenance_gt10fold"] == 1
    assert s["within_provenance_crossing_1000_nm"] == 1
    assert q.iloc[0].source_review_priority == "highest"
    assert q.iloc[0].curation_decision == "deferred"


def test_unmatched_group_rejected():
    e, g = fixture()
    g.loc[0, "n_observations"] = 3
    with pytest.raises(ValueError, match="Group field mismatch"):
        module.check_and_build(e, g)


def test_nonprimary_rejected():
    e, g = fixture()
    e.loc[1, "m17_partition"] = "secondary_construct_reserve"
    with pytest.raises(ValueError, match="non-primary-exact"):
        module.check_and_build(e, g)


def test_duplicate_record_rejected():
    e, g = fixture()
    e.loc[1, "reactant_set_id"] = "r1"
    with pytest.raises(ValueError, match="Duplicate reactant_set_id"):
        module.check_and_build(e, g)


def test_wrong_spread_rejected():
    e, g = fixture()
    g.loc[0, "within_provenance_range_pic50"] = 0.5
    with pytest.raises(ValueError, match="spreads differ"):
        module.check_and_build(e, g)
