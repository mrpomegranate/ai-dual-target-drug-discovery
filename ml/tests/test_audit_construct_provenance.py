"""M1.6 target-name classification is deliberately conservative."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_construct_provenance.py"


def load_module():
    spec = importlib.util.spec_from_file_location("construct_provenance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("target", "name", "category", "notation", "signature"),
    [
        ("EGFR", "Epidermal growth factor receptor", "plain_name", "none", ""),
        ("EGFR", "Epidermal growth factor receptor [L858R]", "substitutions_only", "square_brackets", "L858R"),
        ("EGFR", "Epidermal growth factor receptor [T790M,L858R]", "substitutions_only", "square_brackets", "L858R|T790M"),
        ("EGFR", "Epidermal growth factor receptor (EGFR)(L858R)(T790M)", "substitutions_only", "parentheses", "L858R|T790M"),
        ("EGFR", "Epidermal growth factor receptor [1-745,751-1210,C797S]", "mixed_construct", "square_brackets", "C797S"),
        ("EGFR", "Epidermal growth factor receptor [1-770,'NPG',771-1210]", "other_construct", "square_brackets", ""),
        ("HER2", "Receptor tyrosine-protein kinase erbB-2 [V777L]", "substitutions_only", "square_brackets", "V777L"),
        ("HER2", "Receptor tyrosine-protein kinase erbB-2 [676-775,'YVMA',776-1255]", "other_construct", "square_brackets", ""),
        ("HER2", "Receptor tyrosine-protein kinase erbB-2", "plain_name", "none", ""),
        ("EGFR", "Unexpected EGFR string", "unclassified_nonbase", "other", ""),
    ],
)
def test_name_classification(target, name, category, notation, signature):
    assert load_module().classify_name(target, name) == (category, notation, signature)


@pytest.mark.parametrize(
    ("patent", "pmid", "doi", "expected_type", "expected_key"),
    [
        ("US123", "123", "10.1000/test", "patent", "patent:US123"),
        ("", "123", "10.1000/test", "pmid", "pmid:123"),
        ("", "", "https://doi.org/10.1000/Test", "article_doi", "doi:10.1000/test"),
        ("", "", "", "unresolved", ""),
    ],
)
def test_provenance_priority(patent, pmid, doi, expected_type, expected_key):
    value = load_module().provenance_key(pd.Series({
        "patent_number": patent, "pmid": pmid, "article_doi": doi,
    }))
    assert value == (expected_type, expected_key)


def test_small_csv_creates_report_and_figure(tmp_path):
    mod = load_module()
    rows = [
        {"reactant_set_id": "1", "target": "EGFR", "target_name": "Epidermal growth factor receptor", "source": "ChEMBL", "inchi_key": "A", "measurement_status": "exact", "article_doi": "", "pmid": "123", "patent_number": ""},
        {"reactant_set_id": "2", "target": "EGFR", "target_name": "Epidermal growth factor receptor (EGFR)(L858R)(T790M)", "source": "ChEMBL", "inchi_key": "A", "measurement_status": "exact", "article_doi": "", "pmid": "123", "patent_number": ""},
        {"reactant_set_id": "3", "target": "HER2", "target_name": "Receptor tyrosine-protein kinase erbB-2 [V777L]", "source": "US Patent", "inchi_key": "A", "measurement_status": "censored", "article_doi": "", "pmid": "", "patent_number": "US111"},
    ]
    inp = tmp_path / "observations.csv"
    pd.DataFrame(rows).to_csv(inp, index=False)
    report = tmp_path / "report.json"
    inventory = tmp_path / "inventory.csv"
    figure = tmp_path / "08.png"
    output = mod.audit(inp, report, inventory, figure)
    assert output["observations"] == 3
    assert output["reported_inchikey_overlap"]["both_targets_all_constructs"] == 1
    assert output["reported_inchikey_overlap"]["per_target"]["EGFR"]["inchikeys_with_multiple_construct_categories"] == 1
    assert report.exists() and inventory.exists() and figure.exists()
    assert len(pd.read_csv(inventory)) == 3
