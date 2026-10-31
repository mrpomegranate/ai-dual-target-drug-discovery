"""Small ZIP fixtures prove that audit filtering preserves evidence."""

import csv
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_bindingdb_observations.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bindingdb_audit_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_streamed_audit_keeps_exact_censored_and_invalid(tmp_path):
    audit = load_module()
    columns = list(audit.FIELDS.values())
    cols = dict.fromkeys(columns, "")

    def record(rowid, accession, chains, ic50, smiles="CCO", organism="Homo sapiens"):
        return {
            **cols,
            "BindingDB Reactant_set_id": str(rowid),
            "BindingDB MonomerID": str(rowid),
            "Ligand SMILES": smiles,
            audit.FIELDS["uniprot"]: accession,
            audit.FIELDS["chain_count"]: chains,
            audit.FIELDS["ic50_raw"]: ic50,
            audit.FIELDS["organism"]: organism,
        }

    records = [
        record(1, "P00533", "1", "800"),
        record(2, "P00533", "1", "<1000"),
        record(3, "P04626", "1", "14", organism="Human"),
        record(4, "P04626", "1", "garbled"),
        record(5, "P00533", "2", "200"),
        record(6, "P00533", "1", ""),
        record(7, "OTHER", "1", "900"),
        record(8, "P04626", "1", ">1000", smiles="", organism=""),
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
    writer.writeheader()
    writer.writerows(records)
    input_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(input_zip, "w") as z:
        z.writestr("BindingDB_All.tsv", stream.getvalue())

    obs = tmp_path / "interim" / "observations.csv"
    report = tmp_path / "reports" / "audit.json"
    result = audit.audit(input_zip, obs, report, progress_every=0)
    with obs.open(encoding="utf-8", newline="") as f:
        extracted = list(csv.DictReader(f))
    saved = json.loads(report.read_text(encoding="utf-8"))

    assert len(extracted) == 5
    assert result["totals"]["data_records_scanned"] == 8
    assert result["totals"]["excluded_not_single_chain"] == 1
    assert result["totals"]["excluded_missing_ic50"] == 1
    assert result["totals"]["ic50_exact"] == 2
    assert result["totals"]["ic50_censored"] == 2
    assert result["totals"]["ic50_invalid"] == 1
    assert saved["totals"] == result["totals"]
    assert extracted[0]["activity_at_1000_nm"] == "active"
    assert float(extracted[0]["pic50"]) == pytest.approx(6.096910013)
    assert extracted[1]["pic50"] == ""
    assert extracted[1]["activity_at_1000_nm"] == "active"
    assert extracted[2]["organism_flag"] == "human"
    assert extracted[3]["measurement_status"] == "invalid"
    assert extracted[3]["pic50"] == ""
    assert extracted[4]["organism_flag"] == "missing"
    assert extracted[4]["smiles_flag"] == "missing"


def test_duplicate_headers_fail(tmp_path):
    audit = load_module()
    path = tmp_path / "dup.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("BindingDB_All.tsv", "IC50 (nM)\tIC50 (nM)\n")
    with pytest.raises(ValueError, match="Duplicate headers"):
        audit.audit(path, tmp_path / "out.csv", tmp_path / "summary.json", 0)
