"""Synthetic contract tests for M1.9a exact IC50 disagreement diagnostics."""

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_ic50_disagreement.py"
spec = importlib.util.spec_from_file_location("audit_ic50_disagreement", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def row(record_id, target, struct, nm, prov, source="ChEMBL", partition="primary_exact_candidate", status="provisional_exact_structure"):
    return {
        "reactant_set_id": str(record_id), "target": target, "target_name": "Epidermal growth factor receptor" if target == "EGFR" else "Receptor tyrosine-protein kinase erbB-2",
        "source": source, "provenance_key": prov, "m17_partition": partition,
        "m18c_status": status, "m18c_structure_id": struct,
        "measurement_status": "exact", "ic50_relation": "=", "ic50_nm": str(nm),
        "pic50": str(9 - math.log10(nm)),
        "activity_at_1000_nm": "active" if nm < 1000 else "inactive",
        "pubchem_aid": "", "ph": "", "temperature_c": "",
    }


def frame():
    return pd.DataFrame([
        row(1, "EGFR", "S1", 10, "pmid:1"),
        row(2, "EGFR", "S1", 1000, "pmid:2", source="US Patent"),
        row(3, "EGFR", "S1", 1000, "pmid:2", source="US Patent"),
        row(4, "HER2", "S1", 100, "pmid:3"),
        row(5, "EGFR", "S2", 100, "pmid:4"),
        row(6, "EGFR", "S3", 100, "pmid:5", partition="secondary_construct_reserve"),
        row(7, "EGFR", "S4", 100, "pmid:6", status="hold_rdkit_parse_failure"),
    ])


def test_selected_population():
    x = mod.prepare_primary_exact(frame())
    assert len(x) == 5


def test_pair_count():
    s, m, _ = mod.build_audit(frame())
    assert s["population"]["distinct_molecule_target_pairs"] == 3
    assert len(m) == 3


def test_repeated_group():
    s, m, _ = mod.build_audit(frame())
    assert s["repeated_pairs_total"] == 1
    a = m[(m.target == "EGFR") & (m.m18c_structure_id == "S1")].iloc[0]
    assert a.n_observations == 3
    assert a.range_pic50 == pytest.approx(2.0)
    assert bool(a.crosses_activity_cutoff)


def test_provenance_count():
    s, _, st = mod.build_audit(frame())
    assert s["pairs_with_multiple_provenance_groups"] == 1
    assert len(st) == 4


def test_duplicate_value_signature():
    s, _, _ = mod.build_audit(frame())
    assert s["exact_value_signatures_repeated_within_same_provenance"] == 1
    assert s["extra_rows_in_repeated_value_signatures"] == 1


def test_study_median_spread():
    _, m, _ = mod.build_audit(frame())
    a = m[(m.target == "EGFR") & (m.m18c_structure_id == "S1")].iloc[0]
    assert a.study_median_range_pic50 == pytest.approx(2.0)


def test_missing_required_column():
    with pytest.raises(ValueError, match="Missing required"):
        mod.prepare_primary_exact(frame().drop(columns=["provenance_key"]))


def test_duplicate_record_rejected():
    d = frame()
    d.loc[1, "reactant_set_id"] = "1"
    with pytest.raises(ValueError, match="must be non-null and unique"):
        mod.prepare_primary_exact(d)


def test_bad_pic50_rejected():
    d = frame()
    d.loc[0, "pic50"] = "999"
    with pytest.raises(ValueError, match="do not match"):
        mod.prepare_primary_exact(d)


def test_bad_activity_rejected():
    d = frame()
    d.loc[0, "activity_at_1000_nm"] = "inactive"
    with pytest.raises(ValueError, match="Activity labels"):
        mod.prepare_primary_exact(d)


def test_non_exact_rejected():
    d = frame()
    d.loc[0, "ic50_relation"] = ">"
    with pytest.raises(ValueError, match="Non-exact IC50 qualifier"):
        mod.prepare_primary_exact(d)


def test_no_eligible_data():
    d = frame()
    d["m17_partition"] = "secondary_construct_reserve"
    with pytest.raises(ValueError, match="No eligible"):
        mod.prepare_primary_exact(d)


def test_bad_target_rejected():
    d = frame()
    d.loc[0, "target"] = "PARP1"
    with pytest.raises(ValueError, match="Unexpected target"):
        mod.prepare_primary_exact(d)


def test_no_training_labels_created():
    _, m, st = mod.build_audit(frame())
    assert "curated_pic50" not in m.columns
    assert "curated_pic50" not in st.columns


def test_keeps_targets_separate():
    _, m, _ = mod.build_audit(frame())
    assert m[m.m18c_structure_id == "S1"].target.nunique() == 2
