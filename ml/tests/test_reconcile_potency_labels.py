"""Synthetic-only regression tests for the M1.10 policy."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from reconcile_potency_labels import ACCEPT, reconcile


def make_row(record, target="EGFR", sid="cmpd-A", value=100.0, role="primary_exact_candidate",
             provenance="pmid:A", activity=None, status="provisional_exact_structure"):
    return {
        "reactant_set_id": str(record),
        "target": target,
        "m18c_structure_id": sid,
        "m18c_status": status,
        "m17_partition": role,
        "ic50_nm": value,
        "pic50": 9 - np.log10(value) if role == "primary_exact_candidate" else np.nan,
        "activity_at_1000_nm": activity or ("active" if value < 1000 else "inactive"),
        "provenance_key": provenance,
        "source": "ChEMBL",
        "m18c_candidate_isomeric_smiles": "CCO" if sid == "cmpd-A" else "CCC",
        "ic50_raw": str(value),
    }


def run(rows):
    return reconcile(pd.DataFrame(rows))


def only(rows):
    return run(rows)[0].iloc[0]


def test_single_exact_provisional():
    row = only([make_row("1")])
    assert row.m10_policy_status == "provisional_single_exact"
    assert row.m10_provisional_pic50 == pytest.approx(7.0)
    assert row.m10_label_support == "unreplicated_unverified"


def test_two_close_measurements_consensus():
    row = only([make_row("1", value=100), make_row("2", value=140)])
    assert row.m10_policy_status == "provisional_tight_consensus"
    assert row.m10_provisional_pic50 == pytest.approx((7 + 9 - np.log10(140)) / 2)


def test_moderate_disagreement_held():
    row = only([make_row("1", value=10), make_row("2", value=30)])
    assert row.m10_policy_status == "hold_moderate_disagreement"
    assert np.isnan(row.m10_provisional_pic50)


def test_large_disagreement_held():
    row = only([make_row("1", value=10), make_row("2", value=110)])
    assert row.m10_policy_status == "hold_major_disagreement_gt1_pic50"


def test_exact_threshold_contradiction_held():
    row = only([make_row("1", value=999), make_row("2", value=1000)])
    assert row.m10_policy_status == "hold_exact_activity_contradiction"


def test_censored_opposite_activity_vetoes_singleton():
    row = only([make_row("1", value=100), make_row("2", value=2000, role="primary_classification_candidate", activity="inactive")])
    assert row.m10_policy_status == "hold_censored_activity_contradiction"
    assert row.n_censored_classification_observations == 1


def test_censored_same_activity_does_not_veto():
    row = only([make_row("1", value=100), make_row("2", value=500, role="primary_classification_candidate", activity="active")])
    assert row.m10_policy_status == "provisional_single_exact"


def test_target_never_merged():
    labels, *_ = run([make_row("1"), make_row("2", target="HER2")])
    assert len(labels) == 2


def test_structure_id_never_merged():
    labels, *_ = run([make_row("1"), make_row("2", sid="cmpd-B")])
    assert len(labels) == 2


def test_annotated_reserve_not_training_label():
    labels, _, trace, summary = run([make_row("1"), make_row("2", role="secondary_construct_reserve")])
    assert len(labels) == 1
    assert len(trace) == 1
    assert bool(labels.iloc[0].has_annotated_construct_for_same_structure_target)
    assert summary["candidate_molecule_target_pairs"] == 1


def test_held_structure_not_used():
    labels, *_ = run([make_row("1", status="hold_inchikey_connectivity_disagreement"), make_row("2", sid="cmpd-B")])
    assert len(labels) == 1


def test_provenance_median_of_medians_not_raw_weighted():
    rows = [make_row(str(i), value=100, provenance="A") for i in range(1, 5)]
    rows.append(make_row("5", value=125, provenance="B"))
    row = only(rows)
    assert row.n_provenance_keys == 2
    assert row.provenance_median_of_medians_pic50 != pytest.approx(row.unadjusted_observation_median_pic50)


def test_censored_trace_kept():
    _, _, trace, _ = run([make_row("1"), make_row("2", role="primary_classification_candidate")])
    assert len(trace) == 2
    assert set(trace.m10_exact_pair_decision) == {"provisional_single_exact"}


def test_all_tiers_reconcile():
    rows = [make_row("1"), make_row("2", sid="cmpd-B")]
    labels, _, _, summary = run(rows)
    assert summary["provisional_regression_labels"] + summary["held_regression_pairs"] == len(labels)
    assert all(x in ACCEPT for x in labels.m10_policy_status)


def test_invalid_exact_pic50_rejected():
    row = make_row("1")
    row["pic50"] = 4
    with pytest.raises(ValueError, match="inconsistent"):
        run([row])


def test_duplicate_reactant_id_rejected():
    with pytest.raises(ValueError, match="unique"):
        run([make_row("1"), make_row("1")])


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="Missing"):
        reconcile(pd.DataFrame({"target": ["EGFR"]}))


def test_invalid_cutoffs_rejected():
    with pytest.raises(ValueError, match="cutoff"):
        reconcile(pd.DataFrame([make_row("1")]), tight_max_pic50_range=0.7, relaxed_max_pic50_range=0.4)


def test_sensitivity_relaxed_accepts_more():
    labels, sensitivity, _, _ = run([make_row("1", value=10), make_row("2", value=30)])
    assert labels.iloc[0].m10_policy_status == "hold_moderate_disagreement"
    assert bool(sensitivity.iloc[0].sensitivity_relaxed_0_50_accepted)


def test_sensitivity_ignoring_censored_reveals_risk():
    _, sensitivity, _, _ = run([
        make_row("1", value=100),
        make_row("2", value=2000, role="primary_classification_candidate", activity="inactive"),
    ])
    assert bool(sensitivity.iloc[0].sensitivity_ignore_censored_accepted)
    assert not bool(sensitivity.iloc[0].sensitivity_strict_0_30_accepted)
