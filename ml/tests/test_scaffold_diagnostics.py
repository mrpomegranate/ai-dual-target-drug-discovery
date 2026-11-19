"""Scaffold audit regressions with tiny matched OOF fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dual_target.evaluation.scaffold_diagnostics import (
    rank_difficult_scaffolds,
    scaffold_error_table,
    summarize_scaffold_diagnostics,
    validate_fit_only_oof,
)


def frame():
    rows = []
    for i, (sid, group, fold, egfr, her2) in enumerate([
        ("a", "s0", 0, 7.0, 6.0),
        ("b", "s0", 0, 8.0, None),
        ("c", "s1", 1, 6.0, 7.0),
        ("d", "s1", 1, 5.0, 8.0),
    ]):
        row = {"structure_id": sid, "scaffold_group_id": group, "fold": fold,
               "egfr_mask": True, "egfr_pic50": egfr,
               "her2_mask": her2 is not None, "her2_pic50": her2}
        for target in ("egfr", "her2"):
            y = row[f"{target}_pic50"]
            for key, offset in (("rf", .2), ("gine", .6), ("gatv2", .4)):
                row[f"{target}_{key}_pred"] = (y + offset) if y is not None else np.nan
            row[f"{target}_rf75_gine25_pred"] = (
                .75 * row[f"{target}_rf_pred"] + .25 * row[f"{target}_gine_pred"]
                if y is not None else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_scaffold_audit_has_expected_label_counts():
    f = frame()
    validate_fit_only_oof(f, expected_n=4)
    groups = scaffold_error_table(f, minimum_n=2)
    summary = summarize_scaffold_diagnostics(f, groups)
    assert summary["scaffold_groups"] == 2
    assert summary["targets"]["egfr"]["n_labeled"] == 4
    assert summary["targets"]["her2"]["n_labeled"] == 3
    assert summary["targets"]["egfr"]["reportable_scaffolds"] == 2
    assert summary["targets"]["her2"]["reportable_scaffolds"] == 1
    assert summary["targets"]["egfr"]["pooled_mae"]["rf"] == pytest.approx(.2)
    assert summary["locked_test_scored"] is False


def test_ranking_has_target_specific_minimum_label_count():
    groups = scaffold_error_table(frame(), minimum_n=2)
    ranked = rank_difficult_scaffolds(groups, top_k=2)
    assert set(ranked.target) == {"egfr", "her2"}
    assert len(ranked[ranked.target == "her2"]) == 1


def test_no_group_cross_fold_allowed():
    f = frame()
    f.loc[1, "fold"] = 1
    with pytest.raises(ValueError, match="multiple folds"):
        validate_fit_only_oof(f)


def test_ensemble_cannot_silently_change():
    f = frame()
    f.loc[0, "egfr_rf75_gine25_pred"] = 200
    with pytest.raises(ValueError, match="frozen formula"):
        validate_fit_only_oof(f)


def test_masked_target_labels_must_be_missing():
    f = frame()
    f.loc[1, "her2_pic50"] = 0
    with pytest.raises(ValueError, match="masks"):
        validate_fit_only_oof(f)


def test_duplicate_ids_rejected():
    f = frame()
    f.loc[1, "structure_id"] = "a"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_fit_only_oof(f)
