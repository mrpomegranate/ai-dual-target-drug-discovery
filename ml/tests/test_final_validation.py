"""M3.7c pure scoring and safety tests (do not require models or GPU)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dual_target.evaluation.final_validation import (
    make_detail_rows,
    score_dual_activity,
    score_potency_bins,
    score_scaffolds,
    score_target,
    validate_prediction_table,
)


def _toy():
    frame = pd.DataFrame({
        "structure_id": ["a", "b", "c", "d"],
        "scaffold_group_id": ["s1", "s1", "s2", "s3"],
        "egfr_mask": [1, 1, 1, 0],
        "her2_mask": [1, 1, 0, 1],
        "egfr_pic50": [8., 5., 7., np.nan],
        "her2_pic50": [8., 5., np.nan, 7.],
        "egfr_rf_pred": [7.5, 5.5, 7., 8.],
        "her2_rf_pred": [7.5, 5.5, 7., 7.],
        "egfr_gine_pred": [7., 6., 7., 8.],
        "her2_gine_pred": [7., 6., 7., 7.],
    })
    for target in ("egfr", "her2"):
        frame[f"{target}_rf75_gine25_pred"] = (
            .75 * frame[f"{target}_rf_pred"] + .25 * frame[f"{target}_gine_pred"]
        )
    return frame


def test_correct_masks_and_weighted_ensemble():
    validate_prediction_table(_toy(), expected_n=4)


def test_detect_changed_ensemble():
    frame = _toy()
    frame.loc[0, "her2_rf75_gine25_pred"] += .01
    with pytest.raises(ValueError, match="75/25"):
        validate_prediction_table(frame, expected_n=4)


def test_detect_duplicate_molecules():
    frame = _toy()
    frame.loc[1, "structure_id"] = "a"
    with pytest.raises(ValueError, match="identity"):
        validate_prediction_table(frame, expected_n=4)


def test_detect_masked_nonmissing_label():
    frame = _toy()
    frame.loc[3, "egfr_pic50"] = 5.
    with pytest.raises(ValueError, match="masks"):
        validate_prediction_table(frame, expected_n=4)


def test_detect_missing_prediction():
    frame = _toy()
    frame.loc[0, "egfr_rf_pred"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        validate_prediction_table(frame, expected_n=4)


def test_scores_coverage_and_error():
    summary, detail = score_target(_toy(), target="egfr", model="rf", quantile=.5)
    assert summary["n_labeled"] == 3
    assert summary["mae"] == pytest.approx(1/3)
    assert summary["empirical_coverage"] == 1.
    assert summary["mean_interval_width_pic50"] == 1.
    assert detail.covered.all()


def test_different_quantiles_change_coverage_not_predictions():
    _, low = score_target(_toy(), target="egfr", model="rf", quantile=0.)
    _, high = score_target(_toy(), target="egfr", model="rf", quantile=1.)
    assert low.covered.sum() == 1 and high.covered.sum() == 3
    np.testing.assert_array_equal(low.predicted, high.predicted)


def test_scaffold_minimum_n_flag():
    _, detail = score_target(_toy(), target="egfr", model="rf", quantile=.5)
    scaffolds = score_scaffolds(detail, minimum_n=2)
    assert len(scaffolds) == 2
    assert scaffolds.set_index("scaffold_group_id").loc["s1", "meets_minimum_n"]
    assert not scaffolds.set_index("scaffold_group_id").loc["s2", "meets_minimum_n"]


def test_potency_subgroups_no_fabricated_masked_label():
    _, detail = score_target(_toy(), target="egfr", model="rf", quantile=.5)
    groups = score_potency_bins(detail)
    assert groups.n.sum() == 3
    assert groups.loc[groups.observed_pic50_bin == "≥8", "n"].item() == 1


def test_paired_ranking_uses_only_double_labeled():
    result = score_dual_activity(_toy())
    assert all(v["paired_n"] == 2 and v["dual_active_n"] == 1 for v in result.values())
    assert all(v["regression_min_pic50_ap"] == 1. for v in result.values())


def test_details_do_not_include_unmeasured_targets():
    _, egfr = score_target(_toy(), target="egfr", model="rf", quantile=.5)
    _, her2 = score_target(_toy(), target="her2", model="rf", quantile=.5)
    result = make_detail_rows([egfr, her2])
    assert len(result) == 6
    assert not result.observed.isna().any()


def test_bad_quantile():
    with pytest.raises(ValueError, match="quantile"):
        score_target(_toy(), target="egfr", model="rf", quantile=float("nan"))
