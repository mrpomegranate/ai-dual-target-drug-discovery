"""Pure evaluation tests; no GNN training or reserved labels."""

from __future__ import annotations

import pytest
from dual_target.evaluation.gnn_cv_metrics import score_oof_rows, verify_oof_coverage


def _row(sid, fold, em, hm, ey, hy, ep, hp):
    return {
        "structure_id": sid, "fold": fold,
        "egfr_mask": em, "her2_mask": hm,
        "egfr_pic50": ey, "her2_pic50": hy,
        "egfr_pred": ep, "her2_pred": hp,
    }


def test_missing_label_excluded_from_regression_and_dual_classification():
    rows = [
        _row("A", 0, True, True, 7., 7., 7., 7.),
        _row("B", 0, True, True, 5., 7., 5., 7.),
        _row("C", 0, True, False, 6., None, 5., -100.),
    ]
    scored = score_oof_rows(rows)
    assert scored["egfr_n"] == 3
    assert scored["her2_n"] == 2
    assert scored["paired_n"] == 2
    assert scored["dual_active_n"] == 1
    assert scored["egfr_mae"] == pytest.approx(1/3)
    assert scored["her2_mae"] == 0
    assert scored["dual_min_prediction_ap"] == 1.


def test_missing_or_inconsistent_label_rejected():
    with pytest.raises(ValueError, match="Masked"):
        score_oof_rows([_row("A", 0, True, False, 7., 8., 7., 8.)])


def test_duplicate_prediction_id_rejected():
    row = _row("A", 0, True, True, 7., 7., 7., 7.)
    with pytest.raises(ValueError, match="Repeated"):
        score_oof_rows([row, row])


def test_oof_integrity_checks_missing_and_repeated():
    rows = [_row(f"m{i}", i, True, True, 7., 7., 7., 7.) for i in range(5)]
    verify_oof_coverage(rows, [f"m{i}" for i in range(5)])
    with pytest.raises(ValueError, match="scored in multiple"):
        verify_oof_coverage([*rows, rows[0]], [f"m{i}" for i in range(5)])
    with pytest.raises(ValueError, match="differ"):
        verify_oof_coverage(rows[:-1], [f"m{i}" for i in range(5)])


def test_constant_predictions_have_null_spearman():
    rows = [
        _row("A", 0, True, True, 6., 7., 5., 5.),
        _row("B", 0, True, True, 7., 8., 5., 5.),
    ]
    scored = score_oof_rows(rows)
    assert scored["egfr_spearman"] is None
    assert scored["her2_spearman"] is None
