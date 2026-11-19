"""Synthetic identity/mask/scaffold alignment tests; no real dataset needed."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from dual_target.evaluation.matched_oof import (
    align_rf_gnn_oof,
    compare_aligned,
    read_gnn_oof_csv,
    score_aligned,
    write_aligned_csv,
)


@dataclass(frozen=True)
class Record:
    structure_id: str
    scaffold_group_id: str
    egfr_mask: int
    her2_mask: int
    egfr_pic50: float | None
    her2_pic50: float | None


def fixtures():
    records = (
        Record("A", "scafA", 1, 1, 7.0, 7.0),
        Record("B", "scafB", 1, 0, 6.2, None),
        Record("C", "scafC", 0, 1, None, 5.3),
        Record("D", "scafD", 1, 1, 5.2, 7.0),
    )
    pipeline = SimpleNamespace(
        data=SimpleNamespace(train=records),
        folds=(SimpleNamespace(number=0, holdout=records[:2]),
               SimpleNamespace(number=1, holdout=records[2:])),
    )
    def oof(r, fold, target, predicted):
        return SimpleNamespace(
            structure_id=r.structure_id,
            scaffold_group_id=r.scaffold_group_id,
            fold=fold,
            observed=getattr(r, f"{target}_pic50"),
            predicted=predicted,
        )
    rf_e = SimpleNamespace(task="egfr", oof_predictions=(
        oof(records[0], 0, "egfr", 6.7),
        oof(records[1], 0, "egfr", 6.1),
        oof(records[3], 1, "egfr", 5.4),
    ))
    rf_h = SimpleNamespace(task="her2", oof_predictions=(
        oof(records[0], 0, "her2", 6.9),
        oof(records[2], 1, "her2", 5.8),
        oof(records[3], 1, "her2", 6.8),
    ))
    gnn = [
        {"structure_id": r.structure_id, "fold": (0 if i < 2 else 1),
         "egfr_mask": bool(r.egfr_mask), "her2_mask": bool(r.her2_mask),
         "egfr_pic50": r.egfr_pic50, "her2_pic50": r.her2_pic50,
         "egfr_pred": 6.8, "her2_pred": 6.5}
        for i, r in enumerate(records)
    ]
    return pipeline, rf_e, rf_h, gnn


def test_matched_alignment_and_molecule_level_metrics():
    args = fixtures()
    matched = align_rf_gnn_oof(*args)
    assert [row["structure_id"] for row in matched] == ["A", "B", "C", "D"]
    assert matched[1]["her2_rf_pred"] is None
    assert matched[2]["egfr_rf_pred"] is None
    assert matched[0]["scaffold_group_id"] == "scafA"
    assert score_aligned(matched, "rf")["egfr_n"] == 3
    assert score_aligned(matched, "gnn")["her2_n"] == 3
    summary = compare_aligned(matched)
    assert len(summary["folds"]) == 2
    assert summary["pooled"]["rf"]["paired_n"] == 2
    assert summary["pooled"]["rf"]["dual_min_prediction_ap"] is not None


def test_small_float32_rounding_is_accepted():
    pipe, e, h, gnn = fixtures()
    gnn[0]["egfr_pic50"] += 0.0000001
    assert len(align_rf_gnn_oof(pipe, e, h, gnn)) == 4


@pytest.mark.parametrize("change,expected", [
    ("wrong_fold", "fold mismatch"),
    ("wrong_mask", "mask mismatch"),
    ("wrong_label", "label mismatch"),
    ("missing_id", "identities differ"),
    ("duplicate_id", "Duplicate"),
])
def test_alignment_rejects_silent_data_corruption(change, expected):
    pipe, e, h, gnn = fixtures()
    if change == "wrong_fold":
        gnn[0]["fold"] = 1
    elif change == "wrong_mask":
        gnn[0]["her2_mask"] = False
    elif change == "wrong_label":
        gnn[0]["egfr_pic50"] = 5.0
    elif change == "missing_id":
        gnn.pop()
    else:
        gnn.append(gnn[0].copy())
    with pytest.raises(ValueError, match=expected):
        align_rf_gnn_oof(pipe, e, h, gnn)


def test_rf_scaffold_mismatch_fails():
    pipe, e, h, gnn = fixtures()
    entries = list(e.oof_predictions)
    entries[0] = SimpleNamespace(**{
        **vars(entries[0]), "scaffold_group_id": "OTHER"
    })
    with pytest.raises(ValueError, match="fold/scaffold mismatch"):
        align_rf_gnn_oof(pipe, SimpleNamespace(task="egfr", oof_predictions=entries), h, gnn)


def test_gnn_csv_roundtrip(tmp_path):
    pipe, e, h, gnn = fixtures()
    input_path = tmp_path / "gnn.csv"
    with input_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gnn[0]))
        writer.writeheader()
        writer.writerows([{k: "" if v is None else v for k, v in r.items()} for r in gnn])
    parsed = read_gnn_oof_csv(input_path)
    matched = align_rf_gnn_oof(pipe, e, h, parsed)
    out_path = tmp_path / "matched.csv"
    write_aligned_csv(out_path, matched)
    assert out_path.exists()
    assert len(out_path.read_text().splitlines()) == 5


def test_gnn_csv_rejects_missing_measured_label(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "structure_id,fold,egfr_mask,her2_mask,egfr_pic50,her2_pic50,egfr_pred,her2_pred\n"
        "A,0,True,False,,,7.0,6.0\n", encoding="utf8"
    )
    with pytest.raises(ValueError, match="Missing measured"):
        read_gnn_oof_csv(path)
