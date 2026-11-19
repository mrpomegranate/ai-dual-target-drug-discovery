"""Masked, molecule-aligned evaluation for fit-only molecular GNN CV.

Input rows are out-of-fold predictions from the existing Stage C fit-only
partition. No calibration, threshold, dev-validation, or test records are read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

TARGETS = ("egfr", "her2")
ACTIVITY_THRESHOLD = 6.0


def score_oof_rows(rows: Sequence[Mapping]) -> dict[str, float | int | None]:
    """Compute mask-safe pooled (or single-fold) potency/ranking metrics.

    Missing labels never contribute to a regression metric or paired class.
    Dual activity is evaluated only for molecules measured on both targets.
    The dual ranking score is min(predicted EGFR, predicted HER2) pIC50.
    """
    if not rows:
        raise ValueError("Cannot score an empty out-of-fold prediction set")
    ids = [str(row["structure_id"]) for row in rows]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("Repeated or empty molecule identity in OOF predictions")

    results: dict[str, float | int | None] = {"molecules": len(rows)}
    for target in TARGETS:
        actual: list[float] = []
        predicted: list[float] = []
        for row in rows:
            prediction = float(row[f"{target}_pred"])
            if not math.isfinite(prediction):
                raise ValueError(f"Nonfinite {target} prediction")
            mask = row[f"{target}_mask"]
            if type(mask) is not bool:
                raise TypeError(f"{target}_mask must be bool")
            if mask:
                value = row[f"{target}_pic50"]
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"Missing or nonfinite measured {target} label")
                actual.append(float(value))
                predicted.append(prediction)
            elif row[f"{target}_pic50"] is not None:
                raise ValueError(f"Masked {target} label must be None")
        results[f"{target}_n"] = len(actual)
        if not actual:
            raise ValueError(f"No {target} labels in evaluation cohort")
        a = np.asarray(actual)
        p = np.asarray(predicted)
        results[f"{target}_mae"] = float(np.abs(a - p).mean())
        results[f"{target}_spearman"] = (
            float(spearmanr(a, p).statistic)
            if len(a) > 1 and np.ptp(a) > 0 and np.ptp(p) > 0
            else None
        )

    paired = [row for row in rows if row["egfr_mask"] and row["her2_mask"]]
    results["paired_n"] = len(paired)
    labels = [
        int(row["egfr_pic50"] > ACTIVITY_THRESHOLD
            and row["her2_pic50"] > ACTIVITY_THRESHOLD)
        for row in paired
    ]
    results["dual_active_n"] = sum(labels)
    results["dual_inactive_n"] = len(labels) - sum(labels)
    results["dual_prevalence"] = sum(labels) / len(labels) if labels else None
    results["dual_min_prediction_ap"] = (
        float(average_precision_score(
            labels,
            [min(row["egfr_pred"], row["her2_pred"]) for row in paired],
        ))
        if labels and 0 < sum(labels) < len(labels) else None
    )
    results["balanced_mae"] = (results["egfr_mae"] + results["her2_mae"]) / 2
    return results


def verify_oof_coverage(
    rows: Sequence[Mapping], expected_ids: Sequence[str], expected_folds: int = 5
) -> None:
    """Ensure every fit-role molecule is scored once on its CV holdout."""
    expected = tuple(expected_ids)
    found = tuple(str(row["structure_id"]) for row in rows)
    if len(expected) != len(set(expected)) or set(found) != set(expected):
        raise ValueError("Out-of-fold molecules differ from fit-only identities")
    if len(found) != len(set(found)):
        raise ValueError("A fit molecule was scored in multiple folds")
    folds = {int(row["fold"]) for row in rows}
    if folds != set(range(expected_folds)):
        raise ValueError("Missing or unexpected scaffold CV fold numbers")
