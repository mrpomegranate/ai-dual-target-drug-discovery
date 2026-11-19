"""Per-target split-conformal intervals using ONLY frozen Stage C calibration.

A fresh prediction model must first be trained on ``split.fit`` only. Pass its
predictions on ``split.calibration`` here; DO NOT pass fit-only CV predictions,
threshold-role labels, development validation or locked-test labels.

Intervals target marginal coverage under exchangeability, not scaffold- or
molecule-conditional coverage. Prior adaptive model selection and distribution
shift can invalidate formal coverage. Report observed coverage independently.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from ..calibration.scaffold_calibration import ScaffoldCalibrationSplit
    from ..data.modeling_dataset import MoleculeRecord

FROZEN_STAGE_C_ASSIGNMENT_SHA256 = (
    "2b99214bce37d30e1eed8f6e22984242492560b2999bb174a8d378604de9171c"
)

Target = Literal["egfr", "her2"]


@dataclass(frozen=True)
class SplitConformalCalibrator:
    model_name: str
    target: Target
    alpha: float
    absolute_error_quantile: float
    n_labels: int
    n_scaffolds: int
    stage_c_assignment_sha256: str
    dataset_sha256: str
    outer_manifest_sha256: str

    def intervals(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        """Predict [lower, upper] for an array of point predictions."""
        point = np.asarray(values, dtype=np.float64)
        if point.ndim != 1 or not np.isfinite(point).all():
            raise ValueError("Expected a finite 1D array of point predictions")
        q = self.absolute_error_quantile
        return np.column_stack((point - q, point + q))


def conformal_quantile(residuals: Sequence[float], *, alpha: float = 0.1) -> float:
    """Conservative split-conformal order statistic: ceil((n+1)(1-alpha))."""
    if not isinstance(alpha, (float, int)) or isinstance(alpha, bool) or not (
        math.isfinite(alpha) and 0 < alpha < 1
    ):
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    scores = np.asarray(residuals, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Residuals must be a nonempty 1D array")
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Residuals must be finite nonnegative absolute errors")
    n = scores.size
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError(
            f"Only {n} calibration labels: finite {(1-alpha)*100:.1f}% "
            "split-conformal interval unavailable (required order statistic exceeds n)"
        )
    return float(np.partition(scores, rank - 1)[rank - 1])


def _validate_stage_c(split: ScaffoldCalibrationSplit, expected_hash: str) -> None:
    if split.assignment_sha256 != expected_hash:
        raise ValueError("Stage C assignment digest differs from frozen protocol")
    canonical = [
        [a.structure_id, a.scaffold_group_id, a.role]
        for a in sorted(split.assignments, key=lambda r: r.structure_id)
    ]
    computed = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if computed != split.assignment_sha256:
        raise ValueError("Stage C assignment content does not match its digest")
    roles = (split.fit, split.calibration, split.threshold)
    all_ids: set[str] = set()
    all_scaffolds: set[str] = set()
    for role in roles:
        if not role:
            raise ValueError("Stage C partition cannot be empty")
        ids = [row.structure_id for row in role]
        scaffolds = {row.scaffold_group_id for row in role}
        if len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError("Molecule identity leakage across Stage C roles")
        if all_scaffolds.intersection(scaffolds):
            raise ValueError("Scaffold leakage across Stage C roles")
        all_ids.update(ids)
        all_scaffolds.update(scaffolds)
    if len(split.assignments) != len(all_ids):
        raise ValueError("Stage C assignment count does not match role records")
    assignment_map = {a.structure_id: (a.scaffold_group_id, a.role) for a in split.assignments}
    if len(assignment_map) != len(all_ids) or set(assignment_map) != all_ids:
        raise ValueError("Stage C role assignments do not cover records uniquely")
    for role_name, records in zip(("fit", "calibration", "threshold"), roles, strict=True):
        if any(assignment_map[r.structure_id] != (r.scaffold_group_id, role_name) for r in records):
            raise ValueError("Stage C role assignment content mismatch")


def fit_split_conformal(
    split: ScaffoldCalibrationSplit,
    calibration_predictions: Mapping[str, float],
    *,
    target: Target,
    model_name: str,
    alpha: float = 0.10,
    expected_assignment_sha256: str = FROZEN_STAGE_C_ASSIGNMENT_SHA256,
) -> SplitConformalCalibrator:
    """Calibrate a FROZEN fitted model on all, and only, calibration-role IDs.

    Unmeasured target labels are excluded rather than imputed as inactive.
    Model and ensemble weights must already be selected; predictions on
    calibration cannot be used to optimize them.
    """
    if target not in ("egfr", "her2"):
        raise ValueError("Unsupported target")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string")
    _validate_stage_c(split, expected_assignment_sha256)
    calibration = split.calibration
    by_id = {record.structure_id: record for record in calibration}
    if set(calibration_predictions) != set(by_id):
        raise ValueError("Predictions must cover EXACTLY all calibration-role IDs")
    mask_name, outcome_name = f"{target}_mask", f"{target}_pic50"
    residuals: list[float] = []
    scaffold_ids: set[str] = set()
    for structure_id, record in by_id.items():
        prediction = float(calibration_predictions[structure_id])
        if not math.isfinite(prediction):
            raise ValueError("Calibration predictions must be finite")
        mask = getattr(record, mask_name)
        measured = getattr(record, outcome_name)
        if mask not in (0, 1):
            raise ValueError("Invalid target mask")
        if mask:
            if measured is None or not math.isfinite(measured):
                raise ValueError("Missing accepted calibration potency")
            residuals.append(abs(prediction - measured))
            scaffold_ids.add(record.scaffold_group_id)
        elif measured is not None:
            raise ValueError("Masked calibration potency must be absent")
    q = conformal_quantile(residuals, alpha=alpha)
    return SplitConformalCalibrator(
        model_name=model_name,
        target=target,
        alpha=float(alpha),
        absolute_error_quantile=q,
        n_labels=len(residuals),
        n_scaffolds=len(scaffold_ids),
        stage_c_assignment_sha256=split.assignment_sha256,
        dataset_sha256=split.dataset_sha256,
        outer_manifest_sha256=split.manifest_sha256,
    )


def summarize_interval_coverage(
    calibrator: SplitConformalCalibrator,
    split: ScaffoldCalibrationSplit,
    records: Sequence[MoleculeRecord],
    predictions: Mapping[str, float],
    *,
    minimum_scaffold_n: int = 10,
) -> tuple[dict, list[dict]]:
    """Score an independently designated evaluation role, NOT calibration itself.

    Caller must supply development-validation records or a separately
    authorized locked-test evaluator. Do not use these metrics to adjust the
    model, weights, intervals or other choices after test inspection.
    """
    if isinstance(minimum_scaffold_n, bool) or minimum_scaffold_n < 1:
        raise ValueError("minimum_scaffold_n must be positive")
    _validate_stage_c(split, calibrator.stage_c_assignment_sha256)
    if (calibrator.dataset_sha256 != split.dataset_sha256
            or calibrator.outer_manifest_sha256 != split.manifest_sha256):
        raise ValueError("Calibrator dataset/manifest provenance mismatch")
    records = tuple(records)
    reserved_ids = {r.structure_id for r in (*split.fit, *split.calibration, *split.threshold)}
    reserved_scaffolds = {r.scaffold_group_id for r in (*split.fit, *split.calibration, *split.threshold)}
    if any(r.structure_id in reserved_ids or r.scaffold_group_id in reserved_scaffolds
           for r in records):
        raise ValueError("Coverage evaluation overlaps Stage C fit/calibration/threshold")
    ids = [r.structure_id for r in records]
    if not records or len(ids) != len(set(ids)) or set(predictions) != set(ids):
        raise ValueError("Prediction coverage or evaluation molecule identity mismatch")
    paired: list[tuple[str, float, float, float]] = []
    for record in records:
        pred = float(predictions[record.structure_id])
        if not math.isfinite(pred):
            raise ValueError("Nonfinite predicted potency")
        if getattr(record, f"{calibrator.target}_mask") == 1:
            observed = getattr(record, f"{calibrator.target}_pic50")
            if observed is None or not math.isfinite(observed):
                raise ValueError("Missing observed potency for accepted mask")
            lower, upper = calibrator.intervals([pred])[0]
            paired.append((record.scaffold_group_id, observed, lower, upper))
    if not paired:
        raise ValueError("No accepted target labels to score")
    observed = np.array([r[1] for r in paired])
    lower = np.array([r[2] for r in paired])
    upper = np.array([r[3] for r in paired])
    hits = (observed >= lower) & (observed <= upper)
    summary = {
        "model_name": calibrator.model_name,
        "target": calibrator.target,
        "nominal_coverage": 1 - calibrator.alpha,
        "empirical_coverage": float(hits.mean()),
        "n_labeled": len(paired),
        "mean_interval_width_pic50": float(np.mean(upper - lower)),
        "scaffold_conditional_coverage_guaranteed": False,
    }
    grouped: dict[str, list[int]] = {}
    for i, (scaffold, _, _, _) in enumerate(paired):
        grouped.setdefault(scaffold, []).append(i)
    scaffold_rows: list[dict] = []
    for scaffold, indexes in sorted(grouped.items()):
        scaffold_rows.append({
            "scaffold_group_id": scaffold,
            "n": len(indexes),
            "coverage": float(np.mean(hits[indexes])),
            "mean_interval_width_pic50": float(np.mean((upper - lower)[indexes])),
            "meets_minimum_n": len(indexes) >= minimum_scaffold_n,
        })
    return summary, scaffold_rows
