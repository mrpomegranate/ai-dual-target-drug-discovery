
"""Target-specific potency regression and direct dual-activity classification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error

from ..data.modeling_dataset import MoleculeRecord
from ..morgan_features import make_morgan_fingerprints

Target = Literal["egfr", "her2"]


@dataclass(frozen=True)
class RFConfig:
    """Common fingerprint and Random Forest parameters for both approaches."""

    radius: int = 2
    n_bits: int = 2048
    include_chirality: bool = True
    n_estimators: int = 200  # Fast initial run; reference run can use 1,000.
    max_depth: int | None = None
    min_samples_leaf: int = 1
    min_samples_split: int = 2
    max_features: str | float = "sqrt"
    seed: int = 714
    n_jobs: int = -1


@dataclass(frozen=True)
class TargetExamples:
    molecule_ids: tuple[str, ...]
    scaffold_ids: tuple[str, ...]
    features: np.ndarray
    pic50: np.ndarray


@dataclass(frozen=True)
class DualActivityExamples:
    molecule_ids: tuple[str, ...]
    scaffold_ids: tuple[str, ...]
    features: np.ndarray
    dual_active: np.ndarray


def _features(records: Sequence[MoleculeRecord], config: RFConfig) -> np.ndarray:
    return make_morgan_fingerprints(
        [record.smiles for record in records],
        radius=config.radius,
        n_bits=config.n_bits,
        include_chirality=config.include_chirality,
    )


def _validate_records(records: Sequence[MoleculeRecord]) -> None:
    ids = [record.structure_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate molecule IDs detected")


def make_target_examples(
    records: Sequence[MoleculeRecord], target: Target, config: RFConfig
) -> TargetExamples:
    """Use only accepted pIC50 labels for the selected protein."""
    if target not in ("egfr", "her2"):
        raise ValueError(f"Unsupported target: {target}")

    selected = [record for record in records if getattr(record, f"{target}_mask") == 1]
    if not selected:
        raise ValueError(f"No labeled molecules for {target}")
    _validate_records(selected)

    labels = np.asarray(
        [getattr(record, f"{target}_pic50") for record in selected],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(labels)):
        raise ValueError("Missing or nonfinite target labels")

    return TargetExamples(
        molecule_ids=tuple(record.structure_id for record in selected),
        scaffold_ids=tuple(record.scaffold_group_id for record in selected),
        features=_features(selected, config),
        pic50=labels,
    )


def make_dual_activity_examples(
    records: Sequence[MoleculeRecord], config: RFConfig, threshold_pic50: float = 6.0
) -> DualActivityExamples:
    """Binary dual activity from molecules measured against both targets.

    A positive label requires EGFR and HER2 pIC50 strictly above 6.0.
    Unpaired molecules are excluded, never labeled inactive.
    """
    if not np.isfinite(threshold_pic50):
        raise ValueError("Activity threshold must be finite")

    selected = [
        record
        for record in records
        if record.egfr_mask == 1 and record.her2_mask == 1
    ]
    if not selected:
        raise ValueError("No paired molecules with accepted measurements")
    _validate_records(selected)

    egfr = np.asarray([r.egfr_pic50 for r in selected], dtype=np.float64)
    her2 = np.asarray([r.her2_pic50 for r in selected], dtype=np.float64)
    if not np.all(np.isfinite(egfr)) or not np.all(np.isfinite(her2)):
        raise ValueError("Missing or nonfinite paired target labels")

    labels = ((egfr > threshold_pic50) & (her2 > threshold_pic50)).astype(
        np.int8
    )
    return DualActivityExamples(
        molecule_ids=tuple(record.structure_id for record in selected),
        scaffold_ids=tuple(record.scaffold_group_id for record in selected),
        features=_features(selected, config),
        dual_active=labels,
    )


def fit_random_forest(
    train: TargetExamples, config: RFConfig
) -> RandomForestRegressor:
    """Fit one target-specific potency regressor using training rows."""
    if len(train.pic50) == 0 or train.features.shape[0] != len(train.pic50):
        raise ValueError("Missing or inconsistent regression training examples")

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        min_samples_split=config.min_samples_split,
        max_features=config.max_features,
        random_state=config.seed,
        n_jobs=config.n_jobs,
    )
    model.fit(train.features, train.pic50)
    return model


def evaluate_random_forest(
    model: RandomForestRegressor, validation: TargetExamples
) -> tuple[dict[str, float | int | None], np.ndarray]:
    """Return pIC50 validation MAE, Spearman rho, and aligned predictions."""
    if len(validation.pic50) == 0:
        raise ValueError("No regression validation examples")

    predictions = model.predict(validation.features)
    mae = float(mean_absolute_error(validation.pic50, predictions))

    if (
        len(validation.pic50) < 2
        or np.ptp(validation.pic50) == 0
        or np.ptp(predictions) == 0
    ):
        rho = None
    else:
        statistic = float(spearmanr(validation.pic50, predictions).statistic)
        rho = statistic if np.isfinite(statistic) else None

    metrics = {"n": len(validation.pic50), "mae": mae, "spearman": rho}
    return metrics, predictions


def fit_dual_activity_classifier(
    train: DualActivityExamples, config: RFConfig
) -> RandomForestClassifier:
    """Train one binary classifier using measured paired training molecules."""
    if len(train.dual_active) == 0 or train.features.shape[0] != len(train.dual_active):
        raise ValueError("Missing or inconsistent classifier training examples")
    if set(np.unique(train.dual_active)) != {0, 1}:
        raise ValueError("Classifier training needs both dual-active and negative rows")

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        min_samples_split=config.min_samples_split,
        max_features=config.max_features,
        random_state=config.seed,
        n_jobs=config.n_jobs,
    )
    model.fit(train.features, train.dual_active)
    return model


def evaluate_dual_activity_classifier(
    model: RandomForestClassifier, validation: DualActivityExamples
) -> tuple[dict[str, float | int], np.ndarray]:
    """Return validation average precision and continuous dual-activity scores."""
    labels = validation.dual_active
    if len(labels) == 0 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Classifier validation needs both positive and negative rows")

    positive_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(validation.features)[:, positive_column]
    metrics = {
        "n": len(labels),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }
    return metrics, probabilities
