
"""Training-only Optuna tuning for Morgan Random Forest baselines.

Uses the frozen M2.1 training partition and its scaffold-disjoint internal
folds. Does not access external validation or test labels, fit calibrators,
or optimize decision thresholds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import optuna
from optuna.trial import FrozenTrial

from ..data.pipeline import ModelingPipeline, Task
from ..evaluation.scaffold_cv import fold_manifest
from ..features.morgan_pipeline import MorganFeatureBank
from ..models.random_forest import RFConfig
from .rf_cv import RFCVResult, run_rf_scaffold_cv


@dataclass(frozen=True)
class RFOptimizationResult:
    """Completed study and selected RF configuration."""

    study: optuna.Study
    task: Task
    best_config: RFConfig
    best_score: float
    metric_name: str


def _fold_manifest_hash(pipeline: ModelingPipeline) -> str:
    """Hash stable molecule-to-internal-fold assignments."""
    assignments = [
        asdict(item) for item in fold_manifest(pipeline.folds)
    ]
    encoded = json.dumps(
        assignments, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _suggest_config(
    trial: optuna.Trial,
    base: RFConfig,
) -> RFConfig:
    """Search only parameters supported by the existing RF module."""
    return replace(
        base,
        max_depth=trial.suggest_categorical(
            "max_depth", [8, 12, 20, 30, None]
        ),
        min_samples_leaf=trial.suggest_categorical(
            "min_samples_leaf", [1, 2, 4, 8, 16]
        ),
        min_samples_split=trial.suggest_int(
            "min_samples_split", 2, 20
        ),
        max_features=trial.suggest_categorical(
            "max_features", ["sqrt", 0.3, 0.5, 0.8]
        ),
    )


def _config_from_params(
    base: RFConfig,
    params: dict[str, Any],
) -> RFConfig:
    return replace(
        base,
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_split"]),
        max_features=params["max_features"],
    )


@dataclass(frozen=True)
class RFOptunaObjective:
    """Optuna objective using only internal scaffold-CV folds."""

    pipeline: ModelingPipeline
    bank: MorganFeatureBank
    task: Task
    base_config: RFConfig

    def __call__(self, trial: optuna.Trial) -> float:
        config = _suggest_config(trial, self.base_config)

        result: RFCVResult = run_rf_scaffold_cv(
            pipeline=self.pipeline,
            bank=self.bank,
            task=self.task,
            config=config,
        )

        trial.set_user_attr("metric_name", result.metric_name)
        trial.set_user_attr(
            "pooled_oof_score", result.pooled_oof_score
        )
        trial.set_user_attr(
            "mean_train_score",
            float(
                np.mean([
                    fold.train_score
                    for fold in result.fold_metrics
                ])
            ),
        )

        for fold in result.fold_metrics:
            prefix = f"fold_{fold.fold}"

            trial.set_user_attr(
                f"{prefix}_train", fold.train_score
            )
            trial.set_user_attr(
                f"{prefix}_holdout", fold.holdout_score
            )
            trial.set_user_attr(
                f"{prefix}_n_fit", fold.n_fit
            )
            trial.set_user_attr(
                f"{prefix}_n_holdout", fold.n_holdout
            )

            if fold.holdout_prevalence is not None:
                trial.set_user_attr(
                    f"{prefix}_prevalence",
                    fold.holdout_prevalence,
                )

        return result.mean_holdout_score


def optimize_rf(
    pipeline: ModelingPipeline,
    bank: MorganFeatureBank,
    task: Task,
    *,
    base_config: RFConfig,
    n_trials: int = 20,
    study_name: str | None = None,
    storage: str | None = None,
    callbacks: Sequence[
        Callable[[optuna.Study, FrozenTrial], None]
    ] = (),
) -> RFOptimizationResult:
    """Select RF settings using training-only scaffold CV.

    Callbacks allow separate MLflow tracking without coupling this
    module to an experiment-tracking implementation.
    """
    if task not in ("egfr", "her2", "dual"):
        raise ValueError(f"Unsupported task: {task}")

    if n_trials < 1:
        raise ValueError("n_trials must be positive")

    if base_config.n_estimators < 1:
        raise ValueError("n_estimators must be positive")

    if (
        bank.radius != base_config.radius
        or bank.n_bits != base_config.n_bits
        or bank.include_chirality != base_config.include_chirality
    ):
        raise ValueError(
            "Fingerprint configuration does not match the feature bank"
        )

    if not pipeline.folds:
        raise ValueError("No training-only scaffold folds available")

    metric_name = (
        "average_precision" if task == "dual" else "mae"
    )
    direction = (
        "maximize" if task == "dual" else "minimize"
    )

    provenance = {
        "task": task,
        "metric_name": metric_name,
        "dataset_sha256": pipeline.data.dataset_sha256,
        "manifest_sha256": pipeline.data.manifest_sha256,
        "feature_sha256": bank.fingerprint_sha256,
        "fold_assignments_sha256": _fold_manifest_hash(
            pipeline
        ),
        "base_config": asdict(base_config),
        "evaluation": "training_only_scaffold_group_cv",
    }

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=optuna.samplers.TPESampler(
            seed=base_config.seed
        ),
        storage=storage,
        load_if_exists=storage is not None,
    )

    for key, value in provenance.items():
        if (
            key in study.user_attrs
            and study.user_attrs[key] != value
        ):
            raise ValueError(
                f"Resumed Optuna study has incompatible {key}"
            )

        study.set_user_attr(key, value)

    objective = RFOptunaObjective(
        pipeline, bank, task, base_config
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=list(callbacks),
        gc_after_trial=True,
        show_progress_bar=False,
    )

    return RFOptimizationResult(
        study=study,
        task=task,
        best_config=_config_from_params(
            base_config, study.best_trial.params
        ),
        best_score=float(study.best_value),
        metric_name=metric_name,
    )
