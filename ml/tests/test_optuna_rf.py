
"""Tests for training-only, scaffold-aware Random Forest Optuna tuning."""

from dataclasses import replace

import optuna
import pytest
from dual_target.data.modeling_dataset import ModelingData, MoleculeRecord
from dual_target.data.pipeline import compose_modeling_pipeline
from dual_target.features.morgan_pipeline import build_morgan_feature_bank
from dual_target.models.random_forest import RFConfig
from dual_target.training import optuna_rf
from dual_target.training.rf_cv import FoldMetrics, RFCVResult


@pytest.fixture
def setup_tuning(monkeypatch):
    """Synthetic scaffold groups and a cheap stand-in for RF cross-validation."""
    train = tuple(
        MoleculeRecord(
            structure_id=f"m{group}_{kind}",
            smiles="CCO" if kind == 0 else "CCN",
            scaffold_group_id=f"scaffold_{group}",
            egfr_mask=1,
            her2_mask=1,
            egfr_pic50=7.2 if kind == 0 else 5.5,
            her2_pic50=7.0,
        )
        for group in range(6)
        for kind in range(2)
    )
    validation = (
        MoleculeRecord("validation", "CCC", "external", 1, 1, 7.0, 7.0),
    )
    data = ModelingData(
        train=train,
        validation=validation,
        dataset_sha256="dataset-version-a",
        manifest_sha256="split-version-a",
    )
    pipeline = compose_modeling_pipeline(data, n_splits=3)
    bank = build_morgan_feature_bank(data.train)
    config = RFConfig(n_estimators=5, n_jobs=1)
    calls = []

    def fake_cv(*, pipeline, bank, task, config):
        calls.append((pipeline, bank, task, config))
        score = 1.0 / (config.min_samples_leaf + 1.0)
        metric_name = "average_precision" if task == "dual" else "mae"
        metrics = tuple(
            FoldMetrics(
                fold=i,
                n_fit=8,
                n_holdout=4,
                train_score=0.9 if task == "dual" else 0.1,
                holdout_score=score + (i - 1) * 0.01,
                holdout_prevalence=0.5 if task == "dual" else None,
            )
            for i in range(3)
        )
        return RFCVResult(
            task=task,
            metric_name=metric_name,
            fold_metrics=metrics,
            oof_predictions=(),
            mean_holdout_score=score,
            pooled_oof_score=score + 0.02,
        )

    monkeypatch.setattr(optuna_rf, "run_rf_scaffold_cv", fake_cv)
    return pipeline, bank, config, calls


def _optimize(setup_tuning, task="dual", **kwargs):
    pipeline, bank, config, _ = setup_tuning
    return optuna_rf.optimize_rf(
        pipeline, bank, task, base_config=config, n_trials=2, **kwargs
    )


@pytest.mark.parametrize("task", ["egfr", "her2"])
def test_regression_study_minimizes_mae(setup_tuning, task):
    result = _optimize(setup_tuning, task)
    assert result.task == task
    assert result.metric_name == "mae"
    assert result.study.direction == optuna.study.StudyDirection.MINIMIZE
    assert result.best_score == pytest.approx(result.study.best_value)


def test_classifier_study_maximizes_ap(setup_tuning):
    result = _optimize(setup_tuning)
    assert result.metric_name == "average_precision"
    assert result.study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert 0 < result.best_score < 1


def test_selected_configuration_matches_best_trial(setup_tuning):
    result = _optimize(setup_tuning)
    config = result.best_config
    params = result.study.best_trial.params

    assert config.max_depth == params["max_depth"]
    assert config.min_samples_leaf == params["min_samples_leaf"]
    assert config.min_samples_split == params["min_samples_split"]
    assert config.max_features == params["max_features"]
    assert config.n_estimators == 5
    assert config.seed == 714
    assert config.n_bits == 2048


def test_uses_existing_pipeline_and_feature_bank(setup_tuning):
    pipeline, bank, _, calls = setup_tuning
    _optimize(setup_tuning, "egfr")

    assert len(calls) == 2
    assert all(p is pipeline and b is bank and t == "egfr" for p, b, t, _ in calls)


def test_callback_receives_each_completed_trial(setup_tuning):
    observed = []

    def callback(study, trial):
        observed.append((study.study_name, trial.number, trial.value))

    result = _optimize(
        setup_tuning, callbacks=[callback], study_name="callback_check"
    )

    assert len(observed) == 2
    assert [number for _, number, _ in observed] == [0, 1]
    assert len(result.study.trials) == 2


def test_trial_records_fold_metrics_and_oof_summary(setup_tuning):
    result = _optimize(setup_tuning)
    attrs = result.study.trials[0].user_attrs

    assert attrs["metric_name"] == "average_precision"
    assert attrs["pooled_oof_score"] > 0
    assert attrs["mean_train_score"] == pytest.approx(0.9)

    for i in range(3):
        assert attrs[f"fold_{i}_n_fit"] == 8
        assert attrs[f"fold_{i}_n_holdout"] == 4
        assert attrs[f"fold_{i}_prevalence"] == pytest.approx(0.5)
        assert f"fold_{i}_holdout" in attrs


def test_study_stores_dataset_feature_and_fold_hashes(setup_tuning):
    pipeline, bank, _, _ = setup_tuning
    attrs = _optimize(setup_tuning).study.user_attrs

    assert attrs["dataset_sha256"] == pipeline.data.dataset_sha256
    assert attrs["manifest_sha256"] == pipeline.data.manifest_sha256
    assert attrs["feature_sha256"] == bank.fingerprint_sha256
    assert len(attrs["fold_assignments_sha256"]) == 64
    assert attrs["evaluation"] == "training_only_scaffold_group_cv"


def test_same_seed_produces_same_search_sequence(setup_tuning):
    first = _optimize(setup_tuning)
    second = _optimize(setup_tuning)

    assert [t.params for t in first.study.trials] == [
        t.params for t in second.study.trials
    ]
    assert first.best_config == second.best_config


def test_rejects_unknown_task(setup_tuning):
    with pytest.raises(ValueError, match="Unsupported task"):
        _optimize(setup_tuning, "unknown")


@pytest.mark.parametrize("n_trials", [0, -1])
def test_rejects_nonpositive_trials(setup_tuning, n_trials):
    pipeline, bank, config, _ = setup_tuning

    with pytest.raises(ValueError, match="n_trials must be positive"):
        optuna_rf.optimize_rf(
            pipeline, bank, "dual", base_config=config, n_trials=n_trials
        )


def test_rejects_invalid_number_of_trees(setup_tuning):
    pipeline, bank, config, _ = setup_tuning

    with pytest.raises(ValueError, match="n_estimators must be positive"):
        optuna_rf.optimize_rf(
            pipeline,
            bank,
            "dual",
            base_config=replace(config, n_estimators=0),
        )


@pytest.mark.parametrize(
    "change",
    [
        {"radius": 3},
        {"n_bits": 1024},
        {"include_chirality": False},
    ],
)
def test_rejects_mismatched_fingerprint_parameters(setup_tuning, change):
    pipeline, bank, config, _ = setup_tuning

    with pytest.raises(ValueError, match="Fingerprint configuration"):
        optuna_rf.optimize_rf(
            pipeline, bank, "dual", base_config=replace(config, **change)
        )


def test_rejects_missing_scaffold_folds(setup_tuning):
    pipeline, bank, config, _ = setup_tuning

    with pytest.raises(ValueError, match="No training-only scaffold folds"):
        optuna_rf.optimize_rf(
            replace(pipeline, folds=()),
            bank,
            "dual",
            base_config=config,
        )


def test_compatible_study_can_resume(setup_tuning):
    storage = optuna.storages.InMemoryStorage()

    result1 = _optimize(
        setup_tuning,
        study_name="resume_check",
        storage=storage,
    )
    result2 = _optimize(
        setup_tuning,
        study_name="resume_check",
        storage=storage,
    )

    assert len(result1.study.trials) == 4
    assert len(result2.study.trials) == 4


@pytest.mark.parametrize(
    "field", ["dataset_sha256", "manifest_sha256"]
)
def test_resume_rejects_changed_frozen_data(setup_tuning, field):
    pipeline, bank, config, _ = setup_tuning
    storage = optuna.storages.InMemoryStorage()

    _optimize(
        setup_tuning,
        study_name="provenance_check",
        storage=storage,
    )

    altered_data = replace(
        pipeline.data, **{field: "different-version"}
    )

    with pytest.raises(ValueError, match=f"incompatible {field}"):
        optuna_rf.optimize_rf(
            replace(pipeline, data=altered_data),
            bank,
            "dual",
            base_config=config,
            n_trials=1,
            study_name="provenance_check",
            storage=storage,
        )


def test_resume_rejects_changed_features(setup_tuning):
    pipeline, bank, config, _ = setup_tuning
    storage = optuna.storages.InMemoryStorage()

    _optimize(
        setup_tuning,
        study_name="feature_check",
        storage=storage,
    )

    with pytest.raises(ValueError, match="incompatible feature_sha256"):
        optuna_rf.optimize_rf(
            pipeline,
            replace(bank, fingerprint_sha256="different-features"),
            "dual",
            base_config=config,
            n_trials=1,
            study_name="feature_check",
            storage=storage,
        )


def test_resume_rejects_changed_base_config(setup_tuning):
    pipeline, bank, config, _ = setup_tuning
    storage = optuna.storages.InMemoryStorage()

    _optimize(
        setup_tuning,
        study_name="configuration_check",
        storage=storage,
    )

    with pytest.raises(ValueError, match="incompatible base_config"):
        optuna_rf.optimize_rf(
            pipeline,
            bank,
            "dual",
            base_config=replace(config, n_estimators=20),
            n_trials=1,
            study_name="configuration_check",
            storage=storage,
        )
