
"""Tests for scaffold-aware Random Forest cross-validation."""

from dataclasses import replace

import numpy as np
import pytest
from dual_target.data.modeling_dataset import ModelingData, MoleculeRecord
from dual_target.data.pipeline import compose_modeling_pipeline
from dual_target.features.morgan_pipeline import build_morgan_feature_bank
from dual_target.models.random_forest import RFConfig
from dual_target.training.rf_cv import run_rf_scaffold_cv
from sklearn.metrics import average_precision_score, mean_absolute_error


@pytest.fixture(scope="module")
def setup_cv():
    """Small synthetic dataset with six scaffold groups and all task types."""
    smiles = ("CCO", "CCN", "CCC", "COC", "CCCl", "CCBr")
    records = []

    for group, smiles_value in enumerate(smiles):
        scaffold = f"group_{group}"

        for suffix, egfr, her2 in (
            ("positive", 7.2, 7.0),
            ("negative", 5.5, 6.8),
            ("egfr_only", 7.1, None),
            ("her2_only", None, 6.5),
        ):
            records.append(
                MoleculeRecord(
                    structure_id=f"{group}_{suffix}",
                    smiles=smiles_value,
                    scaffold_group_id=scaffold,
                    egfr_mask=int(egfr is not None),
                    her2_mask=int(her2 is not None),
                    egfr_pic50=egfr,
                    her2_pic50=her2,
                )
            )

    validation = (
        MoleculeRecord(
            structure_id="external_validation",
            smiles="c1ccccc1",
            scaffold_group_id="external_scaffold",
            egfr_mask=1,
            her2_mask=1,
            egfr_pic50=8.0,
            her2_pic50=7.5,
        ),
    )

    data = ModelingData(
        train=tuple(records),
        validation=validation,
        dataset_sha256="synthetic",
        manifest_sha256="synthetic",
    )
    pipeline = compose_modeling_pipeline(data, n_splits=3)
    bank = build_morgan_feature_bank(data.train)
    config = RFConfig(n_estimators=8, n_jobs=1)

    return pipeline, bank, config


@pytest.fixture(scope="module")
def results(setup_cv):
    pipeline, bank, config = setup_cv
    return {
        task: run_rf_scaffold_cv(pipeline, bank, task, config)
        for task in ("egfr", "her2", "dual")
    }


@pytest.mark.parametrize(
    "task,expected",
    [("egfr", 18), ("her2", 18), ("dual", 12)],
)
def test_oof_covers_all_eligible_molecules(results, task, expected):
    result = results[task]
    assert len(result.oof_predictions) == expected
    ids = [row.structure_id for row in result.oof_predictions]
    assert len(ids) == len(set(ids))


def test_oof_matches_frozen_training_order(setup_cv, results):
    pipeline, _, _ = setup_cv

    for task, result in results.items():
        expected = [
            row.structure_id
            for row in pipeline.data.train
            if (
                row.egfr_mask == 1
                if task == "egfr"
                else (
                    row.her2_mask == 1
                    if task == "her2"
                    else row.egfr_mask == 1 and row.her2_mask == 1
                )
            )
        ]
        assert [row.structure_id for row in result.oof_predictions] == expected


def test_no_external_validation_molecules(setup_cv, results):
    pipeline, _, _ = setup_cv
    validation_ids = {row.structure_id for row in pipeline.data.validation}

    for result in results.values():
        ids = {row.structure_id for row in result.oof_predictions}
        assert ids.isdisjoint(validation_ids)


def test_oof_fold_matches_scaffold_assignment(setup_cv, results):
    pipeline, _, _ = setup_cv
    scaffold_to_fold = {
        row.scaffold_group_id: fold.number
        for fold in pipeline.folds
        for row in fold.holdout
    }

    for result in results.values():
        for row in result.oof_predictions:
            assert row.fold == scaffold_to_fold[row.scaffold_group_id]


def test_every_task_has_three_fold_metrics(results):
    for result in results.values():
        assert len(result.fold_metrics) == 3
        assert [fold.fold for fold in result.fold_metrics] == [0, 1, 2]
        assert all(fold.n_fit > 0 and fold.n_holdout > 0 for fold in result.fold_metrics)


@pytest.mark.parametrize("task", ["egfr", "her2"])
def test_regression_oof_metrics(results, task):
    result = results[task]

    observed = np.array([row.observed for row in result.oof_predictions])
    predicted = np.array([row.predicted for row in result.oof_predictions])

    assert result.metric_name == "mae"
    assert np.isfinite(predicted).all()
    assert result.pooled_oof_score == pytest.approx(
        mean_absolute_error(observed, predicted)
    )
    assert result.mean_holdout_score >= 0
    assert all(fold.holdout_score >= 0 for fold in result.fold_metrics)


def test_classification_oof_metrics(results):
    result = results["dual"]

    observed = np.array([row.observed for row in result.oof_predictions])
    predicted = np.array([row.predicted for row in result.oof_predictions])

    assert result.metric_name == "average_precision"
    assert set(observed) == {0.0, 1.0}
    assert np.all((predicted >= 0) & (predicted <= 1))
    assert result.pooled_oof_score == pytest.approx(
        average_precision_score(observed, predicted)
    )

    for fold in result.fold_metrics:
        assert fold.holdout_positives is not None
        assert 0 < fold.holdout_prevalence < 1
        assert 0 <= fold.train_score <= 1
        assert 0 <= fold.holdout_score <= 1


def test_cv_predictions_are_deterministic(setup_cv, results):
    pipeline, bank, config = setup_cv
    repeated = run_rf_scaffold_cv(pipeline, bank, "dual", config)

    assert repeated.oof_predictions == results["dual"].oof_predictions
    assert repeated.fold_metrics == results["dual"].fold_metrics


def test_rejects_unsupported_task(setup_cv):
    pipeline, bank, config = setup_cv

    with pytest.raises(ValueError, match="Unsupported task"):
        run_rf_scaffold_cv(pipeline, bank, "unknown", config)


def test_rejects_fingerprint_configuration_mismatch(setup_cv):
    pipeline, bank, config = setup_cv
    changed = replace(config, n_bits=1024)

    with pytest.raises(ValueError, match="Fingerprint configuration"):
        run_rf_scaffold_cv(pipeline, bank, "egfr", changed)


def test_rejects_incomplete_feature_bank(setup_cv):
    pipeline, _, config = setup_cv
    incomplete = build_morgan_feature_bank(pipeline.data.train[:-1])

    with pytest.raises(ValueError, match="exactly the training molecules"):
        run_rf_scaffold_cv(pipeline, incomplete, "dual", config)


def test_rejects_misaligned_fold_audits(setup_cv):
    pipeline, bank, config = setup_cv
    changed = replace(
        pipeline,
        fold_audits=tuple(reversed(pipeline.fold_audits)),
    )

    with pytest.raises(ValueError, match="misaligned"):
        run_rf_scaffold_cv(changed, bank, "dual", config)


def test_rejects_unready_classification_fold(setup_cv):
    pipeline, bank, config = setup_cv
    first = replace(
        pipeline.fold_audits[0],
        dual_classification_ready=False,
    )
    changed = replace(
        pipeline,
        fold_audits=(first, *pipeline.fold_audits[1:]),
    )

    with pytest.raises(ValueError, match="not ready"):
        run_rf_scaffold_cv(changed, bank, "dual", config)
