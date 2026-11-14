"""Synthetic-only Stage B RF tests; never read frozen validation/test files."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from dual_target.models.random_forest import RFConfig
from dual_target.training import rf_pipeline as stage_b
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score


@dataclass(frozen=True)
class FakeRecord:
    structure_id: str
    scaffold_group_id: str
    egfr_pic50: float
    her2_pic50: float
    egfr_mask: bool
    her2_mask: bool


class FakeBank:
    """In-memory fingerprint bank with strict molecule-to-row lookup."""

    def __init__(self, records: tuple[FakeRecord, ...], *, n_bits: int = 64):
        self.molecule_ids = tuple(record.structure_id for record in records)
        self.radius = 2
        self.n_bits = n_bits
        self.include_chirality = True
        self.fingerprint_sha256 = "synthetic-feature-hash-" + self.molecule_ids[0]
        rng = np.random.default_rng(714)
        self._by_id = {
            record.structure_id: rng.integers(0, 2, size=n_bits, dtype=np.uint8)
            for record in records
        }

    def select_features(self, records: tuple[FakeRecord, ...]) -> np.ndarray:
        return np.stack([self._by_id[record.structure_id] for record in records])


def make_records(prefix: str, size: int) -> tuple[FakeRecord, ...]:
    return tuple(
        FakeRecord(
            structure_id=f"{prefix}-molecule-{index}",
            scaffold_group_id=f"{prefix}-scaffold-{index}",
            egfr_pic50=6.8 if index % 3 == 0 else 5.4 + 0.3 * (index % 3),
            her2_pic50=6.7 if index % 2 == 0 else 5.5,
            egfr_mask=index not in (0, 5),
            her2_mask=index not in (1, 7),
        )
        for index in range(size)
    )


@pytest.fixture
def experiment(monkeypatch: pytest.MonkeyPatch):
    train = make_records("train", 18)
    validation = make_records("validation", 12)
    pipeline = SimpleNamespace(
        data=SimpleNamespace(
            train=train,
            validation=validation,
            dataset_sha256="synthetic-dataset-sha256",
            manifest_sha256="synthetic-manifest-sha256",
        )
    )
    bank = FakeBank(train)
    validation_bank = FakeBank(validation)

    def filter_records(records, task):
        if task == "dual":
            return tuple(r for r in records if r.egfr_mask and r.her2_mask)
        return tuple(r for r in records if getattr(r, f"{task}_mask"))

    def create_validation_bank(records, *, radius, n_bits, include_chirality):
        assert tuple(records) == validation
        assert (radius, n_bits, include_chirality) == (2, 64, True)
        return validation_bank

    monkeypatch.setattr(stage_b, "select_task_records", filter_records)
    monkeypatch.setattr(stage_b, "build_morgan_feature_bank", create_validation_bank)
    configs = {
        "egfr": RFConfig(n_estimators=12, n_bits=64, n_jobs=1, seed=714),
        "her2": RFConfig(n_estimators=12, n_bits=64, n_jobs=1, seed=714),
        "dual": RFConfig(n_estimators=12, n_bits=64, n_jobs=1, seed=714),
    }
    return pipeline, bank, validation_bank, configs


@pytest.mark.parametrize("task", ["egfr", "her2", "dual"])
def test_task_examples_preserve_eligible_row_order(experiment, task):
    pipeline, bank, _, _ = experiment
    examples = stage_b._make_examples(pipeline.data.train, task, bank)
    eligible = stage_b.select_task_records(pipeline.data.train, task)
    assert examples.molecule_ids == tuple(r.structure_id for r in eligible)
    assert examples.scaffold_ids == tuple(r.scaffold_group_id for r in eligible)
    np.testing.assert_array_equal(examples.features, bank.select_features(eligible))


@pytest.mark.parametrize("task", ["egfr", "her2"])
def test_regression_examples_retain_exact_pic50(experiment, task):
    pipeline, bank, _, _ = experiment
    examples = stage_b._make_examples(pipeline.data.train, task, bank)
    expected = [getattr(r, f"{task}_pic50") for r in stage_b.select_task_records(pipeline.data.train, task)]
    np.testing.assert_allclose(examples.pic50, expected)


def test_dual_activity_is_strictly_above_pic50_threshold(experiment):
    pipeline, bank, _, _ = experiment
    train = list(pipeline.data.train)
    train[2] = replace(train[2], egfr_pic50=6.0, her2_pic50=8.0)
    train[3] = replace(train[3], egfr_pic50=8.0, her2_pic50=6.0)
    train[4] = replace(train[4], egfr_pic50=6.01, her2_pic50=6.01)
    examples = stage_b._make_examples(tuple(train), "dual", bank)
    observed = dict(zip(examples.molecule_ids, examples.dual_active, strict=True))
    assert observed[train[2].structure_id] == 0
    assert observed[train[3].structure_id] == 0
    assert observed[train[4].structure_id] == 1


@pytest.mark.parametrize("task", ["egfr", "her2", "dual"])
def test_empty_eligible_task_rejected(experiment, monkeypatch, task):
    pipeline, bank, _, _ = experiment
    monkeypatch.setattr(stage_b, "select_task_records", lambda records, chosen: ())
    with pytest.raises(ValueError, match="No accepted labels"):
        stage_b._make_examples(pipeline.data.train, task, bank)


@pytest.mark.parametrize("task", ["egfr", "her2"])
def test_nonfinite_regression_label_rejected(experiment, task):
    pipeline, bank, _, _ = experiment
    rows = list(pipeline.data.train)
    rows[2] = replace(rows[2], **{f"{task}_pic50": float("nan")})
    with pytest.raises(ValueError, match="must be finite"):
        stage_b._make_examples(tuple(rows), task, bank)


def test_nonfinite_paired_label_rejected(experiment):
    pipeline, bank, _, _ = experiment
    rows = list(pipeline.data.train)
    rows[2] = replace(rows[2], her2_pic50=float("inf"))
    with pytest.raises(ValueError, match="must be finite"):
        stage_b._make_examples(tuple(rows), "dual", bank)


def test_full_fit_evaluates_all_three_tasks(experiment):
    pipeline, bank, validation_bank, configs = experiment
    result = stage_b.fit_and_evaluate_selected_rf(pipeline, bank, configs)
    assert isinstance(result, stage_b.RFStageBResult)
    assert isinstance(result.egfr.model, RandomForestRegressor)
    assert isinstance(result.her2.model, RandomForestRegressor)
    assert isinstance(result.dual.model, RandomForestClassifier)
    for task in ("egfr", "her2", "dual"):
        fit = getattr(result, task)
        assert fit.task == task
        assert fit.config == configs[task]
        assert len(fit.train_predictions) == len(fit.train_examples.molecule_ids)
        assert len(fit.validation_predictions) == len(fit.validation_examples.molecule_ids)
        assert set(fit.validation_examples.molecule_ids).issubset(set(validation_bank.molecule_ids))
        assert set(fit.train_examples.molecule_ids).issubset(set(bank.molecule_ids))
        assert fit.train_metrics and fit.validation_metrics


def test_regression_validation_predictions_match_fitted_models(experiment):
    pipeline, bank, _, configs = experiment
    result = stage_b.fit_and_evaluate_selected_rf(pipeline, bank, configs)
    for fit in (result.egfr, result.her2):
        np.testing.assert_allclose(
            fit.validation_predictions,
            fit.model.predict(fit.validation_examples.features),
        )
        assert fit.validation_metrics["mae"] >= 0


def test_paired_validation_predictions_are_molecule_aligned(experiment):
    pipeline, bank, validation_bank, configs = experiment
    result = stage_b.fit_and_evaluate_selected_rf(pipeline, bank, configs)
    paired = result.paired_validation
    eligible = stage_b.select_task_records(pipeline.data.validation, "dual")
    assert paired.molecule_ids == tuple(r.structure_id for r in eligible)
    assert paired.scaffold_ids == tuple(r.scaffold_group_id for r in eligible)
    features = validation_bank.select_features(eligible)
    np.testing.assert_allclose(paired.predicted_egfr_pic50, result.egfr.model.predict(features))
    np.testing.assert_allclose(paired.predicted_her2_pic50, result.her2.model.predict(features))
    np.testing.assert_allclose(
        paired.regression_ranking_score,
        np.minimum(paired.predicted_egfr_pic50, paired.predicted_her2_pic50),
    )
    classes = list(result.dual.model.classes_)
    np.testing.assert_allclose(
        paired.classifier_score,
        result.dual.model.predict_proba(features)[:, classes.index(1)],
    )
    assert np.all((paired.classifier_score >= 0) & (paired.classifier_score <= 1))
    assert paired.observed_dual_active.dtype.kind in "biu"
    assert paired.regression_average_precision == pytest.approx(
        average_precision_score(paired.observed_dual_active, paired.regression_ranking_score)
    )
    assert paired.classifier_average_precision == pytest.approx(
        average_precision_score(paired.observed_dual_active, paired.classifier_score)
    )


def test_provenance_hashes_propagated(experiment):
    pipeline, bank, validation_bank, configs = experiment
    result = stage_b.fit_and_evaluate_selected_rf(pipeline, bank, configs)
    assert result.dataset_sha256 == "synthetic-dataset-sha256"
    assert result.manifest_sha256 == "synthetic-manifest-sha256"
    assert result.train_feature_sha256 == bank.fingerprint_sha256
    assert result.validation_feature_sha256 == validation_bank.fingerprint_sha256


@pytest.mark.parametrize(
    "keys",
    [(), ("egfr", "her2"), ("egfr", "her2", "dual", "extra")],
)
def test_wrong_task_configuration_set_rejected(experiment, keys):
    pipeline, bank, _, configs = experiment
    provided = {key: configs.get(key, configs["egfr"]) for key in keys}
    with pytest.raises(ValueError, match="exactly egfr, her2, and dual"):
        stage_b.fit_and_evaluate_selected_rf(pipeline, bank, provided)


def test_training_feature_bank_row_misalignment_rejected(experiment):
    pipeline, bank, _, configs = experiment
    bank.molecule_ids = tuple(reversed(bank.molecule_ids))
    with pytest.raises(ValueError, match="frozen train rows"):
        stage_b.fit_and_evaluate_selected_rf(pipeline, bank, configs)


def test_training_validation_molecule_leakage_rejected(experiment):
    pipeline, bank, _, configs = experiment
    copied = replace(pipeline.data.validation[2], structure_id=pipeline.data.train[2].structure_id)
    pipeline.data.validation = (copied, *pipeline.data.validation[1:])
    with pytest.raises(ValueError, match="molecule overlap"):
        stage_b.fit_and_evaluate_selected_rf(pipeline, bank, configs)


def test_training_validation_scaffold_leakage_rejected(experiment):
    pipeline, bank, _, configs = experiment
    copied = replace(
        pipeline.data.validation[2],
        scaffold_group_id=pipeline.data.train[2].scaffold_group_id,
    )
    pipeline.data.validation = (copied, *pipeline.data.validation[1:])
    with pytest.raises(ValueError, match="scaffold overlap"):
        stage_b.fit_and_evaluate_selected_rf(pipeline, bank, configs)


@pytest.mark.parametrize("task", ["egfr", "her2", "dual"])
@pytest.mark.parametrize(
    "overrides",
    [
        {"radius": 3},
        {"n_bits": 128},
        {"include_chirality": False},
    ],
)
def test_fingerprint_config_mismatch_rejected(experiment, task, overrides):
    pipeline, bank, _, configs = experiment
    configs[task] = replace(configs[task], **overrides)
    with pytest.raises(ValueError, match=f"{task} fingerprint configuration mismatch"):
        stage_b.fit_and_evaluate_selected_rf(pipeline, bank, configs)
