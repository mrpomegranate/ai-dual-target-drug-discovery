"""Stage C.4 sigmoid-calibration tests on synthetic scaffold-disjoint data.

No BindingDB/ChEMBL files, original validation labels, or locked test labels are
loaded. These tests verify fitted behavior and enforce partition/provenance
checks before production MLflow orchestration is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from dual_target.calibration import probability as pc
from dual_target.calibration.scaffold_calibration import (
    make_scaffold_calibration_split,
)
from dual_target.models.random_forest import RFConfig
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


@dataclass(frozen=True)
class FakeMolecule:
    structure_id: str
    scaffold_group_id: str
    egfr_pic50: float
    her2_pic50: float
    egfr_mask: bool
    her2_mask: bool


class FakeFeatureBank:
    """Row-indexed synthetic fingerprints, with the production bank interface."""

    radius = 2
    include_chirality = True
    fingerprint_sha256 = "synthetic-feature-sha-256"

    def __init__(self, records, n_bits=32):
        self.molecule_ids = tuple(row.structure_id for row in records)
        self.n_bits = n_bits
        self._features = {}
        for row in records:
            index = int(row.scaffold_group_id.rsplit("-", 1)[-1])
            member = int(row.structure_id.rsplit("-", 1)[-1])
            vec = np.zeros(n_bits, dtype=np.uint8)
            vec[0] = int(index % 2 == 0)
            vec[1] = int(index % 3 == 0)
            vec[2] = int(index % 5 == 0)
            vec[3] = int(index % 7 == 0)
            vec[4 + member] = 1
            vec[8 + index % 20] = 1
            self._features[row.structure_id] = vec

    def select_features(self, records):
        return np.stack([self._features[r.structure_id] for r in records])


def records_for(prefix, n_groups):
    rows = []
    for group in range(n_groups):
        for member in range(3):
            rows.append(
                FakeMolecule(
                    structure_id=f"{prefix}-m-{group:03d}-{member}",
                    scaffold_group_id=f"{prefix}-scaffold-{group:03d}",
                    egfr_pic50=6.8 if group % 2 == 0 else 5.7,
                    her2_pic50=6.6 if group % 2 == 0 else 5.8,
                    egfr_mask=(member != 2),
                    her2_mask=(member != 1),
                )
            )
    return tuple(rows)


@pytest.fixture(scope="module")
def context():
    train = records_for("train", 100)
    validation = records_for("validation", 16)
    pipeline = SimpleNamespace(
        data=SimpleNamespace(
            train=train,
            validation=validation,
            dataset_sha256="synthetic-dataset-hash",
            manifest_sha256="synthetic-manifest-hash",
        )
    )
    split = make_scaffold_calibration_split(
        pipeline,
        seed=719,
        max_attempts=1,
        minimum_target_labels=4,
        minimum_dual_per_class=2,
    )
    bank = FakeFeatureBank(train)
    validation_bank = FakeFeatureBank(validation)
    config = RFConfig(n_bits=32, n_estimators=12, n_jobs=1, seed=714)
    return SimpleNamespace(
        pipeline=pipeline,
        split=split,
        bank=bank,
        validation_bank=validation_bank,
        config=config,
    )


@pytest.fixture(scope="module")
def fitted(context):
    return pc.fit_sigmoid_dual_calibration(
        context.pipeline,
        context.split,
        context.bank,
        context.config,
        expected_assignment_sha256=context.split.assignment_sha256,
    )


@pytest.fixture(scope="module")
def development_scores(context, fitted):
    return pc.evaluate_development_validation(
        fitted, context.pipeline, context.validation_bank
    )


def fit_with(context, *, pipeline=None, split=None, bank=None, config=None, sha=None):
    return pc.fit_sigmoid_dual_calibration(
        context.pipeline if pipeline is None else pipeline,
        context.split if split is None else split,
        context.bank if bank is None else bank,
        context.config if config is None else config,
        expected_assignment_sha256=(
            context.split.assignment_sha256 if sha is None else sha
        ),
    )


def labels_for(records):
    paired = [r for r in records if r.egfr_mask and r.her2_mask]
    return np.array(
        [int(r.egfr_pic50 > 6.0 and r.her2_pic50 > 6.0) for r in paired]
    )


def test_fitted_classifier_and_sigmoid_have_binary_classes(fitted):
    assert list(fitted.forest.classes_) == [0, 1]
    assert list(fitted.sigmoid.classes_) == [0, 1]


def test_fitted_metadata_matches_frozen_assignment(context, fitted):
    assert fitted.assignment_sha256 == context.split.assignment_sha256
    assert fitted.dataset_sha256 == context.pipeline.data.dataset_sha256
    assert fitted.manifest_sha256 == context.pipeline.data.manifest_sha256
    assert fitted.feature_sha256 == context.bank.fingerprint_sha256
    assert fitted.seed_used == 719
    assert fitted.config == context.config


def test_fitted_label_counts_match_paired_audits(context, fitted):
    assert fitted.n_fit_labels == context.split.fit_audit.paired_labels
    assert fitted.n_calibration_labels == context.split.calibration_audit.paired_labels


def test_fitted_training_ids_cover_training_only(context, fitted):
    assert fitted.training_molecule_ids == {
        row.structure_id for row in context.pipeline.data.train
    }
    assert fitted.training_scaffold_ids == {
        row.scaffold_group_id for row in context.pipeline.data.train
    }
    assert not fitted.training_molecule_ids & {
        row.structure_id for row in context.pipeline.data.validation
    }


def test_fit_uses_only_paired_fit_labels_not_calibration_or_threshold(
    context, monkeypatch
):
    captured = {}
    original = pc.fit_dual_activity_classifier

    def record_examples(examples, config):
        captured["ids"] = set(examples.molecule_ids)
        captured["labels"] = examples.dual_active.copy()
        return original(examples, config)

    monkeypatch.setattr(pc, "fit_dual_activity_classifier", record_examples)
    fit_with(context)
    expected = {
        row.structure_id
        for row in context.split.fit
        if row.egfr_mask and row.her2_mask
    }
    assert captured["ids"] == expected
    assert not captured["ids"] & {
        row.structure_id for row in context.split.calibration
    }
    assert not captured["ids"] & {
        row.structure_id for row in context.split.threshold
    }
    assert len(captured["labels"]) == len(expected)


def test_strict_potency_threshold_excludes_exactly_six(context):
    first = context.split.fit[0]
    exact = replace(first, egfr_pic50=6.0, her2_pic50=7.0, egfr_mask=True, her2_mask=True)
    assert pc._make_paired_examples((exact,), context.bank).dual_active.tolist() == [0]
    positive = replace(exact, egfr_pic50=6.00001)
    assert pc._make_paired_examples((positive,), context.bank).dual_active.tolist() == [1]


def test_missing_target_is_excluded_not_classified_negative(context):
    positive = next(r for r in context.split.fit if r.egfr_pic50 > 6)
    masked = replace(positive, her2_mask=False)
    assert pc._make_paired_examples((positive, masked), context.bank).dual_active.tolist() == [1]


def test_calibration_scores_align_to_eligible_rows(context, fitted):
    expected = tuple(
        r.structure_id for r in context.split.calibration if r.egfr_mask and r.her2_mask
    )
    assert fitted.calibration_scores.molecule_ids == expected
    assert np.array_equal(fitted.calibration_scores.observed, labels_for(context.split.calibration))


def test_calibration_metrics_are_labeled_in_sample(fitted):
    scores = fitted.calibration_scores
    assert scores.raw_metrics.n == fitted.n_calibration_labels
    assert scores.calibrated_metrics.n == fitted.n_calibration_labels
    assert scores.raw_metrics.positives == int(scores.observed.sum())


def test_calibration_scores_have_valid_probabilities(fitted):
    scores = fitted.calibration_scores
    for probabilities in (scores.raw_probabilities, scores.calibrated_probabilities):
        assert probabilities.shape == (fitted.n_calibration_labels,)
        assert np.isfinite(probabilities).all()
        assert ((0 <= probabilities) & (probabilities <= 1)).all()


def test_calibrated_probabilities_match_sigmoid_mapping(context, fitted):
    rows = context.split.calibration[:7]
    matrix = context.bank.select_features(rows)
    raw = fitted.predict_raw_positive(matrix)
    expected = fitted.sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1]
    assert np.allclose(fitted.predict_calibrated_positive(matrix), expected)


def test_predict_proba_returns_complementary_class_probabilities(context, fitted):
    matrix = context.bank.select_features(context.split.fit[:5])
    probs = fitted.predict_proba(matrix)
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[:, 1], fitted.predict_calibrated_positive(matrix))
    assert np.allclose(probs[:, 0], 1 - probs[:, 1])


def test_development_scores_align_to_validation_rows(context, development_scores):
    expected = tuple(
        r.structure_id
        for r in context.pipeline.data.validation
        if r.egfr_mask and r.her2_mask
    )
    assert development_scores.molecule_ids == expected
    assert np.array_equal(development_scores.observed, labels_for(context.pipeline.data.validation))


def test_development_scores_match_independent_sklearn_metrics(development_scores):
    for p, m in (
        (development_scores.raw_probabilities, development_scores.raw_metrics),
        (development_scores.calibrated_probabilities, development_scores.calibrated_metrics),
    ):
        y = development_scores.observed
        assert m.n == len(y)
        assert m.positives == int(y.sum())
        assert m.positive_prevalence == pytest.approx(np.mean(y))
        assert m.average_precision == pytest.approx(average_precision_score(y, p))
        assert m.brier_score == pytest.approx(brier_score_loss(y, p))
        assert m.log_loss == pytest.approx(log_loss(y, p, labels=[0, 1]))


def test_development_evaluation_never_refits(context, fitted, monkeypatch):
    def fail_if_refitting(*args, **kwargs):
        raise AssertionError("No estimator fitting during evaluation")

    monkeypatch.setattr(pc, "fit_dual_activity_classifier", fail_if_refitting)
    monkeypatch.setattr(pc.LogisticRegression, "fit", fail_if_refitting)
    result = pc.evaluate_development_validation(
        fitted, context.pipeline, context.validation_bank
    )
    assert result.raw_metrics.n > 0


def test_repeated_fitting_is_deterministic(context, fitted):
    duplicate = fit_with(context)
    x = context.bank.select_features(context.split.calibration[:6])
    assert np.array_equal(fitted.predict_raw_positive(x), duplicate.predict_raw_positive(x))
    assert np.allclose(
        fitted.predict_calibrated_positive(x), duplicate.predict_calibrated_positive(x)
    )


@pytest.mark.parametrize("bad_sha", ["", "not-a-hash", "f" * 64])
def test_rejects_wrong_or_malformed_assignment_hash(context, bad_sha):
    with pytest.raises(ValueError, match="assignment hash"):
        fit_with(context, sha=bad_sha)


@pytest.mark.parametrize("field", ["dataset_sha256", "manifest_sha256"])
def test_rejects_mismatched_split_provenance(context, field):
    modified = replace(context.split, **{field: "changed-provenance"})
    with pytest.raises(ValueError, match="provenance"):
        fit_with(context, split=modified)


def test_rejects_tampered_assignment_manifest_even_with_same_stored_digest(context):
    one = context.split.assignments[0]
    different = replace(one, role="threshold" if one.role != "threshold" else "fit")
    tampered = replace(context.split, assignments=(different,) + context.split.assignments[1:])
    with pytest.raises(ValueError, match="manifest"):
        fit_with(context, split=tampered)


def test_rejects_tampered_digest_even_if_caller_expects_it(context):
    forged = replace(context.split, assignment_sha256="a" * 64)
    with pytest.raises(ValueError, match="manifest contents"):
        fit_with(context, split=forged, sha="a" * 64)


def test_rejects_duplicate_assignment_rows(context):
    a = list(context.split.assignments)
    a[1] = a[0]
    tampered = replace(context.split, assignments=tuple(a))
    with pytest.raises(ValueError, match="duplicated"):
        fit_with(context, split=tampered)


def test_rejects_modified_record_in_a_partition(context):
    row = context.split.fit[0]
    bad = replace(context.split, fit=(replace(row, egfr_pic50=9.9),) + context.split.fit[1:])
    with pytest.raises(ValueError, match="differ from frozen"):
        fit_with(context, split=bad)


def test_rejects_incomplete_three_way_split(context):
    bad = replace(context.split, threshold=context.split.threshold[1:])
    with pytest.raises(ValueError, match="cover only frozen"):
        fit_with(context, split=bad)


def test_rejects_scaffold_moved_across_roles(context):
    first = context.split.fit[0]
    bad = replace(
        context.split,
        calibration=(first,) + context.split.calibration,
    )
    with pytest.raises(ValueError, match="overlap"):
        fit_with(context, split=bad)


def test_rejects_training_feature_bank_wrong_order(context):
    bank = FakeFeatureBank(context.pipeline.data.train[::-1])
    with pytest.raises(ValueError, match="row order"):
        fit_with(context, bank=bank)


@pytest.mark.parametrize("change", ["n_bits", "radius", "include_chirality"])
def test_rejects_training_fingerprint_config_mismatch(context, change):
    value = {
        "n_bits": 64,
        "radius": 3,
        "include_chirality": False,
    }[change]
    bad = replace(context.config, **{change: value})
    with pytest.raises(ValueError, match="fingerprint-bank"):
        fit_with(context, config=bad)


def test_rejects_validation_scaffold_leakage_during_fit(context):
    val = context.pipeline.data.validation
    modified = replace(val[0], scaffold_group_id=context.split.fit[0].scaffold_group_id)
    bad_pipeline = SimpleNamespace(
        data=replace_data(context.pipeline.data, validation=(modified,) + val[1:])
    )
    with pytest.raises(ValueError, match="leaked"):
        fit_with(context, pipeline=bad_pipeline)


def replace_data(data, **changes):
    return SimpleNamespace(**{**vars(data), **changes})


def test_rejects_validation_dataset_hash_mismatch(context, fitted):
    bad = SimpleNamespace(
        data=replace_data(context.pipeline.data, dataset_sha256="other")
    )
    with pytest.raises(ValueError, match="provenance"):
        pc.evaluate_development_validation(fitted, bad, context.validation_bank)


def test_rejects_validation_manifest_hash_mismatch(context, fitted):
    bad = SimpleNamespace(
        data=replace_data(context.pipeline.data, manifest_sha256="other")
    )
    with pytest.raises(ValueError, match="provenance"):
        pc.evaluate_development_validation(fitted, bad, context.validation_bank)


def test_rejects_validation_feature_bank_wrong_row_order(context, fitted):
    bad_bank = FakeFeatureBank(context.pipeline.data.validation[::-1])
    with pytest.raises(ValueError, match="row mismatch"):
        pc.evaluate_development_validation(fitted, context.pipeline, bad_bank)


@pytest.mark.parametrize("field", ["n_bits", "radius", "include_chirality"])
def test_rejects_validation_feature_configuration_mismatch(context, fitted, field):
    bad_bank = FakeFeatureBank(context.pipeline.data.validation)
    bad_bank.__dict__[field] = {
        "n_bits": 64,
        "radius": 3,
        "include_chirality": False,
    }[field]
    with pytest.raises(ValueError, match="fingerprint configuration"):
        pc.evaluate_development_validation(fitted, context.pipeline, bad_bank)


def test_rejects_validation_scaffold_leakage_during_evaluation(context, fitted):
    val = context.pipeline.data.validation
    bad_row = replace(val[0], scaffold_group_id=context.pipeline.data.train[0].scaffold_group_id)
    bad_pipeline = SimpleNamespace(data=replace_data(context.pipeline.data, validation=(bad_row,) + val[1:]))
    with pytest.raises(ValueError, match="overlap"):
        pc.evaluate_development_validation(fitted, bad_pipeline, context.validation_bank)


def test_rejects_validation_without_paired_measurements(context, fitted):
    masked = tuple(replace(r, her2_mask=False) for r in context.pipeline.data.validation)
    bad_pipeline = SimpleNamespace(data=replace_data(context.pipeline.data, validation=masked))
    with pytest.raises(ValueError, match="No accepted paired labels"):
        pc.evaluate_development_validation(fitted, bad_pipeline, context.validation_bank)


@pytest.mark.parametrize("matrix", [
    np.ones(32),
    np.ones((2, 16)),
    np.zeros((0, 32)),
    np.full((2, 32), np.nan),
])
def test_rejects_invalid_fingerprint_shapes_and_values(fitted, matrix):
    with pytest.raises(ValueError, match="fingerprint|nonfinite"):
        fitted.predict_raw_positive(matrix)


@pytest.mark.parametrize("invalid", [
    np.array([np.nan, 0.5]),
    np.array([-0.1, 0.5]),
    np.array([0.5, 1.1]),
    np.array([[0.5, 0.4]]),
    np.array([0.5]),
])
def test_rejects_invalid_probability_inputs(invalid):
    with pytest.raises(ValueError, match="Invalid positive-class probabilities"):
        pc._validated_probabilities(invalid, 2)


@pytest.mark.parametrize("observed", [np.array([0, 0]), np.array([1, 1])])
def test_probability_metrics_reject_single_class(observed):
    with pytest.raises(ValueError, match="both binary classes"):
        pc._metrics(observed, np.array([0.2, 0.8]))


def test_probability_metrics_computes_non_degenerate_binary_case():
    out = pc._metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.6, 0.3]))
    assert out.n == 4
    assert out.positives == 2
    assert out.positive_prevalence == 0.5
    assert out.average_precision == pytest.approx(1.0)
    assert out.brier_score == pytest.approx(0.0675)
    assert out.log_loss > 0


def test_mismatched_probability_lengths_rejected(context):
    ex = pc._make_paired_examples(context.split.fit, context.bank)
    with pytest.raises(ValueError, match="Invalid positive-class probabilities"):
        pc._scores_from_probabilities(
            ex,
            raw=np.array([0.5]),
            calibrated=np.array([0.5]),
        )


def test_rejects_paired_nonfinite_accepted_potency(context):
    row = context.split.fit[0]
    bad_row = replace(row, egfr_mask=True, her2_mask=True, egfr_pic50=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        pc._make_paired_examples((bad_row,), context.bank)


def test_rejects_no_paired_examples(context):
    rows = tuple(replace(r, her2_mask=False) for r in context.split.fit[:3])
    with pytest.raises(ValueError, match="No accepted paired labels"):
        pc._make_paired_examples(rows, context.bank)


def test_rejects_invalid_forest_classes(context, monkeypatch):
    original = pc.fit_dual_activity_classifier

    def bad_class_order(examples, config):
        model = original(examples, config)
        model.classes_ = np.array([1, 0])
        return model

    monkeypatch.setattr(pc, "fit_dual_activity_classifier", bad_class_order)
    with pytest.raises(ValueError, match="forest classes"):
        fit_with(context)


def test_rejects_invalid_sigmoid_classes(context, monkeypatch):
    class BadSigmoid:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, labels):
            self.classes_ = np.array([1, 0])
            return self

    monkeypatch.setattr(pc, "LogisticRegression", BadSigmoid)
    with pytest.raises(ValueError, match="sigmoid classes"):
        fit_with(context)
