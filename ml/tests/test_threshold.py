"""Stage C.6 tests for leakage-safe, deterministic dual-activity thresholds.

Uses synthetic scaffold-disjoint molecules and fingerprints only. No frozen
BindingDB rows, M2.1 final-test labels, or MLflow artifacts are accessed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from dual_target.calibration import threshold as th
from dual_target.calibration.probability import fit_sigmoid_dual_calibration
from dual_target.calibration.scaffold_calibration import make_scaffold_calibration_split
from dual_target.models.random_forest import RFConfig
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class FakeMolecule:
    structure_id: str
    scaffold_group_id: str
    egfr_pic50: float
    her2_pic50: float
    egfr_mask: bool
    her2_mask: bool


class FakeFeatureBank:
    """Deterministic synthetic fingerprints with the real feature-bank contract."""

    radius = 2
    include_chirality = True
    fingerprint_sha256 = "synthetic-feature-sha-256"

    def __init__(self, records, n_bits=32):
        self.molecule_ids = tuple(row.structure_id for row in records)
        self.n_bits = n_bits
        self._features = {}
        for row in records:
            group = int(row.scaffold_group_id.rsplit("-", 1)[-1])
            member = int(row.structure_id.rsplit("-", 1)[-1])
            vec = np.zeros(n_bits, dtype=np.uint8)
            vec[0] = group % 2 == 0
            vec[1] = group % 3 == 0
            vec[2] = group % 5 == 0
            vec[3] = group % 7 == 0
            vec[4 + member] = 1
            vec[8 + group % 20] = 1
            self._features[row.structure_id] = vec

    def select_features(self, records):
        return np.stack([self._features[row.structure_id] for row in records])


def records_for(prefix: str, n_groups: int):
    records = []
    for group in range(n_groups):
        for member in range(3):
            records.append(
                FakeMolecule(
                    structure_id=f"{prefix}-m-{group:03d}-{member}",
                    scaffold_group_id=f"{prefix}-scaffold-{group:03d}",
                    egfr_pic50=6.8 if group % 2 == 0 else 5.7,
                    her2_pic50=6.6 if group % 2 == 0 else 5.8,
                    egfr_mask=member != 2,
                    her2_mask=member != 1,
                )
            )
    return tuple(records)


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
def model(context):
    return fit_sigmoid_dual_calibration(
        context.pipeline,
        context.split,
        context.bank,
        context.config,
        expected_assignment_sha256=context.split.assignment_sha256,
    )


@pytest.fixture(scope="module")
def selection(context, model):
    return th.select_dual_threshold(
        model,
        context.pipeline,
        context.split,
        context.bank,
        expected_assignment_sha256=context.split.assignment_sha256,
        score_kind="raw",
    )


@pytest.fixture(scope="module")
def validation_scores(context, model, selection):
    return th.evaluate_frozen_dual_threshold(
        selection, model, context.pipeline, context.validation_bank
    )


def select_with(context, model, *, pipeline=None, split=None, bank=None, sha=None, score_kind="raw"):
    return th.select_dual_threshold(
        model,
        context.pipeline if pipeline is None else pipeline,
        context.split if split is None else split,
        context.bank if bank is None else bank,
        expected_assignment_sha256=(
            context.split.assignment_sha256 if sha is None else sha
        ),
        score_kind=score_kind,
    )


def evaluate_with(context, model, selection, *, pipeline=None, bank=None):
    return th.evaluate_frozen_dual_threshold(
        selection,
        model,
        context.pipeline if pipeline is None else pipeline,
        context.validation_bank if bank is None else bank,
    )


def paired_rows(records):
    return tuple(row for row in records if row.egfr_mask and row.her2_mask)


def paired_labels(records):
    return np.array(
        [int(r.egfr_pic50 > 6 and r.her2_pic50 > 6) for r in paired_rows(records)],
        dtype=np.int8,
    )


# --- Exact fixed-cutoff mathematics and boundary behavior ---


def test_fixed_threshold_hand_calculated_confusion_matrix():
    metrics = th.evaluate_fixed_threshold(
        np.array([1, 0, 1, 0, 0]),
        np.array([0.9, 0.8, 0.7, 0.4, 0.2]),
        0.75,
    )
    assert (metrics.tp, metrics.fp, metrics.tn, metrics.fn) == (1, 1, 2, 1)
    assert (metrics.n, metrics.positives, metrics.negatives) == (5, 2, 3)
    assert metrics.predicted_positives == 2
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.specificity == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.balanced_accuracy == pytest.approx((0.5 + 2 / 3) / 2)
    assert metrics.accuracy == pytest.approx(3 / 5)


def test_equal_score_meets_inclusive_threshold():
    result = th.evaluate_fixed_threshold([1, 0], [0.5, 0.5], 0.5)
    assert (result.tp, result.fp, result.tn, result.fn) == (1, 1, 0, 0)


@pytest.mark.parametrize("cutoff", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_valid_cutoffs_match_manual_predictions(cutoff):
    observed = np.array([1, 0, 1, 0], dtype=np.int8)
    scores = np.array([1.0, 0.9, 0.1, 0.0])
    result = th.evaluate_fixed_threshold(observed, scores, cutoff)
    predicted = scores >= cutoff
    assert result.predicted_positives == int(predicted.sum())
    assert result.tp == int(np.sum(predicted & (observed == 1)))
    assert result.fp == int(np.sum(predicted & (observed == 0)))
    assert result.f1 == pytest.approx(f1_score(observed, predicted))


def test_no_predicted_positives_is_well_defined():
    result = th.evaluate_fixed_threshold([1, 0], [0.1, 0.2], 0.9)
    assert result.predicted_positives == 0
    assert result.precision == 0
    assert result.recall == 0
    assert result.f1 == 0
    assert (result.tp, result.fp, result.tn, result.fn) == (0, 0, 1, 1)


def test_all_predicted_positive_is_well_defined():
    result = th.evaluate_fixed_threshold([1, 0, 0], [0.5, 0.2, 0.1], 0.0)
    assert (result.tp, result.fp, result.tn, result.fn) == (1, 2, 0, 0)
    assert result.recall == 1
    assert result.specificity == 0
    assert result.f1 == pytest.approx(0.5)


def test_boolean_and_integer_labels_agree():
    scores = [0.9, 0.8, 0.1, 0.2]
    result1 = th.evaluate_fixed_threshold([True, False, True, False], scores, 0.5)
    result2 = th.evaluate_fixed_threshold([1, 0, 1, 0], scores, 0.5)
    assert result1 == result2


def test_input_arrays_are_never_mutated():
    observed = np.array([1, 0, 1, 0])
    scores = np.array([0.5, 0.8, 0.4, 0.2])
    before_y, before_scores = observed.copy(), scores.copy()
    th.choose_f1_threshold(observed, scores)
    assert np.array_equal(observed, before_y)
    assert np.array_equal(scores, before_scores)


def test_multiple_equal_scores_are_single_threshold_candidate():
    curve = th.sweep_f1_thresholds([1, 0, 1, 0], [0.4, 0.4, 0.8, 0.8])
    assert [row.threshold for row in curve] == [0.8, 0.4]


def test_curve_uses_all_unique_scores_descending():
    scores = [0.1, 0.6, 0.3, 0.6, 0.8]
    curve = th.sweep_f1_thresholds([1, 0, 1, 0, 0], scores)
    assert [row.threshold for row in curve] == [0.8, 0.6, 0.3, 0.1]
    assert len(curve) == len(set(scores))


def test_each_curve_row_is_evaluated_using_its_own_cutoff():
    y = [1, 0, 1, 0, 0]
    scores = [0.91, 0.88, 0.63, 0.42, 0.22]
    curve = th.sweep_f1_thresholds(y, scores)
    assert all(th.evaluate_fixed_threshold(y, scores, row.threshold) == row for row in curve)


def test_f1_maximum_is_selected_from_complete_curve():
    chosen, curve = th.choose_f1_threshold(
        [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]
    )
    assert chosen == max(curve, key=lambda row: (row.f1, row.threshold))
    assert chosen.threshold == pytest.approx(0.7)


def test_equal_f1_ties_select_higher_cutoff():
    chosen, curve = th.choose_f1_threshold([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.6])
    assert curve[0].f1 == pytest.approx(curve[-1].f1)
    assert chosen.threshold == pytest.approx(0.9)


def test_order_of_input_rows_does_not_change_selected_threshold():
    y = np.array([1, 0, 0, 1, 1, 0])
    p = np.array([0.92, 0.8, 0.7, 0.6, 0.55, 0.2])
    shuffled = np.array([3, 0, 5, 1, 4, 2])
    a, curve_a = th.choose_f1_threshold(y, p)
    b, curve_b = th.choose_f1_threshold(y[shuffled], p[shuffled])
    assert a == b
    assert curve_a == curve_b


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), -float("inf"), None, "0.5", True, False])
def test_fixed_threshold_rejects_invalid_cutoff(threshold):
    with pytest.raises(ValueError, match="threshold"):
        th.evaluate_fixed_threshold([0, 1], [0.1, 0.9], threshold)


@pytest.mark.parametrize(
    "labels",
    [[], [0], [0, 0], [1, 1], [0, 1, 2], [0.0, float("nan")], [0, float("inf")], ["0", "1"], [[0, 1]]],
)
def test_threshold_rejects_missing_or_invalid_binary_labels(labels):
    with pytest.raises(ValueError, match="binary classes"):
        th.evaluate_fixed_threshold(labels, [0.3] * len(labels), 0.5)


@pytest.mark.parametrize(
    "scores",
    [[0.1], [0.1, 0.2, 0.3], [-0.1, 0.9], [0.1, 1.1], [float("nan"), 0.9], [float("inf"), 0.9], ["0.1", "0.9"], [[0.1, 0.9]]],
)
def test_threshold_rejects_invalid_probability_scores(scores):
    with pytest.raises(ValueError, match="scores"):
        th.evaluate_fixed_threshold([0, 1], scores, 0.5)


def test_identical_scores_do_not_crash_f1_search():
    chosen, curve = th.choose_f1_threshold([1, 0, 1, 0], [0.4] * 4)
    assert len(curve) == 1
    assert chosen.threshold == pytest.approx(0.4)
    assert chosen.f1 == pytest.approx(2 / 3)


# --- End-to-end selection on synthetic scaffold-separated Stage C data ---


def test_selected_cutoff_is_f1_maximizer_on_threshold_role(selection):
    assert selection.objective == "f1"
    assert selection.score_kind == "raw"
    assert selection.selected_metrics == max(
        selection.curve, key=lambda row: (row.f1, row.threshold)
    )
    assert selection.threshold == selection.selected_metrics.threshold


def test_selected_provenance_matches_frozen_split(context, model, selection):
    assert selection.assignment_sha256 == context.split.assignment_sha256
    assert selection.dataset_sha256 == context.pipeline.data.dataset_sha256
    assert selection.manifest_sha256 == context.pipeline.data.manifest_sha256
    assert selection.train_feature_sha256 == context.bank.fingerprint_sha256
    assert selection.seed_used == model.seed_used == 719
    assert selection.n_fit_labels == model.n_fit_labels == context.split.fit_audit.paired_labels
    assert selection.n_calibration_labels == model.n_calibration_labels == context.split.calibration_audit.paired_labels


def test_threshold_examples_are_only_paired_reserved_molecules(context, selection):
    expected = paired_rows(context.split.threshold)
    scores = selection.threshold_predictions
    assert scores.subset == "threshold"
    assert scores.molecule_ids == tuple(row.structure_id for row in expected)
    assert scores.scaffold_ids == tuple(row.scaffold_group_id for row in expected)
    assert len(scores.molecule_ids) == context.split.threshold_audit.paired_labels
    assert np.array_equal(scores.observed, paired_labels(context.split.threshold))


def test_threshold_molecules_do_not_overlap_fit_or_calibration(context, selection):
    selected = set(selection.threshold_predictions.molecule_ids)
    assert not selected & {r.structure_id for r in context.split.fit}
    assert not selected & {r.structure_id for r in context.split.calibration}
    assert not selected & {r.structure_id for r in context.pipeline.data.validation}


def test_threshold_scaffolds_do_not_overlap_fit_calibration_validation(context, selection):
    selected = set(selection.threshold_predictions.scaffold_ids)
    for subset in (context.split.fit, context.split.calibration, context.pipeline.data.validation):
        assert not selected & {r.scaffold_group_id for r in subset}


def test_reserved_labels_use_strict_potency_definition(context, selection):
    expected = [
        int(row.egfr_pic50 > 6.0 and row.her2_pic50 > 6.0)
        for row in paired_rows(context.split.threshold)
    ]
    assert selection.threshold_predictions.observed.tolist() == expected


def test_reserved_missing_target_labels_are_excluded(context, selection):
    total = len(context.split.threshold)
    n_eligible = len(paired_rows(context.split.threshold))
    assert n_eligible < total
    assert len(selection.threshold_predictions.observed) == n_eligible


def test_reserved_decisions_match_score_greater_than_or_equal_to_cutoff(selection):
    pred = selection.threshold_predictions
    assert np.array_equal(pred.predicted, (pred.scores >= selection.threshold).astype(np.int8))
    assert pred.metrics == selection.selected_metrics


def test_reserved_metrics_have_correct_confusion_counts(selection):
    p = selection.threshold_predictions
    assert p.metrics.tp + p.metrics.fp + p.metrics.tn + p.metrics.fn == len(p.observed)
    assert p.metrics.positives == int(p.observed.sum())
    assert p.metrics.negatives == len(p.observed) - int(p.observed.sum())
    assert p.metrics.f1 == pytest.approx(f1_score(p.observed, p.predicted))


def test_selected_scores_are_exactly_raw_forest_probabilities(context, model, selection):
    eligible = paired_rows(context.split.threshold)
    reference = model.predict_raw_positive(context.bank.select_features(eligible))
    assert np.array_equal(selection.threshold_predictions.scores, reference)


def test_selection_does_not_fit_or_update_forest_and_sigmoid(context, model, monkeypatch):
    def never_fit(*args, **kwargs):
        pytest.fail("threshold selection must not refit the model")

    monkeypatch.setattr(model.forest, "fit", never_fit)
    monkeypatch.setattr(model.sigmoid, "fit", never_fit)
    select_with(context, model)


def test_threshold_selection_is_deterministic(context, model, selection):
    second = select_with(context, model)
    assert second.threshold == selection.threshold
    assert second.curve == selection.curve
    assert second.threshold_predictions.molecule_ids == selection.threshold_predictions.molecule_ids
    assert np.array_equal(second.threshold_predictions.predicted, selection.threshold_predictions.predicted)


def test_calibrated_score_kind_can_be_selected_explicitly(context, model):
    result = select_with(context, model, score_kind="calibrated")
    eligible = paired_rows(context.split.threshold)
    expected = model.predict_calibrated_positive(context.bank.select_features(eligible))
    assert result.score_kind == "calibrated"
    assert np.array_equal(result.threshold_predictions.scores, expected)
    assert result.threshold == result.selected_metrics.threshold


@pytest.mark.parametrize("kind", ["unscaled", "probability", "RAW", "", None])
def test_threshold_selection_rejects_unknown_score_source(context, model, kind):
    with pytest.raises(ValueError, match="score_kind"):
        select_with(context, model, score_kind=kind)


def test_threshold_selection_rejects_wrong_assignment_hash(context, model):
    with pytest.raises(ValueError, match="assignment hash"):
        select_with(context, model, sha="a" * 64)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("assignment_sha256", "x" * 64),
        ("dataset_sha256", "wrong-dataset"),
        ("manifest_sha256", "wrong-manifest"),
        ("feature_sha256", "wrong-features"),
        ("seed_used", 999),
        ("n_fit_labels", -1),
        ("n_calibration_labels", -1),
        ("training_molecule_ids", frozenset()),
        ("training_scaffold_ids", frozenset()),
    ],
)
def test_threshold_selection_rejects_mismatched_model_provenance(context, model, field, bad):
    with pytest.raises(ValueError, match="provenance"):
        select_with(context, replace(model, **{field: bad}))


def test_selection_rejects_mismatched_training_bank_row_order(context, model):
    bank = FakeFeatureBank(tuple(reversed(context.pipeline.data.train)))
    with pytest.raises(ValueError, match="row order"):
        select_with(context, model, bank=bank)


def test_selection_rejects_overlap_with_development_validation(context, model):
    validation = (context.pipeline.data.train[0],) + context.pipeline.data.validation
    changed_pipeline = SimpleNamespace(
        data=SimpleNamespace(**{**vars(context.pipeline.data), "validation": validation})
    )
    with pytest.raises(ValueError, match="Development-validation"):
        select_with(context, model, pipeline=changed_pipeline)


def test_selection_rejects_tampered_split_manifest(context, model):
    changed = replace(context.split, assignments=context.split.assignments[:-1])
    with pytest.raises(ValueError, match="manifest"):
        select_with(context, model, split=changed)


def test_selection_does_not_use_calibration_labels_for_cutoff_selection(context, model, monkeypatch):
    captured = []
    original = th.choose_f1_threshold

    def record(y, scores):
        captured.append(np.asarray(y).copy())
        return original(y, scores)

    monkeypatch.setattr(th, "choose_f1_threshold", record)
    select_with(context, model)
    assert len(captured) == 1
    assert np.array_equal(captured[0], paired_labels(context.split.threshold))
    assert {r.structure_id for r in context.split.threshold}.isdisjoint(
        {r.structure_id for r in context.split.calibration}
    )


# --- Frozen-threshold development-validation evaluation ---


def test_validation_prediction_rows_are_aligned(context, validation_scores):
    paired = paired_rows(context.pipeline.data.validation)
    assert validation_scores.subset == "validation"
    assert validation_scores.molecule_ids == tuple(r.structure_id for r in paired)
    assert validation_scores.scaffold_ids == tuple(r.scaffold_group_id for r in paired)
    assert np.array_equal(validation_scores.observed, paired_labels(context.pipeline.data.validation))


def test_validation_decisions_use_exact_frozen_threshold(selection, validation_scores):
    assert validation_scores.metrics.threshold == selection.threshold
    assert np.array_equal(validation_scores.predicted, (validation_scores.scores >= selection.threshold).astype(np.int8))


def test_validation_f1_matches_sklearn(validation_scores):
    assert validation_scores.metrics.f1 == pytest.approx(f1_score(validation_scores.observed, validation_scores.predicted))


def test_validation_confusion_counts_reconcile(validation_scores):
    m = validation_scores.metrics
    assert m.tp + m.fp + m.tn + m.fn == m.n
    assert m.tp + m.fn == m.positives
    assert m.tn + m.fp == m.negatives
    assert m.tp + m.fp == m.predicted_positives


def test_validation_scores_match_existing_stage_c4_evaluator(context, model, validation_scores):
    from dual_target.calibration.probability import evaluate_development_validation

    reference = evaluate_development_validation(model, context.pipeline, context.validation_bank)
    assert np.array_equal(validation_scores.scores, reference.raw_probabilities)
    assert np.array_equal(validation_scores.observed, reference.observed)


def test_validation_scaffolds_disjoint_from_all_training_scaffolds(context, validation_scores):
    assert not set(validation_scores.scaffold_ids) & {r.scaffold_group_id for r in context.pipeline.data.train}


def test_validation_does_not_rechoose_threshold(context, model, selection, monkeypatch):
    def never_choose(*args, **kwargs):
        pytest.fail("development validation cannot select cutoff")

    monkeypatch.setattr(th, "choose_f1_threshold", never_choose)
    evaluate_with(context, model, selection)


def test_validation_does_not_refit_forest_or_calibrator(context, model, selection, monkeypatch):
    def never_fit(*args, **kwargs):
        pytest.fail("development validation cannot refit")

    monkeypatch.setattr(model.forest, "fit", never_fit)
    monkeypatch.setattr(model.sigmoid, "fit", never_fit)
    evaluate_with(context, model, selection)


def test_explicit_calibrated_selection_evaluates_calibrated_validation_scores(context, model):
    calibrated = select_with(context, model, score_kind="calibrated")
    actual = evaluate_with(context, model, calibrated)
    from dual_target.calibration.probability import evaluate_development_validation

    expected = evaluate_development_validation(model, context.pipeline, context.validation_bank)
    assert np.array_equal(actual.scores, expected.calibrated_probabilities)
    assert actual.metrics.threshold == calibrated.threshold


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("assignment_sha256", "x" * 64),
        ("dataset_sha256", "wrong"),
        ("manifest_sha256", "wrong"),
        ("train_feature_sha256", "wrong"),
        ("seed_used", 1),
        ("n_fit_labels", 1),
        ("n_calibration_labels", 1),
        ("objective", "accuracy"),
        ("score_kind", "alien"),
        ("threshold", -0.1),
    ],
)
def test_validation_rejects_mismatched_threshold_provenance(context, model, selection, field, bad):
    with pytest.raises(ValueError, match="provenance"):
        evaluate_with(context, model, replace(selection, **{field: bad}))


def test_validation_rejects_inconsistent_selected_metric_threshold(context, model, selection):
    fake_metrics = replace(selection.selected_metrics, threshold=0.1234)
    with pytest.raises(ValueError, match="provenance"):
        evaluate_with(context, model, replace(selection, selected_metrics=fake_metrics))


def test_validation_rejects_reordered_feature_bank(context, model, selection):
    bank = FakeFeatureBank(tuple(reversed(context.pipeline.data.validation)))
    with pytest.raises(ValueError, match="row mismatch"):
        evaluate_with(context, model, selection, bank=bank)


def test_validation_rejects_dataset_mismatch(context, model, selection):
    changed_pipeline = SimpleNamespace(
        data=SimpleNamespace(**{**vars(context.pipeline.data), "dataset_sha256": "wrong"})
    )
    with pytest.raises(ValueError, match="dataset"):
        evaluate_with(context, model, selection, pipeline=changed_pipeline)


def test_validation_does_not_mutate_selection_or_model(context, model, selection):
    before = selection.threshold
    before_count = len(selection.curve)
    before_config = model.config
    first = evaluate_with(context, model, selection)
    second = evaluate_with(context, model, selection)
    assert selection.threshold == before
    assert len(selection.curve) == before_count
    assert model.config == before_config
    assert np.array_equal(first.predicted, second.predicted)
    assert first.metrics == second.metrics
