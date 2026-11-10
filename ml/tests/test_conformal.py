"""Stage C.5 tests for scaffold-disjoint split-conformal pIC50 intervals.

All records and fingerprints are synthetic. The tests do not load BindingDB,
development-validation CSVs, or the locked test partition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from dual_target.calibration import conformal as cf
from dual_target.calibration.scaffold_calibration import (
    make_scaffold_calibration_split,
)
from dual_target.models.random_forest import RFConfig


@dataclass(frozen=True)
class FakeMolecule:
    structure_id: str
    scaffold_group_id: str
    egfr_pic50: float
    her2_pic50: float
    egfr_mask: bool
    her2_mask: bool


class FakeFeatureBank:
    """Synthetic order-aware bank that implements the production bank contract."""

    radius = 2
    include_chirality = True
    fingerprint_sha256 = "synthetic-morgan-sha256"

    def __init__(self, records, n_bits=32):
        self.molecule_ids = tuple(r.structure_id for r in records)
        self.n_bits = n_bits
        self._features = {}
        for row in records:
            group = int(row.scaffold_group_id.rsplit("-", 1)[-1])
            member = int(row.structure_id.rsplit("-", 1)[-1])
            bits = np.zeros(n_bits, dtype=np.uint8)
            bits[0] = group % 2 == 0
            bits[1] = group % 3 == 0
            bits[2] = group % 5 == 0
            bits[3] = group % 7 == 0
            bits[4 + member] = 1
            bits[8 + group % 20] = 1
            self._features[row.structure_id] = bits

    def select_features(self, records):
        return np.stack([self._features[row.structure_id] for row in records])


def synthetic_records(prefix, n_scaffolds):
    records = []
    for group in range(n_scaffolds):
        for member in range(3):
            records.append(
                FakeMolecule(
                    structure_id=f"{prefix}-m-{group:03d}-{member}",
                    scaffold_group_id=f"{prefix}-scaffold-{group:03d}",
                    egfr_pic50=5.3 + (group % 9) * 0.27 + member * 0.09,
                    her2_pic50=5.6 + (group % 7) * 0.22 + member * 0.11,
                    egfr_mask=member != 2,
                    her2_mask=member != 1,
                )
            )
    return tuple(records)


def data_with(data, **changes):
    return SimpleNamespace(**{**vars(data), **changes})


@pytest.fixture(scope="module")
def context():
    train = synthetic_records("train", 100)
    validation = synthetic_records("validation", 16)
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
    return SimpleNamespace(
        pipeline=pipeline,
        split=split,
        train_bank=FakeFeatureBank(train),
        validation_bank=FakeFeatureBank(validation),
        config=RFConfig(n_bits=32, n_estimators=12, n_jobs=1, seed=714),
    )


def fit_with(context, target="egfr", *, pipeline=None, split=None, bank=None, config=None, sha=None, alpha=0.10):
    return cf.fit_split_conformal_regression(
        context.pipeline if pipeline is None else pipeline,
        context.split if split is None else split,
        context.train_bank if bank is None else bank,
        context.config if config is None else config,
        target,
        alpha=alpha,
        expected_assignment_sha256=(
            context.split.assignment_sha256 if sha is None else sha
        ),
    )


@pytest.fixture(scope="module", params=["egfr", "her2"])
def fitted(request, context):
    return fit_with(context, request.param)


@pytest.fixture(scope="module")
def fits(context):
    return {target: fit_with(context, target) for target in ("egfr", "her2")}


@pytest.mark.parametrize(
    ("residuals", "alpha", "expected_radius", "expected_order"),
    [
        ([0.4] * 9, 0.10, 0.4, 9),
        ([9, 1, 7, 3, 2, 8, 4, 6, 5, 0], 0.10, 9.0, 10),
        ([1, 4, 2, 3, 5], 0.20, 5.0, 5),
        ([1, 4, 2, 3, 5], 0.50, 3.0, 3),
        ([0.0, 0.0, 0.1, 0.1, 0.5, 2.0, 2.0, 3.0, 5.0], 0.20, 3.0, 8),
    ],
)
def test_conformal_quantile_uses_corrected_order_statistic(
    residuals, alpha, expected_radius, expected_order
):
    radius, order = cf.conformal_absolute_quantile(residuals, alpha=alpha)
    assert radius == pytest.approx(expected_radius)
    assert order == expected_order


def test_quantile_does_not_modify_input_array():
    residuals = np.array([4.0, 1.0, 3.0, 2.0, 5.0])
    before = residuals.copy()
    cf.conformal_absolute_quantile(residuals, alpha=0.2)
    assert np.array_equal(residuals, before)


@pytest.mark.parametrize(
    "alpha", [0, 1, -0.1, 1.1, float("nan"), float("inf"), float("-inf")]
)
def test_quantile_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        cf.conformal_absolute_quantile([1.0, 2.0, 3.0], alpha=alpha)


@pytest.mark.parametrize("alpha", [True, "0.1", None])
def test_quantile_rejects_invalid_alpha_type(alpha):
    with pytest.raises(TypeError, match="alpha"):
        cf.conformal_absolute_quantile([1.0, 2.0, 3.0], alpha=alpha)


@pytest.mark.parametrize(
    "residuals",
    [[], [0.2, -0.1], [1.0, float("nan")], [1.0, float("inf")], [[1.0, 2.0]]],
)
def test_quantile_rejects_invalid_residuals(residuals):
    with pytest.raises(ValueError, match="Residuals"):
        cf.conformal_absolute_quantile(residuals, alpha=0.5)


def test_quantile_rejects_too_few_calibration_labels():
    with pytest.raises(ValueError, match="Too few calibration"):
        cf.conformal_absolute_quantile([0.1] * 8, alpha=0.10)


def test_quantile_allows_nonnegative_zero_radius():
    radius, k = cf.conformal_absolute_quantile([0.0] * 10, alpha=0.1)
    assert radius == 0.0
    assert k == 10


@pytest.mark.parametrize("target", ["egfr", "her2"])
def test_target_examples_preserves_label_masks_and_order(context, target):
    ex = cf._target_examples(context.split.calibration, target, context.train_bank)
    expected = tuple(
        row for row in context.split.calibration if getattr(row, f"{target}_mask")
    )
    assert ex.molecule_ids == tuple(r.structure_id for r in expected)
    assert ex.scaffold_ids == tuple(r.scaffold_group_id for r in expected)
    assert np.array_equal(
        ex.pic50, np.array([getattr(r, f"{target}_pic50") for r in expected])
    )
    assert ex.features.shape == (len(expected), context.config.n_bits)


def test_target_examples_includes_pic50_exactly_six(context):
    row = replace(context.split.fit[0], egfr_mask=True, egfr_pic50=6.0)
    ex = cf._target_examples((row,), "egfr", context.train_bank)
    assert ex.pic50.tolist() == [6.0]


def test_target_examples_excludes_unaccepted_labels(context):
    row = replace(context.split.fit[0], egfr_mask=False, egfr_pic50=float("nan"))
    extra = replace(context.split.fit[1], egfr_mask=True)
    ex = cf._target_examples((row, extra), "egfr", context.train_bank)
    assert ex.molecule_ids == (extra.structure_id,)


def test_target_examples_rejects_no_accepted_labels(context):
    rows = tuple(replace(r, egfr_mask=False) for r in context.split.fit[:3])
    with pytest.raises(ValueError, match="No accepted EGFR"):
        cf._target_examples(rows, "egfr", context.train_bank)


def test_target_examples_rejects_nonfinite_accepted_potency(context):
    row = replace(context.split.fit[0], egfr_mask=True, egfr_pic50=float("nan"))
    with pytest.raises(ValueError, match="Nonfinite accepted EGFR"):
        cf._target_examples((row,), "egfr", context.train_bank)


@pytest.mark.parametrize("target", ["egfr", "her2"])
def test_fitted_regression_provenance_and_counts(context, fits, target):
    fitted = fits[target]
    assert fitted.target == target
    assert fitted.assignment_sha256 == context.split.assignment_sha256
    assert fitted.dataset_sha256 == context.pipeline.data.dataset_sha256
    assert fitted.manifest_sha256 == context.pipeline.data.manifest_sha256
    assert fitted.feature_sha256 == context.train_bank.fingerprint_sha256
    assert fitted.seed_used == 719
    assert fitted.config == context.config
    assert fitted.n_fit_labels == len(
        [r for r in context.split.fit if getattr(r, f"{target}_mask")]
    )
    assert fitted.n_calibration_labels == len(
        [r for r in context.split.calibration if getattr(r, f"{target}_mask")]
    )
    assert fitted.alpha == 0.10
    assert fitted.half_width >= 0


def test_fitted_training_id_and_scaffold_guard(context, fits):
    for fitted in fits.values():
        assert fitted.training_molecule_ids == {
            row.structure_id for row in context.pipeline.data.train
        }
        assert fitted.training_scaffold_ids == {
            row.scaffold_group_id for row in context.pipeline.data.train
        }
        assert not fitted.training_molecule_ids & {
            row.structure_id for row in context.pipeline.data.validation
        }


@pytest.mark.parametrize("target", ["egfr", "her2"])
def test_fit_calls_forest_only_with_fit_role_target_labels(context, monkeypatch, target):
    capture = {}
    original = cf.fit_random_forest

    def record_fit(examples, config):
        capture["ids"] = set(examples.molecule_ids)
        capture["potency"] = examples.pic50.copy()
        return original(examples, config)

    monkeypatch.setattr(cf, "fit_random_forest", record_fit)
    fit_with(context, target)
    expected = {
        r.structure_id
        for r in context.split.fit
        if getattr(r, f"{target}_mask")
    }
    assert capture["ids"] == expected
    assert not capture["ids"] & {r.structure_id for r in context.split.calibration}
    assert not capture["ids"] & {r.structure_id for r in context.split.threshold}
    assert np.isfinite(capture["potency"]).all()


@pytest.mark.parametrize("target", ["egfr", "her2"])
def test_calibration_scores_are_from_disjoint_calibration_role(context, fits, target):
    fitted = fits[target]
    scores = fitted.calibration_scores
    expected = [r for r in context.split.calibration if getattr(r, f"{target}_mask")]
    assert scores.molecule_ids == tuple(r.structure_id for r in expected)
    assert np.array_equal(
        scores.observed_pic50,
        [getattr(r, f"{target}_pic50") for r in expected],
    )
    assert not set(scores.molecule_ids) & {r.structure_id for r in context.split.fit}
    assert not set(scores.molecule_ids) & {r.structure_id for r in context.split.threshold}
    assert scores.metrics.n == fitted.n_calibration_labels


@pytest.mark.parametrize("target", ["egfr", "her2"])
def test_calibration_half_width_matches_exact_residual_order_statistic(fits, target):
    fitted = fits[target]
    scores = fitted.calibration_scores
    residuals = np.abs(scores.observed_pic50 - scores.predicted_pic50)
    radius, order = cf.conformal_absolute_quantile(residuals, alpha=fitted.alpha)
    assert fitted.half_width == pytest.approx(radius)
    assert fitted.calibration_order_statistic == order
    assert np.allclose(scores.upper_pic50 - scores.predicted_pic50, radius)
    assert np.allclose(scores.predicted_pic50 - scores.lower_pic50, radius)
    assert scores.covered.dtype == np.dtype("bool")


@pytest.mark.parametrize("target", ["egfr", "her2"])
def test_prediction_intervals_preserve_row_alignment_and_constant_width(context, fits, target):
    fitted = fits[target]
    records = [r for r in context.split.threshold if getattr(r, f"{target}_mask")][:5]
    features = context.train_bank.select_features(records)
    predicted, lower, upper = fitted.predict_intervals(features)
    assert predicted.shape == lower.shape == upper.shape == (len(records),)
    assert np.isfinite(predicted).all()
    assert np.all(lower <= predicted)
    assert np.all(upper >= predicted)
    assert np.allclose(upper - lower, 2.0 * fitted.half_width)
    assert np.allclose(predicted, fitted.forest.predict(features))


@pytest.mark.parametrize("target", ["egfr", "her2"])
def test_development_validation_metrics_recompute_from_predictions(context, fits, target):
    fitted = fits[target]
    result = cf.evaluate_conformal_development_validation(
        fitted, context.pipeline, context.validation_bank
    )
    expected = [r for r in context.pipeline.data.validation if getattr(r, f"{target}_mask")]
    assert result.molecule_ids == tuple(r.structure_id for r in expected)
    assert result.scaffold_ids == tuple(r.scaffold_group_id for r in expected)
    assert result.metrics.n == len(expected)
    assert result.metrics.mae == pytest.approx(
        np.mean(np.abs(result.observed_pic50 - result.predicted_pic50))
    )
    included = (result.observed_pic50 >= result.lower_pic50) & (
        result.observed_pic50 <= result.upper_pic50
    )
    assert np.array_equal(result.covered, included)
    assert result.metrics.empirical_coverage == pytest.approx(np.mean(included))
    assert result.metrics.mean_interval_width == pytest.approx(np.mean(result.upper_pic50 - result.lower_pic50))
    assert result.metrics.median_interval_width == pytest.approx(np.median(result.upper_pic50 - result.lower_pic50))
    assert result.metrics.target_coverage == pytest.approx(0.90)


def test_interval_coverage_includes_exact_boundary_values():
    fake_examples = SimpleNamespace(
        molecule_ids=("a", "b", "c"),
        scaffold_ids=("s1", "s2", "s3"),
        pic50=np.array([1.0, 4.0, 9.0]),
    )
    result = cf._interval_scores(
        fake_examples,
        predicted=np.array([2.0, 3.0, 7.0]),
        lower=np.array([1.0, 2.0, 6.0]),
        upper=np.array([3.0, 4.0, 8.0]),
        alpha=0.1,
    )
    assert result.covered.tolist() == [True, True, False]
    assert result.metrics.empirical_coverage == pytest.approx(2 / 3)
    assert result.metrics.mae == pytest.approx(4 / 3)
    assert result.metrics.mean_interval_width == pytest.approx(2.0)
    assert result.metrics.median_interval_width == pytest.approx(2.0)
    assert result.metrics.target_coverage == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("predicted", "lower", "upper", "message"),
    [
        (np.array([1.0]), np.array([0.0, 0.0]), np.array([2.0, 2.0]), "align"),
        (np.array([1.0, np.nan]), np.array([0.0, 0.0]), np.array([2.0, 2.0]), "align"),
        (np.array([1.0, 1.0]), np.array([1.5, 0.0]), np.array([2.0, 2.0]), "outside"),
        (np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.5, 2.0]), "outside"),
    ],
)
def test_interval_scores_reject_misaligned_or_inconsistent_bounds(predicted, lower, upper, message):
    examples = SimpleNamespace(
        molecule_ids=("a", "b"), scaffold_ids=("s1", "s2"), pic50=np.array([1.0, 1.0])
    )
    with pytest.raises(ValueError, match=message):
        cf._interval_scores(examples, predicted, lower, upper, alpha=0.1)


def test_repeated_fits_are_deterministic(context, fits):
    for target in ("egfr", "her2"):
        second = fit_with(context, target)
        first = fits[target]
        x = context.train_bank.select_features(context.split.calibration[:6])
        assert first.half_width == pytest.approx(second.half_width)
        assert first.calibration_order_statistic == second.calibration_order_statistic
        assert np.array_equal(first.predict_intervals(x)[0], second.predict_intervals(x)[0])


def test_evaluation_does_not_refit(context, fits, monkeypatch):
    def fail_fit(*args, **kwargs):
        raise AssertionError("Evaluation must not retrain the model")

    monkeypatch.setattr(cf, "fit_random_forest", fail_fit)
    for fitted in fits.values():
        scored = cf.evaluate_conformal_development_validation(
            fitted, context.pipeline, context.validation_bank
        )
        assert scored.metrics.n > 0


@pytest.mark.parametrize("invalid_target", ["dual", "EGFR", "parp1", ""])
def test_fit_rejects_unsupported_targets(context, invalid_target):
    with pytest.raises(ValueError, match="target must be"):
        fit_with(context, invalid_target)


@pytest.mark.parametrize("sha", ["", "bad-digest", "f" * 64])
def test_fit_rejects_wrong_assignment_hash(context, sha):
    with pytest.raises(ValueError, match="assignment hash"):
        fit_with(context, sha=sha)


@pytest.mark.parametrize("field", ["dataset_sha256", "manifest_sha256"])
def test_fit_rejects_split_dataset_provenance_tampering(context, field):
    tampered = replace(context.split, **{field: "not-frozen"})
    with pytest.raises(ValueError, match="provenance"):
        fit_with(context, split=tampered)


def test_fit_rejects_manifest_content_changed_without_digest(context):
    old = context.split.assignments[0]
    changed = replace(old, role="threshold" if old.role != "threshold" else "fit")
    tampered = replace(context.split, assignments=(changed,) + context.split.assignments[1:])
    with pytest.raises(ValueError, match="manifest"):
        fit_with(context, split=tampered)


def test_fit_rejects_changed_molecule_record(context):
    old = context.split.fit[0]
    tampered = replace(context.split, fit=(replace(old, her2_pic50=12.0),) + context.split.fit[1:])
    with pytest.raises(ValueError, match="differ from frozen"):
        fit_with(context, split=tampered)


def test_fit_rejects_incomplete_three_way_split(context):
    tampered = replace(context.split, threshold=context.split.threshold[1:])
    with pytest.raises(ValueError, match="cover only frozen"):
        fit_with(context, split=tampered)


def test_fit_rejects_scaffold_overlap_across_roles(context):
    tampered = replace(context.split, calibration=(context.split.fit[0],) + context.split.calibration)
    with pytest.raises(ValueError, match="overlap"):
        fit_with(context, split=tampered)


def test_fit_rejects_train_feature_bank_wrong_row_order(context):
    bank = FakeFeatureBank(context.pipeline.data.train[::-1])
    with pytest.raises(ValueError, match="row order"):
        fit_with(context, bank=bank)


@pytest.mark.parametrize("field", ["n_bits", "radius", "include_chirality"])
def test_fit_rejects_mismatched_fingerprint_config(context, field):
    value = {"n_bits": 64, "radius": 3, "include_chirality": False}[field]
    with pytest.raises(ValueError, match="fingerprint-bank"):
        fit_with(context, config=replace(context.config, **{field: value}))


def test_fit_rejects_development_validation_scaffold_overlap(context):
    bad = replace(
        context.pipeline.data.validation[0],
        scaffold_group_id=context.pipeline.data.train[0].scaffold_group_id,
    )
    pipeline = SimpleNamespace(
        data=data_with(context.pipeline.data, validation=(bad,) + context.pipeline.data.validation[1:])
    )
    with pytest.raises(ValueError, match="leaked"):
        fit_with(context, pipeline=pipeline)


@pytest.mark.parametrize("field", ["dataset_sha256", "manifest_sha256"])
def test_evaluation_rejects_data_provenance_mismatch(context, fits, field):
    pipeline = SimpleNamespace(data=data_with(context.pipeline.data, **{field: "changed"}))
    with pytest.raises(ValueError, match="provenance"):
        cf.evaluate_conformal_development_validation(
            fits["egfr"], pipeline, context.validation_bank
        )


def test_evaluation_rejects_wrong_validation_row_order(context, fits):
    bank = FakeFeatureBank(context.pipeline.data.validation[::-1])
    with pytest.raises(ValueError, match="row mismatch"):
        cf.evaluate_conformal_development_validation(fits["egfr"], context.pipeline, bank)


@pytest.mark.parametrize("field", ["n_bits", "radius", "include_chirality"])
def test_evaluation_rejects_wrong_validation_feature_config(context, fits, field):
    bank = FakeFeatureBank(context.pipeline.data.validation)
    bank.__dict__[field] = {"n_bits": 64, "radius": 3, "include_chirality": False}[field]
    with pytest.raises(ValueError, match="fingerprint configuration"):
        cf.evaluate_conformal_development_validation(fits["egfr"], context.pipeline, bank)


@pytest.mark.parametrize("leak_by", ["molecule", "scaffold"])
def test_evaluation_rejects_validation_leakage(context, fits, leak_by):
    first = context.pipeline.data.validation[0]
    training = context.pipeline.data.train[0]
    change = (
        {"structure_id": training.structure_id}
        if leak_by == "molecule"
        else {"scaffold_group_id": training.scaffold_group_id}
    )
    bad = replace(first, **change)
    changed_records = (bad,) + context.pipeline.data.validation[1:]
    pipeline = SimpleNamespace(
        data=data_with(context.pipeline.data, validation=changed_records)
    )
    bank = FakeFeatureBank(changed_records) if leak_by == "molecule" else context.validation_bank
    with pytest.raises(ValueError, match="leakage"):
        cf.evaluate_conformal_development_validation(fits["egfr"], pipeline, bank)


def test_evaluation_rejects_no_accepted_target_labels(context, fits):
    masked = tuple(replace(r, egfr_mask=False) for r in context.pipeline.data.validation)
    pipeline = SimpleNamespace(data=data_with(context.pipeline.data, validation=masked))
    with pytest.raises(ValueError, match="No accepted EGFR"):
        cf.evaluate_conformal_development_validation(fits["egfr"], pipeline, context.validation_bank)


@pytest.mark.parametrize(
    "features",
    [np.ones(32), np.ones((2, 16)), np.zeros((0, 32)), np.full((2, 32), np.nan), np.full((2, 32), np.inf)],
)
def test_predict_intervals_rejects_invalid_fingerprints(fits, features):
    with pytest.raises(ValueError, match="fingerprint|nonfinite"):
        fits["egfr"].predict_intervals(features)


def test_fit_rejects_alpha_insufficient_for_calibration_count(context):
    # 10% intervals have enough examples; an extremely low alpha does not.
    with pytest.raises(ValueError, match="Too few calibration"):
        fit_with(context, alpha=1e-9)
