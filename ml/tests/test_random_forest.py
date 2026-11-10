
"""Tests for target-specific RF regression and dual-activity classification."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.models.random_forest import (
    RFConfig,
    evaluate_dual_activity_classifier,
    evaluate_random_forest,
    fit_dual_activity_classifier,
    fit_random_forest,
    make_dual_activity_examples,
    make_target_examples,
)


@pytest.fixture
def config():
    """Small forest for fast, deterministic unit tests."""
    return RFConfig(n_estimators=12, n_jobs=1)


@pytest.fixture
def records():
    """Synthetic potency records; these are not BindingDB measurements."""
    values = [
        # ID, SMILES, EGFR pIC50, HER2 pIC50
        ("a", "CCO", 7.4, 6.8),
        ("b", "CCN", 7.1, 5.9),
        ("c", "CCC", 6.0, 7.3),
        ("d", "COC", 6.1, 7.1),
        ("e", "CCCl", 6.7, None),
        ("f", "CCBr", None, 7.0),
        ("g", "c1ccccc1", 5.2, 5.8),
        ("h", "c1ccncc1", 8.1, 8.0),
        ("i", "CC(=O)O", 6.2, 6.0),
    ]

    return tuple(
        MoleculeRecord(
            structure_id=sid,
            smiles=smiles,
            scaffold_group_id=f"scaffold_{sid}",
            egfr_mask=int(egfr is not None),
            her2_mask=int(her2 is not None),
            egfr_pic50=egfr,
            her2_pic50=her2,
        )
        for sid, smiles, egfr, her2 in values
    )


def test_regression_respects_target_masks(records, config):
    egfr = make_target_examples(records, "egfr", config)
    her2 = make_target_examples(records, "her2", config)

    assert len(egfr.molecule_ids) == 8
    assert len(her2.molecule_ids) == 8

    assert "f" not in egfr.molecule_ids
    assert "e" not in her2.molecule_ids

    assert egfr.features.shape == (8, 2048)
    assert her2.features.shape == (8, 2048)
    assert egfr.features.dtype == np.uint8
    assert her2.features.dtype == np.uint8


def test_regression_labels_are_aligned(records, config):
    examples = make_target_examples(records, "egfr", config)

    assert examples.molecule_ids[0] == "a"
    assert examples.pic50[0] == pytest.approx(7.4)
    assert examples.molecule_ids[1] == "b"
    assert examples.pic50[1] == pytest.approx(7.1)


def test_dual_activity_labels(records, config):
    examples = make_dual_activity_examples(records, config)

    assert examples.features.shape == (7, 2048)
    assert examples.molecule_ids == ("a", "b", "c", "d", "g", "h", "i")

    np.testing.assert_array_equal(
        examples.dual_active,
        [1, 0, 0, 1, 0, 1, 0],
    )


def test_activity_threshold_is_strict(records, config):
    examples = make_dual_activity_examples(records, config)
    labels = dict(zip(examples.molecule_ids, examples.dual_active))

    # Exactly pIC50 = 6.0 is not active.
    assert labels["c"] == 0
    assert labels["i"] == 0

    # Both targets must exceed the threshold.
    assert labels["a"] == 1
    assert labels["b"] == 0


def test_unpaired_molecules_are_excluded(records, config):
    examples = make_dual_activity_examples(records, config)

    assert "e" not in examples.molecule_ids
    assert "f" not in examples.molecule_ids


def test_invalid_regression_target(records, config):
    with pytest.raises(ValueError, match="Unsupported target"):
        make_target_examples(records, "invalid", config)


def test_missing_accepted_regression_label(records, config):
    invalid = replace(records[0], egfr_pic50=None)

    with pytest.raises(ValueError, match="Missing or nonfinite"):
        make_target_examples([invalid], "egfr", config)


def test_duplicate_molecule_rejected(records, config):
    with pytest.raises(ValueError, match="Duplicate molecule IDs"):
        make_target_examples([records[0], records[0]], "egfr", config)


def test_invalid_activity_threshold(records, config):
    with pytest.raises(ValueError, match="threshold must be finite"):
        make_dual_activity_examples(
            records, config, threshold_pic50=float("nan")
        )


def test_regression_is_reproducible(records, config):
    train = make_target_examples(records, "egfr", config)

    model_a = fit_random_forest(train, config)
    model_b = fit_random_forest(train, config)

    np.testing.assert_array_equal(
        model_a.predict(train.features),
        model_b.predict(train.features),
    )


def test_regression_metrics(records, config):
    examples = make_target_examples(records, "egfr", config)
    predictions = examples.pic50 + 0.5

    model = SimpleNamespace(predict=lambda features: predictions)

    metrics, output = evaluate_random_forest(model, examples)

    assert metrics["n"] == 8
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["spearman"] == pytest.approx(1.0)
    np.testing.assert_array_equal(output, predictions)


def test_constant_predictions_have_undefined_spearman(records, config):
    examples = make_target_examples(records, "egfr", config)

    model = SimpleNamespace(
        predict=lambda features: np.full(len(examples.pic50), 6.0)
    )

    metrics, _ = evaluate_random_forest(model, examples)

    assert metrics["spearman"] is None


def test_classifier_fits_and_predicts(records, config):
    train = make_dual_activity_examples(records, config)
    model = fit_dual_activity_classifier(train, config)

    metrics, scores = evaluate_dual_activity_classifier(model, train)

    # This is a functionality test, not an unbiased performance estimate.
    assert len(scores) == 7
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert metrics["n"] == 7
    assert metrics["positives"] == 3
    assert metrics["prevalence"] == pytest.approx(3 / 7)
    assert 0.0 <= metrics["average_precision"] <= 1.0


def test_average_precision_computation(records, config):
    examples = make_dual_activity_examples(records, config)

    positive_scores = np.where(examples.dual_active == 1, 0.9, 0.1)

    # Deliberately reverse class order to test positive-class lookup.
    model = SimpleNamespace(
        classes_=np.array([1, 0]),
        predict_proba=lambda features: np.column_stack(
            [positive_scores, 1.0 - positive_scores]
        ),
    )

    metrics, scores = evaluate_dual_activity_classifier(model, examples)

    assert metrics["average_precision"] == pytest.approx(1.0)
    np.testing.assert_array_equal(scores, positive_scores)


def test_classifier_requires_both_classes(records, config):
    examples = make_dual_activity_examples(records, config)
    single_class = replace(
        examples,
        dual_active=np.ones_like(examples.dual_active),
    )

    with pytest.raises(ValueError, match="needs both"):
        fit_dual_activity_classifier(single_class, config)

    model = fit_dual_activity_classifier(examples, config)

    with pytest.raises(ValueError, match="needs both"):
        evaluate_dual_activity_classifier(model, single_class)
