"""Stage C scaffold-calibration partition tests using synthetic molecules only.

These tests do not load the frozen M2.1 dataset, development-validation labels,
or the locked test split. They exercise split reproducibility and safeguards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from dual_target.calibration import scaffold_calibration as sc


@dataclass(frozen=True)
class FakeRecord:
    structure_id: str
    scaffold_group_id: str
    egfr_pic50: float
    her2_pic50: float
    egfr_mask: bool
    her2_mask: bool


def make_records(prefix: str, n_scaffolds: int = 72) -> tuple[FakeRecord, ...]:
    """Give every scaffold paired positives or negatives plus masked examples."""
    records = []
    for group in range(n_scaffolds):
        for member in range(3):
            records.append(
                FakeRecord(
                    structure_id=f"{prefix}-m{group:03d}-{member}",
                    scaffold_group_id=f"{prefix}-scaffold-{group:03d}",
                    egfr_pic50=6.8 if group % 2 == 0 else 5.6,
                    her2_pic50=6.7 if group % 2 == 0 else 5.5,
                    egfr_mask=member != 2 or group % 4 == 0,
                    her2_mask=member != 1 or group % 5 == 0,
                )
            )
    return tuple(records)


def make_pipeline(
    *,
    train: tuple[FakeRecord, ...] | None = None,
    validation: tuple[FakeRecord, ...] | None = None,
):
    return SimpleNamespace(
        data=SimpleNamespace(
            train=make_records("train") if train is None else train,
            validation=make_records("validation", 8)
            if validation is None
            else validation,
            dataset_sha256="synthetic-dataset-sha",
            manifest_sha256="synthetic-manifest-sha",
        )
    )


@pytest.fixture
def example():
    return make_pipeline()


def split_ids(split):
    return (
        {row.structure_id for row in split.fit},
        {row.structure_id for row in split.calibration},
        {row.structure_id for row in split.threshold},
    )


def split_scaffolds(split):
    return (
        {row.scaffold_group_id for row in split.fit},
        {row.scaffold_group_id for row in split.calibration},
        {row.scaffold_group_id for row in split.threshold},
    )


def test_creates_complete_nonempty_three_way_training_split(example):
    split = sc.make_scaffold_calibration_split(example)
    ids = split_ids(split)
    assert all(ids)
    assert all(not ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3))
    assert set.union(*ids) == {r.structure_id for r in example.data.train}
    assert sum(map(len, ids)) == len(example.data.train)


def test_keeps_scaffolds_completely_separate(example):
    split = sc.make_scaffold_calibration_split(example)
    groups = split_scaffolds(split)
    assert all(groups)
    assert all(
        not groups[i] & groups[j]
        for i in range(3)
        for j in range(i + 1, 3)
    )
    assert set.union(*groups) == {r.scaffold_group_id for r in example.data.train}


def test_source_molecule_order_is_preserved_within_each_role(example):
    split = sc.make_scaffold_calibration_split(example)
    order = {row.structure_id: index for index, row in enumerate(example.data.train)}
    for subset in (split.fit, split.calibration, split.threshold):
        indexes = [order[row.structure_id] for row in subset]
        assert indexes == sorted(indexes)


def test_deterministic_split_and_hash_for_same_seed(example):
    a = sc.make_scaffold_calibration_split(example, seed=91)
    b = sc.make_scaffold_calibration_split(example, seed=91)
    assert a.assignments == b.assignments
    assert a.assignment_sha256 == b.assignment_sha256
    assert a.seed_used == b.seed_used


def test_different_seeds_change_assignment(example):
    a = sc.make_scaffold_calibration_split(example, seed=714)
    b = sc.make_scaffold_calibration_split(example, seed=715)
    assert a.assignment_sha256 != b.assignment_sha256


def test_different_requested_fractions_change_group_counts(example):
    a = sc.make_scaffold_calibration_split(example, calibration_fraction=0.20)
    b = sc.make_scaffold_calibration_split(example, calibration_fraction=0.30)
    assert a.calibration_audit.scaffolds != b.calibration_audit.scaffolds
    assert a.requested_calibration_fraction == pytest.approx(0.20)
    assert b.requested_calibration_fraction == pytest.approx(0.30)


def test_default_requested_fractions_apply_to_scaffold_counts(example):
    split = sc.make_scaffold_calibration_split(example)
    counts = [audit.scaffolds for audit in (
        split.fit_audit, split.calibration_audit, split.threshold_audit
    )]
    total_groups = len({r.scaffold_group_id for r in example.data.train})
    assert sum(counts) == total_groups
    assert abs(counts[0] / total_groups - 0.60) < 0.035
    assert abs(counts[1] / total_groups - 0.20) < 0.035
    assert abs(counts[2] / total_groups - 0.20) < 0.035


def test_all_partitions_meet_default_minimum_label_and_class_counts(example):
    split = sc.make_scaffold_calibration_split(example)
    for audit in (split.fit_audit, split.calibration_audit, split.threshold_audit):
        assert audit.egfr_labels >= 10
        assert audit.her2_labels >= 10
        assert audit.dual_active >= 5
        assert audit.not_dual_active >= 5


def test_audits_match_recomputed_source_counts(example):
    split = sc.make_scaffold_calibration_split(example)
    for subset, audit in (
        (split.fit, split.fit_audit),
        (split.calibration, split.calibration_audit),
        (split.threshold, split.threshold_audit),
    ):
        assert audit == sc.audit_calibration_records(subset)
        assert audit.molecules == len(subset)
        assert audit.dual_active + audit.not_dual_active == audit.paired_labels


def test_split_audit_sums_reconstruct_frozen_training_totals(example):
    split = sc.make_scaffold_calibration_split(example)
    original = sc.audit_calibration_records(example.data.train)
    for field in (
        "molecules", "scaffolds", "egfr_labels", "her2_labels", "paired_labels",
        "dual_active", "not_dual_active",
    ):
        assert sum(getattr(a, field) for a in (
            split.fit_audit, split.calibration_audit, split.threshold_audit
        )) == getattr(original, field)


def test_all_assignment_rows_match_training_source_order(example):
    split = sc.make_scaffold_calibration_split(example)
    assert tuple(a.structure_id for a in split.assignments) == tuple(
        r.structure_id for r in example.data.train
    )
    assert tuple(a.scaffold_group_id for a in split.assignments) == tuple(
        r.scaffold_group_id for r in example.data.train
    )
    assert {a.role for a in split.assignments} == {
        "fit", "calibration", "threshold"
    }


def test_assignment_role_matches_actual_subset_membership(example):
    split = sc.make_scaffold_calibration_split(example)
    expected = {
        record.structure_id: role
        for role, subset in (
            ("fit", split.fit),
            ("calibration", split.calibration),
            ("threshold", split.threshold),
        )
        for record in subset
    }
    assert {r.structure_id: r.role for r in split.assignments} == expected


def test_manifest_rows_only_contain_identity_group_and_role(example):
    split = sc.make_scaffold_calibration_split(example)
    rows = split.manifest_rows()
    assert len(rows) == len(example.data.train)
    assert all(set(row) == {"structure_id", "scaffold_group_id", "role"} for row in rows)
    assert all("pic50" not in key for row in rows for key in row)
    assert rows[0] == {
        "structure_id": split.assignments[0].structure_id,
        "scaffold_group_id": split.assignments[0].scaffold_group_id,
        "role": split.assignments[0].role,
    }


def test_hash_is_sha256_of_canonical_sorted_assignments(example):
    split = sc.make_scaffold_calibration_split(example)
    payload = [
        [row.structure_id, row.scaffold_group_id, row.role]
        for row in sorted(split.assignments, key=lambda row: row.structure_id)
    ]
    canonical = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    assert split.assignment_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    assert len(split.assignment_sha256) == 64


def test_dataset_manifest_provenance_is_propagated(example):
    split = sc.make_scaffold_calibration_split(example)
    assert split.dataset_sha256 == example.data.dataset_sha256
    assert split.manifest_sha256 == example.data.manifest_sha256


def test_seed_records_actual_attempt_seed(example):
    split = sc.make_scaffold_calibration_split(example, seed=42)
    assert split.seed_used == 42


def test_retry_after_unsuitable_attempt_is_deterministic(example, monkeypatch):
    original_ready = sc._ready
    calls = 0

    def reject_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return False
        return original_ready(*args, **kwargs)

    monkeypatch.setattr(sc, "_ready", reject_first)
    split = sc.make_scaffold_calibration_split(example, seed=714, max_attempts=3)
    assert split.seed_used == 715
    assert calls >= 4


def test_exhausted_retry_budget_raises(example, monkeypatch):
    monkeypatch.setattr(sc, "_ready", lambda *args, **kwargs: False)
    with pytest.raises(ValueError, match="No valid three-way scaffold split"):
        sc.make_scaffold_calibration_split(example, max_attempts=2)


def test_infeasible_minimum_class_requirement_raises(example):
    with pytest.raises(ValueError, match="No valid three-way scaffold split"):
        sc.make_scaffold_calibration_split(
            example, minimum_dual_per_class=200, max_attempts=2
        )


def test_audit_does_not_count_masked_labels_as_dual_negatives():
    records = (
        FakeRecord("a", "s1", 7.0, 7.0, True, True),
        FakeRecord("b", "s2", 7.0, 7.0, True, False),
        FakeRecord("c", "s3", 7.0, 7.0, False, True),
        FakeRecord("d", "s4", 5.0, 7.0, True, True),
    )
    audit = sc.audit_calibration_records(records)
    assert audit.molecules == 4
    assert audit.egfr_labels == 3
    assert audit.her2_labels == 3
    assert audit.paired_labels == 2
    assert audit.dual_active == 1
    assert audit.not_dual_active == 1


def test_audit_strict_activity_boundary_pic50_six_is_not_active():
    rows = (
        FakeRecord("a", "s1", 6.0, 8.0, True, True),
        FakeRecord("b", "s2", 8.0, 6.0, True, True),
        FakeRecord("c", "s3", 6.001, 6.001, True, True),
    )
    audit = sc.audit_calibration_records(rows)
    assert audit.paired_labels == 3
    assert audit.dual_active == 1
    assert audit.not_dual_active == 2


@pytest.mark.parametrize(
    ("calibration_fraction", "threshold_fraction"),
    [
        (0.0, 0.2), (0.2, 0.0), (-0.1, 0.2), (0.2, -0.1),
        (0.5, 0.5), (0.8, 0.3), (1.0, 0.0), (0.2, 1.0),
        (float("nan"), 0.2), (float("inf"), 0.2),
        (0.2, float("nan")), (0.2, float("-inf")),
    ],
)
def test_invalid_fractions_fail_before_splitting(
    example, calibration_fraction, threshold_fraction
):
    with pytest.raises(ValueError, match="Fractions must"):
        sc.make_scaffold_calibration_split(
            example,
            calibration_fraction=calibration_fraction,
            threshold_fraction=threshold_fraction,
        )


@pytest.mark.parametrize("seed", [-1, True, False, 1.5, "714"])
def test_invalid_seed_rejected(example, seed):
    with pytest.raises(ValueError, match="seed must"):
        sc.make_scaffold_calibration_split(example, seed=seed)


def test_zero_max_attempts_rejected(example):
    with pytest.raises(ValueError, match="max_attempts must"):
        sc.make_scaffold_calibration_split(example, max_attempts=0)


@pytest.mark.parametrize("keyword", ["minimum_target_labels", "minimum_dual_per_class"])
def test_nonpositive_label_minimum_rejected(example, keyword):
    with pytest.raises(ValueError, match="Minimum label/class"):
        sc.make_scaffold_calibration_split(example, **{keyword: 0})


def test_empty_training_partition_rejected(example):
    example.data.train = ()
    with pytest.raises(ValueError, match="training partition is empty"):
        sc.make_scaffold_calibration_split(example)


def test_duplicate_training_molecule_id_rejected(example):
    rows = list(example.data.train)
    rows[1] = replace(rows[1], structure_id=rows[0].structure_id)
    example.data.train = tuple(rows)
    with pytest.raises(ValueError, match="Duplicate training molecule"):
        sc.make_scaffold_calibration_split(example)


@pytest.mark.parametrize("identity", ["", "  ", None])
def test_missing_or_blank_training_structure_id_rejected(example, identity):
    rows = list(example.data.train)
    rows[0] = replace(rows[0], structure_id=identity)
    example.data.train = tuple(rows)
    with pytest.raises(ValueError, match="structure_id"):
        sc.make_scaffold_calibration_split(example)


@pytest.mark.parametrize("scaffold", ["", "  ", None])
def test_missing_or_blank_scaffold_group_id_rejected(example, scaffold):
    rows = list(example.data.train)
    rows[0] = replace(rows[0], scaffold_group_id=scaffold)
    example.data.train = tuple(rows)
    with pytest.raises(ValueError, match="scaffold_group_id"):
        sc.make_scaffold_calibration_split(example)


def test_training_validation_molecule_overlap_rejected(example):
    example.data.validation = (
        replace(example.data.validation[0], structure_id=example.data.train[0].structure_id),
        *example.data.validation[1:],
    )
    with pytest.raises(ValueError, match="Training/development-validation molecule overlap"):
        sc.make_scaffold_calibration_split(example)


def test_training_validation_scaffold_overlap_rejected(example):
    example.data.validation = (
        replace(example.data.validation[0], scaffold_group_id=example.data.train[0].scaffold_group_id),
        *example.data.validation[1:],
    )
    with pytest.raises(ValueError, match="Training/development-validation scaffold overlap"):
        sc.make_scaffold_calibration_split(example)


def test_too_few_training_scaffolds_rejected(example):
    example.data.train = make_records("train", 2)
    with pytest.raises(ValueError, match="At least three distinct training scaffolds"):
        sc.make_scaffold_calibration_split(example)


@pytest.mark.parametrize("target", ["egfr", "her2"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_accepted_potency_label_rejected(example, target, nonfinite):
    rows = list(example.data.train)
    rows[0] = replace(rows[0], **{f"{target}_pic50": nonfinite})
    example.data.train = tuple(rows)
    with pytest.raises(ValueError, match=f"Nonfinite accepted {target}"):
        sc.make_scaffold_calibration_split(example)


def test_masked_missing_potency_is_not_treated_as_accepted_label(example):
    rows = list(example.data.train)
    # Group 1 member 2 is HER2-only, so its EGFR value is masked.
    row = rows[5]
    assert not row.egfr_mask
    rows[5] = replace(row, egfr_pic50=float("nan"))
    example.data.train = tuple(rows)
    split = sc.make_scaffold_calibration_split(example)
    assert len(split.assignments) == len(rows)


def test_unusable_training_data_with_no_eligible_labels_fails(example):
    example.data.train = tuple(
        replace(row, egfr_mask=False, her2_mask=False)
        for row in example.data.train
    )
    with pytest.raises(ValueError, match="No valid three-way scaffold split"):
        sc.make_scaffold_calibration_split(example, max_attempts=2)


def test_no_test_property_is_accessed():
    class NoTestAccess:
        train = make_records("train")
        validation = make_records("validation", 8)
        dataset_sha256 = "dataset"
        manifest_sha256 = "manifest"

        @property
        def test(self):
            raise AssertionError("The locked test partition must never be read")

    result = sc.make_scaffold_calibration_split(SimpleNamespace(data=NoTestAccess()))
    assert len(result.assignments) == len(NoTestAccess.train)


def test_role_assignments_are_independent_of_validation_label_values(example):
    a = sc.make_scaffold_calibration_split(example)
    example.data.validation = tuple(
        replace(row, egfr_pic50=10.0, her2_pic50=4.0)
        for row in example.data.validation
    )
    b = sc.make_scaffold_calibration_split(example)
    assert a.assignments == b.assignments
    assert a.assignment_sha256 == b.assignment_sha256
