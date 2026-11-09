"""M1.11: reversible, masked EGFR/HER2 assembly from M1.10 provisional labels.

No assay-level confirmation, train/test split, fitted model or label imputation.
All source observations remain available for audits and separate construct review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import rdBase

TARGETS = ("EGFR", "HER2")
ACCEPT = {"provisional_single_exact", "provisional_tight_consensus"}
HOLD = {
    "hold_censored_activity_contradiction",
    "hold_exact_activity_contradiction",
    "hold_major_disagreement_gt1_pic50",
    "hold_moderate_disagreement",
}
REQUIRED_LABELS = {
    "target", "m18c_structure_id", "canonical_isomeric_smiles",
    "m10_policy_status", "m10_provisional_pic50", "m10_provisional_activity",
    "m10_label_support", "m10_target_construct_verification",
    "m10_exact_observation_ids", "n_exact_observations", "n_provenance_keys",
    "has_exact_activity_conflict", "has_censored_activity_conflict",
    "has_annotated_construct_for_same_structure_target",
    "min_exact_pic50", "max_exact_pic50", "exact_pic50_range",
}
REQUIRED_SOURCE = {
    "reactant_set_id", "target", "m17_partition", "m18c_status",
    "m18c_structure_id", "m18c_candidate_isomeric_smiles", "pic50",
    "target_name", "construct_category", "mutation_signature", "measurement_status",
    "ic50_raw", "ic50_nm", "ic50_relation", "source", "provenance_key",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_identity(structure_id: str, canonical_smiles: str) -> None:
    if not structure_id or not canonical_smiles:
        raise ValueError("Missing structure ID or canonical isomeric SMILES")
    expected = "RDKISO1-" + hashlib.sha256(
        b"rdkit-isomeric-smiles-policy-v1\0" + canonical_smiles.encode("utf-8")
    ).hexdigest()
    if expected != structure_id:
        raise ValueError(f"M1.8c chemical identity digest disagrees: {structure_id}")


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path}: missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        return list(reader)


def verified_bool(text: str, field: str) -> bool:
    if text not in ("True", "False"):
        raise ValueError(f"Unexpected {field} boolean {text!r}")
    return text == "True"


def source_index(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], list[dict[str, str]], Counter]:
    """Index only approved exact source measurements, retain all reserve rows separately."""
    exact_by_id = {}
    reserve = []
    counts = Counter()
    all_ids = set()
    for row in rows:
        rid = row["reactant_set_id"]
        if not rid or rid in all_ids:
            raise ValueError("Missing or duplicate reactant_set_id in standardized source")
        all_ids.add(rid)
        counts["total_source_observations"] += 1
        role = row["m17_partition"]
        approved = row["m18c_status"] == "provisional_exact_structure"
        if role == "primary_exact_candidate" and approved:
            exact_by_id[rid] = row
            counts["eligible_exact_observations"] += 1
        if role == "secondary_construct_reserve":
            reserve.append(row)
            counts["reserved_construct_observations"] += 1
            if approved:
                counts["reserved_with_structure_id"] += 1
    return exact_by_id, reserve, counts


def build_dataset(
    candidates: list[dict[str, str]],
    exact_by_id: dict[str, dict[str, str]],
    reserve: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict]:
    """Return (masked multi-task rows, all pair disposition rows, construct reserve, QA).

    Empty numeric strings with mask=0 are missing labels, not zero-potency labels.
    """
    by_structure: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    source_used = set()
    label_count = Counter()
    supported_by_exact = Counter()
    all_accepted = set()
    for r in candidates:
        target = r["target"]
        if target not in TARGETS:
            raise ValueError(f"Unexpected target {target!r}")
        sid = r["m18c_structure_id"]
        smiles = r["canonical_isomeric_smiles"]
        check_identity(sid, smiles)
        if target in by_structure[sid]:
            raise ValueError(f"Duplicate molecule-target row: {target} {sid}")
        if by_structure[sid] and any(
            x["canonical_isomeric_smiles"] != smiles for x in by_structure[sid].values()
        ):
            raise ValueError(f"Inconsistent canonical SMILES for structure {sid}")
        status = r["m10_policy_status"]
        if status not in ACCEPT | HOLD:
            raise ValueError(f"Unrecognized M1.10 decision: {status}")
        ids = r["m10_exact_observation_ids"].split(";")
        if not ids or any(not entry for entry in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid original observation IDs in {target} {sid}")
        if len(ids) != int(r["n_exact_observations"]):
            raise ValueError(f"Incorrect exact observation count in {target} {sid}")
        origin_values = []
        for rid in ids:
            if rid in source_used:
                raise ValueError(f"Exact observation {rid} assigned to more than one pair")
            source_used.add(rid)
            original = exact_by_id.get(rid)
            if original is None or original["target"] != target or original["m18c_structure_id"] != sid:
                raise ValueError(f"Original exact observation {rid} not in expected target/structure")
            if original["m18c_candidate_isomeric_smiles"] != smiles:
                raise ValueError(f"SMILES mismatch in original observation {rid}")
            origin_values.append(float(original["pic50"]))
        if not all(math.isfinite(v) for v in origin_values):
            raise ValueError("Nonfinite source pIC50")
        if not math.isclose(min(origin_values), float(r["min_exact_pic50"]), abs_tol=1e-7):
            raise ValueError(f"Original minimum pIC50 mismatched: {target} {sid}")
        if not math.isclose(max(origin_values), float(r["max_exact_pic50"]), abs_tol=1e-7):
            raise ValueError(f"Original maximum pIC50 mismatched: {target} {sid}")
        if not math.isclose(max(origin_values) - min(origin_values), float(r["exact_pic50_range"]), abs_tol=1e-7):
            raise ValueError("Original exact pIC50 range mismatch")
        econf = verified_bool(r["has_exact_activity_conflict"], "exact conflict")
        cconf = verified_bool(r["has_censored_activity_conflict"], "censored conflict")
        verified_bool(r["has_annotated_construct_for_same_structure_target"], "construct overlap")
        if r["m10_target_construct_verification"] != "NOT_sequence_or_assay_verified":
            raise ValueError("Unexpected assay verification claim in provisional dataset")
        accepted = status in ACCEPT
        if accepted:
            pic50_text = r["m10_provisional_pic50"]
            if not pic50_text:
                raise ValueError("Accepted status with no numerical potency")
            p = float(pic50_text)
            if not math.isfinite(p):
                raise ValueError("Nonfinite candidate label")
            if econf or cconf:
                raise ValueError("Accepted candidate has unresolved activity conflict")
            expected_activity = "active" if p > 6.0 else "inactive"
            if r["m10_provisional_activity"] != expected_activity:
                raise ValueError("Activity classification inconsistent with strict 1000 nM cutoff")
            if status == "provisional_single_exact" and len(ids) != 1:
                raise ValueError("Single-exact decision has multiple records")
            if status == "provisional_tight_consensus" and (len(ids) <= 1 or float(r["exact_pic50_range"]) > 0.30000001):
                raise ValueError("Tight consensus inconsistent with 0.30 rule")
            all_accepted.add(sid)
            label_count["accepted_" + target] += 1
        else:
            if r["m10_provisional_pic50"] or r["m10_provisional_activity"]:
                raise ValueError("Held pair must not have proposed numeric/activity label")
            label_count["held_" + target] += 1
        label_count[status] += 1
        supported_by_exact[target] += len(ids)
        by_structure[sid][target] = r
    if source_used != set(exact_by_id):
        raise ValueError("M1.10 candidate rows do not account for every approved exact source observation")

    reserve_manifest = []
    reserved_pairs = set()
    for r in reserve:
        sid = r["m18c_structure_id"]
        if sid:
            reserved_pairs.add((sid, r["target"]))
        reserve_manifest.append({
            "reactant_set_id": r["reactant_set_id"],
            "target": r["target"],
            "m18c_structure_id": sid,
            "m18c_status": r["m18c_status"],
            "original_target_name": r["target_name"],
            "construct_category": r["construct_category"],
            "mutation_signature": r["mutation_signature"],
            "measurement_status": r["measurement_status"],
            "ic50_raw": r["ic50_raw"],
            "ic50_relation": r["ic50_relation"],
            "ic50_nm": r["ic50_nm"],
            "source": r["source"],
            "provenance_key": r["provenance_key"],
            "eligible_for_primary_training": "False",
        })

    assembled = []
    pair_audit = []
    qc = Counter(label_count)
    qc["all_provisional_structure_ids_seen"] = len(by_structure)
    qc["accepted_structure_ids"] = len(all_accepted)
    qc["reserved_construct_rows"] = len(reserve_manifest)
    qc["reserved_pairs_with_structure_id"] = len(reserved_pairs)
    qc["accepted_source_exact_observations"] = len(source_used)
    qc["annotated_construct_overlap_accepted_pairs"] = 0
    for sid, target_dict in sorted(by_structure.items()):
        smiles = next(iter(target_dict.values()))["canonical_isomeric_smiles"]
        for target, r in target_dict.items():
            active = r["m10_policy_status"] in ACCEPT
            overlap = (sid, target) in reserved_pairs
            declared = verified_bool(r["has_annotated_construct_for_same_structure_target"], "construct overlap")
            if overlap != declared:
                raise ValueError(f"M1.10 construct overlap disagrees with M1.8c: {target} {sid}")
            if active and overlap:
                qc["annotated_construct_overlap_accepted_pairs"] += 1
            pair_audit.append({
                "m18c_structure_id": sid, "target": target,
                "m10_policy_status": r["m10_policy_status"],
                "m10_provisional_pic50": r["m10_provisional_pic50"],
                "m11_regression_mask": int(active),
                "m11_selected_in_multitask_dataset": int(sid in all_accepted),
                "original_exact_observation_ids": r["m10_exact_observation_ids"],
                "n_exact_observations": r["n_exact_observations"],
                "n_provenance_keys": r["n_provenance_keys"],
                "has_annotated_construct_for_same_structure_target": str(overlap),
                "m10_label_support": r["m10_label_support"],
            })
        if sid not in all_accepted:
            qc["held_only_structure_ids"] += 1
            continue
        row: dict[str, object] = {
            "m18c_structure_id": sid,
            "canonical_isomeric_smiles": smiles,
            "provisional_identity_policy": "rdkit-isomeric-smiles-policy-v1",
            "assay_construct_verification": "NOT_sequence_or_assay_verified",
            "training_split": "NOT_ASSIGNED",
            "scaffold_split": "NOT_ASSIGNED",
        }
        mask_sum = 0
        for target in TARGETS:
            r = target_dict.get(target)
            key = target.lower()
            accepted = r is not None and r["m10_policy_status"] in ACCEPT
            mask = int(accepted)
            mask_sum += mask
            row[key + "_mask"] = mask
            row[key + "_pic50"] = r["m10_provisional_pic50"] if accepted else ""
            row[key + "_activity"] = r["m10_provisional_activity"] if accepted else ""
            row[key + "_status"] = r["m10_policy_status"] if r else "no_primary_exact_evidence"
            row[key + "_support"] = r["m10_label_support"] if accepted else ""
            row[key + "_n_exact_observations"] = r["n_exact_observations"] if r else "0"
            row[key + "_n_provenance_keys"] = r["n_provenance_keys"] if r else "0"
            row[key + "_annotated_construct_overlap"] = int((sid, target) in reserved_pairs)
            if accepted:
                qc["accepted_single_exact"] += r["m10_policy_status"] == "provisional_single_exact"
                qc["accepted_tight_consensus"] += r["m10_policy_status"] == "provisional_tight_consensus"
        if mask_sum not in (1, 2):
            raise ValueError("Model row must have at least one label")
        row["n_available_regression_tasks"] = mask_sum
        if mask_sum == 2:
            qc["dual_labeled_structures"] += 1
            a, b = row["egfr_activity"], row["her2_activity"]
            row["paired_activity_class"] = (
                "dual_active" if a == b == "active" else
                "dual_inactive" if a == b == "inactive" else
                "EGFR_only_active" if a == "active" else "HER2_only_active"
            )
            qc[str(row["paired_activity_class"])] += 1
        else:
            row["paired_activity_class"] = "not_evaluable_missing_task_label"
            qc["EGFR_only_labeled_structures" if row["egfr_mask"] else "HER2_only_labeled_structures"] += 1
        assembled.append(row)
    if len(assembled) != len(all_accepted):
        raise ValueError("Final row count inconsistent with accepted identity set")
    if qc["accepted_EGFR"] + qc["accepted_HER2"] != sum(
        x["egfr_mask"] + x["her2_mask"] for x in assembled
    ):
        raise ValueError("Mask sums differ from M1.10 accepted-label count")
    return assembled, pair_audit, reserve_manifest, dict(qc)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No output rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path: Path, qc: dict) -> None:
    names = ["Both targets\nlabeled", "EGFR only\nlabeled", "HER2 only\nlabeled", "Held-only\nstructures"]
    values = [qc["dual_labeled_structures"], qc["EGFR_only_labeled_structures"],
              qc["HER2_only_labeled_structures"], qc["held_only_structure_ids"]]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    bars = ax.bar(names, values, color=["#155E75", "#3B82A0", "#8AB6C7", "#BCC8D0"], width=0.65)
    ax.bar_label(bars, padding=5, fmt="{:,.0f}")
    ax.set_ylabel("Unique M1.8c provisional molecular identities")
    ax.set_title("M1.11 — Reconciled provisional potency-label coverage", fontsize=14, pad=17)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(values) * 1.18)
    fig.text(0.02, 0.01, "Counts describe provisional evidence, not confirmed wild-type biochemical inhibition.", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--m10-summary", type=Path, required=True)
    parser.add_argument("--standardized", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--construct-reserve", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    m10 = json.loads(args.m10_summary.read_text(encoding="utf-8"))
    if m10.get("stage") != "M1.10_provisional_exact_potency_label_reconciliation_NOT_validated_assay_labels":
        raise ValueError("Expected M1.10 provisional reconciliation summary")
    input_hash = sha256_file(args.standardized)
    if input_hash != m10["inputs"]["input_sha256"]:
        raise ValueError("Standardized source SHA-256 does not match M1.10 input")
    candidates = read_csv(args.labels, REQUIRED_LABELS)
    source = read_csv(args.standardized, REQUIRED_SOURCE)
    exact, reserve, upstream = source_index(source)
    dataset, pair_audit, reserve_manifest, qc = build_dataset(candidates, exact, reserve)
    counts = m10["analysis"]
    checks = {
        "candidate_molecule_target_pairs": (len(candidates), counts["candidate_molecule_target_pairs"]),
        "provisional_regression_labels": (qc["accepted_EGFR"] + qc["accepted_HER2"], counts["provisional_regression_labels"]),
        "held_regression_pairs": (qc["held_EGFR"] + qc["held_HER2"], counts["held_regression_pairs"]),
        "eligible_exact_observations": (len(exact), counts["eligible_exact_observations"]),
        "total_input_observations": (upstream["total_source_observations"], counts["input_observations"]),
    }
    for target in TARGETS:
        checks[target + "_provisional_labels"] = (
            qc["accepted_" + target], counts["per_target"][target]["provisional_labels"]
        )
    for key, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"M1.11 failed upstream reconciliation: {key} {actual} != {expected}")
    write_csv(args.dataset, dataset)
    write_csv(args.pair_audit, pair_audit)
    write_csv(args.construct_reserve, reserve_manifest)
    make_figure(args.figure, qc)
    summary = {
        "stage": "M1.11_masked_paired_molecular_dataset_PROVISIONAL_NOT_SPLIT_OR_VERIFIED",
        "generated_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "labels_file": args.labels.name,
            "labels_sha256": sha256_file(args.labels),
            "m10_summary_file": args.m10_summary.name,
            "m10_summary_sha256": sha256_file(args.m10_summary),
            "standardized_file": args.standardized.name,
            "standardized_sha256": input_hash,
            "rdkit_version_provenance": rdBase.rdkitVersion,
        },
        "analysis": {
            "accepted_molecule_target_labels": qc["accepted_EGFR"] + qc["accepted_HER2"],
            "EGFR_labels": qc["accepted_EGFR"],
            "HER2_labels": qc["accepted_HER2"],
            "unique_accepted_structures": len(dataset),
            "dual_labeled_structures": qc["dual_labeled_structures"],
            "EGFR_only_labeled_structures": qc["EGFR_only_labeled_structures"],
            "HER2_only_labeled_structures": qc["HER2_only_labeled_structures"],
            "dual_activity_class_counts": {k: qc.get(k, 0) for k in
                ("dual_active", "dual_inactive", "EGFR_only_active", "HER2_only_active")},
            "all_m10_candidate_pairs_retained_in_disposition_audit": len(pair_audit),
            "held_molecule_target_pairs": qc["held_EGFR"] + qc["held_HER2"],
            "held_only_structures_excluded_from_dataset": qc["held_only_structure_ids"],
            "accepted_single_exact_labels": qc["accepted_single_exact"],
            "accepted_tight_consensus_labels": qc["accepted_tight_consensus"],
            "reserved_annotated_construct_observations": len(reserve_manifest),
            "reserved_annotated_construct_with_approved_identity": upstream["reserved_with_structure_id"],
            "accepted_pair_annotations_overlapping_construct_reserve": qc["annotated_construct_overlap_accepted_pairs"],
            "identity_digests_and_exact_source_links_checked": True,
            "missing_values_are_blank_and_controlled_by_masks": True,
            "train_val_test_splits_created": False,
        },
        "outputs": {
            "multitask_dataset": str(args.dataset),
            "full_pair_dispositions": str(args.pair_audit),
            "separate_construct_reserve": str(args.construct_reserve),
            "figure": str(args.figure),
        },
        "limitations": [
            "Every label is provisional and unverified at protein-sequence/assay-endpoint level.",
            "Missing target labels remain blank, never zero and never imputed.",
            "A structure with one held target label can still enter the dataset for its other accepted target.",
            "Held-only structures are preserved in the pair disposition export but excluded from modeling rows.",
            "Annotated constructs are separate from primary training candidates, including overlapping structures.",
            "No Bemis-Murcko scaffold split, source-held-out split, training, or external validation performed.",
            "Dual-active status requires both accepted target labels and is only a provisional classification.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"M1.10 accepted labels reconciled: {summary['analysis']['accepted_molecule_target_labels']:,}")
    print(f"Unique provisional molecular structures: {len(dataset):,}")
    print(f"Dual-labeled structures: {qc['dual_labeled_structures']:,}")
    print(f"EGFR-only: {qc['EGFR_only_labeled_structures']:,}; HER2-only: {qc['HER2_only_labeled_structures']:,}")
    print(f"Held-only structures preserved in audit, excluded from dataset: {qc['held_only_structure_ids']:,}")
    print(f"Annotated-construct reserve observations: {len(reserve_manifest):,}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
