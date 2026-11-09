"""M1.8c: join verified M1.7/M1.8a/M1.8b evidence and assign provisional IDs.

Observation-level audit only. Never collapse records, transform original SMILES,
strip salt, neutralize, normalize tautomer, assign missing stereochemistry, or
aggregate potency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from dual_target.standardization import (
    classify_structure,
    structure_id,
    unspecified_stereogenic_elements,
)
from rdkit import rdBase

REQUIRED_POLICY_FIELDS = {
    "reactant_set_id", "smiles", "inchi_key", "target", "m17_partition",
    "ic50_raw", "pic50", "provenance_key", "construct_category",
}
REQUIRED_QA_FIELDS = {
    "reactant_set_id", "smiles", "inchi_key", "target", "m17_partition",
    "rdkit_status", "rdkit_canonical_isomeric_smiles", "rdkit_inchi_key",
    "rdkit_fragment_count", "source_key_comparison",
}
REQUIRED_PRIORITY_FIELDS = {"smiles", "identity_review_class", "review_priority"}
HIGH_CLASSES = {"different_connectivity_block", "rdkit_unavailable"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_unique_rows(path: Path, required: set[str], key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
        indexed = {}
        for row in reader:
            identifier = row[key]
            if not identifier or identifier in indexed:
                raise ValueError(f"{path}: missing or duplicate key {key}={identifier!r}")
            indexed[identifier] = row
    return indexed


def write_figure(summary: dict, output: Path) -> None:
    """Show both the large usable pool and small, important review holds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [
        "primary_exact_candidate", "primary_classification_candidate",
        "secondary_construct_reserve", "primary_hold_measurement",
    ]
    held = [
        "hold_inchikey_connectivity_disagreement",
        "hold_rdkit_parse_failure",
        "hold_multifragment",
    ]
    labels = ["Primary exact", "Primary censored", "Construct reserve", "Measurement hold"]
    data = summary["by_m17_partition"]
    fig, (ax_total, ax_holds) = plt.subplots(
        1, 2, figsize=(13, 5.1), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    accepted = [data.get(group, {}).get("provisional_exact_structure", 0) for group in order]
    ax_total.barh(labels, accepted, label="Provisional structure ID")
    ax_total.invert_yaxis()
    ax_total.set_xlabel("Observations assigned provisional structure IDs")
    ax_total.set_title("Provisional identity assignment")
    for i, count in enumerate(accepted):
        ax_total.text(count + 150, i, f"{count:,}", va="center", fontsize=9)
    ax_total.set_xlim(0, max(accepted) * 1.17)

    left = [0] * len(order)
    for status in held:
        values = [data.get(group, {}).get(status, 0) for group in order]
        ax_holds.barh(
            labels, values, left=left, label=status.replace("hold_", "").replace("_", " ")
        )
        left = [x + y for x, y in zip(left, values, strict=True)]
    ax_holds.invert_yaxis()
    ax_holds.set_xlabel("Observations held for review")
    ax_holds.set_title("Review holds (magnified)")
    for i, count in enumerate(left):
        ax_holds.text(count + 0.7, i, str(count), va="center", fontsize=9)
    ax_holds.set_xlim(0, max(left) * 1.4)
    ax_holds.legend(fontsize=8, loc="lower right")
    fig.suptitle("M1.8c — conservative identity decisions (observation level)", fontsize=13)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(
    policy_path: Path,
    qa_path: Path,
    priority_path: Path,
    output: Path,
    summary_path: Path,
    collisions_path: Path,
    figure_path: Path | None,
) -> dict:
    qa_by_id = read_unique_rows(qa_path, REQUIRED_QA_FIELDS, "reactant_set_id")
    priority_by_smiles = read_unique_rows(priority_path, REQUIRED_PRIORITY_FIELDS, "smiles")
    flagged = set(priority_by_smiles)

    # Fail closed: a stale/incomplete priority list cannot silently promote
    # connectivity-discrepant or RDKit-unparseable structures.
    actual_high = {
        item["smiles"] for item in qa_by_id.values()
        if item["rdkit_status"] != "parsed"
        or item["source_key_comparison"] == "different_connectivity_block"
    }
    if actual_high != flagged:
        raise ValueError(
            "High-priority SMILES do not match M1.8a identity QA "
            f"(missing={len(actual_high - flagged)}, extra={len(flagged - actual_high)})"
        )
    for smiles, review in priority_by_smiles.items():
        if review["review_priority"] != "high" or review["identity_review_class"] not in HIGH_CLASSES:
            raise ValueError(f"Unexpected priority-review classification for SMILES {smiles!r}")

    result_counts = Counter()
    by_target = defaultdict(Counter)
    by_partition = defaultdict(Counter)
    raw_smiles_per_canon = defaultdict(set)
    canon_per_rdkit_key = defaultdict(set)
    observed_ids = set()
    ids_by_target = defaultdict(set)
    exact_ids_by_target = defaultdict(set)
    original_partition_counts = Counter()
    raw_unspecified = {}
    out_fields = [
        "m18c_status", "m18c_structure_id", "m18c_candidate_isomeric_smiles",
        "m18c_candidate_rdkit_inchikey", "m18c_source_key_comparison",
        "m18c_fragment_count", "m18c_unspecified_stereogenic_elements",
        "m18c_review_reason",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".partial")
    try:
        with policy_path.open(encoding="utf-8-sig", newline="") as src, tmp.open(
            "w", encoding="utf-8", newline=""
        ) as dst:
            reader = csv.DictReader(src)
            missing = REQUIRED_POLICY_FIELDS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{policy_path}: missing required columns {sorted(missing)}")
            writer = csv.DictWriter(dst, fieldnames=[*(reader.fieldnames or []), *out_fields])
            writer.writeheader()
            for record in reader:
                record_id = record["reactant_set_id"]
                if not record_id or record_id in observed_ids:
                    raise ValueError(f"Missing/duplicate M1.7 reactant_set_id: {record_id!r}")
                observed_ids.add(record_id)
                qa = qa_by_id.get(record_id)
                if qa is None:
                    raise ValueError(f"Missing M1.8a QA record: {record_id}")
                for key in ("smiles", "inchi_key", "target", "m17_partition"):
                    if record[key] != qa[key]:
                        raise ValueError(f"M1.7/M1.8a mismatch at {record_id} for {key}")

                status = classify_structure(qa, flagged)
                canonical = qa["rdkit_canonical_isomeric_smiles"]
                rdkit_key = qa["rdkit_inchi_key"]
                if status == "provisional_exact_structure":
                    identity = structure_id(canonical)
                    ids_by_target[record["target"]].add(identity)
                    if record["m17_partition"] == "primary_exact_candidate":
                        exact_ids_by_target[record["target"]].add(identity)
                else:
                    identity = ""
                if canonical:
                    raw_smiles_per_canon[canonical].add(record["smiles"])
                    if rdkit_key:
                        canon_per_rdkit_key[rdkit_key].add(canonical)
                if canonical not in raw_unspecified:
                    raw_unspecified[canonical] = unspecified_stereogenic_elements(canonical)
                stereos = raw_unspecified[canonical]

                writer.writerow({
                    **record,
                    "m18c_status": status,
                    "m18c_structure_id": identity,
                    "m18c_candidate_isomeric_smiles": canonical,
                    "m18c_candidate_rdkit_inchikey": rdkit_key,
                    "m18c_source_key_comparison": qa["source_key_comparison"],
                    "m18c_fragment_count": qa["rdkit_fragment_count"],
                    "m18c_unspecified_stereogenic_elements": "" if stereos is None else stereos,
                    "m18c_review_reason": (
                        "none_provisional_not_final_curated_identity"
                        if identity else status
                    ),
                })
                result_counts[status] += 1
                by_target[record["target"]][status] += 1
                by_partition[record["m17_partition"]][status] += 1
                original_partition_counts[record["m17_partition"]] += 1
        if observed_ids != set(qa_by_id):
            raise ValueError("M1.7 and M1.8a record ID sets disagree")
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)

    # Report representation collisions WITHOUT changing or merging structures.
    collisions_path.parent.mkdir(parents=True, exist_ok=True)
    collision_rows = []
    for canon, raw_set in raw_smiles_per_canon.items():
        if len(raw_set) > 1:
            collision_rows.append({
                "collision_type": "multiple_raw_smiles_same_isomeric_canonical",
                "identifier": canon,
                "distinct_structures_or_representations": len(raw_set),
            })
    for key, canon_set in canon_per_rdkit_key.items():
        if len(canon_set) > 1:
            collision_rows.append({
                "collision_type": "one_rdkit_inchikey_multiple_canonical_smiles",
                "identifier": key,
                "distinct_structures_or_representations": len(canon_set),
            })
    with collisions_path.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=[
            "collision_type", "identifier", "distinct_structures_or_representations",
        ])
        writer.writeheader()
        writer.writerows(sorted(collision_rows, key=lambda r: (r["collision_type"], r["identifier"])))

    summary = {
        "stage": "M1.8c_provisional_structure_identity_NOT_assay_aggregation",
        "generated_utc": datetime.now(UTC).isoformat(),
        "rdkit_version_for_stereochemistry_flags": rdBase.rdkitVersion,
        "structure_identity_policy": "RDKit M1.8a canonical isomeric SMILES; SHA256 policy-v1 token; exact textual canonical identity only",
        "no_chemical_transforms": "No salt stripping, fragment selection, uncharging, tautomer normalization, stereo completion, or raw SMILES edits",
        "input": {
            "policy_file": policy_path.name, "policy_sha256": sha256_file(policy_path),
            "identity_file": qa_path.name, "identity_sha256": sha256_file(qa_path),
            "priority_file": priority_path.name, "priority_sha256": sha256_file(priority_path),
        },
        "observations_retained": len(observed_ids),
        "unique_raw_smiles": len({r["smiles"] for r in qa_by_id.values()}),
        "high_priority_raw_smiles_quarantined": len(flagged),
        "observations_by_structure_status": dict(sorted(result_counts.items())),
        "by_target": {k: dict(sorted(v.items())) for k, v in sorted(by_target.items())},
        "by_m17_partition": {k: dict(sorted(v.items())) for k, v in sorted(by_partition.items())},
        "m17_partition_totals_retained": dict(sorted(original_partition_counts.items())),
        "canonical_smiles_with_multiple_raw_input_representations": sum(
            len(v) > 1 for v in raw_smiles_per_canon.values()
        ),
        "rdkit_inchikeys_corresponding_to_multiple_canonical_smiles": sum(
            len(v) > 1 for v in canon_per_rdkit_key.values()
        ),
        "provisional_structure_id_overlap_egfr_her2_all_roles": len(
            ids_by_target.get("EGFR", set()) & ids_by_target.get("HER2", set())
        ),
        "provisional_primary_exact_id_overlap_egfr_her2": len(
            exact_ids_by_target.get("EGFR", set()) & exact_ids_by_target.get("HER2", set())
        ),
        "output": {
            "observation_file": str(output),
            "collision_inventory": str(collisions_path),
        },
        "limitations": [
            "A structure ID is provisional and does not imply confirmed formulation, stereoisomer purity or assay equivalence.",
            "RDKit canonical strings come from the prior M1.8a audit; re-run upstream QA after RDKit or policy changes.",
            "Unspecified potentially stereogenic elements are flagged, never assigned by inference.",
            "InChIKey is descriptive, not the primary identity join key.",
            "High-priority and multicomponent records are retained with no approved structure ID.",
            "No IC50 labels merged, no dataset split, and no model trained.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if figure_path:
        write_figure(summary, figure_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="M1.7 decisions CSV")
    parser.add_argument("--identity", required=True, type=Path, help="M1.8a identity QA CSV")
    parser.add_argument("--priority", required=True, type=Path, help="M1.8b high-priority CSV")
    parser.add_argument("--output", required=True, type=Path, help="New M1.8c observations CSV")
    parser.add_argument("--summary", required=True, type=Path, help="M1.8c JSON report")
    parser.add_argument("--collisions", required=True, type=Path, help="Representation collisions CSV")
    parser.add_argument("--figure", type=Path, help="Optional summary PNG")
    args = parser.parse_args()
    result = run(
        args.input, args.identity, args.priority,
        args.output, args.summary, args.collisions, args.figure,
    )
    print(f"Observations retained: {result['observations_retained']:,}")
    print(f"Distinct raw SMILES: {result['unique_raw_smiles']:,}")
    for name, value in result["observations_by_structure_status"].items():
        print(f"  {name}: {value:,}")
    print(f"High-priority raw SMILES quarantined: {result['high_priority_raw_smiles_quarantined']:,}")
    print(f"Primary exact overlap (provisional IDs only): {result['provisional_primary_exact_id_overlap_egfr_her2']:,}")
    print(f"Summary: {args.summary}")
    print(f"Observation QA: {args.output}")
    print(f"Collision inventory: {args.collisions}")
    if args.figure:
        print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
