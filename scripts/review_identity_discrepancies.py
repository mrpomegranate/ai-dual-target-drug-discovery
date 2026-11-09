"""M1.8b: Investigate, but never auto-resolve, molecular-identity disagreements.

Inputs: M1.8a row-level identity QA CSV (NOT the original BindingDB archive).
Outputs: one row per raw SMILES, a priority review subset, and a public-safe JSON summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dual_target.identity_review import classify_identity, review_priority

REQUIRED_COLUMNS = {
    "reactant_set_id",
    "target",
    "m17_partition",
    "source",
    "smiles",
    "inchi_key",
    "rdkit_status",
    "rdkit_canonical_isomeric_smiles",
    "rdkit_inchi_key",
    "rdkit_fragment_count",
    "source_key_comparison",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_join(series: pd.Series) -> str:
    return "; ".join(sorted({str(x).strip() for x in series if str(x).strip()}))


def integer_or_none(value: str) -> int | None:
    if not str(value).strip():
        return None
    return int(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--priority", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    observations = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(observations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if observations["smiles"].eq("").any():
        raise ValueError("Unexpected empty raw SMILES: retain and investigate upstream")

    # A raw SMILES must always map to the same source and RDKit key in this QA.
    check_columns = ["inchi_key", "rdkit_inchi_key", "rdkit_status", "source_key_comparison"]
    conflicts = observations.groupby("smiles", sort=False)[check_columns].nunique(dropna=False)
    if conflicts.gt(1).any(axis=None):
        raise ValueError("Same raw SMILES has conflicting identity fields across rows")

    unique = observations.drop_duplicates("smiles").copy()
    source_groups = observations.groupby("smiles", sort=False)
    unique["observations_for_raw_smiles"] = unique["smiles"].map(source_groups.size())
    unique["targets_reported"] = unique["smiles"].map(source_groups["target"].agg(clean_join))
    unique["sources_reported"] = unique["smiles"].map(source_groups["source"].agg(clean_join))
    unique["m17_partitions"] = unique["smiles"].map(source_groups["m17_partition"].agg(clean_join))
    unique["identity_review_class"] = unique.apply(
        lambda r: classify_identity(r.rdkit_status, r.inchi_key, r.rdkit_inchi_key),
        axis=1,
    )
    unique["review_priority"] = unique.apply(
        lambda r: review_priority(
            r.identity_review_class, integer_or_none(r.rdkit_fragment_count)
        ),
        axis=1,
    )
    unique["contains_extended_smiles_annotation"] = unique["smiles"].str.contains(
        " |", regex=False
    )

    # Sanity-check classes against the precomputed M1.8a labels.
    mapping = {
        "exact_match": "exact_match",
        "source_unspecified_stereochemical_block": "same_connectivity_block_only",
        "same_connectivity_other_inchi_layer_difference": "same_connectivity_block_only",
        "different_connectivity_block": "different_connectivity_block",
        "rdkit_unavailable": "rdkit_key_unavailable",
    }
    inferred = unique.identity_review_class.map(mapping)
    if (inferred != unique.source_key_comparison).any():
        raise ValueError("M1.8a comparison labels disagree with M1.8b recalculation")

    # An InChIKey is not guaranteed to identify one canonical isomeric SMILES.
    valid = unique[unique.rdkit_inchi_key.ne("")]
    key_to_canonical_count = valid.groupby("rdkit_inchi_key")[
        "rdkit_canonical_isomeric_smiles"
    ].nunique()
    nonunique_key_count = int((key_to_canonical_count > 1).sum())
    reported_to_canonical_count = unique.groupby("inchi_key")[
        "rdkit_canonical_isomeric_smiles"
    ].nunique()
    reported_nonunique_key_count = int((reported_to_canonical_count > 1).sum())

    cols = [
        "reactant_set_id", "smiles", "target", "targets_reported", "source",
        "sources_reported", "m17_partition", "m17_partitions",
        "observations_for_raw_smiles", "inchi_key", "rdkit_inchi_key",
        "rdkit_canonical_isomeric_smiles", "rdkit_status", "rdkit_fragment_count",
        "source_key_comparison", "identity_review_class", "review_priority",
        "contains_extended_smiles_annotation",
    ]
    review = unique.loc[:, cols].sort_values(
        ["review_priority", "identity_review_class", "reactant_set_id"]
    )
    priority = review.loc[review.review_priority.eq("high")]
    for output in (args.review, args.priority, args.summary):
        output.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.review, index=False)
    priority.to_csv(args.priority, index=False)

    by_source = []
    for source, block in observations.groupby("source", sort=True):
        source_unique = block.drop_duplicates("smiles").copy()
        source_unique["review_class"] = source_unique.apply(
            lambda r: classify_identity(r.rdkit_status, r.inchi_key, r.rdkit_inchi_key),
            axis=1,
        )
        by_source.append({
            "source": source,
            "observations": len(block),
            "unique_smiles_within_source": len(source_unique),
            "different_connectivity_unique_smiles": int(
                source_unique.review_class.eq("different_connectivity_block").sum()
            ),
            "rdkit_unavailable_unique_smiles": int(
                source_unique.review_class.eq("rdkit_unavailable").sum()
            ),
        })

    class_counts = Counter(unique.identity_review_class)
    summary = {
        "stage": "M1.8b_structure_identity_discrepancy_triage_NOT_curation",
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_name": args.input.name,
        "input_sha256": sha256(args.input),
        "observations_retained": len(observations),
        "unique_raw_smiles": len(unique),
        "identity_class_counts_unique_smiles": dict(sorted(class_counts.items())),
        "review_priority_counts_unique_smiles": {
            str(k): int(v) for k, v in unique.review_priority.value_counts().items()
        },
        "high_priority_distinct_smiles": len(priority),
        "high_priority_observation_count": int(
            unique.loc[unique.review_priority.eq("high"), "observations_for_raw_smiles"].sum()
        ),
        "multi_fragment_distinct_smiles": int(
            pd.to_numeric(unique.rdkit_fragment_count, errors="coerce").gt(1).sum()
        ),
        "raw_smiles_with_multiple_source_keys": int(
            (observations.groupby("smiles").inchi_key.nunique() > 1).sum()
        ),
        "reported_keys_with_multiple_canonical_smiles": reported_nonunique_key_count,
        "rdkit_keys_with_multiple_canonical_smiles": nonunique_key_count,
        "by_source": by_source,
        "outputs": {
            "structure_review_csv": args.review.name,
            "high_priority_csv": args.priority.name,
        },
        "limitations": [
            "All counts refer to unique raw SMILES unless explicitly labeled observations.",
            "Source key missing stereo block is a review clue, not proven stereo root cause.",
            "Different connectivity InChI blocks flag disagreement, not necessarily incorrect molecules.",
            "No stereoisomer collapse, salt stripping, tautomer normalization, or IC50 aggregation.",
            "This input omits experimental IC50 and study IDs; assay reconciliation requires upstream evidence.",
            "No molecule split or model training performed.",
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(observations):,} observations / {len(unique):,} distinct SMILES")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count:,}")
    print(f"High-priority review: {len(priority):,} structures / {summary['high_priority_observation_count']:,} observations")
    print(f"Nonunique canonical SMILES per RDKit key: {nonunique_key_count}")
    print(f"Summary: {args.summary}")
    print(f"Manual review: {args.priority}")


if __name__ == "__main__":
    main()
