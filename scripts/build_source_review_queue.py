"""Build a non-destructive, manual source-review queue from M1.9c exports.

This script DOES NOT infer the assay type, correct measurements or merge labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

KEYS = ["target", "m18c_structure_id", "provenance_key"]
REQUIRED_EVIDENCE = KEYS + [
    "reactant_set_id", "review_rank", "ic50_nm", "pic50", "source",
    "m17_partition", "m18c_status", "article_doi", "pmid",
    "patent_number", "pubchem_aid", "ph", "temperature_c",
    "candidate_duplicate_signature", "target_name",
]
REQUIRED_GROUP = KEYS + [
    "n_observations", "min_ic50_nm", "max_ic50_nm",
    "within_provenance_range_pic50", "candidate_duplicate_rows",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_value(s: pd.Series) -> pd.Series:
    return s.notna() & s.astype("string").str.strip().ne("")


def check_and_build(evidence: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reconcile M1.9c source groups; return a queue, not curation decisions."""
    for name, frame, required in [
        ("evidence", evidence, REQUIRED_EVIDENCE),
        ("groups", groups, REQUIRED_GROUP),
    ]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    if evidence["reactant_set_id"].duplicated().any():
        raise ValueError("Duplicate reactant_set_id: individual observations not unique")
    if not (evidence["m17_partition"] == "primary_exact_candidate").all():
        raise ValueError("Selected evidence contains non-primary-exact rows")
    if not (evidence["m18c_status"] == "provisional_exact_structure").all():
        raise ValueError("Selected evidence contains nonapproved structure identity rows")
    if evidence[KEYS].isna().any().any() or groups[KEYS].isna().any().any():
        raise ValueError("Missing target, structure ID or provenance key")
    if groups.duplicated(subset=KEYS).any():
        raise ValueError("Provenance groups are not unique")
    if (evidence["ic50_nm"] <= 0).any() or evidence["ic50_nm"].isna().any():
        raise ValueError("IC50 must be exact and positive")
    if evidence["pic50"].isna().any():
        raise ValueError("Missing exact pIC50")

    ag = evidence.groupby(KEYS, dropna=False).agg(
        rows=("reactant_set_id", "size"),
        minimum_ic50=("ic50_nm", "min"),
        maximum_ic50=("ic50_nm", "max"),
        recomputed_min_pic50=("pic50", "min"),
        recomputed_max_pic50=("pic50", "max"),
        distinct_ic50=("ic50_nm", "nunique"),
        sources=("source", "nunique"),
        candidate_duplicate_rows_calc=("candidate_duplicate_signature", "sum"),
    ).reset_index()
    joined = groups.merge(ag, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("M1.9c evidence and provenance groups do not have identical group keys")
    for a, b in [
        ("n_observations", "rows"),
        ("min_ic50_nm", "minimum_ic50"),
        ("max_ic50_nm", "maximum_ic50"),
        ("candidate_duplicate_rows", "candidate_duplicate_rows_calc"),
    ]:
        if not pd.Series((joined[a] - joined[b]).abs() < 1e-7).all():
            raise ValueError(f"Group field mismatch: {a} vs {b}")
    spread = joined["recomputed_max_pic50"] - joined["recomputed_min_pic50"]
    if not ((spread - joined["within_provenance_range_pic50"]).abs() < 1e-5).all():
        raise ValueError("Group pIC50 spreads differ from underlying observations")

    # Populate review hints from source rows. These are NOT validations.
    meta = evidence.groupby(KEYS, dropna=False).agg(
        article_doi_candidates=("article_doi", lambda s: "; ".join(sorted(set(s.dropna().astype(str).str.strip()) - {""}))[:240]),
        pmid_candidates=("pmid", lambda s: "; ".join(sorted(set(s.dropna().astype(str).str.strip().str.replace(r"\.0$", "", regex=True)) - {""}))[:160]),
        patent_candidates=("patent_number", lambda s: "; ".join(sorted(set(s.dropna().astype(str).str.strip()) - {""}))[:180]),
        has_pubchem_aid=("pubchem_aid", lambda s: bool(has_value(s).any())),
        has_ph=("ph", lambda s: bool(has_value(s).any())),
        has_temperature=("temperature_c", lambda s: bool(has_value(s).any())),
        source_categories=("source", lambda s: "; ".join(sorted(s.astype(str).unique()))),
        n_target_name_strings=("target_name", "nunique"),
        review_rank=("review_rank", "min"),
    ).reset_index()
    joined = joined.merge(meta, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_from_observations"))
    if "review_rank_from_observations" in joined:
        joined["review_rank"] = joined["review_rank_from_observations"]
    joined["within_group_threshold_crossing"] = (joined.minimum_ic50 < 1000) & (joined.maximum_ic50 >= 1000)
    joined["gt10fold_spread"] = spread > 1.0
    joined["source_review_priority"] = "routine"
    joined.loc[joined.gt10fold_spread, "source_review_priority"] = "high"
    joined.loc[joined.within_group_threshold_crossing, "source_review_priority"] = "highest"
    joined["review_status"] = "needs_original_source_review"
    joined["confirmed_assay_endpoint"] = ""
    joined["confirmed_target_construct"] = ""
    joined["source_table_or_supplement"] = ""
    joined["verified_units"] = ""
    joined["possible_duplicate_reporting"] = "unresolved"
    joined["curation_decision"] = "deferred"
    joined["reviewer_notes"] = ""
    joined["n_distinct_ic50_values"] = joined.distinct_ic50
    joined["observed_pic50_range"] = spread
    ordered_cols = [
        "target", "review_rank", "m18c_structure_id", "provenance_key",
        "source_categories", "n_observations", "n_distinct_ic50_values",
        "min_ic50_nm", "max_ic50_nm", "observed_pic50_range",
        "within_group_threshold_crossing", "gt10fold_spread",
        "candidate_duplicate_rows", "has_pubchem_aid", "has_ph", "has_temperature",
        "article_doi_candidates", "pmid_candidates", "patent_candidates", "n_target_name_strings",
        "source_review_priority", "review_status", "confirmed_assay_endpoint",
        "confirmed_target_construct", "source_table_or_supplement", "verified_units",
        "possible_duplicate_reporting", "curation_decision", "reviewer_notes",
    ]
    queue = joined[ordered_cols].sort_values(
        by=["source_review_priority", "observed_pic50_range", "target", "review_rank"],
        key=lambda s: s.map({"highest": 0, "high": 1, "routine": 2}) if s.name == "source_review_priority" else (-s if s.name == "observed_pic50_range" else s),
    ).reset_index(drop=True)
    summary = {
        "stage": "M1.9d1_source_review_QUEUE_not_curation",
        "observations": len(evidence),
        "selected_molecule_target_pairs": int(evidence.groupby(["target", "m18c_structure_id"]).ngroups),
        "provenance_groups": len(groups),
        "distinct_reactant_set_ids": int(evidence.reactant_set_id.nunique()),
        "within_provenance_gt10fold": int(queue.gt10fold_spread.sum()),
        "within_provenance_crossing_1000_nm": int(queue.within_group_threshold_crossing.sum()),
        "source_review_priority_counts": {str(k): int(v) for k, v in queue.source_review_priority.value_counts().items()},
        "suspected_duplicate_signature_rows": int(evidence.candidate_duplicate_signature.sum()),
        "unique_target_name_strings": int(evidence.target_name.nunique()),
        "metadata_rows_present": {
            col: int(has_value(evidence[col]).sum())
            for col in ["article_doi", "pmid", "patent_number", "pubchem_aid", "ph", "temperature_c"]
        },
        "source_counts": {str(k): int(v) for k, v in evidence.source.value_counts().items()},
        "limits": [
            "The 20 cases are deliberately selected extremes, not a representative random sample.",
            "Provenance keys are publication/patent identifiers, not assay or laboratory identifiers.",
            "A paper title or abstract mentioning mutants does not prove a specific BindingDB row is mutant.",
            "Source link matching to original tables is not performed automatically.",
            "No IC50 values, chemical identities, potency labels or train/test allocations are changed.",
        ],
    }
    return queue, summary


def plot_coverage(summary: dict, out: Path) -> None:
    labels = ["PMID", "DOI", "Patent", "PubChem AID", "pH", "Temperature"]
    key = ["pmid", "article_doi", "patent_number", "pubchem_aid", "ph", "temperature_c"]
    n = summary["observations"]
    counts = [summary["metadata_rows_present"][k] for k in key]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ys = range(len(labels))
    ax.barh(ys, [100 * x / n for x in counts])
    ax.set_yticks(list(ys), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Percent of selected observations with metadata")
    ax.set_title("M1.9d — Metadata coverage in 20 prioritized cases (n=1,001)")
    for y, val in zip(ys, counts, strict=True):
        ax.text(100 * val / n + 1.0, y, f"{val}/{n}", va="center", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    evidence = pd.read_csv(args.evidence, low_memory=False, dtype={"reactant_set_id": "string", "pmid": "string"})
    groups = pd.read_csv(args.provenance, low_memory=False)
    queue, summary = check_and_build(evidence, groups)
    summary["generated_utc"] = datetime.now(UTC).isoformat()
    summary["inputs"] = {
        "evidence_file": args.evidence.name, "evidence_sha256": sha256(args.evidence),
        "provenance_file": args.provenance.name, "provenance_sha256": sha256(args.provenance),
    }
    summary["outputs"] = {"review_queue": str(args.queue), "figure": str(args.figure)}
    args.queue.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.queue, index=False)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_coverage(summary, args.figure)
    print(f"Validated individual observations: {summary['observations']:,}")
    print(f"Validated provenance groups: {summary['provenance_groups']:,}")
    print(f">10-fold within-group disagreement: {summary['within_provenance_gt10fold']}")
    print(f"Activity threshold crossed within group: {summary['within_provenance_crossing_1000_nm']}")
    print(f"Manual review queue: {args.queue}")
    print(f"Summary: {args.summary}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
