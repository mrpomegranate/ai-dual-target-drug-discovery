"""M1.9b: prioritize IC50 disagreements without modifying any measurement.

Inputs: M1.9a molecule-target and molecule-target-provenance summary CSVs.
Provenance keys are not laboratory or assay IDs. No labels are merged.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

KEY = ["target", "m18c_structure_id"]
STUDY_KEY = [*KEY, "provenance_key"]
MOL_REQUIRED = {
    *KEY,
    "n_observations",
    "n_provenance_groups",
    "n_source_categories",
    "min_pic50",
    "max_pic50",
    "range_pic50",
    "minimum_ic50_nm",
    "maximum_ic50_nm",
    "n_active",
    "n_inactive",
    "crosses_activity_cutoff",
    "n_study_groups",
    "study_groups_within_range_gt1",
    "study_median_range_pic50",
}
STUDY_REQUIRED = {
    *STUDY_KEY,
    "n_observations",
    "median_pic50",
    "min_pic50",
    "max_pic50",
    "within_study_range_pic50",
    "n_exact_duplicate_values",
}
TIER_ORDER = [
    "T0_crossing_gt10fold",
    "T1_crossing_up_to_10fold",
    "T2_gt10fold_non_crossing",
    "T3_other_repeated",
]


def sha256_file(path: Path) -> str:
    """Checksum a file without loading all bytes into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(mol: pd.DataFrame, study: pd.DataFrame) -> None:
    """Require matching molecule/target/study aggregates before triage."""
    for name, frame, required in (
        ("molecule", mol, MOL_REQUIRED),
        ("study", study, STUDY_REQUIRED),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} CSV is empty")
        if not frame["target"].isin(["EGFR", "HER2"]).all():
            raise ValueError(f"Unexpected {name} target label")
        if frame[list(KEY)].isna().any().any():
            raise ValueError(f"Missing {name} molecular key")
    if mol.duplicated(KEY).any():
        raise ValueError("Duplicate molecule-target keys in molecule CSV")
    if study.duplicated(STUDY_KEY).any():
        raise ValueError("Duplicate molecule-target-provenance keys in study CSV")
    if study["provenance_key"].isna().any():
        raise ValueError("Missing provenance identifier")
    for frame in (mol, study):
        if (frame["n_observations"] < 1).any():
            raise ValueError("Observation counts must be positive")
        if (frame["min_pic50"] > frame["max_pic50"]).any():
            raise ValueError("pIC50 minimum exceeds maximum")
    if (mol["n_active"] + mol["n_inactive"] != mol["n_observations"]).any():
        raise ValueError("Molecule activity counts do not reconcile")
    if (mol["n_study_groups"] != mol["n_provenance_groups"]).any():
        raise ValueError("Provenance-group metadata is inconsistent")
    if ((mol["max_pic50"] - mol["min_pic50"] - mol["range_pic50"]).abs() > 1e-6).any():
        raise ValueError("Molecule pIC50 ranges are inconsistent")
    if ((study["max_pic50"] - study["min_pic50"] - study["within_study_range_pic50"]).abs() > 1e-6).any():
        raise ValueError("Study pIC50 ranges are inconsistent")
    expected_cross = (mol["n_active"] > 0) & (mol["n_inactive"] > 0)
    if not (expected_cross == mol["crosses_activity_cutoff"]).all():
        raise ValueError("Activity-threshold crossing flags are inconsistent")
    grouped = study.groupby(KEY, dropna=False).agg(
        study_rows=("provenance_key", "size"),
        study_observations=("n_observations", "sum"),
        study_min=("median_pic50", "min"),
        study_max=("median_pic50", "max"),
        within_gt1=("within_study_range_pic50", lambda x: int((x > 1.0).sum())),
    ).reset_index()
    joined = mol.merge(grouped, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or len(joined) != len(grouped):
        raise ValueError("Molecule/study keys do not reconcile")
    if (joined["n_observations"] != joined["study_observations"]).any():
        raise ValueError("Molecule/study observation totals differ")
    if (joined["n_provenance_groups"] != joined["study_rows"]).any():
        raise ValueError("Molecule/study provenance-group counts differ")
    if (joined["study_groups_within_range_gt1"] != joined["within_gt1"]).any():
        raise ValueError("Within-study >10-fold counts differ")
    expected_spread = joined["study_max"] - joined["study_min"]
    if ((expected_spread - joined["study_median_range_pic50"]).abs() > 1e-6).any():
        raise ValueError("Between-study median ranges differ")


def classify_case(row: pd.Series) -> str:
    """Tier an observation-disagreement pattern, not an assay's correctness."""
    if row["n_observations"] < 2:
        return "not_repeated"
    crossing = bool(row["crosses_activity_cutoff"])
    severe = row["range_pic50"] > 1.0
    if crossing and severe:
        return TIER_ORDER[0]
    if crossing:
        return TIER_ORDER[1]
    if severe:
        return TIER_ORDER[2]
    return TIER_ORDER[3]


def triage(mol: pd.DataFrame, study: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Prioritize cases, retain provenance group evidence, and summarize."""
    validate(mol, study)
    repeated = mol.loc[mol["n_observations"] > 1].copy()
    repeated["triage_tier"] = repeated.apply(classify_case, axis=1)
    repeated["tier_order"] = repeated["triage_tier"].map({v: i for i, v in enumerate(TIER_ORDER)})
    repeated["within_provenance_gt10fold"] = repeated["study_groups_within_range_gt1"] > 0
    repeated["between_provenance_median_gt10fold"] = (
        (repeated["n_provenance_groups"] > 1)
        & (repeated["study_median_range_pic50"] > 1.0)
    )
    repeated["extreme_gt10000fold"] = repeated["range_pic50"] > 4.0
    repeated = repeated.sort_values(
        ["tier_order", "range_pic50", "n_provenance_groups", "target", "m18c_structure_id"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    repeated.insert(0, "review_rank", range(1, len(repeated) + 1))
    # Only the highest-priority tiers receive a detailed publication/patent queue.
    priority = repeated.loc[repeated["triage_tier"] != TIER_ORDER[3]].copy()
    study_cases = study.merge(
        priority[[*KEY, "triage_tier", "review_rank", "range_pic50", "crosses_activity_cutoff"]],
        on=KEY,
        how="inner",
        validate="many_to_one",
    ).sort_values(["review_rank", "median_pic50", "provenance_key"])
    primary = {
        "input_molecule_target_pairs": len(mol),
        "input_provenance_groups": len(study),
        "input_exact_ic50_observations": int(mol["n_observations"].sum()),
        "repeated_pairs": len(repeated),
        "priority_pairs": len(priority),
        "priority_provenance_groups": len(study_cases),
        "threshold_crossing_pairs": int(repeated["crosses_activity_cutoff"].sum()),
        "repeated_pairs_gt10fold": int((repeated["range_pic50"] > 1.0).sum()),
        "repeated_pairs_gt10000fold": int(repeated["extreme_gt10000fold"].sum()),
        "repeated_pairs_gt10fold_one_provenance": int(((repeated["range_pic50"] > 1.0) & (repeated["n_provenance_groups"] == 1)).sum()),
        "repeated_pairs_gt10fold_multi_provenance": int(((repeated["range_pic50"] > 1.0) & (repeated["n_provenance_groups"] > 1)).sum()),
        "pairs_within_any_provenance_gt10fold": int(repeated["within_provenance_gt10fold"].sum()),
        "pairs_between_provenance_medians_gt10fold": int(repeated["between_provenance_median_gt10fold"].sum()),
        "pairs_with_both_within_and_between_gt10fold": int((repeated["within_provenance_gt10fold"] & repeated["between_provenance_median_gt10fold"]).sum()),
        "tiers": {t: int((repeated["triage_tier"] == t).sum()) for t in TIER_ORDER},
        "tiers_by_target": {
            t: {tier: int(((repeated["target"] == t) & (repeated["triage_tier"] == tier)).sum()) for tier in TIER_ORDER}
            for t in ("EGFR", "HER2")
        },
        "top_10_cases": [],
    }
    for row in repeated.head(10).itertuples(index=False):
        primary["top_10_cases"].append({
            "review_rank": int(row.review_rank),
            "target": row.target,
            "m18c_structure_id": row.m18c_structure_id,
            "tier": row.triage_tier,
            "n_observations": int(row.n_observations),
            "n_provenance_groups": int(row.n_provenance_groups),
            "minimum_ic50_nm": float(row.minimum_ic50_nm),
            "maximum_ic50_nm": float(row.maximum_ic50_nm),
            "range_pic50": float(row.range_pic50),
        })
    return repeated, study_cases, primary


def save_figure(repeated: pd.DataFrame, path: Path) -> None:
    """Save descriptive priority counts, not source-attributed causal effects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    targets = ["EGFR", "HER2"]
    bottom = [0, 0]
    for tier in TIER_ORDER:
        counts = [int(((repeated["target"] == t) & (repeated["triage_tier"] == tier)).sum()) for t in targets]
        ax.bar(targets, counts, bottom=bottom, label=tier.replace("_", " "))
        bottom = [b + c for b, c in zip(bottom, counts)]
    for x, n in zip(targets, bottom):
        ax.text(x, n + 35, f"{n:,}", ha="center")
    ax.set_ylim(0, max(bottom) * 1.14)
    ax.set_ylabel("Repeated molecule–target pairs")
    ax.set_title("M1.9b · Prioritize reported IC50 disagreement (not model error)")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule-review", required=True, type=Path)
    parser.add_argument("--study-review", required=True, type=Path)
    parser.add_argument("--priority-cases", required=True, type=Path)
    parser.add_argument("--priority-studies", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    m = pd.read_csv(args.molecule_review, low_memory=False)
    s = pd.read_csv(args.study_review, low_memory=False)
    cases, provenance, info = triage(m, s)
    report = {
        "stage": "M1.9b_descriptive_disagreement_triage_NOT_curation_or_label_aggregation",
        "generated_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "molecule_file": args.molecule_review.name,
            "molecule_sha256": sha256_file(args.molecule_review),
            "study_file": args.study_review.name,
            "study_sha256": sha256_file(args.study_review),
        },
        "analysis": info,
        "interpretation_limits": [
            "M1.9a aggregate input only; individual reactant_set_id and assay conditions are not present here.",
            "A provenance_key groups DOI/PMID/patent records, not verified assays, labs or independent replicates.",
            "A within-group range is not proof of within-laboratory bias.",
            "Observed extrema and medians can be influenced by outliers, duplicated reporting and unmeasured assay differences.",
            "Construct-reference status is provisional; no target sequence verification has occurred.",
            "No chemical structures, IC50 values, source observations or labels are modified or aggregated.",
        ],
        "outputs": {
            "review_cases": str(args.priority_cases),
            "review_provenance_groups": str(args.priority_studies),
            "figure": str(args.figure),
        },
    }
    for path in (args.priority_cases, args.priority_studies, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.priority_cases, index=False)
    provenance.to_csv(args.priority_studies, index=False)
    save_figure(cases, args.figure)
    args.summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Pairs audited: {info['input_molecule_target_pairs']:,}")
    print(f"Repeated pairs: {info['repeated_pairs']:,}")
    print(f"Priority pairs (threshold contradiction or >10-fold spread): {info['priority_pairs']:,}")
    print(f"  Crossing 1000 nM cutoff: {info['threshold_crossing_pairs']:,}")
    print(f"  >10-fold spread: {info['repeated_pairs_gt10fold']:,}")
    print(f"Priority provenance groups: {info['priority_provenance_groups']:,}")
    print(f"Summary: {args.summary}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
