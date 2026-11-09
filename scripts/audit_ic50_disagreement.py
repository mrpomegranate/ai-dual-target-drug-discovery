"""M1.9a: audit repeat IC50 disagreement; never aggregate training labels.

A provenance key is a publication/patent grouping, NOT an assay ID or laboratory ID.
"""

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")

REQUIRED = {
    "reactant_set_id", "target", "target_name", "source", "provenance_key",
    "m17_partition", "m18c_status", "m18c_structure_id", "measurement_status",
    "ic50_relation", "ic50_nm", "pic50", "activity_at_1000_nm",
}
THRESHOLD_NM = 1000.0


def verify_input(df: pd.DataFrame) -> None:
    """Fail closed if the upstream extraction has an unexpected schema."""
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["reactant_set_id"].isna().any() or df["reactant_set_id"].duplicated().any():
        raise ValueError("reactant_set_id must be non-null and unique")
    if not df["target"].isin(["EGFR", "HER2"]).all():
        raise ValueError("Unexpected target label")
    if not df["m17_partition"].isin([
        "primary_exact_candidate", "primary_classification_candidate",
        "primary_hold_measurement", "secondary_construct_reserve",
    ]).all():
        raise ValueError("Unexpected M1.7 partition")


def prepare_primary_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the provisional, primary, exact regression observations."""
    verify_input(df)
    keep = (
        (df["m17_partition"] == "primary_exact_candidate")
        & (df["m18c_status"] == "provisional_exact_structure")
    )
    out = df.loc[keep].copy()
    if out.empty:
        raise ValueError("No eligible primary exact observations")
    for field in ("m18c_structure_id", "provenance_key", "target_name"):
        if out[field].isna().any() or out[field].astype(str).str.strip().eq("").any():
            raise ValueError(f"Selected observations have missing {field}")
    if not out["measurement_status"].eq("exact").all():
        raise ValueError("Non-exact measurement in primary exact pool")
    if not out["ic50_relation"].eq("=").all():
        raise ValueError("Non-exact IC50 qualifier in primary exact pool")
    out["pic50"] = pd.to_numeric(out["pic50"], errors="raise")
    out["ic50_nm"] = pd.to_numeric(out["ic50_nm"], errors="raise")
    if not out["pic50"].map(math.isfinite).all():
        raise ValueError("Nonfinite pIC50 in primary exact pool")
    if not out["ic50_nm"].map(lambda x: math.isfinite(x) and x > 0).all():
        raise ValueError("IC50 must be positive and finite")
    checked_pic50 = 9.0 - out["ic50_nm"].map(math.log10)
    if (checked_pic50 - out["pic50"]).abs().max() > 1e-6:
        raise ValueError("pIC50 values do not match IC50 in nM")
    actual_activity = out["ic50_nm"].lt(THRESHOLD_NM).map({True: "active", False: "inactive"})
    if not (actual_activity == out["activity_at_1000_nm"]).all():
        raise ValueError("Activity labels inconsistent with strict 1000 nM cutoff")
    return out


def build_audit(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Compute observational disagreement; no averaging into final training labels."""
    x = prepare_primary_exact(df)
    keys = ["target", "m18c_structure_id"]
    # Study-level aggregates are purely diagnostic; repeated rows might be
    # independent assays, source duplication, or true replicates.
    skeys = [*keys, "provenance_key"]
    study = x.groupby(skeys, sort=True, dropna=False).agg(
        n_observations=("pic50", "size"),
        median_pic50=("pic50", "median"),
        min_pic50=("pic50", "min"),
        max_pic50=("pic50", "max"),
        n_sources=("source", "nunique"),
        n_exact_duplicate_values=("pic50", lambda s: len(s) - s.nunique()),
    ).reset_index()
    study["within_study_range_pic50"] = study["max_pic50"] - study["min_pic50"]

    mol = x.groupby(keys, sort=True, dropna=False).agg(
        n_observations=("pic50", "size"),
        n_provenance_groups=("provenance_key", "nunique"),
        n_source_categories=("source", "nunique"),
        n_target_name_strings=("target_name", "nunique"),
        min_pic50=("pic50", "min"),
        max_pic50=("pic50", "max"),
        median_observation_pic50=("pic50", "median"),
        minimum_ic50_nm=("ic50_nm", "min"),
        maximum_ic50_nm=("ic50_nm", "max"),
        n_active=("activity_at_1000_nm", lambda s: (s == "active").sum()),
        n_inactive=("activity_at_1000_nm", lambda s: (s == "inactive").sum()),
    ).reset_index()
    mol["range_pic50"] = mol["max_pic50"] - mol["min_pic50"]
    mol["fold_ic50_range"] = 10.0 ** mol["range_pic50"].clip(upper=300)
    mol["crosses_activity_cutoff"] = (mol["n_active"] > 0) & (mol["n_inactive"] > 0)
    mol["multiple_observations"] = mol["n_observations"] > 1
    mol["multiple_provenances"] = mol["n_provenance_groups"] > 1

    # Study medians within molecule-target; flags cross-study variation without
    # claiming study medians are chemically/assay comparable.
    between = study.groupby(keys).agg(
        n_study_groups=("median_pic50", "size"),
        min_study_median=("median_pic50", "min"),
        max_study_median=("median_pic50", "max"),
        study_groups_within_range_gt1=("within_study_range_pic50", lambda s: (s > 1.0).sum()),
    ).reset_index()
    between["study_median_range_pic50"] = (
        between["max_study_median"] - between["min_study_median"]
    )
    mol = mol.merge(between, on=keys, how="left", validate="one_to_one")

    # Possible duplicate reporting: equal exact values from same structure,
    # target and study, not proof the underlying experiment was duplicated.
    signature = x.groupby([*skeys, "ic50_nm"], dropna=False).size()
    repeated_signatures = signature[signature > 1]

    # Compare source categories using matched molecule-targets only. Source
    # composition and within-source study differences can still confound.
    source_med = x.groupby([*keys, "source"], dropna=False)["pic50"].median().reset_index()
    source_spread = source_med.groupby(keys)["pic50"].agg(["size", "min", "max"]).reset_index()
    source_spread["source_median_range_pic50"] = source_spread["max"] - source_spread["min"]
    cross_source = source_spread[source_spread["size"] > 1]

    per_target = []
    for target in ("EGFR", "HER2"):
        subset = mol[mol["target"] == target]
        obs = x[x["target"] == target]
        repeated = subset[subset["multiple_observations"]]
        multi_study = subset[subset["multiple_provenances"]]
        per_target.append({
            "target": target,
            "observations": len(obs),
            "distinct_molecule_target_pairs": len(subset),
            "repeated_pairs": len(repeated),
            "observations_in_repeated_pairs": int(repeated["n_observations"].sum()),
            "repeated_pairs_range_gt_0_3_pic50": int((repeated["range_pic50"] > 0.3).sum()),
            "repeated_pairs_range_gt_1_pic50": int((repeated["range_pic50"] > 1.0).sum()),
            "repeated_pairs_crossing_activity_threshold": int(repeated["crosses_activity_cutoff"].sum()),
            "pairs_with_multiple_provenance_groups": len(multi_study),
            "multi_provenance_pairs_with_range_gt_1_pic50": int((multi_study["range_pic50"] > 1.0).sum()),
            "multi_provenance_pairs_with_study_median_range_gt_1": int((multi_study["study_median_range_pic50"] > 1.0).sum()),
        })

    duplicate_full = x.duplicated(subset=[
        "target", "m18c_structure_id", "provenance_key", "ic50_nm",
        "source", "target_name", "pubchem_aid", "ph", "temperature_c",
    ], keep=False) if {"pubchem_aid", "ph", "temperature_c"}.issubset(x.columns) else pd.Series(False, index=x.index)

    summary = {
        "stage": "M1.9a_primary_exact_ic50_disagreement_diagnostics_NOT_label_aggregation",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "cutoff_nm_strict_less_than": THRESHOLD_NM,
        "population": {
            "input_observations": len(df),
            "primary_exact_with_provisional_structure_id": len(x),
            "not_in_primary_exact_audit": int(len(df) - len(x)),
            "distinct_molecule_target_pairs": len(mol),
            "distinct_provenance_keys_in_primary_exact": int(x["provenance_key"].nunique()),
        },
        "per_target": per_target,
        "repeated_pairs_total": int(mol["multiple_observations"].sum()),
        "observations_in_repeated_pairs": int(mol.loc[mol["multiple_observations"], "n_observations"].sum()),
        "repeated_pairs_range_gt_1_pic50": int((mol["multiple_observations"] & (mol["range_pic50"] > 1.0)).sum()),
        "pairs_with_multiple_provenance_groups": int(mol["multiple_provenances"].sum()),
        "exact_value_signatures_repeated_within_same_provenance": len(repeated_signatures),
        "extra_rows_in_repeated_value_signatures": int((repeated_signatures - 1).sum()),
        "rows_matching_full_context_duplicate_signature": int(duplicate_full.sum()),
        "study_groups": len(study),
        "study_groups_with_repeated_observations": int((study["n_observations"] > 1).sum()),
        "study_groups_with_within_group_range_gt_1": int((study["within_study_range_pic50"] > 1.0).sum()),
        "molecule_target_pairs_in_multiple_source_categories": len(cross_source),
        "matched_molecule_target_pairs_with_source_median_range_gt_1": int((cross_source["source_median_range_pic50"] > 1.0).sum()),
        "limitations": [
            "Study groups use upstream publication/patent provenance_key, NOT verified assay, laboratory, or independent experimental replicates.",
            "Exact IC50 values only; censored measurements and annotated constructs are out of scope for this first disagreement audit.",
            "Repeated identical entries are possible duplicate reporting, NOT confirmed duplicate experiments.",
            "Same nominal target and plain target name do NOT confirm identical protein sequence, assay format, pH, ATP concentration or protocol.",
            "Unadjusted pIC50 ranges reflect chemistry, assay heterogeneity, reporting issues, and outliers; they do not demonstrate source causality.",
            "Study medians are descriptive diagnostics, NOT curated regression labels. No observations are merged or removed.",
        ],
    }
    return summary, mol, study


def save_figure(mol: pd.DataFrame, study: pd.DataFrame, path: Path) -> None:
    """Save a compact descriptive audit chart; no fitted model results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    cats = ("EGFR", "HER2")
    at_least_two = [mol[(mol["target"] == t) & mol["multiple_observations"]] for t in cats]
    counts = {
        "Repeated pairs": [len(x) for x in at_least_two],
        ">3-fold IC50 range": [int((x["range_pic50"] > math.log10(3)).sum()) for x in at_least_two],
        ">10-fold IC50 range": [int((x["range_pic50"] > 1).sum()) for x in at_least_two],
    }
    pos = range(2)
    width = 0.24
    for i, (label, values) in enumerate(counts.items()):
        axs[0].bar([p + (i-1)*width for p in pos], values, width=width, label=label)
    axs[0].set_xticks(list(pos), cats)
    axs[0].set_ylabel("Distinct molecule–target pairs")
    axs[0].set_title("Repeated measurement disagreement")
    axs[0].legend(fontsize=8)

    for t in cats:
        spread = mol.loc[(mol["target"] == t) & mol["multiple_observations"], "range_pic50"]
        axs[1].hist(spread.clip(upper=4), bins=25, alpha=0.60, label=t)
    axs[1].axvline(1, color="black", linewidth=1.1, linestyle="--", label="10-fold difference")
    axs[1].set_xlabel("Within-pair pIC50 range (values above 4 clipped)")
    axs[1].set_ylabel("Distinct molecule–target pairs")
    axs[1].set_title("Spread of repeated exact measurements")
    axs[1].legend(fontsize=8)
    fig.suptitle("M1.9a · Descriptive IC50 disagreement (not model performance)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--molecule-review", required=True, type=Path)
    parser.add_argument("--study-review", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.input, dtype=str, keep_default_na=False, low_memory=False)
    summary, mol, study = build_audit(data)
    summary["input_file"] = args.input.name
    summary["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    for path in (args.summary, args.molecule_review, args.study_review):
        path.parent.mkdir(parents=True, exist_ok=True)
    mol.to_csv(args.molecule_review, index=False)
    study.to_csv(args.study_review, index=False)
    save_figure(mol, study, args.figure)
    summary["outputs"] = {
        "molecule_review": str(args.molecule_review),
        "study_review": str(args.study_review),
        "figure": str(args.figure),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Input observations: {summary['population']['input_observations']:,}")
    print(f"Eligible exact observations: {summary['population']['primary_exact_with_provisional_structure_id']:,}")
    print(f"Distinct molecule–target pairs: {summary['population']['distinct_molecule_target_pairs']:,}")
    print(f"Repeated pairs: {summary['repeated_pairs_total']:,}")
    print(f"Repeated pairs with >10-fold IC50 spread: {summary['repeated_pairs_range_gt_1_pic50']:,}")
    print(f"Pairs with multiple provenance groups: {summary['pairs_with_multiple_provenance_groups']:,}")
    print(f"Summary: {args.summary}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
