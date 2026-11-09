"""M1.7: Reproducible, provisional EGFR/HER2 construct inclusion decisions.

Does not deduplicate assays, confirm wild-type sequence, create train/test splits,
standardize molecules, or establish curated dual-target training labels.

Requires scripts/audit_construct_provenance.py from M1.6 in the same directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from audit_construct_provenance import load_and_classify

plt.switch_backend("Agg")

REQUIRED_FIELDS = {
    "target", "target_name", "organism", "measurement_status", "smiles",
    "pic50", "ic50_nm", "activity_at_1000_nm", "source", "inchi_key",
    "reactant_set_id",
}
EXPLICIT_CONSTRUCTS = {"substitutions_only", "mixed_construct", "other_construct"}
HUMAN_TERMS = {"human", "homo sapiens"}
DETERMINATE = {"active", "inactive"}
DECISIONS = (
    "primary_exact_candidate",
    "primary_classification_candidate",
    "primary_hold_organism",
    "primary_hold_smiles",
    "primary_hold_measurement",
    "secondary_construct_reserve",
    "manual_construct_review",
)
DISPLAY = {
    "primary_exact_candidate": "Primary: exact IC50",
    "primary_classification_candidate": "Primary: censored classification",
    "primary_hold_organism": "Primary: organism review",
    "primary_hold_smiles": "Primary: structure review",
    "primary_hold_measurement": "Primary: measurement review",
    "secondary_construct_reserve": "Annotated construct reserve",
    "manual_construct_review": "Unrecognized construct review",
}


def norm_human(organism: str) -> bool:
    """Normalize explicit human metadata; never impute missing organisms."""
    return organism.strip().casefold() in HUMAN_TERMS


def is_valid_exact(row: dict) -> bool:
    if str(row.get("measurement_status", "")).strip() != "exact":
        return False
    try:
        ic50 = float(row["ic50_nm"])
        pic50 = float(row["pic50"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (math.isfinite(ic50) and ic50 > 0 and math.isfinite(pic50)):
        return False
    # A large numerical mismatch should not silently pass as a regression label.
    return math.isclose(pic50, 9 - math.log10(ic50), rel_tol=0, abs_tol=1e-5)


def is_classifiable_censored(row: dict) -> bool:
    return (
        str(row.get("measurement_status", "")).strip() == "censored"
        and str(row.get("activity_at_1000_nm", "")).strip() in DETERMINATE
    )


def decide(row: dict) -> tuple[str, str, str]:
    """Return partition, assay role, and reason (mutually exclusive partition)."""
    category = str(row["construct_category"]).strip()
    organism_ok = norm_human(str(row.get("organism", "")))
    smiles_ok = bool(str(row.get("smiles", "")).strip())
    exact = is_valid_exact(row)
    classified_censored = is_classifiable_censored(row)

    if category == "unclassified_nonbase":
        return "manual_construct_review", "not_ready", "unrecognized_target_name"

    if category in EXPLICIT_CONSTRUCTS:
        if not organism_ok:
            return "secondary_construct_reserve", "not_ready", "organism_unconfirmed"
        if not smiles_ok:
            return "secondary_construct_reserve", "not_ready", "missing_smiles"
        if exact:
            return "secondary_construct_reserve", "exact_candidate", "annotated_target_construct"
        if classified_censored:
            return "secondary_construct_reserve", "classification_candidate", "annotated_target_construct"
        return "secondary_construct_reserve", "not_ready", "measurement_not_usable"

    if category != "plain_name":
        raise ValueError(f"Unexpected M1.6 construct category: {category!r}")

    if not organism_ok:
        return "primary_hold_organism", "not_ready", "organism_unconfirmed"
    if not smiles_ok:
        return "primary_hold_smiles", "not_ready", "missing_smiles"
    if exact:
        return "primary_exact_candidate", "exact_candidate", "provisional_plain_target"
    if classified_censored:
        return "primary_classification_candidate", "classification_candidate", "provisional_plain_target"
    return "primary_hold_measurement", "not_ready", "measurement_not_usable"


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare(input_csv: Path) -> pd.DataFrame:
    # Centralize target-name parsing in the already tested M1.6 implementation.
    data = load_and_classify(input_csv)
    missing = sorted(REQUIRED_FIELDS - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use raw metadata, not the possibly stale M1.4 organism_flag in older exports.
    expected_flag = data["organism"].map(
        lambda value: "human" if norm_human(value) else ("missing" if not value.strip() else "nonhuman_or_unexpected")
    )
    data["m17_organism_normalized"] = expected_flag
    if "organism_flag" in data.columns:
        data["m17_original_organism_flag_mismatch"] = data["organism_flag"] != expected_flag
    else:
        data["m17_original_organism_flag_mismatch"] = False

    decisions = [decide(row) for row in data.to_dict("records")]
    data[["m17_partition", "m17_assay_role", "m17_reason"]] = pd.DataFrame(
        decisions, index=data.index
    )
    if data["m17_partition"].isna().any():
        raise AssertionError("Some rows were not assigned a partition")
    return data


def counts_by_target(data: pd.DataFrame) -> dict:
    output: dict[str, dict] = {}
    for target, subset in data.groupby("target", sort=True):
        partition_counts = Counter(subset["m17_partition"])
        reserve = subset.loc[subset.m17_partition == "secondary_construct_reserve"]
        output[target] = {
            "observations": len(subset),
            "partitions": {partition: int(partition_counts[partition]) for partition in DECISIONS},
            "reserve_assay_roles": {
                role: int((reserve.m17_assay_role == role).sum())
                for role in ("exact_candidate", "classification_candidate", "not_ready")
            },
            "organism_metadata": {
                key: int((subset.m17_organism_normalized == key).sum())
                for key in ("human", "missing", "nonhuman_or_unexpected")
            },
        }
        assert sum(partition_counts.values()) == len(subset)
    return output


def summary(data: pd.DataFrame, input_csv: Path) -> dict:
    per_target = counts_by_target(data)
    source_detail = []
    for (target, source), subset in data.groupby(["target", "source"], sort=True):
        source_detail.append({
            "target": target,
            "source": source,
            "observations": len(subset),
            "partitions": {
                partition: int((subset.m17_partition == partition).sum())
                for partition in DECISIONS
            },
        })

    exact = data.loc[data.m17_partition == "primary_exact_candidate"]
    reserve = data.loc[
        (data.m17_partition == "secondary_construct_reserve")
        & (data.m17_assay_role == "exact_candidate")
    ]
    egfr_exact = set(exact.loc[exact.target == "EGFR", "inchi_key"]) - {""}
    her2_exact = set(exact.loc[exact.target == "HER2", "inchi_key"]) - {""}
    overlap_by_target = {}
    for target in ("EGFR", "HER2"):
        primary_keys = set(exact.loc[exact.target == target, "inchi_key"]) - {""}
        reserve_keys = set(reserve.loc[reserve.target == target, "inchi_key"]) - {""}
        overlap_by_target[target] = len(primary_keys & reserve_keys)

    return {
        "stage": "M1.7_provisional_construct_policy_NOT_curated_training_labels",
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_csv_name": input_csv.name,
        "input_sha256": sha256(input_csv),
        "observations": len(data),
        "activity_threshold_nm": 1000,
        "policy": {
            "primary": "plain target name, explicit human metadata and present SMILES; exact IC50 for regression, determinate censored only for classification",
            "secondary": "all recognized name-annotated constructs reserved; separately flag assay/organism eligibility",
            "unknown_names": "manual review; never automatically place in reference or construct challenge set",
            "sequence_verification": "NOT performed; plain names are provisional, NOT confirmed wild-type",
            "no_model_split": "No molecular or scaffold train/test allocation has occurred",
        },
        "total_partitions": {
            category: int((data.m17_partition == category).sum()) for category in DECISIONS
        },
        "per_target": per_target,
        "by_source": source_detail,
        "source_reported_inchikey_diagnostics": {
            "primary_exact_egfr_her2_overlap": len(egfr_exact & her2_exact),
            "primary_exact_vs_secondary_exact_same_target": overlap_by_target,
            "interpretation": "Source-reported structure identifiers only; not standardized, paired, scaffold-split or assay-reconciled",
        },
        "source_organism_flag_disagreements": int(data.m17_original_organism_flag_mismatch.sum()),
        "limitations": [
            "Only target-name annotations were examined, not sequences or protein constructs experimentally verified.",
            "Source-reported InChIKeys are not canonicalized chemical identities.",
            "Secondary reserve includes censored, missing-metadata and otherwise unusable observations; only annotated exact candidates could enter regression after further curation.",
            "Dataset role assignment does not prevent source, chemistry, or scaffold overlap. These checks remain mandatory.",
            "Same-molecule reference-vs-mutant comparisons are a separate paired assay analysis, not scaffold-held-out model evaluation.",
            "Publication/patent/source categories are not laboratory identifiers; causal confounding not established.",
            "No raw observations merged, imputed, relabeled, or used to train a model.",
        ],
    }


def plot_partitions(data: pd.DataFrame, output: Path) -> None:
    grouped = data.groupby(["target", "m17_partition"]).size().unstack(fill_value=0)
    grouped = grouped.reindex(index=["EGFR", "HER2"], columns=DECISIONS, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(grouped))
    left = np.zeros(len(y))
    for partition in DECISIONS:
        vals = grouped[partition].to_numpy()
        ax.barh(y, vals, left=left, label=DISPLAY[partition])
        left += vals
    ax.set_yticks(y, list(grouped.index))
    ax.invert_yaxis()
    ax.set_xlabel("Source experimental observations (not unique molecules)")
    ax.set_title("09  Provisional M1.7 construct-policy decisions", loc="left", weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.01, "Plain target names are not sequence-verified wild-type; reserve is not a finalized challenge test set.", fontsize=9)
    fig.subplots_adjust(left=0.14, bottom=0.32, top=0.85)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_source_shift(data: pd.DataFrame, output: Path) -> None:
    selection = data.loc[
        data.m17_partition.isin({"primary_exact_candidate", "secondary_construct_reserve"})
    ].copy()
    selection["cohort"] = np.where(
        selection.m17_partition == "primary_exact_candidate", "Primary exact", "Annotated reserve (all assays)"
    )
    grouped = selection.groupby(["target", "cohort", "source"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6.3))
    y = np.arange(len(grouped))
    left = np.zeros(len(y))
    for src in sorted(selection.source.unique()):
        vals = grouped[src].to_numpy(dtype=float)
        pct = np.divide(100 * vals, grouped.sum(axis=1).to_numpy(), out=np.zeros_like(vals), where=grouped.sum(axis=1).to_numpy() != 0)
        ax.barh(y, pct, left=left, label=src)
        left += pct
    labels = [f"{target} · {cohort} (n={sum(grouped.loc[(target,cohort)]):,})" for target, cohort in grouped.index]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of observations within cohort (%)")
    ax.set_title("10  Source composition differs between candidate pools", loc="left", weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    fig.text(0.01, 0.01, "Descriptive source categories, not proven lab/site effects. Primary exact and all annotated reserve are different assay subsets.", fontsize=8.5)
    fig.subplots_adjust(left=0.38, bottom=0.24, top=0.88)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run(input_csv: Path, decisions_csv: Path, summary_json: Path, figure_dir: Path) -> dict:
    data = prepare(input_csv)
    result = summary(data, input_csv)
    decisions_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(decisions_csv, index=False)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot_partitions(data, figure_dir / "09_provisional_inclusion_decisions.png")
    plot_source_shift(data, figure_dir / "10_source_composition_after_policy.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/interim/bindingdb_egfr_her2_observations.csv"))
    parser.add_argument("--decisions", type=Path, default=Path("data/interim/M1_7_observation_decisions.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/audits/M1_7_construct_policy_summary.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures/source_bias"))
    args = parser.parse_args()
    result = run(args.input, args.decisions, args.summary, args.figure_dir)
    print(f"Audited: {result['observations']:,} observations")
    for target, counts in result["per_target"].items():
        c = counts["partitions"]
        r = counts["reserve_assay_roles"]
        print(
            f"{target}: primary exact={c['primary_exact_candidate']:,}, "
            f"primary classification={c['primary_classification_candidate']:,}, "
            f"annotated reserve={c['secondary_construct_reserve']:,} "
            f"(exact candidate={r['exact_candidate']:,})"
        )
    print(f"Source organism flag discrepancies: {result['source_organism_flag_disagreements']:,}")
    print(f"Decisions: {args.decisions}")
    print(f"Summary: {args.summary}")
    print(f"Figures: {args.figure_dir / '09_provisional_inclusion_decisions.png'}")
    print(f"         {args.figure_dir / '10_source_composition_after_policy.png'}")


if __name__ == "__main__":
    main()
