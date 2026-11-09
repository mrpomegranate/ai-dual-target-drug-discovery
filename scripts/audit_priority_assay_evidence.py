"""M1.9c: trace priority M1.9b cases to original BindingDB observations.

Conservative, read-only scientific QA: NO IC50 aggregation or lab-effect inference.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY = ["target", "m18c_structure_id"]
META = ["source", "article_doi", "pmid", "patent_number", "pubchem_aid", "ph", "temperature_c"]
REQUIRED_OBS = {
    *KEY, "m17_partition", "m18c_status", "measurement_status", "ic50_relation",
    "ic50_nm", "pic50", "activity_at_1000_nm", "provenance_key", "target_name",
    "reactant_set_id", "ic50_raw", *META,
}
REQUIRED_CASES = {
    *KEY, "review_rank", "triage_tier", "n_observations", "n_provenance_groups",
    "minimum_ic50_nm", "maximum_ic50_nm", "range_pic50", "crosses_activity_cutoff",
}
REVIEW_TIERS = {"T0_crossing_gt10fold", "T1_crossing_up_to_10fold", "T2_gt10fold_non_crossing"}
DUP_SIGNATURE = [*KEY, "provenance_key", "source", "target_name", "ic50_raw", "pubchem_aid", "ph", "temperature_c"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def present(col: pd.Series) -> pd.Series:
    return col.notna() & col.astype(str).str.strip().ne("")


def choose_cases(cases: pd.DataFrame, per_target: int) -> pd.DataFrame:
    """Select priority cases for BOTH EGFR and HER2, even if EGFR dominates ranks."""
    if per_target < 1:
        raise ValueError("per_target must be >= 1")
    missing = REQUIRED_CASES.difference(cases.columns)
    if missing:
        raise ValueError(f"Priority case table missing columns: {sorted(missing)}")
    if cases.duplicated(KEY).any():
        raise ValueError("Duplicate target/structure pairs in priority cases")
    if cases["review_rank"].isna().any() or cases["review_rank"].duplicated().any():
        raise ValueError("Review ranks missing or repeated")
    priority = cases.loc[cases["triage_tier"].isin(REVIEW_TIERS)].copy()
    if priority.empty:
        raise ValueError("No priority cases found")
    selected = (
        priority.sort_values("review_rank", kind="stable")
        .groupby("target", sort=True, group_keys=False)
        .head(per_target)
        .sort_values(["target", "review_rank"], kind="stable")
        .copy()
    )
    return selected


def reconstruct(obs: pd.DataFrame, cases: pd.DataFrame, per_target: int = 10):
    """Join and verify exact observations for selected priority molecule–target pairs."""
    missing = REQUIRED_OBS.difference(obs.columns)
    if missing:
        raise ValueError(f"Observation file missing columns: {sorted(missing)}")
    selected = choose_cases(cases, per_target)
    # Require provenance- and structure-aware joins: do not join by compound name/monomer ID.
    eligible = obs.loc[
        (obs["m17_partition"] == "primary_exact_candidate")
        & (obs["m18c_status"] == "provisional_exact_structure")
        & (obs["measurement_status"] == "exact")
    ]
    evidence = eligible.merge(
        selected[[*KEY, "review_rank", "triage_tier", "n_observations", "n_provenance_groups",
                  "minimum_ic50_nm", "maximum_ic50_nm", "range_pic50", "crosses_activity_cutoff"]],
        on=KEY, how="inner", validate="many_to_one",
    )
    if evidence.empty:
        raise ValueError("No observations matched selected cases")
    if not evidence["m17_partition"].eq("primary_exact_candidate").all():
        raise ValueError("Selected cases unexpectedly include non-primary observations")
    if not evidence["m18c_status"].eq("provisional_exact_structure").all():
        raise ValueError("Selected cases unexpectedly include unapproved molecular identities")
    if not evidence["measurement_status"].eq("exact").all():
        raise ValueError("Selected cases unexpectedly include censored/invalid IC50")
    if not evidence["ic50_relation"].eq("=").all():
        raise ValueError("Selected cases unexpectedly include inequality measurement")
    if not evidence["target"].isin(["EGFR", "HER2"]).all():
        raise ValueError("Unexpected target")
    if evidence["provenance_key"].isna().any():
        raise ValueError("Missing provenance_key in selected evidence")
    if evidence["reactant_set_id"].isna().any():
        raise ValueError("Missing source record ID")
    if evidence["reactant_set_id"].duplicated().any():
        raise ValueError("Duplicate BindingDB reactant_set_id in selected evidence")
    ic50 = pd.to_numeric(evidence["ic50_nm"], errors="coerce")
    pic50 = pd.to_numeric(evidence["pic50"], errors="coerce")
    if (ic50.isna() | ~np.isfinite(ic50) | (ic50 <= 0)).any():
        raise ValueError("Invalid exact IC50 in priority evidence")
    if (pic50.isna() | ~np.isfinite(pic50) | ((9 - np.log10(ic50) - pic50).abs() > 1e-6)).any():
        raise ValueError("IC50/pIC50 discrepancy in priority evidence")
    expected_activity = np.where(ic50 < 1000, "active", "inactive")
    if not (evidence["activity_at_1000_nm"].to_numpy() == expected_activity).all():
        raise ValueError("Activity label contradicts strict IC50 <1000 nM policy")
    evidence = evidence.copy()
    evidence["candidate_duplicate_signature"] = evidence.duplicated(
        DUP_SIGNATURE, keep=False
    )
    group = evidence.groupby(KEY, sort=False)
    counts = group.agg(
        n_rows=("reactant_set_id", "size"),
        n_provenance_groups=("provenance_key", "nunique"),
        min_ic50=("ic50_nm", "min"),
        max_ic50=("ic50_nm", "max"),
        min_pic50=("pic50", "min"),
        max_pic50=("pic50", "max"),
    ).reset_index()
    chk = selected.merge(counts, on=KEY, how="left", validate="one_to_one", suffixes=("_expected", "_actual"), indicator=True)
    if not chk["_merge"].eq("both").all() or len(chk) != len(counts):
        raise ValueError("Some priority cases have no matching observations")
    if not (chk["n_observations"] == chk["n_rows"]).all():
        raise ValueError("Observation counts do not match M1.9b")
    if not (chk["n_provenance_groups_expected"] == chk["n_provenance_groups_actual"]).all():
        raise ValueError("Provenance counts do not match M1.9b")
    if ((chk["minimum_ic50_nm"] - chk["min_ic50"]).abs() > 1e-6).any():
        raise ValueError("Minimum IC50 does not match M1.9b")
    if ((chk["maximum_ic50_nm"] - chk["max_ic50"]).abs() > 1e-6).any():
        raise ValueError("Maximum IC50 does not match M1.9b")
    hi_col = "max_pic50_actual" if "max_pic50_actual" in chk.columns else "max_pic50"
    lo_col = "min_pic50_actual" if "min_pic50_actual" in chk.columns else "min_pic50"
    if (((chk[hi_col] - chk[lo_col]) - chk["range_pic50"]).abs() > 1e-6).any():
        raise ValueError("pIC50 range does not match M1.9b")
    activity_by_pair = evidence.groupby(KEY)["activity_at_1000_nm"].nunique().reset_index(name="n_activity_classes")
    crossing = selected.merge(activity_by_pair, on=KEY, validate="one_to_one")
    if not ((crossing["n_activity_classes"] > 1) == crossing["crosses_activity_cutoff"].astype(bool)).all():
        raise ValueError("Activity threshold-crossing metadata does not match M1.9b")

    evidence = evidence.sort_values(["target", "review_rank", "pic50", "provenance_key", "reactant_set_id"], kind="stable")
    study = evidence.groupby([*KEY, "review_rank", "triage_tier", "provenance_key"], dropna=False, sort=False).agg(
        n_observations=("reactant_set_id", "size"),
        median_pic50=("pic50", "median"),
        min_pic50=("pic50", "min"),
        max_pic50=("pic50", "max"),
        min_ic50_nm=("ic50_nm", "min"),
        max_ic50_nm=("ic50_nm", "max"),
        n_source_categories=("source", "nunique"),
        candidate_duplicate_rows=("candidate_duplicate_signature", "sum"),
        article_doi_present=("article_doi", lambda x: int(present(x).sum())),
        pmid_present=("pmid", lambda x: int(present(x).sum())),
        patent_present=("patent_number", lambda x: int(present(x).sum())),
        assay_id_present=("pubchem_aid", lambda x: int(present(x).sum())),
        ph_present=("ph", lambda x: int(present(x).sum())),
        temperature_present=("temperature_c", lambda x: int(present(x).sum())),
    ).reset_index().sort_values(["target", "review_rank", "median_pic50"], kind="stable")
    study["within_provenance_range_pic50"] = study["max_pic50"] - study["min_pic50"]
    return selected, evidence, study


def build_summary(selected: pd.DataFrame, evidence: pd.DataFrame, study: pd.DataFrame) -> dict:
    result = {
        "stage": "M1.9c_priority_observation_evidence_NOT_lab_causality_or_label_aggregation",
        "selected_pairs": len(selected),
        "selected_pairs_by_target": {target: int((selected["target"] == target).sum()) for target in ("EGFR", "HER2")},
        "exact_observation_rows": len(evidence),
        "provenance_groups": len(study),
        "suspected_duplicate_signature_rows": int(evidence["candidate_duplicate_signature"].sum()),
        "metadata_present_rows": {column: int(present(evidence[column]).sum()) for column in META},
        "source_categories": {str(k): int(v) for k, v in evidence["source"].value_counts(dropna=False).items()},
        "pairs_with_multiple_target_name_strings": int((evidence.groupby(KEY)["target_name"].nunique() > 1).sum()),
        "provenance_groups_with_gt10fold_within_group": int((study["within_provenance_range_pic50"] > 1.0).sum()),
        "example_cases": [
            {"rank": int(row.review_rank), "target": row.target,
             "structure_id": row.m18c_structure_id, "n_observations": int(row.n_observations),
             "n_provenance_groups": int(row.n_provenance_groups)}
            for row in selected.sort_values("review_rank").head(5).itertuples(index=False)
        ],
        "cautions": [
            "Selection is deliberately extreme and not representative of all BindingDB observations.",
            "Publication/patent provenance_key is not an assay ID, laboratory ID or evidence of independence.",
            "Matching exact values/context are possible duplicate reporting or technical replicates, not confirmed duplicates.",
            "PubChem AID, when present, is an assay identifier but may not resolve all assay conditions.",
            "Missing pH/temperature metadata does not establish assay equality.",
            "No lab/source causal effect estimated; no observation changed, merged, censored or relabeled.",
        ],
    }
    return result


def make_figure(selected: pd.DataFrame, evidence: pd.DataFrame, path: Path) -> None:
    """One chart: each case's observed pIC50 range; cutoff pIC50=6."""
    fig, ax = plt.subplots(figsize=(10, max(5, len(selected) * 0.34 + 1.5)))
    ordered = selected.sort_values(["target", "review_rank"])
    samples = evidence.groupby(KEY).agg(lo=("pic50", "min"), hi=("pic50", "max"), n=("pic50", "size")).reset_index()
    plot_rows = ordered.merge(samples, on=KEY, validate="one_to_one")
    labels = [f"{r.target} · rank {int(r.review_rank)} (n={int(r.n)})" for r in plot_rows.itertuples(index=False)]
    y = np.arange(len(plot_rows))
    ax.hlines(y, plot_rows["lo"], plot_rows["hi"], linewidth=2.7)
    ax.scatter(plot_rows["lo"], y, s=20)
    ax.scatter(plot_rows["hi"], y, s=20)
    ax.axvline(6.0, color="0.35", linestyle="--", linewidth=1.4, label="IC50=1000 nM (pIC50=6)")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Reported pIC50 (higher = stronger inhibition)")
    ax.set_title("M1.9c · selected priority cases, raw assay disagreement\nBoth targets represented; descriptive, not model predictions")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="M1.8c observation-level CSV")
    parser.add_argument("--cases", type=Path, required=True, help="M1.9b priority_cases CSV")
    parser.add_argument("--m19a-summary", type=Path, help="Optional M1.9a summary for checksum validation")
    parser.add_argument("--per-target", type=int, default=10)
    parser.add_argument("--evidence", type=Path, required=True, help="PRIVATE observation-level evidence CSV")
    parser.add_argument("--study-summary", type=Path, required=True, help="PRIVATE provenance-level case CSV")
    parser.add_argument("--summary", type=Path, required=True, help="Public aggregate JSON")
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    input_hash = sha256_file(args.input)
    cases_hash = sha256_file(args.cases)
    if args.m19a_summary:
        known = json.loads(args.m19a_summary.read_text(encoding="utf-8"))
        if input_hash != known["input_sha256"]:
            raise ValueError("M1.9a and M1.9c standardized-observation input checksums differ")
    obs = pd.read_csv(args.input, low_memory=False)
    cases = pd.read_csv(args.cases, low_memory=False)
    selected, evidence, study = reconstruct(obs, cases, args.per_target)
    summary = build_summary(selected, evidence, study)
    summary.update({
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_file": args.input.name, "input_sha256": input_hash,
        "priority_file": args.cases.name, "priority_sha256": cases_hash,
        "per_target_limit": args.per_target,
        "outputs": {"private_evidence": str(args.evidence), "private_study_summary": str(args.study_summary),
                    "figure": str(args.figure)},
    })
    # Always write full-context private evidence; do not commit to public repo.
    for path in [args.evidence, args.study_summary, args.summary]:
        path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(args.evidence, index=False)
    study.to_csv(args.study_summary, index=False)
    make_figure(selected, evidence, args.figure)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Selected molecule-target cases: {len(selected)} (by target: {summary['selected_pairs_by_target']})")
    print(f"Original exact IC50 observations: {len(evidence)}")
    print(f"Provenance groups: {len(study)}")
    print(f"Potential repeated-value/context rows: {summary['suspected_duplicate_signature_rows']}")
    print(f"Evidence: {args.evidence}")
    print(f"Summary: {args.summary}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
