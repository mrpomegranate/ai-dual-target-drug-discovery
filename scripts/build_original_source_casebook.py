"""Build a conservative, traceable original-publication verification casebook.

Reads M1.9d.1 review queue and M1.9c observation evidence. Does NOT
validate an individual assay, relabel a target, average IC50 or remove data.
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
PRIORITY = {"highest": 0, "high": 1, "routine": 2}
REQ_QUEUE = KEYS + [
    "source_review_priority", "n_observations", "min_ic50_nm", "max_ic50_nm",
    "observed_pic50_range", "review_rank", "within_group_threshold_crossing",
    "article_doi_candidates", "pmid_candidates", "patent_candidates",
]
REQ_EVIDENCE = KEYS + [
    "reactant_set_id", "monomer_id", "ic50_nm", "pic50", "target_name",
    "m17_partition", "m18c_status", "source", "pmid", "article_doi",
]

# Publication-level context checked in PubMed. Crucially, NOT row-to-table verification.
PUBMED_LEADS = {
    "32931277": {
        "title": "Targeting Her2-insYVMA with Covalent Inhibitors—A Focused Compound Screening and Structure-Based Design Approach",
        "context": "Publication studies wild-type HER2 and HER2 A775_G776insYVMA. Exact EGFR row assignments have not been verified.",
    },
    "37197473": {
        "title": "Addressing the Osimertinib Resistance Mutation EGFR-L858R/C797S with Reversible Aminopyrimidines",
        "context": "Full text reports HTRF biochemical EGFR WT/L858R/L858R-C797S and CTG Ba/F3 cellular assays; individual BindingDB rows remain unmatched.",
    },
    "36375337": {
        "title": "Design, synthesis, and biological evaluation of novel quinoline derivatives as small molecule mutant EGFR inhibitors targeting resistance in NSCLC",
        "context": "Publication is about mutant EGFR inhibitors; exact endpoint/units for this BindingDB row not yet checked.",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identifier_to_url(key: str) -> str:
    if key.startswith("pmid:") and key[5:].isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{key[5:]}/"
    if key.startswith("doi:"):
        return f"https://doi.org/{key[4:]}"
    # Never guess a URL from an ambiguous identifier.
    return ""


def validate_evidence(queue: pd.DataFrame, evidence: pd.DataFrame) -> None:
    for label, frame, cols in [
        ("queue", queue, REQ_QUEUE),
        ("evidence", evidence, REQ_EVIDENCE),
    ]:
        missing = sorted(set(cols) - set(frame.columns))
        if missing:
            raise ValueError(f"{label} missing columns: {missing}")
        if frame[KEYS].isna().any().any():
            raise ValueError(f"{label} contains missing group identifiers")
    if queue.duplicated(subset=KEYS).any():
        raise ValueError("Duplicate source group key in queue")
    if evidence["reactant_set_id"].isna().any() or evidence["reactant_set_id"].duplicated().any():
        raise ValueError("Nonunique or missing reactant_set_id")
    if not evidence["m17_partition"].eq("primary_exact_candidate").all():
        raise ValueError("Non-primary-exact row in evidence")
    if not evidence["m18c_status"].eq("provisional_exact_structure").all():
        raise ValueError("Unapproved molecular identity in evidence")
    if evidence["ic50_nm"].isna().any() or not evidence["ic50_nm"].gt(0).all():
        raise ValueError("Missing or nonpositive exact IC50 in evidence")
    if evidence["pic50"].isna().any():
        raise ValueError("Missing pIC50")
    if not queue["source_review_priority"].isin(PRIORITY).all():
        raise ValueError("Unrecognized review priority")
    ag = evidence.groupby(KEYS, dropna=False).agg(
        n=("reactant_set_id", "size"), lo=("ic50_nm", "min"),
        hi=("ic50_nm", "max"), p_lo=("pic50", "min"),
        p_hi=("pic50", "max"),
    ).reset_index()
    merged = queue.merge(ag, on=KEYS, validate="one_to_one", how="outer", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Queue and evidence do not contain identical source groups")
    checks = {
        "observation count": (merged["n_observations"] - merged["n"]).abs() < 1e-7,
        "minimum IC50": (merged["min_ic50_nm"] - merged["lo"]).abs() < 1e-6,
        "maximum IC50": (merged["max_ic50_nm"] - merged["hi"]).abs() < 1e-6,
        "pIC50 spread": (merged["observed_pic50_range"] - (merged["p_hi"] - merged["p_lo"])).abs() < 1e-5,
    }
    for label, valid in checks.items():
        if not valid.all():
            raise ValueError(f"Queue/evidence mismatch: {label}")
    is_crossing = (merged["lo"] < 1000) & (merged["hi"] >= 1000)
    reported = merged["within_group_threshold_crossing"].astype(str).str.lower().eq("true")
    if not is_crossing.eq(reported).all():
        raise ValueError("Activity threshold contradiction mismatch")


def select_groups(
    queue: pd.DataFrame,
    per_target: int = 6,
    focus_pmids: tuple[str, ...] = ("32931277", "37197473", "36375337"),
) -> pd.DataFrame:
    """Select source groups per target, adding named literature leads if needed."""
    if per_target < 1:
        raise ValueError("per_target must be >= 1")
    ranked = queue.copy()
    ranked["_priority_rank"] = ranked["source_review_priority"].map(PRIORITY)
    ranked = ranked.sort_values(
        ["_priority_rank", "observed_pic50_range", "n_observations", "review_rank", "provenance_key"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )
    selected = pd.concat(
        [ranked[ranked.target == target].head(per_target) for target in ["EGFR", "HER2"]],
        ignore_index=True,
    )
    # Add one representative group for each focus publication; these are investigative leads.
    for pmid in focus_pmids:
        matches = ranked[ranked["provenance_key"].eq(f"pmid:{pmid}")]
        if len(matches) and not selected["provenance_key"].eq(f"pmid:{pmid}").any():
            selected = pd.concat([selected, matches.head(1)], ignore_index=True)
    return selected.drop_duplicates(subset=KEYS).drop(columns="_priority_rank").reset_index(drop=True)


def build_casebook(queue: pd.DataFrame, evidence: pd.DataFrame,
                   per_target: int = 6,
                   focus_pmids: tuple[str, ...] = ("32931277", "37197473", "36375337"),
                   ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    validate_evidence(queue, evidence)
    selected = select_groups(queue, per_target, focus_pmids)
    chosen = evidence.merge(selected[KEYS], on=KEYS, how="inner", validate="many_to_one")
    if len(chosen) != selected["n_observations"].sum():
        raise ValueError("Selected source observation counts do not reconcile")
    rows = []
    for group in selected.itertuples(index=False):
        mask = (chosen.target.eq(group.target) &
                chosen.m18c_structure_id.eq(group.m18c_structure_id) &
                chosen.provenance_key.eq(group.provenance_key))
        subset = chosen.loc[mask].sort_values("ic50_nm", kind="mergesort")
        pmid = group.provenance_key[5:] if group.provenance_key.startswith("pmid:") else ""
        lead = PUBMED_LEADS.get(pmid)
        rows.append({
            "target": group.target,
            "review_rank": int(group.review_rank),
            "m18c_structure_id": group.m18c_structure_id,
            "provenance_key": group.provenance_key,
            "source_review_priority": group.source_review_priority,
            "observations_in_group": len(subset),
            "min_ic50_nm": float(subset.ic50_nm.min()),
            "max_ic50_nm": float(subset.ic50_nm.max()),
            "pic50_range": float(subset.pic50.max() - subset.pic50.min()),
            "source_categories": "; ".join(sorted(subset.source.astype(str).unique())),
            "reactant_set_ids": "; ".join(subset.reactant_set_id.astype(str).tolist()),
            "monomer_ids": "; ".join(sorted(set(subset.monomer_id.dropna().astype(str)))),
            "reported_ic50_nm_in_ascending_order": "; ".join(f"{v:g}" for v in subset.ic50_nm),
            "target_name_strings": "; ".join(sorted(subset.target_name.astype(str).unique())),
            "lookup_url": source_identifier_to_url(group.provenance_key),
            "paper_title_verified_from_pubmed": lead["title"] if lead else "",
            "publication_context_review": "pubmed_abstract_checked" if lead else "not_checked",
            "publication_context_note": lead["context"] if lead else "",
            "exact_compound_matched_to_source_table": "not_checked",
            "original_table_or_supplement_location": "",
            "measured_endpoint_verified": "not_checked",
            "biochemical_or_cellular_assay": "not_checked",
            "experimental_protein_construct_verified": "not_checked",
            "units_and_conversion_verified": "not_checked",
            "independence_of_measurements_verified": "not_checked",
            "decision": "defer_aggregation",
            "reviewer_evidence_notes": "",
        })
    worksheet = pd.DataFrame(rows)
    evidence_cols = [
        "target", "m18c_structure_id", "provenance_key", "reactant_set_id",
        "monomer_id", "ic50_nm", "pic50", "target_name", "source", "pmid",
        "article_doi", "pubchem_aid", "patent_number", "bindingdb_doi",
    ]
    subset_cols = [c for c in evidence_cols if c in chosen.columns]
    extracted = chosen[subset_cols].sort_values(
        ["target", "provenance_key", "m18c_structure_id", "ic50_nm"], kind="mergesort",
    ).reset_index(drop=True)
    summary = {
        "stage": "M1.9d2_original_publication_review_casebook_NOT_row_verification_or_labels",
        "full_queue_groups_reconciled": len(queue),
        "full_evidence_rows_reconciled": len(evidence),
        "selected_source_groups": len(worksheet),
        "selected_observation_rows": len(extracted),
        "selected_by_target": worksheet.target.value_counts().to_dict(),
        "selected_by_review_priority": worksheet.source_review_priority.value_counts().to_dict(),
        "publication_abstracts_prechecked": int(worksheet.publication_context_review.eq("pubmed_abstract_checked").sum()),
        "row_to_table_matches_confirmed": 0,
        "label_or_chemical_identity_changes": 0,
        "limitations": [
            "PubMed publication context is not a match between an individual BindingDB row and a paper table.",
            "Reported target names may conceal mutant constructs or mixed biochemical/cellular endpoints.",
            "The source groups are not laboratory IDs or proven independent assays.",
            "Only a deliberately focused subset is reviewed. No conclusions about whole-dataset rates.",
            "All original IC50 values are preserved. No labels are aggregated or altered.",
        ],
    }
    return worksheet, extracted, summary


def plot_casebook(worksheet: pd.DataFrame, output: Path) -> None:
    tab = worksheet.groupby(["target", "source_review_priority"]).size().unstack(fill_value=0)
    for cat in ["highest", "high", "routine"]:
        if cat not in tab:
            tab[cat] = 0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    left = [0] * len(tab)
    for col in ["highest", "high", "routine"]:
        widths = tab[col].to_list()
        ax.barh(tab.index, widths, left=left, label=col)
        left = [a+b for a,b in zip(left, widths, strict=True)]
    ax.set_xlabel("Selected molecule–target–publication/patent groups")
    ax.set_title("M1.9d.2 — Focused source-verification queue (not confirmed assays)")
    ax.legend(loc="best")
    ax.invert_yaxis()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--worksheet", type=Path, required=True)
    parser.add_argument("--selected-records", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--per-target", type=int, default=6)
    parser.add_argument("--focus-pmids", default="32931277,37197473,36375337")
    args = parser.parse_args()
    queue = pd.read_csv(args.queue, low_memory=False)
    evidence = pd.read_csv(args.evidence, low_memory=False, dtype={"reactant_set_id": "string", "pmid": "string"})
    focus_pmids = tuple(x.strip() for x in args.focus_pmids.split(",") if x.strip())
    worksheet, extracted, summary = build_casebook(queue, evidence, args.per_target, focus_pmids)
    summary["generated_utc"] = datetime.now(UTC).isoformat()
    summary["inputs"] = {
        "queue": args.queue.name,
        "queue_sha256": sha256(args.queue),
        "evidence": args.evidence.name,
        "evidence_sha256": sha256(args.evidence),
    }
    summary["outputs"] = {
        "worksheet": str(args.worksheet),
        "selected_records": str(args.selected_records),
        "figure": str(args.figure),
    }
    for path in (args.worksheet, args.selected_records, args.summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    worksheet.to_csv(args.worksheet, index=False)
    extracted.to_csv(args.selected_records, index=False)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_casebook(worksheet, args.figure)
    print(f"Validated {len(queue):,} provenance groups and {len(evidence):,} source observations")
    print(f"Selected {len(worksheet)} groups and {len(extracted)} records for original-source review")
    print(f"Publication abstracts checked: {summary['publication_abstracts_prechecked']} group references")
    print("Exact row-to-table matches confirmed: 0 (must be completed manually)")
    print(f"Worksheet: {args.worksheet}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
