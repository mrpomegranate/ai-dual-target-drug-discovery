"""M1.6: Read-only name-construct and experimental provenance diagnostics.

This is not sequence-verified wild-type classification, laboratory attribution,
assay harmonization, deduplication, or a curated molecule-level dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

BASE_NAMES = {
    "EGFR": "Epidermal growth factor receptor",
    "HER2": "Receptor tyrosine-protein kinase erbB-2",
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SUBSTITUTION = re.compile(rf"^[{AMINO_ACIDS}][1-9][0-9]*[{AMINO_ACIDS}]$")
PARENTHESES = re.compile(r"\(([^()]*)\)")
CATEGORIES = (
    "plain_name",
    "substitutions_only",
    "mixed_construct",
    "other_construct",
    "unclassified_nonbase",
)
LABELS = {
    "plain_name": "Plain target name",
    "substitutions_only": "Substitutions only",
    "mixed_construct": "Substitutions + other changes",
    "other_construct": "Other construct annotation",
    "unclassified_nonbase": "Other non-base name",
}
REQUIRED = {
    "target", "target_name", "source", "inchi_key", "measurement_status",
    "article_doi", "pmid", "patent_number", "reactant_set_id",
}


def classify_name(target: str, name: str) -> tuple[str, str, str]:
    """Return category, notation, and mutation signature (not sequence-verified)."""
    base = BASE_NAMES[target]
    name = name.strip()
    if name == base:
        return "plain_name", "none", ""

    if name.startswith(base + " [") and name.endswith("]"):
        parts = [part.strip() for part in name[len(base) + 2 : -1].split(",")]
        variants = sorted({part for part in parts if SUBSTITUTION.fullmatch(part)})
        if variants and len(variants) == len(parts):
            category = "substitutions_only"
        elif variants:
            category = "mixed_construct"
        else:
            category = "other_construct"
        return category, "square_brackets", "|".join(variants)

    if name.startswith(base + " "):
        suffix = name[len(base) + 1 :].strip()
        chunks = PARENTHESES.findall(suffix)
        if chunks and "".join(f"({part})" for part in chunks) == suffix:
            variants = sorted({part for part in chunks if SUBSTITUTION.fullmatch(part)})
            allowed_aliases = {target, "EGFR", "HER2", "erbB-2"}
            if variants and all(
                SUBSTITUTION.fullmatch(part) or part in allowed_aliases for part in chunks
            ):
                return "substitutions_only", "parentheses", "|".join(variants)
            return "other_construct", "parentheses", "|".join(variants)

    return "unclassified_nonbase", "other", ""


def provenance_key(row: pd.Series) -> tuple[str, str]:
    """A publication/patent grouping proxy, never laboratory identification."""
    patent = row["patent_number"].strip()
    pmid = row["pmid"].strip()
    doi = row["article_doi"].strip().lower()
    if patent:
        return "patent", f"patent:{patent.upper()}"
    if pmid:
        return "pmid", f"pmid:{pmid}"
    if doi:
        return "article_doi", f"doi:{doi.removeprefix('https://doi.org/')}"
    return "unresolved", ""


def load_and_classify(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    missing = sorted(REQUIRED - set(data.columns))
    if missing:
        raise ValueError(f"M1.4 observation CSV missing columns: {missing}")
    if data.empty:
        raise ValueError("Observation CSV is empty")
    data = data.copy()
    if not data.target.isin(BASE_NAMES).all():
        raise ValueError("Unexpected targets in observation CSV")
    if data.reactant_set_id.duplicated().any():
        raise ValueError("Duplicate BindingDB reactant_set_id in source observations")
    annotations = [
        classify_name(target, name)
        for target, name in zip(data.target, data.target_name, strict=True)
    ]
    data[["construct_category", "name_notation", "mutation_signature"]] = (
        pd.DataFrame(annotations, index=data.index)
    )
    studies = [provenance_key(row) for _, row in data.iterrows()]
    data[["provenance_type", "provenance_key"]] = pd.DataFrame(
        studies, index=data.index
    )
    return data


def source_groups(data: pd.DataFrame) -> list[dict]:
    result = []
    for (target, source), subset in data.groupby(["target", "source"], sort=True):
        result.append(
            {
                "target": target,
                "source": source,
                "observations": len(subset),
                "construct_categories": {
                    cat: int((subset.construct_category == cat).sum())
                    for cat in CATEGORIES
                },
                "name_notations": {
                    kind: int((subset.name_notation == kind).sum())
                    for kind in ("none", "square_brackets", "parentheses", "other")
                },
                "provenance_resolved_rows": int(subset.provenance_key.ne("").sum()),
                "provenance_missing_rows": int(subset.provenance_key.eq("").sum()),
                "distinct_provenance_groups": int(
                    subset.loc[subset.provenance_key.ne(""), "provenance_key"].nunique()
                ),
            }
        )
    return result


def source_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summary(data: pd.DataFrame, input_path: Path) -> dict:
    counts = data.construct_category.value_counts()
    nonempty_inchi = data.loc[data.inchi_key.ne("")].copy()
    by_target = {}
    for target, subset in nonempty_inchi.groupby("target"):
        categories_per_key = subset.groupby("inchi_key").construct_category.nunique()
        names_per_key = subset.groupby("inchi_key").target_name.nunique()
        by_target[target] = {
            "reported_distinct_inchikeys": int(subset.inchi_key.nunique()),
            "inchikeys_with_multiple_construct_categories": int(
                (categories_per_key > 1).sum()
            ),
            "inchikeys_with_multiple_target_names": int((names_per_key > 1).sum()),
        }
    egfr = set(nonempty_inchi.loc[nonempty_inchi.target == "EGFR", "inchi_key"])
    her2 = set(nonempty_inchi.loc[nonempty_inchi.target == "HER2", "inchi_key"])
    return {
        "stage": "M1.6_construct_name_and_study_provenance_diagnostic_not_curated_labels",
        "generated_utc": datetime.now(UTC).isoformat(),
        "source_csv": input_path.name,
        "source_csv_sha256": source_sha256(input_path),
        "observations": len(data),
        "all_construct_categories": {
            cat: int(counts.get(cat, 0)) for cat in CATEGORIES
        },
        "groups": source_groups(data),
        "reported_inchikey_overlap": {
            "both_targets_all_constructs": len(egfr & her2),
            "caveat": (
                "Source-reported InChIKey overlap, not paired curated labels; "
                "constructs, exact/censored assays and repeat measurements not reconciled."
            ),
            "per_target": by_target,
        },
        "limitations": [
            "Plain target name does not establish wild-type protein or full-length construct.",
            "Mutation signatures are parsed from target names, not verified against sequences.",
            "Parenthetical EGFR mutations are recognized; unsupported names are flagged.",
            "Construct categories are distinct even when source-reported InChIKeys match.",
            "Publication/PMID/patent are provenance proxies, not laboratory identifiers.",
            "Different assay protocols, selection, and experimental error are not yet adjusted.",
            "No observations are removed, merged, imputed, or used as final labels.",
        ],
    }


def plot_construct_composition(data: pd.DataFrame, output: Path) -> None:
    grouped = (
        data.groupby(["target", "source", "construct_category"], sort=True)
        .size().unstack(fill_value=0).reindex(columns=CATEGORIES, fill_value=0)
    )
    labels = [f"{target} · {source}" for target, source in grouped.index]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    left = np.zeros(len(grouped))
    yy = np.arange(len(grouped))
    for cat in CATEGORIES:
        percent = (100 * grouped[cat].to_numpy() / grouped.sum(axis=1).to_numpy())
        ax.barh(yy, percent, left=left, label=LABELS[cat])
        left += percent
    ax.set_yticks(yy, labels)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("Share of source observations (%)")
    ax.set_title("08  Target-name construct categories by source", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    fig.text(
        0.03, 0.015,
        "Name annotations only; neither a plain name nor a mutation string verifies the tested protein sequence.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.39, bottom=0.24, top=0.91)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def audit(input_path: Path, summary_path: Path, inventory_path: Path, figure_path: Path) -> dict:
    data = load_and_classify(input_path)
    result = summary(data, input_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    inventory = (
        data.groupby(
            ["target", "source", "target_name", "construct_category", "name_notation", "mutation_signature"],
            dropna=False,
            sort=True,
        ).size().rename("observations").reset_index()
    )
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(inventory_path, index=False)
    plot_construct_composition(data, figure_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/interim/bindingdb_egfr_her2_observations.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/audits/M1_6_construct_provenance_profile.json"))
    parser.add_argument("--inventory", type=Path, default=Path("reports/audits/M1_6_target_name_inventory.csv"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/source_bias/08_construct_name_categories_by_source.png"))
    args = parser.parse_args()
    result = audit(args.input, args.summary, args.inventory, args.figure)
    print(f"Audited {result['observations']:,} observations")
    for group in result["groups"]:
        categories = group["construct_categories"]
        print(f"  {group['target']} | {group['source']}: n={group['observations']:,}, plain={categories['plain_name']:,}, substitutions={categories['substitutions_only']:,}, other={categories['mixed_construct']+categories['other_construct']+categories['unclassified_nonbase']:,}")
    print(f"Reported InChIKeys appearing under both targets: {result['reported_inchikey_overlap']['both_targets_all_constructs']:,} (NOT curated paired labels)")
    print(f"Saved summary: {args.summary}")
    print(f"Saved inventory: {args.inventory}")
    print(f"Saved figure: {args.figure}")


if __name__ == "__main__":
    main()
