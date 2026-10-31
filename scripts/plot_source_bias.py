"""M1.5: Plot exploratory source and assay confounding diagnostics.

Input:  M1.4 observation-level CSV (not molecule-level curated labels).
Output: reports/figures/source_bias/*.png and reports/audits/*.json.

All comparisons are DESCRIPTIVE. Different sources measure different molecules
and target constructs. Figures cannot establish causal laboratory or assay bias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "target",
    "source",
    "target_name",
    "measurement_status",
    "pic50",
    "activity_at_1000_nm",
    "inchi_key",
    "article_doi",
    "pmid",
    "patent_number",
    "ph",
    "temperature_c",
)
META_FIELDS = {
    "Article DOI": "article_doi",
    "PMID": "pmid",
    "Patent number": "patent_number",
    "pH": "ph",
    "Temperature": "temperature_c",
}
SOURCE_SHORT = {
    "Curated from the literature by BindingDB": "BindingDB literature",
    "US Patent": "US Patent",
    "ChEMBL": "ChEMBL",
    "WIPO": "WIPO",
}
TARGET_ORDER = ("EGFR", "HER2")
STATUSES = ("active", "inactive", "indeterminate")


def short_source(name: str) -> str:
    return SOURCE_SHORT.get(name, name)


def load_observations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in M1.4 CSV: {missing}")
    if df.empty:
        raise ValueError("No observations in CSV")
    df = df.copy()
    df["target"] = df["target"].str.strip()
    df["source"] = df["source"].str.strip()
    df["target_name"] = df["target_name"].str.strip()
    df["measurement_status"] = df["measurement_status"].str.strip()
    df["activity_at_1000_nm"] = df["activity_at_1000_nm"].str.strip()
    df["pic50_numeric"] = pd.to_numeric(df["pic50"], errors="coerce")
    df["construct_annotated"] = df["target_name"].str.contains(r"\[", regex=True)

    if not df["target"].isin(TARGET_ORDER).all():
        raise ValueError("Unexpected target outside EGFR and HER2")
    if not df["measurement_status"].isin(("exact", "censored", "invalid")).all():
        raise ValueError("Unexpected measurement status")
    if not df["activity_at_1000_nm"].isin(STATUSES).all():
        raise ValueError("Unexpected activity classification")
    if df.loc[df["measurement_status"] == "exact", "pic50_numeric"].isna().any():
        raise ValueError("Exact IC50 record is missing numerical pIC50")
    if df.loc[df["measurement_status"] != "exact", "pic50_numeric"].notna().any():
        raise ValueError("Censored/invalid IC50 must not have exact pIC50")
    return df


def ordered_groups(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (target, source)
        for target in TARGET_ORDER
        for source in sorted(df.loc[df["target"] == target, "source"].unique())
    ]


def initialize_plot():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, outdir: Path, filename: str, footnote: str) -> str:
    fig.text(0.025, 0.018, footnote, fontsize=8, alpha=0.8)
    path = outdir / filename
    fig.savefig(path, bbox_inches="tight", pad_inches=0.23)
    plt.close(fig)
    return str(path.as_posix())


def source_counts(df: pd.DataFrame, outdir: Path) -> str:
    order = sorted(df["source"].unique())
    grouped = df.groupby(["source", "target"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x = np.arange(len(order))
    width = 0.34
    for index, target in enumerate(TARGET_ORDER):
        values = grouped.reindex(order, fill_value=0).get(target, pd.Series(0, index=order))
        bars = ax.bar(x + (index - 0.5) * width, values, width, label=target)
        ax.bar_label(bars, labels=[f"{v:,}" if v else "" for v in values], padding=3, fontsize=9)
    ax.set_xticks(x, [short_source(s) for s in order], rotation=9)
    ax.set_ylabel("Number of experimental observations")
    ax.set_title("01  Source composition differs between EGFR and HER2", loc="left", pad=16)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.17)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.20)
    return save(fig, outdir, "01_observations_by_source.png",
                "M1.4 raw observations. Sources are curation categories, not individual laboratories.")


def construct_share(df: pd.DataFrame, outdir: Path) -> str:
    groups = ordered_groups(df)
    unannotated, annotated, labels, totals = [], [], [], []
    for target, source in groups:
        data = df[(df.target == target) & (df.source == source)]
        n = len(data)
        a = int(data.construct_annotated.sum())
        unannotated.append(100 * (n-a)/n)
        annotated.append(100 * a/n)
        labels.append(f"{target} · {short_source(source)}")
        totals.append(n)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(labels))
    ax.barh(y, unannotated, label="Name not bracket-annotated")
    ax.barh(y, annotated, left=unannotated, label="Bracket-annotated target name")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 116)
    ax.set_xlabel("Share of observations (%)")
    ax.set_title("02  Target-construct annotations are source-dependent", loc="left", pad=16)
    ax.invert_yaxis()
    for i, (a, total) in enumerate(zip(annotated, totals, strict=True)):
        ax.text(101.7, i, f"{a:.1f}%  n={total:,}", va="center", fontsize=9)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, -0.20), ncol=2)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.22, left=0.27)
    return save(fig, outdir, "02_construct_annotations_by_source.png",
                "Bracket annotation is a screening flag, NOT proof of mutant status; unannotated names do not prove wild type.")


def potency_boxplot(df: pd.DataFrame, target: str, outdir: Path) -> str:
    exact = df[(df.target == target) & (df.measurement_status == "exact")]
    order = sorted(exact.source.unique())
    values = [exact.loc[exact.source == s, "pic50_numeric"].to_numpy() for s in order]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.boxplot(values, tick_labels=[short_source(s) for s in order], showfliers=False,
               showmeans=False, widths=0.5, patch_artist=False)
    medians = [float(np.median(arr)) for arr in values]
    for i, (arr, median) in enumerate(zip(values, medians, strict=True), start=1):
        ax.text(i, median + 0.20, f"median {median:.2f}\nn={len(arr):,}",
                ha="center", fontsize=8.5)
    ax.set_ylabel("Exact pIC50 = 9 − log₁₀(IC50 in nM)")
    ax.set_title(f"0{'3' if target == 'EGFR' else '4'}  {target}: reported potency differs by source", loc="left", pad=16)
    ax.tick_params(axis="x", labelrotation=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.19)
    return save(fig, outdir, f"0{'3' if target == 'EGFR' else '4'}_exact_pic50_by_source_{target.lower()}.png",
                "Exact IC50 only; no adjustment for molecule, assay, construct, or repeat-measurement differences. Outliers hidden in boxplots.")


def activity_share(df: pd.DataFrame, outdir: Path) -> str:
    groups = ordered_groups(df)
    labels = [f"{target} · {short_source(source)}" for target, source in groups]
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(11, 6.1))
    left = np.zeros(len(groups))
    for activity in STATUSES:
        counts = [int(((df.target == t) & (df.source == s) &
                       (df.activity_at_1000_nm == activity)).sum()) for t, s in groups]
        totals = [int(((df.target == t) & (df.source == s)).sum()) for t, s in groups]
        percentages = np.asarray(counts) / np.asarray(totals) * 100
        ax.barh(y, percentages, left=left, label=activity.capitalize())
        for idx, p in enumerate(percentages):
            if p >= 9:
                ax.text(left[idx] + p/2, idx, f"{p:.0f}%", ha="center", va="center", fontsize=8)
        left += percentages
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percent of observations")
    ax.set_title("05  Activity evidence differs across sources", loc="left", pad=16)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.21), ncol=3)
    fig.subplots_adjust(bottom=0.22, left=0.27)
    return save(fig, outdir, "05_activity_by_source.png",
                "Strict IC50 < 1000 nM; censored bounds classified conservatively. Not prevalence or model accuracy.")


def metadata_completeness(df: pd.DataFrame, outdir: Path) -> str:
    groups = ordered_groups(df)
    matrix = np.empty((len(groups), len(META_FIELDS)))
    labels = []
    for i, (target, source) in enumerate(groups):
        subset = df[(df.target == target) & (df.source == source)]
        labels.append(f"{target} · {short_source(source)}")
        for j, field in enumerate(META_FIELDS.values()):
            matrix[i, j] = 100 * subset[field].str.strip().ne("").mean()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(len(META_FIELDS)), list(META_FIELDS.keys()))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if value < 50 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Field populated (%)")
    ax.set_title("06  Provenance and assay metadata completeness", loc="left", pad=16)
    fig.subplots_adjust(left=0.28, bottom=0.15)
    return save(fig, outdir, "06_metadata_completeness.png",
                "A blank publication DOI does not mean no provenance; patents often use patent numbers instead.")


def repeated_molecules(df: pd.DataFrame, outdir: Path) -> str:
    groups = ordered_groups(df)
    labels, total_obs, distinct, multiples = [], [], [], []
    for target, source in groups:
        subset = df[(df.target == target) & (df.source == source)]
        inchi = subset.loc[subset.inchi_key.str.strip() != "", "inchi_key"]
        labels.append(f"{target} · {short_source(source)}")
        total_obs.append(len(subset))
        distinct.append(inchi.nunique())
        multiples.append(len(subset)/inchi.nunique() if inchi.nunique() else float('nan'))
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(11, 6.4))
    height = 0.36
    ax.barh(y-height/2, total_obs, height=height, label="Observations")
    ax.barh(y+height/2, distinct, height=height, label="Distinct reported InChIKeys")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("07  Repeated molecules appear in the observation table", loc="left", pad=16)
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.17)
    ax.set_axisbelow(True)
    for i, ratio in enumerate(multiples):
        if math.isfinite(ratio):
            ax.text(total_obs[i] + max(total_obs)*0.015, i, f"{ratio:.2f}× obs/key", va="center", fontsize=9)
    ax.set_xlim(0, max(total_obs)*1.20)
    fig.subplots_adjust(left=0.29, bottom=0.12)
    return save(fig, outdir, "07_observations_vs_distinct_inchikeys.png",
                "InChIKey is the source identifier, not RDKit-normalized identity; repeated rows may reflect genuine repeat assays.")


def write_summary(df: pd.DataFrame, original_csv: Path, summary_path: Path, figure_paths: list[str]) -> None:
    entries = []
    for target, source in ordered_groups(df):
        subset = df[(df.target == target) & (df.source == source)]
        exact = subset[subset.measurement_status == "exact"]
        entries.append({
            "target": target,
            "source": source,
            "observations": len(subset),
            "exact_ic50": len(exact),
            "censored_ic50": int((subset.measurement_status == "censored").sum()),
            "invalid_ic50": int((subset.measurement_status == "invalid").sum()),
            "exact_pic50_median": (round(float(exact.pic50_numeric.median()), 4) if len(exact) else None),
            "bracket_annotated_target_names": int(subset.construct_annotated.sum()),
            "bracket_annotation_percent": round(float(100 * subset.construct_annotated.mean()), 2),
            "activity_counts": {
                k: int((subset.activity_at_1000_nm == k).sum()) for k in STATUSES
            },
            "distinct_reported_inchi_keys": int(
                subset.loc[subset.inchi_key.str.strip().ne(""), "inchi_key"].nunique()
            ),
            "metadata_population_percent": {
                title: round(float(100 * subset[col].str.strip().ne("").mean()), 2)
                for title, col in META_FIELDS.items()
            },
        })
    obj = {
        "stage": "M1.5_source_and_construct_descriptive_diagnostics",
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_file": original_csv.name,
        "input_sha256": hashlib.sha256(original_csv.read_bytes()).hexdigest(),
        "records": len(df),
        "plot_versions": {"matplotlib": matplotlib.__version__, "pandas": pd.__version__},
        "groups": entries,
        "figures": [Path(f).name for f in figure_paths],
        "limitations": [
            "Source is BindingDB curation source; not individual lab/site identity.",
            "Observations are not independent molecules; duplicates and repeated assays preserved.",
            "Bracket notation is a proxy for construct annotation, not validated genotype.",
            "Only exact IC50 records are used in pIC50 boxplots; censored records excluded.",
            "No causal source bias, model bias, or model performance is established.",
            "A full independent molecule/assay curation and source-held-out evaluation remain future work.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("data/interim/bindingdb_egfr_her2_observations.csv"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("reports/figures/source_bias"))
    parser.add_argument("--summary", type=Path,
                        default=Path("reports/audits/M1_5_source_bias_profile.json"))
    args = parser.parse_args()
    initialize_plot()
    df = load_observations(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    outputs = [
        source_counts(df, args.outdir),
        construct_share(df, args.outdir),
        potency_boxplot(df, "EGFR", args.outdir),
        potency_boxplot(df, "HER2", args.outdir),
        activity_share(df, args.outdir),
        metadata_completeness(df, args.outdir),
        repeated_molecules(df, args.outdir),
    ]
    write_summary(df, args.input, args.summary, outputs)
    print(f"M1.5 source diagnostics: {len(df):,} observation rows; {len(outputs)} PNG images")
    for out in outputs:
        print("  " + out)
    print(f"  {args.summary}")


if __name__ == "__main__":
    main()
