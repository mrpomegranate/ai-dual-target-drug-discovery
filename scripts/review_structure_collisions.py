"""M1.8d — inspect structure representation collisions, without label merging."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

REQUIRED = {
    "reactant_set_id", "smiles", "target", "source", "m17_partition",
    "m18c_status", "m18c_structure_id", "m18c_candidate_isomeric_smiles",
    "m18c_candidate_rdkit_inchikey",
}
APPROVED = "provisional_exact_structure"
TYPE_RAW = "multiple_raw_smiles_same_isomeric_canonical"
TYPE_KEY = "one_rdkit_inchikey_multiple_canonical_smiles"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collision_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return approved observations and collision inventory (no chemistry transforms)."""
    missing = sorted(REQUIRED.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["reactant_set_id"].eq("").any() or df["reactant_set_id"].duplicated().any():
        raise ValueError("Each observation must have a nonempty unique reactant_set_id")

    approved = df.loc[df["m18c_status"].eq(APPROVED)].copy()
    for field in ("m18c_structure_id", "m18c_candidate_isomeric_smiles", "m18c_candidate_rdkit_inchikey"):
        if approved[field].eq("").any():
            raise ValueError(f"Approved observation missing {field}")

    if (approved.groupby("m18c_candidate_isomeric_smiles")["m18c_structure_id"].nunique() > 1).any():
        raise ValueError("A canonical SMILES has multiple provisional structure IDs")
    if (approved.groupby("m18c_structure_id")["m18c_candidate_isomeric_smiles"].nunique() > 1).any():
        raise ValueError("A provisional structure ID has multiple canonical SMILES")

    colliding_raw = approved.groupby("m18c_candidate_isomeric_smiles")["smiles"].nunique()
    colliding_raw = colliding_raw[colliding_raw > 1].sort_index()
    colliding_key = approved.groupby("m18c_candidate_rdkit_inchikey")["m18c_candidate_isomeric_smiles"].nunique()
    colliding_key = colliding_key[colliding_key > 1].sort_index()

    rows = []
    for ctype, identifiers, col in (
        (TYPE_RAW, colliding_raw.index.tolist(), "m18c_candidate_isomeric_smiles"),
        (TYPE_KEY, colliding_key.index.tolist(), "m18c_candidate_rdkit_inchikey"),
    ):
        for identifier in identifiers:
            group = approved.loc[approved[col].eq(identifier)]
            canon = sorted(set(group["m18c_candidate_isomeric_smiles"]))
            raw = sorted(set(group["smiles"]))
            rows.append(
                {
                    "collision_type": ctype,
                    "identifier": identifier,
                    "observations": len(group),
                    "unique_raw_smiles": len(raw),
                    "unique_canonical_smiles": len(canon),
                    "unique_structure_ids": group["m18c_structure_id"].nunique(),
                    "targets": "; ".join(sorted(set(group["target"]))),
                    "sources": "; ".join(sorted(set(group["source"]))),
                    "m17_partitions": "; ".join(sorted(set(group["m17_partition"]))),
                    "sample_record_ids": "; ".join(group["reactant_set_id"].head(8).tolist()),
                    "canonical_smiles_json": json.dumps(canon, ensure_ascii=False),
                    "raw_smiles_json": json.dumps(raw, ensure_ascii=False),
                    "structure_ids_json": json.dumps(sorted(set(group["m18c_structure_id"]))),
                    "recommended_action": (
                        "Canonical-SMILES representations already share an exact structure ID; retain raw forms"
                        if ctype == TYPE_RAW else
                        "Retain distinct structure IDs and labels; flag an equivalence family for split-leakage review"
                    ),
                }
            )
    return approved, rows


def check_inventory(rows: list[dict], inventory_path: Path) -> None:
    inventory = pd.read_csv(inventory_path, dtype=str, keep_default_na=False)
    required = {"collision_type", "identifier", "distinct_structures_or_representations"}
    if not required.issubset(inventory.columns):
        raise ValueError("M1.8c inventory missing required columns")
    observed = {
        (r["collision_type"], r["identifier"]): (
            r["unique_raw_smiles"] if r["collision_type"] == TYPE_RAW else r["unique_canonical_smiles"]
        )
        for r in rows
    }
    expected = {}
    for _, r in inventory.iterrows():
        key = (r["collision_type"], r["identifier"])
        if key in expected:
            raise ValueError("Repeated collision key in M1.8c inventory")
        expected[key] = int(r["distinct_structures_or_representations"])
    if observed != expected:
        raise ValueError("Current collision groups do not match the M1.8c inventory")


def create_report(df: pd.DataFrame, input_path: Path, inventory: Path, review: Path, summary_path: Path, figure: Path) -> dict:
    approved, rows = collision_groups(df)
    check_inventory(rows, inventory)

    review.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(review, index=False, encoding="utf-8")

    by_type = {}
    for ctype, field in ((TYPE_RAW, "m18c_candidate_isomeric_smiles"), (TYPE_KEY, "m18c_candidate_rdkit_inchikey")):
        ids = {r["identifier"] for r in rows if r["collision_type"] == ctype}
        sub = approved[approved[field].isin(ids)]
        by_type[ctype] = {
            "collision_groups": len(ids),
            "observation_rows": len(sub),
            "unique_structure_ids": sub["m18c_structure_id"].nunique(),
            "by_target": dict(Counter(sub["target"])),
            "by_m17_partition": dict(Counter(sub["m17_partition"])),
        }

    summary = {
        "stage": "M1.8d_representation_collision_review_NOT_label_aggregation",
        "generated_utc": datetime.now(UTC).isoformat(),
        "input_file": input_path.name,
        "input_sha256": sha256(input_path),
        "m18c_collision_inventory_file": inventory.name,
        "m18c_collision_inventory_sha256": sha256(inventory),
        "all_observations": len(df),
        "approved_observations": len(approved),
        "held_observations": len(df) - len(approved),
        "collision_results": by_type,
        "collision_groups_total": len(rows),
        "inclusion_policy": "Canonical isomeric SMILES remains exact-structure identity. Shared InChIKey alone does not merge or average labels.",
        "split_policy": "Future splits must keep the same exact ID together and review same-InChIKey families for leakage; scaffold/source controls remain separately required.",
        "limitations": [
            "A collision is a representation/identifier diagnostic, not evidence that an assay measurement is wrong.",
            "Same InChIKey can correspond to multiple input canonical SMILES, including tautomeric/resonance-like forms.",
            "No pIC50/IC50 averaging, no target construct verification, no scaffold split, no model training.",
            "Source publication identifiers and assay matching require further reconciliation.",
        ],
        "output": {"review_csv": str(review), "figure": str(figure)},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    short = ["Same canonical SMILES\nfrom different raw SMILES", "Same recalculated InChIKey\nfrom different canonical SMILES"]
    types = [TYPE_RAW, TYPE_KEY]
    fig = Figure(figsize=(11.5, 4.3), layout="constrained")
    axes = fig.subplots(1, 2)
    for ax, name, stat in zip(axes, short, ["collision_groups", "observation_rows"], strict=True):
        vals = [by_type[t][stat] for t in types]
        bars = ax.barh(short, vals)
        ax.bar_label(bars, padding=3)
        ax.set_xlabel("Collision groups" if stat == "collision_groups" else "Affected observation rows")
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_title("Representation groups" if stat == "collision_groups" else "Observations involved")
    fig.suptitle("M1.8d — Molecular representation collision audit")
    fig.savefig(figure, dpi=170)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    report = create_report(df, args.input, args.inventory, args.review, args.summary, args.figure)
    print(f"Observations reviewed: {report['all_observations']:,}")
    print(f"Approved structure rows: {report['approved_observations']:,}")
    for category, stats in report["collision_results"].items():
        print(f"{category}: {stats['collision_groups']} groups across {stats['observation_rows']} observations")
    print(f"Review: {args.review}\nSummary: {args.summary}\nFigure: {args.figure}")


if __name__ == "__main__":
    main()
