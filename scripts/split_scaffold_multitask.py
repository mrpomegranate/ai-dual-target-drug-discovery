"""M2.1: deterministic, molecule-/Bemis-Murcko-scaffold-disjoint EGFR/HER2 splits.

Consumes the *unchanged* M1.11 provisional dataset, never the original raw BindingDB.
No observed pIC50 or activity class is used to choose train/validation/test groups.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

POLICY = "M2.1-rdkit-murcko-stereo-off-acyclic-identity-v1"
FRACTIONS = {"train": 0.80, "validation": 0.10, "test": 0.10}
PARTITIONS = tuple(FRACTIONS)
IDENTITY_PREFIX = b"rdkit-isomeric-smiles-policy-v1\0"
REQUIRED = {
    "m18c_structure_id", "canonical_isomeric_smiles", "training_split", "scaffold_split",
    "egfr_mask", "her2_mask", "egfr_pic50", "her2_pic50",
    "n_available_regression_tasks", "paired_activity_class",
}
MANIFEST_COLS = (
    "m18c_structure_id", "training_split", "scaffold_group_id", "scaffold_type",
    "murcko_scaffold_smiles", "egfr_mask", "her2_mask",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        missing = REQUIRED - set(fields)
        if missing:
            raise ValueError(f"Missing M1.11 columns: {sorted(missing)}")
        records = list(reader)
    if not records:
        raise ValueError("M1.11 dataset is empty")
    return records, fields


def compute_scaffold(smiles: str, structure_id: str) -> tuple[str, str, str]:
    """Return (group ID, Murcko scaffold SMILES, kind).

    Acyclic molecules have the empty Murcko scaffold. Keeping all in one group
    would dominate an artificial split, so assign EACH acyclic identity its own
    explicitly tagged group. Such examples are NOT scaffold OOD by definition.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"RDKit cannot parse structure {structure_id}")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Unexpected multi-fragment M1.11 structure {structure_id}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        # Scaffold is a chemical graph representation, NOT an activity label.
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(scaffold), isomericSmiles=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"BM-{digest}", canonical, "ring_scaffold"
    return f"ACYCLIC-{structure_id}", "", "acyclic_identity_group"


def validate_row(row: dict[str, str], seen_ids: set[str], seen_smiles: set[str]) -> tuple[int, int]:
    sid, smiles = row["m18c_structure_id"], row["canonical_isomeric_smiles"]
    if sid in seen_ids or smiles in seen_smiles:
        raise ValueError(f"Duplicate molecular identity or SMILES: {sid}")
    seen_ids.add(sid)
    seen_smiles.add(smiles)
    expected_id = "RDKISO1-" + hashlib.sha256(IDENTITY_PREFIX + smiles.encode("utf-8")).hexdigest()
    if sid != expected_id:
        raise ValueError(f"M1.8c identity digest mismatch: {sid}")
    if row["training_split"] not in {"NOT_ASSIGNED", ""} or row["scaffold_split"] not in {"NOT_ASSIGNED", ""}:
        raise ValueError("Input appears already split; always start from frozen M1.11 data")
    masks = []
    for target in ("egfr", "her2"):
        mask_text, value = row[f"{target}_mask"], row[f"{target}_pic50"].strip()
        if mask_text not in {"0", "1"}:
            raise ValueError(f"Invalid mask for {sid}: {target}={mask_text!r}")
        mask = int(mask_text)
        if mask:
            if not value or not math.isfinite(float(value)):
                raise ValueError(f"Present mask requires finite pIC50: {sid} {target}")
        elif value:
            raise ValueError(f"Missing mask requires blank pIC50: {sid} {target}")
        masks.append(mask)
    if int(row["n_available_regression_tasks"]) != sum(masks):
        raise ValueError(f"n_available_regression_tasks mismatch: {sid}")
    if not any(masks):
        raise ValueError(f"Held-only molecule should not appear in M1.11: {sid}")
    if all(masks):
        a, b = float(row["egfr_pic50"]) > 6, float(row["her2_pic50"]) > 6
        expected = (
            "dual_active" if a and b else "dual_inactive" if not (a or b)
            else "EGFR_only_active" if a else "HER2_only_active"
        )
        if row["paired_activity_class"] != expected:
            raise ValueError(f"Paired classification inconsistent with pIC50: {sid}")
    elif row["paired_activity_class"] not in {"", "not_evaluable_missing_task_label"}:
        raise ValueError(f"Partial-label molecule has a paired class: {sid}")
    return masks[0], masks[1]


def prepare(records: list[dict[str, str]]) -> tuple[dict[str, dict], dict[str, tuple[str, str, str]]]:
    """Group on *structure and topology*, not observed potency or activity."""
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    groups: dict[str, dict] = defaultdict(lambda: {"ids": [], "n": 0, "egfr": 0, "her2": 0, "paired": 0})
    ids: dict[str, tuple[str, str, str]] = {}
    for row in records:
        egfr, her2 = validate_row(row, seen_ids, seen_smiles)
        sid = row["m18c_structure_id"]
        group, scaffold, kind = compute_scaffold(row["canonical_isomeric_smiles"], sid)
        ids[sid] = (group, scaffold, kind)
        g = groups[group]
        g["ids"].append(sid)
        g["n"] += 1
        g["egfr"] += egfr
        g["her2"] += her2
        g["paired"] += egfr * her2
    if len(groups) < 3:
        raise ValueError("At least three independent scaffold groups are needed")
    return dict(groups), ids


def assign_groups(groups: dict[str, dict], seed: int) -> dict[str, str]:
    """Deterministic best-fit balancing of count and *label availability* only.

    No pIC50, activity classification or test performance influences allocation.
    Large scaffold groups go first; ties are shuffled using the recorded seed.
    """
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: groups[key]["n"], reverse=True)
    totals = {k: sum(g[k] for g in groups.values()) for k in ("n", "egfr", "her2", "paired")}
    targets = {part: {k: total * FRACTIONS[part] for k, total in totals.items()} for part in PARTITIONS}
    current = {part: dict.fromkeys(totals, 0) for part in PARTITIONS}
    weights = {"n": 4.0, "egfr": 1.0, "her2": 1.0, "paired": 1.0}
    result: dict[str, str] = {}
    for gid in keys:
        group = groups[gid]
        def delta(part: str, group: dict) -> float:
            # Change in weighted squared normalized deviation from desired counts.
            return sum(
                weights[k] * (
                    ((current[part][k] + group[k] - targets[part][k]) ** 2
                     - (current[part][k] - targets[part][k]) ** 2) / max(targets[part][k], 1.0)
                )
                for k in totals
            )
        partition = min(PARTITIONS, key=lambda part: (delta(part, group), PARTITIONS.index(part)))
        result[gid] = partition
        for k in totals:
            current[partition][k] += group[k]
    if any(current[part]["n"] == 0 for part in PARTITIONS):
        raise ValueError("Unable to obtain nonempty train/validation/test partitions")
    return result


def audit(records: list[dict[str, str]], mapping: dict[str, tuple[str, str, str]],
          groups: dict[str, dict], assignments: dict[str, str]) -> dict[str, object]:
    split_ids: dict[str, set[str]] = {x: set() for x in PARTITIONS}
    split_groups: dict[str, set[str]] = {x: set() for x in PARTITIONS}
    stats: dict[str, Counter] = {x: Counter() for x in PARTITIONS}
    kinds = Counter()
    for row in records:
        sid = row["m18c_structure_id"]
        gid, _, kind = mapping[sid]
        part = assignments[gid]
        split_ids[part].add(sid)
        split_groups[part].add(gid)
        kinds[kind] += 1
        stats[part]["molecules"] += 1
        stats[part]["EGFR_labels"] += int(row["egfr_mask"])
        stats[part]["HER2_labels"] += int(row["her2_mask"])
        if row["egfr_mask"] == "1" and row["her2_mask"] == "1":
            stats[part]["dual_labeled"] += 1
            stats[part][row["paired_activity_class"]] += 1
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            if split_ids[a] & split_ids[b] or split_groups[a] & split_groups[b]:
                raise AssertionError(f"Leaking molecule or scaffold: {a}/{b}")
    if sum(len(x) for x in split_ids.values()) != len(mapping):
        raise AssertionError("Molecule coverage mismatch")
    for part in PARTITIONS:
        if stats[part]["molecules"] == 0 or stats[part]["EGFR_labels"] == 0 or stats[part]["HER2_labels"] == 0:
            raise ValueError(f"Empty target coverage in {part}; adjust grouping strategy")
    return {
        "zero_molecule_id_overlap": True,
        "zero_scaffold_group_id_overlap": True,
        "all_target_masks_preserved": True,
        "total_unique_molecules": len(mapping),
        "total_scaffold_groups": len(groups),
        "largest_group_size": max(g["n"] for g in groups.values()),
        "scaffold_types": dict(sorted(kinds.items())),
        "by_partition": {part: dict(sorted(stats[part].items())) | {"scaffold_groups": len(split_groups[part])} for part in PARTITIONS},
    }


def check_m11_summary(path: Path | None, records: list[dict[str, str]]) -> dict[str, object]:
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    observed = {
        "unique_accepted_structures": len(records),
        "EGFR_labels": sum(row["egfr_mask"] == "1" for row in records),
        "HER2_labels": sum(row["her2_mask"] == "1" for row in records),
        "dual_labeled_structures": sum(row["egfr_mask"] == "1" and row["her2_mask"] == "1" for row in records),
    }
    for key, value in observed.items():
        if obj["analysis"].get(key) != value:
            raise ValueError(f"M1.11 summary mismatch for {key}: expected {obj['analysis'].get(key)}, got {value}")
    source_version = obj.get("inputs", {}).get("rdkit_version_provenance")
    if source_version and source_version != rdBase.rdkitVersion:
        raise ValueError(
            f"RDKit version mismatch: M1.11 used {source_version}; current is {rdBase.rdkitVersion}. "
            "Recreate the locked M1 environment before freezing the split."
        )
    return {"m11_summary_sha256": sha256_file(path), "m11_source_rdkit_version": source_version}


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path: Path, audit_result: dict[str, object]) -> None:
    """Create a descriptive plot from actual split counts, not preview numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = audit_result["by_partition"]
    categories = ("molecules", "EGFR_labels", "HER2_labels", "dual_labeled")
    names = ("Molecules", "EGFR labels", "HER2 labels", "Paired labels")
    x = list(range(len(categories)))
    fig, ax = plt.subplots(figsize=(10, 5.3))
    width = 0.24
    for j, part in enumerate(PARTITIONS):
        heights = [by[part].get(k, 0) for k in categories]
        ax.bar([i + (j - 1) * width for i in x], heights, width=width, label=part.title())
    ax.set_xticks(x, names)
    ax.set_ylabel("Unique molecules or accepted provisional target labels")
    ax.set_title("M2.1 — Scaffold-disjoint partitions (actual local counts)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(input_path: Path, m11_summary: Path | None, dataset: Path, manifest: Path,
        summary: Path, seed: int = 714, figure: Path | None = None) -> dict[str, object]:
    paths = [dataset.resolve(), manifest.resolve(), summary.resolve()]
    if figure is not None:
        paths.append(figure.resolve())
    if len(set(paths)) != len(paths) or input_path.resolve() in paths:
        raise ValueError("Output paths must be distinct and must not overwrite M1.11")
    for path in (dataset, manifest, summary, figure):
        if path is not None and path.exists():
            raise FileExistsError(f"Refusing to overwrite existing/frozen artifact: {path}")
    rows, original_fields = read_csv(input_path)
    source_info = check_m11_summary(m11_summary, rows)
    groups, mapping = prepare(rows)
    assignments = assign_groups(groups, seed)
    quality = audit(rows, mapping, groups, assignments)
    # Do not modify labels, masks, activity classes, or provenance fields.
    output_records = []
    manifest_records = []
    for row in sorted(rows, key=lambda r: r["m18c_structure_id"]):
        sid = row["m18c_structure_id"]
        group, scaffold, kind = mapping[sid]
        part = assignments[group]
        new = dict(row)
        new["training_split"] = part
        new["scaffold_split"] = part  # Both M1 placeholders now indicate the assigned partition.
        new["scaffold_group_id"] = group
        new["murcko_scaffold_smiles"] = scaffold
        new["scaffold_type"] = kind
        output_records.append(new)
        manifest_records.append({
            "m18c_structure_id": sid, "training_split": part,
            "scaffold_group_id": group, "scaffold_type": kind,
            "murcko_scaffold_smiles": scaffold,
            "egfr_mask": row["egfr_mask"], "her2_mask": row["her2_mask"],
        })
    write_csv(dataset, output_records, original_fields + ["scaffold_group_id", "murcko_scaffold_smiles", "scaffold_type"])
    write_csv(manifest, manifest_records, list(MANIFEST_COLS))
    if figure is not None:
        make_figure(figure, quality)
    result = {
        "stage": "M2.1_scaffold_disjoint_split_provisional_assay_labels",
        "policy": POLICY,
        "random_seed": seed,
        "target_fractions": FRACTIONS,
        "rdkit_version": rdBase.rdkitVersion,
        "input": {"path": input_path.name, "sha256": sha256_file(input_path)} | source_info,
        "outputs": {
            "dataset": dataset.as_posix(), "dataset_sha256": sha256_file(dataset),
            "manifest": manifest.as_posix(), "manifest_sha256": sha256_file(manifest),
        },
        "audit": quality,
        "interpretation": {
            "scaffold": "Bemis-Murcko rings and linkers, chirality excluded for grouping",
            "acyclic": "Each empty-Murcko acyclic molecule forms an identity-level group; acyclic split is NOT scaffold OOD",
            "group_balancing": "Only group sizes and availability masks (not pIC50 or activity labels) used",
            "evidence": "All M1.11 target labels remain provisional; no WT assay sequence verification implied",
            "holdout": "Test is frozen for final comparison; no training or tuning on test",
        },
    }
    if figure is not None:
        result["outputs"]["figure"] = figure.as_posix()
        result["outputs"]["figure_sha256"] = sha256_file(figure)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--m11-summary", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=714)
    parser.add_argument("--figure", type=Path, default=None)
    a = parser.parse_args()
    obj = run(a.input, a.m11_summary, a.dataset, a.manifest, a.summary, a.seed, a.figure)
    print(json.dumps(obj["audit"], indent=2))
    print(f"Split manifest SHA-256: {obj['outputs']['manifest_sha256']}")


if __name__ == "__main__":
    main()
