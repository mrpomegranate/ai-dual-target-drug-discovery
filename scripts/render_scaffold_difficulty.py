"""Draw representative chemistry for top fit-only OOF scaffold groups.

Uses ONLY frozen Stage C fit-role records, not calibration, threshold,
development validation or locked-test labels. A representative Murcko core
is a visual example of the group, not an assertion of homogeneous chemistry.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold


def core_from_smiles(smiles: str) -> tuple[str, bool]:
    """Get a displayable core; acyclic molecules use their full SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid molecule SMILES in accepted fit-only record: {smiles}")
    core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if core:
        return Chem.MolToSmiles(Chem.MolFromSmiles(core), canonical=True), False
    return Chem.MolToSmiles(mol, canonical=True), True


def representative_cores(fit_records, needed_scaffolds: set[str]) -> dict[str, dict]:
    """Choose the most frequent core within each preexisting frozen group."""
    groups: dict[str, list[tuple[str, str, bool]]] = {}
    for record in fit_records:
        if record.scaffold_group_id in needed_scaffolds:
            core, acyclic = core_from_smiles(record.smiles)
            groups.setdefault(record.scaffold_group_id, []).append(
                (record.structure_id, core, acyclic)
            )
    if set(groups) != needed_scaffolds:
        raise ValueError("Some ranked scaffold IDs are absent from the frozen fit role")
    result = {}
    for scaffold_id, members in groups.items():
        frequency = Counter(core for _, core, _ in members)
        chosen = min(frequency, key=lambda x: (-frequency[x], x))
        representative_id = min(sid for sid, core, _ in members if core == chosen)
        result[scaffold_id] = {
            "representative_core_smiles": chosen,
            "representative_structure_id": representative_id,
            "fit_molecules_in_scaffold": len(members),
            "distinct_murcko_cores_in_group": len(frequency),
            "acyclic_identity_display": all(acyclic for _, _, acyclic in members),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    from dual_target.calibration.scaffold_calibration import (
        make_scaffold_calibration_split,
    )
    from dual_target.data.pipeline import (
        FROZEN_STAGE_C_ASSIGNMENT_SHA256,
        load_modeling_pipeline,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, default=Path(
        "reports/milestones/M3_7a_fit_only_scaffold_difficulty_top.csv"
    ))
    parser.add_argument("--dataset", type=Path, default=Path(
        "data/interim/M2_1_scaffold_split_molecules.csv"
    ))
    parser.add_argument("--manifest", type=Path, default=Path(
        "data/interim/M2_1_scaffold_split_manifest.csv"
    ))
    parser.add_argument("--summary", type=Path, default=Path(
        "reports/audits/M2_1_scaffold_split_summary.json"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/milestones"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    with args.ranking.open("r", encoding="utf-8-sig", newline="") as stream:
        ranking = list(csv.DictReader(stream))
    if not ranking or not {"target", "scaffold_group_id", "n", "rf_mae"}.issubset(ranking[0]):
        raise ValueError("Expected nonempty, verified M3.7a scaffold ranking CSV")
    if any(r["target"] not in ("egfr", "her2") for r in ranking):
        raise ValueError("Unexpected target in scaffold ranking")
    keyset = [(r["target"], r["scaffold_group_id"]) for r in ranking]
    if len(keyset) != len(set(keyset)):
        raise ValueError("Duplicate scaffold and target in ranking")

    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    split = make_scaffold_calibration_split(
        source, seed=719, calibration_fraction=.20, threshold_fraction=.20,
        minimum_target_labels=10, minimum_dual_per_class=50, max_attempts=1,
    )
    if split.assignment_sha256 != FROZEN_STAGE_C_ASSIGNMENT_SHA256:
        raise ValueError("Stage C assignment changed")
    if len(split.fit) != 5364:
        raise ValueError("Unexpected Stage C fit-only molecule count")
    lookup = representative_cores(
        split.fit, {r["scaffold_group_id"] for r in ranking}
    )
    results = [{**row, **lookup[row["scaffold_group_id"]]} for row in ranking]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / "M3_7a_fit_only_scaffold_structures.csv"]
    outputs += [args.output_dir / f"M3_7a_fit_only_scaffold_cores_{t}.png"
                for t in ("egfr", "her2")]
    if not args.overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Structure report already exists; pass --overwrite")
    with outputs[0].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for target, output in zip(("egfr", "her2"), outputs[1:], strict=True):
        subset = [row for row in results if row["target"] == target]
        mols = [Chem.MolFromSmiles(row["representative_core_smiles"]) for row in subset]
        if not all(mols):
            raise ValueError("Unable to depict at least one selected scaffold core")
        labels = [
            f"{row['scaffold_group_id'][-9:]} | n={row['n']} | RF MAE={float(row['rf_mae']):.2f}"
            for row in subset
        ]
        rendered = Draw.MolsToGridImage(
            mols, molsPerRow=3, subImgSize=(360, 270), legends=labels,
            useSVG=False,
        )
        rendered.save(str(output))
        print(f"Saved {output}")
    print(f"Saved {outputs[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
