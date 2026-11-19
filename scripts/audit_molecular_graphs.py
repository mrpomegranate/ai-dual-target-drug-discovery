"""Audit frozen Stage C fit-role molecules as deterministic RDKit/PyG graphs.

This script reconstructs the audited seed-719 assignment and featurizes ONLY
its fitting records. It never featurizes reserved Stage C roles or test data.
Report contains aggregate statistics and provenance, not potency labels.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

import torch
from dual_target.calibration.scaffold_calibration import (
    make_scaffold_calibration_split,
)
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.data.pipeline import (
    FROZEN_STAGE_C_ASSIGNMENT_SHA256,
    build_fit_only_tuning_pipeline,
    load_modeling_pipeline,
    summarize_labels,
)
from dual_target.features.molecular_graph import (
    ATOM_CARDINALITIES,
    ATOM_FEATURE_NAMES,
    BOND_CARDINALITIES,
    BOND_FEATURE_NAMES,
    GRAPH_SCHEMA_VERSION,
    molecule_to_graph,
)
from rdkit import rdBase
from torch_geometric import __version__ as pyg_version

EXPECTED_DATASET_SHA256 = (
    "83a64d02e45ccd9a6310947efcd5cdef9bddc9603edcd55e1d207eb1cafc18b8"
)
EXPECTED_FIT_COUNTS = {
    "molecules": 5364,
    "egfr_labels": 4465,
    "her2_labels": 1596,
    "paired_labels": 697,
}
# Columns with a final OTHER category. Boolean columns and atomic_number
# deliberately do not have an OTHER category.
ATOM_OTHER_COLUMNS = (1, 2, 3, 4, 7, 8)
BOND_OTHER_COLUMNS = (0, 3)


def _distribution(values: list[int]) -> dict[str, int | float] | None:
    """Small deterministic distribution summary without retaining all graphs."""
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(0.95 * len(ordered)) - 1
    return {
        "min": ordered[0],
        "median": median(ordered),
        "mean": round(mean(ordered), 3),
        "p95_nearest_rank": ordered[rank],
        "max": ordered[-1],
    }


def _validate_graph(graph, record: MoleculeRecord) -> None:
    """Fail closed if features, masks, identities, or bonds are inconsistent."""
    if (graph.structure_id, graph.scaffold_group_id, graph.smiles) != (
        record.structure_id,
        record.scaffold_group_id,
        record.smiles,
    ):
        raise ValueError("Molecule identity, scaffold, or SMILES changed")
    if graph.graph_schema_version != GRAPH_SCHEMA_VERSION:
        raise ValueError("Graph feature-schema version changed")
    if graph.x.ndim != 2 or graph.x.shape[1] != len(ATOM_FEATURE_NAMES):
        raise ValueError("Invalid atom feature dimensions")
    if graph.x.shape[0] < 1 or graph.num_nodes != graph.x.shape[0]:
        raise ValueError("Invalid atom count")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("Invalid bond connectivity dimensions")
    if graph.edge_attr.shape != (
        graph.edge_index.shape[1], len(BOND_FEATURE_NAMES)
    ):
        raise ValueError("Invalid bond feature dimensions")
    if graph.x.dtype != torch.long or graph.edge_attr.dtype != torch.long:
        raise ValueError("Categorical graph features must be int64")
    if graph.y.shape != (1, 2) or graph.y_mask.shape != (1, 2):
        raise ValueError("Invalid two-target label/mask dimensions")
    if graph.y_mask.dtype != torch.bool:
        raise ValueError("Target mask is not Boolean")
    expected_mask = [bool(record.egfr_mask), bool(record.her2_mask)]
    if graph.y_mask[0].tolist() != expected_mask:
        raise ValueError("Accepted-label masks changed")
    for index, (mask, target) in enumerate(zip(expected_mask, ("egfr", "her2"))):
        actual = float(graph.y[0, index])
        if not math.isfinite(actual):
            raise ValueError(f"Nonfinite graph {target} label")
        if mask:
            original = getattr(record, f"{target}_pic50")
            if original is None or not math.isclose(
                actual, original, rel_tol=1e-6, abs_tol=1e-5
            ):
                raise ValueError(f"Accepted {target} potency changed")
        elif actual != 0.0:
            raise ValueError(f"Missing {target} label must use placeholder 0.0")

    for features, cardinalities, kind in (
        (graph.x, ATOM_CARDINALITIES, "atom"),
        (graph.edge_attr, BOND_CARDINALITIES, "bond"),
    ):
        for col, size in enumerate(cardinalities):
            if bool(torch.any((features[:, col] < 0) | (features[:, col] >= size))):
                raise ValueError(f"Out-of-vocabulary {kind} feature column {col}")

    if graph.edge_index.numel() and bool(
        torch.any((graph.edge_index < 0) | (graph.edge_index >= graph.num_nodes))
    ):
        raise ValueError("Bond endpoint outside atom index range")
    if graph.edge_index.shape[1] % 2:
        raise ValueError("Directed bonds must appear in reciprocal pairs")
    # The M3.1 featurizer emits consecutive directed pairs for each bond.
    for i in range(0, graph.edge_index.shape[1], 2):
        u, v = graph.edge_index[:, i].tolist()
        reverse_u, reverse_v = graph.edge_index[:, i + 1].tolist()
        if (u, v) != (reverse_v, reverse_u):
            raise ValueError("Bond directions are not reciprocal")
        if not torch.equal(graph.edge_attr[i], graph.edge_attr[i + 1]):
            raise ValueError("Reciprocal bond features differ")


def audit_fit_graphs(
    records: tuple[MoleculeRecord, ...],
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    """Audit graph conversion; failures are recorded and make passed=False.

    Only aggregate statistics are returned; raw potency values are never saved.
    """
    counts = asdict(summarize_labels(records))
    mismatches = []
    for field, expected in (expected_counts or {}).items():
        actual = counts.get(field)
        if actual != expected:
            mismatches.append(f"{field}: expected {expected}, observed {actual}")

    atom_sizes: list[int] = []
    undirected_bond_sizes: list[int] = []
    atomic_numbers: Counter[int] = Counter()
    atom_other: Counter[str] = Counter()
    bond_other: Counter[str] = Counter()
    total_atoms = 0
    total_directed_edges = 0
    failures: list[dict[str, str]] = []
    failed_total = 0
    converted_ids: set[str] = set()

    for record in records:
        try:
            graph = molecule_to_graph(record)
            _validate_graph(graph, record)
            if record.structure_id in converted_ids:
                raise ValueError("Duplicate converted molecule identity")
            converted_ids.add(record.structure_id)
            n_atoms = int(graph.num_nodes)
            n_edges = int(graph.edge_index.shape[1])
            atom_sizes.append(n_atoms)
            undirected_bond_sizes.append(n_edges // 2)
            total_atoms += n_atoms
            total_directed_edges += n_edges
            atomic_numbers.update(graph.x[:, 0].tolist())
            for col in ATOM_OTHER_COLUMNS:
                name = ATOM_FEATURE_NAMES[col]
                atom_other[name] += int(
                    (graph.x[:, col] == ATOM_CARDINALITIES[col] - 1).sum()
                )
            for col in BOND_OTHER_COLUMNS:
                name = BOND_FEATURE_NAMES[col]
                bond_other[name] += int(
                    (graph.edge_attr[:, col] == BOND_CARDINALITIES[col] - 1).sum()
                )
        except (ValueError, TypeError, RuntimeError, IndexError) as exc:
            failed_total += 1
            if len(failures) < 25:
                failures.append(
                    {"structure_id": record.structure_id, "error": str(exc)}
                )

    return {
        "passed": (
            not mismatches and failed_total == 0 and len(converted_ids) == len(records)
        ),
        "schema_version": GRAPH_SCHEMA_VERSION,
        "atom_features": list(ATOM_FEATURE_NAMES),
        "bond_features": list(BOND_FEATURE_NAMES),
        "label_counts": counts,
        "expected_count_mismatches": mismatches,
        "graphs_converted": len(converted_ids),
        "graphs_failed": failed_total,
        "failure_examples_first_25": failures,
        "atom_count_distribution": _distribution(atom_sizes),
        "bond_count_distribution": _distribution(undirected_bond_sizes),
        "single_atom_graphs": sum(n == 1 for n in atom_sizes),
        "edgeless_graphs": sum(n == 0 for n in undirected_bond_sizes),
        "total_atoms": total_atoms,
        "total_directed_edges": total_directed_edges,
        "atomic_number_counts": {
            str(k): v for k, v in sorted(atomic_numbers.items())
        },
        "atom_other_counts": dict(sorted(atom_other.items())),
        "bond_other_counts": dict(sorted(bond_other.items())),
        "atom_other_rates": {
            k: round(v / total_atoms, 6) if total_atoms else None
            for k, v in sorted(atom_other.items())
        },
        "bond_other_rates": {
            k: round(v / total_directed_edges, 6)
            if total_directed_edges else None
            for k, v in sorted(bond_other.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/interim/M2_1_scaffold_split_molecules.csv"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/audits/M3_1_fit_graph_audit.json"),
    )
    args = parser.parse_args()

    source = load_modeling_pipeline(args.dataset, args.manifest, args.summary)
    if source.data.dataset_sha256 != EXPECTED_DATASET_SHA256:
        raise ValueError("Dataset differs from the expected frozen M2.1 SHA-256")

    split = make_scaffold_calibration_split(
        source,
        seed=719,
        calibration_fraction=0.20,
        threshold_fraction=0.20,
        minimum_target_labels=10,
        minimum_dual_per_class=50,
        max_attempts=1,
    )
    fit_only = build_fit_only_tuning_pipeline(source, split, n_splits=5)
    if fit_only.stage_c_assignment_sha256 != FROZEN_STAGE_C_ASSIGNMENT_SHA256:
        raise ValueError("Unexpected Stage C assignment SHA-256")
    report = audit_fit_graphs(
        fit_only.data.train, expected_counts=EXPECTED_FIT_COUNTS
    )
    report["provenance"] = {
        "dataset_sha256": fit_only.data.dataset_sha256,
        "outer_manifest_sha256": fit_only.data.manifest_sha256,
        "stage_c_assignment_sha256": fit_only.stage_c_assignment_sha256,
        "stage_c_seed": fit_only.stage_c_seed,
        "fit_only_cv_folds": len(fit_only.folds),
        "pytorch_version": torch.__version__,
        "pyg_version": pyg_version,
        "rdkit_version": rdBase.rdkitVersion,
    }
    report["fit_only_cv_folds"] = [
        {
            "fold": fold.number,
            "fit_molecules": len(fold.fit),
            "holdout_molecules": len(fold.holdout),
            "fit_scaffolds": len({r.scaffold_group_id for r in fold.fit}),
            "holdout_scaffolds": len({r.scaffold_group_id for r in fold.holdout}),
            "audit": asdict(summarize_labels(fold.holdout)),
        }
        for fold in fit_only.folds
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Audit report: {args.output}")
    print(
        f"Converted: {report['graphs_converted']}/"
        f"{report['label_counts']['molecules']}; "
        f"failures: {report['graphs_failed']}; passed: {report['passed']}"
    )
    if not report["passed"]:
        print("Graph audit FAILED; inspect failure examples and count mismatches.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
