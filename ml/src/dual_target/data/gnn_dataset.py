"""Stage-C fit-only PyG graph bank and scaffold-CV batch views for M3.

Accepts only an already-audited FitOnlyTuningPipeline. No file IO, split
creation, calibration/threshold data, development validation, or locked test.
Do not use CV holdout graphs to select early-stopping epochs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..features.molecular_graph import GRAPH_SCHEMA_VERSION, molecule_to_graph
from .modeling_dataset import MoleculeRecord
from .pipeline import (
    FROZEN_STAGE_C_ASSIGNMENT_SHA256,
    FitOnlyTuningPipeline,
)

FROZEN_STAGE_C_SEED = 719


@dataclass(frozen=True)
class FitOnlyGraphBank:
    """CPU graphs indexed by frozen molecule ID, in source fitting order."""

    ordered_ids: tuple[str, ...]
    graphs_by_id: Mapping[str, Data]
    records_by_id: Mapping[str, MoleculeRecord]
    dataset_sha256: str
    manifest_sha256: str
    stage_c_assignment_sha256: str
    graph_schema_version: str

    @property
    def molecules(self) -> int:
        return len(self.ordered_ids)


@dataclass(frozen=True)
class FoldGraphViews:
    """References to the existing fit/holdout graphs for one scaffold CV fold."""

    number: int
    fit: tuple[Data, ...]
    holdout: tuple[Data, ...]


def _validate_fit_only_view(pipeline: FitOnlyTuningPipeline) -> None:
    """Check the fit-only boundary and all preexisting fold partitions."""
    if not isinstance(pipeline, FitOnlyTuningPipeline):
        raise TypeError("Requires FitOnlyTuningPipeline, not unrestricted data")
    if pipeline.stage_c_assignment_sha256 != FROZEN_STAGE_C_ASSIGNMENT_SHA256:
        raise ValueError("Stage C assignment hash differs from frozen seed-719 split")
    if pipeline.stage_c_seed != FROZEN_STAGE_C_SEED:
        raise ValueError("Unexpected Stage C seed")
    if not pipeline.data.dataset_sha256 or not pipeline.data.manifest_sha256:
        raise ValueError("Missing frozen dataset or manifest provenance")

    records = tuple(pipeline.data.train)
    ids = [row.structure_id for row in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Fit-only records are empty or contain repeated IDs")
    by_id = {row.structure_id: row for row in records}
    all_ids = set(by_id)
    held_out_once: set[str] = set()
    fold_numbers: set[int] = set()
    if not pipeline.folds:
        raise ValueError("Fit-only pipeline has no scaffold folds")

    for fold in pipeline.folds:
        if fold.number in fold_numbers:
            raise ValueError("Duplicate scaffold fold number")
        fold_numbers.add(fold.number)
        fit_ids = [row.structure_id for row in fold.fit]
        holdout_ids = [row.structure_id for row in fold.holdout]
        if not fit_ids or not holdout_ids:
            raise ValueError(f"Empty fit/holdout in fold {fold.number}")
        if len(fit_ids) != len(set(fit_ids)) or len(holdout_ids) != len(set(holdout_ids)):
            raise ValueError(f"Duplicate molecule within fold {fold.number}")
        fit_set, holdout_set = set(fit_ids), set(holdout_ids)
        if fit_set & holdout_set or fit_set | holdout_set != all_ids:
            raise ValueError(f"Fold {fold.number} does not partition fit-only records")
        if held_out_once & holdout_set:
            raise ValueError("Molecule occurs in more than one CV holdout")
        held_out_once.update(holdout_set)
        if {r.scaffold_group_id for r in fold.fit} & {
            r.scaffold_group_id for r in fold.holdout
        }:
            raise ValueError(f"Scaffold overlap in fold {fold.number}")
        for row in (*fold.fit, *fold.holdout):
            if row != by_id[row.structure_id]:
                raise ValueError("Fold record differs from frozen fit-only record")

    if held_out_once != all_ids:
        raise ValueError("Fit-only CV holdouts do not cover all fitting molecules")


def build_fit_only_graph_bank(pipeline: FitOnlyTuningPipeline) -> FitOnlyGraphBank:
    """Featurize fitting molecules once; never transform reserved records."""
    _validate_fit_only_view(pipeline)
    records = tuple(pipeline.data.train)
    graph_map: dict[str, Data] = {}
    record_map: dict[str, MoleculeRecord] = {}
    for record in records:
        graph = molecule_to_graph(record)
        if graph.structure_id != record.structure_id:
            raise ValueError("Graph molecule identity changed")
        if graph.scaffold_group_id != record.scaffold_group_id:
            raise ValueError("Graph scaffold identity changed")
        if graph.graph_schema_version != GRAPH_SCHEMA_VERSION:
            raise ValueError("Graph feature schema changed")
        if graph.y_mask.shape != (1, 2) or graph.y_mask[0].tolist() != [
            bool(record.egfr_mask), bool(record.her2_mask)
        ]:
            raise ValueError("Graph measurement masks differ from source")
        graph_map[record.structure_id] = graph
        record_map[record.structure_id] = record

    return FitOnlyGraphBank(
        ordered_ids=tuple(row.structure_id for row in records),
        graphs_by_id=MappingProxyType(graph_map),
        records_by_id=MappingProxyType(record_map),
        dataset_sha256=pipeline.data.dataset_sha256,
        manifest_sha256=pipeline.data.manifest_sha256,
        stage_c_assignment_sha256=pipeline.stage_c_assignment_sha256,
        graph_schema_version=GRAPH_SCHEMA_VERSION,
    )


def _select_graphs(
    bank: FitOnlyGraphBank, records: Sequence[MoleculeRecord]
) -> tuple[Data, ...]:
    graphs: list[Data] = []
    for record in records:
        original = bank.records_by_id.get(record.structure_id)
        if original is None or original != record:
            raise ValueError("Requested graph is not an identical fit-only record")
        graphs.append(bank.graphs_by_id[record.structure_id])
    return tuple(graphs)


def select_fold_graphs(
    pipeline: FitOnlyTuningPipeline, bank: FitOnlyGraphBank, fold_number: int
) -> FoldGraphViews:
    """Select the existing fold's graph references, preserving record order.

    The CV holdout is for scoring, NEVER for early stopping or tuning epochs.
    """
    _validate_fit_only_view(pipeline)
    if (
        bank.dataset_sha256 != pipeline.data.dataset_sha256
        or bank.manifest_sha256 != pipeline.data.manifest_sha256
        or bank.stage_c_assignment_sha256 != pipeline.stage_c_assignment_sha256
        or bank.ordered_ids != tuple(r.structure_id for r in pipeline.data.train)
    ):
        raise ValueError("Graph bank provenance does not match fit-only pipeline")
    matching = [fold for fold in pipeline.folds if fold.number == fold_number]
    if len(matching) != 1:
        raise ValueError(f"Unknown scaffold CV fold: {fold_number}")
    fold = matching[0]
    return FoldGraphViews(
        number=fold.number,
        fit=_select_graphs(bank, fold.fit),
        holdout=_select_graphs(bank, fold.holdout),
    )


def make_graph_loader(
    graphs: Sequence[Data],
    *,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = 714,
) -> DataLoader:
    """Batch variable-size graphs on CPU; move only batches to GPU later.

    Use num_workers=0 for a simple, reproducible Windows starting point.
    """
    if not graphs:
        raise ValueError("Cannot batch an empty graph collection")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = torch.Generator().manual_seed(seed)
    return DataLoader(
        list(graphs), batch_size=batch_size, shuffle=shuffle,
        num_workers=0, generator=rng,
    )
