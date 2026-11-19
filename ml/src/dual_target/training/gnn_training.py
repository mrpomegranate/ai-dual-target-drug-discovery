"""Fit-only molecular GNN training with leakage-safe inner scaffold stopping.

This module accepts ONLY graph sequences that the caller selects from a
FitOnlyTuningPipeline fold's `.fit` records. It never opens files, creates
outer splits, consults CV holdouts, or loads reserved target labels.

Protocol: train on inner-fit, choose the epoch on inner-stop scaffolds, then
refit a *fresh* model for that fixed number of epochs on the complete CV-fit
portion before scoring the existing CV holdout in a separate driver.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch_geometric.data import Data

from ..data.gnn_dataset import FoldGraphViews, make_graph_loader
from ..models.gat import MolecularGATv2
from ..models.gnn import MolecularGNN, masked_huber_loss

MolecularRegressor = MolecularGNN | MolecularGATv2


@dataclass(frozen=True)
class GNNTrainingConfig:
    """Prespecified starter training recipe; tune later using fit-only CV."""

    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.10
    architecture: Literal["gine", "gatv2"] = "gine"
    attention_heads: int = 4
    attention_dropout: float = 0.0
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 80
    patience: int = 12
    inner_stop_fraction: float = 0.20  # Fraction of scaffold GROUPS.
    seed: int = 719
    huber_delta: float = 1.0
    grad_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim", "num_layers", "batch_size", "max_epochs", "patience"
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be >= 2")
        if self.architecture not in ("gine", "gatv2"):
            raise ValueError("architecture must be 'gine' or 'gatv2'")
        if (
            isinstance(self.attention_heads, bool)
            or not isinstance(self.attention_heads, int)
            or self.attention_heads < 1
        ):
            raise ValueError("attention_heads must be a positive integer")
        if self.architecture == "gatv2" and self.hidden_dim % self.attention_heads:
            raise ValueError("GATv2 hidden_dim must be divisible by attention_heads")
        if not math.isfinite(self.attention_dropout) or not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not (0 < self.inner_stop_fraction < 1):
            raise ValueError("inner_stop_fraction must be between 0 and 1")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be in [0, 1)")
        for name in ("learning_rate", "huber_delta", "grad_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


DEFAULT_GNN_TRAINING_CONFIG = GNNTrainingConfig()


@dataclass(frozen=True)
class InnerScaffoldSplit:
    train: tuple[Data, ...]
    stop: tuple[Data, ...]
    seed: int


@dataclass(frozen=True)
class PotencyMetrics:
    egfr_count: int
    her2_count: int
    egfr_mae: float
    her2_mae: float

    @property
    def balanced_mae(self) -> float:
        """Primary early-stopping score; lower is better."""
        return (self.egfr_mae + self.her2_mae) / 2.0


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    training_batch_loss: float
    stopping_metrics: PotencyMetrics


@dataclass
class InnerStoppingResult:
    """Best inner-stop model is diagnostic only, NOT the CV-holdout scorer."""

    model: MolecularRegressor
    best_epoch: int
    best_inner_mae: float
    epochs_ran: int
    history: tuple[EpochSummary, ...]
    inner_train_molecules: int
    inner_stop_molecules: int
    inner_train_scaffolds: int
    inner_stop_scaffolds: int


def _identify(graphs: Sequence[Data]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not graphs:
        raise ValueError("Expected a nonempty sequence of fitting graphs")
    ids = tuple(graph.structure_id for graph in graphs)
    scaffolds = tuple(graph.scaffold_group_id for graph in graphs)
    if any(not isinstance(x, str) or not x for x in (*ids, *scaffolds)):
        raise ValueError("Graphs must preserve nonempty molecule/scaffold IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate molecule identity in fitting graphs")
    return ids, scaffolds


def _validated_cv_fit(fold: FoldGraphViews) -> tuple[Data, ...]:
    """Take fit graphs only from the audited M3.2 fold-view interface."""
    if not isinstance(fold, FoldGraphViews):
        raise TypeError("Expected an M3.2 FoldGraphViews, not arbitrary graph data")
    fit = tuple(fold.fit)
    holdout = tuple(fold.holdout)
    fit_ids, fit_scaffolds = _identify(fit)
    holdout_ids, holdout_scaffolds = _identify(holdout)
    if set(fit_ids) & set(holdout_ids):
        raise ValueError("CV fold molecule leakage")
    if set(fit_scaffolds) & set(holdout_scaffolds):
        raise ValueError("CV fold scaffold leakage")
    return fit


def _require_both_targets(graphs: Sequence[Data]) -> None:
    counts = torch.zeros(2, dtype=torch.int64)
    for graph in graphs:
        if tuple(graph.y_mask.shape) != (1, 2) or graph.y_mask.dtype != torch.bool:
            raise ValueError("Every graph must have a boolean [1, 2] y_mask")
        counts += graph.y_mask[0].to(dtype=torch.int64, device="cpu")
    if not torch.all(counts > 0):
        raise ValueError("Both EGFR and HER2 require measured labels in each subset")


def make_inner_scaffold_split(
    cv_fit_graphs: Sequence[Data], *, fraction: float = 0.20, seed: int = 719
) -> InnerScaffoldSplit:
    """Separate stopping scaffolds from *CV-training* graphs only.

    Call this with `select_fold_graphs(...).fit`, never `.holdout`.
    We use a fixed seed, no label-driven reselection of the inner split.
    """
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    graphs = tuple(cv_fit_graphs)
    ids, groups = _identify(graphs)
    if len(set(groups)) < 2:
        raise ValueError("At least two scaffold groups are required")
    train_ix, stop_ix = next(
        GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(
            np.arange(len(graphs)), groups=np.asarray(groups)
        )
    )
    train = tuple(graphs[int(i)] for i in sorted(train_ix))
    stop = tuple(graphs[int(i)] for i in sorted(stop_ix))
    train_ids, train_groups = _identify(train)
    stop_ids, stop_groups = _identify(stop)
    if set(train_ids) & set(stop_ids) or set(train_groups) & set(stop_groups):
        raise ValueError("Inner stopping split has molecule or scaffold leakage")
    if set(train_ids) | set(stop_ids) != set(ids):
        raise ValueError("Inner stopping split does not cover CV-fit records")
    _require_both_targets(train)
    _require_both_targets(stop)
    return InnerScaffoldSplit(train=train, stop=stop, seed=seed)


def evaluate_potency_mae(
    model: MolecularRegressor,
    graphs: Sequence[Data],
    *,
    batch_size: int,
    device: torch.device | str,
) -> PotencyMetrics:
    """Aggregate per-target errors over observed labels, not per-batch means."""
    _require_both_targets(graphs)
    model.eval()
    errors = torch.zeros(2, dtype=torch.float64)
    counts = torch.zeros(2, dtype=torch.int64)
    loader = make_graph_loader(graphs, batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            predictions = model(batch)
            if not torch.isfinite(predictions).all():
                raise ValueError("Nonfinite GNN predictions during evaluation")
            mask = batch.y_mask
            errors += ((predictions - batch.y).abs() * mask).sum(dim=0).double().cpu()
            counts += mask.sum(dim=0).long().cpu()
    return PotencyMetrics(
        egfr_count=int(counts[0]),
        her2_count=int(counts[1]),
        egfr_mae=float(errors[0] / counts[0]),
        her2_mae=float(errors[1] / counts[1]),
    )


def _train_epoch(
    model: MolecularRegressor,
    loader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    huber_delta: float,
    grad_clip_norm: float,
) -> float:
    model.train()
    losses: list[float] = []
    for cpu_batch in loader:
        batch = cpu_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted = model(batch)
        loss = masked_huber_loss(predicted, batch.y, batch.y_mask, delta=huber_delta)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite masked GNN training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(loss.detach().item())
    return float(np.mean(losses))  # Diagnostic average of mini-batch losses.


def _make_model(config: GNNTrainingConfig) -> MolecularRegressor:
    """Instantiate a model without changing the frozen training protocol.

    GINE remains the default for compatibility with previous M3 runs.
    """
    common = {
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    }
    if config.architecture == "gine":
        return MolecularGNN(**common)
    if config.architecture == "gatv2":
        return MolecularGATv2(
            **common,
            heads=config.attention_heads,
            attention_dropout=config.attention_dropout,
        )
    raise ValueError(f"Unsupported graph architecture: {config.architecture}")


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_with_inner_stopping(
    fold: FoldGraphViews,
    *,
    config: GNNTrainingConfig = DEFAULT_GNN_TRAINING_CONFIG,
    device: torch.device | str = "cpu",
) -> InnerStoppingResult:
    """Choose stopping epoch using only `fold.fit`, never `fold.holdout`."""
    cv_fit_graphs = _validated_cv_fit(fold)
    inner = make_inner_scaffold_split(
        cv_fit_graphs, fraction=config.inner_stop_fraction, seed=config.seed
    )
    _seed_torch(config.seed)
    model = _make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = make_graph_loader(
        inner.train, batch_size=config.batch_size, shuffle=True, seed=config.seed
    )
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    history: list[EpochSummary] = []
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        loss = _train_epoch(
            model, loader, optimizer, device=device,
            huber_delta=config.huber_delta, grad_clip_norm=config.grad_clip_norm,
        )
        metrics = evaluate_potency_mae(
            model, inner.stop, batch_size=config.batch_size, device=device
        )
        history.append(EpochSummary(epoch, loss, metrics))
        if metrics.balanced_mae < best_mae:
            best_mae = metrics.balanced_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return InnerStoppingResult(
        model=model,
        best_epoch=best_epoch,
        best_inner_mae=best_mae,
        epochs_ran=len(history),
        history=tuple(history),
        inner_train_molecules=len(inner.train),
        inner_stop_molecules=len(inner.stop),
        inner_train_scaffolds=len({g.scaffold_group_id for g in inner.train}),
        inner_stop_scaffolds=len({g.scaffold_group_id for g in inner.stop}),
    )


def refit_cv_training_graphs(
    fold: FoldGraphViews,
    *,
    epochs: int,
    config: GNNTrainingConfig = DEFAULT_GNN_TRAINING_CONFIG,
    device: torch.device | str = "cpu",
) -> MolecularRegressor:
    """Fit a fresh GNN on all CV-fit graphs for the inner-selected epoch count.

    Callers are responsible for obtaining `epochs` from *inner* stopping.
    No evaluation graphs are used during the refit.
    """
    cv_fit_graphs = _validated_cv_fit(fold)
    _require_both_targets(cv_fit_graphs)
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or not 1 <= epochs <= config.max_epochs
    ):
        raise ValueError("epochs must be a positive inner-selected count <= max_epochs")
    _seed_torch(config.seed)
    model = _make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loader = make_graph_loader(
        cv_fit_graphs, batch_size=config.batch_size, shuffle=True, seed=config.seed
    )
    for _ in range(epochs):
        _train_epoch(
            model, loader, optimizer, device=device,
            huber_delta=config.huber_delta, grad_clip_norm=config.grad_clip_norm,
        )
    return model.eval()
