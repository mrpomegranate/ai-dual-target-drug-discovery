"""Small, bond-aware multi-task molecular GNN for the audited M3 graph schema.

This module defines the model and masked regression objective only. It does NOT
select training/validation data, create folds, set stopping epochs, or evaluate
reserved partitions. Those remain the responsibility of the fit-only pipeline.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_mean_pool

from ..features.molecular_graph import (
    ATOM_CARDINALITIES,
    ATOM_FEATURE_NAMES,
    BOND_CARDINALITIES,
    BOND_FEATURE_NAMES,
)


class CategoricalFeatureEncoder(nn.Module):
    """Sum one embedding per fixed-vocabulary categorical feature column."""

    def __init__(self, cardinalities: tuple[int, ...], hidden_dim: int) -> None:
        super().__init__()
        if not cardinalities or any(n < 2 for n in cardinalities):
            raise ValueError("Feature cardinalities must all be >= 2")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        self.cardinalities = cardinalities
        self.embeddings = nn.ModuleList(
            nn.Embedding(n, hidden_dim) for n in cardinalities
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.size(1) != len(self.cardinalities):
            raise ValueError(
                f"Expected categorical features [N, {len(self.cardinalities)}]"
            )
        if features.dtype != torch.long:
            raise TypeError("Categorical graph features must have torch.long dtype")
        if features.size(0) == 0:
            raise ValueError("Empty categorical feature matrix")
        for i, cardinality in enumerate(self.cardinalities):
            if (features[:, i] < 0).any() or (features[:, i] >= cardinality).any():
                raise ValueError(f"Feature column {i} contains out-of-range indices")
        return torch.stack(
            [embedding(features[:, i]) for i, embedding in enumerate(self.embeddings)]
        ).sum(dim=0)


class MolecularGNN(nn.Module):
    """Shared edge-aware graph encoder with separate EGFR/HER2 pIC50 heads."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or hidden_dim < 2:
            raise ValueError("hidden_dim must be >= 2")
        if isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.atom_encoder = CategoricalFeatureEncoder(ATOM_CARDINALITIES, hidden_dim)
        self.bond_encoder = CategoricalFeatureEncoder(BOND_CARDINALITIES, hidden_dim)
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                train_eps=True,
            )
            for _ in range(num_layers)
        )

        head_width = max(hidden_dim // 2, 1)

        def make_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, head_width),
                nn.ReLU(),
                nn.Linear(head_width, 1),
            )

        self.egfr_head = make_head()
        self.her2_head = make_head()

    def forward(self, graph_batch: Data | Batch) -> Tensor:
        """Return two continuous pIC50 predictions per molecular graph: [B, 2]."""
        if graph_batch.x.ndim != 2 or graph_batch.x.size(1) != len(ATOM_FEATURE_NAMES):
            raise ValueError("Unexpected atom-feature schema")
        if (
            graph_batch.edge_attr.ndim != 2
            or graph_batch.edge_attr.size(1) != len(BOND_FEATURE_NAMES)
            or graph_batch.edge_attr.size(0) != graph_batch.edge_index.size(1)
        ):
            raise ValueError("Unexpected bond-feature schema")

        h = self.atom_encoder(graph_batch.x)
        edge_h = self.bond_encoder(graph_batch.edge_attr) if graph_batch.edge_attr.size(0) else h.new_empty((0, self.hidden_dim))
        for conv in self.convs:
            update = F.relu(conv(h, graph_batch.edge_index, edge_attr=edge_h))
            h = h + F.dropout(update, p=self.dropout, training=self.training)

        batch = getattr(graph_batch, "batch", None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        molecule_h = global_mean_pool(h, batch)
        return torch.cat((self.egfr_head(molecule_h), self.her2_head(molecule_h)), dim=1)


def masked_huber_loss(
    predictions: Tensor,
    targets: Tensor,
    masks: Tensor,
    *,
    delta: float = 1.0,
    task_weights: tuple[float, float] = (1.0, 1.0),
) -> Tensor:
    """Mean Huber over observed labels per target, then normalized task weighting.

    With both targets observed within a batch, default weights yield
    0.5 * mean_Huber(EGFR) + 0.5 * mean_Huber(HER2).
    If a target has no labels in a batch, use only available target(s).
    Callers must log target counts and preserve the full fit-only sampling
    protocol; mini-batch absence does not represent missing-label negatives.
    """
    if predictions.ndim != 2 or predictions.shape != targets.shape or predictions.shape != masks.shape or predictions.size(1) != 2:
        raise ValueError("predictions, targets and masks must all be [B, 2]")
    if masks.dtype != torch.bool:
        raise TypeError("masks must be boolean")
    if predictions.size(0) == 0 or not torch.any(masks):
        raise ValueError("At least one observed label is required")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    if len(task_weights) != 2 or any((not math.isfinite(w)) or w <= 0 for w in task_weights):
        raise ValueError("Both task weights must be positive and finite")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("Prediction and target tensors must contain finite values")

    per_cell = F.huber_loss(predictions, targets, reduction="none", delta=delta)
    valid = masks.to(dtype=predictions.dtype)
    counts = valid.sum(dim=0)
    per_target = (per_cell * valid).sum(dim=0) / counts.clamp(min=1)
    weights = predictions.new_tensor(task_weights) * (counts > 0).to(predictions.dtype)
    return (per_target * weights).sum() / weights.sum()
