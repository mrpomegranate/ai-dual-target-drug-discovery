"""Edge-aware GATv2 molecular potency regressor for M3.6.

This is an alternative to MolecularGNN (GINE), not a replacement. It uses the
same M3.1 categorical graph schema and produces [batch_size, 2] EGFR/HER2
pIC50 predictions. No splits, training, calibration, or test access live here.

Bond attributes influence attention coefficients through GATv2Conv.edge_dim.
This differs from GINE, which adds bond embeddings to neighbor messages.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

from ..features.molecular_graph import (
    ATOM_CARDINALITIES,
    ATOM_FEATURE_NAMES,
    BOND_CARDINALITIES,
    BOND_FEATURE_NAMES,
)
from .gnn import CategoricalFeatureEncoder


class MolecularGATv2(nn.Module):
    """Shared bond-aware multi-head attention encoder and two potency heads.

    Each GATv2Conv produces hidden_dim features by concatenating `heads`
    attention heads, each of width `hidden_dim // heads`.

    Self-loops represent each atom's own features and receive zero synthetic
    edge features (no invented bond). Residual updates match the GINE starter.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.10,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 2:
            raise ValueError("hidden_dim must be an integer >= 2")
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
            raise ValueError("heads must be a positive integer")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(attention_dropout) or not 0 <= attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.atom_encoder = CategoricalFeatureEncoder(ATOM_CARDINALITIES, hidden_dim)
        self.bond_encoder = CategoricalFeatureEncoder(BOND_CARDINALITIES, hidden_dim)
        self.convs = nn.ModuleList(
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                edge_dim=hidden_dim,
                dropout=attention_dropout,
                add_self_loops=True,
                fill_value=0.0,
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
        """Predict EGFR and HER2 pIC50 for each molecular graph: [B, 2]."""
        if graph_batch.x.ndim != 2 or graph_batch.x.size(1) != len(ATOM_FEATURE_NAMES):
            raise ValueError("Unexpected atom-feature schema")
        if (
            graph_batch.edge_attr.ndim != 2
            or graph_batch.edge_attr.size(1) != len(BOND_FEATURE_NAMES)
            or graph_batch.edge_attr.size(0) != graph_batch.edge_index.size(1)
        ):
            raise ValueError("Unexpected bond-feature schema")
        if graph_batch.edge_index.ndim != 2 or graph_batch.edge_index.size(0) != 2:
            raise ValueError("Unexpected edge-index schema")

        h = self.atom_encoder(graph_batch.x)
        edge_h = (
            self.bond_encoder(graph_batch.edge_attr)
            if graph_batch.edge_attr.size(0)
            else h.new_empty((0, self.hidden_dim))
        )
        for conv in self.convs:
            update = F.relu(conv(h, graph_batch.edge_index, edge_attr=edge_h))
            h = h + F.dropout(update, p=self.dropout, training=self.training)

        batch = getattr(graph_batch, "batch", None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        molecule_h = global_mean_pool(h, batch)
        return torch.cat(
            (self.egfr_head(molecule_h), self.her2_head(molecule_h)), dim=1
        )
