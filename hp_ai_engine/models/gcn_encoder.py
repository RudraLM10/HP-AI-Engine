"""
Spatial GCN Encoder for HP AI Engine.

Performs spectral graph convolution over the station network to produce
spatially enriched node embeddings. Each station's embedding encodes
the current state of its neighbourhood — nearby stations' activity
patterns influence its representation.

Architecture:
    Input:  per-station feature vector x_i
    N × GCNConv layers with ReLU, BatchNorm, Dropout, and residual connections
    Output: node embedding z_i of shape [hidden_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SpatialGCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder for the station spatial graph.

    Message passing on the station graph produces embeddings that
    capture inter-station spatial dependencies (demand spillover,
    shared catchment effects, tanker route proximity).

    Args:
        in_channels: Dimension of input node features.
        hidden_dim: Dimension of hidden layers and output embedding.
        num_layers: Number of GCN layers. Default 2.
        dropout: Dropout rate. Default 0.1.
        use_residual: Whether to add residual connections (for num_layers > 1).
        use_batch_norm: Whether to apply batch normalisation between layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm

        # Build GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer: in_channels -> hidden_dim
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity())
        self.dropouts.append(nn.Dropout(dropout))

        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
            )
            self.dropouts.append(nn.Dropout(dropout))

        # Projection layer for residual if input dim != hidden_dim
        self.input_proj = None
        if use_residual and in_channels != hidden_dim:
            self.input_proj = nn.Linear(in_channels, hidden_dim)

        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through the GCN encoder.

        Args:
            x: Node feature matrix [num_nodes, in_channels].
            edge_index: Graph connectivity [2, num_edges].
            edge_weight: Edge weights [num_edges] (from adjacency matrix).

        Returns:
            Node embeddings [num_nodes, hidden_dim].
        """
        # Project input for residual connection if dimensions differ
        residual = self.input_proj(x) if self.input_proj is not None else x

        h = x
        for i in range(self.num_layers):
            h_in = h

            # Graph convolution
            h = self.convs[i](h, edge_index, edge_weight=edge_weight)

            # Batch normalisation
            h = self.batch_norms[i](h)

            # Activation
            h = self.activation(h)

            # Dropout
            h = self.dropouts[i](h)

            # Residual connection (skip from previous layer or input)
            if self.use_residual:
                if i == 0:
                    h = h + residual
                else:
                    h = h + h_in

        return h

    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.convs[-1].out_channels
