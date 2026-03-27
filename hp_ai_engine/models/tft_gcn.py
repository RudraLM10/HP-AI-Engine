"""
Combined TFT-GCN Prediction Engine for HP AI Engine.

The main model class that orchestrates:
1. GCN spatial encoder → station embeddings
2. TFT temporal model → 3 horizon forecasts
3. Context attention → real-time override on short-term prediction

This is the single entry point for inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from hp_ai_engine.models.context_attention import ContextAttentionOverride
from hp_ai_engine.models.gcn_encoder import SpatialGCNEncoder
from hp_ai_engine.models.tft_model import TFTCore


@dataclass
class PredictionOutput:
    """Container for all prediction engine outputs."""

    short_forecast: torch.Tensor           # [num_stations_in_batch, short_horizon]
    mid_forecast: torch.Tensor             # [num_stations_in_batch, mid_horizon]
    long_forecast: torch.Tensor            # [num_stations_in_batch, long_horizon]
    attention_weights: dict = field(default_factory=dict)
    context_override_info: dict = field(default_factory=dict)
    spatial_embeddings: torch.Tensor | None = None


class TFTGCNPredictor(nn.Module):
    """
    End-to-end TFT-GCN prediction engine.

    Architecture:
        1. GCN encoder processes station graph → per-station spatial embeddings
        2. TFT takes static features + spatial embedding + dynamic series → 3 forecasts
        3. Context attention optionally overrides the short-term forecast

    Args:
        gcn_in_channels: Input dimension for GCN node features.
        gcn_hidden_dim: GCN hidden/output embedding dimension.
        gcn_num_layers: Number of GCN layers.
        gcn_dropout: GCN dropout rate.
        num_static_features: Number of static covariates for TFT.
        num_dynamic_features: Number of dynamic past features for TFT.
        num_future_features: Number of known future features for TFT.
        tft_hidden_size: TFT hidden dimension.
        tft_num_heads: TFT attention heads.
        tft_lstm_layers: Number of LSTM layers.
        tft_dropout: TFT dropout rate.
        short_horizon: Short forecast horizon (hours).
        mid_horizon: Mid forecast horizon (hours).
        long_horizon: Long forecast horizon (hours).
        lookback_hours: Past lookback window (hours).
        num_context_signals: Number of real-time context signals.
        deviation_threshold: Context attention z-score threshold.
        enable_context_override: Whether to apply context attention.
    """

    def __init__(
        self,
        gcn_in_channels: int = 8,
        gcn_hidden_dim: int = 64,
        gcn_num_layers: int = 2,
        gcn_dropout: float = 0.1,
        num_static_features: int = 2,
        num_dynamic_features: int = 8,
        num_future_features: int = 8,
        tft_hidden_size: int = 128,
        tft_num_heads: int = 4,
        tft_lstm_layers: int = 2,
        tft_dropout: float = 0.1,
        short_horizon: int = 6,
        mid_horizon: int = 168,
        long_horizon: int = 4320,
        lookback_hours: int = 168,
        num_context_signals: int = 3,
        deviation_threshold: float = 2.0,
        enable_context_override: bool = True,
    ):
        super().__init__()

        self.enable_context_override = enable_context_override

        # GCN spatial encoder
        self.gcn = SpatialGCNEncoder(
            in_channels=gcn_in_channels,
            hidden_dim=gcn_hidden_dim,
            num_layers=gcn_num_layers,
            dropout=gcn_dropout,
            use_residual=True,
            use_batch_norm=True,
        )

        # TFT temporal model
        self.tft = TFTCore(
            num_static_features=num_static_features,
            num_dynamic_features=num_dynamic_features,
            num_future_features=num_future_features,
            spatial_embedding_dim=gcn_hidden_dim,
            hidden_size=tft_hidden_size,
            num_attention_heads=tft_num_heads,
            lstm_layers=tft_lstm_layers,
            dropout=tft_dropout,
            short_horizon=short_horizon,
            mid_horizon=mid_horizon,
            long_horizon=long_horizon,
            lookback_hours=lookback_hours,
        )

        # Context attention override (applied to short-term forecast only)
        if enable_context_override:
            self.context_attention = ContextAttentionOverride(
                num_signals=num_context_signals,
                deviation_threshold=deviation_threshold,
            )
        else:
            self.context_attention = None

    def forward(
        self,
        # Graph inputs
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        # Per-sample TFT inputs
        static_features: torch.Tensor | None = None,
        dynamic_past: torch.Tensor | None = None,
        dynamic_future: torch.Tensor | None = None,
        station_indices: torch.Tensor | None = None,
        # Context attention inputs (optional)
        current_signals: torch.Tensor | None = None,
        baseline_means: torch.Tensor | None = None,
        baseline_stds: torch.Tensor | None = None,
    ) -> PredictionOutput:
        """
        Full forward pass through the TFT-GCN prediction engine.

        Args:
            node_features: [num_nodes, gcn_in_channels] — features for all stations.
            edge_index: [2, num_edges] — graph connectivity.
            edge_weight: [num_edges] — edge weights.
            static_features: [batch, num_static_features] — per-station static features.
            dynamic_past: [batch, lookback, num_dynamic] — historical time series.
            dynamic_future: [batch, max_horizon, num_future] — known future features.
            station_indices: [batch] — index into the graph node list per sample.
            current_signals: [batch, num_signals] — real-time context signals.
            baseline_means: [batch, num_signals] — historical baseline means.
            baseline_stds: [batch, num_signals] — historical baseline stds.

        Returns:
            PredictionOutput with forecasts, attention info, and context override info.
        """
        # --- Step 1: GCN spatial encoding ---
        spatial_embeddings = self.gcn(node_features, edge_index, edge_weight)
        # spatial_embeddings shape: [num_nodes, gcn_hidden_dim]

        # Select embeddings for the stations in the current batch
        if station_indices is not None:
            batch_spatial = spatial_embeddings[station_indices]  # [batch, gcn_hidden_dim]
        else:
            batch_spatial = spatial_embeddings

        # --- Step 2: TFT temporal forecasting ---
        short_pred, mid_pred, long_pred, tft_info = self.tft(
            static_features=static_features,
            dynamic_past=dynamic_past,
            dynamic_future=dynamic_future,
            spatial_embedding=batch_spatial,
        )

        # --- Step 3: Context attention override (short-term only) ---
        context_info = {}
        if (
            self.enable_context_override
            and self.context_attention is not None
            and current_signals is not None
            and baseline_means is not None
            and baseline_stds is not None
        ):
            short_pred, context_info = self.context_attention(
                tft_prediction=short_pred,
                current_signals=current_signals,
                baseline_means=baseline_means,
                baseline_stds=baseline_stds,
            )

        return PredictionOutput(
            short_forecast=short_pred,
            mid_forecast=mid_pred,
            long_forecast=long_pred,
            attention_weights=tft_info,
            context_override_info=context_info,
            spatial_embeddings=batch_spatial,
        )
