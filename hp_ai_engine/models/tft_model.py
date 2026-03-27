"""
Temporal Fusion Transformer (TFT) for HP AI Engine.

A custom implementation of the TFT architecture for multi-horizon CNG demand
forecasting. The key components are:

1. Variable Selection Network (VSN) — learns which input features matter most
2. Gated Residual Network (GRN) — the fundamental building block
3. LSTM Encoder-Decoder — captures sequential temporal patterns
4. Interpretable Multi-Head Attention — attends to the full lookback window
5. Three Decoder Heads — simultaneous forecasts at 6h, 7d, and 6mo horizons

The GCN spatial embedding is injected as an additional static covariate.

Reference: Bryan Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", 2021.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gated Residual Network (GRN) — fundamental building block
# ---------------------------------------------------------------------------

class GatedResidualNetwork(nn.Module):
    """
    GRN block: the core computation unit of TFT.

    Architecture:
        input -> Linear -> ELU -> Linear -> GLU Gate -> LayerNorm + Residual

    If context is provided, it is added after the first linear layer,
    enabling static enrichment of temporal representations.

    Args:
        input_dim: Dimension of the input.
        hidden_dim: Dimension of the hidden layer.
        output_dim: Dimension of the output.
        context_dim: Optional context vector dimension (for static enrichment).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * 2)  # ×2 for GLU gate
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim else None
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Skip connection projection if dimensions differ
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., input_dim].
            context: Optional context tensor [..., context_dim].

        Returns:
            Output tensor [..., output_dim].
        """
        residual = self.skip_proj(x) if self.skip_proj is not None else x

        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)
        h = self.dropout(h)

        # GLU gate
        h = self.fc2(h)
        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)

        # Residual + LayerNorm
        return self.layer_norm(h + residual)


# ---------------------------------------------------------------------------
# Variable Selection Network (VSN)
# ---------------------------------------------------------------------------

class VariableSelectionNetwork(nn.Module):
    """
    Learns which input features matter most via feature-wise GRN processing
    followed by softmax attention across variables.

    Args:
        num_variables: Number of input variables.
        input_dim: Dimension of each variable's embedding.
        hidden_dim: Hidden dimension for GRNs.
        context_dim: Optional static context dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_variables: int,
        input_dim: int,
        hidden_dim: int,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_variables = num_variables

        # Per-variable GRNs
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_variables)
        ])

        # Flattened GRN for computing variable weights
        self.weight_grn = GatedResidualNetwork(
            num_variables * input_dim, hidden_dim, num_variables,
            context_dim=context_dim, dropout=dropout,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Tensor [..., num_variables, input_dim].
            context: Optional context tensor [..., context_dim].

        Returns:
            selected: Weighted combination [..., hidden_dim].
            weights: Variable importance weights [..., num_variables].
        """
        # Process each variable through its own GRN
        variable_outputs = []
        for i in range(self.num_variables):
            var_input = inputs[..., i, :]
            var_output = self.variable_grns[i](var_input)
            variable_outputs.append(var_output)

        # Stack: [..., num_variables, hidden_dim]
        stacked = torch.stack(variable_outputs, dim=-2)

        # Compute variable importance weights
        flat_inputs = inputs.flatten(start_dim=-2)  # [..., num_variables * input_dim]
        weights = self.weight_grn(flat_inputs, context)
        weights = F.softmax(weights, dim=-1)  # [..., num_variables]

        # Weighted combination
        selected = (stacked * weights.unsqueeze(-1)).sum(dim=-2)

        return selected, weights


# ---------------------------------------------------------------------------
# Interpretable Multi-Head Attention
# ---------------------------------------------------------------------------

class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention where attention weights are averaged across heads
    to produce a single interpretable attention vector.

    Unlike standard multi-head attention, this uses additive attention
    within each head for better interpretability.

    Args:
        hidden_dim: Dimension of input/output.
        num_heads: Number of attention heads.
        dropout: Dropout rate on attention weights.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: [batch, q_len, hidden_dim]
            keys: [batch, k_len, hidden_dim]
            values: [batch, v_len, hidden_dim]
            mask: Optional attention mask [batch, q_len, k_len].

        Returns:
            output: [batch, q_len, hidden_dim]
            attention_weights: [batch, q_len, k_len] (averaged across heads)
        """
        batch = queries.size(0)

        # Project and reshape to [batch, num_heads, seq_len, head_dim]
        q = self.q_proj(queries).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(values).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)

        # Average attention weights across heads for interpretability
        avg_attn_weights = attn_weights.mean(dim=1)

        return output, avg_attn_weights


# ---------------------------------------------------------------------------
# TFT Core Model
# ---------------------------------------------------------------------------

class TFTCore(nn.Module):
    """
    Full Temporal Fusion Transformer architecture.

    Pipeline:
    1. Static covariate encoder (produces context vectors for enrichment)
    2. Variable Selection Networks (past inputs and known future inputs)
    3. LSTM encoder → LSTM decoder
    4. Static enrichment (blend static context into temporal representations)
    5. Temporal self-attention over decoder outputs
    6. Position-wise feed-forward
    7. Three decoder heads:
       - short_head: forecasts 0–6 hours
       - mid_head: forecasts 1–7 days
       - long_head: forecasts 1–6 months

    The spatial_embedding from GCN is concatenated to static covariates
    before the static covariate encoder.

    Args:
        num_static_features: Number of static input features.
        num_dynamic_features: Number of dynamic past features.
        num_future_features: Number of known future features.
        spatial_embedding_dim: Dimension of GCN spatial embedding.
        hidden_size: Main hidden dimension. Default 128.
        num_attention_heads: Attention heads. Default 4.
        lstm_layers: LSTM layers. Default 2.
        dropout: Dropout rate. Default 0.1.
        short_horizon: Short forecast horizon. Default 6.
        mid_horizon: Mid forecast horizon. Default 168.
        long_horizon: Long forecast horizon. Default 4320.
        lookback_hours: Past lookback window. Default 168.
    """

    def __init__(
        self,
        num_static_features: int,
        num_dynamic_features: int,
        num_future_features: int,
        spatial_embedding_dim: int = 64,
        hidden_size: int = 128,
        num_attention_heads: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        short_horizon: int = 6,
        mid_horizon: int = 168,
        long_horizon: int = 4320,
        lookback_hours: int = 168,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.short_horizon = short_horizon
        self.mid_horizon = mid_horizon
        self.long_horizon = long_horizon
        self.lookback_hours = lookback_hours

        # -- Static covariate encoder --
        static_input_dim = num_static_features + spatial_embedding_dim
        self.static_encoder = GatedResidualNetwork(
            static_input_dim, hidden_size, hidden_size, dropout=dropout
        )
        # Static context vectors for different uses
        self.static_context_variable = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )
        self.static_context_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )
        self.static_context_state_h = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )
        self.static_context_state_c = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )

        # -- Input embeddings --
        self.past_input_proj = nn.Linear(num_dynamic_features, hidden_size)
        self.future_input_proj = nn.Linear(num_future_features, hidden_size)

        # -- Variable Selection Networks --
        self.past_vsn = VariableSelectionNetwork(
            num_variables=1,  # treating entire past vector as one variable for simplicity
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            context_dim=hidden_size,
            dropout=dropout,
        )
        self.future_vsn = VariableSelectionNetwork(
            num_variables=1,
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            context_dim=hidden_size,
            dropout=dropout,
        )

        # -- LSTM encoder-decoder --
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # -- Post-LSTM gate + norm --
        self.post_lstm_gate = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )

        # -- Static enrichment --
        self.static_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size,
            context_dim=hidden_size, dropout=dropout,
        )

        # -- Temporal self-attention --
        self.attention = InterpretableMultiHeadAttention(
            hidden_size, num_attention_heads, dropout=dropout,
        )
        self.post_attention_gate = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout,
        )

        # -- Position-wise feed-forward --
        self.ff = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout,
        )

        # -- Three decoder heads (independent projections) --
        self.short_head = nn.Linear(hidden_size, short_horizon)
        self.mid_head = nn.Linear(hidden_size, mid_horizon)
        self.long_head = nn.Linear(hidden_size, long_horizon)

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_past: torch.Tensor,
        dynamic_future: torch.Tensor,
        spatial_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass through the TFT.

        Args:
            static_features: [batch, num_static_features]
            dynamic_past: [batch, lookback_hours, num_dynamic_features]
            dynamic_future: [batch, max_horizon, num_future_features]
            spatial_embedding: [batch, spatial_embedding_dim] from GCN

        Returns:
            short_pred: [batch, short_horizon]
            mid_pred: [batch, mid_horizon]
            long_pred: [batch, long_horizon]
            info: dict with attention weights and variable importance
        """
        batch_size = static_features.size(0)

        # ---- Static encoding ----
        static_input = torch.cat([static_features, spatial_embedding], dim=-1)
        static_encoded = self.static_encoder(static_input)

        # Generate context vectors for different uses
        cs_variable = self.static_context_variable(static_encoded)
        cs_enrichment = self.static_context_enrichment(static_encoded)
        cs_h = self.static_context_state_h(static_encoded)
        cs_c = self.static_context_state_c(static_encoded)

        # ---- Input projection ----
        past_proj = self.past_input_proj(dynamic_past)    # [batch, lookback, hidden]
        future_proj = self.future_input_proj(dynamic_future)  # [batch, horizon, hidden]

        # ---- Variable selection ----
        # Treating projected inputs as single variables (can be extended to per-feature VSN)
        past_selected, past_var_weights = self.past_vsn(
            past_proj.unsqueeze(-2), cs_variable.unsqueeze(1).expand(-1, past_proj.size(1), -1)
        )
        future_selected, future_var_weights = self.future_vsn(
            future_proj.unsqueeze(-2),
            cs_variable.unsqueeze(1).expand(-1, future_proj.size(1), -1),
        )

        # ---- LSTM encoding ----
        # Initialize LSTM hidden state from static context
        num_lstm_layers = self.lstm_encoder.num_layers
        h_0 = cs_h.unsqueeze(0).expand(num_lstm_layers, -1, -1).contiguous()
        c_0 = cs_c.unsqueeze(0).expand(num_lstm_layers, -1, -1).contiguous()

        encoder_output, (h_n, c_n) = self.lstm_encoder(past_selected, (h_0, c_0))
        decoder_output, _ = self.lstm_decoder(future_selected, (h_n, c_n))

        # Combine encoder and decoder outputs along time axis
        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)

        # Post-LSTM gate
        lstm_output = self.post_lstm_gate(lstm_output)

        # ---- Static enrichment ----
        enrichment_context = cs_enrichment.unsqueeze(1).expand(-1, lstm_output.size(1), -1)
        enriched = self.static_enrichment(lstm_output, enrichment_context)

        # ---- Temporal self-attention ----
        attn_output, attn_weights = self.attention(enriched, enriched, enriched)
        attn_output = self.post_attention_gate(attn_output)

        # ---- Position-wise feed-forward ----
        ff_output = self.ff(attn_output)

        # ---- Decoder heads ----
        # Use the mean of all temporal positions as the aggregate representation
        aggregate = ff_output.mean(dim=1)  # [batch, hidden_size]

        short_pred = self.short_head(aggregate)
        mid_pred = self.mid_head(aggregate)
        long_pred = self.long_head(aggregate)

        info = {
            "attention_weights": attn_weights,
            "past_variable_weights": past_var_weights,
            "future_variable_weights": future_var_weights,
        }

        return short_pred, mid_pred, long_pred, info
