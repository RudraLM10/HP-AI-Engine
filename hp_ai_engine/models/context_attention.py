"""
Context Attention Override for HP AI Engine.

Monitors real-time signals (traffic, rainfall, AQI) and applies a learned
scaling factor to TFT predictions when current conditions deviate significantly
from historical baselines.

This acts as ​the "fast reaction" layer — the TFT captures learned patterns,
but if a sudden rainstorm or road closure happens right now, the context
attention overrides the forecast in a controlled, bounded way.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContextAttentionOverride(nn.Module):
    """
    Real-time contextual override mechanism.

    Process:
    1. Compute z-score deviation of each real-time signal from baseline
    2. If max(|z-scores|) > deviation_threshold → apply override
    3. Small MLP maps deviations → scaling factor in [scaling_min, scaling_max]
    4. adjusted_prediction = tft_prediction × scaling_factor

    The override is bounded (default [0.5, 1.5]) so it cannot produce
    unreasonably large or small predictions.

    Signal attributions are computed via input-gradient magnitude,
    providing a simple feature importance ranking for why the override fired.

    Args:
        num_signals: Number of real-time context signals. Default 3
                     (traffic_speed, rainfall, AQI).
        deviation_threshold: Z-score threshold to activate override. Default 2.0.
        scaling_min: Minimum scaling factor. Default 0.5.
        scaling_max: Maximum scaling factor. Default 1.5.
        mlp_hidden: Hidden dimension of the override MLP. Default 32.
    """

    def __init__(
        self,
        num_signals: int = 3,
        deviation_threshold: float = 2.0,
        scaling_min: float = 0.5,
        scaling_max: float = 1.5,
        mlp_hidden: int = 32,
    ):
        super().__init__()

        self.num_signals = num_signals
        self.deviation_threshold = deviation_threshold
        self.scaling_min = scaling_min
        self.scaling_max = scaling_max

        # MLP: deviations → scaling factor
        self.override_mlp = nn.Sequential(
            nn.Linear(num_signals, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),  # output in [0, 1], then rescaled to [scaling_min, scaling_max]
        )

    def compute_deviations(
        self,
        current_signals: torch.Tensor,
        baseline_means: torch.Tensor,
        baseline_stds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute z-score deviations of current signals from baselines.

        Args:
            current_signals: [batch, num_signals] current values.
            baseline_means: [batch, num_signals] rolling 30-day means.
            baseline_stds: [batch, num_signals] rolling 30-day stds.

        Returns:
            z_scores: [batch, num_signals].
        """
        # Clamp stds to avoid division by zero
        safe_stds = torch.clamp(baseline_stds, min=1e-6)
        return (current_signals - baseline_means) / safe_stds

    def forward(
        self,
        tft_prediction: torch.Tensor,
        current_signals: torch.Tensor,
        baseline_means: torch.Tensor,
        baseline_stds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Apply context-aware override to TFT predictions.

        Args:
            tft_prediction: [batch, horizon] raw TFT forecast.
            current_signals: [batch, num_signals] current real-time values.
            baseline_means: [batch, num_signals] historical baseline means.
            baseline_stds: [batch, num_signals] historical baseline stds.

        Returns:
            adjusted_prediction: [batch, horizon] (possibly modified).
            info: dict with:
                override_active: [batch] bool tensor
                scaling_factor: [batch] applied scaling factor
                z_scores: [batch, num_signals]
                signal_attributions: [batch, num_signals] feature importance
        """
        # Compute z-score deviations
        z_scores = self.compute_deviations(current_signals, baseline_means, baseline_stds)

        # Determine which samples have significant deviations
        max_abs_z = z_scores.abs().max(dim=-1).values  # [batch]
        override_active = max_abs_z > self.deviation_threshold  # [batch]

        # Compute scaling factor via MLP
        # Enable gradient for attribution computation
        z_input = z_scores.detach().requires_grad_(True)
        raw_scale = self.override_mlp(z_input)  # [batch, 1]

        # Rescale from [0, 1] to [scaling_min, scaling_max]
        scaling_factor = self.scaling_min + raw_scale.squeeze(-1) * (
            self.scaling_max - self.scaling_min
        )  # [batch]

        # Compute signal attributions via input gradient magnitude
        if z_input.grad is not None:
            z_input.grad.zero_()
        raw_scale.sum().backward(retain_graph=True)
        signal_attributions = z_input.grad.abs() if z_input.grad is not None else z_scores.abs()

        # Normalise attributions to sum to 1
        attr_sum = signal_attributions.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        signal_attributions = signal_attributions / attr_sum

        # Apply override only to samples where deviation exceeds threshold
        effective_scale = torch.where(
            override_active.unsqueeze(-1).expand_as(tft_prediction),
            scaling_factor.unsqueeze(-1).expand_as(tft_prediction),
            torch.ones_like(tft_prediction),
        )
        adjusted_prediction = tft_prediction * effective_scale

        info = {
            "override_active": override_active,
            "scaling_factor": scaling_factor,
            "z_scores": z_scores.detach(),
            "signal_attributions": signal_attributions.detach(),
        }

        return adjusted_prediction, info
