"""
Custom loss functions for HP AI Engine.

Combines Huber loss (robust to outliers) with a MAPE penalty (scales with
volume) for multi-horizon CNG demand forecasting. The MAPE term is weighted
by station throughput volume so high-volume stations have proportionally
more influence on training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberMAPELoss(nn.Module):
    """
    Combined loss: Huber + volume-weighted MAPE penalty.

    loss = Huber(y_true, y_pred, delta) + mape_weight × MAPE(y_true, y_pred, volumes)

    The Huber component provides robustness to outlier transactions (e.g.
    large fleet fills). The MAPE component ensures that percentage accuracy
    scales appropriately with station throughput.

    Args:
        huber_delta: Huber loss threshold. Default 1.0.
        mape_weight: Weight for the MAPE penalty term. Default 0.3.
        mape_epsilon: Small constant to avoid division by zero in MAPE. Default 1.0.
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        mape_weight: float = 0.3,
        mape_epsilon: float = 1.0,
    ):
        super().__init__()
        self.huber_delta = huber_delta
        self.mape_weight = mape_weight
        self.mape_epsilon = mape_epsilon

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        station_volumes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the combined loss.

        Args:
            y_pred: Predicted values [batch, horizon].
            y_true: Ground truth values [batch, horizon].
            station_volumes: Optional per-station average throughput [batch]
                            for volume-weighted MAPE. If None, uniform weighting.

        Returns:
            Scalar loss tensor.
        """
        # Huber loss
        huber = F.huber_loss(y_pred, y_true, delta=self.huber_delta, reduction="none")

        # MAPE loss: |y_true - y_pred| / (|y_true| + epsilon)
        abs_error = torch.abs(y_true - y_pred)
        mape = abs_error / (torch.abs(y_true) + self.mape_epsilon)

        if station_volumes is not None:
            # Volume-weight: higher volume stations contribute more
            # Normalise volumes to sum to batch_size to keep loss scale stable
            weights = station_volumes / (station_volumes.sum() + 1e-8) * station_volumes.size(0)
            weights = weights.unsqueeze(-1)  # [batch, 1]

            huber = huber * weights
            mape = mape * weights

        # Combine
        loss = huber.mean() + self.mape_weight * mape.mean()

        return loss


class MultiHorizonLoss(nn.Module):
    """
    Weighted combination of losses from the three forecast horizons.

    loss = w_short × loss(short) + w_mid × loss(mid) + w_long × loss(long)

    Short-term accuracy is prioritised as it drives immediate operational
    decisions (dispenser management, tanker dispatch).

    Args:
        base_loss: Loss function to apply per-horizon. Default HuberMAPELoss.
        weight_short: Weight for 0–6h forecast loss. Default 0.5.
        weight_mid: Weight for 1–7d forecast loss. Default 0.3.
        weight_long: Weight for 1–6mo forecast loss. Default 0.2.
    """

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        weight_short: float = 0.5,
        weight_mid: float = 0.3,
        weight_long: float = 0.2,
    ):
        super().__init__()
        self.base_loss = base_loss or HuberMAPELoss()
        self.weight_short = weight_short
        self.weight_mid = weight_mid
        self.weight_long = weight_long

    def forward(
        self,
        pred_short: torch.Tensor,
        pred_mid: torch.Tensor,
        pred_long: torch.Tensor,
        target_short: torch.Tensor,
        target_mid: torch.Tensor,
        target_long: torch.Tensor,
        station_volumes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the weighted multi-horizon loss.

        Returns:
            total_loss: Scalar weighted loss.
            loss_breakdown: Dict with per-horizon losses for logging.
        """
        loss_short = self.base_loss(pred_short, target_short, station_volumes)
        loss_mid = self.base_loss(pred_mid, target_mid, station_volumes)
        loss_long = self.base_loss(pred_long, target_long, station_volumes)

        total = (
            self.weight_short * loss_short
            + self.weight_mid * loss_mid
            + self.weight_long * loss_long
        )

        breakdown = {
            "loss_short": loss_short.item(),
            "loss_mid": loss_mid.item(),
            "loss_long": loss_long.item(),
            "loss_total": total.item(),
        }

        return total, breakdown
