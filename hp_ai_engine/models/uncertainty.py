"""
MC Dropout Uncertainty Quantification for HP AI Engine.

Wraps a trained TFT-GCN model and produces calibrated uncertainty estimates
by running multiple forward passes with stochastic dropout active during
inference. This avoids the need for a separate probabilistic model.

Usage:
    mc = MCDropoutInference(n_passes=30)
    result = mc.predict_with_uncertainty(model, batch, graph_data)
    # result.mean_forecast — point estimate
    # result.confidence_intervals[0.90] — 90% CI (lower, upper)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class UncertaintyOutput:
    """Container for uncertainty-quantified predictions."""

    mean_forecast: torch.Tensor                                   # [batch, horizon]
    std_forecast: torch.Tensor                                    # [batch, horizon]
    confidence_intervals: dict[float, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )  # level -> (lower, upper)
    all_passes: torch.Tensor | None = None  # [n_passes, batch, horizon]


class MCDropoutInference:
    """
    Monte Carlo Dropout inference wrapper.

    Activates dropout during inference and runs N forward passes to
    approximate Bayesian uncertainty. The spread of predictions across
    passes provides calibrated confidence intervals.

    Args:
        n_passes: Number of MC forward passes. Default 30.
        ci_levels: Confidence interval levels to compute.
        keep_all_passes: Whether to retain all individual pass outputs.
    """

    def __init__(
        self,
        n_passes: int = 30,
        ci_levels: list[float] | None = None,
        keep_all_passes: bool = False,
    ):
        self.n_passes = n_passes
        self.ci_levels = ci_levels or [0.80, 0.90, 0.95]
        self.keep_all_passes = keep_all_passes

    def _enable_dropout(self, model: nn.Module) -> None:
        """Set all Dropout layers to train mode (active during inference)."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def _restore_eval(self, model: nn.Module) -> None:
        """Restore model to full eval mode."""
        model.eval()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        model: nn.Module,
        horizon: str = "short",
        **forward_kwargs,
    ) -> UncertaintyOutput:
        """
        Run MC Dropout inference and produce uncertainty estimates.

        Args:
            model: Trained TFTGCNPredictor model.
            horizon: Which forecast horizon to quantify ('short', 'mid', 'long').
            **forward_kwargs: Arguments to pass to model.forward().

        Returns:
            UncertaintyOutput with mean, std, and confidence intervals.
        """
        model.eval()
        self._enable_dropout(model)

        all_predictions = []

        for _ in range(self.n_passes):
            output = model(**forward_kwargs)

            # Select the appropriate horizon forecast
            if horizon == "short":
                pred = output.short_forecast
            elif horizon == "mid":
                pred = output.mid_forecast
            elif horizon == "long":
                pred = output.long_forecast
            else:
                raise ValueError(f"Unknown horizon: {horizon}. Use 'short', 'mid', or 'long'.")

            all_predictions.append(pred)

        self._restore_eval(model)

        # Stack: [n_passes, batch, horizon]
        stacked = torch.stack(all_predictions, dim=0)

        # Compute statistics
        mean_forecast = stacked.mean(dim=0)
        std_forecast = stacked.std(dim=0)

        # Compute confidence intervals from percentiles
        confidence_intervals: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
        for level in self.ci_levels:
            alpha = (1 - level) / 2
            lower_pct = alpha * 100
            upper_pct = (1 - alpha) * 100

            lower = torch.quantile(stacked, alpha, dim=0)
            upper = torch.quantile(stacked, 1 - alpha, dim=0)
            confidence_intervals[level] = (lower, upper)

        return UncertaintyOutput(
            mean_forecast=mean_forecast,
            std_forecast=std_forecast,
            confidence_intervals=confidence_intervals,
            all_passes=stacked if self.keep_all_passes else None,
        )

    @torch.no_grad()
    def predict_all_horizons(
        self,
        model: nn.Module,
        **forward_kwargs,
    ) -> dict[str, UncertaintyOutput]:
        """
        Run MC Dropout for all three forecast horizons simultaneously.

        Returns:
            Dict with keys 'short', 'mid', 'long', each containing UncertaintyOutput.
        """
        model.eval()
        self._enable_dropout(model)

        all_short, all_mid, all_long = [], [], []

        for _ in range(self.n_passes):
            output = model(**forward_kwargs)
            all_short.append(output.short_forecast)
            all_mid.append(output.mid_forecast)
            all_long.append(output.long_forecast)

        self._restore_eval(model)

        results = {}
        for name, preds in [("short", all_short), ("mid", all_mid), ("long", all_long)]:
            stacked = torch.stack(preds, dim=0)
            mean_f = stacked.mean(dim=0)
            std_f = stacked.std(dim=0)

            cis: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
            for level in self.ci_levels:
                alpha = (1 - level) / 2
                lower = torch.quantile(stacked, alpha, dim=0)
                upper = torch.quantile(stacked, 1 - alpha, dim=0)
                cis[level] = (lower, upper)

            results[name] = UncertaintyOutput(
                mean_forecast=mean_f,
                std_forecast=std_f,
                confidence_intervals=cis,
                all_passes=stacked if self.keep_all_passes else None,
            )

        return results
