"""
SHAP Explainability for HP AI Engine.

Computes SHAP values for individual station forecasts, producing:
- Per-feature attribution scores (how much each input contributed)
- Ranked feature importance list
- Human-readable narrative explanations

Example narrative:
    "Demand ↓340 kg: road closure effect via traffic speed (60%),
     heavy rainfall (30%), lower vehicle density (10%)"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ExplanationOutput:
    """Container for explainability results."""

    shap_values: dict[str, float]                     # feature_name -> SHAP value
    ranked_features: list[tuple[str, float]]           # sorted by |SHAP value|
    forecast_delta: float                              # total change from baseline
    narrative: str                                     # human-readable explanation


class SHAPExplainer:
    """
    SHAP-based forecast explainer.

    For each station's forecast, computes how much each input feature
    contributed to the deviation from a baseline prediction. Uses a
    simplified permutation-based approach compatible with the TFT-GCN
    architecture.

    Args:
        feature_names: List of names for input features (in order they appear
                      in the dynamic_past tensor).
        n_background_samples: Number of background samples for baseline
                             computation. Default 100.
        device: Torch device for computation.
    """

    def __init__(
        self,
        feature_names: list[str],
        n_background_samples: int = 100,
        device: str = "cpu",
    ):
        self.feature_names = feature_names
        self.n_background_samples = n_background_samples
        self.device = device

    @torch.no_grad()
    def explain(
        self,
        model: nn.Module,
        sample_kwargs: dict,
        baseline_kwargs: dict | None = None,
        horizon: str = "short",
    ) -> ExplanationOutput:
        """
        Compute SHAP-like attributions for a single forecast.

        Uses a simplified perturbation approach: for each feature, replace it
        with the baseline value and measure the forecast change. This is a
        first-order approximation of SHAP values.

        Args:
            model: Trained TFTGCNPredictor.
            sample_kwargs: Forward kwargs for the sample to explain.
            baseline_kwargs: Forward kwargs for the baseline (mean input).
                           If None, uses zeros.
            horizon: Which horizon to explain ('short', 'mid', 'long').

        Returns:
            ExplanationOutput with attributions and narrative.
        """
        model.eval()

        # Get the model's prediction for the sample
        output = model(**sample_kwargs)
        if horizon == "short":
            sample_pred = output.short_forecast.mean().item()
        elif horizon == "mid":
            sample_pred = output.mid_forecast.mean().item()
        else:
            sample_pred = output.long_forecast.mean().item()

        # Get baseline prediction
        if baseline_kwargs is not None:
            baseline_output = model(**baseline_kwargs)
        else:
            # Use zero baseline
            baseline_kwargs = {
                k: torch.zeros_like(v) if isinstance(v, torch.Tensor) else v
                for k, v in sample_kwargs.items()
            }
            baseline_output = model(**baseline_kwargs)

        if horizon == "short":
            baseline_pred = baseline_output.short_forecast.mean().item()
        elif horizon == "mid":
            baseline_pred = baseline_output.mid_forecast.mean().item()
        else:
            baseline_pred = baseline_output.long_forecast.mean().item()

        forecast_delta = sample_pred - baseline_pred

        # Feature ablation: replace each feature with baseline value
        dynamic_past = sample_kwargs.get("dynamic_past")
        if dynamic_past is None:
            return ExplanationOutput(
                shap_values={},
                ranked_features=[],
                forecast_delta=forecast_delta,
                narrative="No dynamic features available for attribution.",
            )

        baseline_past = baseline_kwargs.get("dynamic_past", torch.zeros_like(dynamic_past))
        num_features = min(dynamic_past.shape[-1], len(self.feature_names))

        attributions: dict[str, float] = {}
        for i in range(num_features):
            # Create perturbed input with feature i replaced by baseline
            perturbed = dynamic_past.clone()
            perturbed[..., i] = baseline_past[..., i]

            perturbed_kwargs = {**sample_kwargs, "dynamic_past": perturbed}
            perturbed_output = model(**perturbed_kwargs)

            if horizon == "short":
                perturbed_pred = perturbed_output.short_forecast.mean().item()
            elif horizon == "mid":
                perturbed_pred = perturbed_output.mid_forecast.mean().item()
            else:
                perturbed_pred = perturbed_output.long_forecast.mean().item()

            # Attribution = how much the forecast changes when this feature is removed
            attribution = sample_pred - perturbed_pred
            attributions[self.feature_names[i]] = attribution

        # Rank by absolute attribution
        ranked = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)

        # Generate narrative
        narrative = self.generate_narrative(attributions, forecast_delta)

        return ExplanationOutput(
            shap_values=attributions,
            ranked_features=ranked,
            forecast_delta=forecast_delta,
            narrative=narrative,
        )

    def generate_narrative(
        self,
        shap_values: dict[str, float],
        forecast_delta: float,
    ) -> str:
        """
        Generate a human-readable explanation string.

        Args:
            shap_values: Feature name -> attribution value.
            forecast_delta: Total forecast change from baseline.

        Returns:
            Narrative string, e.g.:
            "Demand ↓340 kg: traffic speed (60%), rainfall (30%), vehicle density (10%)"
        """
        if not shap_values or forecast_delta == 0:
            return "Forecast is at baseline level — no significant deviations detected."

        # Direction
        direction = "↑" if forecast_delta > 0 else "↓"
        delta_str = f"Demand {direction}{abs(forecast_delta):.0f} kg"

        # Top contributors (normalise to percentages)
        total_abs = sum(abs(v) for v in shap_values.values())
        if total_abs == 0:
            return f"{delta_str}: no dominant contributing features identified."

        ranked = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)

        # Take top 3 contributors
        contributions = []
        for name, value in ranked[:3]:
            pct = abs(value) / total_abs * 100
            if pct < 5:
                break
            contributions.append(f"{name} ({pct:.0f}%)")

        if not contributions:
            return f"{delta_str}: multiple small factors contributing equally."

        return f"{delta_str}: {', '.join(contributions)}"


class TFTAttentionExplainer:
    """
    Extracts interpretability directly from TFT's built-in attention weights.

    Unlike SHAP (which is model-agnostic but approximate), this uses the
    attention weights and variable selection weights produced by the TFT
    during inference — these are exact and free to compute.

    This complements the SHAP explainer for faster, real-time explanations.
    """

    @staticmethod
    def extract_temporal_importance(
        attention_weights: torch.Tensor,
        lookback_hours: int = 168,
    ) -> dict[str, float]:
        """
        Extract which past time steps the model attended to most.

        Args:
            attention_weights: [batch, q_len, k_len] from TFT attention.
            lookback_hours: Number of lookback hours.

        Returns:
            Dict mapping time descriptions to attention scores.
        """
        # Average across batch and query positions
        avg_attn = attention_weights.mean(dim=(0, 1))  # [k_len]

        if len(avg_attn) < lookback_hours:
            lookback_hours = len(avg_attn)

        # Aggregate into interpretable time windows
        attn_last_6h = avg_attn[-6:].sum().item() if lookback_hours >= 6 else 0
        attn_last_24h = avg_attn[-24:].sum().item() if lookback_hours >= 24 else 0
        attn_last_week = avg_attn[:lookback_hours].sum().item()

        total = attn_last_week if attn_last_week > 0 else 1.0

        return {
            "last_6_hours": attn_last_6h / total,
            "last_24_hours": attn_last_24h / total,
            "last_7_days": 1.0,  # normalised base
        }

    @staticmethod
    def extract_variable_importance(
        variable_weights: torch.Tensor,
        feature_names: list[str],
    ) -> list[tuple[str, float]]:
        """
        Extract which input variables the model considers most important.

        Args:
            variable_weights: [batch, num_variables] from VSN.
            feature_names: Names corresponding to each variable.

        Returns:
            Sorted list of (feature_name, importance_score) tuples.
        """
        avg_weights = variable_weights.mean(dim=0)  # [num_variables]
        importance = []
        for i, name in enumerate(feature_names):
            if i < len(avg_weights):
                importance.append((name, avg_weights[i].item()))

        return sorted(importance, key=lambda x: x[1], reverse=True)
