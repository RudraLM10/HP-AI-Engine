"""
MDU (Mobile Dispensing Unit) vs Permanent Station decision engine.

Uses the Coefficient of Variation (CV) from the long-term TFT forecast
to decide whether a candidate location warrants immediate permanent
infrastructure or should be tested with an MDU first.

CV < 0.2  → Consistent demand → Permanent station
CV >= 0.2 → Variable demand → Deploy MDU, collect 6 months, then decide
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("mdu_decision", component="location")


@dataclass
class StationDecision:
    """Decision output for MDU vs permanent station."""

    site_id: str
    recommendation: Literal["permanent", "mdu_first"]
    cv_value: float
    monthly_demand_profile: list[float]  # average demand per month
    confidence: float                     # 0 to 1
    rationale: str                        # human-readable explanation
    estimated_capex_cr: float = 0.0       # estimated capital expenditure in crore


class MDUDecisionEngine:
    """
    CV-based decision engine for MDU vs permanent station deployment.

    The Coefficient of Variation (CV = std / mean) measures demand
    consistency. Low CV = stable, predictable demand → safe to invest
    in permanent infrastructure. High CV = volatile demand → test
    with an MDU first before committing ₹80-100 crore.

    Args:
        cv_threshold: CV below which permanent station is recommended.
                     Default 0.2.
        mdu_evaluation_months: Duration of MDU pilot before re-evaluation.
                              Default 6.
        permanent_capex_cr: Typical permanent station cost (crore). Default 90.
        mdu_capex_cr: Typical MDU deployment cost (crore). Default 5.
    """

    def __init__(
        self,
        cv_threshold: float = 0.2,
        mdu_evaluation_months: int = 6,
        permanent_capex_cr: float = 90.0,
        mdu_capex_cr: float = 5.0,
    ):
        self.cv_threshold = cv_threshold
        self.mdu_evaluation_months = mdu_evaluation_months
        self.permanent_capex_cr = permanent_capex_cr
        self.mdu_capex_cr = mdu_capex_cr

    def _compute_monthly_profile(
        self,
        long_forecast: np.ndarray,
    ) -> list[float]:
        """
        Aggregate hourly long-term forecast into monthly averages.

        Args:
            long_forecast: Hourly forecast values (up to 4320 hours = 6 months).

        Returns:
            List of monthly average demand values (up to 6 months).
        """
        hours_per_month = 720  # approximate
        monthly = []
        for start in range(0, len(long_forecast), hours_per_month):
            end = min(start + hours_per_month, len(long_forecast))
            chunk = long_forecast[start:end]
            if len(chunk) > 0:
                monthly.append(float(np.mean(chunk)))
        return monthly

    def _compute_cv(self, monthly_demand: list[float]) -> float:
        """Compute Coefficient of Variation from monthly demands."""
        if not monthly_demand or len(monthly_demand) < 2:
            return 0.0

        mean_val = np.mean(monthly_demand)
        std_val = np.std(monthly_demand, ddof=1)

        if mean_val == 0:
            return 0.0

        return float(std_val / mean_val)

    def _estimate_confidence(
        self,
        cv: float,
        num_months: int,
    ) -> float:
        """
        Estimate confidence in the recommendation.

        Higher confidence when:
        - CV is far from the threshold (clear decision)
        - More months of data available (better statistical power)
        """
        # Distance from threshold (normalised)
        distance = abs(cv - self.cv_threshold) / max(self.cv_threshold, 0.01)

        # Data sufficiency factor
        data_factor = min(num_months / 6.0, 1.0)

        return round(min(0.5 + 0.3 * distance + 0.2 * data_factor, 0.99), 2)

    def decide(
        self,
        site_id: str,
        long_term_forecast: np.ndarray | torch.Tensor | list[float],
    ) -> StationDecision:
        """
        Make MDU vs permanent station recommendation for a site.

        Args:
            site_id: Identifier for the candidate site.
            long_term_forecast: Hourly demand forecast for up to 6 months.

        Returns:
            StationDecision with recommendation, CV, and rationale.
        """
        # Convert to numpy
        if isinstance(long_term_forecast, torch.Tensor):
            forecast_np = long_term_forecast.detach().cpu().numpy().flatten()
        elif isinstance(long_term_forecast, list):
            forecast_np = np.array(long_term_forecast)
        else:
            forecast_np = long_term_forecast.flatten()

        # Compute monthly profile and CV
        monthly = self._compute_monthly_profile(forecast_np)
        cv = self._compute_cv(monthly)
        confidence = self._estimate_confidence(cv, len(monthly))

        # Decision
        if cv < self.cv_threshold:
            recommendation = "permanent"
            capex = self.permanent_capex_cr
            rationale = (
                f"CV = {cv:.3f} (below {self.cv_threshold} threshold). "
                f"Demand is consistent across months — "
                f"monthly avg: {np.mean(monthly):.0f} kg/day, "
                f"std: {np.std(monthly):.0f} kg/day. "
                f"Permanent station investment (₹{capex:.0f} Cr) is justified."
            )
        else:
            recommendation = "mdu_first"
            capex = self.mdu_capex_cr
            rationale = (
                f"CV = {cv:.3f} (above {self.cv_threshold} threshold). "
                f"Demand varies significantly across months — "
                f"monthly avg: {np.mean(monthly):.0f} kg/day, "
                f"std: {np.std(monthly):.0f} kg/day. "
                f"Deploy MDU first (₹{capex:.0f} Cr) for {self.mdu_evaluation_months} months. "
                f"Re-evaluate with real data before committing to permanent station."
            )

        decision = StationDecision(
            site_id=site_id,
            recommendation=recommendation,
            cv_value=round(cv, 4),
            monthly_demand_profile=[round(m, 1) for m in monthly],
            confidence=confidence,
            rationale=rationale,
            estimated_capex_cr=capex,
        )

        logger.info(
            f"Site {site_id}: {recommendation} (CV={cv:.3f}, confidence={confidence})",
            extra={"site_id": site_id, "cv": cv, "recommendation": recommendation},
        )

        return decision

    def reevaluate_mdu(
        self,
        site_id: str,
        mdu_actuals: list[float],
    ) -> StationDecision:
        """
        Re-evaluate an MDU deployment after the pilot period.

        Uses actual monthly demand data (not forecasts) collected during
        the MDU pilot to make a final recommendation.

        Args:
            site_id: Site identifier.
            mdu_actuals: Actual monthly demand during MDU pilot (kg/day avg).

        Returns:
            StationDecision with updated recommendation.
        """
        cv = self._compute_cv(mdu_actuals)
        confidence = self._estimate_confidence(cv, len(mdu_actuals))

        if cv < self.cv_threshold:
            recommendation = "permanent"
            rationale = (
                f"MDU evaluation complete. Actual CV = {cv:.3f} "
                f"(below {self.cv_threshold}). "
                f"Real demand data confirms consistent utilisation. "
                f"Convert to permanent station."
            )
            capex = self.permanent_capex_cr
        else:
            recommendation = "mdu_first"
            rationale = (
                f"MDU evaluation complete. Actual CV = {cv:.3f} "
                f"(still above {self.cv_threshold}). "
                f"Demand remains variable. Maintain MDU or consider relocation."
            )
            capex = self.mdu_capex_cr

        logger.info(
            f"MDU re-evaluation {site_id}: {recommendation} "
            f"(actual CV={cv:.3f})",
        )

        return StationDecision(
            site_id=site_id,
            recommendation=recommendation,
            cv_value=round(cv, 4),
            monthly_demand_profile=[round(m, 1) for m in mdu_actuals],
            confidence=confidence,
            rationale=rationale,
            estimated_capex_cr=capex,
        )
