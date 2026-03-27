"""
Multi-Criteria Decision Analysis (MCDA) site scoring for HP AI Engine.

Scores candidate CNG station sites on six weighted criteria:
1. Vehicle density (25%) — CNG vehicles per sq km in catchment
2. Traffic volume (20%) — average daily vehicles on adjacent roads
3. Predicted demand (20%) — TFT long-term forecast for the zone
4. Competitor proximity (15%) — inverse distance to nearest competitor
5. Catchment demographics (15%) — population density × income bracket
6. Land availability (5%) — binary / external input

Each criterion is min-max normalised to [0, 100] across all candidates.
Final score = weighted sum → range [0, 100].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("site_scoring", component="location")


@dataclass
class CandidateSite:
    """A candidate location for a new CNG station."""

    site_id: str
    location: tuple[float, float]  # (lat, lon)
    vehicle_density: float          # CNG vehicles per sq km
    traffic_volume: float           # average daily vehicles on adjacent roads
    predicted_demand: float         # TFT long-term forecast (kg/day)
    competitor_distance_km: float   # distance to nearest competitor station
    population_density: float       # people per sq km in catchment
    income_bracket: float           # encoded: low=1, mid=2, high=3
    land_available: bool = True     # from external input / manual flag


@dataclass
class ScoredSite:
    """A scored and ranked candidate site."""

    site_id: str
    location: tuple[float, float]
    total_score: float
    breakdown: dict[str, float]     # criterion_name -> weighted contribution
    rank: int = 0


# Default scoring weights (sum to 1.0)
DEFAULT_WEIGHTS = {
    "vehicle_density": 0.25,
    "traffic_volume": 0.20,
    "predicted_demand": 0.20,
    "competitor_proximity": 0.15,
    "catchment_demographics": 0.15,
    "land_availability": 0.05,
}


class SiteScoringModel:
    """
    Weighted MCDA site scoring model for CNG station placement.

    Transparent and auditable: every site's score decomposes into six
    named criteria, suitable for board-level governance and regulatory approval.

    Args:
        weights: Dict mapping criterion name to weight (must sum to ~1.0).
                Defaults to architecture-specified weights.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        # Validate weights sum
        total = sum(self.weights.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Weights must sum to ~1.0, got {total:.4f}")

    def _extract_raw_scores(
        self,
        candidates: list[CandidateSite],
    ) -> dict[str, list[float]]:
        """Extract raw criterion values from candidates."""
        return {
            "vehicle_density": [c.vehicle_density for c in candidates],
            "traffic_volume": [c.traffic_volume for c in candidates],
            "predicted_demand": [c.predicted_demand for c in candidates],
            # Competitor proximity: farther is better (less competition)
            "competitor_proximity": [c.competitor_distance_km for c in candidates],
            # Demographics: population_density × income_bracket
            "catchment_demographics": [
                c.population_density * c.income_bracket for c in candidates
            ],
            "land_availability": [
                100.0 if c.land_available else 0.0 for c in candidates
            ],
        }

    def _normalise_to_100(self, values: list[float]) -> list[float]:
        """Min-max normalise values to [0, 100]."""
        if not values:
            return []
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return [50.0] * len(values)  # all equal → midpoint
        return [(v - v_min) / (v_max - v_min) * 100.0 for v in values]

    def score_candidates(
        self,
        candidates: list[CandidateSite],
    ) -> list[ScoredSite]:
        """
        Score and rank all candidate sites.

        Args:
            candidates: List of candidate sites with raw criterion values.

        Returns:
            List of ScoredSite, sorted by total_score descending (rank 1 = best).
        """
        if not candidates:
            return []

        # Extract and normalise
        raw = self._extract_raw_scores(candidates)
        normalised: dict[str, list[float]] = {}

        for criterion, values in raw.items():
            normalised[criterion] = self._normalise_to_100(values)

        # Compute weighted scores
        scored_sites: list[ScoredSite] = []

        for i, candidate in enumerate(candidates):
            breakdown: dict[str, float] = {}
            total = 0.0

            for criterion, weight in self.weights.items():
                norm_value = normalised[criterion][i]
                weighted_contribution = norm_value * weight
                breakdown[criterion] = round(weighted_contribution, 2)
                total += weighted_contribution

            scored_sites.append(ScoredSite(
                site_id=candidate.site_id,
                location=candidate.location,
                total_score=round(total, 2),
                breakdown=breakdown,
            ))

        # Rank by total score (descending)
        scored_sites.sort(key=lambda s: s.total_score, reverse=True)
        for rank, site in enumerate(scored_sites, start=1):
            site.rank = rank

        logger.info(
            f"Scored {len(scored_sites)} candidate sites. "
            f"Top: {scored_sites[0].site_id} ({scored_sites[0].total_score:.1f})"
        )

        return scored_sites

    def recalibrate_weights(
        self,
        actual_throughput: list[float],
        predicted_scores: list[float],
    ) -> dict[str, float]:
        """
        Recalibrate scoring weights based on actual vs predicted performance.

        Run a retrospective validation: score existing stations and compare
        with their actual throughput. If high-scoring stations don't show
        high throughput, adjust weights.

        This is a placeholder for a proper weight optimisation—in production,
        use linear regression or gradient-based optimisation.

        Args:
            actual_throughput: Actual kg/day for N existing stations.
            predicted_scores: Model-predicted scores for those same stations.

        Returns:
            Suggested updated weights.
        """
        if len(actual_throughput) != len(predicted_scores):
            raise ValueError("Must provide equal-length actual and predicted arrays")

        # Compute correlation as a simple diagnostic
        if len(actual_throughput) < 3:
            logger.warning("Not enough stations for weight recalibration")
            return self.weights

        correlation = np.corrcoef(actual_throughput, predicted_scores)[0, 1]
        logger.info(
            f"Weight recalibration: correlation between scores and throughput = {correlation:.3f}"
        )

        if correlation < 0.5:
            logger.warning(
                "Low correlation between scores and actual throughput — "
                "weights may need manual review"
            )

        # Return current weights (proper optimisation would go here)
        return self.weights
