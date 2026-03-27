"""
Data drift detection for HP AI Engine.

Monitors two types of drift:
1. Covariate drift — input feature distributions shift (e.g. traffic patterns
   change after a new highway opens)
2. Concept drift — the relationship between inputs and demand changes
   (e.g. a new OEM launches an affordable CNG car, changing demand patterns)

Uses Population Stability Index (PSI) and CUSUM charts for detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("drift_detection", component="scalability")


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    feature_name: str
    drift_type: Literal["covariate", "concept"]
    psi_value: float
    threshold: float
    is_drifted: bool
    severity: Literal["none", "warning", "critical"]
    recommendation: str


class DriftDetector:
    """
    Drift detection using Population Stability Index (PSI) and CUSUM.

    PSI quantifies the shift between the training data distribution
    and the current production data distribution for each feature.

    PSI < 0.10 → No significant drift
    0.10 ≤ PSI < 0.25 → Moderate drift (warning)
    PSI ≥ 0.25 → Significant drift (critical → trigger retraining)

    CUSUM is used for concept drift: it monitors the cumulative sum
    of prediction residuals. A sustained positive or negative CUSUM
    indicates the model's relationship with the target has shifted.

    Args:
        psi_warning_threshold: PSI value for warning. Default 0.10.
        psi_critical_threshold: PSI value for critical drift. Default 0.25.
        cusum_threshold: CUSUM threshold for concept drift alert. Default 5.0.
        num_bins: Number of bins for PSI histogram. Default 10.
    """

    def __init__(
        self,
        psi_warning_threshold: float = 0.10,
        psi_critical_threshold: float = 0.25,
        cusum_threshold: float = 5.0,
        num_bins: int = 10,
    ):
        self.psi_warning = psi_warning_threshold
        self.psi_critical = psi_critical_threshold
        self.cusum_threshold = cusum_threshold
        self.num_bins = num_bins

        # CUSUM state
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Compute Population Stability Index between two distributions.

        PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)

        where P is the reference (training) distribution and Q is the
        current (production) distribution, both discretised into bins.

        Args:
            reference: Reference (training) feature values.
            current: Current (production) feature values.

        Returns:
            PSI value (non-negative).
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=self.num_bins)

        # Compute bin proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Normalise to proportions
        ref_pct = ref_counts / max(ref_counts.sum(), 1)
        cur_pct = cur_counts / max(cur_counts.sum(), 1)

        # Avoid log(0) by adding small epsilon
        epsilon = 1e-6
        ref_pct = np.clip(ref_pct, epsilon, None)
        cur_pct = np.clip(cur_pct, epsilon, None)

        # PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    def check_covariate_drift(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> DriftResult:
        """
        Check for covariate drift on a single feature.

        Args:
            feature_name: Name of the feature being monitored.
            reference: Training data distribution for this feature.
            current: Current production data for this feature.

        Returns:
            DriftResult with PSI value and recommendation.
        """
        psi = self.compute_psi(reference, current)

        if psi >= self.psi_critical:
            severity = "critical"
            recommendation = (
                f"CRITICAL: Feature '{feature_name}' has drifted significantly "
                f"(PSI={psi:.4f}). Trigger automated retraining pipeline."
            )
        elif psi >= self.psi_warning:
            severity = "warning"
            recommendation = (
                f"WARNING: Feature '{feature_name}' shows moderate drift "
                f"(PSI={psi:.4f}). Monitor closely; schedule retraining if trend continues."
            )
        else:
            severity = "none"
            recommendation = f"Feature '{feature_name}' is stable (PSI={psi:.4f})."

        return DriftResult(
            feature_name=feature_name,
            drift_type="covariate",
            psi_value=round(psi, 4),
            threshold=self.psi_critical,
            is_drifted=psi >= self.psi_critical,
            severity=severity,
            recommendation=recommendation,
        )

    def check_concept_drift(
        self,
        residuals: np.ndarray,
        target_mean: float = 0.0,
    ) -> DriftResult:
        """
        Check for concept drift using CUSUM on prediction residuals.

        Monitors cumulative deviations: if the model consistently
        over-predicts or under-predicts, CUSUM crosses the threshold.

        Args:
            residuals: Array of (actual - predicted) for recent predictions.
            target_mean: Expected mean residual (typically 0 for an unbiased model).

        Returns:
            DriftResult with CUSUM value and recommendation.
        """
        # Reset CUSUM for this batch
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        max_cusum = 0.0

        for r in residuals:
            deviation = r - target_mean
            self._cusum_pos = max(0, self._cusum_pos + deviation)
            self._cusum_neg = max(0, self._cusum_neg - deviation)
            max_cusum = max(max_cusum, self._cusum_pos, self._cusum_neg)

        is_drifted = max_cusum > self.cusum_threshold

        if is_drifted:
            direction = "over-predicting" if self._cusum_neg > self._cusum_pos else "under-predicting"
            severity = "critical"
            recommendation = (
                f"CONCEPT DRIFT: Model is systematically {direction} "
                f"(CUSUM={max_cusum:.2f} > {self.cusum_threshold}). "
                f"Trigger retraining with updated data."
            )
        else:
            severity = "none"
            recommendation = f"No concept drift detected (CUSUM={max_cusum:.2f})."

        return DriftResult(
            feature_name="model_residuals",
            drift_type="concept",
            psi_value=round(max_cusum, 4),
            threshold=self.cusum_threshold,
            is_drifted=is_drifted,
            severity=severity,
            recommendation=recommendation,
        )

    def run_full_check(
        self,
        reference_features: dict[str, np.ndarray],
        current_features: dict[str, np.ndarray],
        residuals: np.ndarray | None = None,
    ) -> list[DriftResult]:
        """
        Run full drift check on all features and concept drift.

        Args:
            reference_features: Dict of feature_name -> training data arrays.
            current_features: Dict of feature_name -> production data arrays.
            residuals: Optional prediction residuals for concept drift check.

        Returns:
            List of DriftResult for all checks.
        """
        results: list[DriftResult] = []

        # Covariate drift per feature
        for feat_name in reference_features:
            if feat_name in current_features:
                result = self.check_covariate_drift(
                    feat_name, reference_features[feat_name], current_features[feat_name]
                )
                results.append(result)
                if result.severity != "none":
                    logger.warning(result.recommendation)

        # Concept drift
        if residuals is not None and len(residuals) > 0:
            concept_result = self.check_concept_drift(residuals)
            results.append(concept_result)
            if concept_result.severity != "none":
                logger.warning(concept_result.recommendation)

        drifted_count = sum(1 for r in results if r.is_drifted)
        logger.info(
            f"Drift check complete: {drifted_count}/{len(results)} features drifted",
            extra={"drifted": drifted_count, "total_checked": len(results)},
        )

        return results
