"""
Dispenser and staffing rule engine for HP AI Engine.

Threshold-based operational rules driven by the 0–6 hour forecast.
Produces actionable instructions for station managers — what to do,
when to do it, and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("dispenser_rules", component="optimiser")


@dataclass
class StationState:
    """Current operational state of a station."""

    station_id: str
    active_dispensers: int
    total_dispensers: int
    current_staff: int
    current_inventory_kg: float
    storage_capacity_kg: float
    current_utilisation: float  # 0.0 to 1.0

    @property
    def inventory_pct(self) -> float:
        """Current inventory as a percentage of capacity."""
        if self.storage_capacity_kg == 0:
            return 0.0
        return self.current_inventory_kg / self.storage_capacity_kg


@dataclass
class Action:
    """Operational action recommended by the rule engine."""

    station_id: str
    action_type: Literal[
        "open_dispenser", "close_dispenser",
        "add_staff", "reduce_staff",
        "alert_manager", "request_tanker",
    ]
    urgency: Literal["immediate", "within_1h", "within_3h"]
    reason: str
    details: dict | None = None
    timestamp: datetime | None = None


class DispenserRuleEngine:
    """
    Threshold-based rule engine for dispenser and staffing management.

    Rules are evaluated against the 0–6 hour forecast and current station state.
    Thresholds are configurable to account for station-specific operating norms.

    Default thresholds:
    - utilisation > 80% → open all dispensers
    - utilisation > 60% → open N-1 dispensers and alert staff
    - utilisation < 30% → reduce to minimum dispensers
    - demand spike > 2× current → pre-emptive staffing alert
    - inventory < 25% of capacity → request tanker

    Args:
        high_threshold: Utilisation above which all dispensers open. Default 0.80.
        mid_threshold: Moderate utilisation threshold. Default 0.60.
        low_threshold: Low utilisation threshold for reducing dispensers. Default 0.30.
        spike_multiplier: Demand increase ratio triggering pre-emptive staffing. Default 2.0.
        inventory_alert_pct: Inventory level triggering tanker request. Default 0.25.
        min_dispensers: Minimum dispensers that must stay open. Default 1.
    """

    def __init__(
        self,
        high_threshold: float = 0.80,
        mid_threshold: float = 0.60,
        low_threshold: float = 0.30,
        spike_multiplier: float = 2.0,
        inventory_alert_pct: float = 0.25,
        min_dispensers: int = 1,
    ):
        self.high_threshold = high_threshold
        self.mid_threshold = mid_threshold
        self.low_threshold = low_threshold
        self.spike_multiplier = spike_multiplier
        self.inventory_alert_pct = inventory_alert_pct
        self.min_dispensers = min_dispensers

    def _predict_utilisation(
        self,
        forecast_6h: list[float],
        state: StationState,
    ) -> float:
        """
        Estimate peak utilisation from the 6-hour forecast.

        Utilisation = peak_predicted_demand / (dispenser_capacity × active_dispensers).
        """
        if not forecast_6h or state.total_dispensers == 0:
            return 0.0

        peak_demand = max(forecast_6h)
        # Approximate per-dispenser capacity: 50 kg/hour
        capacity = state.total_dispensers * 50.0

        return min(peak_demand / max(capacity, 1.0), 1.5)

    def evaluate(
        self,
        forecast_6h: list[float],
        state: StationState,
    ) -> list[Action]:
        """
        Evaluate rules and generate actions.

        Args:
            forecast_6h: Predicted demand (kg) for each of the next 6 hours.
            state: Current station operational state.

        Returns:
            List of recommended actions, sorted by urgency.
        """
        actions: list[Action] = []
        now = datetime.now()

        predicted_util = self._predict_utilisation(forecast_6h, state)
        current_demand = forecast_6h[0] if forecast_6h else 0
        peak_demand = max(forecast_6h) if forecast_6h else 0

        # Rule 1: High utilisation → open all dispensers
        if predicted_util > self.high_threshold:
            if state.active_dispensers < state.total_dispensers:
                actions.append(Action(
                    station_id=state.station_id,
                    action_type="open_dispenser",
                    urgency="immediate",
                    reason=(
                        f"Predicted utilisation {predicted_util:.0%} exceeds "
                        f"{self.high_threshold:.0%}. Open all {state.total_dispensers} dispensers."
                    ),
                    details={
                        "dispensers_to_open": state.total_dispensers - state.active_dispensers,
                        "predicted_utilisation": round(predicted_util, 2),
                    },
                    timestamp=now,
                ))

        # Rule 2: Moderate utilisation → open N-1 dispensers and alert
        elif predicted_util > self.mid_threshold:
            target_dispensers = state.total_dispensers - 1
            if state.active_dispensers < target_dispensers:
                actions.append(Action(
                    station_id=state.station_id,
                    action_type="open_dispenser",
                    urgency="within_1h",
                    reason=(
                        f"Predicted utilisation {predicted_util:.0%}. "
                        f"Recommend {target_dispensers} active dispensers."
                    ),
                    details={"target_dispensers": target_dispensers},
                    timestamp=now,
                ))

            actions.append(Action(
                station_id=state.station_id,
                action_type="alert_manager",
                urgency="within_1h",
                reason=f"Moderate demand expected. Peak forecast: {peak_demand:.0f} kg/h.",
                timestamp=now,
            ))

        # Rule 3: Low utilisation → reduce dispensers
        elif predicted_util < self.low_threshold:
            if state.active_dispensers > self.min_dispensers:
                actions.append(Action(
                    station_id=state.station_id,
                    action_type="close_dispenser",
                    urgency="within_3h",
                    reason=(
                        f"Low utilisation {predicted_util:.0%}. "
                        f"Reduce to {self.min_dispensers} dispenser(s)."
                    ),
                    details={
                        "dispensers_to_close": state.active_dispensers - self.min_dispensers,
                    },
                    timestamp=now,
                ))

        # Rule 4: Demand spike → pre-emptive staffing
        if current_demand > 0 and peak_demand > current_demand * self.spike_multiplier:
            actions.append(Action(
                station_id=state.station_id,
                action_type="add_staff",
                urgency="within_1h",
                reason=(
                    f"Demand spike predicted: {peak_demand:.0f} kg/h "
                    f"(current: {current_demand:.0f} kg/h, "
                    f"{peak_demand / current_demand:.1f}× increase)."
                ),
                timestamp=now,
            ))

        # Rule 5: Low inventory → request tanker
        if state.inventory_pct < self.inventory_alert_pct:
            # Estimate hours until stockout
            avg_demand = sum(forecast_6h) / max(len(forecast_6h), 1)
            hours_to_stockout = (
                state.current_inventory_kg / max(avg_demand, 1)
                if avg_demand > 0 else float("inf")
            )

            urgency = "immediate" if hours_to_stockout < 2 else "within_1h"
            actions.append(Action(
                station_id=state.station_id,
                action_type="request_tanker",
                urgency=urgency,
                reason=(
                    f"Inventory at {state.inventory_pct:.0%} of capacity "
                    f"({state.current_inventory_kg:.0f} kg). "
                    f"Estimated {hours_to_stockout:.1f}h until stockout."
                ),
                details={
                    "current_inventory_kg": state.current_inventory_kg,
                    "hours_to_stockout": round(hours_to_stockout, 1),
                },
                timestamp=now,
            ))

        # Sort by urgency: immediate > within_1h > within_3h
        urgency_order = {"immediate": 0, "within_1h": 1, "within_3h": 2}
        actions.sort(key=lambda a: urgency_order.get(a.urgency, 3))

        if actions:
            logger.info(
                f"Station {state.station_id}: {len(actions)} actions generated",
                extra={"station_id": state.station_id, "num_actions": len(actions)},
            )

        return actions
