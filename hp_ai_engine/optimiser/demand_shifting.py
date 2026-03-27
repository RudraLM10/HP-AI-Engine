"""
Non-price demand shifting engine for HP AI Engine.

Redistributes demand between stations and time windows using CRM mechanisms
that work within India's regulated CNG pricing framework:
- Priority queue tokens for advance booking at off-peak hours
- Loyalty point multipliers for off-peak refuelling
- Fleet operator scheduling to preferred low-utilisation windows
- Redirect suggestions when a station approaches capacity
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("demand_shifting", component="optimiser")


@dataclass
class Incentive:
    """A demand shifting incentive recommendation."""

    station_id: str
    mechanism: Literal[
        "loyalty_multiplier", "priority_queue", "fleet_scheduling", "redirect"
    ]
    target_window: tuple[datetime, datetime]
    expected_shift_kg: float
    redirect_to: str | None = None
    description: str = ""


@dataclass
class NetworkForecast:
    """Aggregated forecast data for a station."""

    station_id: str
    forecast_6h: list[float]  # hourly demand forecast
    current_utilisation: float  # 0.0 to 1.0
    location: tuple[float, float]


class DemandShiftingEngine:
    """
    Non-price demand redistribution across the station network.

    All mechanisms operate within India's regulated CNG price framework —
    they shift when and where customers refuel without changing the
    regulated headline price.

    Args:
        overload_threshold: Utilisation above which to shift demand away. Default 0.75.
        underload_threshold: Utilisation below which to attract demand. Default 0.40.
        redirect_max_km: Maximum distance for redirect suggestions. Default 5.0.
        loyalty_multiplier_offpeak: Loyalty point multiplier for off-peak. Default 2.0.
    """

    def __init__(
        self,
        overload_threshold: float = 0.75,
        underload_threshold: float = 0.40,
        redirect_max_km: float = 5.0,
        loyalty_multiplier_offpeak: float = 2.0,
    ):
        self.overload_threshold = overload_threshold
        self.underload_threshold = underload_threshold
        self.redirect_max_km = redirect_max_km
        self.loyalty_multiplier_offpeak = loyalty_multiplier_offpeak

    def _find_offpeak_windows(
        self,
        forecast_6h: list[float],
        base_time: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Identify hours with below-average demand."""
        if not forecast_6h:
            return []

        avg_demand = sum(forecast_6h) / len(forecast_6h)
        windows = []

        for i, demand in enumerate(forecast_6h):
            if demand < avg_demand * 0.7:  # 30% below average
                start = base_time + timedelta(hours=i)
                end = start + timedelta(hours=1)
                windows.append((start, end))

        return windows

    def _find_nearby_alternatives(
        self,
        station: NetworkForecast,
        all_stations: list[NetworkForecast],
    ) -> list[NetworkForecast]:
        """Find nearby stations with spare capacity."""
        from hp_ai_engine.utils.geo import haversine

        alternatives = []
        for other in all_stations:
            if other.station_id == station.station_id:
                continue
            if other.current_utilisation >= self.overload_threshold:
                continue

            dist = haversine(
                station.location[0], station.location[1],
                other.location[0], other.location[1],
            )
            if dist <= self.redirect_max_km:
                alternatives.append(other)

        # Sort by utilisation (lowest first = most spare capacity)
        alternatives.sort(key=lambda s: s.current_utilisation)
        return alternatives

    def generate_incentives(
        self,
        network_forecasts: list[NetworkForecast],
    ) -> list[Incentive]:
        """
        Generate demand shifting incentives for the entire network.

        Args:
            network_forecasts: Forecast and state for all stations.

        Returns:
            List of Incentive recommendations.
        """
        now = datetime.now()
        incentives: list[Incentive] = []

        overloaded = [s for s in network_forecasts if s.current_utilisation > self.overload_threshold]
        underloaded = [s for s in network_forecasts if s.current_utilisation < self.underload_threshold]

        for station in overloaded:
            # 1. Redirect to nearby stations with spare capacity
            alternatives = self._find_nearby_alternatives(station, network_forecasts)
            if alternatives:
                best_alt = alternatives[0]
                incentives.append(Incentive(
                    station_id=station.station_id,
                    mechanism="redirect",
                    target_window=(now, now + timedelta(hours=6)),
                    expected_shift_kg=sum(station.forecast_6h) * 0.15,
                    redirect_to=best_alt.station_id,
                    description=(
                        f"Redirect ~15% of traffic to {best_alt.station_id} "
                        f"(utilisation: {best_alt.current_utilisation:.0%})"
                    ),
                ))

            # 2. Loyalty multiplier for off-peak hours
            offpeak = self._find_offpeak_windows(station.forecast_6h, now)
            for window in offpeak[:2]:  # cap at 2 windows
                incentives.append(Incentive(
                    station_id=station.station_id,
                    mechanism="loyalty_multiplier",
                    target_window=window,
                    expected_shift_kg=sum(station.forecast_6h) * 0.05,
                    description=(
                        f"{self.loyalty_multiplier_offpeak}× loyalty points for "
                        f"refuelling between {window[0].strftime('%H:%M')} "
                        f"and {window[1].strftime('%H:%M')}"
                    ),
                ))

            # 3. Fleet scheduling to off-peak
            if offpeak:
                incentives.append(Incentive(
                    station_id=station.station_id,
                    mechanism="fleet_scheduling",
                    target_window=offpeak[0],
                    expected_shift_kg=sum(station.forecast_6h) * 0.10,
                    description=(
                        "Contact fleet operators to reschedule refuelling "
                        f"to off-peak window starting {offpeak[0][0].strftime('%H:%M')}"
                    ),
                ))

        # 4. Priority queue for underloaded stations to attract demand
        for station in underloaded:
            incentives.append(Incentive(
                station_id=station.station_id,
                mechanism="priority_queue",
                target_window=(now, now + timedelta(hours=6)),
                expected_shift_kg=sum(station.forecast_6h) * 0.10,
                description=(
                    "Offer priority queue / no-wait token for advance bookings "
                    f"at {station.station_id} (current utilisation: "
                    f"{station.current_utilisation:.0%})"
                ),
            ))

        if incentives:
            logger.info(
                f"Generated {len(incentives)} demand shifting incentives "
                f"({len(overloaded)} overloaded, {len(underloaded)} underloaded stations)"
            )

        return incentives
