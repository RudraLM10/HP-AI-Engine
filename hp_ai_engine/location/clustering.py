"""
DBSCAN-based demand zone clustering for HP AI Engine.

Identifies coherent high-demand zones across the station network
without requiring a pre-specified number of clusters. Zones with
high demand density but low station density are flagged as underserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

from hp_ai_engine.utils.geo import haversine
from hp_ai_engine.utils.logging import get_logger

logger = get_logger("clustering", component="location")


@dataclass
class DemandZone:
    """A spatially coherent demand zone identified by DBSCAN."""

    zone_id: int
    centroid: tuple[float, float]            # (lat, lon)
    member_stations: list[str]
    total_predicted_demand_kg: float
    area_sq_km: float
    station_density: float                    # stations per sq km
    is_underserved: bool = False
    avg_demand_per_station: float = 0.0


class DemandZoneClustering:
    """
    DBSCAN clustering to identify high-demand zones.

    DBSCAN is chosen over K-Means because:
    - No need to pre-specify cluster count K
    - Handles irregularly shaped urban zones
    - Automatically identifies noise points (isolated stations)

    Args:
        eps_km: Maximum distance between stations in the same zone. Default 5.0.
        min_samples: Minimum stations to form a zone. Default 3.
        underserved_threshold: Demand-to-station ratio above which a zone
                              is flagged as underserved. Default 2.0.
    """

    def __init__(
        self,
        eps_km: float = 5.0,
        min_samples: int = 3,
        underserved_threshold: float = 2.0,
    ):
        self.eps_km = eps_km
        self.min_samples = min_samples
        self.underserved_threshold = underserved_threshold

    def _compute_pairwise_distances(
        self,
        coords: list[tuple[float, float]],
    ) -> np.ndarray:
        """Compute pairwise Haversine distance matrix in km."""
        n = len(coords)
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        return dist_matrix

    def _estimate_zone_area(self, coords: list[tuple[float, float]]) -> float:
        """Estimate zone area in sq km from bounding box of member stations."""
        if len(coords) < 2:
            return 1.0

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]

        # Width and height of bounding box
        width_km = haversine(min(lats), min(lons), min(lats), max(lons))
        height_km = haversine(min(lats), min(lons), max(lats), min(lons))

        return max(width_km * height_km, 0.01)

    def find_demand_zones(
        self,
        station_ids: list[str],
        coords: list[tuple[float, float]],
        demand_values: list[float],
    ) -> list[DemandZone]:
        """
        Run DBSCAN to identify demand zones.

        Each station is weighted by its long-term predicted demand.
        Zones where demand significantly exceeds station infrastructure
        are flagged as underserved.

        Args:
            station_ids: Station identifiers.
            coords: (latitude, longitude) per station.
            demand_values: 6-month average predicted demand per station (kg/day).

        Returns:
            List of DemandZone objects.
        """
        if not station_ids:
            return []

        # Compute distance matrix
        dist_matrix = self._compute_pairwise_distances(coords)

        # Run DBSCAN with precomputed distances
        dbscan = DBSCAN(
            eps=self.eps_km,
            min_samples=self.min_samples,
            metric="precomputed",
        )
        labels = dbscan.fit_predict(dist_matrix)

        # Build zones
        unique_labels = set(labels)
        unique_labels.discard(-1)  # remove noise label

        zones: list[DemandZone] = []
        # Compute network-wide average for underserved comparison
        avg_demand_per_station_global = (
            np.mean(demand_values) if demand_values else 0
        )

        for zone_id in sorted(unique_labels):
            mask = labels == zone_id
            member_ids = [station_ids[i] for i in range(len(station_ids)) if mask[i]]
            member_coords = [coords[i] for i in range(len(coords)) if mask[i]]
            member_demands = [demand_values[i] for i in range(len(demand_values)) if mask[i]]

            total_demand = sum(member_demands)
            area = self._estimate_zone_area(member_coords)
            station_density = len(member_ids) / max(area, 0.01)
            avg_demand_per_station = total_demand / max(len(member_ids), 1)

            # Centroid
            centroid = (
                np.mean([c[0] for c in member_coords]),
                np.mean([c[1] for c in member_coords]),
            )

            # Underserved: high demand per station relative to network average
            is_underserved = (
                avg_demand_per_station > avg_demand_per_station_global * self.underserved_threshold
            )

            zones.append(DemandZone(
                zone_id=int(zone_id),
                centroid=(round(centroid[0], 6), round(centroid[1], 6)),
                member_stations=member_ids,
                total_predicted_demand_kg=round(total_demand, 1),
                area_sq_km=round(area, 2),
                station_density=round(station_density, 4),
                is_underserved=is_underserved,
                avg_demand_per_station=round(avg_demand_per_station, 1),
            ))

        # Log noise points
        noise_count = sum(1 for l in labels if l == -1)
        logger.info(
            f"DBSCAN: {len(zones)} zones identified, {noise_count} isolated stations, "
            f"{sum(1 for z in zones if z.is_underserved)} underserved zones"
        )

        return zones
