"""
Tanker routing optimisation using Google OR-Tools VRPTW solver.

Solves the Vehicle Routing Problem with Time Windows:
- Identifies stations at stockout risk from 0-6h forecast
- Formulates VRPTW with tanker capacities and travel times
- Uses Clarke-Wright savings algorithm + local search
- Supports emergency rerouting when new stockout alerts arrive

Routes recalculate every 6 hours on updated forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("tanker_routing", component="optimiser")


@dataclass
class Tanker:
    """Tanker vehicle specification."""

    tanker_id: str
    capacity_kg: float
    current_location: tuple[float, float]  # (lat, lon)
    available: bool = True
    current_load_kg: float = 0.0


@dataclass
class StationDemand:
    """Station with demand/inventory information for routing."""

    station_id: str
    location: tuple[float, float]  # (lat, lon)
    current_inventory_kg: float
    forecast_demand_6h: float  # total predicted demand over next 6 hours
    safety_stock_kg: float  # minimum inventory to maintain
    urgency_hours: float  # hours until stockout at forecast demand rate

    @property
    def deficit_kg(self) -> float:
        """Amount of CNG needed to avoid stockout."""
        return max(0, self.forecast_demand_6h - self.current_inventory_kg + self.safety_stock_kg)


@dataclass
class TankerRoute:
    """Optimised route for a single tanker."""

    tanker_id: str
    stops: list[str]  # ordered station_ids
    delivery_amounts: dict[str, float]  # station_id -> kg to deliver
    total_distance_km: float = 0.0
    total_time_hours: float = 0.0
    total_delivery_kg: float = 0.0


class TankerRouter:
    """
    VRPTW-based tanker route optimiser.

    Uses Google OR-Tools to solve the vehicle routing problem with:
    - Time windows derived from stockout urgency
    - Vehicle capacity constraints from tanker specs
    - Depot(s) as CNG supply sources

    Args:
        depot_location: (lat, lon) of the CNG supply depot.
        avg_speed_kmh: Average tanker road speed. Default 30.
        time_limit_seconds: OR-Tools solver time limit. Default 30.
        safety_stock_hours: Hours of safety stock to maintain. Default 4.
    """

    def __init__(
        self,
        depot_location: tuple[float, float],
        avg_speed_kmh: float = 30.0,
        time_limit_seconds: int = 30,
        safety_stock_hours: float = 4.0,
    ):
        self.depot_location = depot_location
        self.avg_speed_kmh = avg_speed_kmh
        self.time_limit_seconds = time_limit_seconds
        self.safety_stock_hours = safety_stock_hours

    def _identify_at_risk_stations(
        self,
        stations: list[StationDemand],
    ) -> list[StationDemand]:
        """Identify stations at risk of stockout within 6 hours."""
        at_risk = [s for s in stations if s.deficit_kg > 0]
        # Sort by urgency (most urgent first)
        at_risk.sort(key=lambda s: s.urgency_hours)

        logger.info(
            f"Identified {len(at_risk)}/{len(stations)} stations at stockout risk"
        )
        return at_risk

    def _build_distance_matrix(
        self,
        locations: list[tuple[float, float]],
    ) -> np.ndarray:
        """Build travel time matrix (in minutes) between all locations."""
        from hp_ai_engine.utils.geo import haversine

        n = len(locations)
        matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = haversine(
                        locations[i][0], locations[i][1],
                        locations[j][0], locations[j][1],
                    )
                    # Convert distance to time in minutes
                    travel_time_min = (dist / self.avg_speed_kmh) * 60
                    matrix[i, j] = travel_time_min

        return matrix

    def optimise_routes(
        self,
        stations: list[StationDemand],
        tankers: list[Tanker],
        distance_matrix: np.ndarray | None = None,
    ) -> list[TankerRoute]:
        """
        Solve the VRPTW to produce optimised tanker routes.

        Args:
            stations: All stations with demand/inventory data.
            tankers: Available tanker fleet.
            distance_matrix: Optional precomputed distance matrix.
                           If None, computed from station coordinates.

        Returns:
            List of TankerRoute, one per assigned tanker.
        """
        # Filter to at-risk stations only
        at_risk = self._identify_at_risk_stations(stations)
        if not at_risk:
            logger.info("No stations at risk — no routes needed")
            return []

        available_tankers = [t for t in tankers if t.available]
        if not available_tankers:
            logger.warning("No tankers available for dispatch")
            return []

        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        except ImportError:
            logger.warning("OR-Tools not installed — using greedy fallback")
            return self._greedy_route(at_risk, available_tankers)

        # Build location list: [depot, station_0, station_1, ...]
        locations = [self.depot_location] + [s.location for s in at_risk]

        if distance_matrix is None:
            time_matrix = self._build_distance_matrix(locations)
        else:
            time_matrix = distance_matrix

        # Scale to integers for OR-Tools (multiply by 100 for precision)
        scale = 100
        int_matrix = (time_matrix * scale).astype(int).tolist()

        # Create routing model
        num_locations = len(locations)
        num_vehicles = len(available_tankers)
        depot_index = 0

        manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Capacity constraint
        demands = [0] + [int(s.deficit_kg) for s in at_risk]

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        capacities = [int(t.capacity_kg) for t in available_tankers]
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # no slack
            capacities,
            True,  # start cumul to zero
            "Capacity",
        )

        # Time windows
        routing.AddDimension(
            transit_callback_index,
            60 * scale,  # 60 min allowed waiting
            480 * scale,  # 8 hour max per vehicle
            False,
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        for i, station in enumerate(at_risk):
            index = manager.NodeToIndex(i + 1)  # +1 because depot is 0
            urgency_min = int(station.urgency_hours * 60 * scale)
            time_dimension.CumulVar(index).SetRange(0, max(urgency_min, 60 * scale))

        # Search parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.SAVINGS
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = self.time_limit_seconds

        # Solve
        solution = routing.SolveWithParameters(search_params)

        if solution is None:
            logger.warning("OR-Tools found no feasible solution — using greedy fallback")
            return self._greedy_route(at_risk, available_tankers)

        # Extract routes from solution
        routes = []
        for v in range(num_vehicles):
            index = routing.Start(v)
            stops = []
            deliveries = {}

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node > 0:  # skip depot
                    station = at_risk[node - 1]
                    stops.append(station.station_id)
                    deliveries[station.station_id] = station.deficit_kg
                index = solution.Value(routing.NextVar(index))

            if stops:
                total_delivery = sum(deliveries.values())
                routes.append(TankerRoute(
                    tanker_id=available_tankers[v].tanker_id,
                    stops=stops,
                    delivery_amounts=deliveries,
                    total_delivery_kg=total_delivery,
                ))

        logger.info(
            f"Routes optimised: {len(routes)} tankers assigned "
            f"to {sum(len(r.stops) for r in routes)} stations"
        )
        return routes

    def _greedy_route(
        self,
        stations: list[StationDemand],
        tankers: list[Tanker],
    ) -> list[TankerRoute]:
        """
        Simple greedy fallback: assign stations to tankers by urgency.

        Used when OR-Tools is not available or finds no feasible solution.
        """
        routes = []
        station_queue = list(stations)  # already sorted by urgency

        for tanker in tankers:
            if not station_queue:
                break

            remaining_capacity = tanker.capacity_kg
            stops = []
            deliveries = {}

            while station_queue and remaining_capacity > 0:
                station = station_queue[0]
                delivery = min(station.deficit_kg, remaining_capacity)

                if delivery > 0:
                    stops.append(station.station_id)
                    deliveries[station.station_id] = delivery
                    remaining_capacity -= delivery
                    station_queue.pop(0)
                else:
                    break

            if stops:
                routes.append(TankerRoute(
                    tanker_id=tanker.tanker_id,
                    stops=stops,
                    delivery_amounts=deliveries,
                    total_delivery_kg=sum(deliveries.values()),
                ))

        return routes

    def emergency_reroute(
        self,
        active_routes: list[TankerRoute],
        new_alert: StationDemand,
        tankers: list[Tanker],
    ) -> list[TankerRoute]:
        """
        Re-route when a new stockout alert arrives mid-cycle.

        Inserts the emergency station into the nearest tanker's route
        if capacity allows, otherwise dispatches a new tanker.

        Args:
            active_routes: Currently active tanker routes.
            new_alert: Station with urgent stockout risk.
            tankers: Full tanker fleet.

        Returns:
            Updated list of routes.
        """
        logger.warning(
            f"Emergency reroute: station {new_alert.station_id} "
            f"(urgency: {new_alert.urgency_hours:.1f}h)",
            extra={"station_id": new_alert.station_id},
        )

        # Try to insert into existing route with available capacity
        for route in active_routes:
            tanker = next((t for t in tankers if t.tanker_id == route.tanker_id), None)
            if tanker and (tanker.capacity_kg - route.total_delivery_kg) >= new_alert.deficit_kg:
                route.stops.insert(0, new_alert.station_id)  # insert as first stop
                route.delivery_amounts[new_alert.station_id] = new_alert.deficit_kg
                route.total_delivery_kg += new_alert.deficit_kg
                logger.info(
                    f"Inserted {new_alert.station_id} into tanker {tanker.tanker_id} route"
                )
                return active_routes

        # No existing route can accommodate — dispatch new tanker
        idle_tankers = [
            t for t in tankers
            if t.available and t.tanker_id not in {r.tanker_id for r in active_routes}
        ]
        if idle_tankers:
            new_route = TankerRoute(
                tanker_id=idle_tankers[0].tanker_id,
                stops=[new_alert.station_id],
                delivery_amounts={new_alert.station_id: new_alert.deficit_kg},
                total_delivery_kg=new_alert.deficit_kg,
            )
            active_routes.append(new_route)
            logger.info(f"Dispatched tanker {idle_tankers[0].tanker_id} for emergency")
        else:
            logger.error("No tankers available for emergency dispatch")

        return active_routes
