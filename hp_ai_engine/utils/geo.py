"""
Geographic utility functions for HP AI Engine.

Provides distance calculations (Haversine and road-network-based),
OpenStreetMap network loading, and spatial helper functions.
"""

from __future__ import annotations

import math
from typing import Sequence

import networkx as nx
import numpy as np

# osmnx is an optional heavy dependency — import lazily
_osmnx = None


def _get_osmnx():
    """Lazy-load osmnx to avoid import overhead when not needed."""
    global _osmnx
    if _osmnx is None:
        import osmnx as ox
        _osmnx = ox
    return _osmnx


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees).
        lat2, lon2: Latitude and longitude of point 2 (degrees).

    Returns:
        Distance in kilometres.
    """
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS_KM * c


def haversine_matrix(
    coords: Sequence[tuple[float, float]],
) -> np.ndarray:
    """
    Compute pairwise Haversine distance matrix.

    Args:
        coords: List of (latitude, longitude) pairs.

    Returns:
        Symmetric distance matrix of shape [N, N] in kilometres.
    """
    n = len(coords)
    matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix


# ---------------------------------------------------------------------------
# OSM Road Network
# ---------------------------------------------------------------------------

def get_osm_network(
    city_name: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    network_type: str = "drive",
) -> nx.MultiDiGraph:
    """
    Download the road network graph from OpenStreetMap.

    Args:
        city_name: Name of the city (e.g. "Mumbai, India"). Used if bbox is None.
        bbox: Bounding box (north, south, east, west) in degrees.
        network_type: OSM network type ('drive', 'walk', 'bike', 'all').

    Returns:
        NetworkX MultiDiGraph with road network.
    """
    ox = _get_osmnx()

    if bbox is not None:
        return ox.graph_from_bbox(
            north=bbox[0], south=bbox[1], east=bbox[2], west=bbox[3],
            network_type=network_type,
        )
    elif city_name is not None:
        return ox.graph_from_place(city_name, network_type=network_type)
    else:
        raise ValueError("Either city_name or bbox must be provided.")


def road_distance(
    graph: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> float:
    """
    Compute shortest road distance between two points using the OSM network.

    Args:
        graph: OSM road network graph.
        origin: (latitude, longitude) of origin.
        destination: (latitude, longitude) of destination.

    Returns:
        Road distance in kilometres. Returns float('inf') if no path exists.
    """
    ox = _get_osmnx()

    orig_node = ox.nearest_nodes(graph, origin[1], origin[0])
    dest_node = ox.nearest_nodes(graph, destination[1], destination[0])

    try:
        distance_m = nx.shortest_path_length(graph, orig_node, dest_node, weight="length")
        return distance_m / 1000.0
    except nx.NetworkXNoPath:
        return float("inf")


def road_distance_matrix(
    coords: Sequence[tuple[float, float]],
    graph: nx.MultiDiGraph,
) -> np.ndarray:
    """
    Compute pairwise road distance matrix using the OSM network.

    Args:
        coords: List of (latitude, longitude) pairs.
        graph: OSM road network graph.

    Returns:
        Distance matrix of shape [N, N] in kilometres.
        Unreachable pairs have distance float('inf').
    """
    n = len(coords)
    matrix = np.full((n, n), fill_value=float("inf"), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)

    ox = _get_osmnx()
    nodes = [ox.nearest_nodes(graph, coord[1], coord[0]) for coord in coords]

    for i in range(n):
        for j in range(i + 1, n):
            try:
                dist_m = nx.shortest_path_length(graph, nodes[i], nodes[j], weight="length")
                dist_km = dist_m / 1000.0
                matrix[i, j] = dist_km
                matrix[j, i] = dist_km
            except nx.NetworkXNoPath:
                pass  # already inf

    return matrix


def connectivity_coefficient(
    graph: nx.MultiDiGraph,
    origin: tuple[float, float],
    destination: tuple[float, float],
    max_paths: int = 5,
) -> int:
    """
    Count the number of distinct shortest paths between two nodes.

    Used to augment edge weights in the GCN adjacency matrix — stations
    connected by more road paths have stronger edges.

    Args:
        graph: OSM road network graph.
        origin: (latitude, longitude).
        destination: (latitude, longitude).
        max_paths: Maximum paths to enumerate (for performance).

    Returns:
        Number of distinct shortest paths (capped at max_paths).
    """
    ox = _get_osmnx()

    orig_node = ox.nearest_nodes(graph, origin[1], origin[0])
    dest_node = ox.nearest_nodes(graph, destination[1], destination[0])

    try:
        paths = list(nx.all_shortest_paths(graph, orig_node, dest_node, weight="length"))
        return min(len(paths), max_paths)
    except nx.NetworkXNoPath:
        return 0
