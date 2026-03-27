"""
GCN adjacency matrix builder for HP AI Engine.

Constructs a weighted graph where nodes are stations and edges encode
spatial relationships via inverse-squared road distances augmented by
OpenStreetMap connectivity coefficients.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from hp_ai_engine.utils.geo import (
    connectivity_coefficient,
    haversine_matrix,
    road_distance_matrix,
)
from hp_ai_engine.utils.logging import get_logger

logger = get_logger("graph_builder", component="data")


class StationGraphBuilder:
    """
    Constructs and maintains a dynamic weighted adjacency matrix for the station network.

    Nodes: CNG stations
    Edge weights: (1 / (road_distance_km² + ε)) × connectivity_coefficient

    The connectivity_coefficient is the number of distinct shortest road paths
    between two stations (capped at max_paths), providing a richer spatial signal
    than distance alone.

    Args:
        max_distance_km: Maximum distance to consider an edge. Beyond this,
                         stations are considered unconnected. Default 50 km.
        epsilon: Small value added to distances to avoid division by zero.
        use_road_distance: If True, compute road distances via OSM network.
                          If False, use Haversine (faster, no network needed).
        max_paths: Maximum distinct paths to count for connectivity coefficient.
    """

    def __init__(
        self,
        max_distance_km: float = 50.0,
        epsilon: float = 1e-6,
        use_road_distance: bool = False,
        max_paths: int = 5,
    ):
        self.max_distance_km = max_distance_km
        self.epsilon = epsilon
        self.use_road_distance = use_road_distance
        self.max_paths = max_paths

        # Internal state
        self._station_ids: list[str] = []
        self._coords: list[tuple[float, float]] = []
        self._graph: Data | None = None

    def build_static_graph(
        self,
        station_ids: list[str],
        coords: list[tuple[float, float]],
        road_network=None,
    ) -> Data:
        """
        Build the full adjacency matrix from scratch.

        Args:
            station_ids: List of station identifiers (defines node ordering).
            coords: List of (latitude, longitude) for each station.
            road_network: Optional OSM networkx graph for road distances.

        Returns:
            torch_geometric.data.Data with edge_index, edge_attr, and num_nodes.
        """
        self._station_ids = list(station_ids)
        self._coords = list(coords)
        n = len(station_ids)

        logger.info(f"Building station graph: {n} nodes")

        # Compute distance matrix
        if self.use_road_distance and road_network is not None:
            dist_matrix = road_distance_matrix(coords, road_network)
        else:
            dist_matrix = haversine_matrix(coords)

        # Build edge list
        edge_sources: list[int] = []
        edge_targets: list[int] = []
        edge_weights: list[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                dist = dist_matrix[i, j]
                if dist > self.max_distance_km or dist == float("inf"):
                    continue

                # Inverse squared distance weight
                weight = 1.0 / (dist ** 2 + self.epsilon)

                # Connectivity coefficient (if road network available)
                if self.use_road_distance and road_network is not None:
                    conn = connectivity_coefficient(
                        road_network, coords[i], coords[j], self.max_paths
                    )
                    weight *= max(conn, 1)

                # Bidirectional edges
                edge_sources.extend([i, j])
                edge_targets.extend([j, i])
                edge_weights.extend([weight, weight])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)

        # Placeholder node features (will be filled by dataset/model)
        x = torch.zeros(n, 1, dtype=torch.float32)

        self._graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n,
        )

        logger.info(
            f"Graph built: {n} nodes, {edge_index.shape[1]} edges",
            extra={"nodes": n, "edges": edge_index.shape[1]},
        )
        return self._graph

    def add_node(
        self,
        station_id: str,
        coord: tuple[float, float],
        road_network=None,
    ) -> Data:
        """
        Add a new station node to an existing graph.

        Computes edges between the new node and all existing nodes
        within max_distance_km, then appends to the graph.

        Args:
            station_id: New station identifier.
            coord: (latitude, longitude) of the new station.
            road_network: Optional OSM network for road distances.

        Returns:
            Updated torch_geometric.data.Data.
        """
        if self._graph is None:
            raise RuntimeError("No existing graph. Call build_static_graph first.")

        new_idx = len(self._station_ids)
        self._station_ids.append(station_id)
        self._coords.append(coord)

        new_sources: list[int] = []
        new_targets: list[int] = []
        new_weights: list[float] = []

        for i, existing_coord in enumerate(self._coords[:-1]):
            from hp_ai_engine.utils.geo import haversine
            dist = haversine(coord[0], coord[1], existing_coord[0], existing_coord[1])

            if dist > self.max_distance_km:
                continue

            weight = 1.0 / (dist ** 2 + self.epsilon)

            new_sources.extend([new_idx, i])
            new_targets.extend([i, new_idx])
            new_weights.extend([weight, weight])

        # Expand graph
        old_edge_index = self._graph.edge_index
        old_edge_attr = self._graph.edge_attr

        if new_sources:
            new_edge_index = torch.tensor([new_sources, new_targets], dtype=torch.long)
            new_edge_attr = torch.tensor(new_weights, dtype=torch.float32).unsqueeze(-1)

            edge_index = torch.cat([old_edge_index, new_edge_index], dim=1)
            edge_attr = torch.cat([old_edge_attr, new_edge_attr], dim=0)
        else:
            edge_index = old_edge_index
            edge_attr = old_edge_attr

        # Expand node features
        x = torch.zeros(len(self._station_ids), 1, dtype=torch.float32)

        self._graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self._station_ids),
        )

        logger.info(
            f"Added node '{station_id}': graph now {self._graph.num_nodes} nodes, "
            f"{edge_index.shape[1]} edges"
        )
        return self._graph

    def remove_node(self, station_id: str) -> Data:
        """
        Remove a station node from the graph.

        Re-indexes all remaining nodes and filters edges accordingly.

        Args:
            station_id: Station to remove.

        Returns:
            Updated torch_geometric.data.Data.
        """
        if station_id not in self._station_ids:
            raise ValueError(f"Station '{station_id}' not in graph.")

        remove_idx = self._station_ids.index(station_id)
        self._station_ids.pop(remove_idx)
        self._coords.pop(remove_idx)

        # Filter edges
        old_edge_index = self._graph.edge_index
        old_edge_attr = self._graph.edge_attr

        # Keep edges that don't involve the removed node
        mask = (old_edge_index[0] != remove_idx) & (old_edge_index[1] != remove_idx)
        edge_index = old_edge_index[:, mask]
        edge_attr = old_edge_attr[mask]

        # Re-index: shift indices above the removed node down by 1
        edge_index = edge_index.clone()
        edge_index[edge_index > remove_idx] -= 1

        x = torch.zeros(len(self._station_ids), 1, dtype=torch.float32)

        self._graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self._station_ids),
        )

        logger.info(f"Removed node '{station_id}': graph now {self._graph.num_nodes} nodes")
        return self._graph

    @property
    def station_ids(self) -> list[str]:
        """Current list of station IDs in node order."""
        return list(self._station_ids)

    @property
    def graph(self) -> Data | None:
        """Current graph, or None if not yet built."""
        return self._graph

    def get_adjacency_matrix(self) -> np.ndarray:
        """Return the adjacency matrix as a dense numpy array."""
        if self._graph is None:
            raise RuntimeError("No graph built yet.")

        n = self._graph.num_nodes
        adj = np.zeros((n, n), dtype=np.float64)
        edge_index = self._graph.edge_index.numpy()
        edge_attr = self._graph.edge_attr.numpy().flatten()

        for k in range(edge_index.shape[1]):
            i, j = edge_index[0, k], edge_index[1, k]
            adj[i, j] = edge_attr[k]

        return adj
