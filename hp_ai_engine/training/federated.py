"""
Federated Learning (FedAvg) for HP AI Engine.

Implements Federated Averaging for city cluster models:
- Each city cluster trains independently on its own data
- Periodically, model weights are aggregated centrally
- Aggregated weights are distributed back to clusters
- Each cluster validates the new global model locally
- If performance degrades, the cluster rolls back to its previous weights

This architecture ensures:
1. No station goes offline if the central server fails
2. Raw data never leaves city clusters (data sovereignty for PSU)
3. All clusters benefit from the collective knowledge of the network
"""

from __future__ import annotations

from copy import deepcopy
from typing import OrderedDict

import torch
import torch.nn as nn

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("federated", component="training")


class FedAvgAggregator:
    """
    Federated Averaging aggregation server.

    Computes a weighted average of model parameters from multiple
    city cluster models, where weights are proportional to each
    cluster's data volume.

    Args:
        rollback_threshold: Maximum allowed MAPE degradation before
                          a cluster rolls back. Default 0.05 (5% increase).
    """

    def __init__(self, rollback_threshold: float = 0.05):
        self.rollback_threshold = rollback_threshold
        self.aggregation_round = 0
        self._global_state: OrderedDict | None = None

    def aggregate(
        self,
        cluster_weights: list[OrderedDict[str, torch.Tensor]],
        cluster_sizes: list[int],
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Compute weighted average of cluster model weights.

        Args:
            cluster_weights: List of model state_dicts from each cluster.
            cluster_sizes: Data volume (number of samples) per cluster.

        Returns:
            Aggregated state_dict (weighted average).
        """
        if not cluster_weights:
            raise ValueError("No cluster weights provided for aggregation.")

        total_samples = sum(cluster_sizes)
        if total_samples == 0:
            raise ValueError("Total cluster data volume is zero.")

        # Compute normalised weights
        normalised_weights = [s / total_samples for s in cluster_sizes]

        # Weighted average of all parameters
        aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
        param_keys = cluster_weights[0].keys()

        for key in param_keys:
            weighted_params = [
                w * cluster_weights[i][key].float()
                for i, w in enumerate(normalised_weights)
                if key in cluster_weights[i]
            ]
            if weighted_params:
                aggregated[key] = torch.stack(weighted_params).sum(dim=0)

        self._global_state = aggregated
        self.aggregation_round += 1

        logger.info(
            f"FedAvg round {self.aggregation_round}: aggregated {len(cluster_weights)} "
            f"clusters ({total_samples} total samples)",
            extra={"round": self.aggregation_round, "num_clusters": len(cluster_weights)},
        )

        return aggregated

    @staticmethod
    def should_rollback(
        old_local_mape: float,
        new_global_mape: float,
        threshold: float = 0.05,
    ) -> bool:
        """
        Determine whether a cluster should roll back to its previous weights.

        Rollback occurs when the global model degrades this cluster's
        performance by more than the threshold.

        Args:
            old_local_mape: MAPE with the cluster's own local model.
            new_global_mape: MAPE with the new global aggregated model.
            threshold: Maximum allowed MAPE increase (as a fraction, e.g. 0.05 = 5%).

        Returns:
            True if the cluster should roll back.
        """
        if old_local_mape == 0:
            return new_global_mape > threshold

        degradation = (new_global_mape - old_local_mape) / max(old_local_mape, 1e-8)
        return degradation > threshold

    @property
    def global_state(self) -> OrderedDict[str, torch.Tensor] | None:
        """Current global aggregated state dict."""
        return self._global_state


class CityClusterClient:
    """
    Federated learning client for a single city cluster.

    Manages:
    - Local model training on cluster data
    - Sending weights to the aggregation server
    - Receiving and evaluating global model updates
    - Rolling back if global update degrades local performance

    Args:
        cluster_id: Unique identifier for this city cluster.
        model: Local TFTGCNPredictor model instance.
        rollback_threshold: MAPE degradation threshold for rollback.
    """

    def __init__(
        self,
        cluster_id: str,
        model: nn.Module,
        rollback_threshold: float = 0.05,
    ):
        self.cluster_id = cluster_id
        self.model = model
        self.rollback_threshold = rollback_threshold

        # Store local weights for potential rollback
        self._local_state: OrderedDict | None = None
        self._local_mape: float = float("inf")

    def save_local_state(self, current_mape: float) -> None:
        """Snapshot current model weights and performance."""
        self._local_state = deepcopy(self.model.state_dict())
        self._local_mape = current_mape

    def get_weights(self) -> OrderedDict[str, torch.Tensor]:
        """Return current model weights for aggregation."""
        return deepcopy(self.model.state_dict())

    def receive_global_weights(
        self,
        global_state: OrderedDict[str, torch.Tensor],
    ) -> None:
        """Load global aggregated weights into the local model."""
        self.model.load_state_dict(global_state)
        logger.info(
            f"Cluster '{self.cluster_id}': received global weights",
            extra={"cluster_id": self.cluster_id},
        )

    def evaluate_and_decide(
        self,
        val_mape: float,
    ) -> bool:
        """
        Evaluate the global model on local data and decide: keep or rollback.

        Args:
            val_mape: Validation MAPE after loading global weights.

        Returns:
            True if global weights were kept, False if rolled back.
        """
        should_rollback = FedAvgAggregator.should_rollback(
            self._local_mape, val_mape, self.rollback_threshold
        )

        if should_rollback:
            if self._local_state is not None:
                self.model.load_state_dict(self._local_state)
                logger.info(
                    f"Cluster '{self.cluster_id}': rolled back "
                    f"(global mape={val_mape:.4f} > local mape={self._local_mape:.4f})",
                    extra={"cluster_id": self.cluster_id},
                )
            return False
        else:
            # Accept global weights and update local baseline
            self._local_state = deepcopy(self.model.state_dict())
            self._local_mape = val_mape
            logger.info(
                f"Cluster '{self.cluster_id}': accepted global weights "
                f"(mape={val_mape:.4f})",
                extra={"cluster_id": self.cluster_id},
            )
            return True


class FederatedCoordinator:
    """
    Orchestrates the entire federated learning cycle.

    Usage:
        coordinator = FederatedCoordinator()
        coordinator.register_cluster("Mumbai_West", model_mw)
        coordinator.register_cluster("Mumbai_East", model_me)

        # Run one aggregation round
        coordinator.run_round(evaluate_fn=my_eval_fn)
    """

    def __init__(self, rollback_threshold: float = 0.05):
        self.aggregator = FedAvgAggregator(rollback_threshold=rollback_threshold)
        self.clusters: dict[str, CityClusterClient] = {}

    def register_cluster(
        self,
        cluster_id: str,
        model: nn.Module,
        initial_mape: float = float("inf"),
    ) -> None:
        """Register a city cluster client."""
        client = CityClusterClient(cluster_id, model, self.aggregator.rollback_threshold)
        client.save_local_state(initial_mape)
        self.clusters[cluster_id] = client
        logger.info(f"Registered cluster: {cluster_id}")

    def run_round(
        self,
        evaluate_fn: callable,
        min_clusters: int = 3,
    ) -> dict[str, bool]:
        """
        Execute one federated learning round.

        Args:
            evaluate_fn: Function(cluster_id, model) -> float (MAPE).
            min_clusters: Minimum participating clusters required.

        Returns:
            Dict mapping cluster_id -> bool (True=accepted, False=rolled back).
        """
        if len(self.clusters) < min_clusters:
            logger.warning(
                f"Not enough clusters for aggregation: "
                f"{len(self.clusters)} < {min_clusters}"
            )
            return {}

        # Collect weights and sizes
        weights = []
        sizes = []
        cluster_ids = []

        for cid, client in self.clusters.items():
            weights.append(client.get_weights())
            # Use 1 as default size if not available
            sizes.append(1)
            cluster_ids.append(cid)

        # Aggregate
        global_state = self.aggregator.aggregate(weights, sizes)

        # Distribute and evaluate
        results = {}
        for cid, client in self.clusters.items():
            client.save_local_state(client._local_mape)
            client.receive_global_weights(global_state)

            # Evaluate global model on local data
            new_mape = evaluate_fn(cid, client.model)

            # Decide: keep or rollback
            accepted = client.evaluate_and_decide(new_mape)
            results[cid] = accepted

        accepted_count = sum(1 for v in results.values() if v)
        logger.info(
            f"Round complete: {accepted_count}/{len(results)} clusters accepted global weights"
        )

        return results
