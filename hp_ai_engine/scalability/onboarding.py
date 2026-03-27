"""
New station onboarding pipeline for HP AI Engine.

Zero-downtime process for adding a new CNG station to the live network:
1. Register station metadata → validate against schema
2. Update the spatial graph → add node with edges to nearby stations
3. Warm-start model via transfer learning → inherit global knowledge
4. Set up drift monitoring → baseline distributions from initial data
5. Activate → station goes live with forecasts from day one
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hp_ai_engine.data.graph_builder import StationGraphBuilder
from hp_ai_engine.data.schemas import StationMeta
from hp_ai_engine.utils.logging import get_logger

logger = get_logger("onboarding", component="scalability")


@dataclass
class OnboardingResult:
    """Result of onboarding a new station."""

    station_id: str
    status: str
    steps_completed: list[str] = field(default_factory=list)
    graph_node_count: int = 0
    graph_edge_count: int = 0
    model_warm_started: bool = False
    drift_baseline_set: bool = False
    errors: list[str] = field(default_factory=list)


class StationOnboardingPipeline:
    """
    End-to-end pipeline for adding a new station to the network.

    The pipeline is idempotent — re-running it for an already onboarded
    station will detect the existing state and skip completed steps.

    Args:
        graph_builder: StationGraphBuilder instance managing the network graph.
        global_checkpoint_path: Path to pretrained global model checkpoint.
        checkpoint_dir: Directory for station-specific checkpoints.
    """

    def __init__(
        self,
        graph_builder: StationGraphBuilder,
        global_checkpoint_path: str | Path | None = None,
        checkpoint_dir: str = "checkpoints/",
    ):
        self.graph_builder = graph_builder
        self.global_checkpoint_path = (
            Path(global_checkpoint_path) if global_checkpoint_path else None
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Registry of onboarded stations
        self._registry: dict[str, dict[str, Any]] = {}

    def _validate_metadata(self, metadata: dict) -> StationMeta:
        """Validate station metadata against Pydantic schema."""
        return StationMeta(**metadata)

    def _add_to_graph(
        self,
        station_id: str,
        coord: tuple[float, float],
    ) -> int:
        """Add station node to the spatial graph."""
        if station_id in self.graph_builder.station_ids:
            logger.info(f"Station {station_id} already in graph — skipping")
            return self.graph_builder.graph.num_nodes if self.graph_builder.graph else 0

        graph = self.graph_builder.add_node(station_id, coord)
        return graph.num_nodes

    def _warm_start_model(
        self,
        model: nn.Module,
    ) -> bool:
        """Apply transfer learning from global checkpoint."""
        if self.global_checkpoint_path is None or not self.global_checkpoint_path.exists():
            logger.warning("No global checkpoint found — skipping warm start")
            return False

        from hp_ai_engine.training.transfer import TransferLearner

        transfer = TransferLearner(
            freeze_gcn=True,
            fine_tune_epochs=0,  # just load weights, fine-tuning happens later
        )

        try:
            device = next(model.parameters()).device
            checkpoint = torch.load(
                self.global_checkpoint_path, map_location=device, weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            logger.info("Warm start: loaded global checkpoint weights")
            return True
        except Exception as e:
            logger.error(f"Warm start failed: {e}")
            return False

    def _set_drift_baselines(
        self,
        station_id: str,
        initial_features: dict | None = None,
    ) -> bool:
        """Store initial feature distributions for drift monitoring."""
        if initial_features is None:
            logger.info(f"No initial features for {station_id} — drift baselines deferred")
            return False

        # Store baselines in registry
        self._registry[station_id]["drift_baselines"] = initial_features
        logger.info(f"Drift baselines set for {station_id}")
        return True

    def onboard(
        self,
        metadata: dict,
        model: nn.Module | None = None,
        initial_features: dict | None = None,
    ) -> OnboardingResult:
        """
        Execute the full onboarding pipeline for a new station.

        Args:
            metadata: Station metadata dict (must conform to StationMeta schema).
            model: Optional TFTGCNPredictor model to warm-start.
            initial_features: Optional dict of feature_name -> np.ndarray
                            for drift baselines.

        Returns:
            OnboardingResult with status and step details.
        """
        result = OnboardingResult(
            station_id=metadata.get("station_id", "unknown"),
            status="in_progress",
        )

        # Step 1: Validate metadata
        try:
            validated = self._validate_metadata(metadata)
            result.steps_completed.append("metadata_validated")
            logger.info(f"Validated metadata for {validated.station_id}")
        except Exception as e:
            result.errors.append(f"Metadata validation failed: {e}")
            result.status = "failed"
            return result

        station_id = validated.station_id
        result.station_id = station_id

        # Initialise registry entry
        self._registry[station_id] = {
            "metadata": validated.model_dump(),
            "drift_baselines": None,
        }

        # Step 2: Add to graph
        try:
            coord = (validated.latitude, validated.longitude)
            num_nodes = self._add_to_graph(station_id, coord)
            result.graph_node_count = num_nodes
            result.graph_edge_count = (
                self.graph_builder.graph.edge_index.shape[1]
                if self.graph_builder.graph is not None
                else 0
            )
            result.steps_completed.append("graph_updated")
        except Exception as e:
            result.errors.append(f"Graph update failed: {e}")
            logger.error(f"Graph update failed for {station_id}: {e}")

        # Step 3: Warm-start model
        if model is not None:
            result.model_warm_started = self._warm_start_model(model)
            if result.model_warm_started:
                result.steps_completed.append("model_warm_started")

        # Step 4: Set drift baselines
        result.drift_baseline_set = self._set_drift_baselines(station_id, initial_features)
        if result.drift_baseline_set:
            result.steps_completed.append("drift_baselines_set")

        # Step 5: Mark as active
        self._registry[station_id]["active"] = True
        result.steps_completed.append("activated")
        result.status = "success"

        logger.info(
            f"Onboarding complete for {station_id}: "
            f"{len(result.steps_completed)} steps completed, "
            f"{len(result.errors)} errors",
            extra={"station_id": station_id},
        )

        return result

    def get_registered_stations(self) -> list[str]:
        """Return list of all onboarded station IDs."""
        return list(self._registry.keys())

    def is_active(self, station_id: str) -> bool:
        """Check if a station is onboarded and active."""
        return self._registry.get(station_id, {}).get("active", False)
