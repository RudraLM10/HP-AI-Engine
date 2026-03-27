"""
Automated retraining pipeline for HP AI Engine.

When drift detection triggers a retraining event:
1. Refreshes the training data window (sliding window of recent data)
2. Snapshots the current model as a rollback checkpoint
3. Retrains with a warm start from the current weights
4. Validates on a holdout set
5. If validation MAPE improves: promote new model, archive old
6. If validation MAPE degrades: roll back to snapshot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("retraining", component="scalability")


@dataclass
class RetrainingResult:
    """Summary of a retraining attempt."""

    triggered_by: str          # e.g. "covariate_drift", "concept_drift", "scheduled"
    old_mape: float
    new_mape: float
    improvement_pct: float
    model_promoted: bool
    rolled_back: bool
    epochs_trained: int


class AutomatedRetrainer:
    """
    Automated retraining pipeline for continuous model improvement.

    Designed for unmanned operation: drift detection triggers retraining,
    and validation ensures no regressions are deployed.

    Args:
        checkpoint_dir: Directory for model checkpoints.
        max_retrain_epochs: Maximum epochs per retraining. Default 30.
        learning_rate: Learning rate for retraining (warm start). Default 0.0001.
        min_improvement_pct: Minimum MAPE improvement to promote. Default 1.0.
        gradient_clip: Max gradient norm. Default 1.0.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/",
        max_retrain_epochs: int = 30,
        learning_rate: float = 0.0001,
        min_improvement_pct: float = 1.0,
        gradient_clip: float = 1.0,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_retrain_epochs = max_retrain_epochs
        self.learning_rate = learning_rate
        self.min_improvement_pct = min_improvement_pct
        self.gradient_clip = gradient_clip
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_model(self, model: nn.Module, tag: str) -> Path:
        """Save a snapshot of the current model state."""
        path = self.checkpoint_dir / f"snapshot_{tag}.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)
        logger.info(f"Saved model snapshot: {path.name}")
        return path

    def _restore_snapshot(self, model: nn.Module, snapshot_path: Path) -> None:
        """Restore model from a snapshot."""
        device = next(model.parameters()).device
        ckpt = torch.load(snapshot_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Restored model from snapshot: {snapshot_path.name}")

    def retrain(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        graph_data,
        loss_fn: nn.Module,
        evaluate_fn: Callable[[nn.Module, DataLoader], float],
        trigger_reason: str = "scheduled",
    ) -> RetrainingResult:
        """
        Execute one retraining cycle.

        Args:
            model: Current production model.
            train_loader: Updated training DataLoader (recent data window).
            val_loader: Validation DataLoader.
            graph_data: Station graph data.
            loss_fn: Loss function.
            evaluate_fn: Function(model, val_loader) -> MAPE score.
            trigger_reason: What triggered retraining.

        Returns:
            RetrainingResult with metrics and decision.
        """
        device = next(model.parameters()).device
        logger.info(f"Retraining triggered by: {trigger_reason}")

        # Step 1: Evaluate current model
        model.eval()
        old_mape = evaluate_fn(model, val_loader)
        logger.info(f"Current model MAPE: {old_mape:.4f}")

        # Step 2: Snapshot current model
        snapshot = self._snapshot_model(model, tag="pre_retrain")

        # Step 3: Warm-start retraining
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_retrain_epochs
        )

        best_val_mape = old_mape
        epochs_trained = 0

        for epoch in range(1, self.max_retrain_epochs + 1):
            model.train()
            for batch in train_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                optimizer.zero_grad()

                fwd_kwargs = {
                    "node_features": graph_data.x.to(device),
                    "edge_index": graph_data.edge_index.to(device),
                    "static_features": batch.get("static"),
                    "dynamic_past": batch.get("dynamic_past"),
                    "dynamic_future": batch.get("dynamic_future"),
                    "station_indices": batch.get("station_indices"),
                }
                if graph_data.edge_attr is not None:
                    fwd_kwargs["edge_weight"] = graph_data.edge_attr.squeeze(-1).to(device)

                output = model(**fwd_kwargs)

                loss, _ = loss_fn(
                    pred_short=output.short_forecast,
                    pred_mid=output.mid_forecast,
                    pred_long=output.long_forecast,
                    target_short=batch["target_short"],
                    target_mid=batch["target_mid"],
                    target_long=batch["target_long"],
                )

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                optimizer.step()

            scheduler.step()
            epochs_trained = epoch

            # Validate every 5 epochs
            if epoch % 5 == 0:
                model.eval()
                current_mape = evaluate_fn(model, val_loader)
                logger.info(f"Retrain epoch {epoch}: MAPE={current_mape:.4f}")

                if current_mape < best_val_mape:
                    best_val_mape = current_mape

        # Step 4: Final evaluation
        model.eval()
        new_mape = evaluate_fn(model, val_loader)
        improvement_pct = ((old_mape - new_mape) / max(old_mape, 1e-8)) * 100

        # Step 5: Promote or rollback
        if improvement_pct >= self.min_improvement_pct:
            model_promoted = True
            rolled_back = False
            # Save promoted model
            self._snapshot_model(model, tag="promoted")
            logger.info(
                f"Retraining successful: MAPE {old_mape:.4f} → {new_mape:.4f} "
                f"(↓{improvement_pct:.1f}%). Model promoted."
            )
        else:
            model_promoted = False
            rolled_back = True
            self._restore_snapshot(model, snapshot)
            logger.info(
                f"Retraining did not improve sufficiently: "
                f"MAPE {old_mape:.4f} → {new_mape:.4f} "
                f"({improvement_pct:+.1f}%). Rolled back."
            )

        return RetrainingResult(
            triggered_by=trigger_reason,
            old_mape=round(old_mape, 4),
            new_mape=round(new_mape, 4),
            improvement_pct=round(improvement_pct, 2),
            model_promoted=model_promoted,
            rolled_back=rolled_back,
            epochs_trained=epochs_trained,
        )
