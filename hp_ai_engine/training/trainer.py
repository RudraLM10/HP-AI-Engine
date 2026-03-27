"""
Training loop for the TFT-GCN prediction engine.

Features:
- OneCycleLR scheduler
- Gradient clipping
- Early stopping on validation MAPE
- Checkpoint saving (best model + periodic saves)
- Multi-horizon metric logging per epoch
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from hp_ai_engine.training.loss import MultiHorizonLoss
from hp_ai_engine.utils.logging import get_logger
from hp_ai_engine.utils.metrics import mape as compute_mape

logger = get_logger("trainer", component="training")


@dataclass
class TrainingResult:
    """Results from a training run."""
    best_epoch: int = 0
    best_val_mape: float = float("inf")
    training_history: list[dict] = field(default_factory=list)
    total_time_seconds: float = 0.0


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    mape_short: float = 0.0
    mape_mid: float = 0.0
    mape_long: float = 0.0
    rmse_short: float = 0.0
    rmse_mid: float = 0.0
    rmse_long: float = 0.0
    total_samples: int = 0


class TFTGCNTrainer:
    """
    Training orchestrator for the TFT-GCN prediction engine.

    Args:
        model: TFTGCNPredictor model instance.
        graph_data: torch_geometric.data.Data with station graph.
        learning_rate: Initial learning rate. Default 0.001.
        max_epochs: Maximum training epochs. Default 100.
        patience: Early stopping patience. Default 10.
        min_delta: Minimum MAPE improvement to reset patience. Default 0.001.
        gradient_clip: Max gradient norm. Default 1.0.
        checkpoint_dir: Directory for saving checkpoints. Default 'checkpoints/'.
        save_top_k: Number of best checkpoints to keep. Default 3.
        device: Torch device. Default auto-detects GPU.
        loss_fn: Custom loss function. Default MultiHorizonLoss.
    """

    def __init__(
        self,
        model: nn.Module,
        graph_data,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 0.001,
        gradient_clip: float = 1.0,
        checkpoint_dir: str = "checkpoints/",
        save_top_k: int = 3,
        device: str | None = None,
        loss_fn: nn.Module | None = None,
    ):
        self.model = model
        self.graph_data = graph_data
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_top_k = save_top_k
        self.loss_fn = loss_fn or MultiHorizonLoss()

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track best checkpoints for save_top_k
        self._best_checkpoints: list[tuple[float, Path]] = []  # (mape, path)

    def _prepare_batch(self, batch: dict) -> dict:
        """Move batch tensors to device and prepare graph data."""
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value

        # Add graph data
        prepared["node_features"] = self.graph_data.x.to(self.device)
        prepared["edge_index"] = self.graph_data.edge_index.to(self.device)
        if self.graph_data.edge_attr is not None:
            prepared["edge_weight"] = self.graph_data.edge_attr.squeeze(-1).to(self.device)

        return prepared

    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        loss_accum = {"loss_short": 0.0, "loss_mid": 0.0, "loss_long": 0.0}

        for batch in train_loader:
            batch = self._prepare_batch(batch)
            optimizer.zero_grad()

            # Forward pass
            output = self.model(
                node_features=batch["node_features"],
                edge_index=batch["edge_index"],
                edge_weight=batch.get("edge_weight"),
                static_features=batch.get("static"),
                dynamic_past=batch.get("dynamic_past"),
                dynamic_future=batch.get("dynamic_future"),
                station_indices=batch.get("station_indices"),
            )

            # Multi-horizon loss
            loss, breakdown = self.loss_fn(
                pred_short=output.short_forecast,
                pred_mid=output.mid_forecast,
                pred_long=output.long_forecast,
                target_short=batch["target_short"],
                target_mid=batch["target_mid"],
                target_long=batch["target_long"],
            )

            # Backward pass
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            for k, v in breakdown.items():
                if k in loss_accum:
                    loss_accum[k] += v
            num_batches += 1

        avg_metrics = {
            "train_loss": total_loss / max(num_batches, 1),
            "train_loss_short": loss_accum["loss_short"] / max(num_batches, 1),
            "train_loss_mid": loss_accum["loss_mid"] / max(num_batches, 1),
            "train_loss_long": loss_accum["loss_long"] / max(num_batches, 1),
        }
        return avg_metrics

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict:
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_short_preds, all_short_targets = [], []

        for batch in val_loader:
            batch = self._prepare_batch(batch)

            output = self.model(
                node_features=batch["node_features"],
                edge_index=batch["edge_index"],
                edge_weight=batch.get("edge_weight"),
                static_features=batch.get("static"),
                dynamic_past=batch.get("dynamic_past"),
                dynamic_future=batch.get("dynamic_future"),
                station_indices=batch.get("station_indices"),
            )

            loss, _ = self.loss_fn(
                pred_short=output.short_forecast,
                pred_mid=output.mid_forecast,
                pred_long=output.long_forecast,
                target_short=batch["target_short"],
                target_mid=batch["target_mid"],
                target_long=batch["target_long"],
            )

            total_loss += loss.item()
            all_short_preds.append(output.short_forecast.cpu())
            all_short_targets.append(batch["target_short"].cpu())
            num_batches += 1

        # Compute MAPE on short-term (primary validation metric)
        if all_short_preds:
            preds_cat = torch.cat(all_short_preds)
            targets_cat = torch.cat(all_short_targets)
            val_mape = compute_mape(targets_cat, preds_cat)
        else:
            val_mape = float("inf")

        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_mape_short": val_mape,
        }

    def _save_checkpoint(self, epoch: int, val_mape: float) -> None:
        """Save model checkpoint with top-k management."""
        ckpt_path = self.checkpoint_dir / f"tft_gcn_epoch{epoch}_mape{val_mape:.4f}.pt"

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_mape": val_mape,
        }, ckpt_path)

        # Manage top-k checkpoints
        self._best_checkpoints.append((val_mape, ckpt_path))
        self._best_checkpoints.sort(key=lambda x: x[0])

        while len(self._best_checkpoints) > self.save_top_k:
            _, worst_path = self._best_checkpoints.pop()
            if worst_path.exists():
                worst_path.unlink()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingResult:
        """
        Execute the full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.

        Returns:
            TrainingResult with training history and best metrics.
        """
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        total_steps = len(train_loader) * self.max_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate * 10,
            total_steps=total_steps,
            pct_start=0.3,
        )

        result = TrainingResult()
        patience_counter = 0
        start_time = time.time()

        logger.info(f"Starting training: {self.max_epochs} epochs, device={self.device}")

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler)

            # Validate
            val_metrics = self._validate(val_loader)

            # Combine metrics
            epoch_metrics = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_s": time.time() - epoch_start,
            }
            result.training_history.append(epoch_metrics)

            # Check for improvement
            val_mape = val_metrics["val_mape_short"]
            if val_mape < result.best_val_mape - self.min_delta:
                result.best_val_mape = val_mape
                result.best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(epoch, val_mape)
                logger.info(
                    f"Epoch {epoch}: val_mape={val_mape:.4f} (new best) | "
                    f"train_loss={train_metrics['train_loss']:.4f}"
                )
            else:
                patience_counter += 1
                if epoch % 5 == 0:
                    logger.info(
                        f"Epoch {epoch}: val_mape={val_mape:.4f} | "
                        f"patience={patience_counter}/{self.patience}"
                    )

            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        result.total_time_seconds = time.time() - start_time
        logger.info(
            f"Training complete: best_epoch={result.best_epoch}, "
            f"best_mape={result.best_val_mape:.4f}, "
            f"time={result.total_time_seconds:.1f}s"
        )

        return result

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> EvaluationResult:
        """
        Evaluate model on test set.

        Returns:
            EvaluationResult with per-horizon MAPE and RMSE.
        """
        self.model.eval()
        all_preds = {"short": [], "mid": [], "long": []}
        all_targets = {"short": [], "mid": [], "long": []}

        for batch in test_loader:
            batch = self._prepare_batch(batch)

            output = self.model(
                node_features=batch["node_features"],
                edge_index=batch["edge_index"],
                edge_weight=batch.get("edge_weight"),
                static_features=batch.get("static"),
                dynamic_past=batch.get("dynamic_past"),
                dynamic_future=batch.get("dynamic_future"),
                station_indices=batch.get("station_indices"),
            )

            all_preds["short"].append(output.short_forecast.cpu())
            all_preds["mid"].append(output.mid_forecast.cpu())
            all_preds["long"].append(output.long_forecast.cpu())
            all_targets["short"].append(batch["target_short"].cpu())
            all_targets["mid"].append(batch["target_mid"].cpu())
            all_targets["long"].append(batch["target_long"].cpu())

        from hp_ai_engine.utils.metrics import mape as calc_mape, rmse as calc_rmse

        result = EvaluationResult()
        for horizon in ["short", "mid", "long"]:
            preds = torch.cat(all_preds[horizon])
            targets = torch.cat(all_targets[horizon])
            setattr(result, f"mape_{horizon}", calc_mape(targets, preds))
            setattr(result, f"rmse_{horizon}", calc_rmse(targets, preds))

        result.total_samples = len(test_loader.dataset)
        logger.info(
            f"Evaluation: mape_short={result.mape_short:.2f}%, "
            f"mape_mid={result.mape_mid:.2f}%, mape_long={result.mape_long:.2f}%"
        )

        return result

    def load_best_checkpoint(self) -> None:
        """Load the best checkpoint by validation MAPE."""
        if not self._best_checkpoints:
            # Scan checkpoint directory
            ckpts = list(self.checkpoint_dir.glob("tft_gcn_*.pt"))
            if not ckpts:
                logger.warning("No checkpoints found")
                return
            # Pick the one with lowest MAPE from filename
            best = min(ckpts, key=lambda p: float(p.stem.split("mape")[-1]))
        else:
            _, best = self._best_checkpoints[0]

        ckpt = torch.load(best, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {best.name} (epoch={ckpt['epoch']})")
