"""
Transfer learning for new station warm-start.

When a new CNG station is added to the network, it has little to no
historical data. Transfer learning from the global pretrained model
provides a warm start — the new station inherits knowledge from hundreds
of other stations, then fine-tunes on its own data as it accumulates.

This reduces convergence from 6-12 months (training from scratch) to
2-3 weeks of live operation.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("transfer", component="training")


class TransferLearner:
    """
    Warm-start a model for a new station using transfer learning.

    Process:
    1. Load the global pretrained TFT-GCN checkpoint
    2. Freeze GCN layers (spatial knowledge transfers directly)
    3. Reset the final decoder heads (station-specific outputs)
    4. Fine-tune on new station's data for a configurable number of epochs
    5. Optionally unfreeze GCN layers after initial convergence

    Args:
        freeze_gcn: Whether to freeze GCN layers during fine-tuning.
        fine_tune_epochs: Number of fine-tuning epochs. Default 20.
        learning_rate: Learning rate for fine-tuning (lower than original). Default 0.0001.
        unfreeze_after: Unfreeze GCN after this many epochs. None = never unfreeze.
        device: Torch device.
    """

    def __init__(
        self,
        freeze_gcn: bool = True,
        fine_tune_epochs: int = 20,
        learning_rate: float = 0.0001,
        unfreeze_after: int | None = 10,
        device: str | None = None,
    ):
        self.freeze_gcn = freeze_gcn
        self.fine_tune_epochs = fine_tune_epochs
        self.learning_rate = learning_rate
        self.unfreeze_after = unfreeze_after
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _freeze_gcn_params(self, model: nn.Module) -> None:
        """Freeze all parameters in the GCN encoder."""
        if hasattr(model, "gcn"):
            for param in model.gcn.parameters():
                param.requires_grad = False
            logger.info("Froze GCN encoder parameters")

    def _unfreeze_gcn_params(self, model: nn.Module) -> None:
        """Unfreeze GCN parameters for full fine-tuning."""
        if hasattr(model, "gcn"):
            for param in model.gcn.parameters():
                param.requires_grad = True
            logger.info("Unfroze GCN encoder parameters")

    def _reset_decoder_heads(self, model: nn.Module) -> None:
        """Reset the final decoder heads to random weights."""
        if hasattr(model, "tft"):
            tft = model.tft
            for head_name in ["short_head", "mid_head", "long_head"]:
                head = getattr(tft, head_name, None)
                if head is not None and isinstance(head, nn.Linear):
                    nn.init.xavier_uniform_(head.weight)
                    if head.bias is not None:
                        nn.init.zeros_(head.bias)
            logger.info("Reset decoder head weights")

    def warm_start(
        self,
        model: nn.Module,
        global_checkpoint_path: str | Path,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        graph_data=None,
        loss_fn: nn.Module | None = None,
    ) -> nn.Module:
        """
        Execute transfer learning from a global checkpoint.

        Args:
            model: Uninitialised TFTGCNPredictor model (same architecture).
            global_checkpoint_path: Path to pretrained global model checkpoint.
            train_loader: DataLoader for the new station's data.
            val_loader: Optional validation DataLoader.
            graph_data: Updated station graph including the new station.
            loss_fn: Loss function. Defaults to MultiHorizonLoss.

        Returns:
            Fine-tuned model.
        """
        # Load global weights
        checkpoint = torch.load(
            global_checkpoint_path, map_location=self.device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model = model.to(self.device)

        logger.info(
            f"Loaded global checkpoint from {Path(global_checkpoint_path).name} "
            f"(epoch {checkpoint.get('epoch', '?')})"
        )

        # Freeze GCN
        if self.freeze_gcn:
            self._freeze_gcn_params(model)

        # Reset decoder heads for station-specific learning
        self._reset_decoder_heads(model)

        # Setup optimizer (only for unfrozen parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(trainable_params, lr=self.learning_rate)

        if loss_fn is None:
            from hp_ai_engine.training.loss import MultiHorizonLoss
            loss_fn = MultiHorizonLoss()

        # Fine-tuning loop
        best_val_loss = float("inf")
        for epoch in range(1, self.fine_tune_epochs + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                optimizer.zero_grad()

                # Build forward kwargs
                fwd_kwargs = {
                    "static_features": batch.get("static"),
                    "dynamic_past": batch.get("dynamic_past"),
                    "dynamic_future": batch.get("dynamic_future"),
                    "station_indices": batch.get("station_indices"),
                }
                if graph_data is not None:
                    fwd_kwargs["node_features"] = graph_data.x.to(self.device)
                    fwd_kwargs["edge_index"] = graph_data.edge_index.to(self.device)
                    if graph_data.edge_attr is not None:
                        fwd_kwargs["edge_weight"] = graph_data.edge_attr.squeeze(-1).to(self.device)

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
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            # Unfreeze GCN after threshold
            if (
                self.unfreeze_after is not None
                and epoch == self.unfreeze_after
                and self.freeze_gcn
            ):
                self._unfreeze_gcn_params(model)
                # Refresh optimizer to include GCN params
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = Adam(trainable_params, lr=self.learning_rate * 0.1)

            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Transfer fine-tune epoch {epoch}: loss={avg_loss:.4f}")

        logger.info("Transfer learning complete")
        return model
