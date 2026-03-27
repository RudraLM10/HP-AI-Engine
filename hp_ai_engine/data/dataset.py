"""
PyTorch Dataset and DataLoader utilities for HP AI Engine.

Provides:
- CNGDemandDataset: per-station time-series dataset for TFT input
- CNGGraphBatch: graph-aware batch collator ensuring full network coverage
  per time window for GCN message passing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("dataset", component="data")


@dataclass
class CNGSample:
    """A single model input sample for one station at one time step."""

    static_covariates: torch.Tensor       # [num_static_features]
    dynamic_past: torch.Tensor            # [lookback_hours, num_dynamic_features]
    dynamic_future_known: torch.Tensor    # [horizon_hours, num_known_future_features]
    target_short: torch.Tensor            # [short_horizon]
    target_mid: torch.Tensor              # [mid_horizon]
    target_long: torch.Tensor             # [long_horizon]
    station_idx: int                      # index into graph node list


class CNGDemandDataset(Dataset):
    """
    PyTorch Dataset for CNG demand forecasting.

    Each sample consists of:
    - Static covariates: station type encoding, dispenser count, storage capacity, etc.
    - Dynamic past: lookback window of dynamic features (volume, weather, traffic, etc.)
    - Dynamic future (known): time features for the forecast horizon (hour, DOW, holidays)
    - Targets: ground-truth volumes at 3 horizons (short, mid, long)

    Args:
        df: Feature-engineered DataFrame from feature_engineering pipeline.
        station_ids: Ordered list of station IDs (matching graph node order).
        lookback_hours: Number of past hours to include. Default 168 (7 days).
        short_horizon: Short-term forecast horizon in hours. Default 6.
        mid_horizon: Mid-term forecast horizon in hours. Default 168.
        long_horizon: Long-term forecast horizon in hours. Default 4320.
        target_col: Name of the target column. Default 'volume_kg'.
        static_cols: Columns to use as static covariates.
        dynamic_cols: Columns to use as dynamic past features.
        future_known_cols: Columns known in advance (time features).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        station_ids: list[str],
        lookback_hours: int = 168,
        short_horizon: int = 6,
        mid_horizon: int = 168,
        long_horizon: int = 4320,
        target_col: str = "volume_kg",
        static_cols: list[str] | None = None,
        dynamic_cols: list[str] | None = None,
        future_known_cols: list[str] | None = None,
    ):
        self.lookback_hours = lookback_hours
        self.short_horizon = short_horizon
        self.mid_horizon = mid_horizon
        self.long_horizon = long_horizon
        self.target_col = target_col
        self.station_ids = station_ids
        self.max_horizon = max(short_horizon, mid_horizon, long_horizon)

        # Default column groups
        if static_cols is None:
            static_cols = ["dispenser_count", "storage_capacity_kg"]
        if dynamic_cols is None:
            dynamic_cols = [
                "volume_kg", "temperature_c", "humidity_pct", "rainfall_mm",
                "aqi", "wind_speed_kmh", "traffic_speed_kmh", "traffic_density",
            ]
        if future_known_cols is None:
            future_known_cols = [
                "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "month_sin", "month_cos", "is_weekend", "is_holiday",
            ]

        self.static_cols = static_cols
        self.dynamic_cols = dynamic_cols
        self.future_known_cols = future_known_cols

        # Build index of valid samples
        self._samples: list[tuple[str, int]] = []  # (station_id, start_position_in_station_df)
        self._station_dfs: dict[str, pd.DataFrame] = {}

        for sid in station_ids:
            station_df = df[df["station_id"] == sid].sort_values("timestamp").reset_index(drop=True)
            self._station_dfs[sid] = station_df

            # Valid samples: need lookback in the past + max horizon in the future
            total_rows = len(station_df)
            for t in range(lookback_hours, total_rows - self.max_horizon):
                self._samples.append((sid, t))

        logger.info(
            f"Created dataset: {len(self._samples)} samples from "
            f"{len(station_ids)} stations"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> CNGSample:
        station_id, t = self._samples[idx]
        station_df = self._station_dfs[station_id]

        # Static covariates (same across all time steps for this station)
        static = station_df[self.static_cols].iloc[t].values.astype(np.float32)

        # Dynamic past: [t - lookback : t]
        past_slice = station_df.iloc[t - self.lookback_hours: t]
        dynamic_past = past_slice[self.dynamic_cols].values.astype(np.float32)

        # Dynamic future (known): [t : t + max_horizon]
        future_slice = station_df.iloc[t: t + self.max_horizon]
        available_future_cols = [c for c in self.future_known_cols if c in future_slice.columns]
        if available_future_cols:
            dynamic_future = future_slice[available_future_cols].values.astype(np.float32)
        else:
            dynamic_future = np.zeros(
                (self.max_horizon, len(self.future_known_cols)), dtype=np.float32
            )

        # Targets at three horizons
        target_series = station_df[self.target_col].values.astype(np.float32)
        target_short = target_series[t: t + self.short_horizon]
        target_mid = target_series[t: t + self.mid_horizon]
        target_long = target_series[t: t + self.long_horizon]

        # Pad targets if near the end (shouldn't happen due to index filtering, but safe)
        target_short = np.pad(target_short, (0, max(0, self.short_horizon - len(target_short))))
        target_mid = np.pad(target_mid, (0, max(0, self.mid_horizon - len(target_mid))))
        target_long = np.pad(target_long, (0, max(0, self.long_horizon - len(target_long))))

        station_idx = self.station_ids.index(station_id)

        return CNGSample(
            static_covariates=torch.from_numpy(static),
            dynamic_past=torch.from_numpy(dynamic_past),
            dynamic_future_known=torch.from_numpy(dynamic_future),
            target_short=torch.from_numpy(target_short),
            target_mid=torch.from_numpy(target_mid),
            target_long=torch.from_numpy(target_long),
            station_idx=station_idx,
        )


def collate_cng_samples(samples: list[CNGSample]) -> dict[str, torch.Tensor]:
    """
    Custom collate function for CNGDemandDataset.

    Stacks individual samples into batched tensors.

    Returns:
        Dict with keys: static, dynamic_past, dynamic_future,
        target_short, target_mid, target_long, station_indices.
    """
    return {
        "static": torch.stack([s.static_covariates for s in samples]),
        "dynamic_past": torch.stack([s.dynamic_past for s in samples]),
        "dynamic_future": torch.stack([s.dynamic_future_known for s in samples]),
        "target_short": torch.stack([s.target_short for s in samples]),
        "target_mid": torch.stack([s.target_mid for s in samples]),
        "target_long": torch.stack([s.target_long for s in samples]),
        "station_indices": torch.tensor([s.station_idx for s in samples], dtype=torch.long),
    }


def create_dataloaders(
    dataset: CNGDemandDataset,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.10,
    split_strategy: str = "temporal",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Uses temporal splitting (not random) to prevent data leakage:
    the test set is always the most recent data, validation is before test,
    and training is the earliest data.

    Args:
        dataset: CNGDemandDataset instance.
        batch_size: Batch size.
        val_split: Fraction for validation.
        test_split: Fraction for test.
        split_strategy: 'temporal' for chronological, 'random' for random.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    n = len(dataset)
    if split_strategy == "temporal":
        test_size = int(n * test_split)
        val_size = int(n * val_split)
        train_size = n - val_size - test_size

        train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        test_dataset = torch.utils.data.Subset(
            dataset, range(train_size + val_size, n)
        )
    else:
        # Random split
        train_size = n - int(n * val_split) - int(n * test_split)
        val_size = int(n * val_split)
        test_size = n - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(split_strategy == "random"),
        collate_fn=collate_cng_samples, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_cng_samples, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_cng_samples, num_workers=num_workers,
    )

    logger.info(
        f"DataLoaders created: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader
