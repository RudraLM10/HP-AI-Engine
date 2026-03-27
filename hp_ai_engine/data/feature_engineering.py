"""
Feature engineering pipeline for HP AI Engine.

Transforms raw merged station-hourly data into model-ready features:
- Lag features (1h, 6h, 12h, 24h, 48h, 168h)
- Rolling statistics (mean, std over configurable windows)
- Cyclical time encodings (hour, day-of-week, month)
- Z-score / min-max normalisation per station
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from hp_ai_engine.utils.logging import get_logger
from hp_ai_engine.utils.time_utils import get_time_features_batch

logger = get_logger("feature_engineering", component="data")


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "volume_kg",
    lags: list[int] | None = None,
    group_col: str = "station_id",
) -> pd.DataFrame:
    """
    Add lagged values of the target column as new features.

    Args:
        df: DataFrame sorted by (group_col, timestamp).
        target_col: Column to lag.
        lags: List of lag periods in hours. Defaults to [1, 6, 12, 24, 48, 168].
        group_col: Column to group by (each station gets independent lags).

    Returns:
        DataFrame with additional columns: {target_col}_lag_{lag}h.
    """
    if lags is None:
        lags = [1, 6, 12, 24, 48, 168]

    df = df.copy()
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}h"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)

    logger.info(f"Added {len(lags)} lag features for '{target_col}'")
    return df


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "volume_kg",
    windows: list[int] | None = None,
    group_col: str = "station_id",
    functions: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add rolling window statistics as new features.

    Args:
        df: DataFrame sorted by (group_col, timestamp).
        target_col: Column to compute rolling stats on.
        windows: List of window sizes in hours. Defaults to [6, 12, 24, 168].
        group_col: Column to group by.
        functions: Stats to compute. Defaults to ["mean", "std"].

    Returns:
        DataFrame with additional columns: {target_col}_rolling_{func}_{window}h.
    """
    if windows is None:
        windows = [6, 12, 24, 168]
    if functions is None:
        functions = ["mean", "std"]

    df = df.copy()
    for window in windows:
        grouped = df.groupby(group_col)[target_col]
        rolling = grouped.transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_mean_{window}h"] = rolling

        if "std" in functions:
            rolling_std = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f"{target_col}_rolling_std_{window}h"] = rolling_std.fillna(0)

    logger.info(f"Added rolling features for '{target_col}' with windows {windows}")
    return df


# ---------------------------------------------------------------------------
# Cyclical time features
# ---------------------------------------------------------------------------

def add_cyclical_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add cyclical time encodings (sin/cos for hour, day-of-week, month)
    and binary flags (is_weekend, is_holiday).

    Args:
        df: DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with 8 additional time feature columns.
    """
    df = df.copy()
    timestamps = pd.DatetimeIndex(df[timestamp_col])
    time_features = get_time_features_batch(timestamps)

    for col in time_features.columns:
        df[col] = time_features[col].values

    logger.info("Added cyclical time features")
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class FeatureNormaliser:
    """
    Normalises features using z-score or min-max scaling.

    Stores fit statistics for inverse transformation during inference.

    Usage:
        normaliser = FeatureNormaliser(method="zscore", per_station=True)
        df_normed = normaliser.fit_transform(df, cols=["temperature_c", "volume_kg"])
        df_original = normaliser.inverse_transform(df_normed, cols=["volume_kg"])
    """

    def __init__(
        self,
        method: Literal["zscore", "minmax"] = "zscore",
        per_station: bool = True,
        group_col: str = "station_id",
    ):
        self.method = method
        self.per_station = per_station
        self.group_col = group_col
        self.stats: dict[str, dict] = {}  # col -> {mean, std} or {min, max}

    def fit(self, df: pd.DataFrame, cols: list[str]) -> FeatureNormaliser:
        """Compute normalisation statistics from training data."""
        for col in cols:
            if self.per_station:
                if self.method == "zscore":
                    means = df.groupby(self.group_col)[col].mean().to_dict()
                    stds = df.groupby(self.group_col)[col].std().to_dict()
                    # Replace zero stds to avoid division by zero
                    stds = {k: v if v > 0 else 1.0 for k, v in stds.items()}
                    self.stats[col] = {"means": means, "stds": stds}
                else:
                    mins = df.groupby(self.group_col)[col].min().to_dict()
                    maxs = df.groupby(self.group_col)[col].max().to_dict()
                    self.stats[col] = {"mins": mins, "maxs": maxs}
            else:
                if self.method == "zscore":
                    self.stats[col] = {
                        "mean": df[col].mean(),
                        "std": max(df[col].std(), 1e-8),
                    }
                else:
                    self.stats[col] = {
                        "min": df[col].min(),
                        "max": df[col].max(),
                    }

        logger.info(f"Fit normalisation ({self.method}) on {len(cols)} columns")
        return self

    def transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Apply normalisation using stored statistics."""
        df = df.copy()
        for col in cols:
            if col not in self.stats:
                raise ValueError(f"Column '{col}' not fitted. Call fit() first.")

            if self.per_station:
                if self.method == "zscore":
                    means = df[self.group_col].map(self.stats[col]["means"])
                    stds = df[self.group_col].map(self.stats[col]["stds"])
                    df[col] = (df[col] - means) / stds
                else:
                    mins = df[self.group_col].map(self.stats[col]["mins"])
                    maxs = df[self.group_col].map(self.stats[col]["maxs"])
                    ranges = maxs - mins
                    ranges = ranges.replace(0, 1)
                    df[col] = (df[col] - mins) / ranges
            else:
                if self.method == "zscore":
                    df[col] = (df[col] - self.stats[col]["mean"]) / self.stats[col]["std"]
                else:
                    range_val = self.stats[col]["max"] - self.stats[col]["min"]
                    range_val = max(range_val, 1e-8)
                    df[col] = (df[col] - self.stats[col]["min"]) / range_val

        return df

    def fit_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, cols).transform(df, cols)

    def inverse_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Reverse normalisation to original scale."""
        df = df.copy()
        for col in cols:
            if col not in self.stats:
                raise ValueError(f"Column '{col}' not fitted.")

            if self.per_station:
                if self.method == "zscore":
                    means = df[self.group_col].map(self.stats[col]["means"])
                    stds = df[self.group_col].map(self.stats[col]["stds"])
                    df[col] = df[col] * stds + means
                else:
                    mins = df[self.group_col].map(self.stats[col]["mins"])
                    maxs = df[self.group_col].map(self.stats[col]["maxs"])
                    df[col] = df[col] * (maxs - mins) + mins
            else:
                if self.method == "zscore":
                    df[col] = df[col] * self.stats[col]["std"] + self.stats[col]["mean"]
                else:
                    df[col] = df[col] * (self.stats[col]["max"] - self.stats[col]["min"]) + self.stats[col]["min"]

        return df


# ---------------------------------------------------------------------------
# Full feature engineering pipeline
# ---------------------------------------------------------------------------

def run_feature_pipeline(
    df: pd.DataFrame,
    target_col: str = "volume_kg",
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    normalise_cols: list[str] | None = None,
    normalise_method: Literal["zscore", "minmax"] = "zscore",
) -> tuple[pd.DataFrame, FeatureNormaliser | None]:
    """
    Execute the full feature engineering pipeline.

    1. Sort by station and timestamp
    2. Add lag features
    3. Add rolling features
    4. Add cyclical time features
    5. Normalise specified columns
    6. Drop rows with NaN from lag/rolling operations

    Args:
        df: Merged station-hourly DataFrame from ingestion pipeline.
        target_col: Target column name.
        lags: Lag periods in hours.
        rolling_windows: Rolling window sizes in hours.
        normalise_cols: Columns to normalise. If None, normalises all numeric
                       columns except target and time features.
        normalise_method: 'zscore' or 'minmax'.

    Returns:
        Tuple of (processed DataFrame, fitted FeatureNormaliser or None).
    """
    logger.info("Starting feature engineering pipeline")

    # Sort
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # Add features
    df = add_lag_features(df, target_col, lags)
    df = add_rolling_features(df, target_col, rolling_windows)
    df = add_cyclical_time_features(df)

    # Drop rows where lag features are NaN (beginning of each station's series)
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)
    logger.info(f"Dropped {dropped} rows with NaN from lag/rolling operations")

    # Normalise
    normaliser = None
    if normalise_cols is None:
        # Default: normalise all numeric columns except target, time features, and IDs
        exclude = {target_col, "station_id", "timestamp", "catchment_id", "city_cluster",
                   "station_type", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                   "month_sin", "month_cos", "is_weekend", "is_holiday"}
        normalise_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    if normalise_cols:
        normaliser = FeatureNormaliser(method=normalise_method, per_station=True)
        df = normaliser.fit_transform(df, normalise_cols)

    logger.info(
        f"Feature engineering complete: {len(df)} rows, {len(df.columns)} columns"
    )
    return df, normaliser
