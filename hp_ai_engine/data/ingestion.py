"""
Data ingestion pipeline for HP AI Engine.

Loads data from CSV, Parquet, or TimescaleDB, validates against Pydantic schemas,
aligns timestamps to hourly granularity, and produces a merged multi-source DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type, TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("ingestion", component="data")

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# File loaders with schema validation
# ---------------------------------------------------------------------------

def load_csv(
    path: str | Path,
    schema_cls: Type[T],
    parse_dates: list[str] | None = None,
) -> list[T]:
    """
    Load a CSV file and validate each row against a Pydantic schema.

    Args:
        path: Path to the CSV file.
        schema_cls: Pydantic model class to validate against.
        parse_dates: Column names to parse as datetime.

    Returns:
        List of validated Pydantic model instances.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path, parse_dates=parse_dates or [])
    records: list[T] = []
    errors: list[tuple[int, str]] = []

    for idx, row in df.iterrows():
        try:
            record = schema_cls(**row.to_dict())
            records.append(record)
        except ValidationError as e:
            errors.append((int(idx), str(e)))

    if errors:
        logger.warning(
            f"Validation errors in {path.name}: {len(errors)} of {len(df)} rows failed",
            extra={"error_count": len(errors), "file": str(path)},
        )

    logger.info(
        f"Loaded {len(records)} valid records from {path.name}",
        extra={"file": str(path), "schema": schema_cls.__name__},
    )
    return records


def load_parquet(
    path: str | Path,
    schema_cls: Type[T],
) -> pd.DataFrame:
    """
    Load a Parquet file and validate columns against a Pydantic schema.

    Returns a DataFrame (not individual model instances) for performance
    with large datasets. Validation is done on a sample.

    Args:
        path: Path to the Parquet file.
        schema_cls: Pydantic model class for column validation.

    Returns:
        DataFrame with columns matching the schema.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    # Validate schema on first 100 rows
    sample = df.head(100)
    error_count = 0
    for _, row in sample.iterrows():
        try:
            schema_cls(**row.to_dict())
        except ValidationError:
            error_count += 1

    if error_count > 0:
        logger.warning(
            f"Schema validation: {error_count}/100 sample rows failed for {path.name}",
            extra={"file": str(path), "schema": schema_cls.__name__},
        )

    logger.info(f"Loaded {len(df)} rows from {path.name}")
    return df


def records_to_dataframe(records: list[BaseModel]) -> pd.DataFrame:
    """Convert a list of Pydantic model instances to a pandas DataFrame."""
    return pd.DataFrame([r.model_dump() for r in records])


# ---------------------------------------------------------------------------
# Multi-source ingestion pipeline
# ---------------------------------------------------------------------------

class DataIngestionPipeline:
    """
    Orchestrates loading and merging data from all six sources.

    The pipeline:
    1. Loads each data source from configured file paths
    2. Validates each row against its Pydantic schema
    3. Aligns all timestamps to hourly granularity
    4. Merges sources into a unified station-hourly DataFrame

    Usage:
        pipeline = DataIngestionPipeline(
            dispensing_path="data/raw/dispensing.csv",
            weather_path="data/raw/weather.csv",
            traffic_path="data/raw/traffic.csv",
            vehicle_path="data/raw/vehicles.csv",
            catchment_path="data/raw/catchments.csv",
            station_path="data/raw/stations.csv",
        )
        merged_df = pipeline.run()
    """

    def __init__(
        self,
        dispensing_path: str | Path | None = None,
        weather_path: str | Path | None = None,
        traffic_path: str | Path | None = None,
        vehicle_path: str | Path | None = None,
        catchment_path: str | Path | None = None,
        station_path: str | Path | None = None,
    ):
        self.dispensing_path = dispensing_path
        self.weather_path = weather_path
        self.traffic_path = traffic_path
        self.vehicle_path = vehicle_path
        self.catchment_path = catchment_path
        self.station_path = station_path

    def _load_and_floor(
        self,
        path: str | Path | None,
        schema_cls: Type[BaseModel],
        time_col: str = "timestamp",
    ) -> pd.DataFrame | None:
        """Load a source, validate, and floor timestamps to hourly."""
        if path is None:
            return None

        path = Path(path)
        if not path.exists():
            logger.warning(f"Source file not found, skipping: {path}")
            return None

        if path.suffix == ".parquet":
            df = load_parquet(path, schema_cls)
        else:
            records = load_csv(path, schema_cls, parse_dates=[time_col])
            df = records_to_dataframe(records)

        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col]).dt.floor("h")

        return df

    def run(self) -> pd.DataFrame:
        """
        Execute the full ingestion and merge pipeline.

        Returns:
            DataFrame indexed by (station_id, timestamp) with all features.
        """
        from hp_ai_engine.data.schemas import (
            CatchmentProfile,
            DispensingRecord,
            StationMeta,
            TrafficRecord,
            VehiclePopulation,
            WeatherRecord,
        )

        logger.info("Starting data ingestion pipeline")

        # Load each source
        dispensing_df = self._load_and_floor(self.dispensing_path, DispensingRecord)
        weather_df = self._load_and_floor(self.weather_path, WeatherRecord)
        traffic_df = self._load_and_floor(self.traffic_path, TrafficRecord)
        vehicle_df = self._load_and_floor(self.vehicle_path, VehiclePopulation, time_col="date")
        catchment_df = self._load_and_floor(self.catchment_path, CatchmentProfile)
        station_df = self._load_and_floor(self.station_path, StationMeta)

        # Start with dispensing as the base (it defines the station-time grid)
        if dispensing_df is None:
            raise ValueError("Dispensing data is required — it is the target variable.")

        # Aggregate dispensing to hourly per station
        base = (
            dispensing_df
            .groupby(["station_id", "timestamp"])
            .agg(volume_kg=("volume_kg", "sum"))
            .reset_index()
        )

        # Merge weather
        if weather_df is not None:
            weather_cols = ["station_id", "timestamp", "temperature_c", "humidity_pct",
                           "rainfall_mm", "aqi", "wind_speed_kmh"]
            weather_df = weather_df[[c for c in weather_cols if c in weather_df.columns]]
            base = base.merge(weather_df, on=["station_id", "timestamp"], how="left")

        # Merge traffic
        if traffic_df is not None:
            traffic_cols = ["station_id", "timestamp", "avg_speed_kmh", "density_vehicles_per_km"]
            traffic_df = traffic_df[[c for c in traffic_cols if c in traffic_df.columns]]
            base = base.merge(traffic_df, on=["station_id", "timestamp"], how="left")
            base.rename(columns={
                "avg_speed_kmh": "traffic_speed_kmh",
                "density_vehicles_per_km": "traffic_density",
            }, inplace=True)

        # Merge station metadata (static columns)
        if station_df is not None:
            static_cols = ["station_id", "station_type", "dispenser_count",
                          "storage_capacity_kg", "catchment_id", "city_cluster"]
            station_df = station_df[[c for c in static_cols if c in station_df.columns]]
            base = base.merge(station_df, on="station_id", how="left")

        # Forward-fill NaN from merges
        base = base.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        base = base.fillna(method="ffill").fillna(0)

        logger.info(
            f"Ingestion complete: {len(base)} rows, {base['station_id'].nunique()} stations",
            extra={"rows": len(base), "stations": base["station_id"].nunique()},
        )

        return base
