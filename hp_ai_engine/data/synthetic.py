"""
Synthetic data generator for HP AI Engine.

Generates realistic synthetic datasets for all six data sources to enable
model development and testing before real HPCL dispensing data is available.

Patterns are based on documented CNG demand characteristics from PNGRB reports:
- Bimodal daily demand (morning and evening peaks)
- Weekly seasonality (lower Sundays)
- Station-type scaling (highway > retail)
- Monsoon effects on demand
- Gradual growth trend
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from hp_ai_engine.utils.logging import get_logger

logger = get_logger("synthetic", component="data")


# ---------------------------------------------------------------------------
# Station metadata generation
# ---------------------------------------------------------------------------

def generate_stations(
    num_stations: int = 50,
    bbox: dict | None = None,
    clusters: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic station metadata.

    Args:
        num_stations: Number of stations to generate.
        bbox: Bounding box dict with keys lat_min, lat_max, lon_min, lon_max.
        clusters: List of city cluster names to assign stations to.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns matching StationMeta schema.
    """
    rng = np.random.default_rng(seed)

    if bbox is None:
        bbox = {"lat_min": 18.85, "lat_max": 19.30, "lon_min": 72.75, "lon_max": 73.05}
    if clusters is None:
        clusters = ["Mumbai_West", "Mumbai_East", "Navi_Mumbai", "Thane"]

    station_types: list[Literal["retail", "fleet", "highway", "mixed"]] = [
        "retail", "fleet", "highway", "mixed"
    ]
    type_weights = [0.5, 0.2, 0.15, 0.15]

    records = []
    for i in range(num_stations):
        lat = rng.uniform(bbox["lat_min"], bbox["lat_max"])
        lon = rng.uniform(bbox["lon_min"], bbox["lon_max"])
        st_type = rng.choice(station_types, p=type_weights)

        # Dispenser count depends on station type
        dispenser_map = {"retail": (2, 4), "fleet": (3, 6), "highway": (4, 8), "mixed": (3, 5)}
        d_min, d_max = dispenser_map[st_type]

        records.append({
            "station_id": f"ST_{i:04d}",
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "station_type": st_type,
            "dispenser_count": int(rng.integers(d_min, d_max + 1)),
            "storage_capacity_kg": round(rng.uniform(2000, 8000), 0),
            "catchment_id": f"CA_{i % 20:03d}",
            "city_cluster": clusters[i % len(clusters)],
        })

    df = pd.DataFrame(records)
    logger.info(f"Generated {num_stations} synthetic stations")
    return df


# ---------------------------------------------------------------------------
# Dispensing volume generation
# ---------------------------------------------------------------------------

def _daily_demand_curve(hours: np.ndarray) -> np.ndarray:
    """
    Generate a bimodal daily demand curve.

    Peaks at 8-10 AM (commute) and 5-7 PM (return commute).
    """
    morning_peak = np.exp(-0.5 * ((hours - 9) / 1.5) ** 2)
    evening_peak = np.exp(-0.5 * ((hours - 18) / 1.5) ** 2)
    base = 0.15  # minimum baseline demand even at night
    return base + 0.5 * morning_peak + 0.35 * evening_peak


def generate_dispensing(
    stations_df: pd.DataFrame,
    num_days: int = 365,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hourly dispensing volume data.

    Features:
    - Bimodal daily pattern (morning + evening peaks)
    - Weekly seasonality (Sundays ~70% of weekday volume)
    - Station-type scaling (highway 2×, fleet 1.8×, mixed 1.3× vs retail)
    - Monthly growth trend (+0.5% per month)
    - Gaussian noise (±15%)
    - Monsoon dip (June-Sept: -10% due to reduced road travel)

    Args:
        stations_df: Station metadata DataFrame.
        num_days: Number of days to generate.
        start_date: Start date string (YYYY-MM-DD).
        seed: Random seed.

    Returns:
        DataFrame with columns: station_id, timestamp, volume_kg.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    hours = np.arange(24)
    daily_curve = _daily_demand_curve(hours)

    type_scale = {"retail": 1.0, "fleet": 1.8, "highway": 2.0, "mixed": 1.3}

    records = []
    for _, station in stations_df.iterrows():
        base_volume = rng.uniform(30, 80)  # base kg per peak hour
        scale = type_scale.get(station["station_type"], 1.0)

        for day in range(num_days):
            current_date = start + timedelta(days=day)
            dow = current_date.weekday()
            month = current_date.month

            # Weekly factor: Sunday is lower
            weekly_factor = 0.7 if dow == 6 else (0.85 if dow == 5 else 1.0)

            # Monthly growth: +0.5% per month
            growth_factor = 1.0 + 0.005 * (month - 1 + day // 30 * 0.5)

            # Monsoon dip: June-September
            monsoon_factor = 0.9 if month in (6, 7, 8, 9) else 1.0

            for hour_idx in range(24):
                ts = current_date + timedelta(hours=hour_idx)
                volume = (
                    base_volume
                    * scale
                    * daily_curve[hour_idx]
                    * weekly_factor
                    * growth_factor
                    * monsoon_factor
                )
                # Add noise
                noise = rng.normal(1.0, 0.15)
                volume = max(0, volume * noise)

                records.append({
                    "station_id": station["station_id"],
                    "timestamp": ts,
                    "volume_kg": round(volume, 2),
                })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} dispensing records for {len(stations_df)} stations")
    return df


# ---------------------------------------------------------------------------
# Weather data generation
# ---------------------------------------------------------------------------

def generate_weather(
    stations_df: pd.DataFrame,
    num_days: int = 365,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hourly weather data based on Indian climatology.

    Features:
    - Seasonal temperature (cold winters, hot summers for North India)
    - Monsoon rainfall distribution (June-September heavy)
    - AQI correlation with season (worse in winter/post-Diwali)
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)

    records = []
    for _, station in stations_df.iterrows():
        for day in range(num_days):
            current_date = start + timedelta(days=day)
            month = current_date.month

            # Seasonal temperature profile (Delhi-like)
            month_temp_base = {
                1: 14, 2: 17, 3: 23, 4: 30, 5: 35, 6: 35,
                7: 32, 8: 31, 9: 30, 10: 28, 11: 22, 12: 16,
            }
            temp_base = month_temp_base[month]

            # Monsoon rainfall
            month_rain_prob = {
                1: 0.02, 2: 0.02, 3: 0.03, 4: 0.05, 5: 0.08, 6: 0.35,
                7: 0.55, 8: 0.50, 9: 0.35, 10: 0.10, 11: 0.03, 12: 0.02,
            }

            # AQI profile (worse in winter)
            month_aqi_base = {
                1: 280, 2: 220, 3: 180, 4: 150, 5: 140, 6: 100,
                7: 80, 8: 75, 9: 90, 10: 200, 11: 350, 12: 300,
            }

            for hour_idx in range(24):
                ts = current_date + timedelta(hours=hour_idx)

                # Temperature with diurnal variation
                diurnal = -3 * np.cos(2 * np.pi * hour_idx / 24)
                temp = temp_base + diurnal + rng.normal(0, 2)

                # Humidity inversely related to temperature
                humidity = min(100, max(20, 80 - (temp - 25) * 2 + rng.normal(0, 5)))

                # Rainfall (sporadic during monsoon)
                if rng.random() < month_rain_prob[month]:
                    rainfall = rng.exponential(5.0)
                else:
                    rainfall = 0.0

                # AQI
                aqi = max(0, int(month_aqi_base[month] + rng.normal(0, 30)))

                # Wind
                wind = max(0, rng.normal(12, 5))

                records.append({
                    "station_id": station["station_id"],
                    "timestamp": ts,
                    "temperature_c": round(temp, 1),
                    "humidity_pct": round(humidity, 1),
                    "rainfall_mm": round(max(0, rainfall), 1),
                    "aqi": min(500, aqi),
                    "wind_speed_kmh": round(wind, 1),
                })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} weather records")
    return df


# ---------------------------------------------------------------------------
# Traffic data generation
# ---------------------------------------------------------------------------

def generate_traffic(
    stations_df: pd.DataFrame,
    num_days: int = 365,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hourly traffic data.

    Speed inversely correlated with dispensing peak hours (rush hour congestion).
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)

    records = []
    for _, station in stations_df.iterrows():
        base_speed = rng.uniform(25, 45)  # km/h (urban roads)

        for day in range(num_days):
            current_date = start + timedelta(days=day)
            dow = current_date.weekday()

            for hour_idx in range(24):
                ts = current_date + timedelta(hours=hour_idx)

                # Speed dips during rush hours
                rush_dip = 1.0
                if hour_idx in range(8, 11):
                    rush_dip = 0.6
                elif hour_idx in range(17, 20):
                    rush_dip = 0.65
                elif hour_idx in range(0, 6):
                    rush_dip = 1.4  # free-flowing at night

                # Weekend is smoother
                if dow >= 5:
                    rush_dip = min(rush_dip * 1.2, 1.4)

                speed = base_speed * rush_dip + rng.normal(0, 3)
                density = max(0, 200 / max(speed, 1) + rng.normal(0, 5))

                records.append({
                    "station_id": station["station_id"],
                    "timestamp": ts,
                    "avg_speed_kmh": round(max(5, speed), 1),
                    "density_vehicles_per_km": round(max(0, density), 1),
                    "source": "synthetic",
                })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} traffic records")
    return df


# ---------------------------------------------------------------------------
# Vehicle population generation
# ---------------------------------------------------------------------------

def generate_vehicle_population(
    stations_df: pd.DataFrame,
    num_months: int = 12,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic monthly CNG vehicle registration data.

    Growth following PNGRB projections (~5-8% annual growth).
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)

    catchment_ids = stations_df["catchment_id"].unique()
    records = []

    for catchment in catchment_ids:
        base_vehicles = int(rng.uniform(5000, 50000))
        monthly_growth_rate = rng.uniform(0.004, 0.008)  # 5-10% annually

        for month_idx in range(num_months):
            current_date = start + pd.DateOffset(months=month_idx)
            total = int(base_vehicles * (1 + monthly_growth_rate) ** month_idx)
            new_reg = int(total * monthly_growth_rate)

            records.append({
                "catchment_id": catchment,
                "date": current_date.date(),
                "total_cng_vehicles": total,
                "new_registrations_month": new_reg,
                "vehicle_mix": {
                    "auto": round(rng.uniform(0.30, 0.45), 2),
                    "bus": round(rng.uniform(0.05, 0.15), 2),
                    "car": round(rng.uniform(0.20, 0.35), 2),
                    "taxi": round(rng.uniform(0.10, 0.20), 2),
                },
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} vehicle population records")
    return df


# ---------------------------------------------------------------------------
# Catchment profile generation
# ---------------------------------------------------------------------------

def generate_catchments(
    stations_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic catchment area profiles."""
    rng = np.random.default_rng(seed)

    catchment_ids = stations_df["catchment_id"].unique()
    income_brackets = ["low", "mid", "high"]
    income_weights = [0.3, 0.5, 0.2]

    records = []
    for catchment in catchment_ids:
        records.append({
            "catchment_id": catchment,
            "population": int(rng.uniform(50_000, 500_000)),
            "area_sq_km": round(rng.uniform(5, 50), 1),
            "avg_income_bracket": rng.choice(income_brackets, p=income_weights),
            "commercial_density": round(rng.uniform(10, 200), 1),
            "residential_density": round(rng.uniform(500, 5000), 1),
        })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} catchment profiles")
    return df


# ---------------------------------------------------------------------------
# Full synthetic dataset generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates a complete synthetic dataset for all six data sources.

    Usage:
        generator = SyntheticDataGenerator(num_stations=50, num_days=365, seed=42)
        data = generator.generate_all()
        # data is a dict with keys: stations, dispensing, weather, traffic, vehicles, catchments
    """

    def __init__(
        self,
        num_stations: int = 50,
        num_days: int = 365,
        start_date: str = "2024-01-01",
        seed: int = 42,
        bbox: dict | None = None,
        clusters: list[str] | None = None,
    ):
        self.num_stations = num_stations
        self.num_days = num_days
        self.start_date = start_date
        self.seed = seed
        self.bbox = bbox
        self.clusters = clusters

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate all six synthetic data sources."""
        logger.info(
            f"Generating full synthetic dataset: {self.num_stations} stations, "
            f"{self.num_days} days"
        )

        stations = generate_stations(
            self.num_stations, self.bbox, self.clusters, self.seed
        )
        dispensing = generate_dispensing(
            stations, self.num_days, self.start_date, self.seed
        )
        weather = generate_weather(
            stations, self.num_days, self.start_date, self.seed + 1
        )
        traffic = generate_traffic(
            stations, self.num_days, self.start_date, self.seed + 2
        )
        vehicles = generate_vehicle_population(
            stations, num_months=max(1, self.num_days // 30), start_date=self.start_date,
            seed=self.seed + 3
        )
        catchments = generate_catchments(stations, self.seed + 4)

        data = {
            "stations": stations,
            "dispensing": dispensing,
            "weather": weather,
            "traffic": traffic,
            "vehicles": vehicles,
            "catchments": catchments,
        }

        logger.info("Synthetic dataset generation complete")
        return data

    def save_all(self, output_dir: str = "data/synthetic/") -> None:
        """Generate and save all datasets to CSV files."""
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        data = self.generate_all()
        for name, df in data.items():
            filepath = output_path / f"{name}.csv"
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} to {filepath}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from hp_ai_engine.utils.logging import setup_logging
    setup_logging(level="INFO", json_format=False)

    generator = SyntheticDataGenerator(num_stations=50, num_days=365, seed=42)
    generator.save_all()
