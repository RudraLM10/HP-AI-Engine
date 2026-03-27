"""
Pydantic data models for the six input data sources.

Each schema validates incoming records and serves as the canonical
type definition for data flowing through the entire pipeline.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# 1. CNG Dispensing Logs
# ---------------------------------------------------------------------------

class DispensingRecord(BaseModel):
    """Single CNG dispensing transaction at a station."""

    station_id: str = Field(..., description="Unique identifier for the station")
    timestamp: datetime = Field(..., description="Transaction timestamp (IST)")
    volume_kg: float = Field(..., ge=0, description="CNG volume dispensed in kg")
    dispenser_id: str = Field(..., description="Identifier of the dispenser used")
    transaction_type: Literal["retail", "fleet"] = Field(
        ..., description="Whether the transaction was retail or fleet"
    )

    @field_validator("volume_kg")
    @classmethod
    def volume_must_be_reasonable(cls, v: float) -> float:
        """Flag unreasonable single-transaction volumes (>500 kg ~ a bus fill)."""
        if v > 500:
            raise ValueError(f"Single transaction volume {v} kg exceeds 500 kg limit")
        return v


# ---------------------------------------------------------------------------
# 2. Weather & AQI
# ---------------------------------------------------------------------------

class WeatherRecord(BaseModel):
    """Hourly weather and air quality observation for a station's locality."""

    station_id: str
    timestamp: datetime
    temperature_c: float = Field(..., ge=-10, le=55, description="Temperature in Celsius")
    humidity_pct: float = Field(..., ge=0, le=100, description="Relative humidity %")
    rainfall_mm: float = Field(..., ge=0, description="Rainfall in mm (hourly)")
    aqi: int = Field(..., ge=0, le=500, description="Air Quality Index (0-500)")
    wind_speed_kmh: float = Field(..., ge=0, description="Wind speed in km/h")


# ---------------------------------------------------------------------------
# 3. Road Traffic Conditions
# ---------------------------------------------------------------------------

class TrafficRecord(BaseModel):
    """Hourly traffic conditions near a station."""

    station_id: str
    timestamp: datetime
    avg_speed_kmh: float = Field(..., ge=0, description="Average road speed in km/h")
    density_vehicles_per_km: float = Field(
        ..., ge=0, description="Vehicle density per km of road"
    )
    source: Literal["google_maps", "tomtom", "sensor"] = Field(
        default="google_maps", description="Traffic data source"
    )


# ---------------------------------------------------------------------------
# 4. CNG Vehicle Population
# ---------------------------------------------------------------------------

class VehiclePopulation(BaseModel):
    """Monthly CNG vehicle registration data for a catchment area."""

    catchment_id: str
    date: date = Field(..., description="First day of the reporting month")
    total_cng_vehicles: int = Field(..., ge=0, description="Total registered CNG vehicles")
    new_registrations_month: int = Field(
        ..., ge=0, description="New CNG registrations this month"
    )
    vehicle_mix: dict[str, float] = Field(
        ...,
        description="Vehicle type distribution, e.g. {'auto': 0.4, 'bus': 0.2, 'car': 0.3, 'taxi': 0.1}",
    )

    @field_validator("vehicle_mix")
    @classmethod
    def mix_must_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        """Vehicle mix fractions should sum to approximately 1.0."""
        total = sum(v.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Vehicle mix fractions sum to {total}, expected ~1.0")
        return v


# ---------------------------------------------------------------------------
# 5. Catchment Area Profile
# ---------------------------------------------------------------------------

class CatchmentProfile(BaseModel):
    """Demographic and economic profile of a station's catchment area."""

    catchment_id: str
    population: int = Field(..., ge=0, description="Total population in catchment")
    area_sq_km: float = Field(..., gt=0, description="Catchment area in sq km")
    avg_income_bracket: Literal["low", "mid", "high"] = Field(
        ..., description="Average income bracket of the catchment population"
    )
    commercial_density: float = Field(
        ..., ge=0, description="Commercial establishments per sq km"
    )
    residential_density: float = Field(
        ..., ge=0, description="Residential units per sq km"
    )


# ---------------------------------------------------------------------------
# 6. Station Metadata & Network Map
# ---------------------------------------------------------------------------

class StationMeta(BaseModel):
    """Physical and operational metadata for a CNG station."""

    station_id: str = Field(..., description="Unique station identifier")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    station_type: Literal["retail", "fleet", "highway", "mixed"] = Field(
        ..., description="Primary operating type"
    )
    dispenser_count: int = Field(..., ge=1, description="Number of CNG dispensers")
    storage_capacity_kg: float = Field(
        ..., gt=0, description="Total CNG storage capacity in kg"
    )
    catchment_id: str = Field(..., description="ID of the catchment area this station belongs to")
    city_cluster: str = Field(
        ..., description="City cluster this station is assigned to (e.g. 'Mumbai_West')"
    )


# ---------------------------------------------------------------------------
# Aggregated station data (wrapper for merged multi-source data)
# ---------------------------------------------------------------------------

class StationHourlyRecord(BaseModel):
    """
    Merged hourly record for a single station, combining all six data sources.
    This is the canonical input format for the feature engineering pipeline.
    """

    station_id: str
    timestamp: datetime

    # Dispensing
    volume_kg: float = Field(default=0.0, ge=0)

    # Weather
    temperature_c: float = 0.0
    humidity_pct: float = 0.0
    rainfall_mm: float = 0.0
    aqi: int = 0
    wind_speed_kmh: float = 0.0

    # Traffic
    traffic_speed_kmh: float = 0.0
    traffic_density: float = 0.0

    # Vehicle population (interpolated to hourly)
    vehicle_population: int = 0

    # Static (from StationMeta)
    station_type: str = "retail"
    dispenser_count: int = 1
    storage_capacity_kg: float = 0.0
    catchment_profile: str = "mid"
