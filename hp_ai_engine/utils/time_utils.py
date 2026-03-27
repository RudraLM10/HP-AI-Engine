"""
Time and date utility functions for HP AI Engine.

Handles timezone conversions, hourly binning, cyclical time feature
generation, and Indian public holiday integration.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd


# Indian Standard Time offset
IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Indian public holidays (major national holidays — extendable per year)
# ---------------------------------------------------------------------------

def _get_indian_holidays(year: int) -> set[date]:
    """
    Return a set of major Indian public holidays for a given year.

    This is a static list of nationally observed holidays. For production,
    replace with a dynamic API or maintained database.
    """
    holidays = {
        date(year, 1, 26),   # Republic Day
        date(year, 3, 29),   # Holi (approximate — varies yearly)
        date(year, 4, 14),   # Ambedkar Jayanti
        date(year, 5, 1),    # May Day
        date(year, 8, 15),   # Independence Day
        date(year, 10, 2),   # Gandhi Jayanti
        date(year, 10, 24),  # Dussehra (approximate)
        date(year, 11, 12),  # Diwali (approximate)
        date(year, 12, 25),  # Christmas
    }
    return holidays


def is_indian_holiday(dt: date | datetime) -> bool:
    """Check if a date is an Indian public holiday."""
    if isinstance(dt, datetime):
        dt = dt.date()
    holidays = _get_indian_holidays(dt.year)
    return dt in holidays


# ---------------------------------------------------------------------------
# Hourly binning
# ---------------------------------------------------------------------------

def to_hourly_bins(timestamps: pd.Series | Sequence[datetime]) -> pd.DatetimeIndex:
    """
    Floor timestamps to hourly bins.

    Args:
        timestamps: Sequence of datetime objects or pandas Series.

    Returns:
        DatetimeIndex with timestamps floored to the nearest hour.
    """
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)
    return pd.DatetimeIndex(timestamps.dt.floor("h"))


# ---------------------------------------------------------------------------
# Cyclical time features
# ---------------------------------------------------------------------------

def get_time_features(dt: datetime | pd.Timestamp) -> dict[str, float]:
    """
    Extract time-based features from a datetime, including cyclical encodings.

    Returns:
        Dict with keys:
            hour_sin, hour_cos — cyclical hour of day
            dow_sin, dow_cos   — cyclical day of week
            month_sin, month_cos — cyclical month of year
            is_weekend         — 1.0 if Saturday or Sunday
            is_holiday         — 1.0 if Indian public holiday
    """
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()  # 0=Monday, 6=Sunday
    month = dt.month

    return {
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
        "month_sin": float(np.sin(2 * np.pi * (month - 1) / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * (month - 1) / 12.0)),
        "is_weekend": 1.0 if dow >= 5 else 0.0,
        "is_holiday": 1.0 if is_indian_holiday(dt) else 0.0,
    }


def get_time_features_batch(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute cyclical time features for a batch of timestamps.

    Args:
        timestamps: DatetimeIndex to process.

    Returns:
        DataFrame with columns: hour_sin, hour_cos, dow_sin, dow_cos,
        month_sin, month_cos, is_weekend, is_holiday.
    """
    hours = timestamps.hour + timestamps.minute / 60.0
    dows = timestamps.weekday
    months = timestamps.month

    df = pd.DataFrame(index=timestamps)
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12.0)
    df["is_weekend"] = (dows >= 5).astype(float)
    df["is_holiday"] = pd.Series(
        [1.0 if is_indian_holiday(ts) else 0.0 for ts in timestamps],
        index=timestamps,
    )

    return df
