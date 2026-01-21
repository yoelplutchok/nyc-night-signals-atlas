"""
Timezone-aware time utilities for nighttime filtering.

Per R7: All timestamps converted to America/New_York.
       Nighttime window logic is centralized, tested, handles cross-midnight windows.
       DST edge cases have unit tests.
"""

from datetime import datetime, time
from typing import Optional, Tuple, Union

import pandas as pd
import pytz

from sleep_esi.io_utils import read_yaml
from sleep_esi.paths import CONFIG_DIR


# NYC timezone
NYC_TZ = pytz.timezone("America/New_York")


def _load_nighttime_config() -> dict:
    """Load nighttime configuration from params.yml."""
    params = read_yaml(CONFIG_DIR / "params.yml")
    return params.get("nighttime", {})


# =============================================================================
# Timestamp Conversion
# =============================================================================

def to_nyc_timezone(
    timestamps: Union[pd.Series, pd.DatetimeIndex],
    source_tz: Optional[str] = None,
) -> pd.Series:
    """
    Convert timestamps to NYC timezone (America/New_York).
    
    Per R7: All timestamps must be converted to America/New_York immediately.
    
    Args:
        timestamps: Series or DatetimeIndex of timestamps
        source_tz: Source timezone if timestamps are naive.
                   If None and timestamps are naive, assumes UTC.
    
    Returns:
        Series of timezone-aware timestamps in NYC time
    """
    # Convert to Series if DatetimeIndex
    if isinstance(timestamps, pd.DatetimeIndex):
        timestamps = timestamps.to_series()
    
    # Ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps)
    
    # Handle timezone conversion
    if timestamps.dt.tz is None:
        # Naive timestamps - localize first
        if source_tz is None:
            source_tz = "UTC"
        timestamps = timestamps.dt.tz_localize(source_tz)
    
    # Convert to NYC timezone
    return timestamps.dt.tz_convert(NYC_TZ)


def ensure_nyc_timezone(timestamps: pd.Series) -> pd.Series:
    """
    Ensure timestamps are in NYC timezone, converting if necessary.
    
    Args:
        timestamps: Series of timestamps (may or may not have timezone)
    
    Returns:
        Series of timezone-aware timestamps in NYC time
    """
    return to_nyc_timezone(timestamps)


# =============================================================================
# Nighttime Filtering (Cross-Midnight Aware)
# =============================================================================

def is_nighttime(
    timestamps: pd.Series,
    start_hour: int = 22,
    end_hour: int = 7,
) -> pd.Series:
    """
    Determine if timestamps fall within nighttime window.
    
    Handles cross-midnight windows correctly (e.g., 22:00 to 07:00).
    
    Args:
        timestamps: Series of timezone-aware timestamps (should be NYC time)
        start_hour: Start of night window (default 22 = 10 PM)
        end_hour: End of night window (default 7 = 7 AM)
    
    Returns:
        Boolean Series indicating nighttime
    
    Examples:
        - 22:00-07:00: night is 22,23,0,1,2,3,4,5,6 (not 7)
        - 23:00-06:00: night is 23,0,1,2,3,4,5 (not 6)
    """
    hours = timestamps.dt.hour
    
    if start_hour > end_hour:
        # Cross-midnight window (e.g., 22:00 to 07:00)
        # Night is: hour >= start_hour OR hour < end_hour
        return (hours >= start_hour) | (hours < end_hour)
    else:
        # Same-day window (e.g., 01:00 to 05:00)
        # Night is: start_hour <= hour < end_hour
        return (hours >= start_hour) & (hours < end_hour)


def filter_nighttime(
    df: pd.DataFrame,
    timestamp_column: str,
    start_hour: int = 22,
    end_hour: int = 7,
    ensure_timezone: bool = True,
) -> pd.DataFrame:
    """
    Filter DataFrame to nighttime records only.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of timestamp column
        start_hour: Start of night window (default 22)
        end_hour: End of night window (default 7)
        ensure_timezone: If True, ensure timestamps are in NYC timezone
    
    Returns:
        Filtered DataFrame with only nighttime records
    """
    timestamps = df[timestamp_column].copy()
    
    if ensure_timezone:
        timestamps = ensure_nyc_timezone(timestamps)
    
    night_mask = is_nighttime(timestamps, start_hour, end_hour)
    return df[night_mask].copy()


def get_nighttime_window(window: str = "primary") -> Tuple[int, int]:
    """
    Get nighttime window hours from config.
    
    Args:
        window: 'primary' (22:00-07:00) or 'sensitivity' (23:00-06:00)
    
    Returns:
        Tuple of (start_hour, end_hour)
    """
    config = _load_nighttime_config()
    window_config = config.get(window, {})
    
    start_hour = window_config.get("start_hour", 22)
    end_hour = window_config.get("end_hour", 7)
    
    return start_hour, end_hour


# =============================================================================
# Time Window Validation (R14)
# =============================================================================

def assert_temporal_coverage(
    df: pd.DataFrame,
    timestamp_column: str,
    required_start: int,
    required_end: int,
    context: str = "",
) -> None:
    """
    Assert that data covers the required time window.
    
    Per R14: Temporal window checks are executable assertions.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of timestamp column
        required_start: Required start year (inclusive)
        required_end: Required end year (inclusive)
        context: Optional context for error message
    
    Raises:
        ValueError: If data doesn't cover required window
    """
    timestamps = df[timestamp_column]
    
    if len(timestamps) == 0:
        raise ValueError(f"No data to check temporal coverage ({context})")
    
    data_start = timestamps.min().year
    data_end = timestamps.max().year
    
    if data_start > required_start or data_end < required_end:
        msg = (
            f"Temporal coverage insufficient: data covers {data_start}-{data_end}, "
            f"but required {required_start}-{required_end}"
        )
        if context:
            msg = f"{msg} ({context})"
        raise ValueError(msg)


def get_year_range(
    df: pd.DataFrame,
    timestamp_column: str,
) -> Tuple[int, int]:
    """
    Get the year range covered by the data.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of timestamp column
    
    Returns:
        Tuple of (min_year, max_year)
    """
    timestamps = df[timestamp_column]
    return timestamps.min().year, timestamps.max().year


# =============================================================================
# Date/Time Helpers
# =============================================================================

def get_date_range(
    start_year: int,
    end_year: int,
    start_month: int = 1,
    end_month: int = 12,
) -> Tuple[datetime, datetime]:
    """
    Get date range for a given year/month window.
    
    Args:
        start_year: Start year
        end_year: End year
        start_month: Start month (default 1)
        end_month: End month (default 12)
    
    Returns:
        Tuple of (start_datetime, end_datetime) in NYC timezone
    """
    start = NYC_TZ.localize(datetime(start_year, start_month, 1))
    
    # End of the last month
    if end_month == 12:
        end = NYC_TZ.localize(datetime(end_year + 1, 1, 1))
    else:
        end = NYC_TZ.localize(datetime(end_year, end_month + 1, 1))
    
    return start, end


def filter_date_range(
    df: pd.DataFrame,
    timestamp_column: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Filter DataFrame to a specific year range.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of timestamp column
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
    
    Returns:
        Filtered DataFrame
    """
    timestamps = df[timestamp_column]
    years = timestamps.dt.year
    
    mask = (years >= start_year) & (years <= end_year)
    return df[mask].copy()


# =============================================================================
# DST Transition Helpers
# =============================================================================

def get_dst_transitions(year: int) -> dict:
    """
    Get DST transition dates for NYC in a given year.
    
    Useful for debugging DST-related issues.
    
    Args:
        year: Year to check
    
    Returns:
        Dictionary with 'spring_forward' and 'fall_back' dates
    """
    # Get all transitions for the timezone
    transitions = NYC_TZ._utc_transition_times
    
    spring_forward = None
    fall_back = None
    
    for i, t in enumerate(transitions):
        if t.year == year:
            # Check if this is spring forward (gap) or fall back (overlap)
            if i < len(transitions) - 1:
                if spring_forward is None:
                    spring_forward = t
                else:
                    fall_back = t
    
    return {
        "year": year,
        "spring_forward": spring_forward,
        "fall_back": fall_back,
    }


def is_dst(timestamp: Union[datetime, pd.Timestamp]) -> bool:
    """
    Check if a timestamp is during DST in NYC.
    
    Args:
        timestamp: Timestamp to check (should be NYC timezone-aware)
    
    Returns:
        True if DST is in effect
    """
    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    
    if timestamp.tzinfo is None:
        timestamp = NYC_TZ.localize(timestamp)
    
    return bool(timestamp.dst())

