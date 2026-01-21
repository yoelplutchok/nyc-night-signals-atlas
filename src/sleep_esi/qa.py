"""
Quality assurance utilities for geospatial data.

Per R2: CRS mismatches are hard errors. No silent overrides.
Per R3: Bounds sanity checks on every geometry read/write.
Per R20: Directionality checks ("higher = worse") enforced by tests.
"""

from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from sleep_esi.io_utils import read_yaml
from sleep_esi.paths import CONFIG_DIR


# =============================================================================
# Load bounds config
# =============================================================================

def _load_bounds_config() -> dict:
    """Load bounds check configuration from params.yml."""
    params = read_yaml(CONFIG_DIR / "params.yml")
    return params.get("bounds_checks", {})


# =============================================================================
# CRS Validation (R2)
# =============================================================================

class CRSError(Exception):
    """Raised when CRS validation fails."""
    pass


class BoundsError(Exception):
    """Raised when bounds validation fails."""
    pass


def assert_crs_not_none(gdf: gpd.GeoDataFrame, context: str = "") -> None:
    """
    Assert that the GeoDataFrame has a CRS set.
    
    Args:
        gdf: GeoDataFrame to check
        context: Optional context string for error message
        
    Raises:
        CRSError: If CRS is None
    """
    if gdf.crs is None:
        msg = "GeoDataFrame has no CRS set"
        if context:
            msg = f"{msg} ({context})"
        raise CRSError(msg)


def assert_expected_crs(
    gdf: gpd.GeoDataFrame,
    expected_epsg: int,
    context: str = "",
) -> None:
    """
    Assert that the GeoDataFrame has the expected CRS.
    
    Per R2: CRS mismatches are hard errors.
    
    Args:
        gdf: GeoDataFrame to check
        expected_epsg: Expected EPSG code (e.g., 4326, 2263)
        context: Optional context string for error message
        
    Raises:
        CRSError: If CRS is None or doesn't match expected
    """
    assert_crs_not_none(gdf, context)
    
    expected_crs = CRS.from_epsg(expected_epsg)
    
    if not gdf.crs.equals(expected_crs):
        msg = f"CRS mismatch: expected EPSG:{expected_epsg}, got {gdf.crs}"
        if context:
            msg = f"{msg} ({context})"
        raise CRSError(msg)


def get_crs_epsg(gdf: gpd.GeoDataFrame) -> Optional[int]:
    """
    Get the EPSG code of a GeoDataFrame's CRS.
    
    Returns:
        EPSG code or None if CRS is not set or not identifiable
    """
    if gdf.crs is None:
        return None
    return gdf.crs.to_epsg()


def safe_reproject(
    gdf: gpd.GeoDataFrame,
    target_epsg: int,
    context: str = "",
) -> gpd.GeoDataFrame:
    """
    Safely reproject a GeoDataFrame to target CRS.
    
    Per R2: Only uses to_crs(), never set_crs with override.
    
    Args:
        gdf: GeoDataFrame to reproject
        target_epsg: Target EPSG code
        context: Optional context string for error message
        
    Returns:
        Reprojected GeoDataFrame
        
    Raises:
        CRSError: If source CRS is None
    """
    assert_crs_not_none(gdf, context)
    
    target_crs = CRS.from_epsg(target_epsg)
    
    if gdf.crs.equals(target_crs):
        return gdf  # Already in target CRS
    
    return gdf.to_crs(target_crs)


# =============================================================================
# Bounds Validation (R3)
# =============================================================================

def get_bounds(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float]:
    """
    Get bounds of a GeoDataFrame as (minx, miny, maxx, maxy).
    """
    return tuple(gdf.total_bounds)


def check_bounds_epsg4326(
    gdf: gpd.GeoDataFrame,
    lon_min: float = -75.0,
    lon_max: float = -73.0,
    lat_min: float = 40.0,
    lat_max: float = 41.5,
    context: str = "",
) -> bool:
    """
    Check if GeoDataFrame bounds are plausible for NYC in EPSG:4326.
    
    Args:
        gdf: GeoDataFrame in EPSG:4326
        lon_min, lon_max: Expected longitude range
        lat_min, lat_max: Expected latitude range
        context: Optional context for error message
        
    Returns:
        True if bounds are plausible
        
    Raises:
        BoundsError: If bounds are outside expected range
    """
    minx, miny, maxx, maxy = get_bounds(gdf)
    
    errors = []
    if minx < lon_min or maxx > lon_max:
        errors.append(f"Longitude out of range: [{minx}, {maxx}] not in [{lon_min}, {lon_max}]")
    if miny < lat_min or maxy > lat_max:
        errors.append(f"Latitude out of range: [{miny}, {maxy}] not in [{lat_min}, {lat_max}]")
    
    if errors:
        msg = "EPSG:4326 bounds check failed: " + "; ".join(errors)
        if context:
            msg = f"{msg} ({context})"
        raise BoundsError(msg)
    
    return True


def check_bounds_epsg2263(
    gdf: gpd.GeoDataFrame,
    x_min: float = 900000,
    x_max: float = 1100000,
    y_min: float = 110000,
    y_max: float = 280000,
    context: str = "",
) -> bool:
    """
    Check if GeoDataFrame bounds are plausible for NYC in EPSG:2263.
    
    Args:
        gdf: GeoDataFrame in EPSG:2263 (NY StatePlane feet)
        x_min, x_max: Expected X range
        y_min, y_max: Expected Y range
        context: Optional context for error message
        
    Returns:
        True if bounds are plausible
        
    Raises:
        BoundsError: If bounds are outside expected range
    """
    minx, miny, maxx, maxy = get_bounds(gdf)
    
    errors = []
    if minx < x_min or maxx > x_max:
        errors.append(f"X out of range: [{minx}, {maxx}] not in [{x_min}, {x_max}]")
    if miny < y_min or maxy > y_max:
        errors.append(f"Y out of range: [{miny}, {maxy}] not in [{y_min}, {y_max}]")
    
    if errors:
        msg = "EPSG:2263 bounds check failed: " + "; ".join(errors)
        if context:
            msg = f"{msg} ({context})"
        raise BoundsError(msg)
    
    return True


def validate_bounds(
    gdf: gpd.GeoDataFrame,
    context: str = "",
) -> bool:
    """
    Validate bounds based on the GeoDataFrame's CRS.
    
    Automatically selects the appropriate bounds check.
    
    Args:
        gdf: GeoDataFrame to validate
        context: Optional context for error message
        
    Returns:
        True if bounds are valid
        
    Raises:
        CRSError: If CRS is None
        BoundsError: If bounds are outside expected range
    """
    assert_crs_not_none(gdf, context)
    
    epsg = get_crs_epsg(gdf)
    bounds_config = _load_bounds_config()
    
    if epsg == 4326:
        config = bounds_config.get("epsg_4326", {})
        return check_bounds_epsg4326(
            gdf,
            lon_min=config.get("lon_min", -75.0),
            lon_max=config.get("lon_max", -73.0),
            lat_min=config.get("lat_min", 40.0),
            lat_max=config.get("lat_max", 41.5),
            context=context,
        )
    elif epsg == 2263:
        config = bounds_config.get("epsg_2263", {})
        return check_bounds_epsg2263(
            gdf,
            x_min=config.get("x_min", 900000),
            x_max=config.get("x_max", 1100000),
            y_min=config.get("y_min", 110000),
            y_max=config.get("y_max", 280000),
            context=context,
        )
    else:
        # For other CRS, just check that bounds are finite
        bounds = get_bounds(gdf)
        if not all(np.isfinite(bounds)):
            raise BoundsError(f"Non-finite bounds: {bounds} ({context})")
        return True


# =============================================================================
# Geometry Validation
# =============================================================================

def check_geometry_validity(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Check geometry validity and return a summary.
    
    Args:
        gdf: GeoDataFrame to check
        
    Returns:
        DataFrame with validity info for each geometry
    """
    return pd.DataFrame({
        "is_valid": gdf.geometry.is_valid,
        "is_empty": gdf.geometry.is_empty,
        "geom_type": gdf.geometry.geom_type,
    })


def assert_all_valid(gdf: gpd.GeoDataFrame, context: str = "") -> None:
    """
    Assert all geometries are valid.
    
    Args:
        gdf: GeoDataFrame to check
        context: Optional context for error message
        
    Raises:
        ValueError: If any geometry is invalid
    """
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        msg = f"{n_invalid} invalid geometries found"
        if context:
            msg = f"{msg} ({context})"
        raise ValueError(msg)


# =============================================================================
# Directionality Checks (R20)
# =============================================================================

def check_directionality_higher_worse(
    raw_values: pd.Series,
    standardized_values: pd.Series,
    tolerance: float = 0.8,
) -> bool:
    """
    Check that higher raw values correspond to higher standardized values.
    
    Per R20: Directionality ("higher=worse") is enforced by tests.
    
    Args:
        raw_values: Original metric values
        standardized_values: Standardized (z-score or similar) values
        tolerance: Minimum correlation to pass (default 0.8)
        
    Returns:
        True if correlation is positive and above tolerance
        
    Raises:
        ValueError: If directionality is wrong (negative correlation)
    """
    # Remove NaN pairs
    mask = raw_values.notna() & standardized_values.notna()
    raw_clean = raw_values[mask]
    std_clean = standardized_values[mask]
    
    if len(raw_clean) < 3:
        raise ValueError("Insufficient data for directionality check")
    
    correlation = raw_clean.corr(std_clean)
    
    if correlation < 0:
        raise ValueError(
            f"Directionality violation: correlation={correlation:.3f} "
            "(expected positive: higher raw → higher standardized)"
        )
    
    if correlation < tolerance:
        raise ValueError(
            f"Weak directionality: correlation={correlation:.3f} "
            f"(expected >= {tolerance})"
        )
    
    return True


# =============================================================================
# CD Label Validation (Reporting QA)
# =============================================================================

def assert_cd_labels_present(df: pd.DataFrame, context: str = "dataframe") -> None:
    """
    QA check to ensure cd_lookup has been joined into a reporting table.
    
    Per NYC_Night_Signals_Plan.md Section 2: Never print bare boro_cd without label.
    
    Args:
        df: DataFrame that should have CD labels
        context: Description for error messages
    
    Raises:
        AssertionError if cd_label column is missing or has nulls when boro_cd is present
    """
    if "boro_cd" in df.columns:
        assert "cd_label" in df.columns, \
            f"{context}: Has 'boro_cd' but missing 'cd_label'. " \
            f"Join cd_lookup.parquet before generating reports."
        
        null_labels = df["cd_label"].isna().sum()
        assert null_labels == 0, \
            f"{context}: Found {null_labels} null cd_label values after join."


# =============================================================================
# Data Quality Summaries
# =============================================================================

def compute_na_rates(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute NA rates for all columns in a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary of column_name -> NA rate (0-1)
    """
    return (df.isna().sum() / len(df)).to_dict()


def compute_coverage_stats(
    gdf: gpd.GeoDataFrame,
    value_column: str,
) -> dict:
    """
    Compute coverage statistics for a value column.
    
    Args:
        gdf: GeoDataFrame with values
        value_column: Column to analyze
        
    Returns:
        Dictionary with coverage stats
    """
    values = gdf[value_column]
    return {
        "n_total": len(values),
        "n_valid": values.notna().sum(),
        "n_missing": values.isna().sum(),
        "coverage_rate": values.notna().sum() / len(values),
        "min": values.min() if values.notna().any() else None,
        "max": values.max() if values.notna().any() else None,
        "mean": values.mean() if values.notna().any() else None,
    }

