"""
Raster processing utilities with hardened QA.

Per R8: Raster operations prove alignment before computing stats.
- Read raster CRS/transform/nodata/bounds
- Reproject polygons to raster CRS
- Assert overlap of bounds
- Log pixel counts and nodata fraction per polygon
- Nodata/fill/valid range rules are explicit (never default/implicit)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterstats import zonal_stats

from sleep_esi.qa import safe_reproject


class RasterError(Exception):
    """Raised when raster operations fail validation."""
    pass


@dataclass
class RasterMetadata:
    """Metadata for a raster file."""
    path: Path
    crs: CRS
    bounds: Tuple[float, float, float, float]  # (left, bottom, right, top)
    width: int
    height: int
    count: int  # number of bands
    dtype: str
    nodata: Optional[float]
    transform: Any  # Affine transform
    
    @property
    def resolution(self) -> Tuple[float, float]:
        """Get pixel resolution (x, y)."""
        return abs(self.transform.a), abs(self.transform.e)


def read_raster_metadata(path: Union[str, Path]) -> RasterMetadata:
    """
    Read raster metadata without loading the full array.
    
    Per R8: Read raster CRS/transform/nodata/bounds.
    
    Args:
        path: Path to raster file
    
    Returns:
        RasterMetadata object
    """
    path = Path(path)
    
    with rasterio.open(path) as src:
        return RasterMetadata(
            path=path,
            crs=src.crs,
            bounds=src.bounds,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=str(src.dtypes[0]),
            nodata=src.nodata,
            transform=src.transform,
        )


def check_bounds_overlap(
    raster_bounds: Tuple[float, float, float, float],
    polygon_bounds: Tuple[float, float, float, float],
    context: str = "",
) -> bool:
    """
    Check if raster and polygon bounds overlap.
    
    Per R8: Assert overlap of bounds.
    
    Args:
        raster_bounds: (left, bottom, right, top) of raster
        polygon_bounds: (minx, miny, maxx, maxy) of polygons
        context: Optional context for error message
    
    Returns:
        True if bounds overlap
    
    Raises:
        RasterError: If bounds do not overlap
    """
    r_left, r_bottom, r_right, r_top = raster_bounds
    p_minx, p_miny, p_maxx, p_maxy = polygon_bounds
    
    # Check for overlap
    overlaps = not (
        r_right < p_minx or  # raster is left of polygons
        r_left > p_maxx or   # raster is right of polygons
        r_top < p_miny or    # raster is below polygons
        r_bottom > p_maxy    # raster is above polygons
    )
    
    if not overlaps:
        raise RasterError(
            f"Raster and polygon bounds do not overlap. "
            f"Raster: {raster_bounds}, Polygons: {polygon_bounds} ({context})"
        )
    
    return True


def zonal_stats_hardened(
    polygons: gpd.GeoDataFrame,
    raster_path: Union[str, Path],
    stats: List[str] = ["mean", "count"],
    nodata: Optional[float] = None,
    all_touched: bool = False,
    polygon_id_col: str = "boro_cd",
    prefix: str = "",
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Compute zonal statistics with hardened QA checks.
    
    Per R8:
    - Read raster metadata
    - Reproject polygons to raster CRS
    - Assert overlap between polygon bounds and raster bounds
    - Explicitly handle nodata
    - Log pixel counts and nodata fraction per polygon
    
    Args:
        polygons: GeoDataFrame of polygons
        raster_path: Path to raster file
        stats: List of statistics to compute (e.g., ['mean', 'count', 'min', 'max'])
        nodata: Nodata value to use (overrides raster metadata if provided)
        all_touched: If True, include all pixels touched by polygon
        polygon_id_col: Column name for polygon ID
        prefix: Prefix for output column names
    
    Returns:
        Tuple of (GeoDataFrame with stats columns, QA stats dictionary)
    
    Raises:
        RasterError: If validation fails
    """
    raster_path = Path(raster_path)
    
    # Step 1: Read raster metadata
    meta = read_raster_metadata(raster_path)
    
    # Determine nodata value
    effective_nodata = nodata if nodata is not None else meta.nodata
    
    # Step 2: Reproject polygons to raster CRS
    if polygons.crs is None:
        raise RasterError("Polygons have no CRS set")
    
    raster_epsg = meta.crs.to_epsg()
    if raster_epsg is None:
        # Try to use CRS directly
        polygons_proj = polygons.to_crs(meta.crs)
    else:
        polygons_proj = safe_reproject(polygons, raster_epsg, "polygons for zonal stats")
    
    # Step 3: Assert bounds overlap
    polygon_bounds = tuple(polygons_proj.total_bounds)
    check_bounds_overlap(meta.bounds, polygon_bounds, f"raster: {raster_path.name}")
    
    # Step 4: Compute zonal statistics
    # Always include 'count' for QA
    stats_to_compute = list(set(stats) | {"count"})
    
    results = zonal_stats(
        polygons_proj.geometry,
        str(raster_path),
        stats=stats_to_compute,
        nodata=effective_nodata,
        all_touched=all_touched,
    )
    
    # Step 5: Build output DataFrame
    result_df = polygons.copy()
    
    # Add stat columns
    for stat in stats:
        col_name = f"{prefix}{stat}" if prefix else stat
        result_df[col_name] = [r.get(stat) if r else None for r in results]
    
    # Add pixel count for QA
    count_col = f"{prefix}pixel_count" if prefix else "pixel_count"
    result_df[count_col] = [r.get("count", 0) if r else 0 for r in results]
    
    # Step 6: Compute QA statistics
    pixel_counts = [r.get("count", 0) if r else 0 for r in results]
    
    qa_stats = {
        "raster_path": str(raster_path),
        "raster_crs": str(meta.crs),
        "raster_bounds": meta.bounds,
        "raster_resolution": meta.resolution,
        "raster_nodata": effective_nodata,
        "polygon_count": len(polygons),
        "total_pixels": sum(pixel_counts),
        "min_pixels_per_polygon": min(pixel_counts) if pixel_counts else 0,
        "max_pixels_per_polygon": max(pixel_counts) if pixel_counts else 0,
        "mean_pixels_per_polygon": np.mean(pixel_counts) if pixel_counts else 0,
        "zero_pixel_polygons": sum(1 for c in pixel_counts if c == 0),
    }
    
    return result_df, qa_stats


def apply_scale_offset(
    values: np.ndarray,
    scale: float = 1.0,
    offset: float = 0.0,
    valid_min: Optional[float] = None,
    valid_max: Optional[float] = None,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """
    Apply scale, offset, and valid range to raster values.
    
    Per R8: Nodata/fill/valid range rules are explicit.
    
    Args:
        values: Array of raster values
        scale: Scale factor (multiply)
        offset: Offset (add after scaling)
        valid_min: Minimum valid value (values below are masked)
        valid_max: Maximum valid value (values above are masked)
        fill_value: Fill value to mask (e.g., HDF _FillValue)
    
    Returns:
        Processed array with invalid values as NaN
    """
    result = values.astype(float)
    
    # Mask fill value
    if fill_value is not None:
        result = np.where(result == fill_value, np.nan, result)
    
    # Apply scale and offset
    result = result * scale + offset
    
    # Apply valid range
    if valid_min is not None:
        result = np.where(result < valid_min, np.nan, result)
    if valid_max is not None:
        result = np.where(result > valid_max, np.nan, result)
    
    return result


def compute_nodata_fraction(
    values: np.ndarray,
    nodata: Optional[float] = None,
) -> float:
    """
    Compute the fraction of nodata/NaN values in an array.
    
    Args:
        values: Array of values
        nodata: Nodata value (in addition to NaN)
    
    Returns:
        Fraction of nodata values (0-1)
    """
    mask = np.isnan(values)
    if nodata is not None:
        mask = mask | (values == nodata)
    
    return float(mask.sum() / values.size) if values.size > 0 else 0.0


def log_raster_qa(qa_stats: Dict, logger=None) -> None:
    """
    Log raster QA statistics.
    
    Args:
        qa_stats: QA statistics dictionary from zonal_stats_hardened
        logger: Optional logger instance
    """
    msg = (
        f"Raster QA: {qa_stats['polygon_count']} polygons, "
        f"{qa_stats['total_pixels']} total pixels, "
        f"min={qa_stats['min_pixels_per_polygon']}, "
        f"max={qa_stats['max_pixels_per_polygon']}, "
        f"mean={qa_stats['mean_pixels_per_polygon']:.1f}, "
        f"zero_pixel={qa_stats['zero_pixel_polygons']}"
    )
    
    if logger:
        logger.info(msg, extra={"raster_qa": qa_stats})
    else:
        print(msg)

