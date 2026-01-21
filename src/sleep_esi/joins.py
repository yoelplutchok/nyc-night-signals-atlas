"""
Hardened spatial join utilities.

Per R6: Point→polygon joins must use the shared hardened join utility:
- within first, then nearest with max distance
- logged distance distribution
- hard failure beyond threshold
"""

from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from sleep_esi.io_utils import read_yaml
from sleep_esi.paths import CONFIG_DIR
from sleep_esi.qa import assert_crs_not_none, safe_reproject


class SpatialJoinError(Exception):
    """Raised when spatial join fails validation."""
    pass


def _load_join_config() -> dict:
    """Load spatial join configuration from params.yml."""
    params = read_yaml(CONFIG_DIR / "params.yml")
    return params.get("spatial_join", {})


def spatial_join_points_to_polygons(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    polygon_id_col: str = "boro_cd",
    max_distance: Optional[float] = None,
    projected_crs: int = 2263,
    return_stats: bool = False,
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Hardened spatial join of points to polygons.
    
    Per R6:
    1. Assert CRS and reproject to projected CRS (EPSG:2263)
    2. sjoin(within) first
    3. Unmatched → sjoin_nearest with max distance threshold
    4. Log unmatched counts and distance distribution
    5. Fail if max distance > threshold
    
    Args:
        points: GeoDataFrame of points to join
        polygons: GeoDataFrame of polygons to join to
        polygon_id_col: Column name for polygon ID (default: boro_cd)
        max_distance: Maximum distance for nearest join (in CRS units).
                     If None, uses config default.
        projected_crs: EPSG code for projected CRS (default: 2263)
        return_stats: If True, return stats dict even on success
    
    Returns:
        Tuple of (joined GeoDataFrame, stats dictionary)
    
    Raises:
        SpatialJoinError: If join fails validation (too many unmatched, 
                         distances exceed threshold, etc.)
    """
    # Load config defaults
    config = _load_join_config()
    if max_distance is None:
        max_distance = config.get("max_distance_ft", 500)
    
    # Step 1: Assert CRS and reproject
    assert_crs_not_none(points, "points input")
    assert_crs_not_none(polygons, "polygons input")
    
    points_proj = safe_reproject(points, projected_crs, "points")
    polygons_proj = safe_reproject(polygons, projected_crs, "polygons")
    
    # Initialize stats
    stats = {
        "total_points": len(points_proj),
        "matched_within": 0,
        "matched_nearest": 0,
        "unmatched": 0,
        "max_distance_used": 0,
        "mean_distance": None,
        "p95_distance": None,
        "distances": [],
    }
    
    # Step 2: sjoin within first
    # Keep original index for tracking
    points_proj = points_proj.copy()
    points_proj["_original_idx"] = points_proj.index
    
    joined_within = gpd.sjoin(
        points_proj,
        polygons_proj[[polygon_id_col, "geometry"]],
        how="inner",
        predicate="within",
    )
    
    matched_within_idx = set(joined_within["_original_idx"])
    stats["matched_within"] = len(matched_within_idx)
    
    # Step 3: Find unmatched points
    unmatched_mask = ~points_proj["_original_idx"].isin(matched_within_idx)
    unmatched_points = points_proj[unmatched_mask].copy()
    
    if len(unmatched_points) > 0:
        # Step 4: sjoin_nearest for unmatched
        joined_nearest = gpd.sjoin_nearest(
            unmatched_points,
            polygons_proj[[polygon_id_col, "geometry"]],
            how="left",
            distance_col="_join_distance",
            max_distance=max_distance,
        )
        
        # Track distances
        distances = joined_nearest["_join_distance"].dropna()
        if len(distances) > 0:
            stats["distances"] = distances.tolist()
            stats["max_distance_used"] = float(distances.max())
            stats["mean_distance"] = float(distances.mean())
            stats["p95_distance"] = float(np.percentile(distances, 95))
        
        # Count matched via nearest
        matched_nearest_mask = joined_nearest[polygon_id_col].notna()
        stats["matched_nearest"] = int(matched_nearest_mask.sum())
        stats["unmatched"] = int((~matched_nearest_mask).sum())
        
        # Check for distances exceeding threshold
        if stats["max_distance_used"] > max_distance:
            raise SpatialJoinError(
                f"Nearest join exceeded max distance: {stats['max_distance_used']:.1f} > {max_distance}"
            )
        
        # Combine results
        # Remove the join columns from within results to avoid conflicts
        joined_within_clean = joined_within.drop(columns=["index_right"], errors="ignore")
        joined_nearest_clean = joined_nearest.drop(columns=["index_right", "_join_distance"], errors="ignore")
        
        # Concatenate
        result = pd.concat([joined_within_clean, joined_nearest_clean], ignore_index=True)
    else:
        # All points matched within
        result = joined_within.drop(columns=["index_right"], errors="ignore")
    
    # Clean up
    result = result.drop(columns=["_original_idx"], errors="ignore")
    
    # Validate: check for any remaining unmatched
    if stats["unmatched"] > 0:
        unmatched_rate = stats["unmatched"] / stats["total_points"]
        if unmatched_rate > 0.01:  # More than 1% unmatched
            raise SpatialJoinError(
                f"Too many unmatched points: {stats['unmatched']} ({unmatched_rate:.1%})"
            )
    
    # Convert back to GeoDataFrame if needed
    if not isinstance(result, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result, geometry="geometry", crs=projected_crs)
    
    return result, stats


def aggregate_points_to_polygons(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    polygon_id_col: str = "boro_cd",
    agg_funcs: Optional[Dict[str, str]] = None,
    count_col: str = "point_count",
    max_distance: Optional[float] = None,
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Join points to polygons and aggregate counts/statistics.
    
    Args:
        points: GeoDataFrame of points
        polygons: GeoDataFrame of polygons
        polygon_id_col: Column name for polygon ID
        agg_funcs: Dictionary of column -> aggregation function
        count_col: Name for count column in output
        max_distance: Maximum distance for nearest join
    
    Returns:
        Tuple of (aggregated GeoDataFrame with polygon geometry, join stats)
    """
    # Perform join
    joined, stats = spatial_join_points_to_polygons(
        points, polygons, polygon_id_col, max_distance
    )
    
    # Count points per polygon
    counts = joined.groupby(polygon_id_col).size().reset_index(name=count_col)
    
    # Additional aggregations if specified
    if agg_funcs:
        aggs = joined.groupby(polygon_id_col).agg(agg_funcs).reset_index()
        counts = counts.merge(aggs, on=polygon_id_col, how="left")
    
    # Merge back to polygons to include polygons with zero points
    result = polygons[[polygon_id_col, "geometry"]].merge(
        counts,
        on=polygon_id_col,
        how="left",
    )
    
    # Fill NaN counts with 0
    result[count_col] = result[count_col].fillna(0).astype("Int64")
    
    return result, stats


def log_join_stats(stats: Dict, logger=None) -> None:
    """
    Log spatial join statistics.
    
    Args:
        stats: Statistics dictionary from spatial join
        logger: Optional logger instance (uses print if None)
    """
    msg = (
        f"Spatial join stats: "
        f"{stats['total_points']} total, "
        f"{stats['matched_within']} within, "
        f"{stats['matched_nearest']} nearest, "
        f"{stats['unmatched']} unmatched"
    )
    
    if stats.get("max_distance_used"):
        msg += f" | max_dist={stats['max_distance_used']:.1f}"
    if stats.get("mean_distance"):
        msg += f", mean_dist={stats['mean_distance']:.1f}"
    if stats.get("p95_distance"):
        msg += f", p95_dist={stats['p95_distance']:.1f}"
    
    if logger:
        logger.info(msg, extra={"join_stats": stats})
    else:
        print(msg)

