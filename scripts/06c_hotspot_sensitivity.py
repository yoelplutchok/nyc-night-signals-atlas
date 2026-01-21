#!/usr/bin/env python3
"""
06c_hotspot_sensitivity.py

Hotspot Sensitivity Analysis + Map-Grade Exports with Artifact Flagging.

Per NYC_Night_Signals_Plan.md Section 3.3 (Script 06c):

1. Merge Script 06b investigation results into hotspot layers by cell_id
2. Add artifact flags based on config thresholds:
   - is_repeat_location_dominant: top_latlon_share >= repeat_location_share_threshold
   - is_suspected_artifact: top_latlon_share >= suspected_artifact_share_threshold 
     OR max_single_day_count >= suspected_artifact_single_day_threshold
3. Run sensitivity analysis across grid_sizes x thresholds
4. Output:
   - hotspot_cells_ge10.geojson (analysis-grade)
   - hotspot_cells_ge50.geojson (map-grade)
   - hotspot_sensitivity_summary.csv
   - cd_hotspot_concentration.parquet (raw)
   - cd_hotspot_concentration_clean.parquet (excludes suspected artifacts)

Requirements:
- Deterministic given fixed parameters
- Artifact flags are config-driven (not hardcoded)
- Raw outputs preserved; clean is for robustness lens only
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from sleep_esi.hashing import write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_df, atomic_write_gdf, read_yaml, read_gdf
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, RAW_DIR, PROCESSED_DIR
from sleep_esi.qa import safe_reproject, assert_cd_labels_present
from sleep_esi.schemas import ensure_boro_cd_dtype
from sleep_esi.time_utils import ensure_nyc_timezone, filter_nighttime


# =============================================================================
# Constants
# =============================================================================

RAW_311_DIR = RAW_DIR / "311_noise"
HOTSPOTS_DIR = PROCESSED_DIR / "hotspots"

# Outputs
OUTPUT_GE10 = HOTSPOTS_DIR / "hotspot_cells_ge10.geojson"
OUTPUT_GE50 = HOTSPOTS_DIR / "hotspot_cells_ge50.geojson"
OUTPUT_SENSITIVITY = HOTSPOTS_DIR / "hotspot_sensitivity_summary.csv"
OUTPUT_CD_CONCENTRATION = HOTSPOTS_DIR / "cd_hotspot_concentration.parquet"
OUTPUT_CD_CONCENTRATION_CSV = HOTSPOTS_DIR / "cd_hotspot_concentration.csv"
OUTPUT_CD_CONCENTRATION_CLEAN = HOTSPOTS_DIR / "cd_hotspot_concentration_clean.parquet"
OUTPUT_CD_CONCENTRATION_CLEAN_CSV = HOTSPOTS_DIR / "cd_hotspot_concentration_clean.csv"

# Investigation results from 06b
INVESTIGATION_CSV = HOTSPOTS_DIR / "hotspot_investigation_top_cells.csv"


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_311(logger) -> Tuple[pd.DataFrame, Path]:
    """Load the most recent raw 311 noise data."""
    raw_files = list(RAW_311_DIR.glob("raw_311_noise_*.csv"))
    
    if not raw_files:
        raise FileNotFoundError(
            f"No raw 311 noise files found in {RAW_311_DIR}. "
            "Run 02_fetch_311_noise.py first."
        )
    
    raw_path = max(raw_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading raw 311 data from: {raw_path}")
    
    df = pd.read_csv(raw_path, low_memory=False)
    logger.info(f"Loaded {len(df):,} raw records")
    
    return df, raw_path


def load_cd59(logger) -> gpd.GeoDataFrame:
    """Load the canonical CD59 geometries."""
    cd59_path = GEO_DIR / "cd59.parquet"
    
    if not cd59_path.exists():
        raise FileNotFoundError(
            f"CD59 file not found: {cd59_path}. "
            "Run 00_build_geographies.py first."
        )
    
    gdf = read_gdf(cd59_path)
    gdf = ensure_boro_cd_dtype(gdf)
    logger.info(f"Loaded CD59: {len(gdf)} community districts")
    
    return gdf


def load_cd_lookup(logger) -> pd.DataFrame:
    """Load the CD lookup table for labels."""
    lookup_path = GEO_DIR / "cd_lookup.parquet"
    
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"CD lookup not found: {lookup_path}. "
            "Run 00_build_geographies.py first."
        )
    
    df = pd.read_parquet(lookup_path)
    df = ensure_boro_cd_dtype(df)
    logger.info(f"Loaded CD lookup: {len(df)} entries")
    
    return df


def load_investigation_results(logger) -> pd.DataFrame:
    """Load the investigation results from Script 06b."""
    if not INVESTIGATION_CSV.exists():
        logger.warning(
            f"Investigation results not found: {INVESTIGATION_CSV}. "
            "Run 06b_hotspot_investigation.py first. "
            "Proceeding without artifact flags for investigated cells."
        )
        return pd.DataFrame()
    
    df = pd.read_csv(INVESTIGATION_CSV)
    logger.info(f"Loaded investigation results: {len(df)} cells")
    
    return df


# =============================================================================
# Grid Functions
# =============================================================================

def create_grid(
    bounds: Tuple[float, float, float, float],
    cell_size: float,
    crs: str,
) -> gpd.GeoDataFrame:
    """Create a regular square grid covering the given bounds."""
    minx, miny, maxx, maxy = bounds
    
    n_cols = int(np.ceil((maxx - minx) / cell_size))
    n_rows = int(np.ceil((maxy - miny) / cell_size))
    
    cells = []
    cell_id = 0
    
    for row in range(n_rows):
        for col in range(n_cols):
            x0 = minx + col * cell_size
            y0 = miny + row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            
            cells.append({
                "cell_id": cell_id,
                "row": row,
                "col": col,
                "geometry": box(x0, y0, x1, y1),
            })
            cell_id += 1
    
    gdf = gpd.GeoDataFrame(cells, crs=crs)
    
    return gdf


def clip_grid_to_boundary(
    grid: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Clip grid to NYC boundary."""
    if grid.crs != boundary.crs:
        boundary = boundary.to_crs(grid.crs)
    
    nyc_boundary = boundary.union_all()
    grid["intersects_nyc"] = grid.geometry.intersects(nyc_boundary)
    grid_clipped = grid[grid["intersects_nyc"]].copy()
    grid_clipped = grid_clipped.drop(columns=["intersects_nyc"])
    
    return grid_clipped


def assign_cells_to_cds(
    grid: gpd.GeoDataFrame,
    cd59: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assign each grid cell to a CD based on cell centroid."""
    if grid.crs != cd59.crs:
        cd59_proj = cd59.to_crs(grid.crs)
    else:
        cd59_proj = cd59
    
    grid_centroids = grid.copy()
    grid_centroids["geometry"] = grid_centroids.geometry.centroid
    
    joined = gpd.sjoin(
        grid_centroids,
        cd59_proj[["boro_cd", "geometry"]],
        how="left",
        predicate="within",
    )
    
    grid_with_cd = grid.merge(
        joined[["cell_id", "boro_cd"]],
        on="cell_id",
        how="left",
    )
    
    return grid_with_cd


# =============================================================================
# Point Processing
# =============================================================================

def prepare_311_points(df: pd.DataFrame, logger) -> gpd.GeoDataFrame:
    """Prepare 311 data as GeoDataFrame with valid coordinates."""
    # Filter to records with valid coordinates
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    df_valid = df[has_coords].copy()
    
    # Parse timestamps
    df_valid["created_date"] = pd.to_datetime(df_valid["created_date"], errors="coerce")
    valid_ts = df_valid["created_date"].notna()
    df_valid = df_valid[valid_ts].copy()
    
    # Convert to NYC timezone
    df_valid["ts_nyc"] = ensure_nyc_timezone(df_valid["created_date"])
    
    # Create geometry
    gdf = gpd.GeoDataFrame(
        df_valid,
        geometry=gpd.points_from_xy(df_valid["longitude"], df_valid["latitude"]),
        crs="EPSG:4326",
    )
    
    return gdf


def filter_to_nighttime_and_years(
    gdf: gpd.GeoDataFrame,
    start_hour: int,
    end_hour: int,
    year_start: int,
    year_end: int,
) -> gpd.GeoDataFrame:
    """Filter to nighttime hours and date range."""
    gdf = gdf.copy()
    gdf["year"] = gdf["ts_nyc"].dt.year
    gdf = gdf[(gdf["year"] >= year_start) & (gdf["year"] <= year_end)]
    gdf = filter_nighttime(gdf, "ts_nyc", start_hour=start_hour, end_hour=end_hour)
    
    return gdf


def assign_points_to_grid(
    points: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Assign points to grid cells via spatial join."""
    if points.crs != grid.crs:
        points_proj = points.to_crs(grid.crs)
    else:
        points_proj = points
    
    # Spatial join
    joined = gpd.sjoin(points_proj, grid[["cell_id", "geometry"]], how="left", predicate="within")
    
    # Aggregate counts per cell
    cell_counts = joined.groupby("cell_id").size().reset_index(name="count")
    cell_counts["cell_id"] = cell_counts["cell_id"].astype(int)
    
    return joined, cell_counts


# =============================================================================
# Artifact Flagging
# =============================================================================

def compute_artifact_flags_for_all_cells(
    points_with_cells: gpd.GeoDataFrame,
    flag_config: Dict,
    latlon_decimals: int,
    logger,
) -> pd.DataFrame:
    """
    Compute artifact flags for ALL cells (not just top N).
    
    Returns DataFrame with cell_id and flag columns.
    """
    logger.info("Computing artifact flags for all cells...")
    
    repeat_threshold = flag_config.get("repeat_location_share_threshold", 0.90)
    artifact_share_threshold = flag_config.get("suspected_artifact_share_threshold", 0.95)
    artifact_day_threshold = flag_config.get("suspected_artifact_single_day_threshold", 200)
    
    logger.info(f"  repeat_location_share_threshold: {repeat_threshold}")
    logger.info(f"  suspected_artifact_share_threshold: {artifact_share_threshold}")
    logger.info(f"  suspected_artifact_single_day_threshold: {artifact_day_threshold}")
    
    # Filter to assigned points
    assigned = points_with_cells[points_with_cells["cell_id"].notna()].copy()
    assigned["cell_id"] = assigned["cell_id"].astype(int)
    
    # Round coordinates
    assigned["lat_rounded"] = assigned["latitude"].round(latlon_decimals)
    assigned["lon_rounded"] = assigned["longitude"].round(latlon_decimals)
    assigned["latlon_rounded"] = (
        assigned["lat_rounded"].astype(str) + "," + 
        assigned["lon_rounded"].astype(str)
    )
    
    # Add date for daily counts
    assigned["date"] = assigned["ts_nyc"].dt.date
    
    results = []
    
    for cell_id, cell_points in assigned.groupby("cell_id"):
        n = len(cell_points)
        
        # Top coordinate share
        latlon_counts = cell_points["latlon_rounded"].value_counts()
        top_latlon_count = latlon_counts.iloc[0] if len(latlon_counts) > 0 else 0
        top_latlon_share = top_latlon_count / n if n > 0 else 0
        top_latlon = latlon_counts.index[0] if len(latlon_counts) > 0 else None
        
        # Max single-day count
        daily_counts = cell_points.groupby("date").size()
        max_single_day_count = int(daily_counts.max()) if len(daily_counts) > 0 else 0
        
        # Top address share (if available)
        top_address_share = None
        if "incident_address" in cell_points.columns:
            addresses = cell_points["incident_address"].fillna("UNKNOWN").str.upper().str.strip()
            address_counts = addresses.value_counts()
            top_address_count = address_counts.iloc[0] if len(address_counts) > 0 else 0
            top_address_share = top_address_count / n if n > 0 else 0
        
        # Compute flags
        is_repeat_location_dominant = top_latlon_share >= repeat_threshold
        is_suspected_artifact = (
            top_latlon_share >= artifact_share_threshold or
            max_single_day_count >= artifact_day_threshold
        )
        
        results.append({
            "cell_id": int(cell_id),
            "complaint_count": n,
            "top_latlon_share": round(top_latlon_share, 4),
            "top_latlon": top_latlon,
            "top_address_share": round(top_address_share, 4) if top_address_share is not None else None,
            "max_single_day_count": max_single_day_count,
            "is_repeat_location_dominant": is_repeat_location_dominant,
            "is_suspected_artifact": is_suspected_artifact,
        })
    
    df = pd.DataFrame(results)
    
    n_repeat = df["is_repeat_location_dominant"].sum()
    n_artifact = df["is_suspected_artifact"].sum()
    logger.info(f"  Cells flagged as repeat_location_dominant: {n_repeat}")
    logger.info(f"  Cells flagged as suspected_artifact: {n_artifact}")
    
    return df


# =============================================================================
# Concentration Metrics
# =============================================================================

def compute_gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for spatial concentration."""
    values = np.array(values, dtype=float)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumulative = np.cumsum(sorted_values)
    
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    
    return float(gini)


def compute_cd_concentration_metrics(
    grid_stats: gpd.GeoDataFrame,
    cd_lookup: pd.DataFrame,
    top_n: int,
    hotspot_threshold: int,
    exclude_artifacts: bool = False,
    logger = None,
) -> pd.DataFrame:
    """Compute concentration metrics for each CD."""
    # Filter to cells with counts and assigned to CDs
    cells_with_counts = grid_stats[
        (grid_stats["count"] > 0) & (grid_stats["boro_cd"].notna())
    ].copy()
    cells_with_counts["boro_cd"] = cells_with_counts["boro_cd"].astype(int)
    
    # Optionally exclude suspected artifacts
    if exclude_artifacts and "is_suspected_artifact" in cells_with_counts.columns:
        n_before = len(cells_with_counts)
        cells_with_counts = cells_with_counts[~cells_with_counts["is_suspected_artifact"]]
        n_after = len(cells_with_counts)
        if logger:
            logger.info(f"  Excluded {n_before - n_after} suspected artifact cells for clean metrics")
    
    # Compute citywide top 1% threshold
    all_counts = cells_with_counts["count"]
    top_1pct_threshold = all_counts.quantile(0.99) if len(all_counts) > 0 else 0
    cells_with_counts["is_top_1pct"] = cells_with_counts["count"] >= top_1pct_threshold
    cells_with_counts["is_hotspot"] = cells_with_counts["count"] >= hotspot_threshold
    
    results = []
    
    for boro_cd in sorted(cells_with_counts["boro_cd"].unique()):
        cd_cells = cells_with_counts[cells_with_counts["boro_cd"] == boro_cd]
        
        if len(cd_cells) == 0:
            continue
        
        total_complaints = cd_cells["count"].sum()
        
        # Top 1% share
        top_1pct_complaints = cd_cells[cd_cells["is_top_1pct"]]["count"].sum()
        top_1pct_share = top_1pct_complaints / total_complaints if total_complaints > 0 else 0
        
        # Top N cells share
        top_n_cells = cd_cells.nlargest(top_n, "count")
        top_n_complaints = top_n_cells["count"].sum()
        top_n_share = top_n_complaints / total_complaints if total_complaints > 0 else 0
        
        # Hotspot count
        hotspot_count = cd_cells["is_hotspot"].sum()
        
        # Gini coefficient
        gini = compute_gini_coefficient(cd_cells["count"].values)
        
        # Max single cell share
        max_cell_count = cd_cells["count"].max()
        max_cell_share = max_cell_count / total_complaints if total_complaints > 0 else 0
        
        # Artifact counts (if available)
        n_repeat_location = 0
        n_suspected_artifact = 0
        if "is_repeat_location_dominant" in cd_cells.columns:
            n_repeat_location = cd_cells["is_repeat_location_dominant"].sum()
        if "is_suspected_artifact" in cd_cells.columns:
            n_suspected_artifact = cd_cells["is_suspected_artifact"].sum()
        
        results.append({
            "boro_cd": boro_cd,
            "total_complaints": int(total_complaints),
            "cell_count": len(cd_cells),
            "top_1pct_share": top_1pct_share,
            f"top_{top_n}_share": top_n_share,
            "hotspot_count": int(hotspot_count),
            "gini_coefficient": gini,
            "max_cell_count": int(max_cell_count),
            "max_cell_share": max_cell_share,
            "n_repeat_location_dominant": int(n_repeat_location),
            "n_suspected_artifact": int(n_suspected_artifact),
        })
    
    df = pd.DataFrame(results)
    df = ensure_boro_cd_dtype(df)
    
    # Join CD labels
    df = df.merge(
        cd_lookup[["boro_cd", "cd_label", "cd_short"]],
        on="boro_cd",
        how="left",
    )
    
    # Validate labels
    assert_cd_labels_present(df, "CD concentration metrics")
    
    df = df.sort_values("boro_cd").reset_index(drop=True)
    
    return df


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def run_sensitivity_analysis(
    gdf_night_proj: gpd.GeoDataFrame,
    cd59_proj: gpd.GeoDataFrame,
    nyc_bounds: Tuple[float, float, float, float],
    grid_sizes: List[int],
    thresholds: List[int],
    top_n_cells: int,
    logger,
) -> pd.DataFrame:
    """
    Run sensitivity analysis across grid sizes and thresholds.
    
    Returns summary DataFrame with metrics for each combination.
    """
    logger.info("Running sensitivity analysis...")
    logger.info(f"  Grid sizes: {grid_sizes} ft")
    logger.info(f"  Thresholds: {thresholds}")
    
    results = []
    
    for grid_size in grid_sizes:
        logger.info(f"  Processing grid size {grid_size} ft...")
        
        # Create grid
        grid = create_grid(nyc_bounds, grid_size, "EPSG:2263")
        grid = clip_grid_to_boundary(grid, cd59_proj)
        grid = assign_cells_to_cds(grid, cd59_proj)
        
        # Assign points
        _, cell_counts = assign_points_to_grid(gdf_night_proj, grid)
        
        # Merge counts
        grid_stats = grid.merge(cell_counts, on="cell_id", how="left")
        grid_stats["count"] = grid_stats["count"].fillna(0).astype(int)
        
        for threshold in thresholds:
            # Identify hotspots at this threshold
            hotspots = grid_stats[grid_stats["count"] >= threshold]
            n_hotspots = len(hotspots)
            total_complaints = hotspots["count"].sum()
            
            # Top N cells
            top_cells = grid_stats.nlargest(top_n_cells, "count")
            top_n_complaints = top_cells["count"].sum()
            
            # Gini across all cells with complaints
            cells_with_data = grid_stats[grid_stats["count"] > 0]
            mean_gini = 0
            if len(cells_with_data) > 0:
                # Compute per-CD gini and average
                ginis = []
                for boro_cd in cells_with_data["boro_cd"].dropna().unique():
                    cd_cells = cells_with_data[cells_with_data["boro_cd"] == boro_cd]
                    if len(cd_cells) > 1:
                        ginis.append(compute_gini_coefficient(cd_cells["count"].values))
                mean_gini = np.mean(ginis) if ginis else 0
            
            results.append({
                "grid_size_ft": grid_size,
                "grid_size_m": round(grid_size * 0.3048, 0),
                "threshold": threshold,
                "n_hotspot_cells": n_hotspots,
                "total_complaints_in_hotspots": int(total_complaints),
                f"top_{top_n_cells}_complaints": int(top_n_complaints),
                "mean_gini_coefficient": round(mean_gini, 4),
                "total_cells_with_data": len(cells_with_data),
            })
    
    df = pd.DataFrame(results)
    logger.info(f"  Sensitivity analysis complete: {len(df)} configurations")
    
    return df


# =============================================================================
# Main Processing
# =============================================================================

def build_hotspot_layers_with_flags(
    gdf_night_proj: gpd.GeoDataFrame,
    cd59_proj: gpd.GeoDataFrame,
    nyc_bounds: Tuple[float, float, float, float],
    cell_size_ft: int,
    flag_config: Dict,
    latlon_decimals: int,
    cd_lookup: pd.DataFrame,
    thresholds: List[int],
    top_n_cells: int,
    logger,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build hotspot layers with artifact flags.
    
    Returns:
        - grid_stats: Full grid with flags and counts
        - cd_metrics_raw: CD metrics including all cells
        - cd_metrics_clean: CD metrics excluding suspected artifacts
    """
    logger.info(f"Building hotspot layers with cell size {cell_size_ft} ft...")
    
    # Create grid
    grid = create_grid(nyc_bounds, cell_size_ft, "EPSG:2263")
    logger.info(f"  Created grid: {len(grid):,} cells")
    
    grid = clip_grid_to_boundary(grid, cd59_proj)
    logger.info(f"  Clipped to NYC: {len(grid):,} cells")
    
    grid = assign_cells_to_cds(grid, cd59_proj)
    
    # Assign points to cells
    points_with_cells, cell_counts = assign_points_to_grid(gdf_night_proj, grid)
    logger.info(f"  Cells with ≥1 complaint: {len(cell_counts):,}")
    
    # Compute artifact flags for all cells
    artifact_flags = compute_artifact_flags_for_all_cells(
        points_with_cells, flag_config, latlon_decimals, logger
    )
    
    # Merge counts and flags into grid
    grid_stats = grid.merge(cell_counts, on="cell_id", how="left")
    grid_stats["count"] = grid_stats["count"].fillna(0).astype(int)
    
    # Merge artifact flags
    grid_stats = grid_stats.merge(
        artifact_flags[[
            "cell_id", "top_latlon_share", "top_latlon", "top_address_share",
            "max_single_day_count", "is_repeat_location_dominant", "is_suspected_artifact"
        ]],
        on="cell_id",
        how="left",
    )
    
    # Fill NaN flags with False for cells with no complaints
    grid_stats["is_repeat_location_dominant"] = grid_stats["is_repeat_location_dominant"].fillna(False)
    grid_stats["is_suspected_artifact"] = grid_stats["is_suspected_artifact"].fillna(False)
    
    # Add hotspot flags for each threshold
    for threshold in thresholds:
        grid_stats[f"is_hotspot_ge{threshold}"] = grid_stats["count"] >= threshold
    
    # Compute CD metrics (raw and clean)
    logger.info("Computing raw CD concentration metrics...")
    cd_metrics_raw = compute_cd_concentration_metrics(
        grid_stats, cd_lookup, top_n_cells, 
        hotspot_threshold=10, exclude_artifacts=False, logger=logger
    )
    
    logger.info("Computing clean CD concentration metrics (excluding artifacts)...")
    cd_metrics_clean = compute_cd_concentration_metrics(
        grid_stats, cd_lookup, top_n_cells,
        hotspot_threshold=10, exclude_artifacts=True, logger=logger
    )
    
    return grid_stats, cd_metrics_raw, cd_metrics_clean


# =============================================================================
# Validation
# =============================================================================

def validate_outputs(
    grid_stats: gpd.GeoDataFrame,
    cd_metrics_raw: pd.DataFrame,
    cd_metrics_clean: pd.DataFrame,
    logger,
) -> Dict:
    """Validate outputs and return QA stats."""
    logger.info("Validating outputs...")
    
    qa_stats = {"passed": True}
    
    # Check artifact flags are boolean
    if grid_stats["is_repeat_location_dominant"].dtype != bool:
        logger.error("is_repeat_location_dominant is not boolean!")
        qa_stats["passed"] = False
    
    if grid_stats["is_suspected_artifact"].dtype != bool:
        logger.error("is_suspected_artifact is not boolean!")
        qa_stats["passed"] = False
    
    # Check share ranges
    valid_shares = grid_stats["top_latlon_share"].dropna()
    if (valid_shares < 0).any() or (valid_shares > 1).any():
        logger.error("top_latlon_share out of [0, 1] range!")
        qa_stats["passed"] = False
    
    # Check CD count
    qa_stats["cd_count_raw"] = len(cd_metrics_raw)
    qa_stats["cd_count_clean"] = len(cd_metrics_clean)
    
    if len(cd_metrics_raw) != 59:
        logger.warning(f"Expected 59 CDs in raw metrics, got {len(cd_metrics_raw)}")
    
    # Summary stats
    qa_stats["n_repeat_location_dominant"] = int(grid_stats["is_repeat_location_dominant"].sum())
    qa_stats["n_suspected_artifact"] = int(grid_stats["is_suspected_artifact"].sum())
    
    cells_ge10 = grid_stats[grid_stats["count"] >= 10]
    qa_stats["n_hotspots_ge10"] = len(cells_ge10)
    qa_stats["n_hotspots_ge50"] = len(grid_stats[grid_stats["count"] >= 50])
    
    logger.info(f"QA validation {'PASSED' if qa_stats['passed'] else 'FAILED'}")
    
    return qa_stats


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("06c_hotspot_sensitivity") as logger:
        logger.info("Starting 06c_hotspot_sensitivity.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        time_config = config["time_windows"]["primary"]
        night_config = config["nighttime"]["primary"]
        
        year_start = time_config["year_start"]
        year_end = time_config["year_end"]
        night_start = night_config["start_hour"]
        night_end = night_config["end_hour"]
        
        # Investigation config (flags)
        investigation_config = config.get("hotspot_investigation", {})
        flag_config = investigation_config.get("flags", {})
        latlon_decimals = investigation_config.get("latlon_decimals", 5)
        
        # Sensitivity config
        sensitivity_config = config.get("hotspot_sensitivity", {})
        grid_sizes = sensitivity_config.get("grid_sizes_ft", [410, 820, 1640])
        thresholds = sensitivity_config.get("thresholds", [10, 25, 50])
        top_n_cells = sensitivity_config.get("top_n_cells", 100)
        primary_grid_size = sensitivity_config.get("primary_grid_size_ft", 820)
        
        # Hotspot config
        hotspot_config = config.get("hotspots", {})
        cd_top_n = hotspot_config.get("top_n_cells", 10)
        
        logger.info("Configuration:")
        logger.info(f"  Grid sizes: {grid_sizes} ft")
        logger.info(f"  Thresholds: {thresholds}")
        logger.info(f"  Primary grid size: {primary_grid_size} ft")
        logger.info(f"  Flag config: {flag_config}")
        
        try:
            # Load data
            df_raw, raw_path = load_raw_311(logger)
            cd59 = load_cd59(logger)
            cd_lookup = load_cd_lookup(logger)
            
            # Prepare points
            logger.info("Preparing 311 points...")
            gdf_points = prepare_311_points(df_raw, logger)
            gdf_night = filter_to_nighttime_and_years(
                gdf_points, night_start, night_end, year_start, year_end
            )
            logger.info(f"Nighttime records: {len(gdf_night):,}")
            
            # Project to EPSG:2263
            cd59_proj = safe_reproject(cd59, 2263, "CD59")
            gdf_night_proj = safe_reproject(gdf_night, 2263, "311 points")
            
            # Get NYC bounds
            nyc_bounds = cd59_proj.total_bounds
            logger.info(f"NYC bounds (EPSG:2263): {nyc_bounds}")
            
            # Build hotspot layers with flags (primary grid size)
            grid_stats, cd_metrics_raw, cd_metrics_clean = build_hotspot_layers_with_flags(
                gdf_night_proj=gdf_night_proj,
                cd59_proj=cd59_proj,
                nyc_bounds=nyc_bounds,
                cell_size_ft=primary_grid_size,
                flag_config=flag_config,
                latlon_decimals=latlon_decimals,
                cd_lookup=cd_lookup,
                thresholds=thresholds,
                top_n_cells=cd_top_n,
                logger=logger,
            )
            
            # Run sensitivity analysis
            sensitivity_df = run_sensitivity_analysis(
                gdf_night_proj=gdf_night_proj,
                cd59_proj=cd59_proj,
                nyc_bounds=nyc_bounds,
                grid_sizes=grid_sizes,
                thresholds=thresholds,
                top_n_cells=top_n_cells,
                logger=logger,
            )
            
            # Validate
            qa_stats = validate_outputs(grid_stats, cd_metrics_raw, cd_metrics_clean, logger)
            
            # Ensure output directory exists
            HOTSPOTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # =================================================================
            # Write outputs
            # =================================================================
            
            # 1. GeoJSON for analysis-grade (≥10)
            cells_ge10 = grid_stats[grid_stats["count"] >= 10].copy()
            # Convert to WGS84 for GeoJSON
            cells_ge10_wgs = cells_ge10.to_crs("EPSG:4326")
            atomic_write_gdf(cells_ge10_wgs, OUTPUT_GE10)
            logger.info(f"Wrote: {OUTPUT_GE10} ({len(cells_ge10):,} cells)")
            
            # 2. GeoJSON for map-grade (≥50)
            cells_ge50 = grid_stats[grid_stats["count"] >= 50].copy()
            cells_ge50_wgs = cells_ge50.to_crs("EPSG:4326")
            atomic_write_gdf(cells_ge50_wgs, OUTPUT_GE50)
            logger.info(f"Wrote: {OUTPUT_GE50} ({len(cells_ge50):,} cells)")
            
            # 3. Sensitivity summary
            sensitivity_df.to_csv(OUTPUT_SENSITIVITY, index=False)
            logger.info(f"Wrote: {OUTPUT_SENSITIVITY} ({len(sensitivity_df)} rows)")
            
            # 4. CD concentration (raw)
            atomic_write_df(cd_metrics_raw, OUTPUT_CD_CONCENTRATION)
            cd_metrics_raw.to_csv(OUTPUT_CD_CONCENTRATION_CSV, index=False)
            logger.info(f"Wrote: {OUTPUT_CD_CONCENTRATION}")
            logger.info(f"Wrote: {OUTPUT_CD_CONCENTRATION_CSV}")
            
            # 5. CD concentration (clean - excluding artifacts)
            atomic_write_df(cd_metrics_clean, OUTPUT_CD_CONCENTRATION_CLEAN)
            cd_metrics_clean.to_csv(OUTPUT_CD_CONCENTRATION_CLEAN_CSV, index=False)
            logger.info(f"Wrote: {OUTPUT_CD_CONCENTRATION_CLEAN}")
            logger.info(f"Wrote: {OUTPUT_CD_CONCENTRATION_CLEAN_CSV}")
            
            # Log outputs
            logger.log_outputs({
                "hotspot_cells_ge10_geojson": str(OUTPUT_GE10),
                "hotspot_cells_ge50_geojson": str(OUTPUT_GE50),
                "sensitivity_summary_csv": str(OUTPUT_SENSITIVITY),
                "cd_concentration_parquet": str(OUTPUT_CD_CONCENTRATION),
                "cd_concentration_csv": str(OUTPUT_CD_CONCENTRATION_CSV),
                "cd_concentration_clean_parquet": str(OUTPUT_CD_CONCENTRATION_CLEAN),
                "cd_concentration_clean_csv": str(OUTPUT_CD_CONCENTRATION_CLEAN_CSV),
            })
            
            # Log metrics
            logger.log_metrics({
                "primary_grid_size_ft": primary_grid_size,
                "grid_sizes_tested": grid_sizes,
                "thresholds_tested": thresholds,
                "n_hotspots_ge10": len(cells_ge10),
                "n_hotspots_ge50": len(cells_ge50),
                "n_repeat_location_dominant": int(grid_stats["is_repeat_location_dominant"].sum()),
                "n_suspected_artifact": int(grid_stats["is_suspected_artifact"].sum()),
                "qa_stats": qa_stats,
            })
            
            # Write metadata sidecar
            write_metadata_sidecar(
                output_path=OUTPUT_GE10,
                inputs={"raw_311_noise": str(raw_path)},
                config=config,
                run_id=logger.run_id,
                extra={
                    "primary_grid_size_ft": primary_grid_size,
                    "flag_config": flag_config,
                    "thresholds": thresholds,
                    "qa_stats": qa_stats,
                },
            )
            
            # Print summary
            logger.info("=" * 70)
            logger.info("Hotspot Sensitivity Analysis Summary:")
            logger.info(f"  Primary grid size: {primary_grid_size} ft (~{primary_grid_size * 0.3048:.0f} m)")
            logger.info(f"  Grid sizes tested: {grid_sizes}")
            logger.info(f"  Thresholds tested: {thresholds}")
            logger.info("")
            logger.info("Hotspot Layer Outputs:")
            logger.info(f"  hotspot_cells_ge10.geojson: {len(cells_ge10):,} cells (analysis-grade)")
            logger.info(f"  hotspot_cells_ge50.geojson: {len(cells_ge50):,} cells (map-grade)")
            logger.info("")
            logger.info("Artifact Flags:")
            logger.info(f"  Cells with is_repeat_location_dominant: {grid_stats['is_repeat_location_dominant'].sum()}")
            logger.info(f"  Cells with is_suspected_artifact: {grid_stats['is_suspected_artifact'].sum()}")
            logger.info("")
            logger.info("Sensitivity Summary (primary grid):")
            primary_sensitivity = sensitivity_df[sensitivity_df["grid_size_ft"] == primary_grid_size]
            for _, row in primary_sensitivity.iterrows():
                logger.info(
                    f"  Threshold ≥{row['threshold']}: {row['n_hotspot_cells']:,} cells, "
                    f"{row['total_complaints_in_hotspots']:,} complaints"
                )
            logger.info("")
            logger.info("CD Concentration Comparison (raw vs clean):")
            logger.info(f"  Raw total complaints: {cd_metrics_raw['total_complaints'].sum():,}")
            logger.info(f"  Clean total complaints: {cd_metrics_clean['total_complaints'].sum():,}")
            raw_gini = cd_metrics_raw['gini_coefficient'].mean()
            clean_gini = cd_metrics_clean['gini_coefficient'].mean()
            logger.info(f"  Raw mean Gini: {raw_gini:.3f}")
            logger.info(f"  Clean mean Gini: {clean_gini:.3f}")
            logger.info("=" * 70)
            
            logger.info("SUCCESS: Hotspot sensitivity analysis complete")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

