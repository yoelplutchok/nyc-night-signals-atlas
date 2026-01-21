#!/usr/bin/env python3
"""
06_build_hotspots.py

Build Hotspot Concentration analysis for nighttime 311 noise complaints.

Per NYC_Night_Signals_Plan.md Section 3.3.6:
- Create a grid (EPSG:2263) with fixed cell size (default 250m)
- Assign nighttime 311 points to grid cells
- Compute per-cell counts and identify hotspot cells
- Compute per-CD concentration metrics:
  - top_1pct_share: Share of CD's complaints in top 1% cells
  - top_10_share: Share of CD's complaints in top 10 cells
  - hotspot_count: Number of cells with ≥N complaints
  - gini_coefficient: Spatial concentration measure

Outputs:
- data/processed/hotspots/hotspot_cells.parquet (grid cells with counts)
- data/processed/hotspots/hotspot_cells.geojson (map-ready)
- data/processed/hotspots/cd_hotspot_concentration.parquet (per-CD metrics)
- data/processed/hotspots/cd_hotspot_concentration.csv (human-readable)
- data/processed/metadata/hotspot_cells_metadata.json (provenance sidecar)

Requirements:
- Fixed random seed for any stochastic operations
- Log all parameters
- Deterministic grid generation
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
ATLAS_DIR = PROCESSED_DIR / "atlas"
HOTSPOTS_DIR = PROCESSED_DIR / "hotspots"

# Outputs
OUTPUT_CELLS = HOTSPOTS_DIR / "hotspot_cells.parquet"
OUTPUT_CELLS_GEOJSON = HOTSPOTS_DIR / "hotspot_cells.geojson"
OUTPUT_CD_CONCENTRATION = HOTSPOTS_DIR / "cd_hotspot_concentration.parquet"
OUTPUT_CD_CONCENTRATION_CSV = HOTSPOTS_DIR / "cd_hotspot_concentration.csv"
OUTPUT_TOP_HOTSPOTS = HOTSPOTS_DIR / "top_hotspots_citywide.csv"


# =============================================================================
# Data Loading (reuse from Script 03)
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
    
    df = pd.read_csv(raw_path)
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


# =============================================================================
# Grid Generation
# =============================================================================

def create_grid(
    bounds: Tuple[float, float, float, float],
    cell_size: float,
    crs: str,
    logger,
) -> gpd.GeoDataFrame:
    """
    Create a regular square grid covering the given bounds.
    
    Args:
        bounds: (minx, miny, maxx, maxy) in projected coordinates
        cell_size: Cell size in the same units as bounds (feet for EPSG:2263)
        crs: CRS string (e.g., "EPSG:2263")
        logger: Logger instance
    
    Returns:
        GeoDataFrame with grid cells, each having a unique cell_id
    """
    minx, miny, maxx, maxy = bounds
    
    # Calculate grid dimensions
    n_cols = int(np.ceil((maxx - minx) / cell_size))
    n_rows = int(np.ceil((maxy - miny) / cell_size))
    
    logger.info(f"Creating grid: {n_cols} cols × {n_rows} rows = {n_cols * n_rows:,} cells")
    logger.info(f"Cell size: {cell_size} feet ({cell_size * 0.3048:.1f} meters)")
    
    # Generate grid cells
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
    logger.info(f"Created {len(gdf):,} grid cells")
    
    return gdf


def clip_grid_to_boundary(
    grid: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    logger,
) -> gpd.GeoDataFrame:
    """
    Clip grid to NYC boundary (union of all CDs).
    Keep only cells that intersect the boundary.
    """
    logger.info("Clipping grid to NYC boundary...")
    
    # Ensure same CRS
    if grid.crs != boundary.crs:
        boundary = boundary.to_crs(grid.crs)
    
    # Create NYC boundary
    nyc_boundary = boundary.union_all()
    
    # Filter to cells that intersect NYC
    grid["intersects_nyc"] = grid.geometry.intersects(nyc_boundary)
    grid_clipped = grid[grid["intersects_nyc"]].copy()
    grid_clipped = grid_clipped.drop(columns=["intersects_nyc"])
    
    logger.info(f"Kept {len(grid_clipped):,} cells within NYC (dropped {len(grid) - len(grid_clipped):,})")
    
    return grid_clipped


# =============================================================================
# Point Processing
# =============================================================================

def prepare_311_points(df: pd.DataFrame, logger) -> gpd.GeoDataFrame:
    """Prepare 311 data as GeoDataFrame with valid coordinates."""
    logger.info("Preparing 311 points...")
    
    # Filter to records with valid coordinates
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    df_valid = df[has_coords].copy()
    logger.info(f"Records with coordinates: {len(df_valid):,} / {len(df):,}")
    
    # Parse timestamps
    df_valid["created_date"] = pd.to_datetime(df_valid["created_date"], errors="coerce")
    valid_ts = df_valid["created_date"].notna()
    df_valid = df_valid[valid_ts].copy()
    logger.info(f"Records with valid timestamps: {len(df_valid):,}")
    
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
    logger,
) -> gpd.GeoDataFrame:
    """Filter to nighttime hours and date range."""
    logger.info(f"Filtering to {year_start}-{year_end}, nighttime {start_hour:02d}:00-{end_hour:02d}:00...")
    
    gdf = gdf.copy()
    gdf["year"] = gdf["ts_nyc"].dt.year
    gdf = gdf[(gdf["year"] >= year_start) & (gdf["year"] <= year_end)]
    logger.info(f"Records in year range: {len(gdf):,}")
    
    gdf = filter_nighttime(gdf, "ts_nyc", start_hour=start_hour, end_hour=end_hour)
    logger.info(f"Nighttime records: {len(gdf):,}")
    
    return gdf


# =============================================================================
# Spatial Join to Grid
# =============================================================================

def assign_points_to_grid(
    points: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    logger,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Assign points to grid cells via spatial join.
    
    Returns:
        - points_with_cell: Points with cell_id assigned
        - cell_counts: DataFrame with count per cell
    """
    logger.info("Assigning points to grid cells...")
    
    # Ensure same CRS
    if points.crs != grid.crs:
        points_proj = points.to_crs(grid.crs)
    else:
        points_proj = points
    
    # Spatial join
    joined = gpd.sjoin(points_proj, grid[["cell_id", "geometry"]], how="left", predicate="within")
    
    # Count assignments
    assigned = joined["cell_id"].notna().sum()
    logger.info(f"Points assigned to cells: {assigned:,} / {len(points):,}")
    
    # Aggregate counts per cell
    cell_counts = joined.groupby("cell_id").size().reset_index(name="count")
    cell_counts["cell_id"] = cell_counts["cell_id"].astype(int)
    
    logger.info(f"Cells with ≥1 complaint: {len(cell_counts):,}")
    
    return joined, cell_counts


def assign_cells_to_cds(
    grid: gpd.GeoDataFrame,
    cd59: gpd.GeoDataFrame,
    logger,
) -> gpd.GeoDataFrame:
    """
    Assign each grid cell to a CD based on cell centroid.
    """
    logger.info("Assigning grid cells to CDs...")
    
    # Ensure same CRS
    if grid.crs != cd59.crs:
        cd59_proj = cd59.to_crs(grid.crs)
    else:
        cd59_proj = cd59
    
    # Get cell centroids
    grid_centroids = grid.copy()
    grid_centroids["geometry"] = grid_centroids.geometry.centroid
    
    # Spatial join centroids to CDs
    joined = gpd.sjoin(
        grid_centroids,
        cd59_proj[["boro_cd", "geometry"]],
        how="left",
        predicate="within",
    )
    
    # Transfer boro_cd to original grid
    grid_with_cd = grid.merge(
        joined[["cell_id", "boro_cd"]],
        on="cell_id",
        how="left",
    )
    
    assigned = grid_with_cd["boro_cd"].notna().sum()
    logger.info(f"Cells assigned to CDs: {assigned:,} / {len(grid):,}")
    
    return grid_with_cd


# =============================================================================
# Concentration Metrics
# =============================================================================

def compute_cell_statistics(
    grid: gpd.GeoDataFrame,
    cell_counts: pd.DataFrame,
    hotspot_threshold: int,
    logger,
) -> gpd.GeoDataFrame:
    """
    Merge counts into grid and compute cell-level statistics.
    """
    logger.info("Computing cell statistics...")
    
    # Merge counts
    grid_stats = grid.merge(cell_counts, on="cell_id", how="left")
    grid_stats["count"] = grid_stats["count"].fillna(0).astype(int)
    
    # Identify hotspots
    grid_stats["is_hotspot"] = grid_stats["count"] >= hotspot_threshold
    
    # Compute percentile rank
    grid_stats["count_percentile"] = grid_stats["count"].rank(pct=True)
    
    # Top 1% flag
    grid_stats["is_top_1pct"] = grid_stats["count_percentile"] >= 0.99
    
    n_hotspots = grid_stats["is_hotspot"].sum()
    n_top_1pct = grid_stats["is_top_1pct"].sum()
    total_complaints = grid_stats["count"].sum()
    
    logger.info(f"Hotspot cells (≥{hotspot_threshold}): {n_hotspots:,}")
    logger.info(f"Top 1% cells: {n_top_1pct:,}")
    logger.info(f"Total complaints in grid: {total_complaints:,}")
    
    return grid_stats


def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for spatial concentration.
    0 = perfectly equal, 1 = perfectly concentrated
    """
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
    logger,
) -> pd.DataFrame:
    """
    Compute concentration metrics for each CD.
    """
    logger.info("Computing per-CD concentration metrics...")
    
    # Filter to cells with counts and assigned to CDs
    cells_with_counts = grid_stats[
        (grid_stats["count"] > 0) & (grid_stats["boro_cd"].notna())
    ].copy()
    cells_with_counts["boro_cd"] = cells_with_counts["boro_cd"].astype(int)
    
    results = []
    
    for boro_cd in sorted(cells_with_counts["boro_cd"].unique()):
        cd_cells = cells_with_counts[cells_with_counts["boro_cd"] == boro_cd]
        
        if len(cd_cells) == 0:
            continue
        
        total_complaints = cd_cells["count"].sum()
        
        # Top 1% share (share of complaints in top 1% cells citywide)
        top_1pct_complaints = cd_cells[cd_cells["is_top_1pct"]]["count"].sum()
        top_1pct_share = top_1pct_complaints / total_complaints if total_complaints > 0 else 0
        
        # Top N cells share (within CD)
        top_n_cells = cd_cells.nlargest(top_n, "count")
        top_n_complaints = top_n_cells["count"].sum()
        top_n_share = top_n_complaints / total_complaints if total_complaints > 0 else 0
        
        # Hotspot count
        hotspot_count = cd_cells["is_hotspot"].sum()
        
        # Gini coefficient for CD
        gini = compute_gini_coefficient(cd_cells["count"].values)
        
        # Max single cell share
        max_cell_count = cd_cells["count"].max()
        max_cell_share = max_cell_count / total_complaints if total_complaints > 0 else 0
        
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
    
    logger.info(f"Computed metrics for {len(df)} CDs")
    logger.info(f"Mean Gini coefficient: {df['gini_coefficient'].mean():.3f}")
    logger.info(f"Mean top-1% share: {df['top_1pct_share'].mean():.3f}")
    
    return df


def get_top_hotspots_citywide(
    grid_stats: gpd.GeoDataFrame,
    cd_lookup: pd.DataFrame,
    top_n: int,
    logger,
) -> pd.DataFrame:
    """
    Get the top N hotspot cells citywide.
    """
    logger.info(f"Identifying top {top_n} hotspots citywide...")
    
    top_cells = grid_stats.nlargest(top_n, "count")[
        ["cell_id", "boro_cd", "count", "row", "col"]
    ].copy()
    
    # Add CD labels
    top_cells["boro_cd"] = top_cells["boro_cd"].fillna(-1).astype(int)
    top_cells = top_cells.merge(
        cd_lookup[["boro_cd", "cd_label", "cd_short"]],
        on="boro_cd",
        how="left",
    )
    
    # Add centroid coordinates for reference
    centroids = grid_stats.set_index("cell_id").loc[top_cells["cell_id"]].geometry.centroid
    top_cells["centroid_x"] = centroids.x.values
    top_cells["centroid_y"] = centroids.y.values
    
    logger.info(f"Top hotspot: cell {top_cells.iloc[0]['cell_id']} with {top_cells.iloc[0]['count']:,} complaints")
    
    return top_cells


# =============================================================================
# Validation
# =============================================================================

def validate_outputs(
    grid_stats: gpd.GeoDataFrame,
    cd_metrics: pd.DataFrame,
    logger,
) -> Dict:
    """
    Validate outputs and return QA stats.
    """
    logger.info("Validating outputs...")
    
    qa_stats = {}
    passed = True
    
    # Grid stats
    qa_stats["total_cells"] = len(grid_stats)
    qa_stats["cells_with_complaints"] = int((grid_stats["count"] > 0).sum())
    qa_stats["total_complaints_in_grid"] = int(grid_stats["count"].sum())
    qa_stats["max_cell_count"] = int(grid_stats["count"].max())
    
    # CD metrics
    qa_stats["cd_count"] = len(cd_metrics)
    if len(cd_metrics) != 59:
        logger.warning(f"Expected 59 CDs, got {len(cd_metrics)}")
    
    # Check for null labels
    null_labels = cd_metrics["cd_label"].isna().sum()
    if null_labels > 0:
        logger.error(f"Found {null_labels} null cd_label values!")
        passed = False
    
    # Check Gini range
    if (cd_metrics["gini_coefficient"] < 0).any() or (cd_metrics["gini_coefficient"] > 1).any():
        logger.error("Gini coefficient out of [0, 1] range!")
        passed = False
    
    # Check share ranges
    for col in ["top_1pct_share", "max_cell_share"]:
        if col in cd_metrics.columns:
            if (cd_metrics[col] < 0).any() or (cd_metrics[col] > 1).any():
                logger.error(f"{col} out of [0, 1] range!")
                passed = False
    
    qa_stats["passed"] = passed
    logger.info(f"QA validation {'PASSED' if passed else 'FAILED'}")
    
    return qa_stats


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("06_build_hotspots") as logger:
        logger.info("Starting 06_build_hotspots.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        time_config = config["time_windows"]["primary"]
        night_config = config["nighttime"]["primary"]
        
        year_start = time_config["year_start"]
        year_end = time_config["year_end"]
        night_start = night_config["start_hour"]
        night_end = night_config["end_hour"]
        
        # Hotspot-specific config
        hotspot_config = config.get("hotspots", {})
        cell_size_ft = hotspot_config.get("cell_size_ft", 820)  # ~250m in feet
        hotspot_threshold = hotspot_config.get("hotspot_threshold", 10)
        top_n_cells = hotspot_config.get("top_n_cells", 10)
        top_n_citywide = hotspot_config.get("top_n_citywide", 100)
        
        logger.info(f"Building hotspot analysis for {year_start}-{year_end}")
        logger.info(f"Cell size: {cell_size_ft} ft (~{cell_size_ft * 0.3048:.0f} m)")
        logger.info(f"Hotspot threshold: ≥{hotspot_threshold} complaints")
        
        try:
            # Load data
            df_raw, raw_path = load_raw_311(logger)
            cd59 = load_cd59(logger)
            cd_lookup = load_cd_lookup(logger)
            
            # Prepare points
            gdf_points = prepare_311_points(df_raw, logger)
            gdf_night = filter_to_nighttime_and_years(
                gdf_points, night_start, night_end, year_start, year_end, logger
            )
            
            # Project to EPSG:2263
            cd59_proj = safe_reproject(cd59, 2263, "CD59")
            gdf_night_proj = safe_reproject(gdf_night, 2263, "311 points")
            
            # Get NYC bounds
            nyc_bounds = cd59_proj.total_bounds
            logger.info(f"NYC bounds (EPSG:2263): {nyc_bounds}")
            
            # Create grid
            grid = create_grid(nyc_bounds, cell_size_ft, "EPSG:2263", logger)
            grid = clip_grid_to_boundary(grid, cd59_proj, logger)
            
            # Assign cells to CDs
            grid = assign_cells_to_cds(grid, cd59_proj, logger)
            
            # Assign points to grid cells
            _, cell_counts = assign_points_to_grid(gdf_night_proj, grid, logger)
            
            # Compute cell statistics
            grid_stats = compute_cell_statistics(grid, cell_counts, hotspot_threshold, logger)
            
            # Compute CD concentration metrics
            cd_metrics = compute_cd_concentration_metrics(
                grid_stats, cd_lookup, top_n_cells, logger
            )
            
            # Get top hotspots citywide
            top_hotspots = get_top_hotspots_citywide(
                grid_stats, cd_lookup, top_n_citywide, logger
            )
            
            # Validate
            qa_stats = validate_outputs(grid_stats, cd_metrics, logger)
            
            # Ensure output directory exists
            HOTSPOTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Write outputs
            # 1. Grid cells (parquet)
            # Only save cells with complaints to reduce file size
            cells_with_data = grid_stats[grid_stats["count"] > 0].copy()
            atomic_write_df(cells_with_data.drop(columns=["geometry"]), OUTPUT_CELLS)
            logger.info(f"Wrote: {OUTPUT_CELLS} ({len(cells_with_data):,} cells)")
            
            # 2. Grid cells (GeoJSON) - only hotspots for map display
            hotspot_cells = grid_stats[grid_stats["is_hotspot"]].copy()
            if len(hotspot_cells) > 0:
                atomic_write_gdf(hotspot_cells, OUTPUT_CELLS_GEOJSON)
                logger.info(f"Wrote: {OUTPUT_CELLS_GEOJSON} ({len(hotspot_cells):,} hotspot cells)")
            else:
                logger.warning("No hotspot cells to write to GeoJSON")
            
            # 3. CD concentration metrics (parquet)
            atomic_write_df(cd_metrics, OUTPUT_CD_CONCENTRATION)
            logger.info(f"Wrote: {OUTPUT_CD_CONCENTRATION}")
            
            # 4. CD concentration metrics (CSV)
            cd_metrics.to_csv(OUTPUT_CD_CONCENTRATION_CSV, index=False)
            logger.info(f"Wrote: {OUTPUT_CD_CONCENTRATION_CSV}")
            
            # 5. Top hotspots citywide (CSV)
            top_hotspots.to_csv(OUTPUT_TOP_HOTSPOTS, index=False)
            logger.info(f"Wrote: {OUTPUT_TOP_HOTSPOTS}")
            
            # Log outputs
            logger.log_outputs({
                "hotspot_cells_parquet": str(OUTPUT_CELLS),
                "hotspot_cells_geojson": str(OUTPUT_CELLS_GEOJSON),
                "cd_concentration_parquet": str(OUTPUT_CD_CONCENTRATION),
                "cd_concentration_csv": str(OUTPUT_CD_CONCENTRATION_CSV),
                "top_hotspots_citywide": str(OUTPUT_TOP_HOTSPOTS),
            })
            
            # Log metrics
            logger.log_metrics({
                "cell_size_ft": cell_size_ft,
                "hotspot_threshold": hotspot_threshold,
                "total_grid_cells": len(grid),
                "cells_with_complaints": int((grid_stats["count"] > 0).sum()),
                "hotspot_count": int(grid_stats["is_hotspot"].sum()),
                "total_complaints": int(grid_stats["count"].sum()),
                "mean_gini": float(cd_metrics["gini_coefficient"].mean()),
                "mean_top_1pct_share": float(cd_metrics["top_1pct_share"].mean()),
                "qa_stats": qa_stats,
            })
            
            # Write metadata sidecar
            write_metadata_sidecar(
                output_path=OUTPUT_CELLS,
                inputs={"raw_311_noise": str(raw_path)},
                config=config,
                run_id=logger.run_id,
                extra={
                    "cell_size_ft": cell_size_ft,
                    "cell_size_m": cell_size_ft * 0.3048,
                    "hotspot_threshold": hotspot_threshold,
                    "total_cells": len(grid),
                    "cells_with_complaints": int((grid_stats["count"] > 0).sum()),
                    "hotspot_count": int(grid_stats["is_hotspot"].sum()),
                    "qa_stats": qa_stats,
                },
            )
            
            # Print summary
            logger.info("=" * 70)
            logger.info("Hotspot Concentration Summary:")
            logger.info(f"  Grid cells: {len(grid):,}")
            logger.info(f"  Cells with complaints: {(grid_stats['count'] > 0).sum():,}")
            logger.info(f"  Hotspot cells (≥{hotspot_threshold}): {grid_stats['is_hotspot'].sum():,}")
            logger.info(f"  Total complaints: {grid_stats['count'].sum():,}")
            logger.info("")
            logger.info("Top 5 CDs by Gini coefficient (most concentrated):")
            top_gini = cd_metrics.nlargest(5, "gini_coefficient")
            for _, row in top_gini.iterrows():
                logger.info(f"  {row['cd_short']}: Gini={row['gini_coefficient']:.3f}")
            logger.info("")
            logger.info("Top 5 CDs by top-1% share:")
            top_share = cd_metrics.nlargest(5, "top_1pct_share")
            for _, row in top_share.iterrows():
                logger.info(f"  {row['cd_short']}: {row['top_1pct_share']:.1%}")
            logger.info("=" * 70)
            
            logger.info("SUCCESS: Built hotspot concentration analysis")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

