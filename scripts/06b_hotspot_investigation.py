#!/usr/bin/env python3
"""
06b_hotspot_investigation.py

Validate extreme hotspot cells to distinguish:
- Genuine persistent location problems
vs
- Geocoding artifacts / proxy coordinates / repeated caller locations

Per NYC_Night_Signals_Plan.md Section 3.3 (Script 06b):

For top N cells (default N=20), output:
- cell_id
- complaint_count
- unique_latlon_rounded (round lat/lon to 5 or 6 decimals)
- top_latlon_share (share of complaints at most common rounded lat/lon)
- unique_addresses + top_address_share (if address fields exist)
- top complaint types/descriptors
- temporal spread: unique dates, month distribution, year distribution

Outputs:
- data/processed/hotspots/hotspot_investigation_top_cells.csv
- data/processed/metadata/hotspot_investigation_metadata.json

Requirements:
- Deterministic given fixed parameters
- Explicitly report which address/descriptor fields exist
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
from sleep_esi.io_utils import atomic_write_df, read_yaml, read_gdf
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, RAW_DIR, PROCESSED_DIR
from sleep_esi.qa import safe_reproject
from sleep_esi.schemas import ensure_boro_cd_dtype
from sleep_esi.time_utils import ensure_nyc_timezone, filter_nighttime


# =============================================================================
# Constants
# =============================================================================

RAW_311_DIR = RAW_DIR / "311_noise"
HOTSPOTS_DIR = PROCESSED_DIR / "hotspots"

# Outputs
OUTPUT_INVESTIGATION = HOTSPOTS_DIR / "hotspot_investigation_top_cells.csv"

# Default parameters
DEFAULT_TOP_N = 20
DEFAULT_LATLON_DECIMALS = 5


# =============================================================================
# Privacy Utilities
# =============================================================================

import re

def redact_address(address: Optional[str]) -> Optional[str]:
    """
    Redact house numbers and unit designators from an address for privacy.

    NYC address formats handled:
    - "672 EAST 231 STREET" → "EAST 231 STREET"
    - "78-15 PARSONS BOULEVARD" → "PARSONS BOULEVARD" (Queens hyphenated)
    - "2400 7 AVENUE" → "7 AVENUE"
    - "1234A BROADWAY" → "BROADWAY" (with unit letter suffix)
    - "123 BROADWAY APT 3F" → "BROADWAY" (apartment designator)
    - "456 WEST 125TH STREET #2B" → "WEST 125TH STREET" (unit with #)
    - "789 5TH AVENUE UNIT 12" → "5TH AVENUE" (unit designator)

    Args:
        address: Full address string

    Returns:
        Street-level address with house numbers and units removed, or None if empty
    """
    if address is None or address == "UNKNOWN":
        return None

    address = str(address).strip().upper()

    if not address:
        return None

    # Step 1: Remove apartment/unit designators from the end
    # Matches: APT, APARTMENT, UNIT, SUITE, STE, FL, FLOOR, #, followed by unit number
    unit_patterns = [
        r"\s+(APT|APARTMENT|UNIT|SUITE|STE|FL|FLOOR|RM|ROOM)\s*\.?\s*#?\s*\w+\s*$",
        r"\s+#\s*\w+\s*$",  # Just "#3F" at the end
    ]
    for pattern in unit_patterns:
        address = re.sub(pattern, "", address, flags=re.IGNORECASE)

    # Step 2: Remove leading house numbers
    # Pattern: "123", "123-45" (Queens), "1234A" (with unit letter)
    # Must be followed by a space and then the street name
    house_number_pattern = r"^\d+(-\d+)?[A-Z]?\s+"
    redacted = re.sub(house_number_pattern, "", address)

    # Clean up any extra whitespace
    redacted = " ".join(redacted.split())

    return redacted if redacted else None


# =============================================================================
# Data Loading (reuse from Script 06)
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
# Grid Functions (from Script 06)
# =============================================================================

def create_grid(
    bounds: Tuple[float, float, float, float],
    cell_size: float,
    crs: str,
    logger,
) -> gpd.GeoDataFrame:
    """Create a regular square grid covering the given bounds."""
    minx, miny, maxx, maxy = bounds
    
    n_cols = int(np.ceil((maxx - minx) / cell_size))
    n_rows = int(np.ceil((maxy - miny) / cell_size))
    
    logger.info(f"Creating grid: {n_cols} cols × {n_rows} rows = {n_cols * n_rows:,} cells")
    
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
    logger,
) -> gpd.GeoDataFrame:
    """Clip grid to NYC boundary."""
    logger.info("Clipping grid to NYC boundary...")
    
    if grid.crs != boundary.crs:
        boundary = boundary.to_crs(grid.crs)
    
    nyc_boundary = boundary.union_all()
    grid["intersects_nyc"] = grid.geometry.intersects(nyc_boundary)
    grid_clipped = grid[grid["intersects_nyc"]].copy()
    grid_clipped = grid_clipped.drop(columns=["intersects_nyc"])
    
    logger.info(f"Kept {len(grid_clipped):,} cells within NYC")
    
    return grid_clipped


def assign_cells_to_cds(
    grid: gpd.GeoDataFrame,
    cd59: gpd.GeoDataFrame,
    logger,
) -> gpd.GeoDataFrame:
    """Assign each grid cell to a CD based on cell centroid."""
    logger.info("Assigning grid cells to CDs...")
    
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


def assign_points_to_cells(
    points: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    logger,
) -> gpd.GeoDataFrame:
    """Assign points to grid cells via spatial join."""
    logger.info("Assigning points to grid cells...")
    
    if points.crs != grid.crs:
        points_proj = points.to_crs(grid.crs)
    else:
        points_proj = points
    
    # Keep point data + assign cell_id
    joined = gpd.sjoin(
        points_proj, 
        grid[["cell_id", "boro_cd", "geometry"]], 
        how="left", 
        predicate="within"
    )
    
    assigned = joined["cell_id"].notna().sum()
    logger.info(f"Points assigned to cells: {assigned:,} / {len(points):,}")
    
    return joined


# =============================================================================
# Investigation Analysis
# =============================================================================

def detect_available_fields(df: pd.DataFrame, logger) -> Dict[str, bool]:
    """Detect which optional fields exist in the data."""
    fields = {
        "incident_address": "incident_address" in df.columns,
        "street_name": "street_name" in df.columns,
        "descriptor": "descriptor" in df.columns,
        "complaint_type": "complaint_type" in df.columns,
        "borough": "borough" in df.columns,
        "city": "city" in df.columns,
    }
    
    logger.info("Available fields in raw data:")
    for field, exists in fields.items():
        status = "✓ present" if exists else "✗ missing"
        logger.info(f"  {field}: {status}")
    
    return fields


def investigate_cell(
    cell_points: gpd.GeoDataFrame,
    cell_id: int,
    boro_cd: int,
    cd_label: str,
    latlon_decimals: int,
    available_fields: Dict[str, bool],
) -> Dict:
    """
    Compute investigation diagnostics for a single cell.
    """
    n = len(cell_points)
    
    result = {
        "cell_id": int(cell_id),
        "boro_cd": int(boro_cd),
        "cd_label": cd_label,
        "complaint_count": n,
    }
    
    # -------------------------------------------------------------------------
    # Coordinate uniqueness (round to specified decimals)
    # -------------------------------------------------------------------------
    cell_points = cell_points.copy()
    cell_points["lat_rounded"] = cell_points["latitude"].round(latlon_decimals)
    cell_points["lon_rounded"] = cell_points["longitude"].round(latlon_decimals)
    cell_points["latlon_rounded"] = (
        cell_points["lat_rounded"].astype(str) + "," + 
        cell_points["lon_rounded"].astype(str)
    )
    
    unique_latlon = cell_points["latlon_rounded"].nunique()
    result["unique_latlon_rounded"] = int(unique_latlon)
    
    # Top lat/lon share
    latlon_counts = cell_points["latlon_rounded"].value_counts()
    top_latlon_count = latlon_counts.iloc[0] if len(latlon_counts) > 0 else 0
    result["top_latlon_share"] = round(top_latlon_count / n, 4) if n > 0 else 0
    result["top_latlon"] = latlon_counts.index[0] if len(latlon_counts) > 0 else None
    
    # -------------------------------------------------------------------------
    # Address uniqueness (if available)
    # -------------------------------------------------------------------------
    if available_fields.get("incident_address"):
        addresses = cell_points["incident_address"].fillna("UNKNOWN").str.upper().str.strip()
        unique_addresses = addresses.nunique()
        result["unique_addresses"] = int(unique_addresses)
        
        address_counts = addresses.value_counts()
        top_address_count = address_counts.iloc[0] if len(address_counts) > 0 else 0
        result["top_address_share"] = round(top_address_count / n, 4) if n > 0 else 0
        result["top_address"] = address_counts.index[0] if len(address_counts) > 0 else None
        
        # Privacy-safe redacted address (remove house numbers, keep street only)
        result["top_address_redacted"] = redact_address(result["top_address"])
    else:
        result["unique_addresses"] = None
        result["top_address_share"] = None
        result["top_address"] = None
        result["top_address_redacted"] = None
    
    # -------------------------------------------------------------------------
    # Complaint type distribution
    # -------------------------------------------------------------------------
    if available_fields.get("complaint_type"):
        type_counts = cell_points["complaint_type"].fillna("UNKNOWN").value_counts()
        result["top_complaint_type"] = type_counts.index[0] if len(type_counts) > 0 else None
        result["top_complaint_type_share"] = round(type_counts.iloc[0] / n, 4) if n > 0 else 0
        
        # Top 3 types
        top_3_types = type_counts.head(3).to_dict()
        result["complaint_type_top3"] = json.dumps(top_3_types)
    else:
        result["top_complaint_type"] = None
        result["top_complaint_type_share"] = None
        result["complaint_type_top3"] = None
    
    # -------------------------------------------------------------------------
    # Descriptor distribution (sub-type detail)
    # -------------------------------------------------------------------------
    if available_fields.get("descriptor"):
        desc_counts = cell_points["descriptor"].fillna("UNKNOWN").value_counts()
        result["top_descriptor"] = desc_counts.index[0] if len(desc_counts) > 0 else None
        result["top_descriptor_share"] = round(desc_counts.iloc[0] / n, 4) if n > 0 else 0
        
        # Top 3 descriptors
        top_3_desc = desc_counts.head(3).to_dict()
        result["descriptor_top3"] = json.dumps(top_3_desc)
    else:
        result["top_descriptor"] = None
        result["top_descriptor_share"] = None
        result["descriptor_top3"] = None
    
    # -------------------------------------------------------------------------
    # Temporal spread
    # -------------------------------------------------------------------------
    cell_points["date"] = cell_points["ts_nyc"].dt.date
    cell_points["month"] = cell_points["ts_nyc"].dt.month
    cell_points["year"] = cell_points["ts_nyc"].dt.year
    
    result["unique_dates"] = int(cell_points["date"].nunique())
    result["unique_months"] = int(cell_points["month"].nunique())
    result["unique_years"] = int(cell_points["year"].nunique())
    
    # Date range
    date_min = cell_points["date"].min()
    date_max = cell_points["date"].max()
    result["date_min"] = str(date_min)
    result["date_max"] = str(date_max)
    
    # Year distribution
    year_counts = cell_points["year"].value_counts().sort_index().to_dict()
    result["year_distribution"] = json.dumps(year_counts)
    
    # Month distribution (aggregated across years)
    month_counts = cell_points["month"].value_counts().sort_index().to_dict()
    result["month_distribution"] = json.dumps(month_counts)
    
    # -------------------------------------------------------------------------
    # Repeat caller proxy: single-day spikes
    # -------------------------------------------------------------------------
    daily_counts = cell_points.groupby("date").size()
    result["max_single_day_count"] = int(daily_counts.max())
    result["mean_daily_count"] = round(daily_counts.mean(), 2)
    result["median_daily_count"] = round(daily_counts.median(), 2)
    
    # Days with unusually high counts (>3x median)
    median_daily = daily_counts.median()
    if median_daily > 0:
        spike_days = (daily_counts > 3 * median_daily).sum()
    else:
        spike_days = 0
    result["spike_days_count"] = int(spike_days)
    
    return result


def run_investigation(
    points_with_cells: gpd.GeoDataFrame,
    cd_lookup: pd.DataFrame,
    top_n: int,
    latlon_decimals: int,
    available_fields: Dict[str, bool],
    logger,
) -> pd.DataFrame:
    """
    Run investigation on top N hotspot cells.
    """
    logger.info(f"Running investigation on top {top_n} hotspot cells...")
    
    # Get cell counts
    cell_counts = (
        points_with_cells
        .groupby("cell_id")
        .agg(
            count=("cell_id", "size"),
            boro_cd=("boro_cd", "first"),
        )
        .reset_index()
    )
    cell_counts["cell_id"] = cell_counts["cell_id"].astype(int)
    cell_counts = cell_counts.sort_values("count", ascending=False)
    
    # Get top N cells
    top_cells = cell_counts.head(top_n).copy()
    
    logger.info(f"Top {len(top_cells)} cells identified:")
    for _, row in top_cells.head(5).iterrows():
        logger.info(f"  Cell {row['cell_id']}: {row['count']:,} complaints")
    
    # Create lookup for CD labels
    cd_label_map = cd_lookup.set_index("boro_cd")["cd_label"].to_dict()
    
    # Investigate each cell
    results = []
    for _, row in top_cells.iterrows():
        cell_id = row["cell_id"]
        boro_cd = int(row["boro_cd"]) if pd.notna(row["boro_cd"]) else -1
        cd_label = cd_label_map.get(boro_cd, "Unknown")
        
        cell_points = points_with_cells[points_with_cells["cell_id"] == cell_id]
        
        result = investigate_cell(
            cell_points=cell_points,
            cell_id=cell_id,
            boro_cd=boro_cd,
            cd_label=cd_label,
            latlon_decimals=latlon_decimals,
            available_fields=available_fields,
        )
        results.append(result)
        
        logger.info(
            f"Cell {cell_id}: {result['complaint_count']:,} complaints, "
            f"{result['unique_latlon_rounded']} unique coords, "
            f"top latlon share = {result['top_latlon_share']:.1%}"
        )
    
    df = pd.DataFrame(results)
    return df


# =============================================================================
# Validation
# =============================================================================

def validate_investigation(df: pd.DataFrame, logger) -> Dict:
    """Validate investigation outputs."""
    logger.info("Validating investigation outputs...")
    
    qa_stats = {
        "row_count": len(df),
        "total_complaints_investigated": int(df["complaint_count"].sum()),
        "passed": True,
    }
    
    # Check for required columns
    required_cols = [
        "cell_id", "boro_cd", "cd_label", "complaint_count",
        "unique_latlon_rounded", "top_latlon_share",
        "unique_dates", "unique_years",
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        qa_stats["passed"] = False
    
    # Check share ranges
    if "top_latlon_share" in df.columns:
        if (df["top_latlon_share"] < 0).any() or (df["top_latlon_share"] > 1).any():
            logger.error("top_latlon_share out of [0, 1] range!")
            qa_stats["passed"] = False
    
    # Log key findings
    qa_stats["cells_with_dominant_coord"] = int((df["top_latlon_share"] > 0.5).sum())
    qa_stats["cells_with_single_address"] = 0
    if "top_address_share" in df.columns and df["top_address_share"].notna().any():
        qa_stats["cells_with_dominant_address"] = int((df["top_address_share"] > 0.5).sum())
    
    logger.info(f"QA validation {'PASSED' if qa_stats['passed'] else 'FAILED'}")
    
    return qa_stats


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("06b_hotspot_investigation") as logger:
        logger.info("Starting 06b_hotspot_investigation.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        time_config = config["time_windows"]["primary"]
        night_config = config["nighttime"]["primary"]
        
        year_start = time_config["year_start"]
        year_end = time_config["year_end"]
        night_start = night_config["start_hour"]
        night_end = night_config["end_hour"]
        
        # Investigation-specific config
        investigation_config = config.get("hotspot_investigation", {})
        top_n = investigation_config.get("top_n_cells", DEFAULT_TOP_N)
        latlon_decimals = investigation_config.get("latlon_decimals", DEFAULT_LATLON_DECIMALS)
        
        # Also use hotspot config for grid params
        hotspot_config = config.get("hotspots", {})
        cell_size_ft = hotspot_config.get("cell_size_ft", 820)
        
        logger.info(f"Investigation parameters:")
        logger.info(f"  Top N cells: {top_n}")
        logger.info(f"  Lat/lon decimals: {latlon_decimals}")
        logger.info(f"  Cell size: {cell_size_ft} ft")
        
        try:
            # Load data
            df_raw, raw_path = load_raw_311(logger)
            cd59 = load_cd59(logger)
            cd_lookup = load_cd_lookup(logger)
            
            # Detect available fields
            available_fields = detect_available_fields(df_raw, logger)
            
            # Prepare points
            gdf_points = prepare_311_points(df_raw, logger)
            gdf_night = filter_to_nighttime_and_years(
                gdf_points, night_start, night_end, year_start, year_end, logger
            )
            
            # Project to EPSG:2263
            cd59_proj = safe_reproject(cd59, 2263, "CD59")
            gdf_night_proj = safe_reproject(gdf_night, 2263, "311 points")
            
            # Get NYC bounds and create grid
            nyc_bounds = cd59_proj.total_bounds
            logger.info(f"NYC bounds (EPSG:2263): {nyc_bounds}")
            
            grid = create_grid(nyc_bounds, cell_size_ft, "EPSG:2263", logger)
            grid = clip_grid_to_boundary(grid, cd59_proj, logger)
            grid = assign_cells_to_cds(grid, cd59_proj, logger)
            
            # Assign points to cells
            points_with_cells = assign_points_to_cells(gdf_night_proj, grid, logger)
            
            # Filter to points assigned to cells
            points_assigned = points_with_cells[points_with_cells["cell_id"].notna()].copy()
            points_assigned["cell_id"] = points_assigned["cell_id"].astype(int)
            logger.info(f"Points assigned to cells: {len(points_assigned):,}")
            
            # Run investigation
            df_investigation = run_investigation(
                points_with_cells=points_assigned,
                cd_lookup=cd_lookup,
                top_n=top_n,
                latlon_decimals=latlon_decimals,
                available_fields=available_fields,
                logger=logger,
            )
            
            # Validate
            qa_stats = validate_investigation(df_investigation, logger)
            
            # Ensure output directory exists
            HOTSPOTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Write output
            df_investigation.to_csv(OUTPUT_INVESTIGATION, index=False)
            logger.info(f"Wrote: {OUTPUT_INVESTIGATION} ({len(df_investigation)} rows)")
            
            # Log outputs
            logger.log_outputs({
                "investigation_csv": str(OUTPUT_INVESTIGATION),
            })
            
            # Log metrics
            logger.log_metrics({
                "top_n_cells": top_n,
                "latlon_decimals": latlon_decimals,
                "total_complaints_investigated": int(df_investigation["complaint_count"].sum()),
                "cells_with_dominant_coord": int((df_investigation["top_latlon_share"] > 0.5).sum()),
                "available_fields": available_fields,
                "qa_stats": qa_stats,
            })
            
            # Write metadata sidecar
            write_metadata_sidecar(
                output_path=OUTPUT_INVESTIGATION,
                inputs={"raw_311_noise": str(raw_path)},
                config=config,
                run_id=logger.run_id,
                extra={
                    "top_n_cells": top_n,
                    "latlon_decimals": latlon_decimals,
                    "cell_size_ft": cell_size_ft,
                    "available_fields": available_fields,
                    "qa_stats": qa_stats,
                },
            )
            
            # Print summary
            logger.info("=" * 70)
            logger.info("Hotspot Investigation Summary:")
            logger.info(f"  Cells investigated: {len(df_investigation)}")
            logger.info(f"  Total complaints in top cells: {df_investigation['complaint_count'].sum():,}")
            logger.info("")
            
            # Key findings
            logger.info("Key Findings:")
            
            # Cells with dominant coordinates (potential geocoding artifacts)
            dominant_coord = df_investigation[df_investigation["top_latlon_share"] > 0.5]
            if len(dominant_coord) > 0:
                logger.info(f"  Cells with >50% at single coordinate: {len(dominant_coord)}")
                for _, row in dominant_coord.iterrows():
                    logger.info(
                        f"    Cell {row['cell_id']} ({row['cd_label']}): "
                        f"{row['top_latlon_share']:.1%} at {row['top_latlon']}"
                    )
            else:
                logger.info("  No cells with >50% at single coordinate (good sign)")
            
            # Top cell details
            top_cell = df_investigation.iloc[0]
            logger.info("")
            logger.info(f"Top Hotspot Cell Details (Cell {top_cell['cell_id']}):")
            logger.info(f"  Location: {top_cell['cd_label']}")
            logger.info(f"  Complaints: {top_cell['complaint_count']:,}")
            logger.info(f"  Unique coordinates: {top_cell['unique_latlon_rounded']}")
            logger.info(f"  Top coordinate share: {top_cell['top_latlon_share']:.1%}")
            if top_cell.get("unique_addresses"):
                logger.info(f"  Unique addresses: {top_cell['unique_addresses']}")
                logger.info(f"  Top address share: {top_cell['top_address_share']:.1%}")
            logger.info(f"  Date range: {top_cell['date_min']} to {top_cell['date_max']}")
            logger.info(f"  Unique dates: {top_cell['unique_dates']}")
            logger.info(f"  Max single-day count: {top_cell['max_single_day_count']}")
            
            logger.info("=" * 70)
            
            logger.info("SUCCESS: Hotspot investigation complete")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

