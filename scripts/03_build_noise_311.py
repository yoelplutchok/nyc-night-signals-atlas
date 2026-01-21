#!/usr/bin/env python3
"""
03_build_noise_311.py

Build Reported Disturbance Index (RDI) from 311 noise complaints.

Per Section 4.1.1:
- Filter to nighttime hours (22:00-07:00 primary, 23:00-06:00 sensitivity)
- Spatial join to Community Districts
- Compute metrics per CD: count, rate per 1k pop, rate per km2
- Standardize: robust z (headline), classic z, percentile rank

Outputs:
- data/processed/domains/noise_311_cd.parquet

Per R7: Timezone-aware, DST-safe nighttime filtering
Per R6: Hardened spatial joins
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sleep_esi.hashing import write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_df, read_yaml, read_gdf, read_df
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, XWALK_DIR, DOMAINS_DIR, RAW_DIR
from sleep_esi.qa import safe_reproject
from sleep_esi.schemas import ensure_boro_cd_dtype, validate_boro_cd
from sleep_esi.time_utils import (
    ensure_nyc_timezone,
    filter_nighttime,
    assert_temporal_coverage,
    get_nighttime_window,
)
from sleep_esi.joins import (
    spatial_join_points_to_polygons,
    aggregate_points_to_polygons,
    log_join_stats,
)

# =============================================================================
# Constants
# =============================================================================

# Raw data file pattern
RAW_311_DIR = RAW_DIR / "311_noise"

# Output path
OUTPUT_PATH = DOMAINS_DIR / "noise_311_cd.parquet"


# =============================================================================
# Standardization Functions (Section 5.1)
# =============================================================================

def robust_z_score(series: pd.Series) -> pd.Series:
    """
    Compute robust z-score using median and MAD.
    
    Per Section 5.1: Headline standardization.
    
    z_robust = (x - median) / MAD
    where MAD = median absolute deviation * 1.4826 (for normal consistency)
    """
    median = series.median()
    mad = (series - median).abs().median() * 1.4826
    
    if mad == 0 or pd.isna(mad):
        # All values are the same
        return pd.Series(0.0, index=series.index)
    
    return (series - median) / mad


def classic_z_score(series: pd.Series) -> pd.Series:
    """
    Compute classic z-score using mean and standard deviation.
    
    z = (x - mean) / std
    """
    mean = series.mean()
    std = series.std()
    
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    
    return (series - mean) / std


def percentile_rank(series: pd.Series) -> pd.Series:
    """
    Compute percentile rank (0-100).
    
    Higher values = higher percentile.
    """
    return series.rank(pct=True) * 100


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_311(logger) -> pd.DataFrame:
    """
    Load the most recent raw 311 noise data.
    """
    # Find most recent raw file
    raw_files = list(RAW_311_DIR.glob("raw_311_noise_*.csv"))
    
    if not raw_files:
        raise FileNotFoundError(
            f"No raw 311 noise files found in {RAW_311_DIR}. "
            "Run 02_fetch_311_noise.py first."
        )
    
    # Use most recent
    raw_path = max(raw_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading raw 311 data from: {raw_path}")
    
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df):,} raw records")
    
    return df


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


def load_cd_population(logger) -> pd.DataFrame:
    """
    Load CD population from tract crosswalk.
    
    Aggregates tract population to CD level.
    """
    xwalk_path = XWALK_DIR / "cd_to_tract_weights.parquet"
    
    if not xwalk_path.exists():
        raise FileNotFoundError(
            f"Crosswalk not found: {xwalk_path}. "
            "Run 01_build_crosswalks.py first."
        )
    
    xwalk = pd.read_parquet(xwalk_path)
    
    # Sum population per CD
    cd_pop = xwalk.groupby("boro_cd")["tract_pop"].sum().reset_index()
    cd_pop = cd_pop.rename(columns={"tract_pop": "population"})
    cd_pop = ensure_boro_cd_dtype(cd_pop)
    
    logger.info(f"Loaded population for {len(cd_pop)} CDs")
    logger.info(f"Total population: {cd_pop['population'].sum():,}")
    
    return cd_pop


# =============================================================================
# Processing
# =============================================================================

def prepare_311_points(df: pd.DataFrame, logger) -> gpd.GeoDataFrame:
    """
    Prepare 311 data as GeoDataFrame with valid coordinates.
    """
    logger.info("Preparing 311 points...")
    
    # Filter to records with valid coordinates
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    df_valid = df[has_coords].copy()
    
    logger.info(f"Records with coordinates: {len(df_valid):,} / {len(df):,}")
    
    # Create geometry
    gdf = gpd.GeoDataFrame(
        df_valid,
        geometry=gpd.points_from_xy(df_valid["longitude"], df_valid["latitude"]),
        crs="EPSG:4326",
    )
    
    return gdf


def filter_to_nighttime(
    gdf: gpd.GeoDataFrame,
    timestamp_col: str,
    start_hour: int,
    end_hour: int,
    logger,
) -> gpd.GeoDataFrame:
    """
    Filter GeoDataFrame to nighttime hours only.
    
    Per R7: Timezone-aware, DST-safe filtering.
    """
    logger.info(f"Filtering to nighttime hours ({start_hour:02d}:00 - {end_hour:02d}:00)...")
    
    # Parse timestamps and convert to NYC timezone
    gdf = gdf.copy()
    gdf[timestamp_col] = pd.to_datetime(gdf[timestamp_col], errors="coerce")
    
    # Filter out invalid timestamps
    valid_ts = gdf[timestamp_col].notna()
    gdf = gdf[valid_ts].copy()
    
    # Convert to NYC timezone
    gdf["_ts_nyc"] = ensure_nyc_timezone(gdf[timestamp_col])
    
    # Filter to nighttime
    gdf_night = filter_nighttime(
        gdf,
        "_ts_nyc",
        start_hour=start_hour,
        end_hour=end_hour,
        ensure_timezone=False,  # Already converted
    )
    
    # Clean up temp column
    gdf_night = gdf_night.drop(columns=["_ts_nyc"])
    
    logger.info(f"Nighttime records: {len(gdf_night):,} / {len(gdf):,} ({100*len(gdf_night)/len(gdf):.1f}%)")
    
    return gdf_night


def compute_cd_metrics(
    noise_counts: pd.DataFrame,
    cd_pop: pd.DataFrame,
    cd59: gpd.GeoDataFrame,
    logger,
) -> pd.DataFrame:
    """
    Compute noise complaint metrics per CD.
    
    Metrics:
    - noise311_count: Raw count of nighttime complaints
    - noise311_rate_per_1k_pop: Rate per 1,000 population
    - noise311_rate_per_km2: Rate per square kilometer
    """
    logger.info("Computing CD metrics...")
    
    # Ensure all 59 CDs are present
    df = cd59[["boro_cd"]].copy()
    df = ensure_boro_cd_dtype(df)
    
    # Merge counts
    df = df.merge(noise_counts, on="boro_cd", how="left")
    df["noise311_count"] = df["noise311_count"].fillna(0).astype("Int64")
    
    # Merge population
    df = df.merge(cd_pop, on="boro_cd", how="left")
    
    # Calculate area in km2 (project to EPSG:2263 for accurate area)
    cd59_proj = safe_reproject(cd59, 2263, "CD59 for area calculation")
    cd59_proj["area_sqft"] = cd59_proj.geometry.area
    cd59_proj["area_km2"] = cd59_proj["area_sqft"] * 0.0929 / 1_000_000  # sq ft to km2
    
    df = df.merge(
        cd59_proj[["boro_cd", "area_km2"]],
        on="boro_cd",
        how="left",
    )
    
    # Compute rates
    df["noise311_rate_per_1k_pop"] = (df["noise311_count"] / df["population"]) * 1000
    df["noise311_rate_per_km2"] = df["noise311_count"] / df["area_km2"]
    
    # Handle division by zero
    df["noise311_rate_per_1k_pop"] = df["noise311_rate_per_1k_pop"].replace([np.inf, -np.inf], np.nan)
    df["noise311_rate_per_km2"] = df["noise311_rate_per_km2"].replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Metrics computed for {len(df)} CDs")
    logger.info(f"Total complaints: {df['noise311_count'].sum():,}")
    logger.info(f"Mean rate per 1k pop: {df['noise311_rate_per_1k_pop'].mean():.1f}")
    logger.info(f"Mean rate per km2: {df['noise311_rate_per_km2'].mean():.1f}")
    
    return df


def standardize_metrics(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Standardize metrics using multiple methods.
    
    Per Section 5.1:
    - Headline: robust z (median/MAD)
    - Sensitivities: classic z, percentile rank
    
    Per R20: Higher = worse (more noise exposure)
    """
    logger.info("Standardizing metrics...")
    
    # Primary metric: rate per 1k pop
    primary = df["noise311_rate_per_1k_pop"]
    
    # Log transform for better distribution (per Section 4.1.1)
    df["noise311_log_rate"] = np.log1p(primary)
    
    # Robust z-score (headline)
    df["z_noise311_robust"] = robust_z_score(df["noise311_log_rate"])
    
    # Classic z-score (sensitivity)
    df["z_noise311_classic"] = classic_z_score(df["noise311_log_rate"])
    
    # Percentile rank (sensitivity)
    df["pct_noise311"] = percentile_rank(df["noise311_log_rate"])
    
    # Log summary
    logger.info("Standardized metrics:")
    logger.info(f"  Robust z: mean={df['z_noise311_robust'].mean():.3f}, std={df['z_noise311_robust'].std():.3f}")
    logger.info(f"  Classic z: mean={df['z_noise311_classic'].mean():.3f}, std={df['z_noise311_classic'].std():.3f}")
    logger.info(f"  Percentile: min={df['pct_noise311'].min():.1f}, max={df['pct_noise311'].max():.1f}")
    
    return df


def build_noise_311(logger) -> pd.DataFrame:
    """
    Main function to build 311 noise metrics.
    """
    # Load config
    config = read_yaml(CONFIG_DIR / "params.yml")
    time_config = config["time_windows"]["primary"]
    night_config = config["nighttime"]["primary"]
    
    year_start = time_config["year_start"]
    year_end = time_config["year_end"]
    night_start = night_config["start_hour"]
    night_end = night_config["end_hour"]
    
    logger.info(f"Building RDI for {year_start}-{year_end}")
    logger.info(f"Nighttime window: {night_start:02d}:00 - {night_end:02d}:00")
    
    # Load data
    df_raw = load_raw_311(logger)
    cd59 = load_cd59(logger)
    cd_pop = load_cd_population(logger)
    
    # Prepare as GeoDataFrame
    gdf = prepare_311_points(df_raw, logger)
    
    # Parse timestamps
    gdf["created_date"] = pd.to_datetime(gdf["created_date"], errors="coerce")
    
    # Assert temporal coverage (R14)
    assert_temporal_coverage(
        gdf,
        "created_date",
        year_start,
        year_end,
        context="311 noise data",
    )
    
    # Filter to nighttime
    gdf_night = filter_to_nighttime(
        gdf,
        "created_date",
        night_start,
        night_end,
        logger,
    )
    
    # Spatial join to CDs (R6)
    logger.info("Performing spatial join to Community Districts...")
    
    # Project CD59 to EPSG:4326 for join (points are in 4326)
    cd59_4326 = safe_reproject(cd59, 4326, "CD59 to EPSG:4326")
    
    joined, stats = spatial_join_points_to_polygons(
        gdf_night,
        cd59_4326,
        polygon_id_col="boro_cd",
        max_distance=500,  # feet (EPSG:2263)
    )
    
    log_join_stats(stats, logger)
    
    # Aggregate counts per CD
    noise_counts = joined.groupby("boro_cd").size().reset_index(name="noise311_count")
    noise_counts = ensure_boro_cd_dtype(noise_counts)
    
    logger.info(f"Complaints joined to CDs: {noise_counts['noise311_count'].sum():,}")
    
    # Compute metrics
    df_metrics = compute_cd_metrics(noise_counts, cd_pop, cd59, logger)
    
    # Standardize
    df_metrics = standardize_metrics(df_metrics, logger)
    
    # Add metadata columns
    df_metrics["year_start"] = year_start
    df_metrics["year_end"] = year_end
    df_metrics["domain"] = "noise_311"
    df_metrics["units"] = "complaints"
    df_metrics["source_id"] = "nyc_311"
    
    # Ensure boro_cd is Int64
    df_metrics = ensure_boro_cd_dtype(df_metrics)
    
    # Sort by boro_cd for determinism
    df_metrics = df_metrics.sort_values("boro_cd").reset_index(drop=True)
    
    # Validate
    validate_boro_cd(df_metrics, "noise_311 output")
    
    if len(df_metrics) != 59:
        logger.warning(f"Expected 59 CDs, got {len(df_metrics)}")
    
    return df_metrics


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("03_build_noise_311") as logger:
        logger.info("Starting 03_build_noise_311.py")
        
        # Load config for logging
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        try:
            # Build noise 311 metrics
            df = build_noise_311(logger)
            
            # Ensure output directory exists
            DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Write output
            atomic_write_df(df, OUTPUT_PATH)
            logger.info(f"Wrote: {OUTPUT_PATH}")
            
            # Log outputs
            logger.log_outputs({"noise_311_cd": str(OUTPUT_PATH)})
            
            # Log metrics
            logger.log_metrics({
                "cd_count": len(df),
                "total_complaints": int(df["noise311_count"].sum()),
                "mean_rate_per_1k": float(df["noise311_rate_per_1k_pop"].mean()),
                "mean_rate_per_km2": float(df["noise311_rate_per_km2"].mean()),
                "z_robust_range": [
                    float(df["z_noise311_robust"].min()),
                    float(df["z_noise311_robust"].max()),
                ],
            })
            
            # Write metadata sidecar
            raw_files = list(RAW_311_DIR.glob("raw_311_noise_*.csv"))
            raw_path = max(raw_files, key=lambda p: p.stat().st_mtime) if raw_files else None
            
            write_metadata_sidecar(
                output_path=OUTPUT_PATH,
                inputs={"raw_311_noise": str(raw_path) if raw_path else "unknown"},
                config=config,
                run_id=logger.run_id,
                extra={
                    "cd_count": len(df),
                    "total_complaints": int(df["noise311_count"].sum()),
                },
            )
            
            # Print summary
            logger.info("=" * 60)
            logger.info("RDI Summary:")
            logger.info(f"  CDs: {len(df)}")
            logger.info(f"  Total nighttime complaints: {df['noise311_count'].sum():,}")
            logger.info(f"  Highest rate: CD {df.loc[df['noise311_rate_per_1k_pop'].idxmax(), 'boro_cd']} ({df['noise311_rate_per_1k_pop'].max():.1f} per 1k)")
            logger.info(f"  Lowest rate: CD {df.loc[df['noise311_rate_per_1k_pop'].idxmin(), 'boro_cd']} ({df['noise311_rate_per_1k_pop'].min():.1f} per 1k)")
            logger.info("=" * 60)
            
            logger.info("SUCCESS: Built RDI from 311 noise complaints")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

