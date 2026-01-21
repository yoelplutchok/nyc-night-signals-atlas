#!/usr/bin/env python3
"""
03_build_311_night_features.py

Build CD-level 311 nighttime complaint features for the NYC Night Signals Atlas.

Per NYC_Night_Signals_Plan.md Section 3.3.3:
- Compute CD-level features for 2021–2023 nighttime complaints
- Features include: counts, rates, type shares, time bins, weekend uplift, seasonality
- All outputs include cd_label via cd_lookup join

Outputs:
- data/processed/atlas/311_cd_features.parquet (main feature table)
- data/processed/atlas/311_cd_features.csv (human-readable)
- data/processed/atlas/311_cd_features.geojson (map-ready with geometry)

Per R7: Timezone-aware, DST-safe nighttime filtering
Per R6: Hardened spatial joins
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from sleep_esi.hashing import write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_df, atomic_write_gdf, read_yaml, read_gdf
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, XWALK_DIR, RAW_DIR, PROCESSED_DIR
from sleep_esi.qa import safe_reproject, assert_cd_labels_present
from sleep_esi.schemas import ensure_boro_cd_dtype, validate_boro_cd
from sleep_esi.time_utils import ensure_nyc_timezone, filter_nighttime
from sleep_esi.joins import spatial_join_points_to_polygons, log_join_stats

# =============================================================================
# Constants
# =============================================================================

RAW_311_DIR = RAW_DIR / "311_noise"
ATLAS_DIR = PROCESSED_DIR / "atlas"

# Output paths
OUTPUT_PARQUET = ATLAS_DIR / "311_cd_features.parquet"
OUTPUT_CSV = ATLAS_DIR / "311_cd_features.csv"
OUTPUT_GEOJSON = ATLAS_DIR / "311_cd_features.geojson"


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_311(logger) -> pd.DataFrame:
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


def load_cd_population(logger) -> pd.DataFrame:
    """Load CD population from tract crosswalk."""
    xwalk_path = XWALK_DIR / "cd_to_tract_weights.parquet"
    
    if not xwalk_path.exists():
        raise FileNotFoundError(
            f"Crosswalk not found: {xwalk_path}. "
            "Run 01_build_crosswalks.py first."
        )
    
    xwalk = pd.read_parquet(xwalk_path)
    
    cd_pop = xwalk.groupby("boro_cd")["tract_pop"].sum().reset_index()
    cd_pop = cd_pop.rename(columns={"tract_pop": "population"})
    cd_pop = ensure_boro_cd_dtype(cd_pop)
    
    logger.info(f"Loaded population for {len(cd_pop)} CDs")
    logger.info(f"Total population: {cd_pop['population'].sum():,}")
    
    return cd_pop


# =============================================================================
# Processing Functions
# =============================================================================

def prepare_311_geodataframe(df: pd.DataFrame, logger) -> gpd.GeoDataFrame:
    """Prepare 311 data as GeoDataFrame with valid coordinates and parsed timestamps."""
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
    
    # Filter to year range
    gdf = gdf.copy()
    gdf["year"] = gdf["ts_nyc"].dt.year
    gdf = gdf[(gdf["year"] >= year_start) & (gdf["year"] <= year_end)]
    logger.info(f"Records in year range: {len(gdf):,}")
    
    # Filter to nighttime
    gdf = filter_nighttime(gdf, "ts_nyc", start_hour=start_hour, end_hour=end_hour)
    logger.info(f"Nighttime records: {len(gdf):,}")
    
    return gdf


def assign_to_cds(gdf: gpd.GeoDataFrame, cd59: gpd.GeoDataFrame, logger) -> Tuple[gpd.GeoDataFrame, Dict]:
    """Spatially join points to Community Districts."""
    logger.info("Performing spatial join to Community Districts...")
    
    cd59_4326 = safe_reproject(cd59, 4326, "CD59 to EPSG:4326")
    
    joined, stats = spatial_join_points_to_polygons(
        gdf,
        cd59_4326,
        polygon_id_col="boro_cd",
        max_distance=500,  # feet
    )
    
    log_join_stats(stats, logger)
    joined = ensure_boro_cd_dtype(joined)
    
    return joined, stats


def compute_time_features(gdf: gpd.GeoDataFrame, logger) -> gpd.GeoDataFrame:
    """Add time-based features to each complaint record."""
    logger.info("Computing time features...")
    
    gdf = gdf.copy()
    
    # Hour of day
    gdf["hour"] = gdf["ts_nyc"].dt.hour
    
    # Day of week (Monday=0, Sunday=6)
    gdf["dow"] = gdf["ts_nyc"].dt.weekday
    
    # Month
    gdf["month"] = gdf["ts_nyc"].dt.month
    
    # Year
    gdf["year"] = gdf["ts_nyc"].dt.year
    
    return gdf


def compute_cd_features(
    joined: gpd.GeoDataFrame,
    cd59: gpd.GeoDataFrame,
    cd_pop: pd.DataFrame,
    cd_lookup: pd.DataFrame,
    config: dict,
    logger,
) -> pd.DataFrame:
    """
    Compute all CD-level features.
    
    Returns DataFrame with 59 rows and all feature columns.
    """
    logger.info("Computing CD-level features...")
    
    # Load config parameters
    night_bins = config.get("night_bins", {})
    bins_config = night_bins.get("bins", [])
    late_night_hours = night_bins.get("late_night_hours", [1, 2, 3])
    weekend_config = config.get("weekend", {})
    weekend_nights = weekend_config.get("weekend_nights", [4, 5])
    weekday_nights = weekend_config.get("weekday_nights", [0, 1, 2, 3, 6])
    warm_season = config.get("warm_season", {}).get("primary", {}).get("months", [6, 7, 8])
    complaint_types = config.get("noise_311", {}).get("complaint_types", [])
    
    # Start with all 59 CDs
    df = cd59[["boro_cd"]].copy()
    df = ensure_boro_cd_dtype(df)
    
    # =========================================================================
    # 1. Basic Counts
    # =========================================================================
    counts = joined.groupby("boro_cd").size().reset_index(name="count_night")
    counts = ensure_boro_cd_dtype(counts)
    df = df.merge(counts, on="boro_cd", how="left")
    df["count_night"] = df["count_night"].fillna(0).astype("Int64")
    
    logger.info(f"Total nighttime complaints: {df['count_night'].sum():,}")
    
    # =========================================================================
    # 2. Rates (per 1k pop, per km²)
    # =========================================================================
    # Merge population
    df = df.merge(cd_pop, on="boro_cd", how="left")
    
    # Calculate area in km²
    cd59_proj = safe_reproject(cd59, 2263, "CD59 for area")
    cd59_proj["area_km2"] = cd59_proj.geometry.area * 0.0929 / 1_000_000  # sq ft to km²
    df = df.merge(cd59_proj[["boro_cd", "area_km2"]], on="boro_cd", how="left")
    
    # Compute rates
    df["rate_per_1k_pop"] = (df["count_night"] / df["population"]) * 1000
    df["rate_per_km2"] = df["count_night"] / df["area_km2"]
    
    # Handle division issues
    df["rate_per_1k_pop"] = df["rate_per_1k_pop"].replace([np.inf, -np.inf], np.nan)
    df["rate_per_km2"] = df["rate_per_km2"].replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Mean rate per 1k pop: {df['rate_per_1k_pop'].mean():.1f}")
    logger.info(f"Mean rate per km²: {df['rate_per_km2'].mean():.1f}")
    
    # =========================================================================
    # 3. Complaint Type Shares
    # =========================================================================
    logger.info("Computing complaint type shares...")
    
    # Get type counts per CD
    type_counts = joined.groupby(["boro_cd", "complaint_type"]).size().unstack(fill_value=0)
    type_counts = type_counts.reset_index()
    type_counts = ensure_boro_cd_dtype(type_counts)
    
    # Merge with main df
    df = df.merge(type_counts, on="boro_cd", how="left")
    
    # Compute shares for each complaint type
    for ctype in complaint_types:
        col_name = f"share_{ctype.lower().replace(' ', '_').replace('-', '_')}"
        if ctype in df.columns:
            df[col_name] = df[ctype] / df["count_night"]
            df[col_name] = df[col_name].fillna(0)
        else:
            df[col_name] = 0.0
    
    # Drop raw type count columns
    for ctype in complaint_types:
        if ctype in df.columns:
            df = df.drop(columns=[ctype])
    
    # =========================================================================
    # 4. Time-of-Night Bin Shares
    # =========================================================================
    logger.info("Computing time-of-night bin shares...")
    
    for bin_def in bins_config:
        bin_name = bin_def["name"]
        bin_start = bin_def["start"]
        bin_end = bin_def["end"]
        
        # Handle cross-midnight bins
        if bin_start < bin_end:
            mask = (joined["hour"] >= bin_start) & (joined["hour"] < bin_end)
        else:
            # For bins like 22-24 (end=24 means 00:00)
            if bin_end == 24:
                mask = joined["hour"] >= bin_start
            else:
                mask = (joined["hour"] >= bin_start) | (joined["hour"] < bin_end)
        
        bin_counts = joined[mask].groupby("boro_cd").size().reset_index(name=f"count_{bin_name}")
        bin_counts = ensure_boro_cd_dtype(bin_counts)
        df = df.merge(bin_counts, on="boro_cd", how="left")
        df[f"count_{bin_name}"] = df[f"count_{bin_name}"].fillna(0)
        df[f"share_{bin_name}"] = df[f"count_{bin_name}"] / df["count_night"]
        df[f"share_{bin_name}"] = df[f"share_{bin_name}"].fillna(0)
    
    # =========================================================================
    # 5. Late-Night Share (01:00-04:00)
    # =========================================================================
    logger.info("Computing late-night share (01:00-04:00)...")
    
    late_night_mask = joined["hour"].isin(late_night_hours)
    late_night_joined = joined[late_night_mask]
    logger.info(f"Late-night complaints (01:00-04:00): {len(late_night_joined):,}")
    
    if len(late_night_joined) > 0:
        late_counts = late_night_joined.groupby("boro_cd").size().reset_index(name="count_late_night")
        late_counts = ensure_boro_cd_dtype(late_counts)
        df = df.merge(late_counts, on="boro_cd", how="left")
    
    # Ensure column exists even if no late-night complaints
    if "count_late_night" not in df.columns:
        df["count_late_night"] = 0
    df["count_late_night"] = df["count_late_night"].fillna(0).astype(int)
    df["late_night_share"] = df["count_late_night"] / df["count_night"]
    df["late_night_share"] = df["late_night_share"].fillna(0)
    
    # =========================================================================
    # 6. Weekend Uplift
    # =========================================================================
    logger.info("Computing weekend uplift...")
    
    # Weekend nights (Fri, Sat)
    weekend_mask = joined["dow"].isin(weekend_nights)
    weekend_joined = joined[weekend_mask]
    if len(weekend_joined) > 0:
        weekend_counts = weekend_joined.groupby("boro_cd").size().reset_index(name="count_weekend")
        weekend_counts = ensure_boro_cd_dtype(weekend_counts)
        df = df.merge(weekend_counts, on="boro_cd", how="left")
    if "count_weekend" not in df.columns:
        df["count_weekend"] = 0
    df["count_weekend"] = df["count_weekend"].fillna(0).astype(int)
    
    # Weekday nights (Sun-Thu)
    weekday_mask = joined["dow"].isin(weekday_nights)
    weekday_joined = joined[weekday_mask]
    if len(weekday_joined) > 0:
        weekday_counts = weekday_joined.groupby("boro_cd").size().reset_index(name="count_weekday")
        weekday_counts = ensure_boro_cd_dtype(weekday_counts)
        df = df.merge(weekday_counts, on="boro_cd", how="left")
    if "count_weekday" not in df.columns:
        df["count_weekday"] = 0
    df["count_weekday"] = df["count_weekday"].fillna(0).astype(int)
    
    # Weekend uplift = (weekend_count/2) / (weekday_count/5)
    df["weekend_rate"] = df["count_weekend"] / 2
    df["weekday_rate"] = df["count_weekday"] / 5
    df["weekend_uplift"] = df["weekend_rate"] / df["weekday_rate"]
    df["weekend_uplift"] = df["weekend_uplift"].replace([np.inf, -np.inf], np.nan)
    
    logger.info(f"Mean weekend uplift: {df['weekend_uplift'].mean():.2f}")
    
    # =========================================================================
    # 7. Warm Season Ratio
    # =========================================================================
    logger.info("Computing warm season ratio...")
    
    warm_mask = joined["month"].isin(warm_season)
    warm_joined = joined[warm_mask]
    if len(warm_joined) > 0:
        warm_counts = warm_joined.groupby("boro_cd").size().reset_index(name="count_warm")
        warm_counts = ensure_boro_cd_dtype(warm_counts)
        df = df.merge(warm_counts, on="boro_cd", how="left")
    if "count_warm" not in df.columns:
        df["count_warm"] = 0
    df["count_warm"] = df["count_warm"].fillna(0).astype(int)
    
    cool_mask = ~joined["month"].isin(warm_season)
    cool_joined = joined[cool_mask]
    if len(cool_joined) > 0:
        cool_counts = cool_joined.groupby("boro_cd").size().reset_index(name="count_cool")
        cool_counts = ensure_boro_cd_dtype(cool_counts)
        df = df.merge(cool_counts, on="boro_cd", how="left")
    if "count_cool" not in df.columns:
        df["count_cool"] = 0
    df["count_cool"] = df["count_cool"].fillna(0).astype(int)
    
    # Warm season ratio (normalize by months: 3 warm vs 9 cool)
    df["warm_rate"] = df["count_warm"] / 3
    df["cool_rate"] = df["count_cool"] / 9
    df["warm_season_ratio"] = df["warm_rate"] / df["cool_rate"]
    df["warm_season_ratio"] = df["warm_season_ratio"].replace([np.inf, -np.inf], np.nan)
    
    # Also compute warm share
    df["warm_share"] = df["count_warm"] / df["count_night"]
    df["warm_share"] = df["warm_share"].fillna(0)
    
    logger.info(f"Mean warm season ratio: {df['warm_season_ratio'].mean():.2f}")
    
    # =========================================================================
    # 8. Join CD Labels (REQUIRED per CD labeling policy)
    # =========================================================================
    logger.info("Joining CD labels...")
    
    df = df.merge(
        cd_lookup[["boro_cd", "borough_name", "district_number", "cd_label", "cd_short"]],
        on="boro_cd",
        how="left",
    )
    
    # =========================================================================
    # 9. Add Metadata Columns
    # =========================================================================
    time_config = config["time_windows"]["primary"]
    df["year_start"] = time_config["year_start"]
    df["year_end"] = time_config["year_end"]
    
    # =========================================================================
    # 10. Final Cleanup
    # =========================================================================
    # Ensure proper dtypes
    df = ensure_boro_cd_dtype(df)
    
    # Sort for determinism
    df = df.sort_values("boro_cd").reset_index(drop=True)
    
    # Validate
    validate_boro_cd(df, "311 features output")
    assert_cd_labels_present(df, "311 features output")
    
    return df


def validate_features(df: pd.DataFrame, logger) -> Dict:
    """Validate the output features and return QA stats."""
    logger.info("Validating features...")
    
    qa_stats = {}
    
    # Row count
    qa_stats["row_count"] = len(df)
    if len(df) != 59:
        logger.warning(f"Expected 59 rows, got {len(df)}")
    
    # CD label check
    null_labels = df["cd_label"].isna().sum()
    qa_stats["null_cd_labels"] = int(null_labels)
    if null_labels > 0:
        logger.error(f"Found {null_labels} null cd_label values!")
    
    # Type share sum check
    type_share_cols = [c for c in df.columns if c.startswith("share_noise")]
    if type_share_cols:
        type_share_sum = df[type_share_cols].sum(axis=1)
        qa_stats["type_share_sum_mean"] = float(type_share_sum.mean())
        qa_stats["type_share_sum_min"] = float(type_share_sum.min())
        qa_stats["type_share_sum_max"] = float(type_share_sum.max())
        
        # Check if sums are approximately 1
        bad_sums = ((type_share_sum < 0.99) | (type_share_sum > 1.01)) & (df["count_night"] > 0)
        if bad_sums.any():
            logger.warning(f"Type share sums not ≈1 for {bad_sums.sum()} CDs")
    
    # Bin share sum check
    # Only check the 4 time-of-night bins (evening, early_am, core_night, predawn)
    time_bin_cols = ["share_evening", "share_early_am", "share_core_night", "share_predawn"]
    time_bin_cols = [c for c in time_bin_cols if c in df.columns]
    if time_bin_cols:
        bin_share_sum = df[time_bin_cols].sum(axis=1)
        qa_stats["bin_share_sum_mean"] = float(bin_share_sum.mean())
        qa_stats["bin_share_sum_min"] = float(bin_share_sum.min())
        qa_stats["bin_share_sum_max"] = float(bin_share_sum.max())
        
        bad_bin_sums = ((bin_share_sum < 0.99) | (bin_share_sum > 1.01)) & (df["count_night"] > 0)
        if bad_bin_sums.any():
            logger.warning(f"Bin share sums not ≈1 for {bad_bin_sums.sum()} CDs")
    
    # Coverage stats
    qa_stats["total_complaints"] = int(df["count_night"].sum())
    qa_stats["min_complaints_per_cd"] = int(df["count_night"].min())
    qa_stats["max_complaints_per_cd"] = int(df["count_night"].max())
    qa_stats["mean_complaints_per_cd"] = float(df["count_night"].mean())
    
    logger.info(f"QA Stats: {qa_stats}")
    
    return qa_stats


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("03_build_311_night_features") as logger:
        logger.info("Starting 03_build_311_night_features.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        time_config = config["time_windows"]["primary"]
        night_config = config["nighttime"]["primary"]
        
        year_start = time_config["year_start"]
        year_end = time_config["year_end"]
        night_start = night_config["start_hour"]
        night_end = night_config["end_hour"]
        
        logger.info(f"Building 311 night features for {year_start}-{year_end}")
        logger.info(f"Nighttime window: {night_start:02d}:00 - {night_end:02d}:00")
        
        try:
            # Load data
            df_raw, raw_path = load_raw_311(logger)
            cd59 = load_cd59(logger)
            cd_lookup = load_cd_lookup(logger)
            cd_pop = load_cd_population(logger)
            
            # Prepare GeoDataFrame
            gdf = prepare_311_geodataframe(df_raw, logger)
            
            # Filter to nighttime and year range
            gdf_night = filter_to_nighttime_and_years(
                gdf, night_start, night_end, year_start, year_end, logger
            )
            
            # Add time features
            gdf_night = compute_time_features(gdf_night, logger)
            
            # Spatial join to CDs
            joined, join_stats = assign_to_cds(gdf_night, cd59, logger)
            
            # Compute all CD-level features
            df_features = compute_cd_features(
                joined, cd59, cd_pop, cd_lookup, config, logger
            )
            
            # Validate
            qa_stats = validate_features(df_features, logger)
            
            # Ensure output directory exists
            ATLAS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Write parquet
            atomic_write_df(df_features, OUTPUT_PARQUET)
            logger.info(f"Wrote: {OUTPUT_PARQUET}")
            
            # Write CSV
            df_features.to_csv(OUTPUT_CSV, index=False)
            logger.info(f"Wrote: {OUTPUT_CSV}")
            
            # Write GeoJSON (join to geometry)
            gdf_output = cd59[["boro_cd", "geometry"]].merge(df_features, on="boro_cd")
            gdf_output = ensure_boro_cd_dtype(gdf_output)
            atomic_write_gdf(gdf_output, OUTPUT_GEOJSON)
            logger.info(f"Wrote: {OUTPUT_GEOJSON}")
            
            # Log outputs
            logger.log_outputs({
                "311_cd_features_parquet": str(OUTPUT_PARQUET),
                "311_cd_features_csv": str(OUTPUT_CSV),
                "311_cd_features_geojson": str(OUTPUT_GEOJSON),
            })
            
            # Log metrics
            logger.log_metrics({
                "cd_count": len(df_features),
                "total_complaints": int(df_features["count_night"].sum()),
                "mean_rate_per_1k": float(df_features["rate_per_1k_pop"].mean()),
                "mean_weekend_uplift": float(df_features["weekend_uplift"].mean()),
                "mean_warm_season_ratio": float(df_features["warm_season_ratio"].mean()),
                "join_stats": join_stats,
                "qa_stats": qa_stats,
            })
            
            # Write metadata sidecar
            write_metadata_sidecar(
                output_path=OUTPUT_PARQUET,
                inputs={"raw_311_noise": str(raw_path)},
                config=config,
                run_id=logger.run_id,
                extra={
                    "cd_count": len(df_features),
                    "total_complaints": int(df_features["count_night"].sum()),
                    "columns": list(df_features.columns),
                    "qa_stats": qa_stats,
                    "join_stats": join_stats,
                },
            )
            
            # Print summary
            logger.info("=" * 70)
            logger.info("311 Night Features Summary:")
            logger.info(f"  CDs: {len(df_features)}")
            logger.info(f"  Total nighttime complaints: {df_features['count_night'].sum():,}")
            logger.info(f"  Mean rate per 1k pop: {df_features['rate_per_1k_pop'].mean():.1f}")
            logger.info(f"  Mean weekend uplift: {df_features['weekend_uplift'].mean():.2f}")
            logger.info(f"  Mean warm season ratio: {df_features['warm_season_ratio'].mean():.2f}")
            logger.info(f"  Mean late-night share: {df_features['late_night_share'].mean():.3f}")
            logger.info("=" * 70)
            
            # Show top 5 by rate
            logger.info("Top 5 CDs by complaint rate (per 1k pop):")
            top5 = df_features.nlargest(5, "rate_per_1k_pop")[["boro_cd", "cd_label", "rate_per_1k_pop"]]
            for _, row in top5.iterrows():
                logger.info(f"  {row['cd_label']}: {row['rate_per_1k_pop']:.1f}")
            
            logger.info("SUCCESS: Built 311 night features")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

