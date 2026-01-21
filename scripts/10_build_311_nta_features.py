#!/usr/bin/env python3
"""
10_build_311_nta_features.py

Build NTA-level 311 nighttime complaint features for the NYC Night Signals Atlas.

Per NYC_Night_Signals_Plan.md Section 3.3 (Script 10):
- Mirror Script 03 feature schema at NTA level
- Produce TWO outputs:
  1) Residential/mixed NTAs only (is_residential==True): 197 rows
  2) All NTAs: 262 rows
- Include is_residential, ntatype_label, and nta_name in both
- Join nta_lookup so official nta_name is always present

Outputs (for each variant):
- data/processed/atlas/311_nta_features.parquet (all NTAs)
- data/processed/atlas/311_nta_features_residential.parquet (residential only)
- data/processed/atlas/311_nta_features.csv
- data/processed/atlas/311_nta_features_residential.csv
- data/processed/atlas/311_nta_features.geojson
- data/processed/atlas/311_nta_features_residential.geojson

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
from sleep_esi.qa import safe_reproject
from sleep_esi.time_utils import ensure_nyc_timezone, filter_nighttime


# =============================================================================
# Constants
# =============================================================================

RAW_311_DIR = RAW_DIR / "311_noise"
ATLAS_DIR = PROCESSED_DIR / "atlas"

# Output paths - all NTAs
OUTPUT_PARQUET_ALL = ATLAS_DIR / "311_nta_features.parquet"
OUTPUT_CSV_ALL = ATLAS_DIR / "311_nta_features.csv"
OUTPUT_GEOJSON_ALL = ATLAS_DIR / "311_nta_features.geojson"

# Output paths - residential only
OUTPUT_PARQUET_RES = ATLAS_DIR / "311_nta_features_residential.parquet"
OUTPUT_CSV_RES = ATLAS_DIR / "311_nta_features_residential.csv"
OUTPUT_GEOJSON_RES = ATLAS_DIR / "311_nta_features_residential.geojson"


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


def load_nta(logger) -> gpd.GeoDataFrame:
    """Load the canonical NTA geometries."""
    nta_path = GEO_DIR / "nta.parquet"
    
    if not nta_path.exists():
        raise FileNotFoundError(
            f"NTA file not found: {nta_path}. "
            "Run 09_build_nta.py first."
        )
    
    gdf = read_gdf(nta_path)
    logger.info(f"Loaded NTA: {len(gdf)} NTAs")
    
    return gdf


def load_nta_lookup(logger) -> pd.DataFrame:
    """Load the NTA lookup table for labels."""
    lookup_path = GEO_DIR / "nta_lookup.parquet"
    
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"NTA lookup not found: {lookup_path}. "
            "Run 09_build_nta.py first."
        )
    
    df = pd.read_parquet(lookup_path)
    logger.info(f"Loaded NTA lookup: {len(df)} entries")
    
    return df


def load_nta_population(logger) -> pd.DataFrame:
    """Load NTA population from crosswalk."""
    xwalk_path = XWALK_DIR / "cd_to_nta_weights.parquet"
    
    if not xwalk_path.exists():
        # Try to compute from tract crosswalk if available
        logger.warning(f"NTA crosswalk not found: {xwalk_path}")
        logger.warning("Rates per 1k pop will be unavailable")
        return pd.DataFrame(columns=["ntacode", "population"])
    
    xwalk = pd.read_parquet(xwalk_path)
    
    # Get unique NTA populations (each NTA appears multiple times in CD crosswalk)
    nta_pop = xwalk.groupby("nta2020")["nta_pop"].first().reset_index()
    nta_pop = nta_pop.rename(columns={"nta2020": "ntacode", "nta_pop": "population"})
    
    logger.info(f"Loaded population for {len(nta_pop)} NTAs")
    logger.info(f"Total NTA population: {nta_pop['population'].sum():,}")
    
    return nta_pop


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
    
    # Create GeoDataFrame
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


def spatial_join_to_nta(
    gdf: gpd.GeoDataFrame,
    nta: gpd.GeoDataFrame,
    logger,
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """Spatially join 311 points to NTA polygons."""
    logger.info("Joining 311 points to NTAs...")
    
    # Ensure same CRS
    if gdf.crs != nta.crs:
        gdf = gdf.to_crs(nta.crs)
    
    # Perform spatial join
    joined = gpd.sjoin(
        gdf,
        nta[["ntacode", "nta_name", "geometry"]],
        how="left",
        predicate="within",
    )
    
    # Stats
    n_total = len(joined)
    n_assigned = joined["ntacode"].notna().sum()
    n_unassigned = n_total - n_assigned
    
    stats = {
        "total_points": n_total,
        "assigned": int(n_assigned),
        "unassigned": int(n_unassigned),
        "assignment_rate": round(n_assigned / n_total, 4) if n_total > 0 else 0,
    }
    
    logger.info(f"Points assigned: {n_assigned:,} / {n_total:,} ({stats['assignment_rate']:.1%})")
    logger.info(f"Unassigned points: {n_unassigned:,}")
    
    # Filter to assigned points only
    joined = joined[joined["ntacode"].notna()].copy()
    
    return joined, stats


def compute_time_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add time-based features to each complaint record."""
    gdf = gdf.copy()
    gdf["hour"] = gdf["ts_nyc"].dt.hour
    gdf["dow"] = gdf["ts_nyc"].dt.weekday
    gdf["month"] = gdf["ts_nyc"].dt.month
    gdf["year"] = gdf["ts_nyc"].dt.year
    return gdf


def compute_nta_features(
    joined: gpd.GeoDataFrame,
    nta: gpd.GeoDataFrame,
    nta_pop: pd.DataFrame,
    nta_lookup: pd.DataFrame,
    config: dict,
    logger,
) -> pd.DataFrame:
    """
    Compute all NTA-level features.
    
    Returns DataFrame with all NTAs and feature columns.
    """
    logger.info("Computing NTA-level features...")
    
    # Load config parameters
    night_bins = config.get("night_bins", {})
    bins_config = night_bins.get("bins", [])
    late_night_hours = night_bins.get("late_night_hours", [1, 2, 3])
    weekend_config = config.get("weekend", {})
    weekend_nights = weekend_config.get("weekend_nights", [4, 5])
    weekday_nights = weekend_config.get("weekday_nights", [0, 1, 2, 3, 6])
    warm_season = config.get("warm_season", {}).get("primary", {}).get("months", [6, 7, 8])
    complaint_types = config.get("noise_311", {}).get("complaint_types", [])
    
    # Start with all NTAs from lookup
    df = nta_lookup[["ntacode", "nta_name", "borough_name", "ntatype", "ntatype_label", "is_residential"]].copy()
    
    # =========================================================================
    # 1. Basic Counts
    # =========================================================================
    counts = joined.groupby("ntacode").size().reset_index(name="count_night")
    df = df.merge(counts, on="ntacode", how="left")
    df["count_night"] = df["count_night"].fillna(0).astype("Int64")
    
    logger.info(f"Total nighttime complaints assigned to NTAs: {df['count_night'].sum():,}")
    
    # =========================================================================
    # 2. Rates (per 1k pop, per km²)
    # =========================================================================
    # Merge population
    df = df.merge(nta_pop, on="ntacode", how="left")
    df["population"] = df["population"].fillna(0)
    
    # Calculate area in km² from geometry
    nta_proj = safe_reproject(nta, 2263, "NTA for area")
    nta_proj["area_km2"] = nta_proj.geometry.area * (0.3048 ** 2) / 1_000_000  # sq ft to km²
    df = df.merge(nta_proj[["ntacode", "area_km2"]], on="ntacode", how="left")
    
    # Compute rates
    df["rate_per_1k_pop"] = np.where(
        df["population"] > 0,
        (df["count_night"] / df["population"]) * 1000,
        np.nan
    )
    df["rate_per_km2"] = np.where(
        df["area_km2"] > 0,
        df["count_night"] / df["area_km2"],
        np.nan
    )
    
    logger.info(f"Mean rate per 1k pop: {df['rate_per_1k_pop'].mean():.1f}")
    logger.info(f"Mean rate per km²: {df['rate_per_km2'].mean():.1f}")
    
    # =========================================================================
    # 3. Complaint Type Shares
    # =========================================================================
    logger.info("Computing complaint type shares...")
    
    type_counts = joined.groupby(["ntacode", "complaint_type"]).size().unstack(fill_value=0)
    type_counts = type_counts.reset_index()
    df = df.merge(type_counts, on="ntacode", how="left")
    
    for ctype in complaint_types:
        col_name = f"share_{ctype.lower().replace(' ', '_').replace('-', '_')}"
        if ctype in df.columns:
            df[col_name] = np.where(
                df["count_night"] > 0,
                df[ctype] / df["count_night"],
                0.0
            )
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
        
        if bin_start < bin_end:
            mask = (joined["hour"] >= bin_start) & (joined["hour"] < bin_end)
        else:
            if bin_end == 24:
                mask = joined["hour"] >= bin_start
            else:
                mask = (joined["hour"] >= bin_start) | (joined["hour"] < bin_end)
        
        bin_counts = joined[mask].groupby("ntacode").size().reset_index(name=f"count_{bin_name}")
        df = df.merge(bin_counts, on="ntacode", how="left")
        df[f"count_{bin_name}"] = df[f"count_{bin_name}"].fillna(0)
        df[f"share_{bin_name}"] = np.where(
            df["count_night"] > 0,
            df[f"count_{bin_name}"] / df["count_night"],
            0.0
        )
    
    # =========================================================================
    # 5. Late-Night Share (01:00-04:00)
    # =========================================================================
    logger.info("Computing late-night share...")
    
    late_night_mask = joined["hour"].isin(late_night_hours)
    late_counts = joined[late_night_mask].groupby("ntacode").size().reset_index(name="count_late_night")
    df = df.merge(late_counts, on="ntacode", how="left")
    df["count_late_night"] = df["count_late_night"].fillna(0).astype(int)
    df["late_night_share"] = np.where(
        df["count_night"] > 0,
        df["count_late_night"] / df["count_night"],
        0.0
    )
    
    # =========================================================================
    # 6. Weekend Uplift
    # =========================================================================
    logger.info("Computing weekend uplift...")
    
    weekend_mask = joined["dow"].isin(weekend_nights)
    weekend_counts = joined[weekend_mask].groupby("ntacode").size().reset_index(name="count_weekend")
    df = df.merge(weekend_counts, on="ntacode", how="left")
    df["count_weekend"] = df["count_weekend"].fillna(0).astype(int)
    
    weekday_mask = joined["dow"].isin(weekday_nights)
    weekday_counts = joined[weekday_mask].groupby("ntacode").size().reset_index(name="count_weekday")
    df = df.merge(weekday_counts, on="ntacode", how="left")
    df["count_weekday"] = df["count_weekday"].fillna(0).astype(int)
    
    # Weekend uplift = (weekend_count/2) / (weekday_count/5)
    df["weekend_rate"] = df["count_weekend"] / 2
    df["weekday_rate"] = df["count_weekday"] / 5
    df["weekend_uplift"] = np.where(
        df["weekday_rate"] > 0,
        df["weekend_rate"] / df["weekday_rate"],
        np.nan
    )
    
    logger.info(f"Mean weekend uplift: {df['weekend_uplift'].mean():.2f}")
    
    # =========================================================================
    # 7. Warm Season Ratio
    # =========================================================================
    logger.info("Computing warm season ratio...")
    
    warm_mask = joined["month"].isin(warm_season)
    warm_counts = joined[warm_mask].groupby("ntacode").size().reset_index(name="count_warm")
    df = df.merge(warm_counts, on="ntacode", how="left")
    df["count_warm"] = df["count_warm"].fillna(0).astype(int)
    
    cool_mask = ~joined["month"].isin(warm_season)
    cool_counts = joined[cool_mask].groupby("ntacode").size().reset_index(name="count_cool")
    df = df.merge(cool_counts, on="ntacode", how="left")
    df["count_cool"] = df["count_cool"].fillna(0).astype(int)
    
    # Warm season ratio (normalize by months: 3 warm vs 9 cool)
    df["warm_rate"] = df["count_warm"] / 3
    df["cool_rate"] = df["count_cool"] / 9
    df["warm_season_ratio"] = np.where(
        df["cool_rate"] > 0,
        df["warm_rate"] / df["cool_rate"],
        np.nan
    )
    
    logger.info(f"Mean warm season ratio: {df['warm_season_ratio'].mean():.2f}")
    
    # =========================================================================
    # 8. Add Metadata
    # =========================================================================
    time_config = config["time_windows"]["primary"]
    df["year_start"] = time_config["year_start"]
    df["year_end"] = time_config["year_end"]
    
    # =========================================================================
    # 9. Final Cleanup
    # =========================================================================
    # Drop intermediate rate columns
    df = df.drop(columns=["weekend_rate", "weekday_rate", "warm_rate", "cool_rate"], errors="ignore")
    
    # Sort for determinism
    df = df.sort_values("ntacode").reset_index(drop=True)
    
    return df


def validate_features(df: pd.DataFrame, expected_rows: int, label: str, logger) -> Dict:
    """Validate the output features and return QA stats."""
    logger.info(f"Validating features ({label})...")
    
    qa_stats = {"label": label}
    
    # Row count
    qa_stats["row_count"] = len(df)
    if len(df) != expected_rows:
        logger.warning(f"Expected {expected_rows} rows, got {len(df)}")
    
    # NTA name check
    null_names = df["nta_name"].isna().sum()
    qa_stats["null_nta_names"] = int(null_names)
    if null_names > 0:
        logger.error(f"Found {null_names} null nta_name values!")
    
    # ntacode check
    null_codes = df["ntacode"].isna().sum()
    qa_stats["null_ntacodes"] = int(null_codes)
    if null_codes > 0:
        logger.error(f"Found {null_codes} null ntacode values!")
    
    # Type share sum check (only for NTAs with complaints)
    type_share_cols = [c for c in df.columns if c.startswith("share_noise")]
    if type_share_cols:
        has_complaints = df["count_night"] > 0
        if has_complaints.any():
            type_share_sum = df.loc[has_complaints, type_share_cols].sum(axis=1)
            qa_stats["type_share_sum_mean"] = float(type_share_sum.mean())
            qa_stats["type_share_sum_min"] = float(type_share_sum.min())
            qa_stats["type_share_sum_max"] = float(type_share_sum.max())
            
            bad_sums = (type_share_sum < 0.99) | (type_share_sum > 1.01)
            if bad_sums.any():
                logger.warning(f"Type share sums not ≈1 for {bad_sums.sum()} NTAs")
    
    # Bin share sum check
    time_bin_cols = ["share_evening", "share_early_am", "share_core_night", "share_predawn"]
    time_bin_cols = [c for c in time_bin_cols if c in df.columns]
    if time_bin_cols:
        has_complaints = df["count_night"] > 0
        if has_complaints.any():
            bin_share_sum = df.loc[has_complaints, time_bin_cols].sum(axis=1)
            qa_stats["bin_share_sum_mean"] = float(bin_share_sum.mean())
            qa_stats["bin_share_sum_min"] = float(bin_share_sum.min())
            qa_stats["bin_share_sum_max"] = float(bin_share_sum.max())
            
            bad_bin_sums = (bin_share_sum < 0.99) | (bin_share_sum > 1.01)
            if bad_bin_sums.any():
                logger.warning(f"Bin share sums not ≈1 for {bad_bin_sums.sum()} NTAs")
    
    # Coverage stats
    qa_stats["total_complaints"] = int(df["count_night"].sum())
    qa_stats["ntas_with_complaints"] = int((df["count_night"] > 0).sum())
    qa_stats["ntas_without_complaints"] = int((df["count_night"] == 0).sum())
    qa_stats["min_complaints_per_nta"] = int(df["count_night"].min())
    qa_stats["max_complaints_per_nta"] = int(df["count_night"].max())
    qa_stats["mean_complaints_per_nta"] = float(df["count_night"].mean())
    
    logger.info(f"QA Stats: {qa_stats}")
    
    return qa_stats


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("10_build_311_nta_features") as logger:
        logger.info("Starting 10_build_311_nta_features.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        time_config = config["time_windows"]["primary"]
        night_config = config["nighttime"]["primary"]
        
        year_start = time_config["year_start"]
        year_end = time_config["year_end"]
        night_start = night_config["start_hour"]
        night_end = night_config["end_hour"]
        
        logger.info(f"Building 311 NTA-level night features for {year_start}-{year_end}")
        logger.info(f"Nighttime window: {night_start:02d}:00 - {night_end:02d}:00")
        
        try:
            # Load data
            df_raw, raw_path = load_raw_311(logger)
            nta = load_nta(logger)
            nta_lookup = load_nta_lookup(logger)
            nta_pop = load_nta_population(logger)
            
            # Prepare 311 points
            gdf = prepare_311_geodataframe(df_raw, logger)
            
            # Filter to nighttime and years
            gdf_night = filter_to_nighttime_and_years(
                gdf, night_start, night_end, year_start, year_end, logger
            )
            
            # Add time features
            gdf_night = compute_time_features(gdf_night)
            
            # Spatial join to NTAs
            joined, join_stats = spatial_join_to_nta(gdf_night, nta, logger)
            
            # Compute NTA features
            df_features = compute_nta_features(
                joined, nta, nta_pop, nta_lookup, config, logger
            )
            
            # Split into residential and all
            df_all = df_features.copy()
            df_residential = df_features[df_features["is_residential"] == True].copy()
            
            logger.info(f"All NTAs: {len(df_all)} rows")
            logger.info(f"Residential NTAs: {len(df_residential)} rows")
            
            # Validate both
            qa_all = validate_features(df_all, 262, "all_ntas", logger)
            qa_res = validate_features(df_residential, 197, "residential_ntas", logger)
            
            # Ensure output directory exists
            ATLAS_DIR.mkdir(parents=True, exist_ok=True)
            
            # =========================================================================
            # Write outputs - ALL NTAs
            # =========================================================================
            logger.info("Writing outputs for ALL NTAs...")
            
            # Parquet
            atomic_write_df(df_all, OUTPUT_PARQUET_ALL)
            logger.info(f"Wrote: {OUTPUT_PARQUET_ALL} ({len(df_all)} rows)")
            
            # CSV
            df_all.to_csv(OUTPUT_CSV_ALL, index=False)
            logger.info(f"Wrote: {OUTPUT_CSV_ALL}")
            
            # GeoJSON (join to geometry - keep only geometry from NTA)
            gdf_all = nta[["ntacode", "geometry"]].merge(df_all, on="ntacode", how="right")
            gdf_all = gdf_all.set_geometry("geometry")
            atomic_write_gdf(gdf_all, OUTPUT_GEOJSON_ALL)
            logger.info(f"Wrote: {OUTPUT_GEOJSON_ALL}")
            
            # =========================================================================
            # Write outputs - RESIDENTIAL NTAs
            # =========================================================================
            logger.info("Writing outputs for RESIDENTIAL NTAs...")
            
            # Parquet
            atomic_write_df(df_residential, OUTPUT_PARQUET_RES)
            logger.info(f"Wrote: {OUTPUT_PARQUET_RES} ({len(df_residential)} rows)")
            
            # CSV
            df_residential.to_csv(OUTPUT_CSV_RES, index=False)
            logger.info(f"Wrote: {OUTPUT_CSV_RES}")
            
            # GeoJSON (join to geometry - keep only geometry from NTA)
            gdf_res = nta[["ntacode", "geometry"]].merge(df_residential, on="ntacode", how="right")
            gdf_res = gdf_res.set_geometry("geometry")
            atomic_write_gdf(gdf_res, OUTPUT_GEOJSON_RES)
            logger.info(f"Wrote: {OUTPUT_GEOJSON_RES}")
            
            # Log outputs
            logger.log_outputs({
                "nta_features_all_parquet": str(OUTPUT_PARQUET_ALL),
                "nta_features_all_csv": str(OUTPUT_CSV_ALL),
                "nta_features_all_geojson": str(OUTPUT_GEOJSON_ALL),
                "nta_features_res_parquet": str(OUTPUT_PARQUET_RES),
                "nta_features_res_csv": str(OUTPUT_CSV_RES),
                "nta_features_res_geojson": str(OUTPUT_GEOJSON_RES),
            })
            
            # Log metrics
            logger.log_metrics({
                "join_stats": join_stats,
                "qa_all": qa_all,
                "qa_residential": qa_res,
            })
            
            # Log join stats
            logger.log_join_stats(join_stats)
            
            # Write metadata sidecar
            write_metadata_sidecar(
                output_path=OUTPUT_PARQUET_ALL,
                inputs={"raw_311_noise": str(raw_path)},
                config=config,
                run_id=logger.run_id,
                extra={
                    "all_nta_count": len(df_all),
                    "residential_nta_count": len(df_residential),
                    "join_stats": join_stats,
                    "qa_all": qa_all,
                    "qa_residential": qa_res,
                },
            )
            
            # Print summary
            logger.info("=" * 70)
            logger.info("311 NTA Features Summary:")
            logger.info(f"  All NTAs: {len(df_all)} rows")
            logger.info(f"  Residential NTAs: {len(df_residential)} rows")
            logger.info(f"  Total complaints assigned: {df_all['count_night'].sum():,}")
            logger.info(f"  Assignment rate: {join_stats['assignment_rate']:.1%}")
            logger.info("")
            logger.info("Coverage (All NTAs):")
            logger.info(f"  NTAs with complaints: {qa_all['ntas_with_complaints']}")
            logger.info(f"  NTAs without complaints: {qa_all['ntas_without_complaints']}")
            logger.info(f"  Mean complaints per NTA: {qa_all['mean_complaints_per_nta']:.1f}")
            logger.info("")
            logger.info("Coverage (Residential NTAs):")
            logger.info(f"  NTAs with complaints: {qa_res['ntas_with_complaints']}")
            logger.info(f"  NTAs without complaints: {qa_res['ntas_without_complaints']}")
            logger.info(f"  Mean complaints per NTA: {qa_res['mean_complaints_per_nta']:.1f}")
            logger.info("=" * 70)
            
            logger.info("SUCCESS: Built 311 NTA features")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

