#!/usr/bin/env python3
"""
12_build_temporal_stability.py

Temporal Stability Analysis for NYC Night Signals Atlas.

Analyzes stability of clustering patterns across years (2021, 2022, 2023):
1. CD-level temporal stability:
   - Rebuild features by year
   - Re-run CD typology per year (fixed K)
   - Compute ARI/NMI across years
   - Produce cluster transition tables

2. NTA-level temporal stability:
   - Use residential NTAs only
   - Apply min_cluster_count logic per year
   - Re-run NTA typology per year
   - Compute ARI/NMI across years
   - Identify stable vs volatile NTAs

3. Hotspot persistence:
   - Track overlap of top N hotspot cells by year
   - Identify persistent vs transient hotspots

Outputs:
- data/processed/temporal/cd_clusters_by_year.parquet
- data/processed/temporal/cd_features_by_year.parquet
- data/processed/temporal/nta_clusters_by_year.parquet
- data/processed/temporal/nta_features_by_year.parquet
- data/processed/temporal/cluster_stability_metrics.csv
- data/processed/temporal/cluster_transitions_cd.csv
- data/processed/temporal/cluster_transitions_nta.csv
- data/processed/temporal/hotspot_persistence.csv
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sleep_esi.hashing import write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_df, atomic_write_gdf, read_yaml, read_gdf
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, XWALK_DIR, RAW_DIR, PROCESSED_DIR
from sleep_esi.qa import safe_reproject, assert_cd_labels_present
from sleep_esi.schemas import ensure_boro_cd_dtype, validate_boro_cd
from sleep_esi.time_utils import ensure_nyc_timezone, filter_nighttime
from sleep_esi.joins import spatial_join_points_to_polygons


# =============================================================================
# Constants
# =============================================================================

RAW_311_DIR = RAW_DIR / "311_noise"
ATLAS_DIR = PROCESSED_DIR / "atlas"
TEMPORAL_DIR = PROCESSED_DIR / "temporal"
HOTSPOTS_DIR = PROCESSED_DIR / "hotspots"

# Output paths
OUTPUT_CD_FEATURES_BY_YEAR = TEMPORAL_DIR / "cd_features_by_year.parquet"
OUTPUT_CD_CLUSTERS_BY_YEAR = TEMPORAL_DIR / "cd_clusters_by_year.parquet"
OUTPUT_NTA_FEATURES_BY_YEAR = TEMPORAL_DIR / "nta_features_by_year.parquet"
OUTPUT_NTA_CLUSTERS_BY_YEAR = TEMPORAL_DIR / "nta_clusters_by_year.parquet"
OUTPUT_STABILITY_METRICS = TEMPORAL_DIR / "cluster_stability_metrics.csv"
OUTPUT_TRANSITIONS_CD = TEMPORAL_DIR / "cluster_transitions_cd.csv"
OUTPUT_TRANSITIONS_NTA = TEMPORAL_DIR / "cluster_transitions_nta.csv"
OUTPUT_HOTSPOT_PERSISTENCE = TEMPORAL_DIR / "hotspot_persistence.csv"

YEARS = [2021, 2022, 2023]


# =============================================================================
# Data Loading (adapted from Script 03)
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
    gdf = read_gdf(cd59_path)
    gdf = ensure_boro_cd_dtype(gdf)
    logger.info(f"Loaded CD59: {len(gdf)} community districts")
    return gdf


def load_cd_lookup(logger) -> pd.DataFrame:
    """Load the CD lookup table for labels."""
    lookup_path = GEO_DIR / "cd_lookup.parquet"
    df = pd.read_parquet(lookup_path)
    df = ensure_boro_cd_dtype(df)
    logger.info(f"Loaded CD lookup: {len(df)} entries")
    return df


def load_cd_population(logger) -> pd.DataFrame:
    """Load CD population from tract crosswalk."""
    xwalk_path = XWALK_DIR / "cd_to_tract_weights.parquet"
    xwalk = pd.read_parquet(xwalk_path)
    cd_pop = xwalk.groupby("boro_cd")["tract_pop"].sum().reset_index()
    cd_pop = cd_pop.rename(columns={"tract_pop": "population"})
    cd_pop = ensure_boro_cd_dtype(cd_pop)
    return cd_pop


def load_nta_geo(logger) -> gpd.GeoDataFrame:
    """Load NTA geometries."""
    nta_path = GEO_DIR / "nta.parquet"
    gdf = read_gdf(nta_path)
    logger.info(f"Loaded NTA geometries: {len(gdf)} NTAs")
    return gdf


def load_nta_lookup(logger) -> pd.DataFrame:
    """Load NTA lookup table."""
    lookup_path = GEO_DIR / "nta_lookup.parquet"
    df = pd.read_parquet(lookup_path)
    logger.info(f"Loaded NTA lookup: {len(df)} entries")
    return df


def load_nta_population(logger) -> pd.DataFrame:
    """Load NTA population from crosswalk."""
    xwalk_path = XWALK_DIR / "cd_to_nta_weights.parquet"
    xwalk = pd.read_parquet(xwalk_path)
    # Column names in crosswalk: nta2020, nta_pop
    nta_pop = xwalk.groupby("nta2020")["nta_pop"].sum().reset_index()
    nta_pop = nta_pop.rename(columns={"nta2020": "ntacode", "nta_pop": "population"})
    return nta_pop


# =============================================================================
# 311 Data Preparation
# =============================================================================

def prepare_311_geodataframe(df: pd.DataFrame, logger) -> gpd.GeoDataFrame:
    """Prepare 311 data as GeoDataFrame with valid coordinates and parsed timestamps."""
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


def filter_to_nighttime_year(
    gdf: gpd.GeoDataFrame,
    year: int,
    start_hour: int,
    end_hour: int,
    logger,
) -> gpd.GeoDataFrame:
    """Filter to nighttime hours for a single year."""
    gdf = gdf.copy()
    gdf["year"] = gdf["ts_nyc"].dt.year
    gdf = gdf[gdf["year"] == year]
    
    # Filter to nighttime
    gdf = filter_nighttime(gdf, "ts_nyc", start_hour=start_hour, end_hour=end_hour)
    logger.info(f"  Year {year}: {len(gdf):,} nighttime records")
    
    return gdf


# =============================================================================
# CD Feature Computation (by year)
# =============================================================================

def compute_cd_features_for_year(
    joined: gpd.GeoDataFrame,
    cd59: gpd.GeoDataFrame,
    cd_pop: pd.DataFrame,
    cd_lookup: pd.DataFrame,
    config: dict,
    year: int,
    logger,
) -> pd.DataFrame:
    """Compute CD-level features for a single year."""
    night_bins = config.get("night_bins", {})
    bins_config = night_bins.get("bins", [])
    late_night_hours = night_bins.get("late_night_hours", [1, 2, 3])
    weekend_config = config.get("weekend", {})
    weekend_nights = weekend_config.get("weekend_nights", [4, 5])
    weekday_nights = weekend_config.get("weekday_nights", [0, 1, 2, 3, 6])
    warm_season = config.get("warm_season", {}).get("primary", {}).get("months", [6, 7, 8])
    complaint_types = config.get("noise_311", {}).get("complaint_types", [])
    
    # Add time features
    joined = joined.copy()
    joined["hour"] = joined["ts_nyc"].dt.hour
    joined["dow"] = joined["ts_nyc"].dt.weekday
    joined["month"] = joined["ts_nyc"].dt.month
    
    # Start with all 59 CDs
    df = cd59[["boro_cd"]].copy()
    df = ensure_boro_cd_dtype(df)
    
    # Basic counts
    counts = joined.groupby("boro_cd").size().reset_index(name="count_night")
    counts = ensure_boro_cd_dtype(counts)
    df = df.merge(counts, on="boro_cd", how="left")
    df["count_night"] = df["count_night"].fillna(0).astype("Int64")
    
    # Merge population and area
    df = df.merge(cd_pop, on="boro_cd", how="left")
    cd59_proj = safe_reproject(cd59, 2263, "CD59 for area")
    cd59_proj["area_km2"] = cd59_proj.geometry.area * 0.0929 / 1_000_000
    df = df.merge(cd59_proj[["boro_cd", "area_km2"]], on="boro_cd", how="left")
    
    # Compute rates
    df["rate_per_1k_pop"] = (df["count_night"] / df["population"]) * 1000
    df["rate_per_km2"] = df["count_night"] / df["area_km2"]
    df["rate_per_1k_pop"] = df["rate_per_1k_pop"].replace([np.inf, -np.inf], np.nan)
    df["rate_per_km2"] = df["rate_per_km2"].replace([np.inf, -np.inf], np.nan)
    
    # Complaint type shares
    type_counts = joined.groupby(["boro_cd", "complaint_type"]).size().unstack(fill_value=0)
    type_counts = type_counts.reset_index()
    type_counts = ensure_boro_cd_dtype(type_counts)
    df = df.merge(type_counts, on="boro_cd", how="left")
    
    for ctype in complaint_types:
        col_name = f"share_{ctype.lower().replace(' ', '_').replace('-', '_')}"
        if ctype in df.columns:
            df[col_name] = df[ctype] / df["count_night"]
            df[col_name] = df[col_name].fillna(0)
        else:
            df[col_name] = 0.0
    
    for ctype in complaint_types:
        if ctype in df.columns:
            df = df.drop(columns=[ctype])
    
    # Time-of-night bin shares
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
        
        bin_counts = joined[mask].groupby("boro_cd").size().reset_index(name=f"count_{bin_name}")
        bin_counts = ensure_boro_cd_dtype(bin_counts)
        df = df.merge(bin_counts, on="boro_cd", how="left")
        df[f"count_{bin_name}"] = df[f"count_{bin_name}"].fillna(0)
        df[f"share_{bin_name}"] = df[f"count_{bin_name}"] / df["count_night"]
        df[f"share_{bin_name}"] = df[f"share_{bin_name}"].fillna(0)
    
    # Late-night share
    late_night_mask = joined["hour"].isin(late_night_hours)
    if late_night_mask.any():
        late_counts = joined[late_night_mask].groupby("boro_cd").size().reset_index(name="count_late_night")
        late_counts = ensure_boro_cd_dtype(late_counts)
        df = df.merge(late_counts, on="boro_cd", how="left")
    if "count_late_night" not in df.columns:
        df["count_late_night"] = 0
    df["count_late_night"] = df["count_late_night"].fillna(0).astype(int)
    df["late_night_share"] = df["count_late_night"] / df["count_night"]
    df["late_night_share"] = df["late_night_share"].fillna(0)
    
    # Weekend uplift
    weekend_mask = joined["dow"].isin(weekend_nights)
    if weekend_mask.any():
        weekend_counts = joined[weekend_mask].groupby("boro_cd").size().reset_index(name="count_weekend")
        weekend_counts = ensure_boro_cd_dtype(weekend_counts)
        df = df.merge(weekend_counts, on="boro_cd", how="left")
    if "count_weekend" not in df.columns:
        df["count_weekend"] = 0
    df["count_weekend"] = df["count_weekend"].fillna(0).astype(int)
    
    weekday_mask = joined["dow"].isin(weekday_nights)
    if weekday_mask.any():
        weekday_counts = joined[weekday_mask].groupby("boro_cd").size().reset_index(name="count_weekday")
        weekday_counts = ensure_boro_cd_dtype(weekday_counts)
        df = df.merge(weekday_counts, on="boro_cd", how="left")
    if "count_weekday" not in df.columns:
        df["count_weekday"] = 0
    df["count_weekday"] = df["count_weekday"].fillna(0).astype(int)
    
    df["weekend_rate"] = df["count_weekend"] / 2
    df["weekday_rate"] = df["count_weekday"] / 5
    df["weekend_uplift"] = df["weekend_rate"] / df["weekday_rate"]
    df["weekend_uplift"] = df["weekend_uplift"].replace([np.inf, -np.inf], np.nan)
    
    # Warm season ratio (for single year, compute if we have enough months)
    warm_mask = joined["month"].isin(warm_season)
    if warm_mask.any():
        warm_counts = joined[warm_mask].groupby("boro_cd").size().reset_index(name="count_warm")
        warm_counts = ensure_boro_cd_dtype(warm_counts)
        df = df.merge(warm_counts, on="boro_cd", how="left")
    if "count_warm" not in df.columns:
        df["count_warm"] = 0
    df["count_warm"] = df["count_warm"].fillna(0).astype(int)
    
    cool_mask = ~joined["month"].isin(warm_season)
    if cool_mask.any():
        cool_counts = joined[cool_mask].groupby("boro_cd").size().reset_index(name="count_cool")
        cool_counts = ensure_boro_cd_dtype(cool_counts)
        df = df.merge(cool_counts, on="boro_cd", how="left")
    if "count_cool" not in df.columns:
        df["count_cool"] = 0
    df["count_cool"] = df["count_cool"].fillna(0).astype(int)
    
    df["warm_rate"] = df["count_warm"] / 3
    df["cool_rate"] = df["count_cool"] / 9
    df["warm_season_ratio"] = df["warm_rate"] / df["cool_rate"]
    df["warm_season_ratio"] = df["warm_season_ratio"].replace([np.inf, -np.inf], np.nan)
    
    # Join labels
    df = df.merge(
        cd_lookup[["boro_cd", "borough_name", "cd_label", "cd_short"]],
        on="boro_cd",
        how="left",
    )
    
    # Add year column
    df["year"] = year
    
    # Cleanup
    df = ensure_boro_cd_dtype(df)
    df = df.sort_values("boro_cd").reset_index(drop=True)
    
    return df


# =============================================================================
# NTA Feature Computation (by year)
# =============================================================================

def compute_nta_features_for_year(
    joined: gpd.GeoDataFrame,
    nta_geo: gpd.GeoDataFrame,
    nta_lookup: pd.DataFrame,
    nta_pop: pd.DataFrame,
    config: dict,
    year: int,
    logger,
) -> pd.DataFrame:
    """Compute NTA-level features for a single year (residential only)."""
    night_bins = config.get("night_bins", {})
    bins_config = night_bins.get("bins", [])
    late_night_hours = night_bins.get("late_night_hours", [1, 2, 3])
    weekend_config = config.get("weekend", {})
    weekend_nights = weekend_config.get("weekend_nights", [4, 5])
    weekday_nights = weekend_config.get("weekday_nights", [0, 1, 2, 3, 6])
    warm_season = config.get("warm_season", {}).get("primary", {}).get("months", [6, 7, 8])
    complaint_types = config.get("noise_311", {}).get("complaint_types", [])
    
    # Filter to residential NTAs
    residential_mask = nta_lookup["is_residential"] == True
    residential_ntas = nta_lookup[residential_mask]["ntacode"].tolist()
    nta_geo_res = nta_geo[nta_geo["ntacode"].isin(residential_ntas)].copy()
    
    # Add time features
    joined = joined.copy()
    joined["hour"] = joined["ts_nyc"].dt.hour
    joined["dow"] = joined["ts_nyc"].dt.weekday
    joined["month"] = joined["ts_nyc"].dt.month
    
    # Spatial join to NTAs
    nta_geo_4326 = safe_reproject(nta_geo_res, 4326, "NTA to EPSG:4326")
    joined_nta, _ = spatial_join_points_to_polygons(
        joined,
        nta_geo_4326,
        polygon_id_col="ntacode",
        max_distance=500,
    )
    
    # Start with all residential NTAs
    df = nta_geo_res[["ntacode"]].copy()
    
    # Basic counts
    counts = joined_nta.groupby("ntacode").size().reset_index(name="count_night")
    df = df.merge(counts, on="ntacode", how="left")
    df["count_night"] = df["count_night"].fillna(0).astype("Int64")
    
    # Merge population and area
    df = df.merge(nta_pop, on="ntacode", how="left")
    nta_proj = safe_reproject(nta_geo_res, 2263, "NTA for area")
    nta_proj["area_km2"] = nta_proj.geometry.area * 0.0929 / 1_000_000
    df = df.merge(nta_proj[["ntacode", "area_km2"]], on="ntacode", how="left")
    
    # Compute rates
    df["rate_per_1k_pop"] = (df["count_night"] / df["population"]) * 1000
    df["rate_per_km2"] = df["count_night"] / df["area_km2"]
    df["rate_per_1k_pop"] = df["rate_per_1k_pop"].replace([np.inf, -np.inf], np.nan)
    df["rate_per_km2"] = df["rate_per_km2"].replace([np.inf, -np.inf], np.nan)
    
    # Complaint type shares
    if len(joined_nta) > 0:
        type_counts = joined_nta.groupby(["ntacode", "complaint_type"]).size().unstack(fill_value=0)
        type_counts = type_counts.reset_index()
        df = df.merge(type_counts, on="ntacode", how="left")
    
    for ctype in complaint_types:
        col_name = f"share_{ctype.lower().replace(' ', '_').replace('-', '_')}"
        if ctype in df.columns:
            df[col_name] = df[ctype] / df["count_night"]
            df[col_name] = df[col_name].fillna(0)
        else:
            df[col_name] = 0.0
    
    for ctype in complaint_types:
        if ctype in df.columns:
            df = df.drop(columns=[ctype])
    
    # Time-of-night bin shares
    for bin_def in bins_config:
        bin_name = bin_def["name"]
        bin_start = bin_def["start"]
        bin_end = bin_def["end"]
        
        if bin_start < bin_end:
            mask = (joined_nta["hour"] >= bin_start) & (joined_nta["hour"] < bin_end)
        else:
            if bin_end == 24:
                mask = joined_nta["hour"] >= bin_start
            else:
                mask = (joined_nta["hour"] >= bin_start) | (joined_nta["hour"] < bin_end)
        
        if mask.any():
            bin_counts = joined_nta[mask].groupby("ntacode").size().reset_index(name=f"count_{bin_name}")
            df = df.merge(bin_counts, on="ntacode", how="left")
        if f"count_{bin_name}" not in df.columns:
            df[f"count_{bin_name}"] = 0
        df[f"count_{bin_name}"] = df[f"count_{bin_name}"].fillna(0)
        df[f"share_{bin_name}"] = df[f"count_{bin_name}"] / df["count_night"]
        df[f"share_{bin_name}"] = df[f"share_{bin_name}"].fillna(0)
    
    # Late-night share
    late_night_mask = joined_nta["hour"].isin(late_night_hours)
    if late_night_mask.any():
        late_counts = joined_nta[late_night_mask].groupby("ntacode").size().reset_index(name="count_late_night")
        df = df.merge(late_counts, on="ntacode", how="left")
    if "count_late_night" not in df.columns:
        df["count_late_night"] = 0
    df["count_late_night"] = df["count_late_night"].fillna(0).astype(int)
    df["late_night_share"] = df["count_late_night"] / df["count_night"]
    df["late_night_share"] = df["late_night_share"].fillna(0)
    
    # Weekend uplift
    weekend_mask = joined_nta["dow"].isin(weekend_nights)
    if weekend_mask.any():
        weekend_counts = joined_nta[weekend_mask].groupby("ntacode").size().reset_index(name="count_weekend")
        df = df.merge(weekend_counts, on="ntacode", how="left")
    if "count_weekend" not in df.columns:
        df["count_weekend"] = 0
    df["count_weekend"] = df["count_weekend"].fillna(0).astype(int)
    
    weekday_mask = joined_nta["dow"].isin(weekday_nights)
    if weekday_mask.any():
        weekday_counts = joined_nta[weekday_mask].groupby("ntacode").size().reset_index(name="count_weekday")
        df = df.merge(weekday_counts, on="ntacode", how="left")
    if "count_weekday" not in df.columns:
        df["count_weekday"] = 0
    df["count_weekday"] = df["count_weekday"].fillna(0).astype(int)
    
    df["weekend_rate"] = df["count_weekend"] / 2
    df["weekday_rate"] = df["count_weekday"] / 5
    df["weekend_uplift"] = df["weekend_rate"] / df["weekday_rate"]
    df["weekend_uplift"] = df["weekend_uplift"].replace([np.inf, -np.inf], np.nan)
    
    # Warm season ratio
    warm_mask = joined_nta["month"].isin(warm_season)
    if warm_mask.any():
        warm_counts = joined_nta[warm_mask].groupby("ntacode").size().reset_index(name="count_warm")
        df = df.merge(warm_counts, on="ntacode", how="left")
    if "count_warm" not in df.columns:
        df["count_warm"] = 0
    df["count_warm"] = df["count_warm"].fillna(0).astype(int)
    
    cool_mask = ~joined_nta["month"].isin(warm_season)
    if cool_mask.any():
        cool_counts = joined_nta[cool_mask].groupby("ntacode").size().reset_index(name="count_cool")
        df = df.merge(cool_counts, on="ntacode", how="left")
    if "count_cool" not in df.columns:
        df["count_cool"] = 0
    df["count_cool"] = df["count_cool"].fillna(0).astype(int)
    
    df["warm_rate"] = df["count_warm"] / 3
    df["cool_rate"] = df["count_cool"] / 9
    df["warm_season_ratio"] = df["warm_rate"] / df["cool_rate"]
    df["warm_season_ratio"] = df["warm_season_ratio"].replace([np.inf, -np.inf], np.nan)
    
    # Join lookup
    df = df.merge(
        nta_lookup[["ntacode", "nta_name", "borough_name", "is_residential", "ntatype_label"]],
        on="ntacode",
        how="left",
    )
    
    # Add year column
    df["year"] = year
    
    # Cleanup
    df = df.sort_values("ntacode").reset_index(drop=True)
    
    return df


# =============================================================================
# Clustering
# =============================================================================

def run_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    random_seed: int,
    id_col: str,
    logger,
) -> Tuple[np.ndarray, float]:
    """Run K-means clustering and return labels + silhouette score."""
    from sklearn.metrics import silhouette_score as sil_score
    
    # Filter to columns that exist
    valid_cols = [c for c in feature_cols if c in df.columns]
    
    # Extract feature matrix
    X_raw = df[valid_cols].copy()
    X_raw = X_raw.fillna(X_raw.median())
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # Cluster
    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X)
    
    # Compute silhouette
    sil = sil_score(X, labels)
    
    return labels, sil


def run_clustering_with_eligibility(
    df: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    min_cluster_count: int,
    random_seed: int,
    id_col: str,
    logger,
) -> Tuple[np.ndarray, float, int]:
    """
    Run K-means with eligibility guardrail.
    Returns labels (-1 for ineligible), silhouette, and n_eligible.
    """
    from sklearn.metrics import silhouette_score as sil_score
    
    # Eligibility mask
    eligible_mask = df["count_night"] >= min_cluster_count
    n_eligible = eligible_mask.sum()
    
    if n_eligible < k:
        logger.warning(f"Only {n_eligible} eligible units, less than K={k}")
        # Return all as cluster 0 with low silhouette
        labels = np.zeros(len(df), dtype=int)
        labels[~eligible_mask] = -1
        return labels, 0.0, n_eligible
    
    df_eligible = df[eligible_mask].copy()
    
    # Filter to columns that exist
    valid_cols = [c for c in feature_cols if c in df_eligible.columns]
    
    # Extract feature matrix
    X_raw = df_eligible[valid_cols].copy()
    X_raw = X_raw.fillna(X_raw.median())
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    # Cluster
    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10, max_iter=300)
    eligible_labels = kmeans.fit_predict(X)
    
    # Compute silhouette
    sil = sil_score(X, eligible_labels)
    
    # Build full labels array
    labels = np.full(len(df), -1)
    labels[eligible_mask] = eligible_labels
    
    return labels, sil, n_eligible


# =============================================================================
# Stability Metrics
# =============================================================================

def compute_cluster_stability(
    df_clusters: pd.DataFrame,
    id_col: str,
    logger,
) -> pd.DataFrame:
    """
    Compute ARI and NMI between each pair of years.
    """
    years = sorted(df_clusters["year"].unique())
    
    results = []
    
    for i, year1 in enumerate(years):
        for year2 in years[i+1:]:
            df1 = df_clusters[df_clusters["year"] == year1].set_index(id_col)
            df2 = df_clusters[df_clusters["year"] == year2].set_index(id_col)
            
            # Align on common IDs
            common_ids = df1.index.intersection(df2.index)
            
            # Exclude ineligible (-1) from comparison
            mask1 = df1.loc[common_ids, "cluster_id"] != -1
            mask2 = df2.loc[common_ids, "cluster_id"] != -1
            valid_mask = mask1 & mask2
            
            if valid_mask.sum() < 10:
                logger.warning(f"Too few valid units for {year1} vs {year2}")
                continue
            
            labels1 = df1.loc[common_ids[valid_mask], "cluster_id"].values
            labels2 = df2.loc[common_ids[valid_mask], "cluster_id"].values
            
            ari = adjusted_rand_score(labels1, labels2)
            nmi = normalized_mutual_info_score(labels1, labels2)
            
            results.append({
                "year_1": year1,
                "year_2": year2,
                "ari": ari,
                "nmi": nmi,
                "n_common": int(valid_mask.sum()),
            })
            
            logger.info(f"  {year1} vs {year2}: ARI={ari:.4f}, NMI={nmi:.4f} (n={valid_mask.sum()})")
    
    return pd.DataFrame(results)


def compute_cluster_transitions(
    df_clusters: pd.DataFrame,
    id_col: str,
    label_col: str,
    logger,
) -> pd.DataFrame:
    """
    Compute per-unit cluster transitions across years.
    """
    years = sorted(df_clusters["year"].unique())
    
    # Pivot to wide format
    pivot = df_clusters.pivot(index=id_col, columns="year", values="cluster_id")
    
    # Add label info
    label_info = df_clusters[[id_col, label_col]].drop_duplicates().set_index(id_col)
    pivot = pivot.join(label_info)
    
    # Compute transition columns
    for i, (year1, year2) in enumerate(zip(years[:-1], years[1:])):
        col_name = f"transition_{year1}_{year2}"
        pivot[col_name] = pivot[year1].astype(str) + " -> " + pivot[year2].astype(str)
    
    # Compute overall volatility
    pivot["n_unique_clusters"] = pivot[years].nunique(axis=1)
    pivot["is_stable"] = pivot["n_unique_clusters"] == 1
    
    # Reset index
    pivot = pivot.reset_index()
    
    return pivot


# =============================================================================
# Hotspot Persistence
# =============================================================================

def compute_hotspot_persistence(
    gdf_311: gpd.GeoDataFrame,
    config: dict,
    logger,
) -> pd.DataFrame:
    """
    Compute hotspot cell overlap across years.
    """
    from collections import defaultdict
    
    hotspot_config = config.get("hotspots", {})
    cell_size_ft = hotspot_config.get("cell_size_ft", 820)
    top_n = hotspot_config.get("top_n_citywide", 100)
    
    # Convert to projected CRS
    gdf_proj = safe_reproject(gdf_311, 2263, "311 points for grid")
    
    # Get bounds
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    
    # Create grid cell assignment
    gdf_proj["cell_x"] = ((gdf_proj.geometry.x - minx) // cell_size_ft).astype(int)
    gdf_proj["cell_y"] = ((gdf_proj.geometry.y - miny) // cell_size_ft).astype(int)
    gdf_proj["cell_id"] = gdf_proj["cell_x"].astype(str) + "_" + gdf_proj["cell_y"].astype(str)
    
    # Get year
    gdf_proj["year"] = gdf_proj["ts_nyc"].dt.year
    
    years = sorted(gdf_proj["year"].unique())
    
    # Get top N cells per year
    top_cells_by_year = {}
    for year in years:
        year_data = gdf_proj[gdf_proj["year"] == year]
        cell_counts = year_data.groupby("cell_id").size().sort_values(ascending=False)
        top_cells_by_year[year] = set(cell_counts.head(top_n).index)
    
    # Compute overlaps
    results = []
    for i, year1 in enumerate(years):
        for year2 in years[i+1:]:
            cells1 = top_cells_by_year[year1]
            cells2 = top_cells_by_year[year2]
            
            intersection = len(cells1 & cells2)
            union = len(cells1 | cells2)
            jaccard = intersection / union if union > 0 else 0
            
            results.append({
                "year_1": year1,
                "year_2": year2,
                "overlap_count": intersection,
                "jaccard": jaccard,
                "top_n": top_n,
            })
            
            logger.info(f"  Top {top_n} hotspot overlap {year1} vs {year2}: {intersection} cells (Jaccard={jaccard:.4f})")
    
    # Identify persistent cells (in all years)
    all_years_cells = set.intersection(*top_cells_by_year.values())
    logger.info(f"Persistent hotspot cells (in all {len(years)} years): {len(all_years_cells)}")
    
    # Add summary row
    results.append({
        "year_1": "all",
        "year_2": "all",
        "overlap_count": len(all_years_cells),
        "jaccard": len(all_years_cells) / top_n if top_n > 0 else 0,
        "top_n": top_n,
    })
    
    return pd.DataFrame(results)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("12_build_temporal_stability") as logger:
        logger.info("Starting 12_build_temporal_stability.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        nighttime_config = config.get("nighttime", {}).get("primary", {})
        start_hour = nighttime_config.get("start_hour", 22)
        end_hour = nighttime_config.get("end_hour", 7)
        
        typology_config = config.get("typology", {})
        cd_feature_cols = typology_config.get("features", [])
        cd_k = 9  # Fixed K from pooled clustering
        
        nta_typology_config = config.get("nta_typology", {})
        nta_feature_cols = nta_typology_config.get("features", [])
        nta_k = 9  # Fixed K from pooled clustering
        min_cluster_count = nta_typology_config.get("min_cluster_count", 200)
        # Scale down for single year
        min_cluster_count_year = max(50, min_cluster_count // 3)
        
        random_seed = config.get("random_seeds", {}).get("clustering", 12345)
        
        logger.info(f"Years to analyze: {YEARS}")
        logger.info(f"CD K (fixed): {cd_k}")
        logger.info(f"NTA K (fixed): {nta_k}")
        logger.info(f"NTA min_cluster_count per year: {min_cluster_count_year}")
        logger.info(f"Random seed: {random_seed}")
        
        try:
            # Ensure output directory
            TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
            
            # Load base data
            df_311, raw_path = load_raw_311(logger)
            cd59 = load_cd59(logger)
            cd_lookup = load_cd_lookup(logger)
            cd_pop = load_cd_population(logger)
            nta_geo = load_nta_geo(logger)
            nta_lookup = load_nta_lookup(logger)
            nta_pop = load_nta_population(logger)
            
            # Prepare 311 GeoDataFrame
            logger.info("Preparing 311 points...")
            gdf_311 = prepare_311_geodataframe(df_311, logger)
            logger.info(f"Prepared {len(gdf_311):,} 311 points")
            
            # ==================================================================
            # 1. CD-level temporal analysis
            # ==================================================================
            logger.info("=" * 70)
            logger.info("CD-LEVEL TEMPORAL ANALYSIS")
            logger.info("=" * 70)
            
            cd_features_all = []
            cd_clusters_all = []
            
            for year in YEARS:
                logger.info(f"\n--- Processing CD features for {year} ---")
                
                # Filter to year and nighttime
                gdf_year = filter_to_nighttime_year(gdf_311, year, start_hour, end_hour, logger)
                
                # Spatial join to CDs
                cd59_4326 = safe_reproject(cd59, 4326, "CD59 to 4326")
                joined, _ = spatial_join_points_to_polygons(
                    gdf_year,
                    cd59_4326,
                    polygon_id_col="boro_cd",
                    max_distance=500,
                )
                joined = ensure_boro_cd_dtype(joined)
                
                # Compute features
                df_features = compute_cd_features_for_year(
                    joined, cd59, cd_pop, cd_lookup, config, year, logger
                )
                cd_features_all.append(df_features)
                
                # Run clustering
                labels, sil = run_clustering(
                    df_features, cd_feature_cols, cd_k, random_seed, "boro_cd", logger
                )
                
                df_clusters = df_features[["boro_cd", "cd_label", "cd_short", "year"]].copy()
                df_clusters["cluster_id"] = labels
                df_clusters["silhouette"] = sil
                cd_clusters_all.append(df_clusters)
                
                logger.info(f"  Year {year}: silhouette={sil:.4f}")
            
            # Combine CD results
            df_cd_features = pd.concat(cd_features_all, ignore_index=True)
            df_cd_clusters = pd.concat(cd_clusters_all, ignore_index=True)
            
            # Compute CD stability
            logger.info("\nComputing CD cluster stability...")
            df_cd_stability = compute_cluster_stability(df_cd_clusters, "boro_cd", logger)
            
            # Compute CD transitions
            logger.info("\nComputing CD cluster transitions...")
            df_cd_transitions = compute_cluster_transitions(df_cd_clusters, "boro_cd", "cd_label", logger)
            
            # ==================================================================
            # 2. NTA-level temporal analysis
            # ==================================================================
            logger.info("\n" + "=" * 70)
            logger.info("NTA-LEVEL TEMPORAL ANALYSIS")
            logger.info("=" * 70)
            
            nta_features_all = []
            nta_clusters_all = []
            
            for year in YEARS:
                logger.info(f"\n--- Processing NTA features for {year} ---")
                
                # Filter to year and nighttime
                gdf_year = filter_to_nighttime_year(gdf_311, year, start_hour, end_hour, logger)
                
                # Compute NTA features
                df_features = compute_nta_features_for_year(
                    gdf_year, nta_geo, nta_lookup, nta_pop, config, year, logger
                )
                nta_features_all.append(df_features)
                
                # Run clustering with eligibility
                labels, sil, n_eligible = run_clustering_with_eligibility(
                    df_features, nta_feature_cols, nta_k, min_cluster_count_year,
                    random_seed, "ntacode", logger
                )
                
                df_clusters = df_features[["ntacode", "nta_name", "year"]].copy()
                df_clusters["cluster_id"] = labels
                df_clusters["silhouette"] = sil
                df_clusters["n_eligible"] = n_eligible
                nta_clusters_all.append(df_clusters)
                
                n_low_vol = (labels == -1).sum()
                logger.info(f"  Year {year}: silhouette={sil:.4f}, eligible={n_eligible}, low_volume={n_low_vol}")
            
            # Combine NTA results
            df_nta_features = pd.concat(nta_features_all, ignore_index=True)
            df_nta_clusters = pd.concat(nta_clusters_all, ignore_index=True)
            
            # Compute NTA stability
            logger.info("\nComputing NTA cluster stability...")
            df_nta_stability = compute_cluster_stability(df_nta_clusters, "ntacode", logger)
            
            # Compute NTA transitions
            logger.info("\nComputing NTA cluster transitions...")
            df_nta_transitions = compute_cluster_transitions(df_nta_clusters, "ntacode", "nta_name", logger)
            
            # ==================================================================
            # 3. Hotspot persistence
            # ==================================================================
            logger.info("\n" + "=" * 70)
            logger.info("HOTSPOT PERSISTENCE ANALYSIS")
            logger.info("=" * 70)
            
            # Filter to nighttime for all years
            gdf_night = gdf_311.copy()
            gdf_night["year"] = gdf_night["ts_nyc"].dt.year
            gdf_night = gdf_night[gdf_night["year"].isin(YEARS)]
            gdf_night = filter_nighttime(gdf_night, "ts_nyc", start_hour=start_hour, end_hour=end_hour)
            
            df_hotspot_persistence = compute_hotspot_persistence(gdf_night, config, logger)
            
            # ==================================================================
            # Combine stability metrics
            # ==================================================================
            df_cd_stability["level"] = "CD"
            df_nta_stability["level"] = "NTA"
            df_stability = pd.concat([df_cd_stability, df_nta_stability], ignore_index=True)
            
            # ==================================================================
            # Write outputs
            # ==================================================================
            logger.info("\n" + "=" * 70)
            logger.info("WRITING OUTPUTS")
            logger.info("=" * 70)
            
            # CD outputs
            atomic_write_df(df_cd_features, OUTPUT_CD_FEATURES_BY_YEAR)
            logger.info(f"Wrote: {OUTPUT_CD_FEATURES_BY_YEAR}")
            
            atomic_write_df(df_cd_clusters, OUTPUT_CD_CLUSTERS_BY_YEAR)
            logger.info(f"Wrote: {OUTPUT_CD_CLUSTERS_BY_YEAR}")
            
            df_cd_transitions.to_csv(OUTPUT_TRANSITIONS_CD, index=False)
            logger.info(f"Wrote: {OUTPUT_TRANSITIONS_CD}")
            
            # NTA outputs
            atomic_write_df(df_nta_features, OUTPUT_NTA_FEATURES_BY_YEAR)
            logger.info(f"Wrote: {OUTPUT_NTA_FEATURES_BY_YEAR}")
            
            atomic_write_df(df_nta_clusters, OUTPUT_NTA_CLUSTERS_BY_YEAR)
            logger.info(f"Wrote: {OUTPUT_NTA_CLUSTERS_BY_YEAR}")
            
            df_nta_transitions.to_csv(OUTPUT_TRANSITIONS_NTA, index=False)
            logger.info(f"Wrote: {OUTPUT_TRANSITIONS_NTA}")
            
            # Stability metrics
            df_stability.to_csv(OUTPUT_STABILITY_METRICS, index=False)
            logger.info(f"Wrote: {OUTPUT_STABILITY_METRICS}")
            
            # Hotspot persistence
            df_hotspot_persistence.to_csv(OUTPUT_HOTSPOT_PERSISTENCE, index=False)
            logger.info(f"Wrote: {OUTPUT_HOTSPOT_PERSISTENCE}")
            
            # Log outputs
            logger.log_outputs({
                "cd_features_by_year": str(OUTPUT_CD_FEATURES_BY_YEAR),
                "cd_clusters_by_year": str(OUTPUT_CD_CLUSTERS_BY_YEAR),
                "nta_features_by_year": str(OUTPUT_NTA_FEATURES_BY_YEAR),
                "nta_clusters_by_year": str(OUTPUT_NTA_CLUSTERS_BY_YEAR),
                "stability_metrics": str(OUTPUT_STABILITY_METRICS),
                "cd_transitions": str(OUTPUT_TRANSITIONS_CD),
                "nta_transitions": str(OUTPUT_TRANSITIONS_NTA),
                "hotspot_persistence": str(OUTPUT_HOTSPOT_PERSISTENCE),
            })
            
            # Log metrics
            cd_ari_mean = df_cd_stability["ari"].mean() if len(df_cd_stability) > 0 else None
            cd_nmi_mean = df_cd_stability["nmi"].mean() if len(df_cd_stability) > 0 else None
            nta_ari_mean = df_nta_stability["ari"].mean() if len(df_nta_stability) > 0 else None
            nta_nmi_mean = df_nta_stability["nmi"].mean() if len(df_nta_stability) > 0 else None
            
            # Count stable units
            n_stable_cds = df_cd_transitions["is_stable"].sum() if "is_stable" in df_cd_transitions.columns else 0
            n_stable_ntas = df_nta_transitions["is_stable"].sum() if "is_stable" in df_nta_transitions.columns else 0
            
            logger.log_metrics({
                "cd_ari_mean": float(cd_ari_mean) if cd_ari_mean else None,
                "cd_nmi_mean": float(cd_nmi_mean) if cd_nmi_mean else None,
                "nta_ari_mean": float(nta_ari_mean) if nta_ari_mean else None,
                "nta_nmi_mean": float(nta_nmi_mean) if nta_nmi_mean else None,
                "n_stable_cds": int(n_stable_cds),
                "n_stable_ntas": int(n_stable_ntas),
                "years_analyzed": YEARS,
            })
            
            # Write metadata
            write_metadata_sidecar(
                output_path=OUTPUT_STABILITY_METRICS,
                inputs={"raw_311": str(raw_path)},
                config=config,
                run_id=logger.run_id,
                extra={
                    "years_analyzed": YEARS,
                    "cd_k": cd_k,
                    "nta_k": nta_k,
                    "min_cluster_count_year": min_cluster_count_year,
                    "random_seed": random_seed,
                    "cd_stability": df_cd_stability.to_dict(orient="records"),
                    "nta_stability": df_nta_stability.to_dict(orient="records"),
                    "n_stable_cds": int(n_stable_cds),
                    "n_stable_ntas": int(n_stable_ntas),
                },
            )
            
            # Print summary
            logger.info("\n" + "=" * 70)
            logger.info("TEMPORAL STABILITY SUMMARY")
            logger.info("=" * 70)
            
            logger.info("\nCD-Level Stability:")
            for _, row in df_cd_stability.iterrows():
                logger.info(f"  {int(row['year_1'])} vs {int(row['year_2'])}: ARI={row['ari']:.4f}, NMI={row['nmi']:.4f}")
            logger.info(f"  Mean ARI: {cd_ari_mean:.4f}" if cd_ari_mean else "  No valid comparisons")
            logger.info(f"  Stable CDs (same cluster all 3 years): {n_stable_cds} / 59")
            
            logger.info("\nNTA-Level Stability:")
            for _, row in df_nta_stability.iterrows():
                logger.info(f"  {int(row['year_1'])} vs {int(row['year_2'])}: ARI={row['ari']:.4f}, NMI={row['nmi']:.4f}")
            logger.info(f"  Mean ARI: {nta_ari_mean:.4f}" if nta_ari_mean else "  No valid comparisons")
            logger.info(f"  Stable NTAs (same cluster all 3 years): {n_stable_ntas} / 197")
            
            logger.info("\nHotspot Persistence:")
            for _, row in df_hotspot_persistence.iterrows():
                if row["year_1"] != "all":
                    logger.info(f"  Top 100 overlap {int(row['year_1'])} vs {int(row['year_2'])}: {int(row['overlap_count'])} cells (Jaccard={row['jaccard']:.4f})")
                else:
                    logger.info(f"  Persistent across all years: {int(row['overlap_count'])} cells")
            
            logger.info("\n" + "=" * 70)
            logger.info("SUCCESS: Built temporal stability analysis")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

