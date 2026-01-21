#!/usr/bin/env python3
"""
08a_build_stability_assets.py

Build Temporal Stability Assets for NYC Night Signals Atlas.

Creates CD-level and NTA-level stability outputs:
1) CD stability table + map layer (59 rows)
2) NTA stability table + map layer (197 residential rows)
3) Hotspot persistence layers (privacy-safe)

Stability classification based on entropy:
- "Structural": entropy <= threshold (same cluster pattern)
- "Semi-structural": intermediate entropy
- "Episodic": entropy >= threshold (volatile)

Outputs:
- data/processed/reports/cd_stability.parquet/csv/geojson
- data/processed/reports/nta_stability_residential.parquet/csv/geojson
- data/processed/reports/hotspot_persistent_all3.geojson
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from sleep_esi.hashing import write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_df, atomic_write_gdf, read_yaml, read_gdf
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, PROCESSED_DIR
from sleep_esi.schemas import ensure_boro_cd_dtype


# =============================================================================
# Constants
# =============================================================================

TEMPORAL_DIR = PROCESSED_DIR / "temporal"
REPORTS_DIR = PROCESSED_DIR / "reports"
HOTSPOTS_DIR = PROCESSED_DIR / "hotspots"

# Inputs
INPUT_CD_CLUSTERS = TEMPORAL_DIR / "cd_clusters_by_year.parquet"
INPUT_NTA_CLUSTERS = TEMPORAL_DIR / "nta_clusters_by_year.parquet"
INPUT_HOTSPOT_CELLS = HOTSPOTS_DIR / "hotspot_cells.parquet"
INPUT_HOTSPOT_INVESTIGATION = HOTSPOTS_DIR / "hotspot_investigation_top_cells.parquet"

# Outputs
OUTPUT_CD_STABILITY = REPORTS_DIR / "cd_stability.parquet"
OUTPUT_CD_STABILITY_CSV = REPORTS_DIR / "cd_stability.csv"
OUTPUT_CD_STABILITY_GEOJSON = REPORTS_DIR / "cd_stability.geojson"

OUTPUT_NTA_STABILITY = REPORTS_DIR / "nta_stability_residential.parquet"
OUTPUT_NTA_STABILITY_CSV = REPORTS_DIR / "nta_stability_residential.csv"
OUTPUT_NTA_STABILITY_GEOJSON = REPORTS_DIR / "nta_stability_residential.geojson"

OUTPUT_HOTSPOT_PERSISTENT = REPORTS_DIR / "hotspot_persistent_all3.geojson"

YEARS = [2021, 2022, 2023]


# =============================================================================
# Entropy Calculation
# =============================================================================

def compute_cluster_entropy(cluster_ids: List[int]) -> float:
    """
    Compute normalized entropy of cluster assignments across years.
    
    Returns value in [0, 1]:
    - 0 = same cluster all years (perfectly stable)
    - 1 = different cluster each year (maximally volatile)
    
    Uses Shannon entropy normalized by log2(n_years).
    """
    if not cluster_ids:
        return 0.0
    
    # Filter out -1 (low volume) for entropy calculation
    valid_ids = [c for c in cluster_ids if c >= 0]
    
    if len(valid_ids) <= 1:
        return 0.0
    
    # Count frequencies
    from collections import Counter
    counts = Counter(valid_ids)
    n = len(valid_ids)
    
    # Compute entropy
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / n
            entropy -= p * np.log2(p)
    
    # Normalize by max possible entropy (log2 of number of observations)
    max_entropy = np.log2(n)
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0
    
    return round(normalized_entropy, 4)


def classify_stability(
    entropy_score: float,
    transition_count: int,
    threshold_structural: float,
    threshold_episodic: float,
) -> str:
    """
    Classify stability based on entropy score.
    
    Returns: "Structural", "Semi-structural", or "Episodic"
    """
    if entropy_score <= threshold_structural:
        return "Structural"
    elif entropy_score >= threshold_episodic:
        return "Episodic"
    else:
        return "Semi-structural"


# =============================================================================
# CD Stability Processing
# =============================================================================

def build_cd_stability(
    df_clusters: pd.DataFrame,
    cd59: gpd.GeoDataFrame,
    cd_lookup: pd.DataFrame,
    threshold_structural: float,
    threshold_episodic: float,
    logger,
) -> pd.DataFrame:
    """
    Build CD-level stability table.
    """
    logger.info("Building CD stability table...")
    
    # Pivot to wide format
    pivot = df_clusters.pivot(index="boro_cd", columns="year", values="cluster_id")
    pivot.columns = [f"cluster_id_{year}" for year in pivot.columns]
    pivot = pivot.reset_index()
    pivot = ensure_boro_cd_dtype(pivot)
    
    # Compute stability metrics
    year_cols = [f"cluster_id_{year}" for year in YEARS]
    
    # Transition count (# distinct clusters)
    pivot["transition_count"] = pivot[year_cols].nunique(axis=1)
    
    # Stability flags
    pivot["is_stable_all_3"] = pivot["transition_count"] == 1
    
    # is_stable_2_of_3: at least 2 years have the same cluster
    def has_2_of_3_stable(row):
        counts = row[year_cols].value_counts()
        return counts.max() >= 2
    
    pivot["is_stable_2_of_3"] = pivot.apply(has_2_of_3_stable, axis=1)
    
    # Entropy score
    def calc_entropy(row):
        cluster_ids = [row[col] for col in year_cols]
        return compute_cluster_entropy(cluster_ids)
    
    pivot["entropy_score"] = pivot.apply(calc_entropy, axis=1)
    
    # Stability class
    pivot["stability_class"] = pivot.apply(
        lambda row: classify_stability(
            row["entropy_score"],
            row["transition_count"],
            threshold_structural,
            threshold_episodic,
        ),
        axis=1,
    )
    
    # Join labels
    pivot = pivot.merge(
        cd_lookup[["boro_cd", "cd_label", "cd_short", "borough_name"]],
        on="boro_cd",
        how="left",
    )
    
    # Sort
    pivot = pivot.sort_values("boro_cd").reset_index(drop=True)
    
    # Log summary
    stability_counts = pivot["stability_class"].value_counts()
    logger.info(f"CD stability classification:")
    for cls, count in stability_counts.items():
        logger.info(f"  {cls}: {count}")
    
    logger.info(f"CDs stable all 3 years: {pivot['is_stable_all_3'].sum()}")
    logger.info(f"CDs stable 2 of 3 years: {pivot['is_stable_2_of_3'].sum()}")
    logger.info(f"Mean entropy: {pivot['entropy_score'].mean():.4f}")
    
    return pivot


# =============================================================================
# NTA Stability Processing
# =============================================================================

def build_nta_stability(
    df_clusters: pd.DataFrame,
    nta_geo: gpd.GeoDataFrame,
    nta_lookup: pd.DataFrame,
    threshold_structural: float,
    threshold_episodic: float,
    logger,
) -> pd.DataFrame:
    """
    Build NTA-level stability table (residential only).
    """
    logger.info("Building NTA stability table...")
    
    # Filter to residential NTAs
    residential_ntas = nta_lookup[nta_lookup["is_residential"] == True]["ntacode"].tolist()
    df_clusters = df_clusters[df_clusters["ntacode"].isin(residential_ntas)].copy()
    
    # Pivot to wide format
    pivot = df_clusters.pivot(index="ntacode", columns="year", values="cluster_id")
    pivot.columns = [f"cluster_id_{year}" for year in pivot.columns]
    pivot = pivot.reset_index()
    
    # Compute stability metrics
    year_cols = [f"cluster_id_{year}" for year in YEARS]
    
    # Transition count (# distinct clusters, excluding -1)
    def count_transitions_excl_lowvol(row):
        valid = [row[col] for col in year_cols if row[col] >= 0]
        if not valid:
            return 0
        return len(set(valid))
    
    pivot["transition_count"] = pivot.apply(count_transitions_excl_lowvol, axis=1)
    
    # Stability flags
    pivot["is_stable_all_3"] = pivot["transition_count"] == 1
    
    # is_stable_2_of_3
    def has_2_of_3_stable_nta(row):
        valid_clusters = [row[col] for col in year_cols if row[col] >= 0]
        if len(valid_clusters) < 2:
            return False
        from collections import Counter
        counts = Counter(valid_clusters)
        return counts.most_common(1)[0][1] >= 2
    
    pivot["is_stable_2_of_3"] = pivot.apply(has_2_of_3_stable_nta, axis=1)
    
    # Count low volume years
    def count_low_vol(row):
        return sum(1 for col in year_cols if row[col] == -1)
    
    pivot["low_volume_years"] = pivot.apply(count_low_vol, axis=1)
    
    # Entropy score
    def calc_entropy_nta(row):
        cluster_ids = [row[col] for col in year_cols]
        return compute_cluster_entropy(cluster_ids)
    
    pivot["entropy_score"] = pivot.apply(calc_entropy_nta, axis=1)
    
    # Stability class
    pivot["stability_class"] = pivot.apply(
        lambda row: classify_stability(
            row["entropy_score"],
            row["transition_count"],
            threshold_structural,
            threshold_episodic,
        ),
        axis=1,
    )
    
    # Join lookup
    pivot = pivot.merge(
        nta_lookup[["ntacode", "nta_name", "borough_name", "is_residential", "ntatype_label"]],
        on="ntacode",
        how="left",
    )
    
    # Sort
    pivot = pivot.sort_values("ntacode").reset_index(drop=True)
    
    # Log summary
    stability_counts = pivot["stability_class"].value_counts()
    logger.info(f"NTA stability classification:")
    for cls, count in stability_counts.items():
        logger.info(f"  {cls}: {count}")
    
    logger.info(f"NTAs stable all 3 years: {pivot['is_stable_all_3'].sum()}")
    logger.info(f"NTAs stable 2 of 3 years: {pivot['is_stable_2_of_3'].sum()}")
    logger.info(f"NTAs with low-volume years: {(pivot['low_volume_years'] > 0).sum()}")
    logger.info(f"Mean entropy: {pivot['entropy_score'].mean():.4f}")
    
    return pivot


# =============================================================================
# Hotspot Persistence Processing
# =============================================================================

def build_hotspot_persistence(
    config: dict,
    logger,
) -> gpd.GeoDataFrame:
    """
    Build hotspot persistence layer (cells present in top N all 3 years).
    Privacy-safe: no raw addresses included.
    """
    logger.info("Building hotspot persistence layer...")
    
    hotspot_config = config.get("hotspots", {})
    cell_size_ft = hotspot_config.get("cell_size_ft", 820)
    top_n = hotspot_config.get("top_n_citywide", 100)
    
    # Load hotspot cells (from Script 06)
    if not INPUT_HOTSPOT_CELLS.exists():
        logger.warning(f"Hotspot cells not found: {INPUT_HOTSPOT_CELLS}")
        return None
    
    df_cells = pd.read_parquet(INPUT_HOTSPOT_CELLS)
    logger.info(f"Loaded {len(df_cells)} hotspot cells")
    
    # Load investigation data for artifact flags
    if INPUT_HOTSPOT_INVESTIGATION.exists():
        df_investigation = pd.read_parquet(INPUT_HOTSPOT_INVESTIGATION)
        artifact_flags = df_investigation[["cell_id", "is_repeat_location_dominant", "is_suspected_artifact", "top_latlon_share"]].copy()
        logger.info(f"Loaded {len(df_investigation)} investigation records")
    else:
        artifact_flags = None
        logger.warning("Investigation data not found, artifact flags will be null")
    
    # Re-compute per-year top N cells
    # Load raw 311 data and compute per-year cell assignments
    from sleep_esi.paths import RAW_DIR
    from sleep_esi.time_utils import ensure_nyc_timezone, filter_nighttime
    from sleep_esi.qa import safe_reproject
    
    raw_311_dir = RAW_DIR / "311_noise"
    raw_files = list(raw_311_dir.glob("raw_311_noise_*.csv"))
    if not raw_files:
        logger.warning("No raw 311 data found for hotspot persistence")
        return None
    
    raw_path = max(raw_files, key=lambda p: p.stat().st_mtime)
    df_311 = pd.read_csv(raw_path, low_memory=False)
    
    # Filter to valid coords and timestamps
    has_coords = df_311["latitude"].notna() & df_311["longitude"].notna()
    df_311 = df_311[has_coords].copy()
    df_311["created_date"] = pd.to_datetime(df_311["created_date"], errors="coerce")
    df_311 = df_311[df_311["created_date"].notna()].copy()
    df_311["ts_nyc"] = ensure_nyc_timezone(df_311["created_date"])
    df_311["year"] = df_311["ts_nyc"].dt.year
    
    # Filter to nighttime and years
    nighttime_config = config.get("nighttime", {}).get("primary", {})
    start_hour = nighttime_config.get("start_hour", 22)
    end_hour = nighttime_config.get("end_hour", 7)
    
    gdf_311 = gpd.GeoDataFrame(
        df_311,
        geometry=gpd.points_from_xy(df_311["longitude"], df_311["latitude"]),
        crs="EPSG:4326",
    )
    gdf_311 = filter_nighttime(gdf_311, "ts_nyc", start_hour=start_hour, end_hour=end_hour)
    gdf_311 = gdf_311[gdf_311["year"].isin(YEARS)]
    
    # Project to EPSG:2263 for grid assignment
    gdf_proj = safe_reproject(gdf_311, 2263, "311 for grid")
    
    # Get bounds and create grid cells
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    gdf_proj["cell_x"] = ((gdf_proj.geometry.x - minx) // cell_size_ft).astype(int)
    gdf_proj["cell_y"] = ((gdf_proj.geometry.y - miny) // cell_size_ft).astype(int)
    gdf_proj["cell_id"] = gdf_proj["cell_x"].astype(str) + "_" + gdf_proj["cell_y"].astype(str)
    
    # Get top N cells per year
    top_cells_by_year = {}
    for year in YEARS:
        year_data = gdf_proj[gdf_proj["year"] == year]
        cell_counts = year_data.groupby("cell_id").size().sort_values(ascending=False)
        top_cells_by_year[year] = set(cell_counts.head(top_n).index)
    
    # Find cells in all 3 years
    persistent_cells = set.intersection(*top_cells_by_year.values())
    logger.info(f"Persistent cells (in top {top_n} all 3 years): {len(persistent_cells)}")
    
    if len(persistent_cells) == 0:
        logger.warning("No persistent hotspot cells found")
        return None
    
    # Build cell geometries
    from shapely.geometry import box
    
    cell_data = []
    for cell_id in persistent_cells:
        parts = cell_id.split("_")
        cell_x = int(parts[0])
        cell_y = int(parts[1])
        
        x_min = minx + cell_x * cell_size_ft
        y_min = miny + cell_y * cell_size_ft
        x_max = x_min + cell_size_ft
        y_max = y_min + cell_size_ft
        
        geom = box(x_min, y_min, x_max, y_max)
        
        # Get total count across all years
        total_count = gdf_proj[gdf_proj["cell_id"] == cell_id].shape[0]
        
        # Get per-year counts
        year_counts = {}
        for year in YEARS:
            year_counts[f"count_{year}"] = gdf_proj[(gdf_proj["cell_id"] == cell_id) & (gdf_proj["year"] == year)].shape[0]
        
        cell_data.append({
            "cell_id": cell_id,
            "geometry": geom,
            "total_count": total_count,
            **year_counts,
        })
    
    gdf_persistent = gpd.GeoDataFrame(cell_data, crs="EPSG:2263")
    
    # Reproject to EPSG:4326 for output
    gdf_persistent = gdf_persistent.to_crs("EPSG:4326")
    
    # Join artifact flags if available
    if artifact_flags is not None:
        # Need to map cell_id - check if formats match
        gdf_persistent = gdf_persistent.merge(
            artifact_flags,
            on="cell_id",
            how="left",
        )
    else:
        gdf_persistent["is_repeat_location_dominant"] = None
        gdf_persistent["is_suspected_artifact"] = None
        gdf_persistent["top_latlon_share"] = None
    
    # Sort by total count
    gdf_persistent = gdf_persistent.sort_values("total_count", ascending=False).reset_index(drop=True)
    
    logger.info(f"Built {len(gdf_persistent)} persistent hotspot cells")
    
    return gdf_persistent


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("08a_build_stability_assets") as logger:
        logger.info("Starting 08a_build_stability_assets.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        stability_config = config.get("stability", {})
        threshold_structural = stability_config.get("entropy_threshold_structural", 0.0)
        threshold_episodic = stability_config.get("entropy_threshold_episodic", 0.9)
        
        logger.info(f"Entropy thresholds: structural <= {threshold_structural}, episodic >= {threshold_episodic}")
        
        try:
            # Ensure output directory
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # ==================================================================
            # 1. CD Stability
            # ==================================================================
            logger.info("=" * 70)
            logger.info("CD STABILITY")
            logger.info("=" * 70)
            
            # Load inputs
            df_cd_clusters = pd.read_parquet(INPUT_CD_CLUSTERS)
            df_cd_clusters = ensure_boro_cd_dtype(df_cd_clusters)
            logger.info(f"Loaded CD clusters: {len(df_cd_clusters)} records")
            
            cd59 = read_gdf(GEO_DIR / "cd59.parquet")
            cd59 = ensure_boro_cd_dtype(cd59)
            cd_lookup = pd.read_parquet(GEO_DIR / "cd_lookup.parquet")
            cd_lookup = ensure_boro_cd_dtype(cd_lookup)
            
            # Build stability table
            df_cd_stability = build_cd_stability(
                df_cd_clusters, cd59, cd_lookup,
                threshold_structural, threshold_episodic, logger
            )
            
            # Write outputs
            atomic_write_df(df_cd_stability, OUTPUT_CD_STABILITY)
            logger.info(f"Wrote: {OUTPUT_CD_STABILITY}")
            
            df_cd_stability.to_csv(OUTPUT_CD_STABILITY_CSV, index=False)
            logger.info(f"Wrote: {OUTPUT_CD_STABILITY_CSV}")
            
            # GeoJSON
            gdf_cd = cd59[["boro_cd", "geometry"]].merge(
                df_cd_stability,
                on="boro_cd",
            )
            gdf_cd = ensure_boro_cd_dtype(gdf_cd)
            atomic_write_gdf(gdf_cd, OUTPUT_CD_STABILITY_GEOJSON)
            logger.info(f"Wrote: {OUTPUT_CD_STABILITY_GEOJSON}")
            
            # ==================================================================
            # 2. NTA Stability
            # ==================================================================
            logger.info("\n" + "=" * 70)
            logger.info("NTA STABILITY")
            logger.info("=" * 70)
            
            # Load inputs
            df_nta_clusters = pd.read_parquet(INPUT_NTA_CLUSTERS)
            logger.info(f"Loaded NTA clusters: {len(df_nta_clusters)} records")
            
            nta_geo = read_gdf(GEO_DIR / "nta.parquet")
            nta_lookup = pd.read_parquet(GEO_DIR / "nta_lookup.parquet")
            
            # Build stability table
            df_nta_stability = build_nta_stability(
                df_nta_clusters, nta_geo, nta_lookup,
                threshold_structural, threshold_episodic, logger
            )
            
            # Write outputs
            atomic_write_df(df_nta_stability, OUTPUT_NTA_STABILITY)
            logger.info(f"Wrote: {OUTPUT_NTA_STABILITY}")
            
            df_nta_stability.to_csv(OUTPUT_NTA_STABILITY_CSV, index=False)
            logger.info(f"Wrote: {OUTPUT_NTA_STABILITY_CSV}")
            
            # GeoJSON - filter nta_geo to residential
            residential_ntas = df_nta_stability["ntacode"].tolist()
            nta_geo_res = nta_geo[nta_geo["ntacode"].isin(residential_ntas)].copy()
            
            gdf_nta = nta_geo_res[["ntacode", "geometry"]].merge(
                df_nta_stability,
                on="ntacode",
            )
            atomic_write_gdf(gdf_nta, OUTPUT_NTA_STABILITY_GEOJSON)
            logger.info(f"Wrote: {OUTPUT_NTA_STABILITY_GEOJSON}")
            
            # ==================================================================
            # 3. Hotspot Persistence
            # ==================================================================
            logger.info("\n" + "=" * 70)
            logger.info("HOTSPOT PERSISTENCE")
            logger.info("=" * 70)
            
            gdf_persistent = build_hotspot_persistence(config, logger)
            
            if gdf_persistent is not None and len(gdf_persistent) > 0:
                atomic_write_gdf(gdf_persistent, OUTPUT_HOTSPOT_PERSISTENT)
                logger.info(f"Wrote: {OUTPUT_HOTSPOT_PERSISTENT}")
            else:
                logger.warning("No persistent hotspots to write")
            
            # ==================================================================
            # Log outputs
            # ==================================================================
            logger.log_outputs({
                "cd_stability_parquet": str(OUTPUT_CD_STABILITY),
                "cd_stability_csv": str(OUTPUT_CD_STABILITY_CSV),
                "cd_stability_geojson": str(OUTPUT_CD_STABILITY_GEOJSON),
                "nta_stability_parquet": str(OUTPUT_NTA_STABILITY),
                "nta_stability_csv": str(OUTPUT_NTA_STABILITY_CSV),
                "nta_stability_geojson": str(OUTPUT_NTA_STABILITY_GEOJSON),
                "hotspot_persistent": str(OUTPUT_HOTSPOT_PERSISTENT),
            })
            
            # Log metrics
            cd_stable_all = int(df_cd_stability["is_stable_all_3"].sum())
            cd_stable_2of3 = int(df_cd_stability["is_stable_2_of_3"].sum())
            nta_stable_all = int(df_nta_stability["is_stable_all_3"].sum())
            nta_stable_2of3 = int(df_nta_stability["is_stable_2_of_3"].sum())
            n_persistent = len(gdf_persistent) if gdf_persistent is not None else 0
            
            logger.log_metrics({
                "cd_stable_all_3": cd_stable_all,
                "cd_stable_2_of_3": cd_stable_2of3,
                "cd_mean_entropy": float(df_cd_stability["entropy_score"].mean()),
                "nta_stable_all_3": nta_stable_all,
                "nta_stable_2_of_3": nta_stable_2of3,
                "nta_mean_entropy": float(df_nta_stability["entropy_score"].mean()),
                "persistent_hotspots": n_persistent,
            })
            
            # Write metadata
            write_metadata_sidecar(
                output_path=OUTPUT_CD_STABILITY,
                inputs={
                    "cd_clusters_by_year": str(INPUT_CD_CLUSTERS),
                    "nta_clusters_by_year": str(INPUT_NTA_CLUSTERS),
                },
                config=config,
                run_id=logger.run_id,
                extra={
                    "threshold_structural": threshold_structural,
                    "threshold_episodic": threshold_episodic,
                    "cd_stability_counts": df_cd_stability["stability_class"].value_counts().to_dict(),
                    "nta_stability_counts": df_nta_stability["stability_class"].value_counts().to_dict(),
                    "persistent_hotspots": n_persistent,
                },
            )
            
            # Print summary
            logger.info("\n" + "=" * 70)
            logger.info("STABILITY ASSETS SUMMARY")
            logger.info("=" * 70)
            
            logger.info("\nCD Stability:")
            logger.info(f"  Structural (entropy=0): {(df_cd_stability['stability_class'] == 'Structural').sum()}")
            logger.info(f"  Semi-structural: {(df_cd_stability['stability_class'] == 'Semi-structural').sum()}")
            logger.info(f"  Episodic (entropy>=0.9): {(df_cd_stability['stability_class'] == 'Episodic').sum()}")
            logger.info(f"  Stable all 3 years: {cd_stable_all}")
            logger.info(f"  Stable 2 of 3 years: {cd_stable_2of3}")
            
            logger.info("\nNTA Stability:")
            logger.info(f"  Structural (entropy=0): {(df_nta_stability['stability_class'] == 'Structural').sum()}")
            logger.info(f"  Semi-structural: {(df_nta_stability['stability_class'] == 'Semi-structural').sum()}")
            logger.info(f"  Episodic (entropy>=0.9): {(df_nta_stability['stability_class'] == 'Episodic').sum()}")
            logger.info(f"  Stable all 3 years: {nta_stable_all}")
            logger.info(f"  Stable 2 of 3 years: {nta_stable_2of3}")
            logger.info(f"  With low-volume years: {(df_nta_stability['low_volume_years'] > 0).sum()}")
            
            logger.info(f"\nPersistent Hotspots: {n_persistent} cells in top 100 all 3 years")
            
            logger.info("\n" + "=" * 70)
            logger.info("SUCCESS: Built temporal stability assets")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

