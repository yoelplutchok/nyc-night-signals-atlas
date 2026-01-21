#!/usr/bin/env python3
"""
04_build_noise_objective.py

Build objective noise exposure metrics from BTS National Transportation Noise Map.
This script computes energy-averaged dB (LAeq) per Community District.

Inputs:
    - NY_rail_road_and_aviation_noise_2020.tif (BTS NTNM 2020)
    - data/processed/geo/cd59.parquet

Outputs:
    - data/processed/domains/noise_obj_full_cd.parquet

Per research_plan.md Section 4.1.2:
    - Tier 1: Full objective noise (road + rail + aviation)
    - dB aggregation MUST use energy averaging (not arithmetic mean)
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from datetime import datetime

from sleep_esi.paths import RAW_DIR, PROCESSED_DIR, CONFIG_DIR, LOGS_DIR, GEO_DIR
from sleep_esi.logging_utils import get_logger
from sleep_esi.io_utils import read_yaml, read_gdf, atomic_write_df
from sleep_esi.hashing import hash_file, get_git_info
from sleep_esi.schemas import DOMAIN_BASE_SCHEMA
from sleep_esi.acoustics import energy_mean_db

# Constants
SCRIPT_NAME = "04_build_noise_objective"
NOISE_RASTER_PATH = RAW_DIR / "noise_objective" / "CONUS_rail_road_and_aviation_noise_2020" / "State_rasters" / "NY_rail_road_and_aviation_noise_2020.tif"


def compute_zonal_stats_energy_db(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    nodata_value: float,
    logger
) -> pd.DataFrame:
    """
    Compute zonal statistics for each polygon using energy-averaged dB.
    
    Per research_plan.md R9: dB aggregation MUST use energy averaging.
    
    Args:
        gdf: GeoDataFrame with polygons (must be in raster CRS)
        raster_path: Path to noise raster
        nodata_value: NoData value in raster
        logger: JSONL logger
        
    Returns:
        DataFrame with columns: boro_cd, noise_obj_db_mean, noise_obj_db_max,
                               pixel_count, nodata_pct, valid_pixel_count
    """
    results = []
    
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        
        logger.info("Raster info loaded", extra={
            "crs": str(raster_crs),
            "bounds": list(raster_bounds),
            "resolution": src.res[0],
            "nodata": nodata_value
        })
        
        for idx, row in gdf.iterrows():
            boro_cd = row['boro_cd']
            geom = row['geometry']
            
            try:
                # Mask raster with polygon
                out_image, out_transform = mask(
                    src, 
                    [mapping(geom)], 
                    crop=True,
                    nodata=nodata_value,
                    all_touched=True  # Include all pixels that touch the polygon
                )
                
                # Get the data array
                data = out_image[0]  # First band
                
                # Count pixels
                total_pixels = data.size
                
                # Handle nodata - the nodata value is a very large negative number
                # Also filter out any values that don't make sense for dB (< 0 or > 200)
                valid_mask = (data != nodata_value) & (data > 0) & (data < 200)
                valid_data = data[valid_mask]
                valid_pixels = len(valid_data)
                nodata_pixels = total_pixels - valid_pixels
                nodata_pct = (nodata_pixels / total_pixels * 100) if total_pixels > 0 else 100
                
                if valid_pixels > 0:
                    # Energy-averaged dB (REQUIRED per research_plan.md R9)
                    db_mean = energy_mean_db(valid_data)
                    db_max = float(np.max(valid_data))
                    db_min = float(np.min(valid_data))
                else:
                    db_mean = np.nan
                    db_max = np.nan
                    db_min = np.nan
                
                results.append({
                    'boro_cd': boro_cd,
                    'noise_obj_db_mean': db_mean,
                    'noise_obj_db_max': db_max,
                    'noise_obj_db_min': db_min,
                    'pixel_count': total_pixels,
                    'valid_pixel_count': valid_pixels,
                    'nodata_pct': nodata_pct
                })
                
            except Exception as e:
                logger.warning(f"Zonal stats error for CD {boro_cd}", extra={
                    "boro_cd": int(boro_cd), 
                    "error": str(e)
                })
                results.append({
                    'boro_cd': boro_cd,
                    'noise_obj_db_mean': np.nan,
                    'noise_obj_db_max': np.nan,
                    'noise_obj_db_min': np.nan,
                    'pixel_count': 0,
                    'valid_pixel_count': 0,
                    'nodata_pct': 100.0
                })
    
    return pd.DataFrame(results)


def standardize_metrics(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Compute standardized versions of a metric.
    
    For noise: higher dB = worse = higher score (positive direction)
    """
    values = df[col].dropna()
    
    if len(values) == 0:
        df[f'z_{col}_robust'] = np.nan
        df[f'z_{col}_classic'] = np.nan
        df[f'pct_{col}'] = np.nan
        return df
    
    # Robust z-score (median/MAD)
    median = values.median()
    mad = np.median(np.abs(values - median))
    if mad > 0:
        df[f'z_{col}_robust'] = (df[col] - median) / (mad * 1.4826)
    else:
        df[f'z_{col}_robust'] = 0.0
    
    # Classic z-score (mean/SD)
    mean = values.mean()
    std = values.std()
    if std > 0:
        df[f'z_{col}_classic'] = (df[col] - mean) / std
    else:
        df[f'z_{col}_classic'] = 0.0
    
    # Percentile rank (0-100)
    df[f'pct_{col}'] = df[col].rank(pct=True) * 100
    
    return df


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("Script starting", extra={"script": SCRIPT_NAME, "run_id": run_id})
    
    # Load config
    params = read_yaml(CONFIG_DIR / "params.yml")
    logger.log_config(params)
    
    # Check if raster exists
    if not NOISE_RASTER_PATH.exists():
        logger.error("Raster not found", extra={"path": str(NOISE_RASTER_PATH)})
        raise FileNotFoundError(f"Noise raster not found: {NOISE_RASTER_PATH}")
    
    # Hash input file for provenance
    raster_hash = hash_file(NOISE_RASTER_PATH)
    logger.log_inputs({
        "raster": str(NOISE_RASTER_PATH),
        "raster_sha256": raster_hash
    })
    
    # Load CD59 geometries
    cd59_path = PROCESSED_DIR / "geo" / "cd59.parquet"
    cd59 = read_gdf(cd59_path)
    logger.info("CD59 loaded", extra={"rows": len(cd59), "crs": str(cd59.crs)})
    
    # Get raster CRS and reproject polygons
    with rasterio.open(NOISE_RASTER_PATH) as src:
        raster_crs = src.crs
        nodata_value = src.nodata
        
    logger.info("Raster CRS identified", extra={"crs": str(raster_crs)})
    
    # Reproject CD59 to raster CRS
    cd59_reproj = cd59.to_crs(raster_crs)
    logger.info("CD59 reprojected", extra={
        "from_crs": str(cd59.crs), 
        "to_crs": str(raster_crs),
        "bounds": list(cd59_reproj.total_bounds)
    })
    
    # Verify NYC bounds are within raster bounds
    with rasterio.open(NOISE_RASTER_PATH) as src:
        raster_bounds = src.bounds
        poly_bounds = cd59_reproj.total_bounds
        
        # Check overlap
        overlap = (
            poly_bounds[0] >= raster_bounds.left and
            poly_bounds[2] <= raster_bounds.right and
            poly_bounds[1] >= raster_bounds.bottom and
            poly_bounds[3] <= raster_bounds.top
        )
        
        if not overlap:
            logger.warning("Bounds partial overlap", extra={
                "raster_bounds": list(raster_bounds),
                "polygon_bounds": list(poly_bounds)
            })
        else:
            logger.info("Bounds validated", extra={"status": "full_overlap"})
    
    # Compute zonal statistics with energy-averaged dB
    logger.info("Computing zonal stats", extra={"method": "energy_averaged_db"})
    stats_df = compute_zonal_stats_energy_db(
        cd59_reproj, 
        NOISE_RASTER_PATH, 
        nodata_value,
        logger
    )
    
    # Log summary statistics
    logger.log_metrics({
        "rows": len(stats_df),
        "db_mean_min": float(stats_df['noise_obj_db_mean'].min()),
        "db_mean_max": float(stats_df['noise_obj_db_mean'].max()),
        "db_mean_mean": float(stats_df['noise_obj_db_mean'].mean()),
        "na_count": int(stats_df['noise_obj_db_mean'].isna().sum()),
        "avg_nodata_pct": float(stats_df['nodata_pct'].mean())
    })
    
    # Rename for clarity and compute coverage
    stats_df = stats_df.rename(columns={'nodata_pct': 'noise_nodata_pct'})
    stats_df['noise_coverage_pct'] = 100 - stats_df['noise_nodata_pct']
    
    # Standardize metrics
    stats_df = standardize_metrics(stats_df, 'noise_obj_db_mean')
    
    # Add metadata columns
    stats_df['year_start'] = 2020
    stats_df['year_end'] = 2020
    stats_df['domain'] = 'noise_objective'
    stats_df['source_id'] = 'BTS_NTNM_2020_rail_road_aviation'
    stats_df['units'] = 'dB_LAeq'
    
    # Ensure boro_cd is Int64
    stats_df['boro_cd'] = stats_df['boro_cd'].astype('Int64')
    
    # Join CD labels (REQUIRED per CD labeling policy)
    cd_lookup = pd.read_parquet(GEO_DIR / "cd_lookup.parquet")
    stats_df = stats_df.merge(
        cd_lookup[["boro_cd", "borough_name", "district_number", "cd_label", "cd_short"]],
        on="boro_cd",
        how="left",
    )
    
    # Write output
    output_path = PROCESSED_DIR / "context" / "noise_obj_cd.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    atomic_write_df(stats_df, output_path)
    
    logger.log_outputs({"noise_obj_cd": str(output_path)})
    
    # Write metadata sidecar
    metadata = {
        'script': SCRIPT_NAME,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'inputs': {
            'raster': str(NOISE_RASTER_PATH),
            'raster_sha256': raster_hash,
            'cd59': str(cd59_path)
        },
        'outputs': {
            'noise_obj_cd': str(output_path)
        },
        'stats': {
            'n_districts': len(stats_df),
            'db_mean_range': [float(stats_df['noise_obj_db_mean'].min()), 
                             float(stats_df['noise_obj_db_mean'].max())],
            'na_count': int(stats_df['noise_obj_db_mean'].isna().sum()),
            'avg_nodata_pct': float(stats_df['noise_nodata_pct'].mean()),
            'avg_coverage_pct': float(stats_df['noise_coverage_pct'].mean())
        },
        'raster_metadata': {
            'crs': str(raster_crs),
            'resolution_m': 30,
            'source': 'BTS National Transportation Noise Map 2020',
            'noise_types': ['road', 'rail', 'aviation'],
            'metric': 'LAeq (24-hr equivalent A-weighted sound level)'
        },
        'git': get_git_info()
    }
    
    metadata_path = PROCESSED_DIR / "metadata" / "noise_obj_cd_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info("Script complete", extra={
        "output": str(output_path),
        "rows": len(stats_df)
    })
    
    print(f"\n✓ Objective noise metrics computed for {len(stats_df)} community districts")
    print(f"  Output: {output_path}")
    print(f"\n  dB (energy-averaged) range: {stats_df['noise_obj_db_mean'].min():.1f} - {stats_df['noise_obj_db_mean'].max():.1f}")
    print(f"  Mean dB across CDs: {stats_df['noise_obj_db_mean'].mean():.1f}")
    print(f"  Avg coverage %: {stats_df['noise_coverage_pct'].mean():.1f}%")


if __name__ == "__main__":
    main()
