#!/usr/bin/env python3
"""
07_build_air.py

Build air quality exposure metrics from NYCCAS (NYC Community Air Survey) rasters.
This script computes annual average PM2.5 and NO2 per Community District.

Inputs:
    - NYCCAS annual average rasters (300m ESRI Grid) for 2021-2023
    - data/processed/geo/cd59.parquet

Outputs:
    - data/processed/domains/air_cd.parquet
    - data/processed/metadata/air_cd_metadata.json

Per Project_Context.md Section 10.2.3:
    - Primary pollutants: PM2.5, NO2
    - Primary: 2021-2023 average
    - Sensitivity: 2022-2023 average
    - Directionality: higher pollution = worse

Data source: NYC Department of Health and Mental Hygiene
    - NYCCAS Air Pollution Rasters (NYC Open Data)
    - https://data.cityofnewyork.us/Environment/NYCCAS-Air-Pollution-Rasters/q68s-8qxv
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
from rasterio.features import rasterize
from shapely.geometry import mapping
from datetime import datetime
from typing import Dict, List, Tuple

from sleep_esi.paths import RAW_DIR, PROCESSED_DIR, CONFIG_DIR, LOGS_DIR, GEO_DIR
from sleep_esi.logging_utils import get_logger
from sleep_esi.io_utils import read_yaml, read_gdf, atomic_write_df, atomic_write_json
from sleep_esi.hashing import hash_file, get_git_info
from sleep_esi.schemas import ensure_boro_cd_dtype

# Constants
SCRIPT_NAME = "07_build_air"
NYCCAS_DIR = RAW_DIR / "air_nyccas" / "AnnAvg_1_15_300m"

# NYCCAS year mapping (Year 1 = 2009)
NYCCAS_BASE_YEAR = 2008  # Year 1 corresponds to Dec 2008 - Dec 2009, so effectively 2009

# Pollutants to process
POLLUTANTS = {
    'pm25': {
        'pattern': 'aa{year}_pm300m',
        'name': 'PM2.5',
        'units': 'μg/m³',
        'description': 'Fine Particulate Matter (annual average)'
    },
    'no2': {
        'pattern': 'aa{year}_no2300m',
        'name': 'NO2',
        'units': 'ppb',
        'description': 'Nitrogen Dioxide (annual average)'
    }
}


def nyccas_year_to_folder_num(calendar_year: int) -> int:
    """Convert calendar year to NYCCAS folder number."""
    return calendar_year - NYCCAS_BASE_YEAR


def get_raster_path(pollutant: str, calendar_year: int) -> Path:
    """Get path to NYCCAS raster for given pollutant and year."""
    folder_num = nyccas_year_to_folder_num(calendar_year)
    folder_name = POLLUTANTS[pollutant]['pattern'].format(year=folder_num)
    return NYCCAS_DIR / folder_name


def compute_zonal_stats_from_esri_grid(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    logger,
    id_col: str = 'boro_cd'
) -> pd.DataFrame:
    """
    Compute zonal statistics for each polygon from an ESRI Grid raster.
    
    Args:
        gdf: GeoDataFrame with polygons (will be reprojected to raster CRS)
        raster_path: Path to ESRI Grid folder
        logger: JSONL logger
        id_col: Column name for polygon ID
        
    Returns:
        DataFrame with zonal statistics
    """
    results = []
    
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata
        data = src.read(1)
        transform = src.transform
        
        logger.info(f"Raster loaded: {raster_path.name}", extra={
            "crs": str(raster_crs),
            "shape": data.shape,
            "nodata": float(nodata) if nodata else None
        })
        
        # CD59 is already in EPSG:2263 which matches NYCCAS CRS (same projection)
        # Both use NAD83 / NY Long Island with US survey feet
        gdf_reproj = gdf
        
        for idx, row in gdf_reproj.iterrows():
            boro_cd = row[id_col]
            geom = row['geometry']
            
            try:
                # Rasterize the polygon
                poly_mask = rasterize(
                    [(geom, 1)],
                    out_shape=data.shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True
                )
                
                # Extract values within polygon
                polygon_pixels = data[poly_mask == 1]
                total_pixels = len(polygon_pixels)
                
                # Filter nodata
                if nodata is not None:
                    valid_mask = polygon_pixels != nodata
                    valid_pixels = polygon_pixels[valid_mask]
                else:
                    valid_pixels = polygon_pixels
                
                n_valid = len(valid_pixels)
                nodata_pct = (total_pixels - n_valid) / total_pixels * 100 if total_pixels > 0 else 100
                
                if n_valid > 0:
                    mean_val = float(np.mean(valid_pixels))
                    median_val = float(np.median(valid_pixels))
                    std_val = float(np.std(valid_pixels))
                    min_val = float(np.min(valid_pixels))
                    max_val = float(np.max(valid_pixels))
                else:
                    mean_val = median_val = std_val = min_val = max_val = np.nan
                
                results.append({
                    id_col: boro_cd,
                    'mean': mean_val,
                    'median': median_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'pixel_count': total_pixels,
                    'valid_pixel_count': n_valid,
                    'nodata_pct': nodata_pct
                })
                
            except Exception as e:
                logger.warning(f"Zonal stats error for CD {boro_cd}", extra={
                    "error": str(e)
                })
                results.append({
                    id_col: boro_cd,
                    'mean': np.nan,
                    'median': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'pixel_count': 0,
                    'valid_pixel_count': 0,
                    'nodata_pct': 100.0
                })
    
    return pd.DataFrame(results)


def standardize_metrics(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Compute standardized versions of a metric.
    
    For air pollution: higher concentration = worse = higher score
    Per R20: Directionality enforced.
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
    
    # Define years
    primary_years = list(range(
        params['time_windows']['primary']['year_start'],
        params['time_windows']['primary']['year_end'] + 1
    ))
    sensitivity_years = list(range(
        params['time_windows']['sensitivity']['year_start'],
        params['time_windows']['sensitivity']['year_end'] + 1
    ))
    
    logger.info("Time windows", extra={
        "primary_years": primary_years,
        "sensitivity_years": sensitivity_years
    })
    
    # Load CD59 geometries in EPSG:2263 (matches NYCCAS CRS)
    # Using the pre-projected version avoids reprojection issues
    cd59_path = PROCESSED_DIR / "geo" / "cd59_epsg2263.parquet"
    cd59 = read_gdf(cd59_path)
    logger.info("CD59 loaded", extra={"rows": len(cd59), "crs": str(cd59.crs.to_epsg())})
    
    # Verify raster files exist
    input_hashes = {}
    for pollutant in POLLUTANTS:
        for year in primary_years:
            raster_path = get_raster_path(pollutant, year)
            if not raster_path.exists():
                logger.error(f"Missing raster: {raster_path}")
                raise FileNotFoundError(f"Missing NYCCAS raster: {raster_path}")
            # Hash the hdr.adf file for provenance
            hdr_path = raster_path / "hdr.adf"
            if hdr_path.exists():
                input_hashes[f"{pollutant}_{year}"] = hash_file(hdr_path)
    
    logger.log_inputs({
        "nyccas_dir": str(NYCCAS_DIR),
        "cd59": str(cd59_path),
        **input_hashes
    })
    
    # Process each pollutant
    all_yearly_stats = {}
    
    for pollutant, config in POLLUTANTS.items():
        logger.info(f"Processing {config['name']}")
        yearly_stats = {}
        
        for year in primary_years:
            raster_path = get_raster_path(pollutant, year)
            logger.info(f"  Year {year}: {raster_path.name}")
            
            stats_df = compute_zonal_stats_from_esri_grid(
                cd59, raster_path, logger, id_col='boro_cd'
            )
            stats_df = stats_df.rename(columns={'mean': f'{pollutant}_{year}'})
            yearly_stats[year] = stats_df[['boro_cd', f'{pollutant}_{year}']]
            
            logger.info(f"  {config['name']} {year} stats", extra={
                "mean": float(stats_df[f'{pollutant}_{year}'].mean()),
                "min": float(stats_df[f'{pollutant}_{year}'].min()),
                "max": float(stats_df[f'{pollutant}_{year}'].max())
            })
        
        all_yearly_stats[pollutant] = yearly_stats
    
    # Build result DataFrame
    result_df = cd59[['boro_cd']].copy()
    
    for pollutant, yearly_stats in all_yearly_stats.items():
        # Merge yearly data
        for year, stats in yearly_stats.items():
            result_df = result_df.merge(stats, on='boro_cd', how='left')
        
        # Compute multi-year averages
        yearly_cols = [f'{pollutant}_{y}' for y in primary_years]
        result_df[f'{pollutant}_mean_primary'] = result_df[yearly_cols].mean(axis=1)
        
        sensitivity_cols = [f'{pollutant}_{y}' for y in sensitivity_years]
        result_df[f'{pollutant}_mean_sensitivity'] = result_df[sensitivity_cols].mean(axis=1)
    
    # Add pixel count from last processed raster (for QC reference)
    last_stats = compute_zonal_stats_from_esri_grid(
        cd59, get_raster_path('pm25', primary_years[-1]), logger
    )
    result_df = result_df.merge(
        last_stats[['boro_cd', 'pixel_count', 'valid_pixel_count', 'nodata_pct']],
        on='boro_cd', how='left'
    )
    
    # Standardize primary metrics
    for pollutant in POLLUTANTS:
        result_df = standardize_metrics(result_df, f'{pollutant}_mean_primary')
    
    # Rename for clarity
    result_df = result_df.rename(columns={
        'z_pm25_mean_primary_robust': 'z_air_pm25_robust',
        'z_pm25_mean_primary_classic': 'z_air_pm25_classic',
        'pct_pm25_mean_primary': 'pct_air_pm25',
        'z_no2_mean_primary_robust': 'z_air_no2_robust',
        'z_no2_mean_primary_classic': 'z_air_no2_classic',
        'pct_no2_mean_primary': 'pct_air_no2'
    })
    
    # Create combined air quality index (average of PM2.5 and NO2 z-scores)
    result_df['z_air_combined_robust'] = (
        result_df['z_air_pm25_robust'] + result_df['z_air_no2_robust']
    ) / 2
    
    # Add metadata columns
    result_df['year_start'] = params['time_windows']['primary']['year_start']
    result_df['year_end'] = params['time_windows']['primary']['year_end']
    result_df['domain'] = 'air'
    result_df['source_id'] = 'NYCCAS_annual_avg_300m'
    
    # Ensure boro_cd is Int64
    result_df = ensure_boro_cd_dtype(result_df)
    
    # Log summary
    logger.log_metrics({
        "rows": len(result_df),
        "pm25_mean_min": float(result_df['pm25_mean_primary'].min()),
        "pm25_mean_max": float(result_df['pm25_mean_primary'].max()),
        "pm25_mean_mean": float(result_df['pm25_mean_primary'].mean()),
        "no2_mean_min": float(result_df['no2_mean_primary'].min()),
        "no2_mean_max": float(result_df['no2_mean_primary'].max()),
        "no2_mean_mean": float(result_df['no2_mean_primary'].mean()),
        "na_count": int(result_df['pm25_mean_primary'].isna().sum())
    })
    
    # QA: Check directionality
    for pollutant in ['pm25', 'no2']:
        raw_vals = result_df[f'{pollutant}_mean_primary'].dropna()
        z_vals = result_df[f'z_air_{pollutant}_robust'].dropna()
        if len(raw_vals) >= 3:
            correlation = raw_vals.corr(z_vals)
            logger.info(f"Directionality check: {pollutant}", extra={
                "correlation": float(correlation)
            })
    
    # Join CD labels (REQUIRED per CD labeling policy)
    cd_lookup = pd.read_parquet(GEO_DIR / "cd_lookup.parquet")
    result_df = result_df.merge(
        cd_lookup[["boro_cd", "borough_name", "district_number", "cd_label", "cd_short"]],
        on="boro_cd",
        how="left",
    )
    
    # Write output
    output_path = PROCESSED_DIR / "context" / "air_cd.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    atomic_write_df(result_df, output_path)
    logger.log_outputs({"air_cd": str(output_path)})
    
    # Write metadata
    metadata = {
        'script': SCRIPT_NAME,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'inputs': {
            'nyccas_dir': str(NYCCAS_DIR),
            'cd59': str(cd59_path),
            **input_hashes
        },
        'outputs': {
            'air_cd': str(output_path)
        },
        'time_windows': {
            'primary': {
                'years': primary_years,
                'label': params['time_windows']['primary']['label']
            },
            'sensitivity': {
                'years': sensitivity_years,
                'label': params['time_windows']['sensitivity']['label']
            }
        },
        'pollutants': {
            pollutant: {
                'name': config['name'],
                'units': config['units'],
                'description': config['description'],
                'mean_range': [
                    float(result_df[f'{pollutant}_mean_primary'].min()),
                    float(result_df[f'{pollutant}_mean_primary'].max())
                ]
            }
            for pollutant, config in POLLUTANTS.items()
        },
        'stats': {
            'n_districts': len(result_df),
            'na_count': int(result_df['pm25_mean_primary'].isna().sum()),
            'avg_nodata_pct': float(result_df['nodata_pct'].mean())
        },
        'raster_metadata': {
            'source': 'NYCCAS (NYC Community Air Survey)',
            'provider': 'NYC DOHMH',
            'url': 'https://data.cityofnewyork.us/Environment/NYCCAS-Air-Pollution-Rasters/q68s-8qxv',
            'crs': 'NAD83 / New York Long Island (EPSG:2263 equivalent)',
            'resolution_ft': 984,
            'resolution_m': 300,
            'method': 'Land Use Regression model'
        },
        'directionality': 'higher_pollution_worse',
        'git': get_git_info()
    }
    
    metadata_path = PROCESSED_DIR / "metadata" / "air_cd_metadata.json"
    atomic_write_json(metadata, metadata_path)
    
    logger.info("Script complete", extra={
        "output": str(output_path),
        "rows": len(result_df)
    })
    
    # Print summary
    print(f"\n✓ Air quality metrics computed for {len(result_df)} community districts")
    print(f"  Output: {output_path}")
    
    print(f"\n  PM2.5 (2021-2023 avg): {result_df['pm25_mean_primary'].min():.2f} - {result_df['pm25_mean_primary'].max():.2f} μg/m³")
    print(f"  NYC mean PM2.5: {result_df['pm25_mean_primary'].mean():.2f} μg/m³")
    
    print(f"\n  NO2 (2021-2023 avg): {result_df['no2_mean_primary'].min():.2f} - {result_df['no2_mean_primary'].max():.2f} ppb")
    print(f"  NYC mean NO2: {result_df['no2_mean_primary'].mean():.2f} ppb")
    
    # Top 5 most polluted CDs (by combined z-score)
    print(f"\n  Top 5 most polluted CDs (combined air quality):")
    top5 = result_df.nlargest(5, 'z_air_combined_robust')[
        ['boro_cd', 'pm25_mean_primary', 'no2_mean_primary', 'z_air_combined_robust']
    ]
    for _, row in top5.iterrows():
        print(f"    CD {int(row['boro_cd'])}: PM2.5={row['pm25_mean_primary']:.2f}, NO2={row['no2_mean_primary']:.1f} (z={row['z_air_combined_robust']:.2f})")
    
    # Top 5 cleanest CDs
    print(f"\n  Top 5 cleanest CDs (combined air quality):")
    bottom5 = result_df.nsmallest(5, 'z_air_combined_robust')[
        ['boro_cd', 'pm25_mean_primary', 'no2_mean_primary', 'z_air_combined_robust']
    ]
    for _, row in bottom5.iterrows():
        print(f"    CD {int(row['boro_cd'])}: PM2.5={row['pm25_mean_primary']:.2f}, NO2={row['no2_mean_primary']:.1f} (z={row['z_air_combined_robust']:.2f})")


if __name__ == "__main__":
    main()
