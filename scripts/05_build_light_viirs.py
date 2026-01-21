#!/usr/bin/env python3
"""
05_build_light_viirs.py

Build light-at-night exposure metrics from VIIRS Black Marble (VNP46A4) annual composites.
This script computes mean radiance per Community District from 2021-2023.

Inputs:
    - VNP46A4 HDF5 annual composites (h10v04 tile) for 2021, 2022, 2023
    - data/processed/geo/cd59.parquet

Outputs:
    - data/processed/domains/light_viirs_cd.parquet

Per Project_Context.md Section 10.2.1:
    - Primary: 2021-2023 average
    - Sensitivity: 2022-2023 average (pandemic distortion check)
    - Layer: AllAngle_Composite_Snow_Free (radiance in nWatts/cm²sr)
    - Directionality: higher light = worse
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sleep_esi.paths import RAW_DIR, PROCESSED_DIR, CONFIG_DIR, LOGS_DIR, GEO_DIR
from sleep_esi.logging_utils import get_logger
from sleep_esi.io_utils import read_yaml, read_gdf, atomic_write_df, atomic_write_json
from sleep_esi.hashing import hash_file, get_git_info
from sleep_esi.schemas import ensure_boro_cd_dtype

# Constants
SCRIPT_NAME = "05_build_light_viirs"
VIIRS_DIR = RAW_DIR / "light_viirs" / "VNP46A4"

# VNP46A4 dataset configuration
RADIANCE_DATASET = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/AllAngle_Composite_Snow_Free"
NUM_OBS_DATASET = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/AllAngle_Composite_Snow_Free_Num"
LAT_DATASET = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lat"
LON_DATASET = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lon"

# Per VNP46A4 documentation
FILL_VALUE = -999.9
VALID_MIN = 0.0


def read_vnp46a4(hdf_path: Path, logger) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Read VNP46A4 HDF5 file and extract radiance data with geolocation.
    
    Per R8: Explicit nodata/fill/valid range handling.
    
    Args:
        hdf_path: Path to HDF5 file
        logger: JSONL logger
        
    Returns:
        Tuple of (radiance array, num_obs array, metadata dict)
    """
    logger.info(f"Reading HDF5: {hdf_path.name}")
    
    with h5py.File(hdf_path, 'r') as f:
        # Read radiance data
        radiance = f[RADIANCE_DATASET][:]
        num_obs = f[NUM_OBS_DATASET][:]
        
        # Read coordinate arrays
        lat = f[LAT_DATASET][:]
        lon = f[LON_DATASET][:]
        
        # Get metadata from root attributes
        meta = {
            'north_bound': float(f.attrs.get('NorthBoundingCoord', 0)),
            'south_bound': float(f.attrs.get('SouthBoundingCoord', 0)),
            'east_bound': float(f.attrs.get('EastBoundingCoord', 0)),
            'west_bound': float(f.attrs.get('WestBoundingCoord', 0)),
            'year_start': str(f.attrs.get('RangeBeginningDate', b'')[:4]),
            'year_end': str(f.attrs.get('RangeEndingDate', b'')[:4]),
            'shape': radiance.shape,
            'lat_range': [float(lat.min()), float(lat.max())],
            'lon_range': [float(lon.min()), float(lon.max())],
        }
    
    # Apply fill value and valid range per R8
    # Set fill values and invalid values to NaN
    radiance = radiance.astype(np.float32)
    invalid_mask = (radiance == FILL_VALUE) | (radiance < VALID_MIN)
    radiance[invalid_mask] = np.nan
    
    n_valid = np.sum(~np.isnan(radiance))
    n_total = radiance.size
    
    logger.info(f"Radiance loaded", extra={
        "shape": radiance.shape,
        "valid_pixels": int(n_valid),
        "total_pixels": int(n_total),
        "valid_pct": float(n_valid / n_total * 100),
        "bounds": [meta['west_bound'], meta['south_bound'], 
                   meta['east_bound'], meta['north_bound']]
    })
    
    return radiance, num_obs, meta


def compute_geotransform(meta: Dict, shape: Tuple[int, int]) -> rasterio.Affine:
    """
    Compute affine transform from HDF5 metadata.
    
    VNP46A4 uses a regular lat/lon grid (EPSG:4326).
    """
    height, width = shape
    
    # Bounds from metadata
    west = meta['west_bound']
    east = meta['east_bound']
    south = meta['south_bound']
    north = meta['north_bound']
    
    # Pixel size
    pixel_width = (east - west) / width
    pixel_height = (north - south) / height
    
    # Affine transform (top-left origin)
    transform = rasterio.Affine(
        pixel_width, 0.0, west,
        0.0, -pixel_height, north
    )
    
    return transform


def compute_zonal_stats_from_array(
    gdf: gpd.GeoDataFrame,
    data: np.ndarray,
    transform: rasterio.Affine,
    logger,
    id_col: str = 'boro_cd'
) -> pd.DataFrame:
    """
    Compute zonal statistics for each polygon from a numpy array.
    
    Per R8: Log pixel counts and nodata fraction per polygon.
    
    Args:
        gdf: GeoDataFrame with polygons (must be in EPSG:4326)
        data: 2D numpy array with radiance values (NaN for nodata)
        transform: Affine transform for the array
        logger: JSONL logger
        id_col: Column name for polygon ID
        
    Returns:
        DataFrame with columns: boro_cd, radiance_mean, pixel_count, 
                               valid_pixel_count, nodata_pct
    """
    results = []
    height, width = data.shape
    
    for idx, row in gdf.iterrows():
        boro_cd = row[id_col]
        geom = row['geometry']
        
        try:
            # Create a mask for this polygon
            # Rasterize the geometry to match the data array
            mask = rasterize(
                [(geom, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )
            
            # Extract values within polygon
            polygon_data = data[mask == 1]
            total_pixels = len(polygon_data)
            
            if total_pixels == 0:
                results.append({
                    id_col: boro_cd,
                    'radiance_mean': np.nan,
                    'radiance_median': np.nan,
                    'radiance_std': np.nan,
                    'pixel_count': 0,
                    'valid_pixel_count': 0,
                    'nodata_pct': 100.0
                })
                continue
            
            # Count valid (non-NaN) pixels
            valid_data = polygon_data[~np.isnan(polygon_data)]
            valid_pixels = len(valid_data)
            nodata_pct = (total_pixels - valid_pixels) / total_pixels * 100
            
            if valid_pixels > 0:
                radiance_mean = float(np.mean(valid_data))
                radiance_median = float(np.median(valid_data))
                radiance_std = float(np.std(valid_data))
            else:
                radiance_mean = np.nan
                radiance_median = np.nan
                radiance_std = np.nan
            
            results.append({
                id_col: boro_cd,
                'radiance_mean': radiance_mean,
                'radiance_median': radiance_median,
                'radiance_std': radiance_std,
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
                id_col: boro_cd,
                'radiance_mean': np.nan,
                'radiance_median': np.nan,
                'radiance_std': np.nan,
                'pixel_count': 0,
                'valid_pixel_count': 0,
                'nodata_pct': 100.0
            })
    
    return pd.DataFrame(results)


def standardize_metrics(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Compute standardized versions of a metric.
    
    For light: higher radiance = worse = higher score (positive direction)
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


def check_bounds_overlap(raster_bounds: Tuple, polygon_bounds: Tuple, logger) -> bool:
    """
    Check if raster and polygon bounds overlap.
    
    Per R8: Assert overlap of bounds.
    """
    r_west, r_south, r_east, r_north = raster_bounds
    p_minx, p_miny, p_maxx, p_maxy = polygon_bounds
    
    overlaps = not (
        r_east < p_minx or
        r_west > p_maxx or
        r_north < p_miny or
        r_south > p_maxy
    )
    
    if not overlaps:
        logger.error("Raster and polygon bounds do not overlap", extra={
            "raster_bounds": raster_bounds,
            "polygon_bounds": polygon_bounds
        })
        raise ValueError("Raster and polygon bounds do not overlap")
    
    logger.info("Bounds overlap validated", extra={
        "raster_bounds": raster_bounds,
        "polygon_bounds": polygon_bounds
    })
    
    return True


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("Script starting", extra={"script": SCRIPT_NAME, "run_id": run_id})
    
    # Load config
    params = read_yaml(CONFIG_DIR / "params.yml")
    logger.log_config(params)
    
    # Define years to process
    primary_years = list(range(
        params['time_windows']['primary']['year_start'],
        params['time_windows']['primary']['year_end'] + 1
    ))
    sensitivity_years = list(range(
        params['time_windows']['sensitivity']['year_start'],
        params['time_windows']['sensitivity']['year_end'] + 1
    ))
    
    logger.info("Time windows configured", extra={
        "primary_years": primary_years,
        "sensitivity_years": sensitivity_years
    })
    
    # Find HDF5 files for each year
    hdf_files = {}
    input_hashes = {}
    
    for year in primary_years:
        year_dir = VIIRS_DIR / str(year)
        if not year_dir.exists():
            logger.error(f"Missing VIIRS data directory for {year}", extra={"path": str(year_dir)})
            raise FileNotFoundError(f"Missing VIIRS data: {year_dir}")
        
        h5_files = list(year_dir.glob("*.h5"))
        if len(h5_files) == 0:
            logger.error(f"No HDF5 files found for {year}", extra={"path": str(year_dir)})
            raise FileNotFoundError(f"No HDF5 files in {year_dir}")
        
        if len(h5_files) > 1:
            logger.warning(f"Multiple HDF5 files for {year}, using first", extra={
                "files": [f.name for f in h5_files]
            })
        
        hdf_files[year] = h5_files[0]
        input_hashes[f"viirs_{year}"] = hash_file(h5_files[0])
    
    logger.log_inputs({
        **{f"viirs_{year}": str(path) for year, path in hdf_files.items()},
        **{f"viirs_{year}_sha256": h for year, h in zip(hdf_files.keys(), input_hashes.values())}
    })
    
    # Load CD59 geometries
    cd59_path = PROCESSED_DIR / "geo" / "cd59.parquet"
    cd59 = read_gdf(cd59_path)
    logger.info("CD59 loaded", extra={"rows": len(cd59), "crs": str(cd59.crs)})
    
    # Ensure CD59 is in EPSG:4326 (same as VIIRS)
    if cd59.crs.to_epsg() != 4326:
        logger.info("Reprojecting CD59 to EPSG:4326")
        cd59 = cd59.to_crs("EPSG:4326")
    
    # Process each year
    yearly_stats = {}
    raster_metadata = {}
    
    for year, hdf_path in hdf_files.items():
        logger.info(f"Processing year {year}")
        
        # Read HDF5 data
        radiance, num_obs, meta = read_vnp46a4(hdf_path, logger)
        raster_metadata[year] = meta
        
        # Compute geotransform
        transform = compute_geotransform(meta, radiance.shape)
        
        # Check bounds overlap
        raster_bounds = (meta['west_bound'], meta['south_bound'], 
                        meta['east_bound'], meta['north_bound'])
        polygon_bounds = tuple(cd59.total_bounds)
        check_bounds_overlap(raster_bounds, polygon_bounds, logger)
        
        # Compute zonal statistics
        stats_df = compute_zonal_stats_from_array(
            cd59, radiance, transform, logger, id_col='boro_cd'
        )
        
        yearly_stats[year] = stats_df
        
        logger.info(f"Year {year} stats computed", extra={
            "mean_radiance": float(stats_df['radiance_mean'].mean()),
            "min_radiance": float(stats_df['radiance_mean'].min()),
            "max_radiance": float(stats_df['radiance_mean'].max()),
            "avg_nodata_pct": float(stats_df['nodata_pct'].mean())
        })
    
    # Compute multi-year averages
    logger.info("Computing multi-year averages")
    
    # Primary: 2021-2023 average
    primary_radiance = pd.concat([
        yearly_stats[y][['boro_cd', 'radiance_mean']].rename(
            columns={'radiance_mean': f'radiance_{y}'}
        ) for y in primary_years
    ], axis=1)
    # Remove duplicate boro_cd columns
    primary_radiance = primary_radiance.loc[:, ~primary_radiance.columns.duplicated()]
    
    radiance_cols = [f'radiance_{y}' for y in primary_years]
    
    # Primary: 2021-2023 average -> radiance_raw (contract-required name)
    primary_radiance['radiance_raw'] = primary_radiance[radiance_cols].mean(axis=1)
    
    # Sensitivity: 2022-2023 average -> radiance_raw_sensitivity
    sensitivity_cols = [f'radiance_{y}' for y in sensitivity_years]
    primary_radiance['radiance_raw_sensitivity'] = primary_radiance[sensitivity_cols].mean(axis=1)
    
    # Build final output DataFrame
    result_df = cd59[['boro_cd']].copy()
    result_df = result_df.merge(
        primary_radiance[['boro_cd', 'radiance_raw', 'radiance_raw_sensitivity'] + radiance_cols],
        on='boro_cd',
        how='left'
    )
    
    # Add pixel count and nodata info from most recent year (2023)
    result_df = result_df.merge(
        yearly_stats[max(primary_years)][['boro_cd', 'pixel_count', 'valid_pixel_count', 'nodata_pct']],
        on='boro_cd',
        how='left'
    )
    
    # Apply log1p transformation (contract-required: radiance_log1p)
    result_df['radiance_log1p'] = np.log1p(result_df['radiance_raw'])
    result_df['radiance_log1p_sensitivity'] = np.log1p(result_df['radiance_raw_sensitivity'])
    
    # Standardize metrics (higher light = worse = higher score)
    # Per contract: robust z (headline), classic z, percentile rank
    result_df = standardize_metrics(result_df, 'radiance_raw')
    result_df = standardize_metrics(result_df, 'radiance_log1p')
    
    # Rename for consistency with domain naming convention
    # Contract: robust z, classic z, percentile rank for both raw and log1p
    result_df = result_df.rename(columns={
        'z_radiance_raw_robust': 'z_light_robust',
        'z_radiance_raw_classic': 'z_light_classic',
        'pct_radiance_raw': 'pct_light',
        'z_radiance_log1p_robust': 'z_light_log1p_robust',
        'z_radiance_log1p_classic': 'z_light_log1p_classic',
        'pct_radiance_log1p': 'pct_light_log1p'
    })
    
    # Add metadata columns
    result_df['year_start'] = params['time_windows']['primary']['year_start']
    result_df['year_end'] = params['time_windows']['primary']['year_end']
    result_df['domain'] = 'light_viirs'
    result_df['source_id'] = 'VNP46A4_v002_h10v04'
    result_df['units'] = 'nWatts_per_cm2_sr'
    
    # Ensure boro_cd is Int64
    result_df = ensure_boro_cd_dtype(result_df)
    
    # Log summary statistics
    logger.log_metrics({
        "rows": len(result_df),
        "radiance_raw_min": float(result_df['radiance_raw'].min()),
        "radiance_raw_max": float(result_df['radiance_raw'].max()),
        "radiance_raw_mean": float(result_df['radiance_raw'].mean()),
        "na_count": int(result_df['radiance_raw'].isna().sum()),
        "avg_nodata_pct": float(result_df['nodata_pct'].mean()),
        "avg_valid_pixels": float(result_df['valid_pixel_count'].mean())
    })
    
    # QA: Check directionality (higher radiance should correlate with higher z-score)
    raw_vals = result_df['radiance_raw'].dropna()
    std_vals = result_df['z_light_robust'].dropna()
    if len(raw_vals) >= 3:
        correlation = raw_vals.corr(std_vals)
        if correlation < 0.8:
            logger.warning("Directionality check: weak correlation", extra={
                "correlation": float(correlation)
            })
        else:
            logger.info("Directionality check passed", extra={
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
    output_path = PROCESSED_DIR / "context" / "light_viirs_cd.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    atomic_write_df(result_df, output_path)
    
    logger.log_outputs({"light_viirs_cd": str(output_path)})
    
    # Write metadata sidecar
    metadata = {
        'script': SCRIPT_NAME,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'inputs': {
            **{f'viirs_{year}': str(path) for year, path in hdf_files.items()},
            **{f'viirs_{year}_sha256': input_hashes[f'viirs_{year}'] for year in hdf_files.keys()},
            'cd59': str(cd59_path)
        },
        'outputs': {
            'light_viirs_cd': str(output_path)
        },
        'time_windows': {
            'primary': {'years': primary_years, 'label': params['time_windows']['primary']['label']},
            'sensitivity': {'years': sensitivity_years, 'label': params['time_windows']['sensitivity']['label']}
        },
        'stats': {
            'n_districts': len(result_df),
            'radiance_raw_range': [
                float(result_df['radiance_raw'].min()),
                float(result_df['radiance_raw'].max())
            ],
            'na_count': int(result_df['radiance_raw'].isna().sum()),
            'avg_nodata_pct': float(result_df['nodata_pct'].mean()),
            'avg_valid_pixels': float(result_df['valid_pixel_count'].mean())
        },
        'raster_metadata': {
            'product': 'VNP46A4',
            'collection': '002',
            'tile': 'h10v04',
            'crs': 'EPSG:4326',
            'resolution_arcsec': 15,
            'resolution_approx_m': 450,
            'layer': 'AllAngle_Composite_Snow_Free',
            'units': 'nWatts/(cm^2 sr)',
            'fill_value': FILL_VALUE,
            'valid_min': VALID_MIN,
            'years_processed': list(raster_metadata.keys())
        },
        'directionality': 'higher_light_worse',
        'dependencies': {
            'h5py': h5py.__version__,
            'note': 'Requires h5py>=3.10 and libgdal-hdf5>=3.8 for HDF5/HDF-EOS2 support'
        },
        'git': get_git_info()
    }
    
    metadata_path = PROCESSED_DIR / "metadata" / "light_viirs_cd_metadata.json"
    atomic_write_json(metadata, metadata_path)
    
    logger.info("Script complete", extra={
        "output": str(output_path),
        "rows": len(result_df)
    })
    
    # Print summary
    print(f"\n✓ Light-at-night metrics computed for {len(result_df)} community districts")
    print(f"  Output: {output_path}")
    print(f"\n  Radiance range (2021-2023 avg): {result_df['radiance_raw'].min():.2f} - {result_df['radiance_raw'].max():.2f} nW/cm²sr")
    print(f"  Mean radiance: {result_df['radiance_raw'].mean():.2f} nW/cm²sr")
    print(f"  Avg nodata %: {result_df['nodata_pct'].mean():.1f}%")
    
    # Top 5 brightest CDs
    print(f"\n  Top 5 brightest CDs (highest light exposure):")
    top5 = result_df.nlargest(5, 'radiance_raw')[['boro_cd', 'radiance_raw', 'z_light_robust']]
    for _, row in top5.iterrows():
        print(f"    CD {int(row['boro_cd'])}: {row['radiance_raw']:.2f} nW/cm²sr (z={row['z_light_robust']:.2f})")
    
    # Bottom 5 darkest CDs
    print(f"\n  Top 5 darkest CDs (lowest light exposure):")
    bottom5 = result_df.nsmallest(5, 'radiance_raw')[['boro_cd', 'radiance_raw', 'z_light_robust']]
    for _, row in bottom5.iterrows():
        print(f"    CD {int(row['boro_cd'])}: {row['radiance_raw']:.2f} nW/cm²sr (z={row['z_light_robust']:.2f})")


if __name__ == "__main__":
    main()
