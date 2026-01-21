#!/usr/bin/env python3
"""
06_build_heat.py

Build nighttime heat exposure metrics from PRISM daily Tmin (minimum temperature).
This script computes warm-season heat metrics per Community District.

Inputs:
    - PRISM daily Tmin rasters (4km) for May-Sep 2021-2023
    - data/processed/geo/cd59.parquet

Outputs:
    - data/processed/domains/heat_cd.parquet
    - data/processed/metadata/heat_cd_metadata.json

Per Project_Context.md Section 10.2.2:
    - Primary: Jun-Aug 2021-2023 (mean Tmin)
    - Sensitivity: May-Sep 2021-2023
    - Hot nights: ≥20°C, ≥24°C, ≥p90 (NYC-relative)
    - Directionality: higher Tmin / more hot nights = worse
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import zipfile
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from shapely.geometry import mapping
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from sleep_esi.paths import RAW_DIR, PROCESSED_DIR, CONFIG_DIR, LOGS_DIR, GEO_DIR
from sleep_esi.logging_utils import get_logger
from sleep_esi.io_utils import read_yaml, read_gdf, atomic_write_df, atomic_write_json
from sleep_esi.hashing import hash_file, get_git_info
from sleep_esi.schemas import ensure_boro_cd_dtype

# Constants
SCRIPT_NAME = "06_build_heat"
PRISM_DIR = RAW_DIR / "heat_prism"

# PRISM data specs
PRISM_NODATA = -9999.0
PRISM_CRS = "EPSG:4269"  # NAD83


def get_prism_files(year: int, months: List[int]) -> List[Path]:
    """
    Get list of PRISM zip files for a given year and months.
    
    Args:
        year: Year (e.g., 2021)
        months: List of months (e.g., [5, 6, 7, 8, 9])
        
    Returns:
        List of zip file paths sorted by date
    """
    year_dir = PRISM_DIR / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"PRISM data directory not found: {year_dir}")
    
    files = []
    for month in months:
        # PRISM filename pattern: prism_tmin_us_25m_YYYYMMDD.zip
        pattern = f"prism_tmin_us_25m_{year}{month:02d}*.zip"
        month_files = sorted(year_dir.glob(pattern))
        files.extend(month_files)
    
    return sorted(files)


def parse_date_from_filename(filepath: Path) -> date:
    """Extract date from PRISM filename."""
    # Pattern: prism_tmin_us_25m_YYYYMMDD.zip
    name = filepath.stem
    date_str = name.split("_")[-1]  # Get YYYYMMDD part
    return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))


def read_prism_tif_from_zip(zip_path: Path, temp_dir: Path) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    """
    Extract and read PRISM TIF from zip file.
    
    Args:
        zip_path: Path to PRISM zip file
        temp_dir: Temporary directory for extraction
        
    Returns:
        Tuple of (data array, transform, metadata dict)
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Find the .tif file in the archive
        tif_names = [n for n in z.namelist() if n.endswith('.tif')]
        if not tif_names:
            raise ValueError(f"No TIF file found in {zip_path}")
        
        tif_name = tif_names[0]
        z.extract(tif_name, temp_dir)
        tif_path = temp_dir / tif_name
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        meta = {
            'crs': src.crs,
            'bounds': src.bounds,
            'nodata': src.nodata,
            'shape': data.shape
        }
    
    # Clean up extracted file
    tif_path.unlink()
    
    return data, transform, meta


def compute_zonal_mean_from_raster(
    gdf: gpd.GeoDataFrame,
    data: np.ndarray,
    transform: rasterio.Affine,
    raster_crs,
    nodata: float,
    id_col: str = 'boro_cd'
) -> Dict[int, float]:
    """
    Compute zonal mean for each polygon.
    
    Returns dict mapping boro_cd -> mean value (or NaN if no valid pixels)
    """
    from rasterio.io import MemoryFile
    from rasterio.features import rasterize
    
    results = {}
    height, width = data.shape
    
    # For each polygon, we'll rasterize it and use as a mask
    for idx, row in gdf.iterrows():
        boro_cd = row[id_col]
        geom = row['geometry']
        
        try:
            # Rasterize the polygon to create a mask
            # This avoids the CRS complexity of using rasterio.mask
            poly_mask = rasterize(
                [(geom, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )
            
            # Extract values where mask == 1
            polygon_pixels = data[poly_mask == 1]
            
            # Filter out nodata values
            valid_mask = polygon_pixels != nodata
            valid_data = polygon_pixels[valid_mask]
            
            if len(valid_data) > 0:
                results[boro_cd] = float(np.mean(valid_data))
            else:
                results[boro_cd] = np.nan
                
        except Exception as e:
            results[boro_cd] = np.nan
    
    return results


def standardize_metrics(df: pd.DataFrame, col: str, higher_is_worse: bool = True) -> pd.DataFrame:
    """
    Compute standardized versions of a metric.
    
    For heat: higher Tmin = worse = higher score (positive direction)
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
    
    # Get configuration
    primary_years = list(range(
        params['time_windows']['primary']['year_start'],
        params['time_windows']['primary']['year_end'] + 1
    ))
    sensitivity_years = list(range(
        params['time_windows']['sensitivity']['year_start'],
        params['time_windows']['sensitivity']['year_end'] + 1
    ))
    
    primary_months = params['warm_season']['primary']['months']
    sensitivity_months = params['warm_season']['sensitivity']['months']
    
    threshold_20c = params['hot_night_thresholds']['primary_c']
    threshold_24c = params['hot_night_thresholds']['sensitivity_c']
    threshold_pct = params['hot_night_thresholds']['percentile']
    
    logger.info("Configuration loaded", extra={
        "primary_years": primary_years,
        "sensitivity_years": sensitivity_years,
        "primary_months": primary_months,
        "sensitivity_months": sensitivity_months,
        "threshold_20c": threshold_20c,
        "threshold_24c": threshold_24c,
        "threshold_pct": threshold_pct
    })
    
    # Load CD59 geometries
    cd59_path = PROCESSED_DIR / "geo" / "cd59.parquet"
    cd59 = read_gdf(cd59_path)
    logger.info("CD59 loaded", extra={"rows": len(cd59), "crs": str(cd59.crs)})
    
    # PRISM uses EPSG:4269 (NAD83), CD59 uses EPSG:4326 (WGS84)
    # For NYC, these are practically identical (< 1m difference)
    # We'll use CD59 in EPSG:4326 directly to avoid reprojection issues
    cd59_reproj = cd59
    logger.info("Using CD59 in EPSG:4326 (compatible with PRISM EPSG:4269 for NYC)")
    
    # Collect all files and verify coverage
    all_files = []
    input_hashes = {}
    
    for year in primary_years:
        year_files = get_prism_files(year, sensitivity_months)  # Get all May-Sep for counting
        all_files.extend(year_files)
        logger.info(f"Found {len(year_files)} files for {year}")
        
        # Hash first and last file of each year for provenance
        if year_files:
            input_hashes[f"prism_{year}_first"] = hash_file(year_files[0])
            input_hashes[f"prism_{year}_last"] = hash_file(year_files[-1])
    
    total_files = len(all_files)
    expected_files = len(primary_years) * sum(
        [31, 30, 31, 31, 30]  # May, Jun, Jul, Aug, Sep days
    )
    
    logger.info("File inventory", extra={
        "total_files": total_files,
        "expected_files": expected_files,
        "coverage_pct": total_files / expected_files * 100 if expected_files > 0 else 0
    })
    
    if total_files < expected_files * 0.95:
        logger.warning("Missing some PRISM files", extra={
            "found": total_files,
            "expected": expected_files
        })
    
    # Process all files and collect daily Tmin per CD
    # Structure: daily_tmin[boro_cd][date] = tmin_value
    daily_tmin = defaultdict(dict)
    
    # Use a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for i, zip_path in enumerate(all_files):
            file_date = parse_date_from_filename(zip_path)
            
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"Processing file {i+1}/{total_files}", extra={
                    "file": zip_path.name,
                    "date": str(file_date)
                })
            
            try:
                data, transform, meta = read_prism_tif_from_zip(zip_path, temp_path)
                
                # Compute zonal means
                zonal_means = compute_zonal_mean_from_raster(
                    cd59_reproj, data, transform, meta['crs'], PRISM_NODATA
                )
                
                for boro_cd, tmin in zonal_means.items():
                    daily_tmin[boro_cd][file_date] = tmin
                    
            except Exception as e:
                logger.warning(f"Error processing {zip_path.name}", extra={
                    "error": str(e)
                })
    
    logger.info("Daily extraction complete", extra={
        "districts_processed": len(daily_tmin),
        "total_observations": sum(len(v) for v in daily_tmin.values())
    })
    
    # Compute metrics per CD
    results = []
    
    # First pass: collect all valid Tmin values for p90 calculation
    all_tmin_values = []
    for boro_cd, dates_dict in daily_tmin.items():
        for d, tmin in dates_dict.items():
            if not np.isnan(tmin) and d.month in primary_months:
                all_tmin_values.append(tmin)
    
    # Calculate p90 threshold (NYC-relative)
    if all_tmin_values:
        p90_threshold = np.percentile(all_tmin_values, threshold_pct)
    else:
        p90_threshold = np.nan
    
    logger.info("P90 threshold calculated", extra={
        "p90_threshold_c": float(p90_threshold) if not np.isnan(p90_threshold) else None,
        "n_observations": len(all_tmin_values)
    })
    
    # Second pass: compute metrics per CD
    for boro_cd in cd59['boro_cd'].values:
        cd_data = daily_tmin.get(boro_cd, {})
        
        # Filter to primary warm season (Jun-Aug) for mean Tmin
        primary_tmin = [
            tmin for d, tmin in cd_data.items()
            if d.month in primary_months and d.year in primary_years and not np.isnan(tmin)
        ]
        
        # Sensitivity warm season (May-Sep)
        sensitivity_tmin = [
            tmin for d, tmin in cd_data.items()
            if d.month in sensitivity_months and d.year in primary_years and not np.isnan(tmin)
        ]
        
        # Mean Tmin (primary and sensitivity)
        tmin_mean_primary = np.mean(primary_tmin) if primary_tmin else np.nan
        tmin_mean_sensitivity = np.mean(sensitivity_tmin) if sensitivity_tmin else np.nan
        
        # Hot nights counts (primary warm season only)
        hot_nights_20c = sum(1 for t in primary_tmin if t >= threshold_20c)
        hot_nights_24c = sum(1 for t in primary_tmin if t >= threshold_24c)
        hot_nights_p90 = sum(1 for t in primary_tmin if t >= p90_threshold) if not np.isnan(p90_threshold) else 0
        
        # Days count for rate calculation
        n_days_primary = len(primary_tmin)
        n_days_sensitivity = len(sensitivity_tmin)
        
        # Hot nights per year (normalized)
        years_covered = len(primary_years)
        hot_nights_20c_per_year = hot_nights_20c / years_covered if years_covered > 0 else 0
        hot_nights_24c_per_year = hot_nights_24c / years_covered if years_covered > 0 else 0
        hot_nights_p90_per_year = hot_nights_p90 / years_covered if years_covered > 0 else 0
        
        results.append({
            'boro_cd': boro_cd,
            'tmin_mean_primary': tmin_mean_primary,
            'tmin_mean_sensitivity': tmin_mean_sensitivity,
            'hot_nights_20c': hot_nights_20c,
            'hot_nights_24c': hot_nights_24c,
            'hot_nights_p90': hot_nights_p90,
            'hot_nights_20c_per_year': hot_nights_20c_per_year,
            'hot_nights_24c_per_year': hot_nights_24c_per_year,
            'hot_nights_p90_per_year': hot_nights_p90_per_year,
            'n_days_primary': n_days_primary,
            'n_days_sensitivity': n_days_sensitivity
        })
    
    result_df = pd.DataFrame(results)
    
    # Log summary before standardization
    logger.log_metrics({
        "rows": len(result_df),
        "tmin_mean_min": float(result_df['tmin_mean_primary'].min()),
        "tmin_mean_max": float(result_df['tmin_mean_primary'].max()),
        "tmin_mean_mean": float(result_df['tmin_mean_primary'].mean()),
        "hot_nights_20c_total": int(result_df['hot_nights_20c'].sum()),
        "hot_nights_24c_total": int(result_df['hot_nights_24c'].sum()),
        "hot_nights_p90_total": int(result_df['hot_nights_p90'].sum()),
        "na_count": int(result_df['tmin_mean_primary'].isna().sum())
    })
    
    # Standardize metrics (higher = worse for heat)
    result_df = standardize_metrics(result_df, 'tmin_mean_primary')
    result_df = standardize_metrics(result_df, 'hot_nights_20c_per_year')
    
    # Rename for clarity
    result_df = result_df.rename(columns={
        'z_tmin_mean_primary_robust': 'z_heat_tmin_robust',
        'z_tmin_mean_primary_classic': 'z_heat_tmin_classic',
        'pct_tmin_mean_primary': 'pct_heat_tmin',
        'z_hot_nights_20c_per_year_robust': 'z_heat_hotnights_robust',
        'z_hot_nights_20c_per_year_classic': 'z_heat_hotnights_classic',
        'pct_hot_nights_20c_per_year': 'pct_heat_hotnights'
    })
    
    # Add metadata columns
    result_df['year_start'] = params['time_windows']['primary']['year_start']
    result_df['year_end'] = params['time_windows']['primary']['year_end']
    result_df['domain'] = 'heat'
    result_df['source_id'] = 'PRISM_tmin_4km_daily'
    result_df['units'] = 'celsius'
    result_df['p90_threshold_c'] = p90_threshold
    
    # Ensure boro_cd is Int64
    result_df = ensure_boro_cd_dtype(result_df)
    
    # QA: Check directionality
    tmin_vals = result_df['tmin_mean_primary'].dropna()
    z_vals = result_df['z_heat_tmin_robust'].dropna()
    if len(tmin_vals) >= 3:
        correlation = tmin_vals.corr(z_vals)
        logger.info("Directionality check", extra={"correlation": float(correlation)})
        if correlation < 0.8:
            logger.warning("Directionality check: weak correlation", extra={
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
    output_path = PROCESSED_DIR / "context" / "heat_cd.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    atomic_write_df(result_df, output_path)
    logger.log_outputs({"heat_cd": str(output_path)})
    
    # Write metadata sidecar
    metadata = {
        'script': SCRIPT_NAME,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'inputs': {
            'prism_dir': str(PRISM_DIR),
            'cd59': str(cd59_path),
            **{k: v for k, v in input_hashes.items()}
        },
        'outputs': {
            'heat_cd': str(output_path)
        },
        'time_windows': {
            'primary': {
                'years': primary_years,
                'months': primary_months,
                'label': params['warm_season']['primary']['label']
            },
            'sensitivity': {
                'years': primary_years,
                'months': sensitivity_months,
                'label': params['warm_season']['sensitivity']['label']
            }
        },
        'thresholds': {
            'hot_night_20c': threshold_20c,
            'hot_night_24c': threshold_24c,
            'hot_night_p90': float(p90_threshold) if not np.isnan(p90_threshold) else None,
            'percentile_used': threshold_pct
        },
        'stats': {
            'n_districts': len(result_df),
            'tmin_mean_range': [
                float(result_df['tmin_mean_primary'].min()),
                float(result_df['tmin_mean_primary'].max())
            ],
            'hot_nights_20c_total': int(result_df['hot_nights_20c'].sum()),
            'hot_nights_24c_total': int(result_df['hot_nights_24c'].sum()),
            'hot_nights_p90_total': int(result_df['hot_nights_p90'].sum()),
            'na_count': int(result_df['tmin_mean_primary'].isna().sum()),
            'files_processed': total_files
        },
        'raster_metadata': {
            'product': 'PRISM Daily Tmin',
            'version': 'stable (25m identifier)',
            'crs': PRISM_CRS,
            'resolution_km': 4,
            'nodata': PRISM_NODATA,
            'units': 'Celsius',
            'source_url': 'https://data.prism.oregonstate.edu'
        },
        'directionality': 'higher_heat_worse',
        'git': get_git_info()
    }
    
    metadata_path = PROCESSED_DIR / "metadata" / "heat_cd_metadata.json"
    atomic_write_json(metadata, metadata_path)
    
    logger.info("Script complete", extra={
        "output": str(output_path),
        "rows": len(result_df)
    })
    
    # Print summary
    print(f"\n✓ Heat metrics computed for {len(result_df)} community districts")
    print(f"  Output: {output_path}")
    print(f"\n  Mean Tmin (Jun-Aug, 2021-2023): {result_df['tmin_mean_primary'].min():.1f}°C - {result_df['tmin_mean_primary'].max():.1f}°C")
    print(f"  NYC mean Tmin: {result_df['tmin_mean_primary'].mean():.1f}°C")
    print(f"  P90 threshold: {p90_threshold:.1f}°C")
    print(f"\n  Hot nights (≥20°C) total: {result_df['hot_nights_20c'].sum():,}")
    print(f"  Hot nights (≥24°C) total: {result_df['hot_nights_24c'].sum():,}")
    print(f"  Hot nights (≥p90) total: {result_df['hot_nights_p90'].sum():,}")
    
    # Top 5 hottest CDs
    print(f"\n  Top 5 hottest CDs (highest mean Tmin):")
    top5 = result_df.nlargest(5, 'tmin_mean_primary')[['boro_cd', 'tmin_mean_primary', 'hot_nights_20c_per_year']]
    for _, row in top5.iterrows():
        print(f"    CD {int(row['boro_cd'])}: {row['tmin_mean_primary']:.1f}°C ({row['hot_nights_20c_per_year']:.0f} hot nights/yr)")
    
    # Bottom 5 coolest CDs
    print(f"\n  Top 5 coolest CDs (lowest mean Tmin):")
    bottom5 = result_df.nsmallest(5, 'tmin_mean_primary')[['boro_cd', 'tmin_mean_primary', 'hot_nights_20c_per_year']]
    for _, row in bottom5.iterrows():
        print(f"    CD {int(row['boro_cd'])}: {row['tmin_mean_primary']:.1f}°C ({row['hot_nights_20c_per_year']:.0f} hot nights/yr)")


if __name__ == "__main__":
    main()
