#!/usr/bin/env python3
"""
07_build_heat_sensitivity.py

Model associations between nighttime minimum temperature (Tmin) and 311 noise complaints.
Estimates heat sensitivity slopes per CD and citywide.

Per NYC_Night_Signals_Plan.md Section 3.3 (Script 07):
- Aggregate to night-level counts per CD
- Join PRISM Tmin for the corresponding date
- Model: nightly_count ~ Tmin + DOW + month + year
- Output per-CD slopes and citywide curve inputs

Outputs:
- data/processed/weather/cd_heat_sensitivity.parquet
- data/processed/weather/citywide_temp_curve.csv
- data/processed/metadata/heat_sensitivity_metadata.json
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
from rasterio.features import rasterize
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sleep_esi.paths import RAW_DIR, PROCESSED_DIR, CONFIG_DIR, LOGS_DIR, GEO_DIR
from sleep_esi.logging_utils import get_logger
from sleep_esi.io_utils import read_yaml, read_gdf, atomic_write_df, atomic_write_json
from sleep_esi.hashing import hash_file, get_git_info, write_metadata_sidecar
from sleep_esi.schemas import ensure_boro_cd_dtype
from sleep_esi.time_utils import ensure_nyc_timezone, filter_nighttime
from sleep_esi.joins import spatial_join_points_to_polygons, log_join_stats

# Constants
SCRIPT_NAME = "07_build_heat_sensitivity"
PRISM_DIR = RAW_DIR / "heat_prism"
RAW_311_DIR = RAW_DIR / "311_noise"
PRISM_NODATA = -9999.0

def get_prism_files(year: int, months: List[int]) -> List[Path]:
    year_dir = PRISM_DIR / str(year)
    if not year_dir.exists():
        return []
    
    files = []
    for month in months:
        pattern = f"prism_tmin_us_25m_{year}{month:02d}*.zip"
        files.extend(sorted(year_dir.glob(pattern)))
    return sorted(files)

def parse_date_from_filename(filepath: Path) -> date:
    name = filepath.stem
    date_str = name.split("_")[-1]
    return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))

def read_prism_tif_from_zip(zip_path: Path, temp_dir: Path) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    with zipfile.ZipFile(zip_path, 'r') as z:
        tif_names = [n for n in z.namelist() if n.endswith('.tif')]
        if not tif_names:
            raise ValueError(f"No TIF file found in {zip_path}")
        tif_name = tif_names[0]
        z.extract(tif_name, temp_dir)
        tif_path = temp_dir / tif_name
    
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        meta = {'crs': src.crs, 'bounds': src.bounds, 'nodata': src.nodata, 'shape': data.shape}
    
    tif_path.unlink()
    return data, transform, meta

def compute_zonal_mean(gdf: gpd.GeoDataFrame, data: np.ndarray, transform: rasterio.Affine, nodata: float) -> Dict[int, float]:
    results = {}
    height, width = data.shape
    for _, row in gdf.iterrows():
        boro_cd = row['boro_cd']
        geom = row['geometry']
        try:
            poly_mask = rasterize([(geom, 1)], out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8, all_touched=True)
            valid_data = data[(poly_mask == 1) & (data != nodata)]
            results[boro_cd] = float(np.mean(valid_data)) if len(valid_data) > 0 else np.nan
        except:
            results[boro_cd] = np.nan
    return results

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(SCRIPT_NAME, run_id)
    logger.info("Starting Script 07: Heat Sensitivity Analysis")
    
    params = read_yaml(CONFIG_DIR / "params.yml")
    years = [2021, 2022, 2023]
    months = [5, 6, 7, 8, 9] # Warm season focus
    
    # 1. Load Geographies
    cd59 = read_gdf(GEO_DIR / "cd59.parquet")
    cd59 = ensure_boro_cd_dtype(cd59)
    cd_lookup = pd.read_parquet(GEO_DIR / "cd_lookup.parquet")
    
    # 2. Get Nightly 311 Counts
    logger.info("Processing 311 counts...")
    raw_311_path = max(RAW_311_DIR.glob("raw_311_noise_*.csv"), key=lambda p: p.stat().st_mtime)
    df_311 = pd.read_csv(raw_311_path)
    df_311['created_date'] = pd.to_datetime(df_311['created_date'], errors='coerce')
    df_311 = df_311.dropna(subset=['created_date', 'latitude', 'longitude'])
    df_311['ts_nyc'] = ensure_nyc_timezone(df_311['created_date'])
    
    # Filter to nighttime
    night_start, night_end = 22, 7
    df_311 = filter_nighttime(df_311, "ts_nyc", start_hour=night_start, end_hour=night_end)
    
    # Assign night date (date the night ENDS, which matches Tmin date)
    df_311['night_date'] = np.where(df_311['ts_nyc'].dt.hour >= 22, 
                                    (df_311['ts_nyc'] + pd.Timedelta(days=1)).dt.date, 
                                    df_311['ts_nyc'].dt.date)
    
    # Spatial join
    gdf_311 = gpd.GeoDataFrame(df_311, geometry=gpd.points_from_xy(df_311.longitude, df_311.latitude), crs="EPSG:4326")
    joined, join_stats = spatial_join_points_to_polygons(gdf_311, cd59, polygon_id_col="boro_cd", max_distance=500)
    log_join_stats(join_stats, logger)
    joined = ensure_boro_cd_dtype(joined)
    
    # Aggregate to nightly counts per CD
    nightly_counts = joined.groupby(['night_date', 'boro_cd']).size().reset_index(name='count')
    nightly_counts['night_date'] = pd.to_datetime(nightly_counts['night_date']).dt.date
    
    # 3. Process PRISM Tmin
    logger.info("Extracting PRISM Tmin...")
    daily_tmin_data = []
    all_files = []
    for year in years:
        all_files.extend(get_prism_files(year, months))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for i, zip_path in enumerate(all_files):
            file_date = parse_date_from_filename(zip_path)
            if i % 100 == 0: logger.info(f"Processing PRISM {i}/{len(all_files)}")
            
            data, transform, meta = read_prism_tif_from_zip(zip_path, temp_path)
            zonal_means = compute_zonal_mean(cd59, data, transform, PRISM_NODATA)
            for boro_cd, tmin in zonal_means.items():
                daily_tmin_data.append({'night_date': file_date, 'boro_cd': boro_cd, 'tmin': tmin})
    
    tmin_df = pd.DataFrame(daily_tmin_data)
    tmin_df['night_date'] = pd.to_datetime(tmin_df['night_date']).dt.date
    
    # 4. Merge
    logger.info("Merging counts and weather...")
    # Create master backbone for all CD x dates to include zeros
    dates = pd.date_range(start=f"{years[0]}-05-01", end=f"{years[-1]}-09-30").date
    backbone = pd.MultiIndex.from_product([dates, cd59['boro_cd'].unique()], names=['night_date', 'boro_cd']).to_frame(index=False)
    
    final_df = backbone.merge(nightly_counts, on=['night_date', 'boro_cd'], how='left')
    final_df['count'] = final_df['count'].fillna(0)
    final_df = final_df.merge(tmin_df, on=['night_date', 'boro_cd'], how='inner') # Keep only dates with Tmin
    
    # Add temporal controls
    final_df['dt'] = pd.to_datetime(final_df['night_date'])
    final_df['dow'] = final_df['dt'].dt.dayofweek
    final_df['month'] = final_df['dt'].dt.month
    final_df['year'] = final_df['dt'].dt.year
    
    # 5. Run Regressions
    logger.info("Running regressions per CD...")
    sensitivity_results = []
    
    for boro_cd, group in final_df.groupby('boro_cd'):
        try:
            # Poisson regression
            # nightly_count ~ tmin + C(dow) + C(month) + C(year)
            model = smf.glm("count ~ tmin + C(dow) + C(month) + C(year)", 
                            data=group, 
                            family=sm.families.Poisson()).fit()
            
            # Extract slope for Tmin
            slope = model.params['tmin']
            se = model.bse['tmin']
            pval = model.pvalues['tmin']
            
            # Percent increase per 1°C = (exp(slope) - 1) * 100
            pct_increase = (np.exp(slope) - 1) * 100
            
            sensitivity_results.append({
                'boro_cd': boro_cd,
                'slope': slope,
                'se': se,
                'pvalue': pval,
                'pct_increase_per_c': pct_increase,
                'n_obs': len(group)
            })
        except Exception as e:
            logger.warning(f"Regression failed for CD {boro_cd}: {e}")
            
    sens_df = pd.DataFrame(sensitivity_results)
    sens_df = ensure_boro_cd_dtype(sens_df)
    sens_df = sens_df.merge(cd_lookup[['boro_cd', 'cd_label']], on='boro_cd')
    
    # 6. Citywide Analysis
    logger.info("Running citywide analysis...")
    citywide_counts = final_df.groupby('night_date').agg({'count': 'sum', 'tmin': 'mean', 'dow': 'first', 'month': 'first', 'year': 'first'}).reset_index()
    citywide_model = smf.glm("count ~ tmin + C(dow) + C(month) + C(year)", 
                             data=citywide_counts, 
                             family=sm.families.Poisson()).fit()
    
    # Generate curve inputs
    t_min, t_max = citywide_counts['tmin'].min(), citywide_counts['tmin'].max()
    t_range = np.linspace(t_min, t_max, 100)
    # Predicted counts at mean of other vars
    pred_df = pd.DataFrame({'tmin': t_range, 'dow': 0, 'month': 7, 'year': 2022}) # Representative baseline
    pred_counts = citywide_model.predict(pred_df)
    curve_df = pd.DataFrame({'tmin': t_range, 'pred_count': pred_counts})
    
    # 7. Write Outputs
    weather_dir = PROCESSED_DIR / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)
    
    atomic_write_df(sens_df, weather_dir / "cd_heat_sensitivity.parquet")
    curve_df.to_csv(weather_dir / "citywide_temp_curve.csv", index=False)
    
    # Remove distance list to avoid bloating metadata (keep summary stats)
    join_stats_summary = {k: v for k, v in join_stats.items() if k != 'distances'}

    metadata = {
        'script': SCRIPT_NAME,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'inputs': {
            'prism_dir': str(PRISM_DIR),
            'raw_311': str(raw_311_path)
        },
        'outputs': {
            'cd_heat_sensitivity': str(weather_dir / "cd_heat_sensitivity.parquet"),
            'citywide_temp_curve': str(weather_dir / "citywide_temp_curve.csv")
        },
        'stats': {
            'citywide_slope': float(citywide_model.params['tmin']),
            'citywide_pct_increase': float((np.exp(citywide_model.params['tmin']) - 1) * 100)
        },
        'join_stats': join_stats_summary,
        'git': get_git_info()
    }
    atomic_write_json(metadata, PROCESSED_DIR / "metadata" / "heat_sensitivity_metadata.json")
    
    logger.info("Script 07 complete.")
    print("\n✓ Heat sensitivity analysis complete.")
    print(f"  Citywide pct increase per 1°C: {(np.exp(citywide_model.params['tmin']) - 1) * 100:.1f}%")

if __name__ == "__main__":
    main()
