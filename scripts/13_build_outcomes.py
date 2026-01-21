#!/usr/bin/env python3
"""
13_build_outcomes.py

Analyze associations between night signatures (clusters) and outcomes:
- Sleep deprivation (CDC PLACES)
- Social inequality (ACS demographics)

Per NYC_Night_Signals_Plan.md Section 3.3 (Script 13):
- Join clusters with sleep proxies and demographics
- Report effect sizes + uncertainty
- Use clusters as grouping variables

Outputs:
- data/processed/outcomes/cd_outcomes.parquet
- data/processed/outcomes/nta_outcomes.parquet
- data/processed/outcomes/cluster_outcome_summary.csv
- data/processed/metadata/outcomes_metadata.json
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sleep_esi.paths import RAW_DIR, PROCESSED_DIR, CONFIG_DIR, LOGS_DIR, GEO_DIR, XWALK_DIR
from sleep_esi.logging_utils import get_logger
from sleep_esi.io_utils import read_yaml, read_df, atomic_write_df, atomic_write_json
from sleep_esi.hashing import hash_file, get_git_info, write_metadata_sidecar
from sleep_esi.schemas import ensure_boro_cd_dtype

# Constants
SCRIPT_NAME = "13_build_outcomes"

ACS_TABLES = {
    'B01003': ['B01003_001E'],  # Total population
    'B17001': ['B17001_001E', 'B17001_002E'],  # Poverty universe, below poverty
    'B02001': ['B02001_003E'],  # Black alone
    'B03003': ['B03003_003E'],  # Hispanic
    'B25070': [  # Rent burden (gross rent as % of income)
        'B25070_001E',  # Total renter households
        'B25070_007E',  # 30-34.9%
        'B25070_008E',  # 35-39.9%
        'B25070_009E',  # 40-49.9%
        'B25070_010E',  # 50% or more
    ],
}

ACS_VARIABLES = [v for vars in ACS_TABLES.values() for v in vars]
NYC_COUNTIES = ['005', '047', '061', '081', '085']

def fetch_acs_data(logger) -> pd.DataFrame:
    """Fetch ACS 5-year 2022 data at tract level for NYC."""
    variables = ','.join(ACS_VARIABLES)
    counties = ','.join(NYC_COUNTIES)
    
    url = (
        f"https://api.census.gov/data/2022/acs/acs5"
        f"?get={variables}"
        f"&for=tract:*"
        f"&in=state:36"
        f"&in=county:{counties}"
    )
    
    logger.info("Fetching ACS data", extra={"url": url[:100] + "..."})
    
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    
    data = response.json()
    headers = data[0]
    rows = data[1:]
    
    df = pd.DataFrame(rows, columns=headers)
    df['geoid'] = '36' + df['county'] + df['tract']
    
    for col in ACS_VARIABLES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def compute_tract_demographics(acs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute demographic rates at tract level."""
    result = acs_df[['geoid']].copy()
    result['population'] = acs_df['B01003_001E']
    
    # Poverty rate
    result['poverty_rate'] = np.where(
        acs_df['B17001_001E'] > 0,
        acs_df['B17001_002E'] / acs_df['B17001_001E'],
        np.nan
    )
    
    # % Black
    result['pct_black'] = np.where(
        result['population'] > 0,
        acs_df['B02001_003E'] / result['population'],
        np.nan
    )
    
    # % Hispanic
    result['pct_hispanic'] = np.where(
        result['population'] > 0,
        acs_df['B03003_003E'] / result['population'],
        np.nan
    )
    
    # Rent burden rate
    rent_burden_cols = ['B25070_007E', 'B25070_008E', 'B25070_009E', 'B25070_010E']
    result['rent_burden_rate'] = np.where(
        acs_df['B25070_001E'] > 0,
        acs_df[rent_burden_cols].sum(axis=1) / acs_df['B25070_001E'],
        np.nan
    )
    
    return result

def load_sleep_data(logger) -> pd.DataFrame:
    """Load CDC PLACES sleep data at tract level."""
    sleep_files = list((RAW_DIR / "cdc_places").glob("places_nyc_tracts_*.csv"))
    if not sleep_files:
        raise FileNotFoundError("No sleep data found in data/raw/cdc_places")
    
    sleep_path = max(sleep_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading sleep data from {sleep_path}")
    return pd.read_csv(sleep_path, dtype={'tract_geoid': str})

def load_tract_nta_mapping(logger) -> pd.DataFrame:
    """Load tract-to-NTA mapping from raw tracts GeoJSON."""
    tract_files = list((RAW_DIR / "census_tracts_2020").glob("census_tracts_2020_*.geojson"))
    if not tract_files:
        raise FileNotFoundError("No tract GeoJSON found in data/raw/census_tracts_2020")
    
    tract_path = max(tract_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading tract mapping from {tract_path}")
    
    # Use fiona/geopandas to read just columns
    import geopandas as gpd
    tracts = gpd.read_file(tract_path)
    
    # Ensure geoid is 11-digit string
    # The API GeoJSON often has 'geoid' or 'geoid20'
    geoid_col = 'geoid' if 'geoid' in tracts.columns else 'geoid20'
    if geoid_col not in tracts.columns:
        # Construct from state+county+tract if needed
        tracts['geoid'] = '36' + tracts['countyfp'] + tracts['tractce']
        geoid_col = 'geoid'
        
    return tracts[[geoid_col, 'nta2020']].rename(columns={geoid_col: 'geoid', 'nta2020': 'ntacode'})

def aggregate_to_geo(
    tract_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    geo_id_col: str,
    weight_col: str = 'w_pop',
    pop_col: str = 'tract_pop',
    logger = None
) -> pd.DataFrame:
    """
    Aggregate tract-level data to a target geography using population-weighted averages.

    For rates (poverty_rate, pct_black, etc.), we compute:
        geo_rate = sum(tract_rate * effective_pop) / sum(effective_pop)

    The crosswalk provides intersection populations (tract_pop) which account for
    tracts that are split across multiple target geographies.

    Args:
        tract_df: DataFrame with 'geoid', 'population', and rate columns
        mapping_df: DataFrame with 'geoid', geo_id_col, weight_col, and optionally pop_col
        geo_id_col: Column name for target geography ID (e.g., 'boro_cd', 'ntacode')
        weight_col: Column with fractional overlap weights (default: 'w_pop')
        pop_col: Column with intersection population (default: 'tract_pop')
        logger: Optional logger instance

    Returns:
        DataFrame with aggregated values for each target geography

    Notes:
        - For CD aggregation: Uses 'tract_pop' from crosswalk (intersection population)
        - For NTA aggregation: Uses 'population' from tract_df (tracts nest in NTAs)
        - Rates are assumed uniform across each tract, weighted by effective population
    """
    # Ensure mapping_df doesn't have 'population' to avoid column collision
    mapping_clean = mapping_df.copy()
    if 'population' in mapping_clean.columns:
        mapping_clean = mapping_clean.drop(columns=['population'])

    merged = mapping_clean.merge(tract_df, on='geoid', how='left')

    # List of rate columns to aggregate
    rate_cols = [c for c in tract_df.columns if c not in ['geoid', 'population']]

    results = []
    for geo_id, group in merged.groupby(geo_id_col):
        # Determine which population column to use for weighting
        # Priority: crosswalk's intersection pop (pop_col) > tract_df's population
        if pop_col in group.columns:
            # Use intersection population from crosswalk (handles split tracts)
            weights_source = group[pop_col].fillna(0).astype(float)
        else:
            # Fallback: use tract population * weight (for simple 1:1 mappings)
            weights_source = group['population'].fillna(0).astype(float) * group[weight_col].fillna(0).astype(float)

        geo_data = {geo_id_col: geo_id}
        
        # Calculate rates for each outcome column
        for col in rate_cols:
            # Filter to rows that have both a valid rate and a non-zero weight
            mask = group[col].notna() & (weights_source > 0)
            
            if mask.any():
                col_weights = weights_source[mask]
                col_values = group.loc[mask, col].astype(float)
                geo_data[col] = (col_values * col_weights).sum() / col_weights.sum()
                
                # Use total population of tracts with data as 'population_est' 
                # (first col only to avoid duplicates, or just track separately)
                if 'population_est' not in geo_data:
                    geo_data['population_est'] = col_weights.sum()
            else:
                geo_data[col] = np.nan
                if 'population_est' not in geo_data:
                    geo_data['population_est'] = 0.0

        results.append(geo_data)

    return pd.DataFrame(results)

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(SCRIPT_NAME, run_id)
    logger.info("Starting Script 13: Outcome Analysis")
    
    # 1. Load Data
    try:
        cd_clusters = read_df(PROCESSED_DIR / "typology" / "cd_clusters.parquet")
        nta_clusters = read_df(PROCESSED_DIR / "typology" / "nta_clusters_residential.parquet")
        sleep_df = load_sleep_data(logger)
        cd_xwalk = read_df(XWALK_DIR / "cd_to_tract_weights.parquet")
        tract_nta_map = load_tract_nta_mapping(logger)
    except Exception as e:
        logger.error(f"Failed to load inputs: {e}")
        sys.exit(1)
        
    # 2. Fetch ACS Demographics
    try:
        acs_raw = fetch_acs_data(logger)
        tract_demo = compute_tract_demographics(acs_raw)
    except Exception as e:
        logger.error(f"Failed to fetch ACS data: {e}")
        sys.exit(1)
        
    # 3. Merge Tract Data (Sleep + Demo)
    tract_all = tract_demo.merge(
        sleep_df.rename(columns={'tract_geoid': 'geoid'}),
        on='geoid',
        how='left'
    )
    
    # 4. Aggregate to CD
    logger.info("Aggregating to CD level...")
    cd_mapping = cd_xwalk.rename(columns={'tract_geoid': 'geoid'})
    # Uses 'tract_pop' from crosswalk (intersection population for split tracts)
    cd_outcomes = aggregate_to_geo(
        tract_all, cd_mapping, 'boro_cd',
        weight_col='w_pop', pop_col='tract_pop', logger=logger
    )
    cd_outcomes = ensure_boro_cd_dtype(cd_outcomes)

    # 5. Aggregate to NTA
    logger.info("Aggregating to NTA level...")
    # For NTAs, tracts nest perfectly in 2020 (no split tracts).
    # Use tract population directly as the weight.
    nta_mapping = tract_nta_map.copy()
    nta_mapping['w_nta'] = 1.0  # Simple 1:1 join, no fractional overlap
    # No 'tract_pop' column, so function will use population * w_nta = population
    nta_outcomes = aggregate_to_geo(
        tract_all, nta_mapping, 'ntacode',
        weight_col='w_nta', pop_col='tract_pop', logger=logger  # Will fallback to population * w
    )
    
    # 6. Join with Clusters
    cd_final = cd_clusters.merge(cd_outcomes, on='boro_cd', how='left')
    nta_final = nta_clusters.merge(nta_outcomes, on='ntacode', how='left')
    
    # 7. Compute Cluster Summary
    outcome_cols = ['sleep', 'poverty_rate', 'pct_black', 'pct_hispanic', 'rent_burden_rate']
    
    # CD level summary
    cd_summary = cd_final.groupby(['cluster_id', 'cluster_label'])[outcome_cols].agg(['mean', 'std', 'count']).reset_index()
    cd_summary.columns = ['_'.join(col).strip('_') for col in cd_summary.columns.values]
    cd_summary['level'] = 'CD'
    
    # NTA level summary
    nta_summary = nta_final.groupby(['cluster_id', 'cluster_label'])[outcome_cols].agg(['mean', 'std', 'count']).reset_index()
    nta_summary.columns = ['_'.join(col).strip('_') for col in nta_summary.columns.values]
    nta_summary['level'] = 'NTA'
    
    combined_summary = pd.concat([cd_summary, nta_summary], axis=0)
    
    # 8. Write Outputs
    outcomes_dir = PROCESSED_DIR / "outcomes"
    outcomes_dir.mkdir(parents=True, exist_ok=True)
    
    atomic_write_df(cd_final, outcomes_dir / "cd_outcomes.parquet")
    atomic_write_df(nta_final, outcomes_dir / "nta_outcomes.parquet")
    atomic_write_df(combined_summary, outcomes_dir / "cluster_outcome_summary.csv")
    
    # Metadata
    input_hashes = {
        'cd_clusters': hash_file(PROCESSED_DIR / "typology" / "cd_clusters.parquet"),
        'nta_clusters': hash_file(PROCESSED_DIR / "typology" / "nta_clusters_residential.parquet"),
        'sleep_data': hash_file(max((RAW_DIR / "cdc_places").glob("places_nyc_tracts_*.csv"), key=lambda p: p.stat().st_mtime))
    }
    
    metadata = {
        'script': SCRIPT_NAME,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'inputs': input_hashes,
        'outputs': {
            'cd_outcomes': str(outcomes_dir / "cd_outcomes.parquet"),
            'nta_outcomes': str(outcomes_dir / "nta_outcomes.parquet"),
            'cluster_outcome_summary': str(outcomes_dir / "cluster_outcome_summary.csv"),
        },
        'git': get_git_info()
    }
    
    atomic_write_json(metadata, PROCESSED_DIR / "metadata" / "outcomes_metadata.json")
    
    logger.info("Script 13 complete.")
    print("\n✓ Outcome analysis complete.")
    print(f"  Summary saved to {outcomes_dir / 'cluster_outcome_summary.csv'}")

if __name__ == "__main__":
    main()
