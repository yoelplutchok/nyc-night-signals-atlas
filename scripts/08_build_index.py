#!/usr/bin/env python3
"""
08_build_index.py

Assemble composite Environmental Sleep Indices from domain metrics.

Inputs:
    - data/processed/domains/noise_obj_full_cd.parquet (objective noise)
    - data/processed/domains/noise_311_cd.parquet (RDI/311)
    - data/processed/domains/light_viirs_cd.parquet (light)
    - data/processed/domains/heat_cd.parquet (heat)
    - data/processed/domains/air_cd.parquet (air)
    - configs/weights.yml

Outputs:
    - data/processed/index/esi_cd.parquet (all index variants)
    - data/processed/index/esi_rank_movement.csv (rank changes across variants)
    - data/processed/index/esi_rank_stability.csv (rank stability metrics)
    - data/processed/metadata/esi_cd_metadata.json

Per Project_Context.md Section 10.2.4:
    - O-ESI variants (objective domains only)
    - RDI_311 (pass-through)
    - Optional SDBI (objective + RDI, explicitly labeled)
    - Rank stability artifacts
    - Equal weights (primary), with sensitivity variants available later

Non-negotiable:
    - Stable sort before aggregation (R15)
    - Atomic writes (R11)
    - Schema validation (R10)
    - Directionality: higher index = worse
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from sleep_esi.paths import PROCESSED_DIR, CONFIG_DIR, LOGS_DIR
from sleep_esi.logging_utils import get_logger
from sleep_esi.io_utils import read_yaml, read_df, atomic_write_df, atomic_write_json
from sleep_esi.hashing import hash_file, get_git_info
from sleep_esi.schemas import ensure_boro_cd_dtype

# Constants
SCRIPT_NAME = "08_build_index"

# =============================================================================
# Domain configuration: maps domain key to file and z-score column
# CRITICAL: Explicit z-score column names for traceability
# =============================================================================
DOMAIN_CONFIG = {
    'noise_obj': {
        'file': 'noise_obj_full_cd.parquet',
        'z_col': 'z_noise_obj_db_mean_robust',  # <-- EXPLICIT: robust z of energy-mean dB
        'pct_col': 'pct_noise_obj_db_mean',
        'raw_col': 'noise_obj_db_mean',
        'name': 'Objective Noise (Full)',
        'units': 'dB (LAeq)',
        'required': True,
    },
    'light': {
        'file': 'light_viirs_cd.parquet',
        'z_col': 'z_light_robust',  # <-- EXPLICIT: robust z of raw radiance
        'pct_col': 'pct_light',
        'raw_col': 'radiance_raw',
        'name': 'Light at Night',
        'units': 'nW/cm²/sr',
        'required': True,
    },
    'heat': {
        'file': 'heat_cd.parquet',
        'z_col': 'z_heat_tmin_robust',  # <-- EXPLICIT: robust z of mean Tmin
        'pct_col': 'pct_heat_tmin',
        'raw_col': 'tmin_mean_primary',
        'name': 'Nighttime Heat',
        'units': '°C',
        'required': True,
    },
    'air': {
        'file': 'air_cd.parquet',
        'z_col': 'z_air_combined_robust',  # <-- EXPLICIT: robust z of combined PM2.5+NO2
        'pct_col': 'pct_air_pm25',
        'raw_col': 'pm25_mean_primary',
        'name': 'Air Quality',
        'units': 'μg/m³',
        'required': True,
    },
}

# RDI config (separate from O-ESI)
RDI_CONFIG = {
    'noise_311': {
        'file': 'noise_311_cd.parquet',
        'z_col': 'z_noise311_robust',  # <-- EXPLICIT: robust z of log complaint rate
        'pct_col': 'pct_noise311',
        'raw_col': 'noise311_rate_per_1k_pop',
        'name': '311 Noise Complaints (RDI)',
        'units': 'complaints/1k pop/year',
    },
}

# =============================================================================
# Canonical index names (per Project_Context.md Section 10.2.4)
# =============================================================================
# Internal working names → Canonical contract names
CANONICAL_NAMES = {
    'oesi_4_equal': 'OESI_4_full_noise',      # Objective ESI with full noise (BTS NTNM)
    'rdi_311': 'RDI_311',                      # Reported Disturbance Index
    'sdbi_5_equal': 'SDBI_5_obj_plus_rdi',    # Combined objective + RDI (explicitly labeled)
}


def load_domain(
    domain_key: str,
    config: dict,
    domains_dir: Path,
    logger,
) -> Optional[pd.DataFrame]:
    """
    Load a domain file and extract relevant columns.
    
    Returns:
        DataFrame with boro_cd, z-score, percentile, and raw value columns
        or None if file not found and not required
    """
    file_path = domains_dir / config['file']
    
    if not file_path.exists():
        if config.get('required', False):
            logger.error(f"Required domain file not found: {file_path}")
            raise FileNotFoundError(f"Required domain file not found: {file_path}")
        else:
            logger.warning(f"Optional domain file not found: {file_path}")
            return None
    
    df = read_df(file_path)
    
    # Validate expected columns exist
    expected_cols = ['boro_cd', config['z_col']]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Domain {domain_key} missing columns: {missing}")
    
    # Extract and rename columns
    result = df[['boro_cd']].copy()
    result[f'z_{domain_key}'] = df[config['z_col']]
    
    if config['pct_col'] in df.columns:
        result[f'pct_{domain_key}'] = df[config['pct_col']]
    
    if config['raw_col'] in df.columns:
        result[f'raw_{domain_key}'] = df[config['raw_col']]
    
    logger.info(f"Loaded domain: {domain_key}", extra={
        "rows": len(result),
        "z_min": float(result[f'z_{domain_key}'].min()),
        "z_max": float(result[f'z_{domain_key}'].max()),
        "z_mean": float(result[f'z_{domain_key}'].mean()),
    })
    
    return result


def compute_composite_index(
    df: pd.DataFrame,
    domain_keys: List[str],
    weights: Dict[str, float],
    index_name: str,
    logger,
) -> pd.DataFrame:
    """
    Compute a weighted composite index from domain z-scores.
    
    Args:
        df: DataFrame with z-score columns (z_{domain_key})
        domain_keys: List of domain keys to include
        weights: Dict mapping domain key to weight
        index_name: Name for the composite index column
        logger: JSONL logger
    
    Returns:
        DataFrame with boro_cd and composite index column
    """
    # Validate weights sum to ~1.0
    weight_sum = sum(weights[k] for k in domain_keys)
    if not np.isclose(weight_sum, 1.0, atol=0.01):
        raise ValueError(f"Weights for {index_name} sum to {weight_sum}, expected 1.0")
    
    # Compute weighted sum
    z_cols = [f'z_{k}' for k in domain_keys]
    
    # Check for missing columns
    missing = [c for c in z_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing z-score columns for {index_name}: {missing}")
    
    # Weighted sum
    weighted_sum = pd.Series(0.0, index=df.index)
    for k in domain_keys:
        weighted_sum += df[f'z_{k}'] * weights[k]
    
    result = df[['boro_cd']].copy()
    result[index_name] = weighted_sum
    
    logger.info(f"Computed composite: {index_name}", extra={
        "domains": domain_keys,
        "weights": weights,
        "min": float(weighted_sum.min()),
        "max": float(weighted_sum.max()),
        "mean": float(weighted_sum.mean()),
    })
    
    return result


def compute_ranks_and_percentiles(
    df: pd.DataFrame,
    index_col: str,
) -> pd.DataFrame:
    """
    Add rank and percentile columns for an index.
    
    Args:
        df: DataFrame with index column
        index_col: Name of index column
    
    Returns:
        DataFrame with rank and percentile columns added
    """
    df = df.copy()
    
    # Rank (1 = highest/worst, per directionality: higher = worse)
    df[f'{index_col}_rank'] = df[index_col].rank(ascending=False, method='min').astype('Int64')
    
    # Percentile (0-100, higher = worse)
    df[f'{index_col}_pct'] = df[index_col].rank(pct=True) * 100
    
    return df


def compute_rank_movement(
    df: pd.DataFrame,
    index_cols: List[str],
) -> pd.DataFrame:
    """
    Compute rank movement between index variants.
    
    Returns DataFrame with boro_cd and rank for each index.
    """
    result = df[['boro_cd']].copy()
    
    for col in index_cols:
        rank_col = f'{col}_rank'
        if rank_col in df.columns:
            result[rank_col] = df[rank_col]
    
    # Sort by boro_cd for stable output (R15)
    result = result.sort_values('boro_cd').reset_index(drop=True)
    
    return result


def compute_rank_stability(
    df: pd.DataFrame,
    index_cols: List[str],
) -> pd.DataFrame:
    """
    Compute rank stability metrics across index variants.
    
    For each CD, computes:
    - Mean rank across variants
    - Rank range (max - min)
    - Rank std dev
    """
    result = df[['boro_cd']].copy()
    
    # Collect rank columns
    rank_data = []
    for col in index_cols:
        rank_col = f'{col}_rank'
        if rank_col in df.columns:
            rank_data.append(df[rank_col])
    
    if not rank_data:
        return result
    
    rank_df = pd.concat(rank_data, axis=1)
    
    result['rank_mean'] = rank_df.mean(axis=1)
    result['rank_min'] = rank_df.min(axis=1)
    result['rank_max'] = rank_df.max(axis=1)
    result['rank_range'] = result['rank_max'] - result['rank_min']
    result['rank_std'] = rank_df.std(axis=1)
    
    # Sort by boro_cd for stable output (R15)
    result = result.sort_values('boro_cd').reset_index(drop=True)
    
    return result


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("Script starting", extra={"script": SCRIPT_NAME, "run_id": run_id})
    
    # Load configs
    params = read_yaml(CONFIG_DIR / "params.yml")
    weights_config = read_yaml(CONFIG_DIR / "weights.yml")
    logger.log_config(params)
    logger.log_config(weights_config)
    
    domains_dir = PROCESSED_DIR / "domains"
    
    # =========================================================================
    # Load all domain data
    # =========================================================================
    
    input_hashes = {}
    domain_dfs = {}
    
    # Load objective domains for O-ESI
    for domain_key, config in DOMAIN_CONFIG.items():
        df = load_domain(domain_key, config, domains_dir, logger)
        if df is not None:
            domain_dfs[domain_key] = df
            input_hashes[domain_key] = hash_file(domains_dir / config['file'])
    
    # Load RDI domain
    for domain_key, config in RDI_CONFIG.items():
        df = load_domain(domain_key, config, domains_dir, logger)
        if df is not None:
            domain_dfs[domain_key] = df
            input_hashes[domain_key] = hash_file(domains_dir / config['file'])
    
    logger.log_inputs({
        "domains_dir": str(domains_dir),
        **input_hashes
    })
    
    # ==========================================================================
    # EXPLICIT: Log which z-score columns are being used for each domain
    # This prevents silent bugs from using wrong columns
    # ==========================================================================
    z_col_mapping = {}
    for domain_key, config in {**DOMAIN_CONFIG, **RDI_CONFIG}.items():
        z_col_mapping[domain_key] = config['z_col']
    
    logger.info("Z-score columns used (EXPLICIT)", extra={
        "z_columns": z_col_mapping,
        "note": "These are the exact columns used for index computation"
    })
    
    # Print to console for visibility
    print("\n  Z-score columns used for index computation:")
    for domain_key, z_col in z_col_mapping.items():
        print(f"    {domain_key}: {z_col}")
    
    # =========================================================================
    # Merge all domains into single DataFrame
    # =========================================================================
    
    # Start with boro_cd from first domain, merge others
    base_domain = list(domain_dfs.keys())[0]
    merged_df = domain_dfs[base_domain].copy()
    
    for domain_key, df in domain_dfs.items():
        if domain_key == base_domain:
            continue
        # Merge on boro_cd (one-to-one expected)
        merge_cols = [c for c in df.columns if c != 'boro_cd']
        merged_df = merged_df.merge(
            df[['boro_cd'] + merge_cols],
            on='boro_cd',
            how='outer',
            validate='one_to_one',
        )
    
    # Sort by boro_cd for determinism (R15)
    merged_df = merged_df.sort_values('boro_cd').reset_index(drop=True)
    
    logger.info("Domains merged", extra={
        "rows": len(merged_df),
        "domains": list(domain_dfs.keys()),
        "columns": list(merged_df.columns),
    })
    
    # =========================================================================
    # Build O-ESI with equal weights (primary)
    # =========================================================================
    
    oesi_domains = ['noise_obj', 'light', 'heat', 'air']
    oesi_weights = weights_config['oesi_weights']['equal']
    
    # Remove the 'label' key from weights dict
    oesi_weight_values = {k: v for k, v in oesi_weights.items() if k != 'label'}
    
    oesi_df = compute_composite_index(
        merged_df,
        oesi_domains,
        oesi_weight_values,
        'oesi_4_equal',
        logger,
    )
    
    merged_df = merged_df.merge(oesi_df, on='boro_cd', how='left', validate='one_to_one')
    merged_df = compute_ranks_and_percentiles(merged_df, 'oesi_4_equal')
    
    # =========================================================================
    # RDI pass-through (already computed, just extract)
    # =========================================================================
    
    merged_df['rdi_311'] = merged_df['z_noise_311']
    merged_df = compute_ranks_and_percentiles(merged_df, 'rdi_311')
    
    logger.info("RDI pass-through", extra={
        "min": float(merged_df['rdi_311'].min()),
        "max": float(merged_df['rdi_311'].max()),
        "mean": float(merged_df['rdi_311'].mean()),
    })
    
    # =========================================================================
    # SDBI: Optional combined index (objective + RDI)
    # =========================================================================
    
    # SDBI: Sleep Disturbance Burden Index
    # Equal weight across 5 domains (4 objective + RDI)
    sdbi_domains = ['noise_obj', 'light', 'heat', 'air', 'noise_311']
    sdbi_weights = {
        'noise_obj': 0.20,
        'light': 0.20,
        'heat': 0.20,
        'air': 0.20,
        'noise_311': 0.20,
    }
    
    sdbi_df = compute_composite_index(
        merged_df,
        sdbi_domains,
        sdbi_weights,
        'sdbi_5_equal',
        logger,
    )
    
    merged_df = merged_df.merge(sdbi_df, on='boro_cd', how='left', validate='one_to_one')
    merged_df = compute_ranks_and_percentiles(merged_df, 'sdbi_5_equal')
    
    # =========================================================================
    # Add canonical column names (aliases for contract compliance)
    # =========================================================================
    
    for internal_name, canonical_name in CANONICAL_NAMES.items():
        if internal_name in merged_df.columns:
            merged_df[canonical_name] = merged_df[internal_name]
            merged_df[f'{canonical_name}_rank'] = merged_df[f'{internal_name}_rank']
            merged_df[f'{canonical_name}_pct'] = merged_df[f'{internal_name}_pct']
    
    logger.info("Canonical column names added", extra={
        "mapping": CANONICAL_NAMES
    })
    
    # =========================================================================
    # Add metadata columns
    # =========================================================================
    
    year_start = params['time_windows']['primary']['year_start']
    year_end = params['time_windows']['primary']['year_end']
    merged_df['year_start'] = year_start
    merged_df['year_end'] = year_end
    
    # Ensure boro_cd is Int64
    merged_df = ensure_boro_cd_dtype(merged_df)
    
    # =========================================================================
    # QA Checks
    # =========================================================================
    
    # Row count
    expected_rows = 59
    if len(merged_df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(merged_df)}")
    
    # Check for NAs in index columns (both internal and canonical names)
    index_cols_internal = ['oesi_4_equal', 'rdi_311', 'sdbi_5_equal']
    index_cols_canonical = list(CANONICAL_NAMES.values())
    
    for col in index_cols_internal + index_cols_canonical:
        na_count = merged_df[col].isna().sum()
        if na_count > 0:
            raise ValueError(f"Index column {col} has {na_count} NA values")
    
    # Directionality check: higher z-score should = higher index rank
    for idx_col in index_cols_internal:
        raw_col = f'{idx_col}'
        rank_col = f'{idx_col}_rank'
        # Rank 1 should have highest value (ascending=False in rank)
        top_ranked = merged_df.loc[merged_df[rank_col] == 1, raw_col].values[0]
        bottom_ranked = merged_df.loc[merged_df[rank_col] == 59, raw_col].values[0]
        if top_ranked <= bottom_ranked:
            logger.warning(f"Directionality may be inverted for {idx_col}")
    
    logger.log_metrics({
        "rows": len(merged_df),
        "na_count": int(sum(merged_df[c].isna().sum() for c in index_cols_internal)),
        "oesi_4_equal_min": float(merged_df['oesi_4_equal'].min()),
        "oesi_4_equal_max": float(merged_df['oesi_4_equal'].max()),
        "rdi_311_min": float(merged_df['rdi_311'].min()),
        "rdi_311_max": float(merged_df['rdi_311'].max()),
        "sdbi_5_equal_min": float(merged_df['sdbi_5_equal'].min()),
        "sdbi_5_equal_max": float(merged_df['sdbi_5_equal'].max()),
    })
    
    # =========================================================================
    # Compute rank artifacts
    # =========================================================================
    
    rank_movement_df = compute_rank_movement(merged_df, index_cols_internal)
    rank_stability_df = compute_rank_stability(merged_df, index_cols_internal)
    
    # =========================================================================
    # Write outputs
    # =========================================================================
    
    # Create output directory
    index_dir = PROCESSED_DIR / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Versioned main index file (canonical artifact)
    versioned_output_path = index_dir / f"esi_cd_{year_start}_{year_end}.parquet"
    atomic_write_df(merged_df, versioned_output_path, index=False)
    
    # "Latest" pointer (convenience, always points to most recent)
    latest_output_path = index_dir / "esi_cd.parquet"
    atomic_write_df(merged_df, latest_output_path, index=False)
    
    logger.log_outputs({
        "esi_cd_versioned": str(versioned_output_path),
        "esi_cd_latest": str(latest_output_path),
    })
    
    # Rank movement CSV (versioned)
    rank_movement_path = index_dir / f"esi_rank_movement_{year_start}_{year_end}.csv"
    atomic_write_df(rank_movement_df, rank_movement_path, index=False)
    
    # Rank stability CSV (versioned)
    rank_stability_path = index_dir / f"esi_rank_stability_{year_start}_{year_end}.csv"
    atomic_write_df(rank_stability_df, rank_stability_path, index=False)
    
    # =========================================================================
    # Write metadata
    # =========================================================================
    
    metadata = {
        'script': SCRIPT_NAME,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'inputs': {
            'domains_dir': str(domains_dir),
            **input_hashes
        },
        'outputs': {
            'esi_cd_versioned': str(versioned_output_path),
            'esi_cd_latest': str(latest_output_path),
            'rank_movement': str(rank_movement_path),
            'rank_stability': str(rank_stability_path),
        },
        'time_window': {
            'year_start': year_start,
            'year_end': year_end,
        },
        # Canonical index names (per Project_Context.md contract)
        'canonical_names': CANONICAL_NAMES,
        'indices': {
            # Canonical names first
            'OESI_4_full_noise': {
                'internal_name': 'oesi_4_equal',
                'name': 'O-ESI (4 domains, full objective noise, equal weights)',
                'domains': oesi_domains,
                'weights': oesi_weight_values,
                'description': 'Objective Environmental Sleep Index using BTS NTNM full noise',
            },
            'RDI_311': {
                'internal_name': 'rdi_311',
                'name': 'RDI (311 Noise Complaints)',
                'domains': ['noise_311'],
                'weights': {'noise_311': 1.0},
                'description': 'Reported Disturbance Index from nighttime 311 complaints',
            },
            'SDBI_5_obj_plus_rdi': {
                'internal_name': 'sdbi_5_equal',
                'name': 'SDBI (5 domains, objective + RDI, equal weights)',
                'domains': sdbi_domains,
                'weights': sdbi_weights,
                'description': 'Sleep Disturbance Burden Index combining objective + reported (NEVER describe as objective-only)',
            },
        },
        # EXPLICIT: Which z-score columns were used (for traceability)
        'z_score_columns_used': {
            domain_key: config['z_col']
            for domain_key, config in {**DOMAIN_CONFIG, **RDI_CONFIG}.items()
        },
        'domain_sources': {
            domain_key: {
                'file': config['file'],
                'z_col': config['z_col'],
                'name': config['name'],
                'units': config['units'],
            }
            for domain_key, config in {**DOMAIN_CONFIG, **RDI_CONFIG}.items()
        },
        'stats': {
            'n_districts': len(merged_df),
            'index_columns_internal': index_cols_internal,
            'index_columns_canonical': index_cols_canonical,
        },
        'directionality': 'higher_index_worse',
        'weighting_scheme': 'equal',
        'git': get_git_info(),
    }
    
    metadata_path = PROCESSED_DIR / "metadata" / "esi_cd_metadata.json"
    atomic_write_json(metadata, metadata_path)
    
    logger.info("Script complete", extra={
        "output_versioned": str(versioned_output_path),
        "output_latest": str(latest_output_path),
        "rows": len(merged_df),
        "indices_internal": index_cols_internal,
        "indices_canonical": index_cols_canonical,
    })
    
    # =========================================================================
    # Print summary
    # =========================================================================
    
    print(f"\n✓ Environmental Sleep Indices computed for {len(merged_df)} community districts")
    print(f"  Versioned output: {versioned_output_path}")
    print(f"  Latest pointer:   {latest_output_path}")
    
    print(f"\n  Canonical index names (per contract):")
    for internal, canonical in CANONICAL_NAMES.items():
        print(f"    {internal} → {canonical}")
    
    print(f"\n  Indices computed:")
    print(f"    - OESI_4_full_noise: {merged_df['OESI_4_full_noise'].min():.2f} to {merged_df['OESI_4_full_noise'].max():.2f}")
    print(f"    - RDI_311: {merged_df['RDI_311'].min():.2f} to {merged_df['RDI_311'].max():.2f}")
    print(f"    - SDBI_5_obj_plus_rdi: {merged_df['SDBI_5_obj_plus_rdi'].min():.2f} to {merged_df['SDBI_5_obj_plus_rdi'].max():.2f}")
    
    # Top 5 worst CDs by O-ESI (use canonical name)
    print(f"\n  Top 5 worst CDs (OESI_4_full_noise, objective exposures only):")
    top5_oesi = merged_df.nlargest(5, 'OESI_4_full_noise')[
        ['boro_cd', 'OESI_4_full_noise', 'OESI_4_full_noise_rank']
    ]
    for _, row in top5_oesi.iterrows():
        print(f"    #{int(row['OESI_4_full_noise_rank'])}: CD {int(row['boro_cd'])} (z={row['OESI_4_full_noise']:.2f})")
    
    # Top 5 worst CDs by RDI (use canonical name)
    print(f"\n  Top 5 worst CDs (RDI_311, 311 complaints):")
    top5_rdi = merged_df.nlargest(5, 'RDI_311')[
        ['boro_cd', 'RDI_311', 'RDI_311_rank']
    ]
    for _, row in top5_rdi.iterrows():
        print(f"    #{int(row['RDI_311_rank'])}: CD {int(row['boro_cd'])} (z={row['RDI_311']:.2f})")
    
    # Top 5 worst CDs by SDBI (use canonical name)
    print(f"\n  Top 5 worst CDs (SDBI_5_obj_plus_rdi, objective + reported):")
    top5_sdbi = merged_df.nlargest(5, 'SDBI_5_obj_plus_rdi')[
        ['boro_cd', 'SDBI_5_obj_plus_rdi', 'SDBI_5_obj_plus_rdi_rank']
    ]
    for _, row in top5_sdbi.iterrows():
        print(f"    #{int(row['SDBI_5_obj_plus_rdi_rank'])}: CD {int(row['boro_cd'])} (z={row['SDBI_5_obj_plus_rdi']:.2f})")
    
    # Rank stability summary
    print(f"\n  Rank stability across indices:")
    most_stable = rank_stability_df.nsmallest(3, 'rank_range')[['boro_cd', 'rank_mean', 'rank_range']]
    print(f"    Most stable (smallest rank range):")
    for _, row in most_stable.iterrows():
        print(f"      CD {int(row['boro_cd'])}: mean rank {row['rank_mean']:.1f}, range {int(row['rank_range'])}")
    
    most_volatile = rank_stability_df.nlargest(3, 'rank_range')[['boro_cd', 'rank_mean', 'rank_range']]
    print(f"    Most volatile (largest rank range):")
    for _, row in most_volatile.iterrows():
        print(f"      CD {int(row['boro_cd'])}: mean rank {row['rank_mean']:.1f}, range {int(row['rank_range'])}")


if __name__ == "__main__":
    main()
