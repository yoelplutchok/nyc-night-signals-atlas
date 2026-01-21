#!/usr/bin/env python3
"""
09_equity_analysis.py

Equity analysis and dual-reality (reporting bias) diagnostics.

Per Project_Context.md Section 10.2.5:
    - Join ACS-derived CD measures (counts/denominators only, no medians)
    - Run pre-specified equity regressions (no model shopping)
    - Run dual-reality reporting-bias diagnostics

ACS Variables (tract-level, aggregated to CD):
    - B01003: Total population
    - B17001: Poverty status (below poverty / universe)
    - B02001: Race (Black alone)
    - B03003: Hispanic origin
    - B25070: Rent burden (≥30% income on rent)

Required Equity Regressions (all run, no cherry-picking):
    M1:  OESI ~ poverty_rate + borough_FE
    M2a: OESI ~ poverty_rate + pct_nonhisp_black + borough_FE
    M2b: OESI ~ poverty_rate + pct_hispanic + borough_FE
    M3a: OESI ~ poverty_rate + pct_nonhisp_black + rent_burden_rate + borough_FE
    M3b: OESI ~ poverty_rate + pct_hispanic + rent_burden_rate + borough_FE

Dual Reality Module:
    Model A (required): RDI_311 ~ z_noise_obj + borough_FE
    Model B (sensitivity): RDI_311 ~ z_noise_obj + demographics + borough_FE

Outputs:
    - data/processed/equity/esi_with_acs.parquet
    - data/processed/equity/equity_regressions.csv
    - data/processed/equity/dual_reality_cd.parquet
    - data/processed/equity/equity_summary.csv
    - data/processed/metadata/equity_metadata.json
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sleep_esi.paths import RAW_DIR, PROCESSED_DIR, CONFIG_DIR, LOGS_DIR
from sleep_esi.logging_utils import get_logger
from sleep_esi.io_utils import read_yaml, read_df, atomic_write_df, atomic_write_json
from sleep_esi.hashing import hash_file, get_git_info
from sleep_esi.schemas import ensure_boro_cd_dtype

# Constants
SCRIPT_NAME = "09_equity_analysis"

# =============================================================================
# ACS Variable Configuration
# =============================================================================
# Per contract: Only count-based variables, no medians
# All aggregated from tract to CD using population weights

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

# Flatten for API request
ACS_VARIABLES = []
for table, vars in ACS_TABLES.items():
    ACS_VARIABLES.extend(vars)

# NYC County FIPS codes
NYC_COUNTIES = ['005', '047', '061', '081', '085']
NYC_COUNTY_TO_BORO = {
    '005': 2,  # Bronx
    '047': 3,  # Brooklyn (Kings)
    '061': 1,  # Manhattan (New York)
    '081': 4,  # Queens
    '085': 5,  # Staten Island (Richmond)
}

# =============================================================================
# Required Regression Models (per contract - no selection allowed)
# =============================================================================

EQUITY_MODELS = {
    'M1': 'OESI_4_full_noise ~ poverty_rate + C(borough)',
    'M2a': 'OESI_4_full_noise ~ poverty_rate + pct_nonhisp_black + C(borough)',
    'M2b': 'OESI_4_full_noise ~ poverty_rate + pct_hispanic + C(borough)',
    'M3a': 'OESI_4_full_noise ~ poverty_rate + pct_nonhisp_black + rent_burden_rate + C(borough)',
    'M3b': 'OESI_4_full_noise ~ poverty_rate + pct_hispanic + rent_burden_rate + C(borough)',
}

DUAL_REALITY_MODELS = {
    'A_baseline': 'RDI_311 ~ z_noise_obj + C(borough)',
    'B_adjusted': 'RDI_311 ~ z_noise_obj + poverty_rate + pct_nonhisp_black + pct_hispanic + rent_burden_rate + C(borough)',
}


def fetch_acs_data(logger) -> pd.DataFrame:
    """
    Fetch ACS 5-year 2022 data at tract level for NYC.
    
    Returns DataFrame with tract-level counts.
    """
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
    
    # Create geoid
    df['geoid'] = '36' + df['county'] + df['tract']
    
    # Convert numeric columns
    for col in ACS_VARIABLES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info("ACS data fetched", extra={
        "rows": len(df),
        "columns": list(df.columns),
    })
    
    return df


def compute_tract_demographics(acs_df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Compute demographic rates at tract level.
    
    All rates are counts/denominators (no median aggregation).
    """
    result = acs_df[['geoid']].copy()
    
    # Total population
    result['population'] = acs_df['B01003_001E']
    
    # Poverty rate = below poverty / poverty universe
    result['poverty_count'] = acs_df['B17001_002E']
    result['poverty_universe'] = acs_df['B17001_001E']
    result['poverty_rate'] = np.where(
        result['poverty_universe'] > 0,
        result['poverty_count'] / result['poverty_universe'],
        np.nan
    )
    
    # % Non-Hispanic Black = Black alone / total population
    # Note: B02001_003E is "Black or African American alone"
    # This is an approximation; for true non-Hispanic Black we'd need B03002
    result['black_count'] = acs_df['B02001_003E']
    result['pct_nonhisp_black'] = np.where(
        result['population'] > 0,
        result['black_count'] / result['population'],
        np.nan
    )
    
    # % Hispanic = Hispanic / total population
    result['hispanic_count'] = acs_df['B03003_003E']
    result['pct_hispanic'] = np.where(
        result['population'] > 0,
        result['hispanic_count'] / result['population'],
        np.nan
    )
    
    # Rent burden ≥30% = (30-34.9% + 35-39.9% + 40-49.9% + 50%+) / total renters
    result['renter_total'] = acs_df['B25070_001E']
    result['rent_burden_count'] = (
        acs_df['B25070_007E'] + 
        acs_df['B25070_008E'] + 
        acs_df['B25070_009E'] + 
        acs_df['B25070_010E']
    )
    result['rent_burden_rate'] = np.where(
        result['renter_total'] > 0,
        result['rent_burden_count'] / result['renter_total'],
        np.nan
    )
    
    logger.info("Tract demographics computed", extra={
        "tracts": len(result),
        "poverty_rate_mean": float(result['poverty_rate'].mean()),
        "pct_black_mean": float(result['pct_nonhisp_black'].mean()),
        "pct_hispanic_mean": float(result['pct_hispanic'].mean()),
        "rent_burden_mean": float(result['rent_burden_rate'].mean()),
    })
    
    return result


def aggregate_to_cd(
    tract_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """
    Aggregate tract-level demographics to CD using population weights.

    For rates, we use population-weighted averages where the weight is the
    intersection population (tract_pop from crosswalk), not tract_pop * w_pop.

    For counts, we sum weighted by tract-CD area overlap (w_area).

    Per contract: No invalid median aggregation.

    Note on weighting (FIX applied 2025-01):
        The crosswalk has tract_pop = intersection population, and
        w_pop = tract_pop / cd_total_pop. Using tract_pop directly as the
        weight gives the correct population-weighted average for rates.
    """
    # Merge tracts with crosswalk
    merged = crosswalk_df.merge(
        tract_df,
        left_on='tract_geoid',
        right_on='geoid',
        how='left',
    )

    # For each CD, compute population-weighted demographics
    # Weight = tract_pop (intersection population from crosswalk)
    
    results = []
    for boro_cd, group in merged.groupby('boro_cd'):
        # Weight = tract_pop (intersection population from crosswalk)
        weights_source = group['tract_pop'].fillna(0).astype(float)
        
        geo_data = {'boro_cd': boro_cd}
        
        # Calculate rates for each demographic column
        rate_cols = ['poverty_rate', 'pct_nonhisp_black', 'pct_hispanic', 'rent_burden_rate']
        for col in rate_cols:
            # Filter to rows that have both a valid rate and a non-zero weight
            mask = group[col].notna() & (weights_source > 0)
            
            if mask.any():
                col_weights = weights_source[mask]
                col_values = group.loc[mask, col].astype(float)
                geo_data[col] = (col_values * col_weights).sum() / col_weights.sum()
                
                # Use total population of tracts with data as 'population'
                if 'population' not in geo_data:
                    geo_data['population'] = col_weights.sum()
            else:
                geo_data[col] = np.nan
                if 'population' not in geo_data:
                    geo_data['population'] = 0.0
        
        # Count-based aggregation (for reference) - use w_area for counts
        # This is an approximation
        geo_data['poverty_count'] = (group['poverty_count'].fillna(0) * group['w_area'].fillna(0)).sum()
        geo_data['black_count'] = (group['black_count'].fillna(0) * group['w_area'].fillna(0)).sum()
        geo_data['hispanic_count'] = (group['hispanic_count'].fillna(0) * group['w_area'].fillna(0)).sum()
        
        results.append(geo_data)
    
    result_df = pd.DataFrame(results)
    result_df = ensure_boro_cd_dtype(result_df)
    
    logger.info("Demographics aggregated to CD", extra={
        "cds": len(result_df),
        "poverty_rate_range": [float(result_df['poverty_rate'].min()), float(result_df['poverty_rate'].max())],
        "pct_black_range": [float(result_df['pct_nonhisp_black'].min()), float(result_df['pct_nonhisp_black'].max())],
    })
    
    return result_df


def run_regression(
    df: pd.DataFrame,
    formula: str,
    model_name: str,
    logger,
) -> Dict:
    """
    Run OLS regression and return results.
    """
    try:
        model = smf.ols(formula, data=df).fit()
        
        # Extract coefficients, SEs, p-values
        results = {
            'model': model_name,
            'formula': formula,
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'r_squared_adj': float(model.rsquared_adj),
            'f_statistic': float(model.fvalue) if model.fvalue else None,
            'f_pvalue': float(model.f_pvalue) if model.f_pvalue else None,
        }
        
        # Add coefficients
        for var in model.params.index:
            # Clean variable name for column naming
            clean_var = var.replace('[', '_').replace(']', '').replace('.', '_')
            results[f'coef_{clean_var}'] = float(model.params[var])
            results[f'se_{clean_var}'] = float(model.bse[var])
            results[f'pval_{clean_var}'] = float(model.pvalues[var])
        
        logger.info(f"Regression: {model_name}", extra={
            "r_squared": float(model.rsquared),
            "n_obs": int(model.nobs),
        })
        
        return results, model
        
    except Exception as e:
        logger.error(f"Regression failed: {model_name}", extra={"error": str(e)})
        return {'model': model_name, 'error': str(e)}, None


def run_dual_reality_analysis(
    df: pd.DataFrame,
    logger,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run dual-reality (reporting bias) analysis.
    
    Model A (required): RDI_311 ~ z_noise_obj + borough_FE
    Model B (sensitivity): RDI_311 ~ z_noise_obj + demographics + borough_FE
    
    Residuals indicate over/under-reporting relative to exposure.
    """
    results = {}
    
    # Model A: Baseline (required)
    formula_a = DUAL_REALITY_MODELS['A_baseline']
    result_a, model_a = run_regression(df, formula_a, 'dual_reality_A', logger)
    results['A_baseline'] = result_a
    
    # Model B: Demographically adjusted (sensitivity)
    formula_b = DUAL_REALITY_MODELS['B_adjusted']
    result_b, model_b = run_regression(df, formula_b, 'dual_reality_B', logger)
    results['B_adjusted'] = result_b
    
    # Add residuals to dataframe
    df = df.copy()
    
    if model_a is not None:
        df['resid_A_raw'] = model_a.resid
        
        # Classify: data deserts (high noise, low complaints = negative residual)
        # Conflict zones (low noise, high complaints = positive residual)
        # Use 1 SD threshold for classification
        resid_std = df['resid_A_raw'].std()
        df['dual_reality_class_A'] = pd.cut(
            df['resid_A_raw'],
            bins=[-np.inf, -resid_std, resid_std, np.inf],
            labels=['data_desert', 'neutral', 'conflict_zone']
        )
    
    if model_b is not None:
        df['resid_B_adjusted'] = model_b.resid
        resid_std_b = df['resid_B_adjusted'].std()
        df['dual_reality_class_B'] = pd.cut(
            df['resid_B_adjusted'],
            bins=[-np.inf, -resid_std_b, resid_std_b, np.inf],
            labels=['data_desert', 'neutral', 'conflict_zone']
        )
    
    logger.info("Dual reality analysis complete", extra={
        "model_a_r2": result_a.get('r_squared'),
        "model_b_r2": result_b.get('r_squared'),
        "data_deserts_A": int((df['dual_reality_class_A'] == 'data_desert').sum()) if 'dual_reality_class_A' in df.columns else None,
        "conflict_zones_A": int((df['dual_reality_class_A'] == 'conflict_zone').sum()) if 'dual_reality_class_A' in df.columns else None,
    })
    
    return df, results


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("Script starting", extra={"script": SCRIPT_NAME, "run_id": run_id})
    
    # Load config
    params = read_yaml(CONFIG_DIR / "params.yml")
    logger.log_config(params)
    
    # =========================================================================
    # Load existing data
    # =========================================================================
    
    # ESI index data
    esi_path = PROCESSED_DIR / "index" / "esi_cd_2021_2023.parquet"
    esi_df = read_df(esi_path)
    logger.info("ESI data loaded", extra={"rows": len(esi_df)})
    
    # Extract borough from boro_cd (first digit)
    esi_df['borough'] = (esi_df['boro_cd'] // 100).astype(int)
    
    # Crosswalk for tract-to-CD aggregation
    xwalk_path = PROCESSED_DIR / "xwalk" / "cd_to_tract_weights.parquet"
    xwalk_df = read_df(xwalk_path)
    logger.info("Crosswalk loaded", extra={"rows": len(xwalk_df)})
    
    input_hashes = {
        'esi_cd': hash_file(esi_path),
        'crosswalk': hash_file(xwalk_path),
    }
    
    # =========================================================================
    # Fetch and process ACS demographics
    # =========================================================================
    
    # Fetch ACS data
    acs_df = fetch_acs_data(logger)
    
    # Save raw ACS data
    acs_raw_path = RAW_DIR / "acs_demographics" / f"acs_demographics_{run_id}.csv"
    acs_raw_path.parent.mkdir(parents=True, exist_ok=True)
    acs_df.to_csv(acs_raw_path, index=False)
    logger.info("Raw ACS saved", extra={"path": str(acs_raw_path)})
    
    # Compute tract-level demographics
    tract_demo_df = compute_tract_demographics(acs_df, logger)
    
    # Aggregate to CD level
    cd_demo_df = aggregate_to_cd(tract_demo_df, xwalk_df, logger)
    
    # =========================================================================
    # Merge ESI with demographics
    # =========================================================================
    
    merged_df = esi_df.merge(
        cd_demo_df,
        on='boro_cd',
        how='left',
        validate='one_to_one',
    )
    
    # Check for missing data
    demo_cols = ['poverty_rate', 'pct_nonhisp_black', 'pct_hispanic', 'rent_burden_rate']
    for col in demo_cols:
        na_count = merged_df[col].isna().sum()
        if na_count > 0:
            logger.warning(f"Missing values in {col}: {na_count}")
    
    logger.info("Data merged", extra={
        "rows": len(merged_df),
        "columns": list(merged_df.columns),
    })
    
    # =========================================================================
    # Run pre-specified equity regressions (no cherry-picking)
    # =========================================================================
    
    logger.info("Running pre-specified equity regressions")
    
    equity_results = []
    for model_name, formula in EQUITY_MODELS.items():
        result, _ = run_regression(merged_df, formula, model_name, logger)
        equity_results.append(result)
    
    equity_results_df = pd.DataFrame(equity_results)
    
    # =========================================================================
    # Run dual-reality analysis
    # =========================================================================
    
    logger.info("Running dual-reality analysis")
    
    merged_df, dual_reality_results = run_dual_reality_analysis(merged_df, logger)
    
    # =========================================================================
    # Create summary statistics
    # =========================================================================
    
    summary_stats = {
        'n_districts': len(merged_df),
        'demographics': {
            'poverty_rate': {
                'mean': float(merged_df['poverty_rate'].mean()),
                'std': float(merged_df['poverty_rate'].std()),
                'min': float(merged_df['poverty_rate'].min()),
                'max': float(merged_df['poverty_rate'].max()),
            },
            'pct_nonhisp_black': {
                'mean': float(merged_df['pct_nonhisp_black'].mean()),
                'std': float(merged_df['pct_nonhisp_black'].std()),
                'min': float(merged_df['pct_nonhisp_black'].min()),
                'max': float(merged_df['pct_nonhisp_black'].max()),
            },
            'pct_hispanic': {
                'mean': float(merged_df['pct_hispanic'].mean()),
                'std': float(merged_df['pct_hispanic'].std()),
                'min': float(merged_df['pct_hispanic'].min()),
                'max': float(merged_df['pct_hispanic'].max()),
            },
            'rent_burden_rate': {
                'mean': float(merged_df['rent_burden_rate'].mean()),
                'std': float(merged_df['rent_burden_rate'].std()),
                'min': float(merged_df['rent_burden_rate'].min()),
                'max': float(merged_df['rent_burden_rate'].max()),
            },
        },
        'correlations': {
            'oesi_poverty': float(merged_df['OESI_4_full_noise'].corr(merged_df['poverty_rate'])),
            'oesi_pct_black': float(merged_df['OESI_4_full_noise'].corr(merged_df['pct_nonhisp_black'])),
            'oesi_pct_hispanic': float(merged_df['OESI_4_full_noise'].corr(merged_df['pct_hispanic'])),
            'oesi_rent_burden': float(merged_df['OESI_4_full_noise'].corr(merged_df['rent_burden_rate'])),
            'rdi_obj_noise': float(merged_df['RDI_311'].corr(merged_df['z_noise_obj'])),
        },
        'dual_reality': {
            'data_deserts': int((merged_df['dual_reality_class_A'] == 'data_desert').sum()) if 'dual_reality_class_A' in merged_df.columns else None,
            'neutral': int((merged_df['dual_reality_class_A'] == 'neutral').sum()) if 'dual_reality_class_A' in merged_df.columns else None,
            'conflict_zones': int((merged_df['dual_reality_class_A'] == 'conflict_zone').sum()) if 'dual_reality_class_A' in merged_df.columns else None,
        },
    }
    
    # =========================================================================
    # Write outputs
    # =========================================================================
    
    equity_dir = PROCESSED_DIR / "equity"
    equity_dir.mkdir(parents=True, exist_ok=True)
    
    # Main merged file
    merged_df = ensure_boro_cd_dtype(merged_df)
    output_path = equity_dir / "esi_with_acs.parquet"
    atomic_write_df(merged_df, output_path, index=False)
    
    # Equity regression results
    regression_path = equity_dir / "equity_regressions.csv"
    atomic_write_df(equity_results_df, regression_path, index=False)
    
    # Dual reality subset (key columns only)
    dual_reality_cols = [
        'boro_cd', 'OESI_4_full_noise', 'RDI_311', 'z_noise_obj',
        'poverty_rate', 'pct_nonhisp_black', 'pct_hispanic', 'rent_burden_rate',
        'resid_A_raw', 'dual_reality_class_A',
    ]
    if 'resid_B_adjusted' in merged_df.columns:
        dual_reality_cols.extend(['resid_B_adjusted', 'dual_reality_class_B'])
    
    dual_reality_df = merged_df[[c for c in dual_reality_cols if c in merged_df.columns]].copy()
    dual_reality_path = equity_dir / "dual_reality_cd.parquet"
    atomic_write_df(dual_reality_df, dual_reality_path, index=False)
    
    # Summary CSV
    summary_df = pd.DataFrame([{
        'metric': k,
        'value': str(v) if isinstance(v, dict) else v
    } for k, v in summary_stats.items()])
    summary_path = equity_dir / "equity_summary.csv"
    atomic_write_df(summary_df, summary_path, index=False)
    
    logger.log_outputs({
        "esi_with_acs": str(output_path),
        "equity_regressions": str(regression_path),
        "dual_reality_cd": str(dual_reality_path),
        "equity_summary": str(summary_path),
    })
    
    # =========================================================================
    # Write metadata
    # =========================================================================
    
    metadata = {
        'script': SCRIPT_NAME,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'inputs': {
            'esi_cd': str(esi_path),
            'crosswalk': str(xwalk_path),
            **input_hashes,
        },
        'outputs': {
            'esi_with_acs': str(output_path),
            'equity_regressions': str(regression_path),
            'dual_reality_cd': str(dual_reality_path),
            'equity_summary': str(summary_path),
        },
        'acs_variables': {
            'source': 'ACS 5-year 2022',
            'tables': list(ACS_TABLES.keys()),
            'variables': ACS_VARIABLES,
        },
        'equity_models': EQUITY_MODELS,
        'dual_reality_models': DUAL_REALITY_MODELS,
        'summary_stats': summary_stats,
        'equity_regression_results': equity_results,
        'dual_reality_results': dual_reality_results,
        'notes': {
            'aggregation': 'Tract-to-CD via population weights, no median aggregation',
            'regressions': 'Pre-specified, no model selection',
            'dual_reality': 'Residuals indicate reporting intensity unexplained by exposure',
        },
        'git': get_git_info(),
    }
    
    metadata_path = PROCESSED_DIR / "metadata" / "equity_metadata.json"
    atomic_write_json(metadata, metadata_path)
    
    logger.info("Script complete", extra={
        "outputs": list(metadata['outputs'].keys()),
    })
    
    # =========================================================================
    # Print summary
    # =========================================================================
    
    print(f"\n✓ Equity analysis complete for {len(merged_df)} community districts")
    print(f"  Output: {output_path}")
    
    print(f"\n  Demographic summary:")
    print(f"    Poverty rate: {merged_df['poverty_rate'].mean()*100:.1f}% mean (range: {merged_df['poverty_rate'].min()*100:.1f}% - {merged_df['poverty_rate'].max()*100:.1f}%)")
    print(f"    % Non-Hispanic Black: {merged_df['pct_nonhisp_black'].mean()*100:.1f}% mean")
    print(f"    % Hispanic: {merged_df['pct_hispanic'].mean()*100:.1f}% mean")
    print(f"    Rent burden ≥30%: {merged_df['rent_burden_rate'].mean()*100:.1f}% mean")
    
    print(f"\n  Equity correlations with OESI_4_full_noise:")
    print(f"    Poverty rate: r = {summary_stats['correlations']['oesi_poverty']:.3f}")
    print(f"    % Black: r = {summary_stats['correlations']['oesi_pct_black']:.3f}")
    print(f"    % Hispanic: r = {summary_stats['correlations']['oesi_pct_hispanic']:.3f}")
    print(f"    Rent burden: r = {summary_stats['correlations']['oesi_rent_burden']:.3f}")
    
    print(f"\n  Pre-specified equity regressions (all run, no selection):")
    for _, row in equity_results_df.iterrows():
        if 'error' not in row or pd.isna(row.get('error')):
            print(f"    {row['model']}: R² = {row['r_squared']:.3f}")
        else:
            print(f"    {row['model']}: ERROR - {row['error']}")
    
    print(f"\n  Dual reality analysis (Model A - baseline):")
    if 'dual_reality_class_A' in merged_df.columns:
        print(f"    Data deserts (high noise, low complaints): {(merged_df['dual_reality_class_A'] == 'data_desert').sum()} CDs")
        print(f"    Neutral: {(merged_df['dual_reality_class_A'] == 'neutral').sum()} CDs")
        print(f"    Conflict zones (low noise, high complaints): {(merged_df['dual_reality_class_A'] == 'conflict_zone').sum()} CDs")
    
    # Show example CDs
    if 'resid_A_raw' in merged_df.columns:
        print(f"\n  Top 3 data deserts (under-reporting):")
        deserts = merged_df.nsmallest(3, 'resid_A_raw')[['boro_cd', 'z_noise_obj', 'RDI_311', 'resid_A_raw']]
        for _, row in deserts.iterrows():
            print(f"    CD {int(row['boro_cd'])}: noise_z={row['z_noise_obj']:.2f}, RDI_z={row['RDI_311']:.2f}, resid={row['resid_A_raw']:.2f}")
        
        print(f"\n  Top 3 conflict zones (over-reporting):")
        conflicts = merged_df.nlargest(3, 'resid_A_raw')[['boro_cd', 'z_noise_obj', 'RDI_311', 'resid_A_raw']]
        for _, row in conflicts.iterrows():
            print(f"    CD {int(row['boro_cd'])}: noise_z={row['z_noise_obj']:.2f}, RDI_z={row['RDI_311']:.2f}, resid={row['resid_A_raw']:.2f}")


if __name__ == "__main__":
    main()
