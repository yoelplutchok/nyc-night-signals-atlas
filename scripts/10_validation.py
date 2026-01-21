#!/usr/bin/env python3
"""
10_validation.py

Ecological validation of ESI against sleep outcomes.

Per Project_Context.md Section 10.2.6 and user specification:
    - Primary: NYC CHS sleep outcomes at UHF42 level
    - Fallback: CDC PLACES tract-level data (with SAE circularity caveats)

Validation approach:
    1. Correlations: Pearson + Spearman with 95% CIs
    2. Parsimonious regressions: Sleep ~ ESI, Sleep ~ ESI + demographics
    3. Domain-level associations: Sleep ~ each domain z-score
    4. Negative control: Sleep ~ unrelated variable (to check for spurious associations)

Data source (fallback):
    - CDC PLACES 2023: "Short sleep duration among adults" (<7 hours)
    - Tract-level, aggregated to CD via crosswalk
    - SAE circularity caveat: PLACES uses similar predictors (demographics, health)
      which may inflate associations. This is EXPLORATORY, not causal.

Outputs:
    - data/processed/validation/validation_models.csv
    - data/processed/validation/esi_with_sleep.parquet
    - data/processed/validation/validation_summary.json
    - data/processed/metadata/validation_metadata.json
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
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sleep_esi.paths import RAW_DIR, PROCESSED_DIR, CONFIG_DIR, LOGS_DIR
from sleep_esi.logging_utils import get_logger
from sleep_esi.io_utils import read_yaml, read_df, atomic_write_df, atomic_write_json
from sleep_esi.hashing import hash_file, get_git_info
from sleep_esi.schemas import ensure_boro_cd_dtype

# Constants
SCRIPT_NAME = "10_validation"

# CDC PLACES API for NYC counties
CDC_PLACES_API = "https://data.cdc.gov/resource/cwsq-ngmh.json"
NYC_COUNTIES = ['New York', 'Kings', 'Queens', 'Bronx', 'Richmond']

# Measures to fetch
MEASURES = {
    'SLEEP': 'Short sleep duration among adults (<7 hours)',
    'MHLTH': 'Frequent mental distress among adults',  # potential confounder
    'GHLTH': 'Fair or poor self-rated health status',  # potential confounder
}

# Negative control: use a variable unlikely to be related to ESI
NEGATIVE_CONTROL = 'MAMMOUSE'  # Mammography use - should not correlate with ESI


def fetch_cdc_places_data(measureid: str, logger) -> pd.DataFrame:
    """
    Fetch CDC PLACES tract-level data for NYC.
    
    Args:
        measureid: CDC PLACES measure ID (e.g., 'SLEEP')
        logger: JSONL logger
    
    Returns:
        DataFrame with tract-level data
    """
    # Build query for all NYC counties
    county_filter = " OR ".join([f"countyname='{c}'" for c in NYC_COUNTIES])
    
    url = (
        f"{CDC_PLACES_API}"
        f"?$where=stateabbr='NY' AND ({county_filter})"
        f"&measureid={measureid}"
        f"&$select=locationname,countyname,data_value,data_value_type,year"
        f"&$limit=5000"
    )
    
    logger.info(f"Fetching CDC PLACES: {measureid}", extra={"url": url[:100]})
    
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    
    data = response.json()
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        logger.warning(f"No data returned for {measureid}")
        return pd.DataFrame()
    
    # Rename columns
    df = df.rename(columns={
        'locationname': 'tract_geoid',
        'data_value': measureid.lower(),
    })
    
    # Convert to numeric
    df[measureid.lower()] = pd.to_numeric(df[measureid.lower()], errors='coerce')
    
    logger.info(f"CDC PLACES fetched: {measureid}", extra={
        "rows": len(df),
        "mean": float(df[measureid.lower()].mean()),
        "min": float(df[measureid.lower()].min()),
        "max": float(df[measureid.lower()].max()),
    })
    
    return df[['tract_geoid', measureid.lower()]]


def aggregate_tract_to_cd(
    tract_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    value_col: str,
    logger,
) -> pd.DataFrame:
    """
    Aggregate tract-level data to CD using population weights.
    
    SAE Circularity Caveat: CDC PLACES uses similar demographic predictors,
    which may inflate correlations with ESI. Treat as exploratory.
    """
    # Merge tracts with crosswalk
    merged = crosswalk_df.merge(
        tract_df,
        on='tract_geoid',
        how='left',
    )
    
    # Aggregate by CD using population weights
    # FIX (2025-01): Use tract_pop directly as weight, not tract_pop * w_pop
    # tract_pop from crosswalk is already the intersection population
    results = []
    for boro_cd, group in merged.groupby('boro_cd'):
        valid = group.dropna(subset=['tract_pop', value_col])

        if len(valid) == 0:
            continue

        # Population-weighted average using intersection population
        weights = valid['tract_pop'].astype(float)
        weight_sum = weights.sum()
        
        if pd.isna(weight_sum) or weight_sum <= 0:
            weighted_mean = np.nan
        else:
            weighted_mean = (valid[value_col] * weights).sum() / weight_sum
        
        results.append({
            'boro_cd': boro_cd,
            value_col: weighted_mean,
            f'{value_col}_n_tracts': len(valid),
        })
    
    result_df = pd.DataFrame(results)
    result_df = ensure_boro_cd_dtype(result_df)
    
    logger.info(f"Aggregated {value_col} to CD", extra={
        "cds": len(result_df),
        "mean": float(result_df[value_col].mean()),
    })
    
    return result_df


def compute_correlation_with_ci(
    x: pd.Series,
    y: pd.Series,
    method: str = 'pearson',
    alpha: float = 0.05,
) -> Dict:
    """
    Compute correlation with confidence interval.
    
    Args:
        x, y: Data series
        method: 'pearson' or 'spearman'
        alpha: Significance level for CI
    
    Returns:
        Dict with r, ci_lower, ci_upper, p_value
    """
    # Drop NAs
    valid = pd.DataFrame({'x': x, 'y': y}).dropna()
    n = len(valid)
    
    if n < 3:
        return {'r': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan, 'p_value': np.nan, 'n': n}
    
    if method == 'pearson':
        r, p = stats.pearsonr(valid['x'], valid['y'])
    else:  # spearman
        r, p = stats.spearmanr(valid['x'], valid['y'])
    
    # Fisher z-transformation for CI
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    ci_lower = np.tanh(z - z_crit * se)
    ci_upper = np.tanh(z + z_crit * se)
    
    return {
        'r': float(r),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(p),
        'n': int(n),
    }


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
        
        results = {
            'model': model_name,
            'formula': formula,
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'r_squared_adj': float(model.rsquared_adj),
            'f_statistic': float(model.fvalue) if model.fvalue else None,
            'f_pvalue': float(model.f_pvalue) if model.f_pvalue else None,
        }
        
        # Add key coefficients
        for var in model.params.index:
            clean_var = var.replace('[', '_').replace(']', '').replace('.', '_')
            results[f'coef_{clean_var}'] = float(model.params[var])
            results[f'se_{clean_var}'] = float(model.bse[var])
            results[f'pval_{clean_var}'] = float(model.pvalues[var])
        
        logger.info(f"Regression: {model_name}", extra={
            "r_squared": float(model.rsquared),
            "n_obs": int(model.nobs),
        })
        
        return results
        
    except Exception as e:
        logger.error(f"Regression failed: {model_name}", extra={"error": str(e)})
        return {'model': model_name, 'error': str(e)}


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = get_logger(SCRIPT_NAME, run_id)
    
    logger.info("Script starting", extra={"script": SCRIPT_NAME, "run_id": run_id})
    
    # =========================================================================
    # Load existing data
    # =========================================================================
    
    # ESI with demographics (from equity analysis)
    esi_path = PROCESSED_DIR / "equity" / "esi_with_acs.parquet"
    esi_df = read_df(esi_path)
    logger.info("ESI data loaded", extra={"rows": len(esi_df)})
    
    # Tract-to-CD crosswalk
    xwalk_path = PROCESSED_DIR / "xwalk" / "cd_to_tract_weights.parquet"
    xwalk_df = read_df(xwalk_path)
    logger.info("Crosswalk loaded", extra={"rows": len(xwalk_df)})
    
    input_hashes = {
        'esi_with_acs': hash_file(esi_path),
        'crosswalk': hash_file(xwalk_path),
    }
    
    # =========================================================================
    # Fetch sleep outcome data (CDC PLACES fallback)
    # =========================================================================
    
    logger.info("Fetching CDC PLACES data (fallback - CHS not programmatically accessible)")
    
    # Fetch sleep data
    sleep_df = fetch_cdc_places_data('SLEEP', logger)
    
    # Fetch potential confounders
    mhlth_df = fetch_cdc_places_data('MHLTH', logger)
    ghlth_df = fetch_cdc_places_data('GHLTH', logger)
    
    # Fetch negative control
    negctrl_df = fetch_cdc_places_data(NEGATIVE_CONTROL, logger)
    
    # Save raw CDC data
    raw_dir = RAW_DIR / "cdc_places"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    all_raw = sleep_df.merge(mhlth_df, on='tract_geoid', how='outer')
    all_raw = all_raw.merge(ghlth_df, on='tract_geoid', how='outer')
    all_raw = all_raw.merge(negctrl_df, on='tract_geoid', how='outer')
    all_raw.to_csv(raw_dir / f"places_nyc_tracts_{run_id}.csv", index=False)
    
    logger.info("Raw CDC PLACES saved", extra={"tracts": len(all_raw)})
    
    # =========================================================================
    # Aggregate to CD level
    # =========================================================================
    
    sleep_cd = aggregate_tract_to_cd(sleep_df, xwalk_df, 'sleep', logger)
    mhlth_cd = aggregate_tract_to_cd(mhlth_df, xwalk_df, 'mhlth', logger)
    ghlth_cd = aggregate_tract_to_cd(ghlth_df, xwalk_df, 'ghlth', logger)
    negctrl_cd = aggregate_tract_to_cd(negctrl_df, xwalk_df, NEGATIVE_CONTROL.lower(), logger)
    
    # Merge with ESI
    merged_df = esi_df.copy()
    merged_df = merged_df.merge(sleep_cd[['boro_cd', 'sleep']], on='boro_cd', how='left')
    merged_df = merged_df.merge(mhlth_cd[['boro_cd', 'mhlth']], on='boro_cd', how='left')
    merged_df = merged_df.merge(ghlth_cd[['boro_cd', 'ghlth']], on='boro_cd', how='left')
    merged_df = merged_df.merge(negctrl_cd[['boro_cd', NEGATIVE_CONTROL.lower()]], on='boro_cd', how='left')
    
    logger.info("Data merged", extra={
        "rows": len(merged_df),
        "sleep_na": int(merged_df['sleep'].isna().sum()),
    })
    
    # =========================================================================
    # Validation 1: Correlations (Pearson + Spearman with CIs)
    # =========================================================================
    
    logger.info("Computing correlations")
    
    correlation_results = []
    
    # ESI indices vs sleep
    for idx in ['OESI_4_full_noise', 'RDI_311', 'SDBI_5_obj_plus_rdi']:
        for method in ['pearson', 'spearman']:
            result = compute_correlation_with_ci(
                merged_df[idx], merged_df['sleep'], method=method
            )
            result['variable_x'] = idx
            result['variable_y'] = 'sleep'
            result['method'] = method
            correlation_results.append(result)
    
    # Domain z-scores vs sleep
    for domain in ['z_noise_obj', 'z_light', 'z_heat', 'z_air', 'z_noise_311']:
        for method in ['pearson', 'spearman']:
            result = compute_correlation_with_ci(
                merged_df[domain], merged_df['sleep'], method=method
            )
            result['variable_x'] = domain
            result['variable_y'] = 'sleep'
            result['method'] = method
            correlation_results.append(result)
    
    # Negative control
    for method in ['pearson', 'spearman']:
        result = compute_correlation_with_ci(
            merged_df['OESI_4_full_noise'], merged_df[NEGATIVE_CONTROL.lower()], method=method
        )
        result['variable_x'] = 'OESI_4_full_noise'
        result['variable_y'] = f'{NEGATIVE_CONTROL.lower()} (negative control)'
        result['method'] = method
        correlation_results.append(result)
    
    corr_df = pd.DataFrame(correlation_results)
    
    # =========================================================================
    # Validation 2: Parsimonious regressions
    # =========================================================================
    
    logger.info("Running validation regressions")
    
    regression_results = []
    
    # V1: Sleep ~ OESI (unadjusted)
    regression_results.append(run_regression(
        merged_df, 'sleep ~ OESI_4_full_noise', 'V1_unadjusted', logger
    ))
    
    # V2: Sleep ~ OESI + borough FE
    regression_results.append(run_regression(
        merged_df, 'sleep ~ OESI_4_full_noise + C(borough)', 'V2_borough_FE', logger
    ))
    
    # V3: Sleep ~ OESI + demographics
    regression_results.append(run_regression(
        merged_df, 
        'sleep ~ OESI_4_full_noise + poverty_rate + pct_nonhisp_black + pct_hispanic + C(borough)',
        'V3_demographics', logger
    ))
    
    # V4: Sleep ~ RDI (comparison)
    regression_results.append(run_regression(
        merged_df, 'sleep ~ RDI_311 + C(borough)', 'V4_RDI', logger
    ))
    
    # V5: Sleep ~ SDBI (combined)
    regression_results.append(run_regression(
        merged_df, 'sleep ~ SDBI_5_obj_plus_rdi + C(borough)', 'V5_SDBI', logger
    ))
    
    # V6: Negative control
    regression_results.append(run_regression(
        merged_df, f'{NEGATIVE_CONTROL.lower()} ~ OESI_4_full_noise + C(borough)',
        'V6_negative_control', logger
    ))
    
    reg_df = pd.DataFrame(regression_results)
    
    # =========================================================================
    # Validation 3: Domain-level associations
    # =========================================================================
    
    logger.info("Running domain-level regressions")
    
    domain_results = []
    
    for domain in ['z_noise_obj', 'z_light', 'z_heat', 'z_air', 'z_noise_311']:
        result = run_regression(
            merged_df, f'sleep ~ {domain} + C(borough)',
            f'domain_{domain}', logger
        )
        domain_results.append(result)
    
    domain_df = pd.DataFrame(domain_results)
    
    # =========================================================================
    # Create summary
    # =========================================================================
    
    summary = {
        'data_source': {
            'primary': 'NYC CHS (not accessible programmatically)',
            'fallback': 'CDC PLACES 2023 (used)',
            'measure': 'Short sleep duration among adults (<7 hours)',
            'geography': 'Census tract → CD59 via population-weighted aggregation',
        },
        'sae_circularity_caveat': (
            "CDC PLACES uses Small Area Estimation (SAE) methods that incorporate "
            "demographic predictors similar to those in ESI. This may inflate "
            "observed correlations. Treat these results as EXPLORATORY, not causal. "
            "True validation requires independent sleep outcome data (e.g., survey microdata)."
        ),
        'n_districts': len(merged_df),
        'sleep_outcome': {
            'mean': float(merged_df['sleep'].mean()),
            'std': float(merged_df['sleep'].std()),
            'min': float(merged_df['sleep'].min()),
            'max': float(merged_df['sleep'].max()),
        },
        'key_correlations': {
            'oesi_sleep_pearson': corr_df[
                (corr_df['variable_x'] == 'OESI_4_full_noise') & 
                (corr_df['variable_y'] == 'sleep') &
                (corr_df['method'] == 'pearson')
            ].iloc[0].to_dict() if len(corr_df) > 0 else None,
            'oesi_sleep_spearman': corr_df[
                (corr_df['variable_x'] == 'OESI_4_full_noise') & 
                (corr_df['variable_y'] == 'sleep') &
                (corr_df['method'] == 'spearman')
            ].iloc[0].to_dict() if len(corr_df) > 0 else None,
        },
        'key_regressions': {
            row['model']: {
                'r_squared': row.get('r_squared'),
                'coef_OESI': row.get('coef_OESI_4_full_noise'),
                'pval_OESI': row.get('pval_OESI_4_full_noise'),
            }
            for _, row in reg_df.iterrows() if 'error' not in row
        },
        'negative_control': {
            'variable': NEGATIVE_CONTROL,
            'interpretation': "Should show no significant association with ESI",
        },
    }
    
    # =========================================================================
    # Write outputs
    # =========================================================================
    
    validation_dir = PROCESSED_DIR / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Main validation models
    all_models = pd.concat([reg_df, domain_df], ignore_index=True)
    models_path = validation_dir / "validation_models.csv"
    atomic_write_df(all_models, models_path, index=False)
    
    # Correlations
    corr_path = validation_dir / "validation_correlations.csv"
    atomic_write_df(corr_df, corr_path, index=False)
    
    # Merged data with sleep
    merged_df = ensure_boro_cd_dtype(merged_df)
    merged_path = validation_dir / "esi_with_sleep.parquet"
    atomic_write_df(merged_df, merged_path, index=False)
    
    # Summary JSON
    summary_path = validation_dir / "validation_summary.json"
    atomic_write_json(summary, summary_path)
    
    logger.log_outputs({
        "validation_models": str(models_path),
        "validation_correlations": str(corr_path),
        "esi_with_sleep": str(merged_path),
        "validation_summary": str(summary_path),
    })
    
    # =========================================================================
    # Write metadata
    # =========================================================================
    
    metadata = {
        'script': SCRIPT_NAME,
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'inputs': {
            'esi_with_acs': str(esi_path),
            'crosswalk': str(xwalk_path),
            **input_hashes,
        },
        'outputs': {
            'validation_models': str(models_path),
            'validation_correlations': str(corr_path),
            'esi_with_sleep': str(merged_path),
            'validation_summary': str(summary_path),
        },
        'data_source': {
            'api': CDC_PLACES_API,
            'measures': MEASURES,
            'negative_control': NEGATIVE_CONTROL,
            'note': 'CDC PLACES fallback used (CHS not programmatically accessible)',
        },
        'sae_circularity_caveat': summary['sae_circularity_caveat'],
        'summary': summary,
        'git': get_git_info(),
    }
    
    metadata_path = PROCESSED_DIR / "metadata" / "validation_metadata.json"
    atomic_write_json(metadata, metadata_path)
    
    logger.info("Script complete")
    
    # =========================================================================
    # Print summary
    # =========================================================================
    
    print(f"\n✓ Validation analysis complete for {len(merged_df)} community districts")
    print(f"  Output: {models_path}")
    
    print(f"\n  ⚠️  SAE CIRCULARITY CAVEAT:")
    print(f"  CDC PLACES uses demographic predictors similar to ESI.")
    print(f"  Correlations may be inflated. Treat as EXPLORATORY.")
    
    print(f"\n  Sleep outcome (short sleep <7h):")
    print(f"    Mean: {merged_df['sleep'].mean():.1f}%")
    print(f"    Range: {merged_df['sleep'].min():.1f}% - {merged_df['sleep'].max():.1f}%")
    
    print(f"\n  Key correlations with OESI_4_full_noise:")
    oesi_corr = corr_df[
        (corr_df['variable_x'] == 'OESI_4_full_noise') & 
        (corr_df['variable_y'] == 'sleep')
    ]
    for _, row in oesi_corr.iterrows():
        print(f"    {row['method'].capitalize()}: r = {row['r']:.3f} ({row['ci_lower']:.3f}, {row['ci_upper']:.3f}), p = {row['p_value']:.4f}")
    
    print(f"\n  Domain-level correlations with sleep (Pearson):")
    domain_corrs = corr_df[
        (corr_df['variable_y'] == 'sleep') & 
        (corr_df['method'] == 'pearson') &
        (corr_df['variable_x'].str.startswith('z_'))
    ]
    for _, row in domain_corrs.iterrows():
        sig = "*" if row['p_value'] < 0.05 else ""
        print(f"    {row['variable_x']}: r = {row['r']:.3f}{sig}")
    
    print(f"\n  Validation regressions (Sleep ~ ESI):")
    for _, row in reg_df.iterrows():
        if 'error' not in row or pd.isna(row.get('error')):
            print(f"    {row['model']}: R² = {row['r_squared']:.3f}")
    
    print(f"\n  Negative control ({NEGATIVE_CONTROL} ~ OESI):")
    neg_ctrl_row = reg_df[reg_df['model'] == 'V6_negative_control'].iloc[0]
    if 'error' not in neg_ctrl_row:
        print(f"    R² = {neg_ctrl_row['r_squared']:.3f} (should be low)")


if __name__ == "__main__":
    main()
