#!/usr/bin/env python3
"""
11_build_nta_typology.py

Build Night Signatures Typology (clustering) for NYC Neighborhood Tabulation Areas (NTAs).

Per NYC_Night_Signals_Plan.md Section 3.3.11:
- Build feature matrix from Script 10 NTA-level features (residential/mixed only)
- Apply eligibility guardrail: only NTAs with count_night >= min_cluster_count are clustered
- NTAs below threshold get cluster_id=-1, cluster_label="Low Volume", is_outlier_cluster=True
- Standardize features (z-score)
- Cluster NTAs using K-Means with K in [8..15]
- Select optimal K using silhouette score with singleton control
- Log all scores for each K

Outputs:
- data/processed/typology/nta_clusters_residential.parquet/csv
- data/processed/typology/nta_clusters_residential.geojson (map-ready with geometry)
- data/processed/typology/nta_cluster_summary_residential.csv
- data/processed/typology/nta_k_selection_scores.csv
- data/processed/metadata/nta_clusters_residential_metadata.json (provenance sidecar)

Key requirements:
- Fixed random seed for reproducibility
- Log all parameters and feature columns used
- Save standardized feature matrix column names
- QA: row counts, no null cluster_id, cluster sizes not degenerate, "Low Volume" rules
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from sleep_esi.hashing import write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_df, atomic_write_gdf, read_yaml, read_gdf
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, PROCESSED_DIR


# =============================================================================
# Constants
# =============================================================================

ATLAS_DIR = PROCESSED_DIR / "atlas"
TYPOLOGY_DIR = PROCESSED_DIR / "typology"

# Input
INPUT_NTA_FEATURES = ATLAS_DIR / "311_nta_features_residential.parquet"
INPUT_NTA_GEO = GEO_DIR / "nta.parquet"
INPUT_NTA_LOOKUP = GEO_DIR / "nta_lookup.parquet"

# Outputs
OUTPUT_CLUSTERS = TYPOLOGY_DIR / "nta_clusters_residential.parquet"
OUTPUT_CLUSTERS_CSV = TYPOLOGY_DIR / "nta_clusters_residential.csv"
OUTPUT_SUMMARY = TYPOLOGY_DIR / "nta_cluster_summary_residential.csv"
OUTPUT_GEOJSON = TYPOLOGY_DIR / "nta_clusters_residential.geojson"
OUTPUT_FEATURE_MATRIX = TYPOLOGY_DIR / "nta_feature_matrix_standardized.csv"
OUTPUT_K_SCORES = TYPOLOGY_DIR / "nta_k_selection_scores.csv"


# =============================================================================
# Data Loading
# =============================================================================

def load_nta_features(logger) -> pd.DataFrame:
    """Load the NTA-level features from Script 10 (residential only)."""
    if not INPUT_NTA_FEATURES.exists():
        raise FileNotFoundError(
            f"NTA features not found: {INPUT_NTA_FEATURES}. "
            "Run 10_build_311_nta_features.py first."
        )
    
    df = pd.read_parquet(INPUT_NTA_FEATURES)
    logger.info(f"Loaded NTA features: {len(df)} NTAs, {len(df.columns)} columns")
    
    return df


def load_nta_geo(logger) -> gpd.GeoDataFrame:
    """Load the canonical NTA geometries."""
    if not INPUT_NTA_GEO.exists():
        raise FileNotFoundError(
            f"NTA geometries not found: {INPUT_NTA_GEO}. "
            "Run 09_build_nta.py first."
        )
    
    gdf = read_gdf(INPUT_NTA_GEO)
    logger.info(f"Loaded NTA geometries: {len(gdf)} NTAs")
    
    return gdf


def load_nta_lookup(logger) -> pd.DataFrame:
    """Load the NTA lookup table."""
    if not INPUT_NTA_LOOKUP.exists():
        raise FileNotFoundError(
            f"NTA lookup not found: {INPUT_NTA_LOOKUP}. "
            "Run 09_build_nta.py first."
        )
    
    df = pd.read_parquet(INPUT_NTA_LOOKUP)
    logger.info(f"Loaded NTA lookup: {len(df)} NTAs")
    
    return df


# =============================================================================
# Eligibility Guardrail
# =============================================================================

def apply_eligibility_guardrail(
    df: pd.DataFrame,
    min_cluster_count: int,
    logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split NTAs into eligible (for clustering) and ineligible (low volume).
    
    Returns:
        - df_eligible: NTAs with count_night >= min_cluster_count
        - df_ineligible: NTAs with count_night < min_cluster_count
    """
    logger.info(f"Applying eligibility guardrail: min_cluster_count = {min_cluster_count}")
    
    eligible_mask = df["count_night"] >= min_cluster_count
    
    df_eligible = df[eligible_mask].copy()
    df_ineligible = df[~eligible_mask].copy()
    
    n_eligible = len(df_eligible)
    n_ineligible = len(df_ineligible)
    
    logger.info(f"Eligible NTAs (count_night >= {min_cluster_count}): {n_eligible}")
    logger.info(f"Ineligible NTAs (count_night < {min_cluster_count}): {n_ineligible}")
    
    # Log top 10 lowest-volume NTAs
    if n_ineligible > 0:
        sorted_ineligible = df_ineligible.sort_values("count_night")
        top_10_lowest = sorted_ineligible.head(10)[["ntacode", "nta_name", "count_night"]]
        logger.info("Top 10 lowest-volume NTAs (excluded):")
        for _, row in top_10_lowest.iterrows():
            logger.info(f"  {row['ntacode']}: {row['nta_name']} - {row['count_night']} complaints")
    
    return df_eligible, df_ineligible


# =============================================================================
# Feature Preparation
# =============================================================================

def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, List[str]]:
    """
    Prepare standardized feature matrix for clustering.
    
    Returns:
        - df_raw: Original values for the feature columns
        - df_scaled: Standardized values
        - scaler: Fitted StandardScaler object
        - valid_cols: List of feature columns actually used
    """
    logger.info(f"Preparing feature matrix with {len(feature_cols)} requested features...")
    
    # Filter to columns that exist
    valid_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing feature columns (skipped): {missing_cols}")
    
    if not valid_cols:
        raise ValueError("No valid feature columns found!")
    
    logger.info(f"Using {len(valid_cols)} feature columns")
    
    # Extract feature matrix
    df_raw = df[valid_cols].copy()
    
    # Handle any NaN values (fill with median)
    nan_counts = df_raw.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found NaN values in features: {nan_counts[nan_counts > 0].to_dict()}")
        df_raw = df_raw.fillna(df_raw.median())
    
    # Standardize (z-score)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_raw)
    df_scaled = pd.DataFrame(scaled_values, columns=valid_cols, index=df_raw.index)
    
    logger.info(f"Feature matrix shape: {df_scaled.shape}")
    logger.info(f"Feature means (after scaling): {df_scaled.mean().mean():.4f}")
    logger.info(f"Feature stds (after scaling): {df_scaled.std().mean():.4f}")
    
    return df_raw, df_scaled, scaler, valid_cols


# =============================================================================
# Clustering
# =============================================================================

def evaluate_k_range(
    X: np.ndarray,
    k_range: List[int],
    random_seed: int,
    logger,
) -> pd.DataFrame:
    """
    Evaluate clustering for each K in the range.
    
    Returns DataFrame with silhouette, Calinski-Harabasz, and cluster size stats.
    """
    logger.info(f"Evaluating K in range {k_range}...")
    
    results = []
    
    for k in k_range:
        # Skip K values that exceed number of samples
        if k >= len(X):
            logger.warning(f"Skipping K={k}: exceeds number of samples ({len(X)})")
            continue
            
        logger.info(f"  Testing K={k}...")
        
        # Fit K-Means
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_seed,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(X)
        
        # Compute metrics
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        inertia = kmeans.inertia_
        
        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        size_min = counts.min()
        size_max = counts.max()
        size_std = counts.std()
        
        # Singleton analysis
        n_singletons = int((counts == 1).sum())
        singleton_fraction = n_singletons / k
        
        results.append({
            "k": k,
            "silhouette_score": sil_score,
            "calinski_harabasz_score": ch_score,
            "inertia": inertia,
            "min_cluster_size": int(size_min),
            "max_cluster_size": int(size_max),
            "cluster_size_std": size_std,
            "n_singletons": n_singletons,
            "singleton_fraction": singleton_fraction,
        })
        
        logger.info(f"    Silhouette: {sil_score:.4f}, CH: {ch_score:.2f}, "
                   f"Min size: {size_min}, Singletons: {n_singletons}")
    
    df_scores = pd.DataFrame(results)
    
    return df_scores


def select_optimal_k(
    df_scores: pd.DataFrame,
    max_singletons: int,
    max_singleton_fraction: float,
    logger,
) -> int:
    """
    Select optimal K based on silhouette score with singleton control.
    
    Preference:
    1. Highest silhouette among K values with acceptable singleton count
    2. If no K meets singleton criteria, use best silhouette overall with warning
    """
    logger.info(f"Selecting optimal K (max_singletons={max_singletons}, "
               f"max_singleton_fraction={max_singleton_fraction:.0%})")
    
    # Filter to K values with acceptable singleton count
    acceptable_mask = (
        (df_scores["n_singletons"] <= max_singletons) |
        (df_scores["singleton_fraction"] <= max_singleton_fraction)
    )
    
    df_acceptable = df_scores[acceptable_mask]
    
    if len(df_acceptable) > 0:
        # Select best silhouette among acceptable K values
        best_row = df_acceptable.loc[df_acceptable["silhouette_score"].idxmax()]
        best_k = int(best_row["k"])
        best_sil = best_row["silhouette_score"]
        best_singletons = int(best_row["n_singletons"])
        
        logger.info(f"Selected K={best_k} (silhouette={best_sil:.4f}, "
                   f"singletons={best_singletons}) from {len(df_acceptable)} acceptable K values")
    else:
        # Fallback: use best silhouette overall with warning
        logger.warning("No K meets singleton criteria! Using best silhouette overall.")
        best_row = df_scores.loc[df_scores["silhouette_score"].idxmax()]
        best_k = int(best_row["k"])
        best_sil = best_row["silhouette_score"]
        best_singletons = int(best_row["n_singletons"])
        
        logger.warning(f"Selected K={best_k} (silhouette={best_sil:.4f}, "
                      f"singletons={best_singletons}) - EXCEEDS SINGLETON THRESHOLD")
    
    # Log runner-up
    sorted_acceptable = df_acceptable.sort_values("silhouette_score", ascending=False) if len(df_acceptable) > 0 else df_scores.sort_values("silhouette_score", ascending=False)
    if len(sorted_acceptable) > 1:
        runner_up = sorted_acceptable.iloc[1]
        logger.info(f"Runner-up: K={int(runner_up['k'])} (silhouette={runner_up['silhouette_score']:.4f})")
    
    return best_k


def fit_final_clustering(
    X: np.ndarray,
    k: int,
    random_seed: int,
    logger,
) -> Tuple[np.ndarray, KMeans]:
    """
    Fit final K-Means model with selected K.
    """
    logger.info(f"Fitting final K-Means with K={k}...")
    
    kmeans = KMeans(
        n_clusters=k,
        random_state=random_seed,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X)
    
    # Log cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"Final cluster sizes: {dict(zip(unique.astype(int), counts.astype(int)))}")
    
    return labels, kmeans


# =============================================================================
# Cluster Summary
# =============================================================================

def compute_cluster_summary(
    df: pd.DataFrame,
    feature_cols: List[str],
    logger,
) -> pd.DataFrame:
    """
    Compute summary statistics (mean, median) for each cluster.
    """
    logger.info("Computing cluster summary...")
    
    # Group by cluster_id
    grouped = df.groupby("cluster_id")
    
    summaries = []
    for cluster_id, group in grouped:
        summary = {
            "cluster_id": cluster_id,
            "n_ntas": len(group),
            "cluster_label": group["cluster_label"].iloc[0] if "cluster_label" in group.columns else f"Cluster_{cluster_id}",
        }
        
        for col in feature_cols:
            if col in group.columns:
                summary[f"{col}_mean"] = group[col].mean()
                summary[f"{col}_median"] = group[col].median()
        
        summaries.append(summary)
    
    df_summary = pd.DataFrame(summaries)
    df_summary = df_summary.sort_values("cluster_id").reset_index(drop=True)
    
    logger.info(f"Created summary with {len(df_summary)} clusters, {len(df_summary.columns)} columns")
    
    return df_summary


def identify_distinguishing_features(
    df_summary: pd.DataFrame,
    feature_cols: List[str],
    logger,
) -> Dict[int, List[str]]:
    """
    For each cluster, identify the top distinguishing features
    (those most different from overall mean).
    """
    logger.info("Identifying distinguishing features per cluster...")
    
    distinguishing = {}
    
    # Exclude "Low Volume" cluster from z-score computation
    df_clustered = df_summary[df_summary["cluster_id"] != -1]
    
    for _, row in df_summary.iterrows():
        cluster_id = int(row["cluster_id"])
        
        if cluster_id == -1:
            distinguishing[cluster_id] = ["Low Volume (excluded from clustering)"]
            continue
        
        # Get mean columns
        mean_cols = [f"{col}_mean" for col in feature_cols if f"{col}_mean" in row.index]
        
        # Calculate z-scores relative to overall means (across clustered clusters only)
        deviations = []
        for col in mean_cols:
            overall_mean = df_clustered[col].mean()
            overall_std = df_clustered[col].std()
            if overall_std > 0:
                z = (row[col] - overall_mean) / overall_std
                feature_name = col.replace("_mean", "")
                deviations.append((feature_name, z, row[col]))
        
        # Sort by absolute z-score
        deviations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top 3
        top_features = []
        for feat, z, val in deviations[:3]:
            direction = "high" if z > 0 else "low"
            top_features.append(f"{feat} ({direction})")
        
        distinguishing[cluster_id] = top_features
        logger.info(f"  Cluster {cluster_id}: {top_features}")
    
    return distinguishing


# =============================================================================
# QA / Validation
# =============================================================================

def validate_clustering(
    df: pd.DataFrame,
    k: int,
    min_cluster_size: int,
    n_ineligible: int,
    logger,
) -> Dict:
    """
    Validate clustering output.
    """
    logger.info("Validating clustering output...")
    
    qa_stats = {}
    passed = True
    
    # Row count (should be 197 for residential NTAs)
    qa_stats["row_count"] = len(df)
    expected_rows = 197
    if len(df) != expected_rows:
        logger.warning(f"Expected {expected_rows} rows, got {len(df)}")
        # Not a hard failure - just warning
    
    # No null cluster_id
    null_clusters = df["cluster_id"].isna().sum()
    qa_stats["null_cluster_ids"] = int(null_clusters)
    if null_clusters > 0:
        logger.error(f"Found {null_clusters} null cluster_id values!")
        passed = False
    
    # cluster_id in [-1, 0..K-1]
    valid_ids = list(range(-1, k))
    invalid_ids = df[~df["cluster_id"].isin(valid_ids)]["cluster_id"].unique()
    qa_stats["invalid_cluster_ids"] = list(invalid_ids)
    if len(invalid_ids) > 0:
        logger.error(f"Found invalid cluster_id values: {invalid_ids}")
        passed = False
    
    # Cluster sizes - convert numpy types to native Python for JSON serialization
    cluster_sizes = df["cluster_id"].value_counts().sort_index()
    qa_stats["cluster_sizes"] = {int(k): int(v) for k, v in cluster_sizes.to_dict().items()}
    
    # Low Volume cluster count
    low_volume_count = int((df["cluster_id"] == -1).sum())
    qa_stats["low_volume_count"] = low_volume_count
    if low_volume_count != n_ineligible:
        logger.error(f"Low Volume count ({low_volume_count}) doesn't match ineligible count ({n_ineligible})")
        passed = False
    
    # Check for degenerate clusters (excluding Low Volume)
    clustered_sizes = cluster_sizes[cluster_sizes.index != -1]
    if len(clustered_sizes) > 0:
        min_size = clustered_sizes.min()
        qa_stats["min_cluster_size"] = int(min_size)
        
        if min_size == 0:
            logger.error("Found cluster with 0 members!")
            passed = False
        elif min_size < min_cluster_size:
            logger.warning(f"Cluster with only {min_size} members (threshold: {min_cluster_size})")
    
    # NTA name check
    null_names = df["nta_name"].isna().sum()
    qa_stats["null_nta_names"] = int(null_names)
    if null_names > 0:
        logger.error(f"Found {null_names} null nta_name values!")
        passed = False
    
    # Cluster label check
    null_labels = df["cluster_label"].isna().sum()
    qa_stats["null_cluster_labels"] = int(null_labels)
    if null_labels > 0:
        logger.error(f"Found {null_labels} null cluster_label values!")
        passed = False
    
    # is_outlier_cluster for Low Volume
    low_vol_outlier = df[df["cluster_id"] == -1]["is_outlier_cluster"].all()
    qa_stats["low_volume_all_outlier"] = bool(low_vol_outlier)
    if not low_vol_outlier and low_volume_count > 0:
        logger.error("Low Volume NTAs should have is_outlier_cluster=True")
        passed = False
    
    qa_stats["passed"] = passed
    logger.info(f"QA validation {'PASSED' if passed else 'FAILED'}")
    
    return qa_stats


def verify_reproducibility(
    X: np.ndarray,
    k: int,
    random_seed: int,
    original_labels: np.ndarray,
    logger,
) -> bool:
    """
    Verify that rerunning clustering produces identical results.
    """
    logger.info("Verifying reproducibility...")
    
    kmeans2 = KMeans(
        n_clusters=k,
        random_state=random_seed,
        n_init=10,
        max_iter=300,
    )
    labels2 = kmeans2.fit_predict(X)
    
    matches = np.array_equal(original_labels, labels2)
    
    if matches:
        logger.info("✓ Reproducibility check PASSED: identical cluster assignments")
    else:
        logger.error("✗ Reproducibility check FAILED: different cluster assignments")
    
    return matches


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("11_build_nta_typology") as logger:
        logger.info("Starting 11_build_nta_typology.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        nta_typology_config = config.get("nta_typology", {})
        k_range = nta_typology_config.get("k_range", [8, 9, 10, 11, 12, 13, 14, 15])
        feature_cols = nta_typology_config.get("features", [])
        min_silhouette = nta_typology_config.get("min_silhouette", 0.1)
        min_cluster_size = nta_typology_config.get("min_cluster_size", 2)
        min_cluster_count = nta_typology_config.get("min_cluster_count", 200)
        
        singleton_control = nta_typology_config.get("singleton_control", {})
        max_singletons = singleton_control.get("max_singletons", 2)
        max_singleton_fraction = singleton_control.get("max_singleton_fraction", 0.10)
        
        random_seed = config.get("random_seeds", {}).get("clustering", 12345)
        
        logger.info(f"K range: {k_range}")
        logger.info(f"Min cluster count: {min_cluster_count}")
        logger.info(f"Random seed: {random_seed}")
        logger.info(f"Feature count: {len(feature_cols)}")
        logger.info(f"Singleton control: max_singletons={max_singletons}, max_fraction={max_singleton_fraction:.0%}")
        
        try:
            # Load data
            df_features = load_nta_features(logger)
            nta_geo = load_nta_geo(logger)
            nta_lookup = load_nta_lookup(logger)
            
            # Apply eligibility guardrail
            df_eligible, df_ineligible = apply_eligibility_guardrail(
                df_features, min_cluster_count, logger
            )
            
            n_eligible = len(df_eligible)
            n_ineligible = len(df_ineligible)
            
            if n_eligible < k_range[0]:
                raise ValueError(
                    f"Only {n_eligible} eligible NTAs, but minimum K is {k_range[0]}. "
                    "Consider lowering min_cluster_count or k_range."
                )
            
            # Prepare feature matrix (only for eligible NTAs)
            df_raw, df_scaled, scaler, valid_cols = prepare_feature_matrix(
                df_eligible, feature_cols, logger
            )
            
            logger.info(f"Feature columns used: {valid_cols}")
            
            # Convert to numpy for sklearn
            X = df_scaled.values
            
            # Evaluate K range
            df_scores = evaluate_k_range(X, k_range, random_seed, logger)
            
            # Select optimal K
            best_k = select_optimal_k(df_scores, max_singletons, max_singleton_fraction, logger)
            
            # Check silhouette threshold
            best_sil = df_scores.loc[df_scores["k"] == best_k, "silhouette_score"].values[0]
            if best_sil < min_silhouette:
                logger.warning(
                    f"Best silhouette ({best_sil:.4f}) below threshold ({min_silhouette}). "
                    "Clustering may not be meaningful."
                )
            
            # Fit final model
            labels, kmeans = fit_final_clustering(X, best_k, random_seed, logger)
            
            # Verify reproducibility
            repro_ok = verify_reproducibility(X, best_k, random_seed, labels, logger)
            
            # Create output DataFrame for eligible NTAs
            df_clusters_eligible = df_eligible[["ntacode", "nta_name", "borough_name", "is_residential", "ntatype_label"]].copy()
            df_clusters_eligible["cluster_id"] = labels
            df_clusters_eligible["cluster_label"] = df_clusters_eligible["cluster_id"].apply(
                lambda x: f"Cluster_{x}"
            )
            df_clusters_eligible["is_outlier_cluster"] = False
            
            # Mark singleton clusters as outliers
            cluster_sizes = pd.Series(labels).value_counts()
            singleton_clusters = cluster_sizes[cluster_sizes == 1].index.tolist()
            df_clusters_eligible.loc[
                df_clusters_eligible["cluster_id"].isin(singleton_clusters),
                "is_outlier_cluster"
            ] = True
            
            logger.info(f"Singleton clusters: {singleton_clusters}")
            
            # Create output DataFrame for ineligible NTAs (Low Volume)
            df_clusters_ineligible = df_ineligible[["ntacode", "nta_name", "borough_name", "is_residential", "ntatype_label"]].copy()
            df_clusters_ineligible["cluster_id"] = -1
            df_clusters_ineligible["cluster_label"] = "Low Volume"
            df_clusters_ineligible["is_outlier_cluster"] = True
            
            # Combine
            df_clusters = pd.concat([df_clusters_eligible, df_clusters_ineligible], ignore_index=True)
            
            # Also add the raw feature values for interpretation
            # Need to use original index alignment
            for col in valid_cols:
                if col in df_eligible.columns:
                    # Map from ntacode
                    eligible_vals = df_eligible.set_index("ntacode")[col]
                    ineligible_vals = df_ineligible.set_index("ntacode")[col] if col in df_ineligible.columns else pd.Series(dtype=float)
                    
                    all_vals = pd.concat([eligible_vals, ineligible_vals])
                    df_clusters[col] = df_clusters["ntacode"].map(all_vals)
            
            # Ensure proper types
            df_clusters["cluster_id"] = df_clusters["cluster_id"].astype("Int64")
            
            # Sort by cluster_id, then ntacode
            df_clusters = df_clusters.sort_values(["cluster_id", "ntacode"]).reset_index(drop=True)
            
            # Compute cluster summary
            df_summary = compute_cluster_summary(df_clusters, valid_cols, logger)
            
            # Identify distinguishing features
            distinguishing = identify_distinguishing_features(df_summary, valid_cols, logger)
            
            # Validate
            qa_stats = validate_clustering(df_clusters, best_k, min_cluster_size, n_ineligible, logger)
            
            # Ensure output directory exists
            TYPOLOGY_DIR.mkdir(parents=True, exist_ok=True)
            
            # Write outputs
            # 1. Cluster assignments
            atomic_write_df(df_clusters, OUTPUT_CLUSTERS)
            logger.info(f"Wrote: {OUTPUT_CLUSTERS}")
            
            df_clusters.to_csv(OUTPUT_CLUSTERS_CSV, index=False)
            logger.info(f"Wrote: {OUTPUT_CLUSTERS_CSV}")
            
            # 2. Cluster summary
            df_summary.to_csv(OUTPUT_SUMMARY, index=False)
            logger.info(f"Wrote: {OUTPUT_SUMMARY}")
            
            # 3. K selection scores
            df_scores.to_csv(OUTPUT_K_SCORES, index=False)
            logger.info(f"Wrote: {OUTPUT_K_SCORES}")
            
            # 4. Standardized feature matrix (for reproducibility)
            df_scaled_with_id = df_scaled.copy()
            df_scaled_with_id.insert(0, "ntacode", df_eligible["ntacode"].values)
            df_scaled_with_id.to_csv(OUTPUT_FEATURE_MATRIX, index=False)
            logger.info(f"Wrote: {OUTPUT_FEATURE_MATRIX}")
            
            # 5. GeoJSON (map-ready)
            # Filter NTA geo to residential only
            residential_ntacodes = df_clusters["ntacode"].tolist()
            nta_geo_residential = nta_geo[nta_geo["ntacode"].isin(residential_ntacodes)].copy()
            
            gdf_output = nta_geo_residential[["ntacode", "geometry"]].merge(
                df_clusters[["ntacode", "nta_name", "cluster_id", "cluster_label", "is_outlier_cluster"]],
                on="ntacode",
            )
            atomic_write_gdf(gdf_output, OUTPUT_GEOJSON)
            logger.info(f"Wrote: {OUTPUT_GEOJSON}")
            
            # Log outputs
            logger.log_outputs({
                "nta_clusters_parquet": str(OUTPUT_CLUSTERS),
                "nta_clusters_csv": str(OUTPUT_CLUSTERS_CSV),
                "nta_cluster_summary": str(OUTPUT_SUMMARY),
                "nta_k_selection_scores": str(OUTPUT_K_SCORES),
                "nta_feature_matrix_standardized": str(OUTPUT_FEATURE_MATRIX),
                "nta_clusters_geojson": str(OUTPUT_GEOJSON),
            })
            
            # Log metrics
            logger.log_metrics({
                "selected_k": best_k,
                "best_silhouette": float(best_sil),
                "best_calinski_harabasz": float(
                    df_scores.loc[df_scores["k"] == best_k, "calinski_harabasz_score"].values[0]
                ),
                "features_used": len(valid_cols),
                "random_seed": random_seed,
                "reproducibility_verified": repro_ok,
                "n_eligible_ntas": n_eligible,
                "n_ineligible_ntas": n_ineligible,
                "n_singletons": int(df_scores.loc[df_scores["k"] == best_k, "n_singletons"].values[0]),
                "cluster_sizes": qa_stats.get("cluster_sizes", {}),
                "qa_stats": qa_stats,
            })
            
            # Write metadata sidecar
            # Convert distinguishing dict keys to strings for JSON serialization
            distinguishing_str = {str(k): v for k, v in distinguishing.items()}
            
            write_metadata_sidecar(
                output_path=OUTPUT_CLUSTERS,
                inputs={
                    "nta_features_residential": str(INPUT_NTA_FEATURES),
                    "nta_geo": str(INPUT_NTA_GEO),
                },
                config=config,
                run_id=logger.run_id,
                extra={
                    "selected_k": best_k,
                    "silhouette_score": float(best_sil),
                    "features_used": valid_cols,
                    "random_seed": random_seed,
                    "reproducibility_verified": repro_ok,
                    "min_cluster_count": min_cluster_count,
                    "n_eligible_ntas": n_eligible,
                    "n_ineligible_ntas": n_ineligible,
                    "k_scores": df_scores.to_dict(orient="records"),
                    "cluster_sizes": qa_stats.get("cluster_sizes", {}),
                    "distinguishing_features": distinguishing_str,
                    "qa_stats": qa_stats,
                },
            )
            
            # Print summary
            logger.info("=" * 70)
            logger.info("NTA Night Signatures Typology Summary:")
            logger.info(f"  Selected K: {best_k}")
            logger.info(f"  Silhouette score: {best_sil:.4f}")
            logger.info(f"  Features used: {len(valid_cols)}")
            logger.info(f"  Eligible NTAs: {n_eligible} (count_night >= {min_cluster_count})")
            logger.info(f"  Low Volume NTAs: {n_ineligible} (count_night < {min_cluster_count})")
            logger.info(f"  Cluster sizes: {qa_stats.get('cluster_sizes', {})}")
            logger.info("")
            logger.info("Distinguishing features per cluster:")
            for cid, feats in distinguishing.items():
                logger.info(f"  Cluster {cid}: {feats}")
            logger.info("=" * 70)
            
            # Show cluster composition by borough
            logger.info("Cluster composition by borough:")
            for cluster_id in sorted(df_clusters["cluster_id"].unique()):
                cluster_df = df_clusters[df_clusters["cluster_id"] == cluster_id]
                boro_counts = cluster_df["borough_name"].value_counts().to_dict()
                sample_ntas = cluster_df["nta_name"].head(5).tolist()
                label = cluster_df["cluster_label"].iloc[0]
                logger.info(f"  Cluster {cluster_id} ({label}, {len(cluster_df)} NTAs): {boro_counts}")
                logger.info(f"    Examples: {sample_ntas[:5]}")
            
            logger.info("")
            logger.info("NOTE: Cluster labels are placeholders. User review required.")
            logger.info("SUCCESS: Built NTA night signatures typology")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

