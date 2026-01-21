#!/usr/bin/env python3
"""
05_build_typology.py

Build Night Signatures Typology (clustering) for NYC Community Districts.

Per NYC_Night_Signals_Plan.md Section 3.3.5:
- Build feature matrix from Script 03 outputs (311 behavior only, no objective layers)
- Standardize features (z-score)
- Cluster CDs using K-Means with K in [5..9]
- Select optimal K using silhouette score (Calinski-Harabasz as secondary)
- Log all scores for each K

Outputs:
- data/processed/typology/cd_clusters.parquet (boro_cd, cd_label, cluster_id, cluster_label_placeholder)
- data/processed/typology/cluster_summary.csv (means/medians of key features by cluster)
- data/processed/typology/cd_clusters.geojson (map-ready with geometry)
- data/processed/metadata/cd_clusters_metadata.json (provenance sidecar)

Key requirements:
- Fixed random seed for reproducibility
- Log all parameters and feature columns used
- Save standardized feature matrix column names
- QA: 59 rows, no null cluster_id, cluster sizes not degenerate
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
from sleep_esi.qa import assert_cd_labels_present
from sleep_esi.schemas import ensure_boro_cd_dtype


# =============================================================================
# Constants
# =============================================================================

ATLAS_DIR = PROCESSED_DIR / "atlas"
TYPOLOGY_DIR = PROCESSED_DIR / "typology"

# Input
INPUT_FEATURES = ATLAS_DIR / "311_cd_features.parquet"

# Outputs
OUTPUT_CLUSTERS = TYPOLOGY_DIR / "cd_clusters.parquet"
OUTPUT_CLUSTERS_CSV = TYPOLOGY_DIR / "cd_clusters.csv"
OUTPUT_SUMMARY = TYPOLOGY_DIR / "cluster_summary.csv"
OUTPUT_GEOJSON = TYPOLOGY_DIR / "cd_clusters.geojson"
OUTPUT_FEATURE_MATRIX = TYPOLOGY_DIR / "feature_matrix_standardized.csv"
OUTPUT_K_SCORES = TYPOLOGY_DIR / "k_selection_scores.csv"


# =============================================================================
# Data Loading
# =============================================================================

def load_311_features(logger) -> pd.DataFrame:
    """Load the 311 CD-level features from Script 03."""
    if not INPUT_FEATURES.exists():
        raise FileNotFoundError(
            f"311 features not found: {INPUT_FEATURES}. "
            "Run 03_build_311_night_features.py first."
        )
    
    df = pd.read_parquet(INPUT_FEATURES)
    df = ensure_boro_cd_dtype(df)
    logger.info(f"Loaded 311 features: {len(df)} CDs, {len(df.columns)} columns")
    
    return df


def load_cd59(logger) -> gpd.GeoDataFrame:
    """Load the canonical CD59 geometries."""
    cd59_path = GEO_DIR / "cd59.parquet"
    
    if not cd59_path.exists():
        raise FileNotFoundError(
            f"CD59 file not found: {cd59_path}. "
            "Run 00_build_geographies.py first."
        )
    
    gdf = read_gdf(cd59_path)
    gdf = ensure_boro_cd_dtype(gdf)
    logger.info(f"Loaded CD59: {len(gdf)} community districts")
    
    return gdf


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
    
    Returns DataFrame with silhouette and Calinski-Harabasz scores for each K.
    """
    logger.info(f"Evaluating K in range {k_range}...")
    
    results = []
    
    for k in k_range:
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
        
        results.append({
            "k": k,
            "silhouette_score": sil_score,
            "calinski_harabasz_score": ch_score,
            "inertia": inertia,
            "cluster_size_min": size_min,
            "cluster_size_max": size_max,
            "cluster_size_std": size_std,
        })
        
        logger.info(f"    Silhouette: {sil_score:.4f}, CH: {ch_score:.2f}, "
                   f"Sizes: {list(counts)}")
    
    df_scores = pd.DataFrame(results)
    
    return df_scores


def select_optimal_k(df_scores: pd.DataFrame, logger) -> int:
    """
    Select optimal K based on silhouette score (primary).
    """
    best_row = df_scores.loc[df_scores["silhouette_score"].idxmax()]
    best_k = int(best_row["k"])
    best_sil = best_row["silhouette_score"]
    
    logger.info(f"Selected K={best_k} with silhouette={best_sil:.4f}")
    
    # Log runner-up
    sorted_df = df_scores.sort_values("silhouette_score", ascending=False)
    if len(sorted_df) > 1:
        runner_up = sorted_df.iloc[1]
        logger.info(f"Runner-up: K={int(runner_up['k'])} with silhouette={runner_up['silhouette_score']:.4f}")
    
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
    logger.info(f"Final cluster sizes: {dict(zip(unique, counts))}")
    
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
            "n_cds": len(group),
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
    
    for _, row in df_summary.iterrows():
        cluster_id = int(row["cluster_id"])
        
        # Get mean columns
        mean_cols = [f"{col}_mean" for col in feature_cols if f"{col}_mean" in row.index]
        
        # Calculate z-scores relative to overall means (across clusters)
        deviations = []
        for col in mean_cols:
            overall_mean = df_summary[col].mean()
            overall_std = df_summary[col].std()
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
    logger,
) -> Dict:
    """
    Validate clustering output.
    """
    logger.info("Validating clustering output...")
    
    qa_stats = {}
    passed = True
    
    # Row count
    qa_stats["row_count"] = len(df)
    if len(df) != 59:
        logger.error(f"Expected 59 rows, got {len(df)}")
        passed = False
    
    # No null cluster_id
    null_clusters = df["cluster_id"].isna().sum()
    qa_stats["null_cluster_ids"] = int(null_clusters)
    if null_clusters > 0:
        logger.error(f"Found {null_clusters} null cluster_id values!")
        passed = False
    
    # cluster_id in [0..K-1]
    valid_range = df["cluster_id"].between(0, k - 1).all()
    qa_stats["cluster_ids_in_valid_range"] = valid_range
    if not valid_range:
        logger.error(f"Some cluster_id values outside [0, {k-1}]")
        passed = False
    
    # Cluster sizes - convert numpy types to native Python for JSON serialization
    cluster_sizes = df["cluster_id"].value_counts().sort_index()
    qa_stats["cluster_sizes"] = {int(k): int(v) for k, v in cluster_sizes.to_dict().items()}
    
    # Check for degenerate clusters
    min_size = cluster_sizes.min()
    qa_stats["min_cluster_size"] = int(min_size)
    
    if min_size == 0:
        logger.error("Found cluster with 0 members!")
        passed = False
    elif min_size < min_cluster_size:
        logger.warning(f"Cluster with only {min_size} members (threshold: {min_cluster_size})")
    
    # CD label check
    null_labels = df["cd_label"].isna().sum()
    qa_stats["null_cd_labels"] = int(null_labels)
    if null_labels > 0:
        logger.error(f"Found {null_labels} null cd_label values!")
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
    with get_logger("05_build_typology") as logger:
        logger.info("Starting 05_build_typology.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        typology_config = config.get("typology", {})
        k_range = typology_config.get("k_range", [5, 6, 7, 8, 9])
        feature_cols = typology_config.get("features", [])
        min_silhouette = typology_config.get("min_silhouette", 0.1)
        min_cluster_size = typology_config.get("min_cluster_size", 2)
        random_seed = config.get("random_seeds", {}).get("clustering", 12345)
        
        logger.info(f"K range: {k_range}")
        logger.info(f"Random seed: {random_seed}")
        logger.info(f"Feature count: {len(feature_cols)}")
        
        try:
            # Load data
            df_features = load_311_features(logger)
            cd59 = load_cd59(logger)
            
            # Prepare feature matrix
            df_raw, df_scaled, scaler, valid_cols = prepare_feature_matrix(
                df_features, feature_cols, logger
            )
            
            logger.info(f"Feature columns used: {valid_cols}")
            
            # Convert to numpy for sklearn
            X = df_scaled.values
            
            # Evaluate K range
            df_scores = evaluate_k_range(X, k_range, random_seed, logger)
            
            # Select optimal K
            best_k = select_optimal_k(df_scores, logger)
            
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
            
            # Create output DataFrame
            df_clusters = df_features[["boro_cd", "cd_label", "cd_short", "borough_name"]].copy()
            df_clusters["cluster_id"] = labels
            
            # Apply final cluster labels from config
            cluster_labels = typology_config.get("cluster_labels", {})
            outlier_clusters = typology_config.get("outlier_clusters", [])
            
            # Map cluster_id to label (use placeholder if not in config)
            df_clusters["cluster_label"] = df_clusters["cluster_id"].apply(
                lambda x: cluster_labels.get(x, cluster_labels.get(str(x), f"Cluster_{x}"))
            )
            
            # Add is_outlier_cluster flag
            df_clusters["is_outlier_cluster"] = df_clusters["cluster_id"].isin(outlier_clusters)
            
            logger.info(f"Applied {len(cluster_labels)} cluster labels from config")
            logger.info(f"Outlier clusters: {outlier_clusters}")
            
            # Also add the raw feature values for interpretation
            for col in valid_cols:
                df_clusters[col] = df_features[col].values
            
            # Ensure proper types
            df_clusters = ensure_boro_cd_dtype(df_clusters)
            df_clusters["cluster_id"] = df_clusters["cluster_id"].astype("Int64")
            
            # Compute cluster summary
            df_summary = compute_cluster_summary(df_clusters, valid_cols, logger)
            
            # Identify distinguishing features
            distinguishing = identify_distinguishing_features(df_summary, valid_cols, logger)
            
            # Validate
            qa_stats = validate_clustering(df_clusters, best_k, min_cluster_size, logger)
            
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
            df_scaled_with_id.insert(0, "boro_cd", df_features["boro_cd"].values)
            df_scaled_with_id.to_csv(OUTPUT_FEATURE_MATRIX, index=False)
            logger.info(f"Wrote: {OUTPUT_FEATURE_MATRIX}")
            
            # 5. GeoJSON (map-ready)
            gdf_output = cd59[["boro_cd", "geometry"]].merge(
                df_clusters[["boro_cd", "cd_label", "cd_short", "cluster_id", "cluster_label", "is_outlier_cluster"]],
                on="boro_cd",
            )
            gdf_output = ensure_boro_cd_dtype(gdf_output)
            atomic_write_gdf(gdf_output, OUTPUT_GEOJSON)
            logger.info(f"Wrote: {OUTPUT_GEOJSON}")
            
            # Log outputs
            logger.log_outputs({
                "cd_clusters_parquet": str(OUTPUT_CLUSTERS),
                "cd_clusters_csv": str(OUTPUT_CLUSTERS_CSV),
                "cluster_summary": str(OUTPUT_SUMMARY),
                "k_selection_scores": str(OUTPUT_K_SCORES),
                "feature_matrix_standardized": str(OUTPUT_FEATURE_MATRIX),
                "cd_clusters_geojson": str(OUTPUT_GEOJSON),
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
                "cluster_sizes": qa_stats.get("cluster_sizes", {}),
                "qa_stats": qa_stats,
            })
            
            # Write metadata sidecar
            # Convert distinguishing dict keys to strings for JSON serialization
            distinguishing_str = {str(k): v for k, v in distinguishing.items()}
            
            write_metadata_sidecar(
                output_path=OUTPUT_CLUSTERS,
                inputs={"311_cd_features": str(INPUT_FEATURES)},
                config=config,
                run_id=logger.run_id,
                extra={
                    "selected_k": best_k,
                    "silhouette_score": float(best_sil),
                    "features_used": valid_cols,
                    "random_seed": random_seed,
                    "reproducibility_verified": repro_ok,
                    "k_scores": df_scores.to_dict(orient="records"),
                    "cluster_sizes": qa_stats.get("cluster_sizes", {}),
                    "distinguishing_features": distinguishing_str,
                    "qa_stats": qa_stats,
                },
            )
            
            # Print summary
            logger.info("=" * 70)
            logger.info("Night Signatures Typology Summary:")
            logger.info(f"  Selected K: {best_k}")
            logger.info(f"  Silhouette score: {best_sil:.4f}")
            logger.info(f"  Features used: {len(valid_cols)}")
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
                cds = cluster_df["cd_short"].tolist()
                logger.info(f"  Cluster {cluster_id} ({len(cluster_df)} CDs): {boro_counts}")
                logger.info(f"    CDs: {cds}")
            
            logger.info("")
            logger.info("NOTE: Cluster labels are placeholders. User review required.")
            logger.info("SUCCESS: Built night signatures typology")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

