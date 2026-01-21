#!/usr/bin/env python3
"""
08b_build_typology_assets.py

Build Typology/Type Assets for NYC Night Signals Atlas.

Creates presentation-ready cluster profiles and map layers for both CD and NTA levels.

Outputs:
1) Cluster profile tables with key feature means
2) Human-readable cluster labels file
3) Presentation-ready joined tables + map layers with stability overlay
4) Bar charts showing key feature means per cluster

Usage:
  python scripts/08b_build_typology_assets.py         # Full run (CD + NTA)
  python scripts/08b_build_typology_assets.py --cd    # CD only
  python scripts/08b_build_typology_assets.py --nta   # NTA only
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import json
import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sleep_esi.hashing import write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_df, atomic_write_gdf, read_yaml, read_gdf
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, PROCESSED_DIR
from sleep_esi.schemas import ensure_boro_cd_dtype


# =============================================================================
# Constants
# =============================================================================

TYPOLOGY_DIR = PROCESSED_DIR / "typology"
REPORTS_DIR = PROCESSED_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Key features for profile tables and charts
KEY_FEATURES = [
    "rate_per_1k_pop",
    "rate_per_km2",
    "share_noise___residential",
    "share_noise___street/sidewalk",
    "share_noise___commercial",
    "share_noise___vehicle",
    "share_noise___helicopter",
    "share_evening",
    "share_early_am",
    "share_core_night",
    "share_predawn",
    "late_night_share",
    "weekend_uplift",
    "warm_season_ratio",
]

# Short names for features (for charts)
FEATURE_SHORT_NAMES = {
    "rate_per_1k_pop": "Rate/1k",
    "rate_per_km2": "Rate/km²",
    "share_noise___residential": "Residential",
    "share_noise___street/sidewalk": "Street",
    "share_noise___commercial": "Commercial",
    "share_noise___vehicle": "Vehicle",
    "share_noise___helicopter": "Helicopter",
    "share_evening": "Evening",
    "share_early_am": "Early AM",
    "share_core_night": "Core Night",
    "share_predawn": "Pre-dawn",
    "late_night_share": "Late Night",
    "weekend_uplift": "Weekend Uplift",
    "warm_season_ratio": "Warm Season",
}


# =============================================================================
# Cluster Label Definitions
# =============================================================================

# CD cluster labels (from Script 05 analysis)
CD_CLUSTER_LABELS = {
    0: {
        "short": "Street-Dominated Activity",
        "long": "High street/sidewalk noise with evening peak",
        "defining": "street/sidewalk share > 0.25, evening share high",
    },
    1: {
        "short": "Weekend-Heavy Residential",
        "long": "Residential noise with strong weekend uplift",
        "defining": "weekend_uplift > 1.9, residential share moderate",
    },
    2: {
        "short": "Helicopter Corridor",
        "long": "Dominated by helicopter noise complaints",
        "defining": "helicopter share > 0.35, unusual time distribution",
    },
    3: {
        "short": "House of Worship Hotspot",
        "long": "Vehicle + house of worship noise concentration",
        "defining": "house_of_worship > 0.07, vehicle high - OUTLIER",
    },
    4: {
        "short": "Low-Intensity Baseline",
        "long": "Low complaint rates, generic noise dominant",
        "defining": "rate_per_1k_pop < 35, generic 'Noise' share high",
    },
    5: {
        "short": "Extreme Residential Disturbance",
        "long": "Very high rates, nearly all residential",
        "defining": "rate_per_1k_pop > 250, residential > 0.94 - OUTLIER",
    },
    6: {
        "short": "Street Activity Evening Peak",
        "long": "Street noise with evening concentration",
        "defining": "street/sidewalk > 0.35, evening > 0.45",
    },
    7: {
        "short": "Nightlife & Entertainment",
        "long": "Late-night and generic noise, entertainment areas",
        "defining": "late_night > 0.28, generic 'Noise' > 0.30",
    },
    8: {
        "short": "Park & Commercial Mix",
        "long": "Park noise with commercial component",
        "defining": "park share > 0.05, commercial > 0.10 - OUTLIER",
    },
}

# NTA cluster labels (from Script 11 analysis)
NTA_CLUSTER_LABELS = {
    -1: {
        "short": "Low Volume",
        "long": "Insufficient complaint volume for clustering",
        "defining": "count_night < 200",
    },
    0: {
        "short": "Residential Evening/Weekend Profile",
        "long": "High residential share with evening peak and weekend uplift",
        "defining": "residential z=+0.70, evening z=+0.94, weekend_uplift z=+0.96",
    },
    1: {
        "short": "Late-Night Activity Profile",
        "long": "Late-night concentration with lower residential share",
        "defining": "late_night z=+0.81, residential z=-1.09",
    },
    2: {
        "short": "Extreme Residential/Seasonal Outlier",
        "long": "Very high rates, extreme warm season spike - single NTA outlier",
        "defining": "rate z=+12.31, warm_season z=+3.70 - OUTLIER",
    },
    3: {
        "short": "Low-Activity Outlier",
        "long": "Unusually low evening activity with atypical temporal pattern",
        "defining": "evening z=-2.72, rate z=+2.27 - OUTLIER",
    },
    4: {
        "short": "Commercial-Heavy Profile",
        "long": "High commercial noise share with evening activity",
        "defining": "commercial z=+1.73, evening z=+0.44",
    },
    5: {
        "short": "Residential Late-Night Profile",
        "long": "High residential share with moderate late-night activity",
        "defining": "residential z=+0.61, late_night z=+0.48",
    },
    6: {
        "short": "Street Activity Profile",
        "long": "High street/sidewalk noise with warm season emphasis",
        "defining": "street z=+2.22, warm_season z=+1.43",
    },
    7: {
        "short": "Helicopter Corridor Profile",
        "long": "Dominated by helicopter noise complaints",
        "defining": "helicopter z=+3.67 - distinctive outlier pattern",
    },
    8: {
        "short": "Balanced/Mixed Profile",
        "long": "No strongly distinguishing features, moderate on all metrics",
        "defining": "all z-scores near 0, balanced noise mix",
    },
}


# =============================================================================
# Stability Classification
# =============================================================================

def add_stability_class2(df: pd.DataFrame, mostly_structural_threshold: float = 0.60) -> pd.DataFrame:
    """
    Add secondary stability classification.
    
    stability_class2:
    - "Strict Structural": entropy = 0 (same cluster all 3 years)
    - "Mostly Structural": is_stable_2_of_3 AND entropy <= threshold
    - "Episodic": everything else
    
    Args:
        df: DataFrame with stability columns
        mostly_structural_threshold: entropy threshold for "Mostly Structural" (config-driven)
    """
    df = df.copy()
    
    def classify(row):
        if row["entropy_score"] == 0:
            return "Strict Structural"
        elif row["is_stable_2_of_3"] and row["entropy_score"] <= mostly_structural_threshold:
            return "Mostly Structural"
        else:
            return "Episodic"
    
    df["is_mostly_structural"] = (df["is_stable_2_of_3"]) & (df["entropy_score"] <= mostly_structural_threshold)
    df["stability_class2"] = df.apply(classify, axis=1)
    
    return df


# =============================================================================
# Profile Table Generation
# =============================================================================

def build_cluster_profiles(
    df_clusters: pd.DataFrame,
    features: List[str],
    id_col: str,
    logger,
) -> pd.DataFrame:
    """
    Build cluster profile table with raw and z-scored means.
    """
    logger.info("Building cluster profile table...")
    
    # Filter to available features
    available_features = [f for f in features if f in df_clusters.columns]
    
    # Compute raw means
    profiles = df_clusters.groupby("cluster_id")[available_features].agg(["mean", "std", "count"])
    
    # Flatten column names
    profiles.columns = [f"{col[0]}_{col[1]}" for col in profiles.columns]
    profiles = profiles.reset_index()
    
    # Get n_units from count column
    count_cols = [c for c in profiles.columns if c.endswith("_count")]
    if count_cols:
        profiles["n_units"] = profiles[count_cols[0]]
        profiles = profiles.drop(columns=count_cols)
    
    # Get cluster labels - prefer existing labels from df_clusters, or leave for later mapping
    if "cluster_label" in df_clusters.columns:
        cluster_labels = df_clusters.groupby("cluster_id")["cluster_label"].first().to_dict()
        profiles["cluster_label"] = profiles["cluster_id"].map(cluster_labels)
    else:
        # Will be filled in by caller with predefined labels
        profiles["cluster_label"] = profiles["cluster_id"].map(lambda x: f"Cluster_{x}")
    
    # Compute z-scored means
    for feat in available_features:
        mean_col = f"{feat}_mean"
        if mean_col in profiles.columns:
            overall_mean = df_clusters[feat].mean()
            overall_std = df_clusters[feat].std()
            if overall_std > 0:
                profiles[f"{feat}_z"] = (profiles[mean_col] - overall_mean) / overall_std
            else:
                profiles[f"{feat}_z"] = 0.0
    
    # Drop std columns for cleaner output
    std_cols = [c for c in profiles.columns if c.endswith("_std")]
    profiles = profiles.drop(columns=std_cols)
    
    # Reorder columns
    base_cols = ["cluster_id", "cluster_label", "n_units"]
    mean_cols = [c for c in profiles.columns if c.endswith("_mean")]
    z_cols = [c for c in profiles.columns if c.endswith("_z")]
    profiles = profiles[base_cols + mean_cols + z_cols]
    
    logger.info(f"Built profiles for {len(profiles)} clusters")
    
    return profiles


def build_cluster_labels_table(
    label_dict: Dict,
    level: str,
) -> pd.DataFrame:
    """
    Build human-readable cluster labels table.
    """
    rows = []
    for cluster_id, labels in label_dict.items():
        rows.append({
            "level": level,
            "cluster_id": cluster_id,
            "cluster_label_short": labels["short"],
            "cluster_label_long": labels["long"],
            "defining_features": labels["defining"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# Chart Generation
# =============================================================================

def create_cluster_bar_chart(
    profiles: pd.DataFrame,
    features: List[str],
    output_path: Path,
    title: str,
    logger,
):
    """
    Create bar chart showing z-scored feature means for each cluster.
    """
    logger.info(f"Creating bar chart: {output_path.name}")
    
    # Get z-score columns
    z_cols = [f"{f}_z" for f in features if f"{f}_z" in profiles.columns]
    
    if not z_cols:
        logger.warning("No z-score columns available for chart")
        return
    
    n_clusters = len(profiles)
    n_features = len(z_cols)
    
    # Set up figure
    fig, axes = plt.subplots(
        nrows=n_clusters, 
        ncols=1, 
        figsize=(12, 2.5 * n_clusters),
        squeeze=False
    )
    
    # Color palette
    colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, n_features))
    
    for idx, (_, row) in enumerate(profiles.iterrows()):
        ax = axes[idx, 0]
        
        cluster_id = int(row["cluster_id"])
        cluster_label = row["cluster_label"]
        n_units = int(row["n_units"])
        
        # Get z-scores
        z_values = [row[col] for col in z_cols]
        feature_names = [FEATURE_SHORT_NAMES.get(col.replace("_z", ""), col.replace("_z", "")) for col in z_cols]
        
        # Create bar chart
        bars = ax.barh(range(n_features), z_values, color=colors, edgecolor="black", linewidth=0.5)
        
        # Color bars by sign
        for bar, val in zip(bars, z_values):
            if val < 0:
                bar.set_color("#4575b4")  # Blue for negative
            else:
                bar.set_color("#d73027")  # Red for positive
        
        # Add reference lines
        ax.axvline(x=0, color="black", linewidth=1)
        ax.axvline(x=-1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axvline(x=1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        
        # Labels
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feature_names, fontsize=9)
        ax.set_xlabel("Z-score (std from mean)", fontsize=10)
        ax.set_title(f"Cluster {cluster_id}: {cluster_label} (n={n_units})", fontsize=11, fontweight="bold")
        ax.set_xlim(-3, 3)
        ax.invert_yaxis()
    
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    
    logger.info(f"Saved chart: {output_path}")


def create_single_cluster_chart(
    profiles: pd.DataFrame,
    cluster_id: int,
    features: List[str],
    output_path: Path,
    logger,
):
    """
    Create single bar chart for one cluster.
    """
    row = profiles[profiles["cluster_id"] == cluster_id].iloc[0]
    
    z_cols = [f"{f}_z" for f in features if f"{f}_z" in profiles.columns]
    z_values = [row[col] for col in z_cols]
    feature_names = [FEATURE_SHORT_NAMES.get(col.replace("_z", ""), col.replace("_z", "")) for col in z_cols]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    y_pos = range(len(z_values))
    bars = ax.barh(y_pos, z_values, edgecolor="black", linewidth=0.5)
    
    # Color bars by sign
    for bar, val in zip(bars, z_values):
        if val < 0:
            bar.set_color("#4575b4")
        else:
            bar.set_color("#d73027")
    
    # Reference lines
    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(x=-1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=1, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel("Z-score (std from mean)", fontsize=11)
    ax.set_title(
        f"Cluster {cluster_id}: {row['cluster_label']} (n={int(row['n_units'])})",
        fontsize=12,
        fontweight="bold"
    )
    ax.set_xlim(-3, 3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# =============================================================================
# CD Assets Builder
# =============================================================================

def build_cd_assets(logger, config: dict):
    """
    Build CD-level typology assets.
    """
    logger.info("=" * 70)
    logger.info("BUILDING CD TYPOLOGY ASSETS")
    logger.info("=" * 70)
    
    # Get stability threshold from config
    stability_config = config.get("stability", {})
    mostly_structural_threshold = stability_config.get("mostly_structural_entropy_threshold", 0.60)
    logger.info(f"Mostly structural threshold: {mostly_structural_threshold}")
    
    # Load inputs
    cd_clusters = pd.read_parquet(TYPOLOGY_DIR / "cd_clusters.parquet")
    cd_clusters = ensure_boro_cd_dtype(cd_clusters)
    logger.info(f"Loaded CD clusters: {len(cd_clusters)} rows")
    
    cd_stability = pd.read_parquet(REPORTS_DIR / "cd_stability.parquet")
    cd_stability = ensure_boro_cd_dtype(cd_stability)
    logger.info(f"Loaded CD stability: {len(cd_stability)} rows")
    
    cd59 = read_gdf(GEO_DIR / "cd59.parquet")
    cd59 = ensure_boro_cd_dtype(cd59)
    
    # =========================================================================
    # 1. Cluster Profile Table
    # =========================================================================
    logger.info("\n--- Cluster Profile Table ---")
    
    cd_profiles = build_cluster_profiles(cd_clusters, KEY_FEATURES, "boro_cd", logger)
    
    output_profiles = REPORTS_DIR / "cd_cluster_profiles.csv"
    cd_profiles.to_csv(output_profiles, index=False)
    logger.info(f"Wrote: {output_profiles}")
    
    # =========================================================================
    # 2. Cluster Labels Table
    # =========================================================================
    logger.info("\n--- Cluster Labels Table ---")
    
    cd_labels_df = build_cluster_labels_table(CD_CLUSTER_LABELS, "CD")
    
    output_labels = REPORTS_DIR / "cluster_labels_cd.csv"
    cd_labels_df.to_csv(output_labels, index=False)
    logger.info(f"Wrote: {output_labels}")
    
    # =========================================================================
    # 3. Presentation-ready Joined Table with Stability
    # =========================================================================
    logger.info("\n--- Joined Types + Stability ---")
    
    # Join clusters with stability
    cd_joined = cd_clusters[["boro_cd", "cd_label", "cd_short", "cluster_id", "cluster_label", "is_outlier_cluster"]].merge(
        cd_stability[[
            "boro_cd", "cluster_id_2021", "cluster_id_2022", "cluster_id_2023",
            "transition_count", "is_stable_all_3", "is_stable_2_of_3", 
            "entropy_score", "stability_class"
        ]],
        on="boro_cd",
        how="left",
    )
    
    # Add stability_class2 (config-driven threshold)
    cd_joined = add_stability_class2(cd_joined, mostly_structural_threshold)
    
    # Rename pooled cluster columns for clarity
    cd_joined = cd_joined.rename(columns={
        "cluster_id": "cluster_id_pooled",
        "cluster_label": "cluster_label_short",
    })
    
    # Add long label
    cd_joined["cluster_label_long"] = cd_joined["cluster_id_pooled"].map(
        lambda x: CD_CLUSTER_LABELS.get(x, {}).get("long", "")
    )
    
    # Write parquet/csv
    output_joined = REPORTS_DIR / "cd_types_with_stability.parquet"
    atomic_write_df(cd_joined, output_joined)
    logger.info(f"Wrote: {output_joined}")
    
    cd_joined.to_csv(REPORTS_DIR / "cd_types_with_stability.csv", index=False)
    logger.info(f"Wrote: {REPORTS_DIR / 'cd_types_with_stability.csv'}")
    
    # GeoJSON
    gdf_cd = cd59[["boro_cd", "geometry"]].merge(cd_joined, on="boro_cd", how="left")
    gdf_cd = ensure_boro_cd_dtype(gdf_cd)
    
    output_geojson = REPORTS_DIR / "cd_types_with_stability.geojson"
    atomic_write_gdf(gdf_cd, output_geojson)
    logger.info(f"Wrote: {output_geojson}")
    
    # =========================================================================
    # 4. Bar Charts
    # =========================================================================
    logger.info("\n--- Bar Charts ---")
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Combined chart
    create_cluster_bar_chart(
        cd_profiles,
        KEY_FEATURES,
        FIGURES_DIR / "cd_cluster_profiles_all.png",
        "CD Cluster Profiles (Z-scored Feature Means)",
        logger,
    )
    
    # Individual charts
    for cluster_id in cd_profiles["cluster_id"].unique():
        create_single_cluster_chart(
            cd_profiles,
            cluster_id,
            KEY_FEATURES,
            FIGURES_DIR / f"cd_cluster_{cluster_id}_profile.png",
            logger,
        )
    
    # =========================================================================
    # Log Summary
    # =========================================================================
    logger.info("\n--- CD Assets Summary ---")
    
    stability_counts = cd_joined["stability_class2"].value_counts()
    logger.info("Stability distribution (class2):")
    for cls, count in stability_counts.items():
        logger.info(f"  {cls}: {count}")
    
    cluster_counts = cd_joined["cluster_id_pooled"].value_counts().sort_index()
    logger.info("\nCluster distribution:")
    for cluster_id, count in cluster_counts.items():
        label = CD_CLUSTER_LABELS.get(cluster_id, {}).get("short", "Unknown")
        logger.info(f"  Cluster {cluster_id} ({label}): {count}")
    
    return {
        "profiles": cd_profiles,
        "labels": cd_labels_df,
        "joined": cd_joined,
    }


# =============================================================================
# NTA Assets Builder
# =============================================================================

def build_nta_assets(logger, config: dict):
    """
    Build NTA-level typology assets.
    """
    logger.info("=" * 70)
    logger.info("BUILDING NTA TYPOLOGY ASSETS")
    logger.info("=" * 70)
    
    # Get stability threshold from config
    stability_config = config.get("stability", {})
    mostly_structural_threshold = stability_config.get("mostly_structural_entropy_threshold", 0.60)
    logger.info(f"Mostly structural threshold: {mostly_structural_threshold}")
    
    # Load inputs
    nta_clusters = pd.read_parquet(TYPOLOGY_DIR / "nta_clusters_residential.parquet")
    logger.info(f"Loaded NTA clusters: {len(nta_clusters)} rows")
    
    nta_stability = pd.read_parquet(REPORTS_DIR / "nta_stability_residential.parquet")
    logger.info(f"Loaded NTA stability: {len(nta_stability)} rows")
    
    nta_geo = read_gdf(GEO_DIR / "nta.parquet")
    
    # Check existing labels
    nta_label_col = "cluster_label" if "cluster_label" in nta_clusters.columns else None
    
    # =========================================================================
    # 1. Build cluster profiles
    # =========================================================================
    logger.info("\n--- Cluster Profile Table ---")
    
    nta_profiles = build_cluster_profiles(nta_clusters, KEY_FEATURES, "ntacode", logger)
    
    # Apply predefined labels to profiles
    nta_profiles["cluster_label"] = nta_profiles["cluster_id"].map(
        lambda x: NTA_CLUSTER_LABELS.get(int(x), {}).get("short", f"Cluster_{x}")
    )
    
    output_profiles = REPORTS_DIR / "nta_cluster_profiles.csv"
    nta_profiles.to_csv(output_profiles, index=False)
    logger.info(f"Wrote: {output_profiles}")
    
    # =========================================================================
    # 2. Cluster Labels - use predefined labels
    # =========================================================================
    logger.info("\n--- Cluster Labels Table ---")
    
    # Use predefined NTA_CLUSTER_LABELS from module level
    nta_labels_df = build_cluster_labels_table(NTA_CLUSTER_LABELS, "NTA")
    
    output_labels = REPORTS_DIR / "cluster_labels_nta.csv"
    nta_labels_df.to_csv(output_labels, index=False)
    logger.info(f"Wrote: {output_labels}")
    
    # =========================================================================
    # 3. Presentation-ready Joined Table with Stability
    # =========================================================================
    logger.info("\n--- Joined Types + Stability ---")
    
    # Get relevant columns from clusters
    cluster_cols = ["ntacode", "nta_name", "cluster_id", "is_outlier_cluster"]
    if "cluster_label" in nta_clusters.columns:
        cluster_cols.append("cluster_label")
    
    nta_joined = nta_clusters[cluster_cols].merge(
        nta_stability[[
            "ntacode", "cluster_id_2021", "cluster_id_2022", "cluster_id_2023",
            "transition_count", "is_stable_all_3", "is_stable_2_of_3",
            "low_volume_years", "entropy_score", "stability_class"
        ]],
        on="ntacode",
        how="left",
    )
    
    # Add stability_class2 (config-driven threshold)
    nta_joined = add_stability_class2(nta_joined, mostly_structural_threshold)
    
    # Rename pooled cluster columns for clarity
    nta_joined = nta_joined.rename(columns={
        "cluster_id": "cluster_id_pooled",
    })
    
    # Use predefined labels from NTA_CLUSTER_LABELS
    nta_joined["cluster_label_short"] = nta_joined["cluster_id_pooled"].map(
        lambda x: NTA_CLUSTER_LABELS.get(x, {}).get("short", f"Cluster {x}")
    )
    
    # Add long label
    nta_joined["cluster_label_long"] = nta_joined["cluster_id_pooled"].map(
        lambda x: NTA_CLUSTER_LABELS.get(x, {}).get("long", "")
    )
    
    # Drop any existing cluster_label column to avoid confusion
    if "cluster_label" in nta_joined.columns:
        nta_joined = nta_joined.drop(columns=["cluster_label"])
    
    # Write parquet/csv
    output_joined = REPORTS_DIR / "nta_types_with_stability_residential.parquet"
    atomic_write_df(nta_joined, output_joined)
    logger.info(f"Wrote: {output_joined}")
    
    nta_joined.to_csv(REPORTS_DIR / "nta_types_with_stability_residential.csv", index=False)
    logger.info(f"Wrote: {REPORTS_DIR / 'nta_types_with_stability_residential.csv'}")
    
    # GeoJSON - filter to residential NTAs
    residential_ntas = nta_joined["ntacode"].tolist()
    nta_geo_res = nta_geo[nta_geo["ntacode"].isin(residential_ntas)][["ntacode", "geometry"]].copy()
    
    gdf_nta = nta_geo_res.merge(nta_joined, on="ntacode", how="left")
    
    output_geojson = REPORTS_DIR / "nta_types_with_stability_residential.geojson"
    atomic_write_gdf(gdf_nta, output_geojson)
    logger.info(f"Wrote: {output_geojson}")
    
    # =========================================================================
    # 4. Bar Charts
    # =========================================================================
    logger.info("\n--- Bar Charts ---")
    
    # Combined chart
    create_cluster_bar_chart(
        nta_profiles,
        KEY_FEATURES,
        FIGURES_DIR / "nta_cluster_profiles_all.png",
        "NTA Cluster Profiles (Z-scored Feature Means)",
        logger,
    )
    
    # Individual charts
    for cluster_id in nta_profiles["cluster_id"].unique():
        create_single_cluster_chart(
            nta_profiles,
            cluster_id,
            KEY_FEATURES,
            FIGURES_DIR / f"nta_cluster_{cluster_id}_profile.png",
            logger,
        )
    
    # =========================================================================
    # Log Summary
    # =========================================================================
    logger.info("\n--- NTA Assets Summary ---")
    
    stability_counts = nta_joined["stability_class2"].value_counts()
    logger.info("Stability distribution (class2):")
    for cls, count in stability_counts.items():
        logger.info(f"  {cls}: {count}")
    
    cluster_counts = nta_joined["cluster_id_pooled"].value_counts().sort_index()
    logger.info("\nCluster distribution:")
    for cluster_id in sorted(cluster_counts.index):
        count = cluster_counts[cluster_id]
        label = NTA_CLUSTER_LABELS.get(int(cluster_id), {}).get("short", "Unknown")
        logger.info(f"  Cluster {cluster_id} ({label}): {count}")
    
    return {
        "profiles": nta_profiles,
        "labels": nta_labels_df,
        "joined": nta_joined,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build typology assets")
    parser.add_argument("--cd", action="store_true", help="Build CD assets only")
    parser.add_argument("--nta", action="store_true", help="Build NTA assets only")
    args = parser.parse_args()
    
    # Default to both if neither specified
    do_cd = args.cd or (not args.cd and not args.nta)
    do_nta = args.nta or (not args.cd and not args.nta)
    
    with get_logger("08b_build_typology_assets") as logger:
        logger.info("Starting 08b_build_typology_assets.py")
        
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        try:
            # Ensure output directories
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            
            outputs = {}
            
            # Build CD assets
            if do_cd:
                cd_outputs = build_cd_assets(logger, config)
                outputs["cd"] = cd_outputs
            
            # Build NTA assets
            if do_nta:
                nta_outputs = build_nta_assets(logger, config)
                outputs["nta"] = nta_outputs
            
            # Log outputs
            output_files = {}
            if do_cd:
                output_files.update({
                    "cd_profiles": str(REPORTS_DIR / "cd_cluster_profiles.csv"),
                    "cd_labels": str(REPORTS_DIR / "cluster_labels_cd.csv"),
                    "cd_types_with_stability": str(REPORTS_DIR / "cd_types_with_stability.geojson"),
                    "cd_charts": str(FIGURES_DIR / "cd_cluster_profiles_all.png"),
                })
            if do_nta:
                output_files.update({
                    "nta_profiles": str(REPORTS_DIR / "nta_cluster_profiles.csv"),
                    "nta_labels": str(REPORTS_DIR / "cluster_labels_nta.csv"),
                    "nta_types_with_stability": str(REPORTS_DIR / "nta_types_with_stability_residential.geojson"),
                    "nta_charts": str(FIGURES_DIR / "nta_cluster_profiles_all.png"),
                })
            
            logger.log_outputs(output_files)
            
            # Write metadata
            write_metadata_sidecar(
                output_path=REPORTS_DIR / "cd_cluster_profiles.csv",
                inputs={
                    "cd_clusters": str(TYPOLOGY_DIR / "cd_clusters.parquet"),
                    "cd_stability": str(REPORTS_DIR / "cd_stability.parquet"),
                },
                config=config,
                run_id=logger.run_id,
                extra={
                    "cd_assets_built": do_cd,
                    "nta_assets_built": do_nta,
                },
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("SUCCESS: Built typology assets")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

