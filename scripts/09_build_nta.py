#!/usr/bin/env python3
"""
09_build_nta.py

Build canonical NTA (Neighborhood Tabulation Area) geometry files.

Per NYC_Night_Signals_Plan.md Section 3.3 (Script 09):
- Use authoritative NYC Planning NTA 2020 boundary dataset
- Produce geometry files with stable codes + official names
- Enables neighborhood-scale analysis with interpretable labels

Data Source:
NTA 2020 boundaries from NYC Department of City Planning.
These are Census-aligned tabulation areas with official neighborhood names.

Outputs:
- data/processed/geo/nta.parquet (GeoParquet, EPSG:4326)
- data/processed/geo/nta_epsg2263.parquet (GeoParquet, EPSG:2263)
- data/processed/geo/nta_lookup.parquet (ntacode, nta_name, borough)
- data/processed/geo/nta_lookup.csv (human-readable)

NTA Types (ntatype field):
- 0: Residential/mixed neighborhoods (primary analysis targets)
- 5-9: Parks, cemeteries, airports, other special areas

Requirements:
- Unique ntacode
- Non-null nta_name
- Geometry validity
- CRS sanity + bounds check
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import pandas as pd

from sleep_esi.hashing import hash_file, write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_gdf, atomic_write_df, read_yaml
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, RAW_DIR
from sleep_esi.qa import (
    assert_all_valid,
    check_bounds_epsg2263,
    check_bounds_epsg4326,
    safe_reproject,
    validate_bounds,
)


# =============================================================================
# Constants
# =============================================================================

RAW_NTA_DIR = RAW_DIR / "ntas_2020"

# NTA type definitions (from NYC Planning documentation)
NTA_TYPES = {
    0: "Residential/Mixed",
    5: "Rikers Island",
    6: "Industrial/Transportation",
    7: "Airport",
    8: "Cemetery",
    9: "Park/Recreation",
}

# Residential types to include for primary analysis
RESIDENTIAL_NTA_TYPES = [0]

# Borough code mapping (same as CDs)
BOROUGH_CODES = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island",
}

BOROUGH_ABBREVIATIONS = {
    "Manhattan": "MN",
    "Bronx": "BX",
    "Brooklyn": "BK",
    "Queens": "QN",
    "Staten Island": "SI",
}


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_nta(logger) -> Tuple[gpd.GeoDataFrame, Path]:
    """
    Load the most recent raw NTA 2020 data.
    
    Returns:
        Tuple of (GeoDataFrame, source_path)
    """
    if not RAW_NTA_DIR.exists():
        raise FileNotFoundError(
            f"NTA raw directory not found: {RAW_NTA_DIR}. "
            "Download NTA 2020 boundaries from NYC Planning."
        )
    
    # Find most recent GeoJSON file
    geojson_files = list(RAW_NTA_DIR.glob("*.geojson"))
    
    if not geojson_files:
        raise FileNotFoundError(
            f"No GeoJSON files found in {RAW_NTA_DIR}. "
            "Download NTA 2020 boundaries from NYC Planning."
        )
    
    raw_path = max(geojson_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading raw NTA data from: {raw_path}")
    
    gdf = gpd.read_file(raw_path)
    logger.info(f"Loaded {len(gdf)} NTA features")
    
    return gdf, raw_path


# =============================================================================
# Processing
# =============================================================================

def process_nta(gdf: gpd.GeoDataFrame, logger) -> gpd.GeoDataFrame:
    """
    Process raw NTA data to canonical format.
    
    Args:
        gdf: Raw NTA GeoDataFrame
        logger: Logger instance
    
    Returns:
        Processed GeoDataFrame with standardized columns
    """
    logger.info("Processing NTA boundaries...")
    
    # Log original schema
    logger.info(f"Original columns: {list(gdf.columns)}")
    
    # Identify key columns (NTA 2020 schema)
    # nta2020 = NTA code
    # ntaname = Official NTA name
    # borocode/boroname = Borough
    # ntatype = Type (0=residential, 9=park, etc.)
    
    if "nta2020" not in gdf.columns:
        raise ValueError("Expected 'nta2020' column not found in NTA data")
    
    if "ntaname" not in gdf.columns:
        raise ValueError("Expected 'ntaname' column not found in NTA data")
    
    # Standardize column names
    gdf = gdf.rename(columns={
        "nta2020": "ntacode",
        "ntaname": "nta_name",
        "borocode": "borough_code",
        "boroname": "borough_name",
    })
    
    # Convert borough_code to int
    if "borough_code" in gdf.columns:
        gdf["borough_code"] = gdf["borough_code"].astype(int)
    
    # Convert ntatype to int
    if "ntatype" in gdf.columns:
        gdf["ntatype"] = gdf["ntatype"].astype(int)
    
    # Log NTA type distribution
    if "ntatype" in gdf.columns:
        type_counts = gdf["ntatype"].value_counts().sort_index()
        logger.info("NTA type distribution:")
        for nta_type, count in type_counts.items():
            type_label = NTA_TYPES.get(int(nta_type), "Unknown")
            logger.info(f"  Type {nta_type} ({type_label}): {count}")
    
    # Ensure CRS is set
    if gdf.crs is None:
        logger.warning("CRS not set in source data, assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    
    # Log borough distribution
    if "borough_name" in gdf.columns:
        boro_counts = gdf["borough_name"].value_counts()
        logger.info(f"NTAs per borough: {dict(boro_counts)}")
    
    logger.info(f"Processed {len(gdf)} NTAs")
    
    return gdf


def build_nta_lookup(gdf: gpd.GeoDataFrame, logger) -> pd.DataFrame:
    """
    Build the NTA lookup table.
    
    Args:
        gdf: Processed NTA GeoDataFrame
        logger: Logger instance
    
    Returns:
        DataFrame with ntacode, nta_name, borough, and type info
    """
    logger.info("Building NTA lookup table...")
    
    # Select columns for lookup
    lookup_cols = ["ntacode", "nta_name", "borough_code", "borough_name"]
    if "ntatype" in gdf.columns:
        lookup_cols.append("ntatype")
    if "ntaabbrev" in gdf.columns:
        lookup_cols.append("ntaabbrev")
    
    # Keep only columns that exist
    lookup_cols = [c for c in lookup_cols if c in gdf.columns]
    
    lookup_df = gdf[lookup_cols].copy()
    
    # Add NTA type label
    if "ntatype" in lookup_df.columns:
        lookup_df["ntatype_label"] = lookup_df["ntatype"].map(NTA_TYPES).fillna("Unknown")
    
    # Add is_residential flag
    if "ntatype" in lookup_df.columns:
        lookup_df["is_residential"] = lookup_df["ntatype"].isin(RESIDENTIAL_NTA_TYPES)
    else:
        lookup_df["is_residential"] = True  # Assume all are residential if no type info
    
    # Add short label for maps
    if "borough_name" in lookup_df.columns:
        lookup_df["nta_short"] = lookup_df["borough_name"].map(BOROUGH_ABBREVIATIONS).fillna("") + "-" + lookup_df["ntacode"]
    
    # Sort by ntacode for determinism
    lookup_df = lookup_df.sort_values("ntacode").reset_index(drop=True)
    
    # Validation
    if lookup_df["ntacode"].duplicated().any():
        dups = lookup_df[lookup_df["ntacode"].duplicated(keep=False)]["ntacode"].tolist()
        raise ValueError(f"Duplicate ntacode values: {dups}")
    
    if lookup_df["nta_name"].isna().any():
        null_codes = lookup_df[lookup_df["nta_name"].isna()]["ntacode"].tolist()
        raise ValueError(f"Null nta_name for ntacodes: {null_codes}")
    
    n_residential = lookup_df["is_residential"].sum()
    logger.info(f"Built NTA lookup with {len(lookup_df)} entries ({n_residential} residential)")
    
    return lookup_df


def validate_nta(gdf: gpd.GeoDataFrame, logger) -> None:
    """
    Validate NTA GeoDataFrame.
    
    Checks:
    - ntacode unique
    - nta_name not null
    - Geometry validity
    - CRS sanity + bounds check
    """
    logger.info("Validating NTA geometries...")
    
    # Check ntacode uniqueness
    if gdf["ntacode"].duplicated().any():
        dups = gdf[gdf["ntacode"].duplicated(keep=False)]["ntacode"].tolist()
        raise ValueError(f"Duplicate ntacode values: {dups}")
    
    # Check nta_name not null
    if gdf["nta_name"].isna().any():
        null_codes = gdf[gdf["nta_name"].isna()]["ntacode"].tolist()
        raise ValueError(f"Null nta_name for ntacodes: {null_codes}")
    
    # Check geometry validity
    assert_all_valid(gdf, "NTA geometries")
    
    # Check CRS and bounds
    validate_bounds(gdf, "NTA")
    
    logger.info("NTA validation passed")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("09_build_nta") as logger:
        logger.info("Starting 09_build_nta.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        try:
            # Load raw NTA data
            gdf_raw, raw_path = load_raw_nta(logger)
            
            # Process to canonical format
            gdf = process_nta(gdf_raw, logger)
            
            # Reproject to EPSG:4326 if not already
            gdf_4326 = safe_reproject(gdf, 4326, "NTA to EPSG:4326")
            check_bounds_epsg4326(gdf_4326, context="NTA EPSG:4326")
            
            # Create EPSG:2263 version
            gdf_2263 = safe_reproject(gdf_4326, 2263, "NTA to EPSG:2263")
            check_bounds_epsg2263(gdf_2263, context="NTA EPSG:2263")
            
            # Keep essential columns for geometry files
            keep_cols = ["ntacode", "nta_name", "borough_code", "borough_name", "geometry"]
            if "ntatype" in gdf_4326.columns:
                keep_cols.insert(-1, "ntatype")
            
            keep_cols = [c for c in keep_cols if c in gdf_4326.columns]
            
            gdf_4326 = gdf_4326[keep_cols].copy()
            gdf_2263 = gdf_2263[keep_cols].copy()
            
            # Sort by ntacode for determinism
            gdf_4326 = gdf_4326.sort_values("ntacode").reset_index(drop=True)
            gdf_2263 = gdf_2263.sort_values("ntacode").reset_index(drop=True)
            
            # Validate
            validate_nta(gdf_4326, logger)
            validate_nta(gdf_2263, logger)
            
            # Build lookup table
            lookup_df = build_nta_lookup(gdf, logger)
            
            # Define output paths
            outputs = {
                "nta_parquet": GEO_DIR / "nta.parquet",
                "nta_2263_parquet": GEO_DIR / "nta_epsg2263.parquet",
                "nta_lookup_parquet": GEO_DIR / "nta_lookup.parquet",
                "nta_lookup_csv": GEO_DIR / "nta_lookup.csv",
            }
            
            # Ensure output directory exists
            GEO_DIR.mkdir(parents=True, exist_ok=True)
            
            # Write outputs
            atomic_write_gdf(gdf_4326, outputs["nta_parquet"])
            logger.info(f"Wrote: {outputs['nta_parquet']} ({len(gdf_4326)} NTAs)")
            
            atomic_write_gdf(gdf_2263, outputs["nta_2263_parquet"])
            logger.info(f"Wrote: {outputs['nta_2263_parquet']} ({len(gdf_2263)} NTAs)")
            
            atomic_write_df(lookup_df, outputs["nta_lookup_parquet"])
            logger.info(f"Wrote: {outputs['nta_lookup_parquet']} ({len(lookup_df)} entries)")
            
            lookup_df.to_csv(outputs["nta_lookup_csv"], index=False)
            logger.info(f"Wrote: {outputs['nta_lookup_csv']}")
            
            # Log outputs
            logger.log_outputs({k: str(v) for k, v in outputs.items()})
            
            # Log metrics
            n_residential = lookup_df["is_residential"].sum() if "is_residential" in lookup_df.columns else len(lookup_df)
            
            logger.log_metrics({
                "total_ntas": len(gdf_4326),
                "residential_ntas": int(n_residential),
                "lookup_entries": len(lookup_df),
                "crs_4326": str(gdf_4326.crs),
                "crs_2263": str(gdf_2263.crs),
                "bounds_4326": list(gdf_4326.total_bounds),
                "bounds_2263": list(gdf_2263.total_bounds),
            })
            
            if "borough_name" in lookup_df.columns:
                logger.log_metrics({
                    "borough_distribution": dict(lookup_df.groupby("borough_name").size()),
                })
            
            # Write metadata sidecar
            write_metadata_sidecar(
                output_path=outputs["nta_parquet"],
                inputs={"nta_raw": str(raw_path)},
                config=config,
                run_id=logger.run_id,
                extra={
                    "total_ntas": len(gdf_4326),
                    "residential_ntas": int(n_residential),
                    "lookup_columns": list(lookup_df.columns),
                },
            )
            
            # Print summary
            logger.info("=" * 70)
            logger.info("NTA Build Summary:")
            logger.info(f"  Total NTAs: {len(gdf_4326)}")
            logger.info(f"  Residential NTAs: {n_residential}")
            logger.info(f"  Lookup entries: {len(lookup_df)}")
            logger.info("")
            if "borough_name" in lookup_df.columns:
                logger.info("NTAs per borough:")
                for boro, count in lookup_df.groupby("borough_name").size().items():
                    logger.info(f"  {boro}: {count}")
            logger.info("")
            if "ntatype_label" in lookup_df.columns:
                logger.info("NTAs by type:")
                for nta_type, count in lookup_df.groupby("ntatype_label").size().items():
                    logger.info(f"  {nta_type}: {count}")
            logger.info("=" * 70)
            
            logger.info(f"SUCCESS: Built NTA geographies with {len(gdf_4326)} NTAs")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

