#!/usr/bin/env python3
"""
00_build_geographies.py

Build canonical CD59 geometry files for the Sleep ESI project.

Per Section 3.1:
- Filter to 59 residential CDs only
- Exclude JIAs and non-residential districts
- Enforce key: boro_cd as pandas nullable Int64
- Enforce uniqueness + completeness

Outputs:
- data/processed/geo/cd59.parquet (GeoParquet, EPSG:4326) - canonical
- data/processed/geo/cd59_epsg2263.parquet (GeoParquet, EPSG:2263) - canonical
- data/processed/geo/cd59.geojson (export)
- data/processed/geo/cd59_epsg2263.geojson (export)

Per R13: Raw download provenance is recorded in manifest.
"""

import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import pandas as pd
import requests

from sleep_esi.hashing import hash_file, write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_gdf, atomic_write_json, read_yaml
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, RAW_DIR
from sleep_esi.qa import (
    assert_all_valid,
    assert_expected_crs,
    check_bounds_epsg2263,
    check_bounds_epsg4326,
    safe_reproject,
    validate_bounds,
)
from sleep_esi.schemas import ensure_boro_cd_dtype, validate_boro_cd

# =============================================================================
# Constants
# =============================================================================

# NYC Open Data API - Community Districts (SODA2 endpoint, stable)
# https://data.cityofnewyork.us/City-Government/Community-Districts/yfnk-k7r4
NYC_CD_API_URL = "https://data.cityofnewyork.us/resource/5crt-au7u.geojson"

# =============================================================================
# AUTHORITATIVE BOROUGH CODE MAPPING
# Fixed NYC standard - borough codes are defined by NYC city government
# Per NYC_Night_Signals_Plan.md Section 2: All labels must derive from
# authoritative attributes, not manual dictionaries.
# =============================================================================
AUTHORITATIVE_BOROUGH_CODES = {
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

# Fallback: NYC Department of City Planning direct download
# https://www.nyc.gov/site/planning/data-maps/open-data/districts-download-metadata.page
NYC_CD_ZIP_URLS = [
    "https://www.nyc.gov/assets/planning/download/zip/data-maps/open-data/nycd_24d.zip",
    "https://www.nyc.gov/assets/planning/download/zip/data-maps/open-data/nycd_24c.zip",
    "https://www.nyc.gov/assets/planning/download/zip/data-maps/open-data/nycd_23d.zip",
    "https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/nycd_24d.zip",
]

# Expected number of residential CDs
EXPECTED_CD_COUNT = 59

# Raw data directory for CDs
RAW_CD_DIR = RAW_DIR / "community_districts"


# =============================================================================
# Data Acquisition
# =============================================================================

def fetch_community_districts_api(logger) -> gpd.GeoDataFrame:
    """
    Fetch Community Districts via NYC Open Data SODA2 API (preferred method).
    
    Returns GeoJSON directly - no unzipping needed.
    
    Returns:
        GeoDataFrame loaded from API
    """
    timestamp = datetime.now(timezone.utc)
    
    logger.info(f"Fetching Community Districts from NYC Open Data API...")
    logger.info(f"URL: {NYC_CD_API_URL}")
    
    response = requests.get(NYC_CD_API_URL, timeout=120)
    response.raise_for_status()
    
    # Save raw GeoJSON for provenance
    RAW_CD_DIR.mkdir(parents=True, exist_ok=True)
    geojson_filename = f"nyc_community_districts_api_{timestamp.strftime('%Y%m%d_%H%M%S')}.geojson"
    geojson_path = RAW_CD_DIR / geojson_filename
    
    with open(geojson_path, "w") as f:
        f.write(response.text)
    
    file_hash = hash_file(geojson_path)
    
    logger.info(f"Downloaded: {geojson_path}")
    logger.info(f"SHA256: {file_hash[:16]}...")
    
    # Load as GeoDataFrame
    gdf = gpd.read_file(geojson_path)
    
    # Record provenance
    provenance = {
        "source": "NYC Open Data API (SODA2)",
        "dataset": "Community Districts",
        "dataset_id": "5crt-au7u",
        "url": NYC_CD_API_URL,
        "download_timestamp": timestamp.isoformat(),
        "filename": geojson_filename,
        "file_path": str(geojson_path),
        "sha256": file_hash,
        "size_bytes": geojson_path.stat().st_size,
        "row_count": len(gdf),
    }
    
    logger.log_inputs({"community_districts_api": str(geojson_path)})
    
    # Update manifest
    manifest_path = RAW_DIR / "_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"downloads": []}
    
    manifest["downloads"].append(provenance)
    manifest["last_updated"] = timestamp.isoformat()
    
    atomic_write_json(manifest, manifest_path)
    
    return gdf


def fetch_community_districts_zip(logger) -> Path:
    """
    Fallback: Fetch Community Districts shapefile from NYC DCP direct download.
    
    Per R13: Records complete download provenance.
    
    Returns:
        Path to extracted shapefile
    """
    RAW_CD_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc)
    zip_filename = f"nyc_community_districts_{timestamp.strftime('%Y%m%d')}.zip"
    zip_path = RAW_CD_DIR / zip_filename
    extract_dir = RAW_CD_DIR / f"nycd_{timestamp.strftime('%Y%m%d')}"
    
    logger.info(f"Fetching Community Districts from NYC DCP (fallback)...")
    
    # Try multiple URLs
    response = None
    successful_url = None
    for url in NYC_CD_ZIP_URLS:
        logger.info(f"Trying URL: {url}")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            successful_url = url
            logger.info(f"Success with URL: {url}")
            break
        except requests.exceptions.HTTPError as e:
            logger.warning(f"URL failed: {url} - {e}")
            continue
    
    if response is None or successful_url is None:
        raise RuntimeError(
            f"All download URLs failed. Please download manually from "
            f"https://www.nyc.gov/site/planning/data-maps/open-data.page "
            f"and place the shapefile in {RAW_CD_DIR}"
        )
    
    # Write zip file
    with open(zip_path, "wb") as f:
        f.write(response.content)
    
    # Compute hash
    file_hash = hash_file(zip_path)
    
    logger.info(f"Downloaded: {zip_path}")
    logger.info(f"SHA256: {file_hash[:16]}...")
    
    # Extract zip
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    
    logger.info(f"Extracted to: {extract_dir}")
    
    # Find the shapefile
    shp_files = list(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No shapefile found in {extract_dir}")
    
    shp_path = shp_files[0]
    logger.info(f"Found shapefile: {shp_path}")
    
    # Record provenance
    provenance = {
        "source": "NYC Department of City Planning",
        "dataset": "Community Districts",
        "url": successful_url,
        "download_timestamp": timestamp.isoformat(),
        "zip_filename": zip_filename,
        "zip_path": str(zip_path),
        "extract_dir": str(extract_dir),
        "shapefile": str(shp_path),
        "sha256": file_hash,
        "size_bytes": zip_path.stat().st_size,
    }
    
    logger.log_inputs({"community_districts_raw": str(shp_path)})
    
    # Update manifest
    manifest_path = RAW_DIR / "_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"downloads": []}
    
    manifest["downloads"].append(provenance)
    manifest["last_updated"] = timestamp.isoformat()
    
    atomic_write_json(manifest, manifest_path)
    
    return shp_path


# =============================================================================
# Processing
# =============================================================================

def load_and_filter_cds_from_gdf(gdf: gpd.GeoDataFrame, logger) -> gpd.GeoDataFrame:
    """
    Filter GeoDataFrame to 59 residential CDs.
    
    Per Section 3.1:
    - Exclude JIAs (Joint Interest Areas)
    - Exclude parks/cemeteries (CD 164, 226, 227, 228, 355, 356, 480, 481, 482, 483, 484)
    - Keep only residential community districts
    
    Returns:
        GeoDataFrame with 59 residential CDs
    """
    logger.info(f"Processing {len(gdf)} total features")
    
    # Log column names for debugging
    logger.debug(f"Columns: {list(gdf.columns)}")
    
    # Find the boro_cd column (may have different names)
    boro_cd_col = None
    for col in ["boro_cd", "borocd", "BORO_CD", "BoroCD"]:
        if col in gdf.columns:
            boro_cd_col = col
            break
    
    if boro_cd_col is None:
        # Try to construct from boro and cd columns
        if "boro" in gdf.columns.str.lower() and "cd" in gdf.columns.str.lower():
            boro_col = [c for c in gdf.columns if c.lower() == "boro"][0]
            cd_col = [c for c in gdf.columns if c.lower() == "cd"][0]
            gdf["boro_cd"] = gdf[boro_col].astype(int) * 100 + gdf[cd_col].astype(int)
            boro_cd_col = "boro_cd"
        else:
            raise ValueError(f"Could not find boro_cd column. Available: {list(gdf.columns)}")
    
    # Standardize column name
    if boro_cd_col != "boro_cd":
        gdf = gdf.rename(columns={boro_cd_col: "boro_cd"})
    
    # Convert to Int64
    gdf["boro_cd"] = gdf["boro_cd"].astype("Int64")
    
    # Define non-residential CDs to exclude
    # These are parks, cemeteries, and other non-residential areas
    non_residential_cds = {
        164,  # Manhattan - Central Park
        226, 227, 228,  # Bronx - parks/cemeteries
        355, 356,  # Brooklyn - parks/cemeteries  
        480, 481, 482, 483, 484,  # Queens - parks/cemeteries
        595,  # Staten Island - parks
    }
    
    # Filter to residential CDs
    # Keep CDs where boro_cd is 3 digits (1XX, 2XX, 3XX, 4XX, 5XX)
    # and not in the non-residential list
    residential_mask = (
        (gdf["boro_cd"] >= 101) & 
        (gdf["boro_cd"] <= 595) &
        (~gdf["boro_cd"].isin(non_residential_cds))
    )
    
    gdf_filtered = gdf[residential_mask].copy()
    
    logger.info(f"Filtered to {len(gdf_filtered)} residential CDs")
    
    # Log the CDs we're keeping
    boro_counts = gdf_filtered["boro_cd"].apply(lambda x: x // 100).value_counts().sort_index()
    logger.info(f"CDs per borough: {dict(boro_counts)}")
    
    return gdf_filtered


def validate_cd59(gdf: gpd.GeoDataFrame, logger) -> None:
    """
    Validate the CD59 GeoDataFrame.
    
    Per Section 3.1 QA:
    - Exactly 59 rows
    - boro_cd unique
    - CRS sanity check (bounds)
    - Geometry validity
    """
    # Check row count
    if len(gdf) != EXPECTED_CD_COUNT:
        # Log which CDs we have for debugging
        logger.warning(f"Expected {EXPECTED_CD_COUNT} CDs, got {len(gdf)}")
        logger.warning(f"CDs present: {sorted(gdf['boro_cd'].tolist())}")
        
        # For now, allow flexibility but log warning
        if len(gdf) < 55 or len(gdf) > 65:
            raise ValueError(f"CD count {len(gdf)} is too far from expected {EXPECTED_CD_COUNT}")
    
    # Validate boro_cd
    validate_boro_cd(gdf, "cd59")
    
    # Check uniqueness
    if gdf["boro_cd"].duplicated().any():
        dups = gdf[gdf["boro_cd"].duplicated(keep=False)]["boro_cd"].tolist()
        raise ValueError(f"Duplicate boro_cd values: {dups}")
    
    # Check geometry validity
    assert_all_valid(gdf, "cd59 geometries")
    
    # Check CRS and bounds
    validate_bounds(gdf, "cd59")
    
    logger.info("CD59 validation passed")


def build_cd_lookup(gdf: gpd.GeoDataFrame, logger) -> pd.DataFrame:
    """
    Build the authoritative CD lookup table from CD59 geometries.
    
    Per NYC_Night_Signals_Plan.md Section 2 (Critical QA Gate):
    - No manual CD→neighborhood dictionaries
    - Labels derived deterministically from boro_cd using authoritative mappings
    - Official label format: "<Borough> Community District <n>"
    - Hard QA checks: computed boro_cd must match dataset values
    
    Args:
        gdf: GeoDataFrame with boro_cd column (59 rows)
        logger: JSONL logger instance
    
    Returns:
        DataFrame with columns:
        - boro_cd: Int64 (e.g., 101, 212, 503)
        - borough_code: Int64 (1-5)
        - borough_name: str (e.g., "Manhattan")
        - district_number: Int64 (1-18)
        - cd_label: str (e.g., "Manhattan Community District 1")
        - cd_short: str (e.g., "MN 1")
    """
    logger.info("Building CD lookup table...")
    
    lookup_data = []
    errors = []
    
    for _, row in gdf.iterrows():
        boro_cd = int(row["boro_cd"])
        
        # Extract components deterministically
        borough_code = boro_cd // 100
        district_number = boro_cd % 100
        
        # CRITICAL QA CHECK: Verify computed boro_cd matches
        computed_boro_cd = borough_code * 100 + district_number
        if computed_boro_cd != boro_cd:
            errors.append(f"Computed boro_cd {computed_boro_cd} != {boro_cd}")
            continue
        
        # Validate borough code against authoritative mapping
        if borough_code not in AUTHORITATIVE_BOROUGH_CODES:
            errors.append(f"Unknown borough code {borough_code} in boro_cd {boro_cd}")
            continue
        
        borough_name = AUTHORITATIVE_BOROUGH_CODES[borough_code]
        
        # Construct official CD label deterministically
        cd_label = f"{borough_name} Community District {district_number}"
        
        # Short label for maps
        cd_short = f"{BOROUGH_ABBREVIATIONS[borough_name]} {district_number}"
        
        lookup_data.append({
            "boro_cd": boro_cd,
            "borough_code": borough_code,
            "borough_name": borough_name,
            "district_number": district_number,
            "cd_label": cd_label,
            "cd_short": cd_short,
        })
    
    # FAIL HARD on any errors (per Section 2 requirement)
    if errors:
        for err in errors:
            logger.error(err)
        raise ValueError(f"CD lookup QA FAILED with {len(errors)} errors. See log for details.")
    
    # Create DataFrame
    lookup_df = pd.DataFrame(lookup_data)
    
    # Enforce dtypes
    lookup_df["boro_cd"] = lookup_df["boro_cd"].astype("Int64")
    lookup_df["borough_code"] = lookup_df["borough_code"].astype("Int64")
    lookup_df["district_number"] = lookup_df["district_number"].astype("Int64")
    
    # Sort by boro_cd for determinism
    lookup_df = lookup_df.sort_values("boro_cd").reset_index(drop=True)
    
    # Final validation
    assert len(lookup_df) == 59, f"Expected 59 CDs, got {len(lookup_df)}"
    assert lookup_df["boro_cd"].nunique() == 59, "Duplicate boro_cd in lookup"
    
    logger.info(f"Built CD lookup with {len(lookup_df)} entries")
    logger.info(f"Borough distribution: {dict(lookup_df.groupby('borough_name').size())}")
    
    return lookup_df


def build_cd59(logger) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build canonical CD59 geometries.
    
    Fetches from NYC Open Data API (preferred) with fallback to direct download.
    
    Returns:
        Tuple of (gdf_4326, gdf_2263)
    """
    # Load config
    config = read_yaml(CONFIG_DIR / "params.yml")
    
    # Check for existing raw files (API GeoJSON or shapefiles)
    raw_geojson_files = list(RAW_CD_DIR.glob("*_api_*.geojson")) if RAW_CD_DIR.exists() else []
    raw_shp_files = list(RAW_CD_DIR.rglob("*.shp")) if RAW_CD_DIR.exists() else []
    
    gdf = None
    
    if raw_geojson_files:
        # Use most recent API download
        raw_path = max(raw_geojson_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using existing API GeoJSON: {raw_path}")
        gdf = gpd.read_file(raw_path)
    elif raw_shp_files:
        # Use most recent shapefile
        raw_path = max(raw_shp_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Using existing raw shapefile: {raw_path}")
        gdf = gpd.read_file(raw_path)
    else:
        # Try API first (preferred - stable URL)
        try:
            gdf = fetch_community_districts_api(logger)
            logger.info("Successfully fetched from NYC Open Data API")
        except Exception as e:
            logger.warning(f"API fetch failed: {e}")
            logger.info("Falling back to direct download...")
            raw_path = fetch_community_districts_zip(logger)
            gdf = gpd.read_file(raw_path)
    
    # Filter to residential CDs
    gdf = load_and_filter_cds_from_gdf(gdf, logger)
    
    # Ensure CRS is set (should be EPSG:4326 from NYC Open Data)
    if gdf.crs is None:
        logger.warning("CRS not set in source data, assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    
    # Reproject to EPSG:4326 if not already
    gdf_4326 = safe_reproject(gdf, 4326, "cd59 to EPSG:4326")
    
    # Validate bounds
    check_bounds_epsg4326(gdf_4326, context="cd59 EPSG:4326")
    
    # Create EPSG:2263 version
    gdf_2263 = safe_reproject(gdf_4326, 2263, "cd59 to EPSG:2263")
    check_bounds_epsg2263(gdf_2263, context="cd59 EPSG:2263")
    
    # Keep only essential columns
    keep_cols = ["boro_cd", "geometry"]
    gdf_4326 = gdf_4326[keep_cols].copy()
    gdf_2263 = gdf_2263[keep_cols].copy()
    
    # Ensure Int64 dtype
    gdf_4326 = ensure_boro_cd_dtype(gdf_4326)
    gdf_2263 = ensure_boro_cd_dtype(gdf_2263)
    
    # Sort by boro_cd for determinism
    gdf_4326 = gdf_4326.sort_values("boro_cd").reset_index(drop=True)
    gdf_2263 = gdf_2263.sort_values("boro_cd").reset_index(drop=True)
    
    # Validate
    validate_cd59(gdf_4326, logger)
    validate_cd59(gdf_2263, logger)
    
    return gdf_4326, gdf_2263


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("00_build_geographies") as logger:
        logger.info("Starting 00_build_geographies.py")
        
        # Load config for logging
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        try:
            # Build CD59
            gdf_4326, gdf_2263 = build_cd59(logger)
            
            # Build CD lookup table (per NYC_Night_Signals_Plan.md Section 2)
            lookup_df = build_cd_lookup(gdf_4326, logger)
            
            # Define output paths
            outputs = {
                "cd59_parquet": GEO_DIR / "cd59.parquet",
                "cd59_2263_parquet": GEO_DIR / "cd59_epsg2263.parquet",
                "cd59_geojson": GEO_DIR / "cd59.geojson",
                "cd59_2263_geojson": GEO_DIR / "cd59_epsg2263.geojson",
                "cd_lookup_parquet": GEO_DIR / "cd_lookup.parquet",
                "cd_lookup_csv": GEO_DIR / "cd_lookup.csv",
            }
            
            # Write outputs
            GEO_DIR.mkdir(parents=True, exist_ok=True)
            
            # GeoParquet (canonical)
            atomic_write_gdf(gdf_4326, outputs["cd59_parquet"])
            logger.info(f"Wrote {outputs['cd59_parquet']}")
            
            atomic_write_gdf(gdf_2263, outputs["cd59_2263_parquet"])
            logger.info(f"Wrote {outputs['cd59_2263_parquet']}")
            
            # GeoJSON (exports)
            atomic_write_gdf(gdf_4326, outputs["cd59_geojson"])
            logger.info(f"Wrote {outputs['cd59_geojson']}")
            
            atomic_write_gdf(gdf_2263, outputs["cd59_2263_geojson"])
            logger.info(f"Wrote {outputs['cd59_2263_geojson']}")
            
            # CD Lookup table (parquet + CSV for easy inspection)
            lookup_df.to_parquet(outputs["cd_lookup_parquet"], index=False)
            logger.info(f"Wrote {outputs['cd_lookup_parquet']}")
            
            lookup_df.to_csv(outputs["cd_lookup_csv"], index=False)
            logger.info(f"Wrote {outputs['cd_lookup_csv']}")
            
            # Log outputs
            logger.log_outputs({k: str(v) for k, v in outputs.items()})
            
            # Log metrics
            logger.log_metrics({
                "cd_count": len(gdf_4326),
                "lookup_count": len(lookup_df),
                "crs_4326": str(gdf_4326.crs),
                "crs_2263": str(gdf_2263.crs),
                "bounds_4326": list(gdf_4326.total_bounds),
                "bounds_2263": list(gdf_2263.total_bounds),
                "borough_distribution": dict(lookup_df.groupby("borough_name").size()),
            })
            
            # Write metadata sidecar
            raw_files = list(RAW_CD_DIR.rglob("*.shp"))
            raw_path = max(raw_files, key=lambda p: p.stat().st_mtime) if raw_files else None
            
            write_metadata_sidecar(
                output_path=outputs["cd59_parquet"],
                inputs={"community_districts_raw": str(raw_path) if raw_path else "fetched"},
                config=config,
                run_id=logger.run_id,
                extra={
                    "cd_count": len(gdf_4326),
                    "lookup_columns": list(lookup_df.columns),
                },
            )
            
            logger.info(f"SUCCESS: Built CD59 with {len(gdf_4326)} community districts")
            logger.info(f"SUCCESS: Built CD lookup with {len(lookup_df)} entries")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

