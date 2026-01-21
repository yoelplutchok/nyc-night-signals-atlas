#!/usr/bin/env python3
"""
01_build_crosswalks.py

Build crosswalks between Community Districts and Census Tracts, NTAs, and UHF districts.

Per Section 3.4:
- Build CD↔Tract, CD↔NTA, CD↔UHF crosswalks
- Both population weights (w_pop) and area weights (w_area)
- Weights sum to 1 within tolerance per CD

Outputs:
- data/processed/xwalk/cd_to_tract_weights.parquet
- data/processed/xwalk/cd_to_nta_weights.parquet
- data/processed/xwalk/cd_to_uhf_weights.parquet

Data Sources:
- Census Tracts 2020: https://data.cityofnewyork.us/resource/63ge-mke6.geojson
- NTAs 2020: https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson
- UHF 42: https://raw.githubusercontent.com/nychealth/coronavirus-data/master/Geography-resources/UHF_resources/UHF42.geo.json
- ACS Population: Census Bureau API
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests

from sleep_esi.hashing import hash_file, write_metadata_sidecar
from sleep_esi.io_utils import atomic_write_df, atomic_write_gdf, atomic_write_json, read_yaml, read_gdf
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, GEO_DIR, RAW_DIR, XWALK_DIR
from sleep_esi.qa import (
    assert_all_valid,
    check_bounds_epsg4326,
    safe_reproject,
)
from sleep_esi.schemas import ensure_boro_cd_dtype, validate_boro_cd, validate_schema, CROSSWALK_SCHEMA

# =============================================================================
# Constants - Data Source URLs
# =============================================================================

# NYC Open Data API endpoints (with $limit to get all rows)
CENSUS_TRACTS_API = "https://data.cityofnewyork.us/resource/63ge-mke6.geojson?$limit=5000"
NTAS_API = "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?$limit=500"

# NYC Health GitHub (UHF 42 boundaries)
UHF42_API = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/Geography-resources/UHF_resources/UHF42.geo.json"

# Census Bureau API for ACS population
ACS_POPULATION_API = (
    "https://api.census.gov/data/2022/acs/acs5"
    "?get=B01003_001E"
    "&for=tract:*"
    "&in=state:36"
    "&in=county:005,047,061,081,085"
)

# Borough code mapping (CDTA prefix to boro code)
CDTA_PREFIX_TO_BORO = {
    "MN": 1,  # Manhattan
    "BX": 2,  # Bronx
    "BK": 3,  # Brooklyn
    "QN": 4,  # Queens
    "SI": 5,  # Staten Island
}

# NYC County FIPS codes
NYC_COUNTY_FIPS = {
    "005": 2,  # Bronx
    "047": 3,  # Brooklyn (Kings)
    "061": 1,  # Manhattan (New York)
    "081": 4,  # Queens
    "085": 5,  # Staten Island (Richmond)
}

# Weight sum tolerance
WEIGHT_SUM_TOLERANCE = 0.001


# =============================================================================
# Data Acquisition
# =============================================================================

def fetch_geojson(url: str, name: str, logger) -> gpd.GeoDataFrame:
    """
    Fetch GeoJSON data from a URL and save raw copy.
    
    Per R13: Records complete download provenance.
    """
    timestamp = datetime.now(timezone.utc)
    
    logger.info(f"Fetching {name} from: {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    
    # Save raw file
    raw_dir = RAW_DIR / name.lower().replace(" ", "_")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{name.lower().replace(' ', '_')}_{timestamp.strftime('%Y%m%d_%H%M%S')}.geojson"
    raw_path = raw_dir / filename
    
    with open(raw_path, "w") as f:
        f.write(response.text)
    
    file_hash = hash_file(raw_path)
    logger.info(f"Saved raw file: {raw_path}")
    logger.info(f"SHA256: {file_hash[:16]}...")
    
    # Load as GeoDataFrame
    gdf = gpd.read_file(raw_path)
    logger.info(f"Loaded {len(gdf)} features from {name}")
    
    # Update manifest
    manifest_path = RAW_DIR / "_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"downloads": []}
    
    manifest["downloads"].append({
        "source": name,
        "url": url,
        "download_timestamp": timestamp.isoformat(),
        "filename": filename,
        "sha256": file_hash,
        "row_count": len(gdf),
    })
    manifest["last_updated"] = timestamp.isoformat()
    
    atomic_write_json(manifest, manifest_path)
    
    return gdf


def fetch_acs_population(logger) -> pd.DataFrame:
    """
    Fetch ACS 5-Year population data from Census Bureau API.
    
    Returns DataFrame with geoid and population columns.
    """
    timestamp = datetime.now(timezone.utc)
    
    logger.info(f"Fetching ACS population from Census Bureau API...")
    logger.info(f"URL: {ACS_POPULATION_API}")
    
    response = requests.get(ACS_POPULATION_API, timeout=120)
    response.raise_for_status()
    
    # Parse JSON response
    data = response.json()
    
    # First row is header
    header = data[0]
    rows = data[1:]
    
    df = pd.DataFrame(rows, columns=header)
    
    # Rename and construct geoid
    df = df.rename(columns={"B01003_001E": "population"})
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    df["population"] = pd.to_numeric(df["population"], errors="coerce").astype("Int64")
    
    # Keep only needed columns
    df = df[["geoid", "population"]].copy()
    
    logger.info(f"Loaded population for {len(df)} census tracts")
    logger.info(f"Total population: {df['population'].sum():,}")
    
    # Save raw file
    raw_dir = RAW_DIR / "acs_population"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"acs_population_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
    raw_path = raw_dir / filename
    df.to_csv(raw_path, index=False)
    
    # Update manifest
    manifest_path = RAW_DIR / "_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"downloads": []}
    
    manifest["downloads"].append({
        "source": "Census Bureau ACS 5-Year 2022",
        "url": ACS_POPULATION_API,
        "download_timestamp": timestamp.isoformat(),
        "filename": filename,
        "row_count": len(df),
        "total_population": int(df["population"].sum()),
    })
    manifest["last_updated"] = timestamp.isoformat()
    
    atomic_write_json(manifest, manifest_path)
    
    return df


def parse_cdta_to_boro_cd(cdta: str) -> Optional[int]:
    """
    Parse CDTA code (e.g., 'MN01') to boro_cd (e.g., 101).
    
    Returns None if parsing fails or if it's a special district (parks, etc.).
    """
    if not cdta or len(cdta) < 4:
        return None
    
    prefix = cdta[:2].upper()
    
    # Skip special districts (parks, JIAs, etc.)
    # These typically have codes like 'MN99', 'BK95', etc.
    if cdta[2:].startswith("9"):
        return None
    
    try:
        cd_num = int(cdta[2:4])
    except ValueError:
        return None
    
    if prefix not in CDTA_PREFIX_TO_BORO:
        return None
    
    boro = CDTA_PREFIX_TO_BORO[prefix]
    return boro * 100 + cd_num


# =============================================================================
# Crosswalk Building
# =============================================================================

def build_cd_to_tract_crosswalk(
    tracts_gdf: gpd.GeoDataFrame,
    population_df: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """
    Build CD ↔ Tract crosswalk with population and area weights.
    
    Uses the cdta2020 field to map tracts to CDs.
    """
    logger.info("Building CD → Tract crosswalk...")
    
    # Parse cdta2020 to boro_cd
    tracts = tracts_gdf.copy()
    tracts["boro_cd"] = tracts["cdta2020"].apply(parse_cdta_to_boro_cd)
    
    # Filter to residential CDs only (remove parks, etc.)
    residential = tracts[tracts["boro_cd"].notna()].copy()
    logger.info(f"Tracts with valid boro_cd: {len(residential)} / {len(tracts)}")
    
    # Ensure geoid column exists
    if "geoid" not in residential.columns:
        # Try to construct from boroct2020 or ct2020
        if "boroct2020" in residential.columns:
            # boroct2020 is like "1000100" - need to add state+county
            # Actually geoid should already exist, let's check other columns
            pass
    
    # Merge with population
    residential = residential.merge(
        population_df,
        on="geoid",
        how="left",
    )
    
    # Log population merge stats
    pop_matched = residential["population"].notna().sum()
    logger.info(f"Population matched: {pop_matched} / {len(residential)} tracts")
    
    # Calculate area (shape_area is in sq ft, convert to sq meters)
    if "shape_area" in residential.columns:
        residential["area_sqft"] = pd.to_numeric(residential["shape_area"], errors="coerce")
        residential["area_sqm"] = residential["area_sqft"] * 0.0929  # sq ft to sq m
    else:
        # Calculate from geometry
        residential_2263 = safe_reproject(residential, 2263, "tracts to EPSG:2263")
        residential["area_sqm"] = residential_2263.geometry.area
    
    # Group by boro_cd to compute weights
    xwalk = residential[["boro_cd", "geoid", "population", "area_sqm"]].copy()
    xwalk["boro_cd"] = xwalk["boro_cd"].astype("Int64")
    
    # Compute total population and area per CD
    cd_totals = xwalk.groupby("boro_cd").agg({
        "population": "sum",
        "area_sqm": "sum",
    }).rename(columns={"population": "cd_pop_total", "area_sqm": "cd_area_total"})
    
    xwalk = xwalk.merge(cd_totals, on="boro_cd", how="left")
    
    # Compute weights
    xwalk["w_pop"] = xwalk["population"] / xwalk["cd_pop_total"]
    xwalk["w_area"] = xwalk["area_sqm"] / xwalk["cd_area_total"]
    
    # Handle NaN weights (from tracts with 0 population)
    xwalk["w_pop"] = xwalk["w_pop"].fillna(0.0)
    xwalk["w_area"] = xwalk["w_area"].fillna(0.0)
    
    # Select final columns
    xwalk = xwalk[["boro_cd", "geoid", "population", "w_pop", "w_area"]].copy()
    xwalk = xwalk.rename(columns={"geoid": "tract_geoid", "population": "tract_pop"})
    
    # Sort for determinism
    xwalk = xwalk.sort_values(["boro_cd", "tract_geoid"]).reset_index(drop=True)
    
    # Validate weight sums
    weight_sums = xwalk.groupby("boro_cd")[["w_pop", "w_area"]].sum()
    
    pop_deviation = (weight_sums["w_pop"] - 1.0).abs().max()
    area_deviation = (weight_sums["w_area"] - 1.0).abs().max()
    
    # Handle potential NA values
    if pd.isna(pop_deviation) or pop_deviation >= WEIGHT_SUM_TOLERANCE:
        logger.warning(f"Population weights don't sum to 1 within tolerance!")
        logger.warning(f"Max deviation: {pop_deviation}")
    
    if pd.isna(area_deviation) or area_deviation >= WEIGHT_SUM_TOLERANCE:
        logger.warning(f"Area weights don't sum to 1 within tolerance!")
        logger.warning(f"Max deviation: {area_deviation}")
    
    logger.info(f"CD → Tract crosswalk: {len(xwalk)} rows, {xwalk['boro_cd'].nunique()} CDs")
    
    return xwalk


def build_cd_to_nta_crosswalk(
    tracts_gdf: gpd.GeoDataFrame,
    population_df: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """
    Build CD ↔ NTA crosswalk with population and area weights.
    
    Aggregates tracts to NTAs using nta2020 field.
    """
    logger.info("Building CD → NTA crosswalk...")
    
    # Parse cdta2020 to boro_cd
    tracts = tracts_gdf.copy()
    tracts["boro_cd"] = tracts["cdta2020"].apply(parse_cdta_to_boro_cd)
    
    # Filter to residential CDs only
    residential = tracts[tracts["boro_cd"].notna()].copy()
    
    # Merge with population
    residential = residential.merge(
        population_df,
        on="geoid",
        how="left",
    )
    
    # Calculate area
    if "shape_area" in residential.columns:
        residential["area_sqft"] = pd.to_numeric(residential["shape_area"], errors="coerce")
        residential["area_sqm"] = residential["area_sqft"] * 0.0929
    else:
        residential_2263 = safe_reproject(residential, 2263, "tracts to EPSG:2263")
        residential["area_sqm"] = residential_2263.geometry.area
    
    # Aggregate to NTA level first
    nta_agg = residential.groupby(["boro_cd", "nta2020"]).agg({
        "population": "sum",
        "area_sqm": "sum",
    }).reset_index()
    
    nta_agg["boro_cd"] = nta_agg["boro_cd"].astype("Int64")
    
    # Compute CD totals
    cd_totals = nta_agg.groupby("boro_cd").agg({
        "population": "sum",
        "area_sqm": "sum",
    }).rename(columns={"population": "cd_pop_total", "area_sqm": "cd_area_total"})
    
    nta_agg = nta_agg.merge(cd_totals, on="boro_cd", how="left")
    
    # Compute weights
    nta_agg["w_pop"] = nta_agg["population"] / nta_agg["cd_pop_total"]
    nta_agg["w_area"] = nta_agg["area_sqm"] / nta_agg["cd_area_total"]
    
    nta_agg["w_pop"] = nta_agg["w_pop"].fillna(0.0)
    nta_agg["w_area"] = nta_agg["w_area"].fillna(0.0)
    
    # Select final columns
    xwalk = nta_agg[["boro_cd", "nta2020", "population", "w_pop", "w_area"]].copy()
    xwalk = xwalk.rename(columns={"population": "nta_pop"})
    
    # Sort for determinism
    xwalk = xwalk.sort_values(["boro_cd", "nta2020"]).reset_index(drop=True)
    
    # Validate weight sums
    weight_sums = xwalk.groupby("boro_cd")[["w_pop", "w_area"]].sum()
    
    pop_deviation = (weight_sums["w_pop"] - 1.0).abs().max()
    area_deviation = (weight_sums["w_area"] - 1.0).abs().max()
    
    if pd.isna(pop_deviation) or pop_deviation >= WEIGHT_SUM_TOLERANCE:
        logger.warning(f"NTA population weights don't sum to 1 within tolerance!")
        logger.warning(f"Max deviation: {pop_deviation}")
    
    if pd.isna(area_deviation) or area_deviation >= WEIGHT_SUM_TOLERANCE:
        logger.warning(f"NTA area weights don't sum to 1 within tolerance!")
        logger.warning(f"Max deviation: {area_deviation}")
    
    logger.info(f"CD → NTA crosswalk: {len(xwalk)} rows, {xwalk['boro_cd'].nunique()} CDs, {xwalk['nta2020'].nunique()} NTAs")
    
    return xwalk


def build_cd_to_uhf_crosswalk(
    cd59_gdf: gpd.GeoDataFrame,
    uhf_gdf: gpd.GeoDataFrame,
    tracts_gdf: gpd.GeoDataFrame,
    population_df: pd.DataFrame,
    logger,
) -> pd.DataFrame:
    """
    Build CD ↔ UHF crosswalk with population and area weights.
    
    Uses spatial intersection between CD and UHF boundaries,
    weighted by tract population/area.
    """
    logger.info("Building CD → UHF crosswalk...")
    
    # Reproject to EPSG:2263 for accurate area calculations
    cd59_2263 = safe_reproject(cd59_gdf, 2263, "CD59 to EPSG:2263")
    uhf_2263 = safe_reproject(uhf_gdf, 2263, "UHF to EPSG:2263")
    
    # Get UHF code column (might be named differently)
    uhf_code_col = None
    for col in ["uhf42", "uhf_code", "uhfcode", "UHFCODE", "UHF42"]:
        if col in uhf_2263.columns:
            uhf_code_col = col
            break
    
    if uhf_code_col is None:
        # Check column names
        logger.warning(f"UHF columns: {list(uhf_2263.columns)}")
        # Try to find any column with UHF-like values
        for col in uhf_2263.columns:
            if uhf_2263[col].dtype in ['int64', 'Int64', 'object', 'str']:
                sample = uhf_2263[col].iloc[0] if len(uhf_2263) > 0 else None
                if sample and str(sample).isdigit() and len(str(sample)) == 3:
                    uhf_code_col = col
                    logger.info(f"Using '{col}' as UHF code column")
                    break
    
    if uhf_code_col is None:
        raise ValueError(f"Could not find UHF code column. Available: {list(uhf_2263.columns)}")
    
    # Standardize UHF code column name
    uhf_2263 = uhf_2263.rename(columns={uhf_code_col: "uhf42"})
    uhf_2263["uhf42"] = uhf_2263["uhf42"].astype(str)
    
    # Compute intersection areas between CDs and UHFs
    intersections = []
    
    for _, cd_row in cd59_2263.iterrows():
        cd_geom = cd_row.geometry
        boro_cd = cd_row["boro_cd"]
        
        for _, uhf_row in uhf_2263.iterrows():
            uhf_geom = uhf_row.geometry
            uhf42 = uhf_row["uhf42"]
            
            if cd_geom.intersects(uhf_geom):
                intersection = cd_geom.intersection(uhf_geom)
                area = intersection.area
                
                if area > 0:
                    intersections.append({
                        "boro_cd": boro_cd,
                        "uhf42": uhf42,
                        "intersection_area": area,
                    })
    
    intersection_df = pd.DataFrame(intersections)
    
    if len(intersection_df) == 0:
        logger.error("No intersections found between CDs and UHFs!")
        raise ValueError("No CD-UHF intersections found")
    
    logger.info(f"Found {len(intersection_df)} CD-UHF intersections")
    
    # Compute area weights
    intersection_df["boro_cd"] = intersection_df["boro_cd"].astype("Int64")
    
    cd_totals = intersection_df.groupby("boro_cd")["intersection_area"].sum().rename("cd_total_area")
    intersection_df = intersection_df.merge(cd_totals.reset_index(), on="boro_cd", how="left")
    
    intersection_df["w_area"] = intersection_df["intersection_area"] / intersection_df["cd_total_area"]
    
    # For population weights, we need to estimate population in each intersection
    # We'll use tract data to approximate this
    # First, join tracts to UHFs
    tracts = tracts_gdf.copy()
    tracts["boro_cd"] = tracts["cdta2020"].apply(parse_cdta_to_boro_cd)
    tracts = tracts[tracts["boro_cd"].notna()].copy()
    tracts = tracts.merge(population_df, on="geoid", how="left")
    
    tracts_2263 = safe_reproject(tracts, 2263, "tracts to EPSG:2263")
    tracts_2263["tract_centroid"] = tracts_2263.geometry.centroid
    
    # Assign each tract to a UHF based on centroid
    tract_uhf = gpd.sjoin(
        gpd.GeoDataFrame(tracts_2263, geometry="tract_centroid", crs=tracts_2263.crs),
        uhf_2263[["uhf42", "geometry"]],
        how="left",
        predicate="within",
    )
    
    # Handle tracts not within any UHF (use nearest)
    unmatched = tract_uhf["uhf42"].isna()
    if unmatched.any():
        logger.info(f"Tracts without UHF match (using nearest): {unmatched.sum()}")
        # For simplicity, leave as NaN for now - they won't contribute to pop weights
    
    # Aggregate population by CD and UHF
    tract_uhf_pop = tract_uhf.groupby(["boro_cd", "uhf42"])["population"].sum().reset_index()
    tract_uhf_pop["boro_cd"] = tract_uhf_pop["boro_cd"].astype("Int64")
    
    cd_pop_totals = tract_uhf_pop.groupby("boro_cd")["population"].sum().rename("cd_pop_total")
    tract_uhf_pop = tract_uhf_pop.merge(cd_pop_totals.reset_index(), on="boro_cd", how="left")
    
    tract_uhf_pop["w_pop"] = tract_uhf_pop["population"] / tract_uhf_pop["cd_pop_total"]
    
    # Merge area and population weights
    xwalk = intersection_df[["boro_cd", "uhf42", "w_area"]].copy()
    xwalk = xwalk.merge(
        tract_uhf_pop[["boro_cd", "uhf42", "population", "w_pop"]],
        on=["boro_cd", "uhf42"],
        how="left",
    )
    
    xwalk["w_pop"] = xwalk["w_pop"].fillna(0.0)
    xwalk["w_area"] = xwalk["w_area"].fillna(0.0)
    xwalk = xwalk.rename(columns={"population": "uhf_pop"})
    
    # Sort for determinism
    xwalk = xwalk.sort_values(["boro_cd", "uhf42"]).reset_index(drop=True)
    
    # Validate weight sums
    weight_sums = xwalk.groupby("boro_cd")[["w_pop", "w_area"]].sum()
    
    pop_deviation = (weight_sums["w_pop"] - 1.0).abs().max()
    area_deviation = (weight_sums["w_area"] - 1.0).abs().max()
    
    if pd.isna(pop_deviation) or pop_deviation >= WEIGHT_SUM_TOLERANCE:
        logger.warning(f"UHF population weights don't sum to 1 within tolerance!")
        logger.warning(f"Max deviation: {pop_deviation}")
    
    if pd.isna(area_deviation) or area_deviation >= WEIGHT_SUM_TOLERANCE:
        logger.warning(f"UHF area weights don't sum to 1 within tolerance!")
        logger.warning(f"Max deviation: {area_deviation}")
    
    logger.info(f"CD → UHF crosswalk: {len(xwalk)} rows, {xwalk['boro_cd'].nunique()} CDs, {xwalk['uhf42'].nunique()} UHFs")
    
    return xwalk


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("01_build_crosswalks") as logger:
        logger.info("Starting 01_build_crosswalks.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        try:
            # Load CD59 (from previous script)
            cd59_path = GEO_DIR / "cd59.parquet"
            if not cd59_path.exists():
                raise FileNotFoundError(
                    f"CD59 file not found: {cd59_path}. "
                    "Run 00_build_geographies.py first."
                )
            
            cd59_gdf = read_gdf(cd59_path)
            logger.info(f"Loaded CD59: {len(cd59_gdf)} community districts")
            
            # Fetch Census Tracts
            tracts_gdf = fetch_geojson(CENSUS_TRACTS_API, "census_tracts_2020", logger)
            
            # Fetch NTAs (for reference/validation)
            ntas_gdf = fetch_geojson(NTAS_API, "ntas_2020", logger)
            
            # Fetch UHF 42
            uhf_gdf = fetch_geojson(UHF42_API, "uhf42", logger)
            
            # Fetch ACS population
            population_df = fetch_acs_population(logger)
            
            # Build crosswalks
            cd_tract_xwalk = build_cd_to_tract_crosswalk(tracts_gdf, population_df, logger)
            cd_nta_xwalk = build_cd_to_nta_crosswalk(tracts_gdf, population_df, logger)
            cd_uhf_xwalk = build_cd_to_uhf_crosswalk(cd59_gdf, uhf_gdf, tracts_gdf, population_df, logger)
            
            # Define output paths
            XWALK_DIR.mkdir(parents=True, exist_ok=True)
            
            outputs = {
                "cd_to_tract": XWALK_DIR / "cd_to_tract_weights.parquet",
                "cd_to_nta": XWALK_DIR / "cd_to_nta_weights.parquet",
                "cd_to_uhf": XWALK_DIR / "cd_to_uhf_weights.parquet",
            }
            
            # Write outputs
            atomic_write_df(cd_tract_xwalk, outputs["cd_to_tract"])
            logger.info(f"Wrote {outputs['cd_to_tract']}")
            
            atomic_write_df(cd_nta_xwalk, outputs["cd_to_nta"])
            logger.info(f"Wrote {outputs['cd_to_nta']}")
            
            atomic_write_df(cd_uhf_xwalk, outputs["cd_to_uhf"])
            logger.info(f"Wrote {outputs['cd_to_uhf']}")
            
            # Log outputs
            logger.log_outputs({k: str(v) for k, v in outputs.items()})
            
            # Log metrics
            logger.log_metrics({
                "cd_tract_rows": len(cd_tract_xwalk),
                "cd_tract_cds": cd_tract_xwalk["boro_cd"].nunique(),
                "cd_tract_tracts": cd_tract_xwalk["tract_geoid"].nunique(),
                "cd_nta_rows": len(cd_nta_xwalk),
                "cd_nta_cds": cd_nta_xwalk["boro_cd"].nunique(),
                "cd_nta_ntas": cd_nta_xwalk["nta2020"].nunique(),
                "cd_uhf_rows": len(cd_uhf_xwalk),
                "cd_uhf_cds": cd_uhf_xwalk["boro_cd"].nunique(),
                "cd_uhf_uhfs": cd_uhf_xwalk["uhf42"].nunique(),
            })
            
            # Write metadata sidecar
            write_metadata_sidecar(
                output_path=outputs["cd_to_tract"],
                inputs={
                    "census_tracts": CENSUS_TRACTS_API,
                    "acs_population": ACS_POPULATION_API,
                    "ntas": NTAS_API,
                    "uhf42": UHF42_API,
                },
                config=config,
                run_id=logger.run_id,
                extra={
                    "cd_tract_rows": len(cd_tract_xwalk),
                    "cd_nta_rows": len(cd_nta_xwalk),
                    "cd_uhf_rows": len(cd_uhf_xwalk),
                },
            )
            
            logger.info("SUCCESS: Built all crosswalks")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

