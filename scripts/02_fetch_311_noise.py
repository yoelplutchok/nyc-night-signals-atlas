#!/usr/bin/env python3
"""
02_fetch_311_noise.py

Fetch 311 noise complaints from NYC Open Data API.

Per Section 4.1.1:
- Fetch noise-related 311 complaints for 2021-2023 (primary)
- Optional: 2019 for exploratory analysis
- Record complete provenance per R13

Outputs:
- data/raw/311_noise/raw_311_noise_YYYYMMDD.csv (raw snapshot)
- data/raw/_manifest.json (updated with provenance)

Data Source:
- 311 Service Requests: https://data.cityofnewyork.us/resource/erm2-nwe9.json
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from sleep_esi.hashing import hash_file
from sleep_esi.io_utils import atomic_write_df, atomic_write_json, read_yaml
from sleep_esi.logging_utils import get_logger
from sleep_esi.paths import CONFIG_DIR, RAW_DIR

# =============================================================================
# Constants
# =============================================================================

# NYC Open Data API endpoint for 311 Service Requests
API_ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# Noise complaint types to fetch (per params.yml)
NOISE_COMPLAINT_TYPES = [
    "Noise - Residential",
    "Noise - Street/Sidewalk",
    "Noise - Commercial",
    "Noise - Vehicle",
    "Noise - Park",
    "Noise",  # Generic noise category
    "Noise - Helicopter",
    "Noise - House of Worship",
]

# Time windows
PRIMARY_START = "2021-01-01"
PRIMARY_END = "2023-12-31"

# Optional exploratory (pre-pandemic baseline)
EXPLORATORY_START = "2019-01-01"
EXPLORATORY_END = "2019-12-31"

# API settings
PAGE_SIZE = 50000  # Max rows per request
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Output directory
RAW_311_DIR = RAW_DIR / "311_noise"


# =============================================================================
# API Fetching
# =============================================================================

def build_query(
    complaint_types: List[str],
    start_date: str,
    end_date: str,
    offset: int = 0,
    limit: int = PAGE_SIZE,
) -> str:
    """
    Build SoQL query for 311 noise complaints.
    
    Uses Socrata Query Language (SoQL) to filter server-side.
    """
    # Build complaint type filter
    type_conditions = " OR ".join([f"complaint_type='{t}'" for t in complaint_types])
    
    # Build full query
    query = (
        f"$where=({type_conditions}) "
        f"AND created_date >= '{start_date}T00:00:00' "
        f"AND created_date <= '{end_date}T23:59:59'"
        f"&$order=created_date"
        f"&$limit={limit}"
        f"&$offset={offset}"
    )
    
    return query


def fetch_page(
    query: str,
    logger,
    retry_count: int = 0,
) -> Optional[List[dict]]:
    """
    Fetch a single page of results from the API.
    
    Returns None if all retries exhausted.
    """
    url = f"{API_ENDPOINT}?{query}"
    
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Request failed, retrying in {RETRY_DELAY}s... ({e})")
            time.sleep(RETRY_DELAY)
            return fetch_page(query, logger, retry_count + 1)
        else:
            logger.error(f"Max retries exceeded: {e}")
            return None


def fetch_all_complaints(
    complaint_types: List[str],
    start_date: str,
    end_date: str,
    logger,
) -> pd.DataFrame:
    """
    Fetch all noise complaints for the given date range.
    
    Handles pagination automatically.
    """
    all_records = []
    offset = 0
    page_num = 0
    
    logger.info(f"Fetching 311 noise complaints from {start_date} to {end_date}")
    logger.info(f"Complaint types: {complaint_types}")
    
    while True:
        page_num += 1
        query = build_query(complaint_types, start_date, end_date, offset, PAGE_SIZE)
        
        logger.info(f"Fetching page {page_num} (offset {offset})...")
        
        records = fetch_page(query, logger)
        
        if records is None:
            raise RuntimeError(f"Failed to fetch page {page_num}")
        
        if len(records) == 0:
            logger.info(f"No more records. Total pages: {page_num - 1}")
            break
        
        all_records.extend(records)
        logger.info(f"Page {page_num}: {len(records)} records (total: {len(all_records)})")
        
        if len(records) < PAGE_SIZE:
            # Last page
            break
        
        offset += PAGE_SIZE
        
        # Small delay to be nice to the API
        time.sleep(0.5)
    
    logger.info(f"Total records fetched: {len(all_records)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    return df


def clean_311_data(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Clean and standardize the 311 data.
    
    - Parse dates
    - Standardize column names
    - Filter to records with valid coordinates
    """
    logger.info("Cleaning 311 data...")
    
    # Standardize column names (lowercase)
    df.columns = df.columns.str.lower()
    
    # Parse created_date
    if "created_date" in df.columns:
        df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    
    # Ensure latitude/longitude are numeric
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Log data quality
    total_rows = len(df)
    has_coords = df["latitude"].notna() & df["longitude"].notna() if "latitude" in df.columns else pd.Series([False] * len(df))
    has_date = df["created_date"].notna() if "created_date" in df.columns else pd.Series([False] * len(df))
    
    logger.info(f"Total records: {total_rows}")
    logger.info(f"Records with coordinates: {has_coords.sum()} ({100*has_coords.mean():.1f}%)")
    logger.info(f"Records with valid date: {has_date.sum()} ({100*has_date.mean():.1f}%)")
    
    # Log complaint type breakdown
    if "complaint_type" in df.columns:
        type_counts = df["complaint_type"].value_counts()
        logger.info("Complaint type breakdown:")
        for ctype, count in type_counts.items():
            logger.info(f"  {ctype}: {count:,}")
    
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    with get_logger("02_fetch_311_noise") as logger:
        logger.info("Starting 02_fetch_311_noise.py")
        
        # Load config
        config = read_yaml(CONFIG_DIR / "params.yml")
        logger.log_config(config)
        
        # Get noise categories from config (or use defaults)
        noise_categories = config.get("noise", {}).get("complaint_categories", NOISE_COMPLAINT_TYPES)
        
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Create output directory
            RAW_311_DIR.mkdir(parents=True, exist_ok=True)
            
            # Fetch primary window (2021-2023)
            logger.info("=" * 60)
            logger.info("Fetching PRIMARY window: 2021-2023")
            logger.info("=" * 60)
            
            df_primary = fetch_all_complaints(
                complaint_types=noise_categories,
                start_date=PRIMARY_START,
                end_date=PRIMARY_END,
                logger=logger,
            )
            
            df_primary = clean_311_data(df_primary, logger)
            
            # Save primary data
            primary_filename = f"raw_311_noise_2021_2023_{timestamp.strftime('%Y%m%d')}.csv"
            primary_path = RAW_311_DIR / primary_filename
            
            atomic_write_df(df_primary, primary_path, index=False)
            logger.info(f"Saved: {primary_path}")
            
            primary_hash = hash_file(primary_path)
            
            # Optionally fetch exploratory window (2019)
            # Uncomment to enable:
            # logger.info("=" * 60)
            # logger.info("Fetching EXPLORATORY window: 2019")
            # logger.info("=" * 60)
            # 
            # df_2019 = fetch_all_complaints(
            #     complaint_types=noise_categories,
            #     start_date=EXPLORATORY_START,
            #     end_date=EXPLORATORY_END,
            #     logger=logger,
            # )
            # 
            # df_2019 = clean_311_data(df_2019, logger)
            # 
            # exploratory_filename = f"raw_311_noise_2019_{timestamp.strftime('%Y%m%d')}.csv"
            # exploratory_path = RAW_311_DIR / exploratory_filename
            # atomic_write_df(df_2019, exploratory_path, index=False)
            
            # Update manifest with provenance
            manifest_path = RAW_DIR / "_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            else:
                manifest = {"downloads": []}
            
            provenance = {
                "source": "NYC Open Data - 311 Service Requests",
                "dataset_id": "erm2-nwe9",
                "api_endpoint": API_ENDPOINT,
                "download_timestamp": timestamp.isoformat(),
                "date_range": {
                    "start": PRIMARY_START,
                    "end": PRIMARY_END,
                },
                "complaint_types": noise_categories,
                "filename": primary_filename,
                "file_path": str(primary_path),
                "sha256": primary_hash,
                "row_count": len(df_primary),
                "columns": list(df_primary.columns),
            }
            
            manifest["downloads"].append(provenance)
            manifest["last_updated"] = timestamp.isoformat()
            
            atomic_write_json(manifest, manifest_path)
            logger.info(f"Updated manifest: {manifest_path}")
            
            # Log outputs
            logger.log_outputs({
                "raw_311_noise": str(primary_path),
            })
            
            # Log metrics
            logger.log_metrics({
                "total_complaints": len(df_primary),
                "date_range_start": PRIMARY_START,
                "date_range_end": PRIMARY_END,
                "complaint_types": len(noise_categories),
                "has_coordinates_pct": float(df_primary["latitude"].notna().mean()) if "latitude" in df_primary.columns else 0,
            })
            
            logger.info(f"SUCCESS: Fetched {len(df_primary):,} noise complaints")
            
        except Exception as e:
            logger.error(f"FAILED: {e}")
            raise


if __name__ == "__main__":
    main()

