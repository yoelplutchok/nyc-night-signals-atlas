"""
I/O utilities with atomic writes and safe reads.

Per R11: All outputs written via temp file → rename/replace.
Per R5: GeoParquet is the internal truth; GeoJSON as export.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import pandas as pd
import yaml


# =============================================================================
# Atomic Write Utilities (R11)
# =============================================================================

@contextmanager
def atomic_write(
    target_path: Union[str, Path],
    mode: str = "w",
    suffix: Optional[str] = None,
):
    """
    Context manager for atomic file writes.
    
    Writes to a temporary file first, then atomically renames to target.
    If an exception occurs, the temp file is cleaned up and target unchanged.
    
    Args:
        target_path: Final destination path
        mode: File mode ('w' for text, 'wb' for binary)
        suffix: Optional suffix for temp file (e.g., '.parquet')
    
    Yields:
        File handle for writing
    
    Example:
        with atomic_write("output.csv") as f:
            f.write("data")
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine suffix from target if not provided
    if suffix is None:
        suffix = target_path.suffix or ".tmp"
    
    # Create temp file in same directory (for atomic rename)
    fd, temp_path = tempfile.mkstemp(
        suffix=suffix,
        prefix=f".{target_path.stem}_",
        dir=target_path.parent,
    )
    temp_path = Path(temp_path)
    
    try:
        # Close the file descriptor from mkstemp, we'll open properly
        os.close(fd)
        
        with open(temp_path, mode) as f:
            yield f
        
        # Atomic rename (works on same filesystem)
        temp_path.replace(target_path)
        
    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_df(
    df: pd.DataFrame,
    target_path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Atomically write a DataFrame to CSV or Parquet.
    
    File format determined by extension.
    
    Args:
        df: DataFrame to write
        target_path: Destination path (.csv or .parquet)
        **kwargs: Additional arguments passed to to_csv/to_parquet
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = target_path.suffix.lower()
    
    # Create temp file
    fd, temp_path = tempfile.mkstemp(
        suffix=suffix,
        prefix=f".{target_path.stem}_",
        dir=target_path.parent,
    )
    os.close(fd)
    temp_path = Path(temp_path)
    
    try:
        if suffix == ".parquet":
            df.to_parquet(temp_path, **kwargs)
        elif suffix == ".csv":
            df.to_csv(temp_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        temp_path.replace(target_path)
        
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_gdf(
    gdf: gpd.GeoDataFrame,
    target_path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Atomically write a GeoDataFrame to GeoParquet or GeoJSON.
    
    Per R5: GeoParquet is preferred for internal use.
    
    Args:
        gdf: GeoDataFrame to write
        target_path: Destination path (.parquet, .geojson, .gpkg)
        **kwargs: Additional arguments passed to writer
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = target_path.suffix.lower()
    
    # Create temp file
    fd, temp_path = tempfile.mkstemp(
        suffix=suffix,
        prefix=f".{target_path.stem}_",
        dir=target_path.parent,
    )
    os.close(fd)
    temp_path = Path(temp_path)
    
    try:
        if suffix == ".parquet":
            gdf.to_parquet(temp_path, **kwargs)
        elif suffix == ".geojson":
            gdf.to_file(temp_path, driver="GeoJSON", **kwargs)
        elif suffix == ".gpkg":
            gdf.to_file(temp_path, driver="GPKG", **kwargs)
        else:
            raise ValueError(f"Unsupported geo format: {suffix}")
        
        temp_path.replace(target_path)
        
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_json(
    data: Any,
    target_path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Atomically write JSON data.
    
    Args:
        data: JSON-serializable data
        target_path: Destination path
        **kwargs: Additional arguments passed to json.dump
    """
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("default", str)
    
    with atomic_write(target_path, mode="w", suffix=".json") as f:
        json.dump(data, f, **kwargs)


def atomic_write_yaml(
    data: Any,
    target_path: Union[str, Path],
    **kwargs,
) -> None:
    """
    Atomically write YAML data.
    
    Args:
        data: YAML-serializable data
        target_path: Destination path
        **kwargs: Additional arguments passed to yaml.safe_dump
    """
    kwargs.setdefault("default_flow_style", False)
    kwargs.setdefault("sort_keys", False)
    
    with atomic_write(target_path, mode="w", suffix=".yml") as f:
        yaml.safe_dump(data, f, **kwargs)


# =============================================================================
# Read Utilities
# =============================================================================

def read_yaml(path: Union[str, Path]) -> dict:
    """Read a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Union[str, Path]) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_gdf(
    path: Union[str, Path],
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Read a GeoDataFrame from file.
    
    Supports GeoParquet, GeoJSON, GeoPackage, Shapefile.
    
    Args:
        path: Path to geo file
        **kwargs: Additional arguments passed to reader
    
    Returns:
        GeoDataFrame
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix == ".parquet":
        return gpd.read_parquet(path, **kwargs)
    else:
        return gpd.read_file(path, **kwargs)


def read_df(
    path: Union[str, Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Read a DataFrame from CSV or Parquet.
    
    Args:
        path: Path to data file
        **kwargs: Additional arguments passed to reader
    
    Returns:
        DataFrame
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    elif suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


# =============================================================================
# Cleanup Utilities
# =============================================================================

def cleanup_temp_files(directory: Union[str, Path], pattern: str = ".*") -> int:
    """
    Clean up orphaned temp files (from failed atomic writes).
    
    Args:
        directory: Directory to clean
        pattern: Glob pattern for temp files (default: hidden files starting with .)
    
    Returns:
        Number of files removed
    """
    directory = Path(directory)
    count = 0
    
    for f in directory.glob(pattern):
        if f.is_file() and f.name.startswith("."):
            f.unlink()
            count += 1
    
    return count

