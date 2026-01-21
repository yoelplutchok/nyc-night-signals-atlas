"""
Hashing utilities for reproducibility and cache validation.

Per R12: Cache is hash-aware, not "file exists".
- Each output has metadata sidecar with:
  - input file hashes
  - config digest
  - code version (git commit if available)
  - runtime library versions
  - timestamp + run_id
- Scripts skip only if hashes match.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sleep_esi.io_utils import atomic_write_json, read_json
from sleep_esi.logging_utils import get_versions
from sleep_esi.paths import METADATA_DIR


# =============================================================================
# File Hashing
# =============================================================================

def hash_file(path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.
    
    Args:
        path: Path to file
        algorithm: Hash algorithm (default: sha256)
    
    Returns:
        Hex digest of file hash
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {path}")
    
    h = hashlib.new(algorithm)
    
    with open(path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    
    return h.hexdigest()


def hash_files(paths: List[Union[str, Path]], algorithm: str = "sha256") -> Dict[str, str]:
    """
    Compute hashes for multiple files.
    
    Args:
        paths: List of file paths
        algorithm: Hash algorithm
    
    Returns:
        Dictionary mapping path (as string) to hash
    """
    return {str(p): hash_file(p, algorithm) for p in paths}


def hash_string(s: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of a string.
    
    Args:
        s: String to hash
        algorithm: Hash algorithm
    
    Returns:
        Hex digest of string hash
    """
    h = hashlib.new(algorithm)
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def hash_dict(d: Dict[str, Any], algorithm: str = "sha256") -> str:
    """
    Compute hash of a dictionary (via JSON serialization).
    
    Args:
        d: Dictionary to hash
        algorithm: Hash algorithm
    
    Returns:
        Hex digest of dict hash
    """
    # Sort keys for deterministic serialization
    s = json.dumps(d, sort_keys=True, default=str)
    return hash_string(s, algorithm)


# =============================================================================
# Git Version Info
# =============================================================================

def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.
    
    Returns:
        Commit hash or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_dirty() -> Optional[bool]:
    """
    Check if git working directory has uncommitted changes.
    
    Returns:
        True if dirty, False if clean, None if not in git repo
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_info() -> Dict[str, Any]:
    """
    Get git repository information.
    
    Returns:
        Dictionary with commit, dirty status, etc.
    """
    return {
        "commit": get_git_commit(),
        "dirty": get_git_dirty(),
    }


# =============================================================================
# Metadata Sidecar
# =============================================================================

def create_metadata_sidecar(
    output_path: Union[str, Path],
    inputs: Dict[str, str],
    config: Dict[str, Any],
    run_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a metadata sidecar for an output file.
    
    Per R12: Every output has metadata sidecar with hashes, config, versions.
    
    Args:
        output_path: Path to the output file
        inputs: Dictionary mapping input names to file paths
        config: Configuration used for this run
        run_id: Unique run identifier
        extra: Additional metadata to include
    
    Returns:
        Metadata dictionary
    """
    output_path = Path(output_path)
    
    # Hash input files
    input_hashes = {}
    for name, path in inputs.items():
        path = Path(path)
        if path.exists():
            input_hashes[name] = {
                "path": str(path),
                "hash": hash_file(path),
            }
        else:
            input_hashes[name] = {
                "path": str(path),
                "hash": None,
                "missing": True,
            }
    
    metadata = {
        "output_file": str(output_path),
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": input_hashes,
        "config_digest": hash_dict(config),
        "config": config,
        "git": get_git_info(),
        "versions": get_versions(),
    }
    
    if extra:
        metadata["extra"] = extra
    
    return metadata


def write_metadata_sidecar(
    output_path: Union[str, Path],
    inputs: Dict[str, str],
    config: Dict[str, Any],
    run_id: str,
    extra: Optional[Dict[str, Any]] = None,
    metadata_dir: Optional[Path] = None,
) -> Path:
    """
    Write a metadata sidecar file for an output.
    
    Args:
        output_path: Path to the output file
        inputs: Dictionary mapping input names to file paths
        config: Configuration used
        run_id: Run identifier
        extra: Additional metadata
        metadata_dir: Directory for sidecar files (default: METADATA_DIR)
    
    Returns:
        Path to the written sidecar file
    """
    if metadata_dir is None:
        metadata_dir = METADATA_DIR
    
    output_path = Path(output_path)
    metadata = create_metadata_sidecar(output_path, inputs, config, run_id, extra)
    
    # Sidecar filename based on output filename
    sidecar_name = f"{output_path.stem}_metadata.json"
    sidecar_path = metadata_dir / sidecar_name
    
    atomic_write_json(metadata, sidecar_path)
    
    return sidecar_path


def read_metadata_sidecar(output_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read metadata sidecar for an output file.
    
    Args:
        output_path: Path to the output file
    
    Returns:
        Metadata dictionary or None if not found
    """
    output_path = Path(output_path)
    sidecar_name = f"{output_path.stem}_metadata.json"
    sidecar_path = METADATA_DIR / sidecar_name
    
    if sidecar_path.exists():
        return read_json(sidecar_path)
    return None


# =============================================================================
# Cache Validation
# =============================================================================

def validate_cache(
    output_path: Union[str, Path],
    inputs: Dict[str, str],
    config: Dict[str, Any],
) -> bool:
    """
    Check if cached output is still valid based on input hashes and config.
    
    Per R12: Scripts skip only if hashes match.
    
    Args:
        output_path: Path to the output file
        inputs: Dictionary mapping input names to file paths
        config: Current configuration
    
    Returns:
        True if cache is valid (all hashes match), False otherwise
    """
    output_path = Path(output_path)
    
    # Output must exist
    if not output_path.exists():
        return False
    
    # Metadata must exist
    metadata = read_metadata_sidecar(output_path)
    if metadata is None:
        return False
    
    # Config must match
    current_config_hash = hash_dict(config)
    if metadata.get("config_digest") != current_config_hash:
        return False
    
    # All input hashes must match
    cached_inputs = metadata.get("inputs", {})
    for name, path in inputs.items():
        path = Path(path)
        
        if name not in cached_inputs:
            return False
        
        cached = cached_inputs[name]
        
        if not path.exists():
            # Input file missing - cache invalid
            return False
        
        current_hash = hash_file(path)
        if cached.get("hash") != current_hash:
            return False
    
    return True


def get_cache_status(
    output_path: Union[str, Path],
    inputs: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get detailed cache status for debugging.
    
    Args:
        output_path: Path to the output file
        inputs: Dictionary mapping input names to file paths
        config: Current configuration
    
    Returns:
        Dictionary with detailed cache status
    """
    output_path = Path(output_path)
    status = {
        "output_exists": output_path.exists(),
        "valid": False,
        "reason": None,
        "input_status": {},
    }
    
    if not output_path.exists():
        status["reason"] = "output_missing"
        return status
    
    metadata = read_metadata_sidecar(output_path)
    if metadata is None:
        status["reason"] = "metadata_missing"
        return status
    
    # Check config
    current_config_hash = hash_dict(config)
    cached_config_hash = metadata.get("config_digest")
    if cached_config_hash != current_config_hash:
        status["reason"] = "config_changed"
        status["config_current"] = current_config_hash[:12]
        status["config_cached"] = cached_config_hash[:12] if cached_config_hash else None
        return status
    
    # Check inputs
    cached_inputs = metadata.get("inputs", {})
    for name, path in inputs.items():
        path = Path(path)
        input_status = {"path": str(path)}
        
        if name not in cached_inputs:
            input_status["status"] = "not_in_cache"
            status["input_status"][name] = input_status
            status["reason"] = f"input_not_cached:{name}"
            return status
        
        if not path.exists():
            input_status["status"] = "missing"
            status["input_status"][name] = input_status
            status["reason"] = f"input_missing:{name}"
            return status
        
        current_hash = hash_file(path)
        cached_hash = cached_inputs[name].get("hash")
        
        if current_hash != cached_hash:
            input_status["status"] = "changed"
            input_status["current_hash"] = current_hash[:12]
            input_status["cached_hash"] = cached_hash[:12] if cached_hash else None
            status["input_status"][name] = input_status
            status["reason"] = f"input_changed:{name}"
            return status
        
        input_status["status"] = "valid"
        status["input_status"][name] = input_status
    
    status["valid"] = True
    status["reason"] = "all_hashes_match"
    return status

