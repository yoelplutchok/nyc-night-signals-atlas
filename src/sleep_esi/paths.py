"""
Canonical path resolution for the Sleep ESI project.

This module provides the single source of truth for all paths in the project.
All scripts MUST import paths from here — no relative '../' paths allowed.

Per Section 2.1 of the research plan:
- Detect root via `.project-root` (primary) and fallback markers
- Expose canonical Paths: RAW_DIR, INTERIM_DIR, PROCESSED_DIR, etc.
"""

from pathlib import Path
from typing import Optional

# Markers to detect project root (in priority order)
ROOT_MARKERS = [".project-root", "pyproject.toml", ".git"]


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root by searching upward for marker files.
    
    Args:
        start_path: Starting directory for search. Defaults to this file's location.
        
    Returns:
        Path to project root directory.
        
    Raises:
        FileNotFoundError: If no root marker is found.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    
    current = start_path
    
    # Search upward until we find a marker or hit filesystem root
    while current != current.parent:
        for marker in ROOT_MARKERS:
            if (current / marker).exists():
                return current
        current = current.parent
    
    # Check root directory itself
    for marker in ROOT_MARKERS:
        if (current / marker).exists():
            return current
    
    raise FileNotFoundError(
        f"Could not find project root. Searched for markers {ROOT_MARKERS} "
        f"starting from {start_path}"
    )


# =============================================================================
# Canonical paths (resolved at import time)
# =============================================================================

PROJECT_ROOT = find_project_root()

# Config
CONFIG_DIR = PROJECT_ROOT / "configs"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Processed subdirectories
GEO_DIR = PROCESSED_DIR / "geo"
XWALK_DIR = PROCESSED_DIR / "xwalk"
DOMAINS_DIR = PROCESSED_DIR / "domains"
INDEX_DIR = PROCESSED_DIR / "index"
EQUITY_DIR = PROCESSED_DIR / "equity"
VALIDATION_DIR = PROCESSED_DIR / "validation"
METADATA_DIR = PROCESSED_DIR / "metadata"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Source and scripts
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Tests
TESTS_DIR = PROJECT_ROOT / "tests"
FIXTURES_DIR = TESTS_DIR / "fixtures"

# Docs
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_dirs_exist() -> None:
    """Create all canonical directories if they don't exist."""
    dirs = [
        CONFIG_DIR,
        RAW_DIR, INTERIM_DIR,
        GEO_DIR, XWALK_DIR, DOMAINS_DIR, INDEX_DIR, 
        EQUITY_DIR, VALIDATION_DIR, METADATA_DIR,
        LOGS_DIR,
        FIGURES_DIR, TABLES_DIR,
        FIXTURES_DIR,
        DOCS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick verification when run directly
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RAW_DIR:      {RAW_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"CONFIG_DIR:   {CONFIG_DIR}")
    print(f"LOGS_DIR:     {LOGS_DIR}")

