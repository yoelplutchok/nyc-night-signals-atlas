"""
Tests for the paths module.

Per Section 2.1: Verify that project root detection and canonical paths work correctly.
"""

import pytest
from pathlib import Path

from sleep_esi.paths import (
    PROJECT_ROOT,
    find_project_root,
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    CONFIG_DIR,
    LOGS_DIR,
    GEO_DIR,
    XWALK_DIR,
    DOMAINS_DIR,
)


class TestProjectRoot:
    """Tests for project root detection."""

    def test_project_root_exists(self):
        """PROJECT_ROOT should be a valid directory."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_project_root_marker_exists(self):
        """The .project-root marker file should exist."""
        marker = PROJECT_ROOT / ".project-root"
        assert marker.exists(), "Missing .project-root marker file"

    def test_find_project_root_from_subdir(self):
        """find_project_root should work from any subdirectory."""
        # Start from src/sleep_esi
        subdir = PROJECT_ROOT / "src" / "sleep_esi"
        found_root = find_project_root(subdir)
        assert found_root == PROJECT_ROOT

    def test_find_project_root_raises_on_invalid_path(self, tmp_path):
        """find_project_root should raise if no marker found."""
        # Create a temp directory with no markers
        with pytest.raises(FileNotFoundError):
            find_project_root(tmp_path)


class TestCanonicalPaths:
    """Tests for canonical path definitions."""

    def test_raw_dir_under_data(self):
        """RAW_DIR should be under data/."""
        assert RAW_DIR.parent.name == "raw" or RAW_DIR.name == "raw"
        assert "data" in str(RAW_DIR)

    def test_processed_dir_under_data(self):
        """PROCESSED_DIR should be under data/."""
        assert "data" in str(PROCESSED_DIR)
        assert PROCESSED_DIR.name == "processed"

    def test_geo_dir_under_processed(self):
        """GEO_DIR should be under processed/."""
        assert GEO_DIR.parent == PROCESSED_DIR

    def test_no_relative_path_components(self):
        """Canonical paths should not contain '..' components."""
        paths_to_check = [
            PROJECT_ROOT, RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
            CONFIG_DIR, LOGS_DIR, GEO_DIR, XWALK_DIR, DOMAINS_DIR,
        ]
        for p in paths_to_check:
            assert ".." not in str(p), f"Path contains '..': {p}"

    def test_all_paths_absolute(self):
        """All canonical paths should be absolute."""
        paths_to_check = [
            PROJECT_ROOT, RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
            CONFIG_DIR, LOGS_DIR, GEO_DIR, XWALK_DIR, DOMAINS_DIR,
        ]
        for p in paths_to_check:
            assert p.is_absolute(), f"Path is not absolute: {p}"


@pytest.mark.smoke
class TestPathsSmoke:
    """Smoke tests for paths module."""

    def test_import_succeeds(self):
        """Basic import should work."""
        from sleep_esi import paths
        assert paths.PROJECT_ROOT is not None

    def test_directories_exist(self):
        """Key directories should exist after setup."""
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "src").exists()
        assert (PROJECT_ROOT / "data").exists()

