"""
Tests for Hotspot Concentration Analysis (Script 06 outputs).

Per NYC_Night_Signals_Plan.md:
- Grid-based analysis of nighttime 311 points
- Per-CD concentration metrics
- Hotspot cells with counts
"""

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
HOTSPOTS_DIR = PROCESSED_DIR / "hotspots"

CELLS_PARQUET = HOTSPOTS_DIR / "hotspot_cells.parquet"
CELLS_GEOJSON = HOTSPOTS_DIR / "hotspot_cells.geojson"
CD_CONCENTRATION = HOTSPOTS_DIR / "cd_hotspot_concentration.parquet"
CD_CONCENTRATION_CSV = HOTSPOTS_DIR / "cd_hotspot_concentration.csv"
TOP_HOTSPOTS = HOTSPOTS_DIR / "top_hotspots_citywide.csv"


class TestHotspotOutputsExist:
    """Tests for output file existence."""

    def test_cells_parquet_exists(self):
        assert CELLS_PARQUET.exists(), f"Missing: {CELLS_PARQUET}"

    def test_cells_geojson_exists(self):
        assert CELLS_GEOJSON.exists(), f"Missing: {CELLS_GEOJSON}"

    def test_cd_concentration_parquet_exists(self):
        assert CD_CONCENTRATION.exists(), f"Missing: {CD_CONCENTRATION}"

    def test_cd_concentration_csv_exists(self):
        assert CD_CONCENTRATION_CSV.exists(), f"Missing: {CD_CONCENTRATION_CSV}"

    def test_top_hotspots_exists(self):
        assert TOP_HOTSPOTS.exists(), f"Missing: {TOP_HOTSPOTS}"


@pytest.fixture
def df_cells():
    """Load hotspot cells."""
    if not CELLS_PARQUET.exists():
        pytest.skip(f"Missing: {CELLS_PARQUET}")
    return pd.read_parquet(CELLS_PARQUET)


@pytest.fixture
def gdf_cells():
    """Load hotspot cells GeoJSON."""
    if not CELLS_GEOJSON.exists():
        pytest.skip(f"Missing: {CELLS_GEOJSON}")
    return gpd.read_file(CELLS_GEOJSON)


@pytest.fixture
def df_concentration():
    """Load CD concentration metrics."""
    if not CD_CONCENTRATION.exists():
        pytest.skip(f"Missing: {CD_CONCENTRATION}")
    return pd.read_parquet(CD_CONCENTRATION)


@pytest.fixture
def df_top_hotspots():
    """Load top hotspots citywide."""
    if not TOP_HOTSPOTS.exists():
        pytest.skip(f"Missing: {TOP_HOTSPOTS}")
    return pd.read_csv(TOP_HOTSPOTS)


class TestCellsParquet:
    """Tests for hotspot_cells.parquet."""

    def test_has_cell_id(self, df_cells):
        """Must have cell_id column."""
        assert "cell_id" in df_cells.columns

    def test_cell_id_unique(self, df_cells):
        """cell_id must be unique."""
        assert df_cells["cell_id"].is_unique

    def test_has_count(self, df_cells):
        """Must have count column."""
        assert "count" in df_cells.columns

    def test_count_positive(self, df_cells):
        """All counts must be positive (cells with 0 excluded)."""
        assert (df_cells["count"] > 0).all()

    def test_has_boro_cd(self, df_cells):
        """Must have boro_cd column."""
        assert "boro_cd" in df_cells.columns

    def test_has_is_hotspot(self, df_cells):
        """Must have is_hotspot column."""
        assert "is_hotspot" in df_cells.columns

    def test_is_hotspot_boolean(self, df_cells):
        """is_hotspot must be boolean."""
        assert df_cells["is_hotspot"].dtype == bool


class TestCellsGeoJSON:
    """Tests for hotspot_cells.geojson (hotspot cells only)."""

    def test_has_geometry(self, gdf_cells):
        """Must have valid geometry."""
        assert gdf_cells.geometry.notna().all()

    def test_all_are_hotspots(self, gdf_cells):
        """All cells in GeoJSON should be hotspots."""
        assert gdf_cells["is_hotspot"].all()

    def test_has_count(self, gdf_cells):
        """Must have count column."""
        assert "count" in gdf_cells.columns

    def test_count_above_threshold(self, gdf_cells):
        """All hotspot cells should have count ≥ threshold (default 10)."""
        # Default threshold is 10
        assert (gdf_cells["count"] >= 10).all()


class TestCDConcentration:
    """Tests for cd_hotspot_concentration.parquet."""

    def test_row_count_is_59(self, df_concentration):
        """Must have 59 CDs."""
        assert len(df_concentration) == 59, f"Expected 59 rows, got {len(df_concentration)}"

    def test_boro_cd_unique(self, df_concentration):
        """boro_cd must be unique."""
        assert df_concentration["boro_cd"].is_unique

    def test_has_cd_label(self, df_concentration):
        """Must have cd_label column."""
        assert "cd_label" in df_concentration.columns

    def test_cd_label_no_nulls(self, df_concentration):
        """cd_label must have no nulls."""
        assert df_concentration["cd_label"].notna().all()

    def test_has_gini_coefficient(self, df_concentration):
        """Must have gini_coefficient column."""
        assert "gini_coefficient" in df_concentration.columns

    def test_gini_in_valid_range(self, df_concentration):
        """Gini coefficient must be in [0, 1]."""
        assert (df_concentration["gini_coefficient"] >= 0).all()
        assert (df_concentration["gini_coefficient"] <= 1).all()

    def test_has_top_1pct_share(self, df_concentration):
        """Must have top_1pct_share column."""
        assert "top_1pct_share" in df_concentration.columns

    def test_top_1pct_share_in_valid_range(self, df_concentration):
        """top_1pct_share must be in [0, 1]."""
        assert (df_concentration["top_1pct_share"] >= 0).all()
        assert (df_concentration["top_1pct_share"] <= 1).all()

    def test_has_hotspot_count(self, df_concentration):
        """Must have hotspot_count column."""
        assert "hotspot_count" in df_concentration.columns

    def test_hotspot_count_non_negative(self, df_concentration):
        """hotspot_count must be non-negative."""
        assert (df_concentration["hotspot_count"] >= 0).all()

    def test_has_total_complaints(self, df_concentration):
        """Must have total_complaints column."""
        assert "total_complaints" in df_concentration.columns

    def test_total_complaints_positive(self, df_concentration):
        """total_complaints must be positive."""
        assert (df_concentration["total_complaints"] > 0).all()

    def test_has_cell_count(self, df_concentration):
        """Must have cell_count column."""
        assert "cell_count" in df_concentration.columns


class TestTopHotspots:
    """Tests for top_hotspots_citywide.csv."""

    def test_has_cell_id(self, df_top_hotspots):
        """Must have cell_id column."""
        assert "cell_id" in df_top_hotspots.columns

    def test_has_count(self, df_top_hotspots):
        """Must have count column."""
        assert "count" in df_top_hotspots.columns

    def test_sorted_by_count(self, df_top_hotspots):
        """Should be sorted by count descending."""
        assert df_top_hotspots["count"].is_monotonic_decreasing

    def test_has_boro_cd(self, df_top_hotspots):
        """Must have boro_cd column."""
        assert "boro_cd" in df_top_hotspots.columns

    def test_has_cd_label(self, df_top_hotspots):
        """Must have cd_label or cd_short column."""
        assert "cd_label" in df_top_hotspots.columns or "cd_short" in df_top_hotspots.columns

    def test_count_100_rows_or_less(self, df_top_hotspots):
        """Should have at most 100 rows (top_n_citywide default)."""
        assert len(df_top_hotspots) <= 100


class TestConcentrationReasonableness:
    """Tests for reasonableness of concentration metrics."""

    def test_mean_gini_reasonable(self, df_concentration):
        """Mean Gini should be in a reasonable range (0.3-0.9 for urban areas)."""
        mean_gini = df_concentration["gini_coefficient"].mean()
        assert 0.1 < mean_gini < 0.95, f"Mean Gini {mean_gini} seems unreasonable"

    def test_some_hotspots_exist(self, df_concentration):
        """At least some CDs should have hotspots."""
        total_hotspots = df_concentration["hotspot_count"].sum()
        assert total_hotspots > 0, "No hotspots found across all CDs"

    def test_max_cell_share_not_always_one(self, df_concentration):
        """Not all CDs should have max_cell_share = 1 (would indicate single-cell CDs)."""
        if "max_cell_share" in df_concentration.columns:
            all_ones = (df_concentration["max_cell_share"] == 1).all()
            assert not all_ones, "All CDs have max_cell_share = 1"


class TestDataConsistency:
    """Tests for data consistency between outputs."""

    def test_cells_total_matches_cd_total(self, df_cells, df_concentration):
        """Total complaints in cells should match CD totals (approximately)."""
        cells_total = df_cells["count"].sum()
        cd_total = df_concentration["total_complaints"].sum()
        # Allow small difference due to cells at CD boundaries
        assert abs(cells_total - cd_total) / cd_total < 0.05, (
            f"Cells total {cells_total} vs CD total {cd_total}"
        )

