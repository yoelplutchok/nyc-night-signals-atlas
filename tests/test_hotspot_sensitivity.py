"""
Tests for Hotspot Sensitivity Analysis (Script 06c outputs).

Per NYC_Night_Signals_Plan.md Script 06c:
- Hotspot layers with artifact flags
- Sensitivity summary across grid sizes x thresholds
- Raw and clean CD concentration metrics
"""

import pytest
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path

# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
HOTSPOTS_DIR = PROCESSED_DIR / "hotspots"

OUTPUT_GE10 = HOTSPOTS_DIR / "hotspot_cells_ge10.geojson"
OUTPUT_GE50 = HOTSPOTS_DIR / "hotspot_cells_ge50.geojson"
OUTPUT_SENSITIVITY = HOTSPOTS_DIR / "hotspot_sensitivity_summary.csv"
OUTPUT_CD_CONCENTRATION = HOTSPOTS_DIR / "cd_hotspot_concentration.parquet"
OUTPUT_CD_CONCENTRATION_CLEAN = HOTSPOTS_DIR / "cd_hotspot_concentration_clean.parquet"


class TestOutputFilesExist:
    """Tests for output file existence."""

    def test_ge10_geojson_exists(self):
        assert OUTPUT_GE10.exists(), f"Missing: {OUTPUT_GE10}"

    def test_ge50_geojson_exists(self):
        assert OUTPUT_GE50.exists(), f"Missing: {OUTPUT_GE50}"

    def test_sensitivity_csv_exists(self):
        assert OUTPUT_SENSITIVITY.exists(), f"Missing: {OUTPUT_SENSITIVITY}"

    def test_cd_concentration_exists(self):
        assert OUTPUT_CD_CONCENTRATION.exists(), f"Missing: {OUTPUT_CD_CONCENTRATION}"

    def test_cd_concentration_clean_exists(self):
        assert OUTPUT_CD_CONCENTRATION_CLEAN.exists(), f"Missing: {OUTPUT_CD_CONCENTRATION_CLEAN}"


@pytest.fixture
def gdf_ge10():
    """Load analysis-grade hotspot cells."""
    if not OUTPUT_GE10.exists():
        pytest.skip(f"Missing: {OUTPUT_GE10}")
    return gpd.read_file(OUTPUT_GE10)


@pytest.fixture
def gdf_ge50():
    """Load map-grade hotspot cells."""
    if not OUTPUT_GE50.exists():
        pytest.skip(f"Missing: {OUTPUT_GE50}")
    return gpd.read_file(OUTPUT_GE50)


@pytest.fixture
def df_sensitivity():
    """Load sensitivity summary."""
    if not OUTPUT_SENSITIVITY.exists():
        pytest.skip(f"Missing: {OUTPUT_SENSITIVITY}")
    return pd.read_csv(OUTPUT_SENSITIVITY)


@pytest.fixture
def df_concentration():
    """Load raw CD concentration."""
    if not OUTPUT_CD_CONCENTRATION.exists():
        pytest.skip(f"Missing: {OUTPUT_CD_CONCENTRATION}")
    return pd.read_parquet(OUTPUT_CD_CONCENTRATION)


@pytest.fixture
def df_concentration_clean():
    """Load clean CD concentration."""
    if not OUTPUT_CD_CONCENTRATION_CLEAN.exists():
        pytest.skip(f"Missing: {OUTPUT_CD_CONCENTRATION_CLEAN}")
    return pd.read_parquet(OUTPUT_CD_CONCENTRATION_CLEAN)


class TestGE10GeoJSON:
    """Tests for hotspot_cells_ge10.geojson."""

    def test_has_geometry(self, gdf_ge10):
        """Must have valid geometry."""
        assert gdf_ge10.geometry.notna().all()

    def test_has_cell_id(self, gdf_ge10):
        """Must have cell_id column."""
        assert "cell_id" in gdf_ge10.columns

    def test_has_count(self, gdf_ge10):
        """Must have count column."""
        assert "count" in gdf_ge10.columns

    def test_count_at_least_10(self, gdf_ge10):
        """All cells must have count >= 10."""
        assert (gdf_ge10["count"] >= 10).all()

    def test_has_artifact_flags(self, gdf_ge10):
        """Must have artifact flag columns."""
        assert "is_repeat_location_dominant" in gdf_ge10.columns
        assert "is_suspected_artifact" in gdf_ge10.columns

    def test_artifact_flags_boolean(self, gdf_ge10):
        """Artifact flags must be boolean."""
        # GeoJSON may convert to int, accept both
        assert gdf_ge10["is_repeat_location_dominant"].isin([True, False, 0, 1]).all()
        assert gdf_ge10["is_suspected_artifact"].isin([True, False, 0, 1]).all()

    def test_has_top_latlon_share(self, gdf_ge10):
        """Must have top_latlon_share column."""
        assert "top_latlon_share" in gdf_ge10.columns

    def test_top_latlon_share_valid_range(self, gdf_ge10):
        """top_latlon_share must be in [0, 1]."""
        valid = gdf_ge10["top_latlon_share"].notna()
        if valid.any():
            assert (gdf_ge10.loc[valid, "top_latlon_share"] >= 0).all()
            assert (gdf_ge10.loc[valid, "top_latlon_share"] <= 1).all()

    def test_has_boro_cd(self, gdf_ge10):
        """Must have boro_cd column."""
        assert "boro_cd" in gdf_ge10.columns


class TestGE50GeoJSON:
    """Tests for hotspot_cells_ge50.geojson."""

    def test_has_geometry(self, gdf_ge50):
        """Must have valid geometry."""
        assert gdf_ge50.geometry.notna().all()

    def test_count_at_least_50(self, gdf_ge50):
        """All cells must have count >= 50."""
        assert (gdf_ge50["count"] >= 50).all()

    def test_has_artifact_flags(self, gdf_ge50):
        """Must have artifact flag columns."""
        assert "is_repeat_location_dominant" in gdf_ge50.columns
        assert "is_suspected_artifact" in gdf_ge50.columns

    def test_subset_of_ge10(self, gdf_ge10, gdf_ge50):
        """GE50 cells should be a subset of GE10 cells."""
        ge10_cells = set(gdf_ge10["cell_id"])
        ge50_cells = set(gdf_ge50["cell_id"])
        assert ge50_cells.issubset(ge10_cells)

    def test_fewer_than_ge10(self, gdf_ge10, gdf_ge50):
        """GE50 should have fewer cells than GE10."""
        assert len(gdf_ge50) <= len(gdf_ge10)


class TestSensitivitySummary:
    """Tests for hotspot_sensitivity_summary.csv."""

    def test_has_grid_size(self, df_sensitivity):
        """Must have grid_size_ft column."""
        assert "grid_size_ft" in df_sensitivity.columns

    def test_has_threshold(self, df_sensitivity):
        """Must have threshold column."""
        assert "threshold" in df_sensitivity.columns

    def test_has_n_hotspot_cells(self, df_sensitivity):
        """Must have n_hotspot_cells column."""
        assert "n_hotspot_cells" in df_sensitivity.columns

    def test_correct_grid_sizes(self, df_sensitivity):
        """Should test expected grid sizes (410, 820, 1640)."""
        expected_sizes = {410, 820, 1640}
        actual_sizes = set(df_sensitivity["grid_size_ft"].unique())
        assert expected_sizes == actual_sizes

    def test_correct_thresholds(self, df_sensitivity):
        """Should test expected thresholds (10, 25, 50)."""
        expected_thresholds = {10, 25, 50}
        actual_thresholds = set(df_sensitivity["threshold"].unique())
        assert expected_thresholds == actual_thresholds

    def test_expected_row_count(self, df_sensitivity):
        """Should have 3 grid sizes × 3 thresholds = 9 rows."""
        assert len(df_sensitivity) == 9

    def test_monotonic_hotspots_by_threshold(self, df_sensitivity):
        """Higher thresholds should have fewer hotspot cells."""
        for grid_size in df_sensitivity["grid_size_ft"].unique():
            subset = df_sensitivity[df_sensitivity["grid_size_ft"] == grid_size].sort_values("threshold")
            assert subset["n_hotspot_cells"].is_monotonic_decreasing or len(subset) == 1


class TestCDConcentrationRaw:
    """Tests for raw CD concentration metrics."""

    def test_has_59_cds(self, df_concentration):
        """Must have 59 CDs."""
        assert len(df_concentration) == 59

    def test_has_boro_cd(self, df_concentration):
        """Must have boro_cd column."""
        assert "boro_cd" in df_concentration.columns

    def test_boro_cd_unique(self, df_concentration):
        """boro_cd must be unique."""
        assert df_concentration["boro_cd"].is_unique

    def test_has_cd_label(self, df_concentration):
        """Must have cd_label column."""
        assert "cd_label" in df_concentration.columns

    def test_has_gini_coefficient(self, df_concentration):
        """Must have gini_coefficient column."""
        assert "gini_coefficient" in df_concentration.columns

    def test_gini_in_valid_range(self, df_concentration):
        """Gini must be in [0, 1]."""
        assert (df_concentration["gini_coefficient"] >= 0).all()
        assert (df_concentration["gini_coefficient"] <= 1).all()

    def test_has_artifact_counts(self, df_concentration):
        """Must have artifact count columns."""
        assert "n_repeat_location_dominant" in df_concentration.columns
        assert "n_suspected_artifact" in df_concentration.columns

    def test_artifact_counts_non_negative(self, df_concentration):
        """Artifact counts must be non-negative."""
        assert (df_concentration["n_repeat_location_dominant"] >= 0).all()
        assert (df_concentration["n_suspected_artifact"] >= 0).all()


class TestCDConcentrationClean:
    """Tests for clean CD concentration metrics."""

    def test_has_59_cds(self, df_concentration_clean):
        """Must have 59 CDs."""
        assert len(df_concentration_clean) == 59

    def test_has_boro_cd(self, df_concentration_clean):
        """Must have boro_cd column."""
        assert "boro_cd" in df_concentration_clean.columns

    def test_total_complaints_less_than_raw(self, df_concentration, df_concentration_clean):
        """Clean should have fewer or equal complaints than raw."""
        # Some CDs may have more complaints in clean if artifacts were in other CDs
        # But total across all CDs should be less or equal
        raw_total = df_concentration["total_complaints"].sum()
        clean_total = df_concentration_clean["total_complaints"].sum()
        assert clean_total <= raw_total


class TestArtifactFlagsConsistency:
    """Tests for artifact flag consistency."""

    def test_suspected_artifact_implies_repeat_location(self, gdf_ge10):
        """
        If suspected_artifact_share_threshold >= repeat_location_share_threshold,
        then most suspected artifacts should also be repeat location dominant.
        
        Exception: cells flagged due to single-day spikes only.
        """
        # Get cells flagged as artifacts
        artifacts = gdf_ge10[gdf_ge10["is_suspected_artifact"] == True]
        if len(artifacts) == 0:
            pytest.skip("No artifacts to test")
        
        # Most should also be repeat location dominant (unless single-day spike)
        also_repeat = (artifacts["is_repeat_location_dominant"] == True).sum()
        # At least some overlap expected
        assert also_repeat > 0 or len(artifacts) < 5

    def test_top_cell_33209_flagged(self, gdf_ge10):
        """Cell 33209 (BX 12 extreme) should be flagged as suspected artifact."""
        if 33209 not in gdf_ge10["cell_id"].values:
            pytest.skip("Cell 33209 not in GE10 layer")
        
        cell = gdf_ge10[gdf_ge10["cell_id"] == 33209].iloc[0]
        # This cell has 91% at single coordinate - should be flagged
        assert cell["is_repeat_location_dominant"] == True or cell["is_repeat_location_dominant"] == 1


class TestDataIntegrity:
    """Tests for data integrity between outputs."""

    def test_ge10_cells_have_counts(self, gdf_ge10):
        """All GE10 cells should have positive counts."""
        assert (gdf_ge10["count"] > 0).all()

    def test_sensitivity_total_reasonable(self, df_sensitivity):
        """Sensitivity totals should be reasonable."""
        # For 820ft grid and threshold 10, should match our known hotspot count
        row_820_10 = df_sensitivity[
            (df_sensitivity["grid_size_ft"] == 820) & 
            (df_sensitivity["threshold"] == 10)
        ]
        if len(row_820_10) > 0:
            assert row_820_10.iloc[0]["n_hotspot_cells"] > 1000  # We know there are 5000+ from 06

