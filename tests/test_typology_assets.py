"""
Tests for Typology Assets (Script 08b) outputs.

Validates:
- CD cluster profiles and labels
- NTA cluster profiles and labels
- Joined types + stability tables
- GeoJSON map layers
- Chart outputs
"""

import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def reports_dir():
    """Path to reports outputs."""
    return Path("data/processed/reports")


@pytest.fixture(scope="module")
def figures_dir(reports_dir):
    """Path to figures outputs."""
    return reports_dir / "figures"


# =============================================================================
# CD Profile Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def cd_profiles(reports_dir):
    """Load CD cluster profiles."""
    path = reports_dir / "cd_cluster_profiles.csv"
    if not path.exists():
        pytest.skip(f"CD profiles not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def cd_labels(reports_dir):
    """Load CD cluster labels."""
    path = reports_dir / "cluster_labels_cd.csv"
    if not path.exists():
        pytest.skip(f"CD labels not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def cd_types_stability(reports_dir):
    """Load CD types with stability parquet."""
    path = reports_dir / "cd_types_with_stability.parquet"
    if not path.exists():
        pytest.skip(f"CD types not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def cd_types_geojson(reports_dir):
    """Load CD types GeoJSON."""
    path = reports_dir / "cd_types_with_stability.geojson"
    if not path.exists():
        pytest.skip(f"CD types GeoJSON not found: {path}")
    return gpd.read_file(path)


# =============================================================================
# NTA Profile Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def nta_profiles(reports_dir):
    """Load NTA cluster profiles."""
    path = reports_dir / "nta_cluster_profiles.csv"
    if not path.exists():
        pytest.skip(f"NTA profiles not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def nta_labels(reports_dir):
    """Load NTA cluster labels."""
    path = reports_dir / "cluster_labels_nta.csv"
    if not path.exists():
        pytest.skip(f"NTA labels not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def nta_types_stability(reports_dir):
    """Load NTA types with stability parquet."""
    path = reports_dir / "nta_types_with_stability_residential.parquet"
    if not path.exists():
        pytest.skip(f"NTA types not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def nta_types_geojson(reports_dir):
    """Load NTA types GeoJSON."""
    path = reports_dir / "nta_types_with_stability_residential.geojson"
    if not path.exists():
        pytest.skip(f"NTA types GeoJSON not found: {path}")
    return gpd.read_file(path)


# =============================================================================
# Test Class: CD Cluster Profiles
# =============================================================================

class TestCDProfiles:
    """Test CD cluster profile table."""
    
    def test_has_required_columns(self, cd_profiles):
        """Should have required columns."""
        required = ["cluster_id", "cluster_label", "n_units"]
        for col in required:
            assert col in cd_profiles.columns, f"Missing column: {col}"
    
    def test_has_mean_columns(self, cd_profiles):
        """Should have mean feature columns."""
        mean_cols = [c for c in cd_profiles.columns if c.endswith("_mean")]
        assert len(mean_cols) >= 10, f"Expected at least 10 mean columns, got {len(mean_cols)}"
    
    def test_has_z_columns(self, cd_profiles):
        """Should have z-score columns."""
        z_cols = [c for c in cd_profiles.columns if c.endswith("_z")]
        assert len(z_cols) >= 10, f"Expected at least 10 z columns, got {len(z_cols)}"
    
    def test_n_units_sum_to_59(self, cd_profiles):
        """Total units should sum to 59 CDs."""
        assert cd_profiles["n_units"].sum() == 59
    
    def test_all_clusters_present(self, cd_profiles):
        """All 9 clusters should be present."""
        assert len(cd_profiles) == 9


# =============================================================================
# Test Class: CD Cluster Labels
# =============================================================================

class TestCDLabels:
    """Test CD cluster labels table."""
    
    def test_has_required_columns(self, cd_labels):
        """Should have required columns."""
        required = ["level", "cluster_id", "cluster_label_short", "cluster_label_long", "defining_features"]
        for col in required:
            assert col in cd_labels.columns, f"Missing column: {col}"
    
    def test_all_rows_are_cd(self, cd_labels):
        """All rows should be level=CD."""
        assert (cd_labels["level"] == "CD").all()
    
    def test_no_null_labels(self, cd_labels):
        """No null labels."""
        assert cd_labels["cluster_label_short"].notna().all()
        assert cd_labels["cluster_label_long"].notna().all()
    
    def test_all_clusters_present(self, cd_labels):
        """All 9 clusters should have labels."""
        assert len(cd_labels) == 9


# =============================================================================
# Test Class: CD Types with Stability
# =============================================================================

class TestCDTypesStability:
    """Test CD types + stability joined table."""
    
    def test_59_rows(self, cd_types_stability):
        """Should have 59 CDs."""
        assert len(cd_types_stability) == 59
    
    def test_has_pooled_cluster(self, cd_types_stability):
        """Should have pooled cluster columns."""
        assert "cluster_id_pooled" in cd_types_stability.columns
        assert "cluster_label_short" in cd_types_stability.columns
    
    def test_has_stability_columns(self, cd_types_stability):
        """Should have stability columns."""
        required = [
            "cluster_id_2021", "cluster_id_2022", "cluster_id_2023",
            "transition_count", "entropy_score", "stability_class"
        ]
        for col in required:
            assert col in cd_types_stability.columns, f"Missing column: {col}"
    
    def test_has_stability_class2(self, cd_types_stability):
        """Should have stability_class2."""
        assert "stability_class2" in cd_types_stability.columns
        assert "is_mostly_structural" in cd_types_stability.columns
    
    def test_valid_stability_class2(self, cd_types_stability):
        """stability_class2 should have valid values."""
        valid = {"Strict Structural", "Mostly Structural", "Episodic"}
        assert cd_types_stability["stability_class2"].isin(valid).all()
    
    def test_has_is_outlier(self, cd_types_stability):
        """Should have is_outlier_cluster."""
        assert "is_outlier_cluster" in cd_types_stability.columns


# =============================================================================
# Test Class: CD Types GeoJSON
# =============================================================================

class TestCDTypesGeoJSON:
    """Test CD types GeoJSON map layer."""
    
    def test_59_rows(self, cd_types_geojson):
        """Should have 59 CDs."""
        assert len(cd_types_geojson) == 59
    
    def test_valid_geometry(self, cd_types_geojson):
        """All geometries should be valid."""
        assert cd_types_geojson.geometry.is_valid.all()
    
    def test_crs_is_4326(self, cd_types_geojson):
        """CRS should be EPSG:4326."""
        assert cd_types_geojson.crs is not None
        assert cd_types_geojson.crs.to_epsg() == 4326
    
    def test_has_boro_cd(self, cd_types_geojson):
        """Should have boro_cd."""
        assert "boro_cd" in cd_types_geojson.columns


# =============================================================================
# Test Class: NTA Cluster Profiles
# =============================================================================

class TestNTAProfiles:
    """Test NTA cluster profile table."""
    
    def test_has_required_columns(self, nta_profiles):
        """Should have required columns."""
        required = ["cluster_id", "cluster_label", "n_units"]
        for col in required:
            assert col in nta_profiles.columns, f"Missing column: {col}"
    
    def test_has_mean_columns(self, nta_profiles):
        """Should have mean feature columns."""
        mean_cols = [c for c in nta_profiles.columns if c.endswith("_mean")]
        assert len(mean_cols) >= 10
    
    def test_n_units_sum_to_197(self, nta_profiles):
        """Total units should sum to 197 NTAs."""
        assert nta_profiles["n_units"].sum() == 197


# =============================================================================
# Test Class: NTA Types with Stability
# =============================================================================

class TestNTATypesStability:
    """Test NTA types + stability joined table."""
    
    def test_197_rows(self, nta_types_stability):
        """Should have 197 residential NTAs."""
        assert len(nta_types_stability) == 197
    
    def test_has_pooled_cluster(self, nta_types_stability):
        """Should have pooled cluster columns."""
        assert "cluster_id_pooled" in nta_types_stability.columns
    
    def test_has_stability_class2(self, nta_types_stability):
        """Should have stability_class2."""
        assert "stability_class2" in nta_types_stability.columns


# =============================================================================
# Test Class: NTA Types GeoJSON
# =============================================================================

class TestNTATypesGeoJSON:
    """Test NTA types GeoJSON map layer."""
    
    def test_197_rows(self, nta_types_geojson):
        """Should have 197 residential NTAs."""
        assert len(nta_types_geojson) == 197
    
    def test_valid_geometry(self, nta_types_geojson):
        """All geometries should be valid."""
        assert nta_types_geojson.geometry.is_valid.all()
    
    def test_crs_is_4326(self, nta_types_geojson):
        """CRS should be EPSG:4326."""
        assert nta_types_geojson.crs is not None
        assert nta_types_geojson.crs.to_epsg() == 4326


# =============================================================================
# Test Class: Charts
# =============================================================================

class TestCharts:
    """Test chart outputs."""
    
    def test_cd_combined_chart_exists(self, figures_dir):
        """CD combined chart should exist."""
        path = figures_dir / "cd_cluster_profiles_all.png"
        assert path.exists(), f"Chart not found: {path}"
    
    def test_cd_individual_charts_exist(self, figures_dir):
        """CD individual charts should exist."""
        for i in range(9):
            path = figures_dir / f"cd_cluster_{i}_profile.png"
            assert path.exists(), f"Chart not found: {path}"
    
    def test_nta_combined_chart_exists(self, figures_dir):
        """NTA combined chart should exist."""
        path = figures_dir / "nta_cluster_profiles_all.png"
        if not path.exists():
            pytest.skip("NTA charts not yet built")
        assert path.exists()


# =============================================================================
# Test Class: Metadata
# =============================================================================

class TestMetadata:
    """Test metadata sidecar."""
    
    def test_metadata_exists(self):
        """Metadata sidecar should exist."""
        metadata_path = Path("data/processed/metadata/cd_cluster_profiles_metadata.json")
        assert metadata_path.exists(), f"Metadata not found: {metadata_path}"

