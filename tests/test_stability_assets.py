"""
Tests for Temporal Stability Assets (Script 08a) outputs.

Validates:
- Row counts (59 CDs, 197 NTAs)
- Required columns and IDs
- Bounded entropy [0, 1]
- Valid stability classes
- GeoJSON validity
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
def cd_stability(reports_dir):
    """Load CD stability parquet."""
    path = reports_dir / "cd_stability.parquet"
    if not path.exists():
        pytest.skip(f"CD stability not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def cd_stability_csv(reports_dir):
    """Load CD stability CSV."""
    path = reports_dir / "cd_stability.csv"
    if not path.exists():
        pytest.skip(f"CD stability CSV not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def cd_stability_geojson(reports_dir):
    """Load CD stability GeoJSON."""
    path = reports_dir / "cd_stability.geojson"
    if not path.exists():
        pytest.skip(f"CD stability GeoJSON not found: {path}")
    return gpd.read_file(path)


@pytest.fixture(scope="module")
def nta_stability(reports_dir):
    """Load NTA stability parquet."""
    path = reports_dir / "nta_stability_residential.parquet"
    if not path.exists():
        pytest.skip(f"NTA stability not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def nta_stability_csv(reports_dir):
    """Load NTA stability CSV."""
    path = reports_dir / "nta_stability_residential.csv"
    if not path.exists():
        pytest.skip(f"NTA stability CSV not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def nta_stability_geojson(reports_dir):
    """Load NTA stability GeoJSON."""
    path = reports_dir / "nta_stability_residential.geojson"
    if not path.exists():
        pytest.skip(f"NTA stability GeoJSON not found: {path}")
    return gpd.read_file(path)


@pytest.fixture(scope="module")
def hotspot_persistent(reports_dir):
    """Load persistent hotspot GeoJSON."""
    path = reports_dir / "hotspot_persistent_all3.geojson"
    if not path.exists():
        pytest.skip(f"Hotspot persistent not found: {path}")
    return gpd.read_file(path)


# =============================================================================
# Test Class: CD Stability Row Counts
# =============================================================================

class TestCDStabilityRowCounts:
    """Test CD stability row counts."""
    
    def test_59_cds(self, cd_stability):
        """Should have 59 CDs."""
        assert len(cd_stability) == 59
    
    def test_parquet_csv_match(self, cd_stability, cd_stability_csv):
        """Parquet and CSV should have same row count."""
        assert len(cd_stability) == len(cd_stability_csv)
    
    def test_geojson_row_count(self, cd_stability_geojson, cd_stability):
        """GeoJSON should have same row count as parquet."""
        assert len(cd_stability_geojson) == len(cd_stability)


# =============================================================================
# Test Class: CD Stability Schema
# =============================================================================

class TestCDStabilitySchema:
    """Test CD stability schema."""
    
    def test_required_columns(self, cd_stability):
        """Should have required columns."""
        required = [
            "boro_cd",
            "cluster_id_2021",
            "cluster_id_2022",
            "cluster_id_2023",
            "transition_count",
            "is_stable_all_3",
            "is_stable_2_of_3",
            "entropy_score",
            "stability_class",
            "cd_label",
        ]
        for col in required:
            assert col in cd_stability.columns, f"Missing column: {col}"
    
    def test_no_null_boro_cd(self, cd_stability):
        """No null boro_cd."""
        assert cd_stability["boro_cd"].notna().all()
    
    def test_no_null_cd_label(self, cd_stability):
        """No null cd_label."""
        assert cd_stability["cd_label"].notna().all()


# =============================================================================
# Test Class: CD Stability Values
# =============================================================================

class TestCDStabilityValues:
    """Test CD stability value validity."""
    
    def test_entropy_bounded(self, cd_stability):
        """Entropy should be in [0, 1]."""
        assert cd_stability["entropy_score"].between(0, 1).all()
    
    def test_transition_count_range(self, cd_stability):
        """Transition count should be in [1, 3]."""
        assert cd_stability["transition_count"].between(1, 3).all()
    
    def test_valid_stability_class(self, cd_stability):
        """Stability class should be valid."""
        valid_classes = {"Structural", "Semi-structural", "Episodic"}
        assert cd_stability["stability_class"].isin(valid_classes).all()
    
    def test_stable_all_3_implies_transition_1(self, cd_stability):
        """is_stable_all_3 should imply transition_count == 1."""
        stable = cd_stability[cd_stability["is_stable_all_3"] == True]
        if len(stable) > 0:
            assert (stable["transition_count"] == 1).all()
    
    def test_stable_all_3_implies_entropy_0(self, cd_stability):
        """is_stable_all_3 should imply entropy_score == 0."""
        stable = cd_stability[cd_stability["is_stable_all_3"] == True]
        if len(stable) > 0:
            assert (stable["entropy_score"] == 0).all()


# =============================================================================
# Test Class: CD Stability GeoJSON
# =============================================================================

class TestCDStabilityGeoJSON:
    """Test CD stability GeoJSON."""
    
    def test_valid_geometry(self, cd_stability_geojson):
        """All geometries should be valid."""
        assert cd_stability_geojson.geometry.is_valid.all()
    
    def test_no_empty_geometry(self, cd_stability_geojson):
        """No empty geometries."""
        assert not cd_stability_geojson.geometry.is_empty.any()
    
    def test_crs_is_4326(self, cd_stability_geojson):
        """CRS should be EPSG:4326."""
        assert cd_stability_geojson.crs is not None
        assert cd_stability_geojson.crs.to_epsg() == 4326


# =============================================================================
# Test Class: NTA Stability Row Counts
# =============================================================================

class TestNTAStabilityRowCounts:
    """Test NTA stability row counts."""
    
    def test_197_ntas(self, nta_stability):
        """Should have 197 residential NTAs."""
        assert len(nta_stability) == 197
    
    def test_parquet_csv_match(self, nta_stability, nta_stability_csv):
        """Parquet and CSV should have same row count."""
        assert len(nta_stability) == len(nta_stability_csv)
    
    def test_geojson_row_count(self, nta_stability_geojson, nta_stability):
        """GeoJSON should have same row count as parquet."""
        assert len(nta_stability_geojson) == len(nta_stability)


# =============================================================================
# Test Class: NTA Stability Schema
# =============================================================================

class TestNTAStabilitySchema:
    """Test NTA stability schema."""
    
    def test_required_columns(self, nta_stability):
        """Should have required columns."""
        required = [
            "ntacode",
            "cluster_id_2021",
            "cluster_id_2022",
            "cluster_id_2023",
            "transition_count",
            "is_stable_all_3",
            "is_stable_2_of_3",
            "entropy_score",
            "stability_class",
            "nta_name",
        ]
        for col in required:
            assert col in nta_stability.columns, f"Missing column: {col}"
    
    def test_no_null_ntacode(self, nta_stability):
        """No null ntacode."""
        assert nta_stability["ntacode"].notna().all()
    
    def test_no_null_nta_name(self, nta_stability):
        """No null nta_name."""
        assert nta_stability["nta_name"].notna().all()


# =============================================================================
# Test Class: NTA Stability Values
# =============================================================================

class TestNTAStabilityValues:
    """Test NTA stability value validity."""
    
    def test_entropy_bounded(self, nta_stability):
        """Entropy should be in [0, 1]."""
        assert nta_stability["entropy_score"].between(0, 1).all()
    
    def test_transition_count_range(self, nta_stability):
        """Transition count should be in [0, 3]."""
        # 0 is possible if all years are low volume
        assert nta_stability["transition_count"].between(0, 3).all()
    
    def test_valid_stability_class(self, nta_stability):
        """Stability class should be valid."""
        valid_classes = {"Structural", "Semi-structural", "Episodic"}
        assert nta_stability["stability_class"].isin(valid_classes).all()
    
    def test_has_low_volume_years(self, nta_stability):
        """Should have low_volume_years column."""
        assert "low_volume_years" in nta_stability.columns


# =============================================================================
# Test Class: NTA Stability GeoJSON
# =============================================================================

class TestNTAStabilityGeoJSON:
    """Test NTA stability GeoJSON."""
    
    def test_valid_geometry(self, nta_stability_geojson):
        """All geometries should be valid."""
        assert nta_stability_geojson.geometry.is_valid.all()
    
    def test_no_empty_geometry(self, nta_stability_geojson):
        """No empty geometries."""
        assert not nta_stability_geojson.geometry.is_empty.any()
    
    def test_crs_is_4326(self, nta_stability_geojson):
        """CRS should be EPSG:4326."""
        assert nta_stability_geojson.crs is not None
        assert nta_stability_geojson.crs.to_epsg() == 4326


# =============================================================================
# Test Class: Hotspot Persistent
# =============================================================================

class TestHotspotPersistent:
    """Test persistent hotspot layer."""
    
    def test_has_cell_id(self, hotspot_persistent):
        """Should have cell_id column."""
        assert "cell_id" in hotspot_persistent.columns
    
    def test_has_total_count(self, hotspot_persistent):
        """Should have total_count column."""
        assert "total_count" in hotspot_persistent.columns
    
    def test_has_year_counts(self, hotspot_persistent):
        """Should have per-year count columns."""
        for year in [2021, 2022, 2023]:
            assert f"count_{year}" in hotspot_persistent.columns
    
    def test_total_count_positive(self, hotspot_persistent):
        """Total count should be positive."""
        assert (hotspot_persistent["total_count"] > 0).all()
    
    def test_valid_geometry(self, hotspot_persistent):
        """All geometries should be valid."""
        assert hotspot_persistent.geometry.is_valid.all()
    
    def test_crs_is_4326(self, hotspot_persistent):
        """CRS should be EPSG:4326."""
        assert hotspot_persistent.crs is not None
        assert hotspot_persistent.crs.to_epsg() == 4326
    
    def test_no_raw_addresses(self, hotspot_persistent):
        """Should not contain raw addresses (privacy)."""
        # Check that certain address-related columns are not present
        forbidden_cols = ["top_address", "incident_address"]
        for col in forbidden_cols:
            assert col not in hotspot_persistent.columns, f"Found forbidden column: {col}"


# =============================================================================
# Test Class: Metadata
# =============================================================================

class TestMetadata:
    """Test metadata sidecar."""
    
    def test_metadata_exists(self):
        """Metadata sidecar should exist."""
        metadata_path = Path("data/processed/metadata/cd_stability_metadata.json")
        assert metadata_path.exists(), f"Metadata not found: {metadata_path}"
    
    def test_metadata_has_required_fields(self):
        """Metadata should have required fields."""
        import json
        metadata_path = Path("data/processed/metadata/cd_stability_metadata.json")
        if not metadata_path.exists():
            pytest.skip("Metadata not found")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert "inputs" in metadata
        assert "run_id" in metadata
        assert "extra" in metadata
        
        extra = metadata.get("extra", {})
        assert "threshold_structural" in extra
        assert "threshold_episodic" in extra

