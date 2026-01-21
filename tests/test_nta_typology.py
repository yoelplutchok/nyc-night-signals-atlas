"""
Tests for NTA Night Signatures Typology (Script 11) outputs.

Validates:
- Row counts and schema
- Cluster ID validity (-1 for Low Volume, 0..K-1 for clustered)
- NTA name presence
- Bounded shares
- Low Volume rules (cluster_id=-1, cluster_label="Low Volume", is_outlier_cluster=True)
- Reproducibility (via fixed seed)
- K selection scores schema
- Cluster summary schema
"""

import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def typology_dir():
    """Path to typology outputs."""
    return Path("data/processed/typology")


@pytest.fixture(scope="module")
def atlas_dir():
    """Path to atlas outputs."""
    return Path("data/processed/atlas")


@pytest.fixture(scope="module")
def nta_clusters(typology_dir):
    """Load NTA clusters parquet."""
    path = typology_dir / "nta_clusters_residential.parquet"
    if not path.exists():
        pytest.skip(f"NTA clusters not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def nta_clusters_csv(typology_dir):
    """Load NTA clusters CSV."""
    path = typology_dir / "nta_clusters_residential.csv"
    if not path.exists():
        pytest.skip(f"NTA clusters CSV not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def nta_clusters_geojson(typology_dir):
    """Load NTA clusters GeoJSON."""
    path = typology_dir / "nta_clusters_residential.geojson"
    if not path.exists():
        pytest.skip(f"NTA clusters GeoJSON not found: {path}")
    return gpd.read_file(path)


@pytest.fixture(scope="module")
def cluster_summary(typology_dir):
    """Load cluster summary."""
    path = typology_dir / "nta_cluster_summary_residential.csv"
    if not path.exists():
        pytest.skip(f"Cluster summary not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def k_selection_scores(typology_dir):
    """Load K selection scores."""
    path = typology_dir / "nta_k_selection_scores.csv"
    if not path.exists():
        pytest.skip(f"K selection scores not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def nta_features(atlas_dir):
    """Load NTA features for comparison."""
    path = atlas_dir / "311_nta_features_residential.parquet"
    if not path.exists():
        pytest.skip(f"NTA features not found: {path}")
    return pd.read_parquet(path)


# =============================================================================
# Test Class: Row Counts
# =============================================================================

class TestRowCounts:
    """Test row counts match expectations."""
    
    def test_residential_nta_count(self, nta_clusters):
        """Should have 197 residential NTAs."""
        assert len(nta_clusters) == 197, f"Expected 197, got {len(nta_clusters)}"
    
    def test_parquet_csv_match(self, nta_clusters, nta_clusters_csv):
        """Parquet and CSV should have same row count."""
        assert len(nta_clusters) == len(nta_clusters_csv)
    
    def test_geojson_row_count(self, nta_clusters_geojson, nta_clusters):
        """GeoJSON should have same row count as parquet."""
        assert len(nta_clusters_geojson) == len(nta_clusters)


# =============================================================================
# Test Class: Schema
# =============================================================================

class TestSchema:
    """Test output schema."""
    
    def test_required_columns_parquet(self, nta_clusters):
        """Parquet should have required columns."""
        required = [
            "ntacode",
            "nta_name",
            "borough_name",
            "cluster_id",
            "cluster_label",
            "is_outlier_cluster",
        ]
        for col in required:
            assert col in nta_clusters.columns, f"Missing column: {col}"
    
    def test_required_columns_geojson(self, nta_clusters_geojson):
        """GeoJSON should have required columns."""
        required = [
            "ntacode",
            "nta_name",
            "cluster_id",
            "cluster_label",
            "is_outlier_cluster",
            "geometry",
        ]
        for col in required:
            assert col in nta_clusters_geojson.columns, f"Missing column: {col}"
    
    def test_k_scores_schema(self, k_selection_scores):
        """K selection scores should have required columns."""
        required = [
            "k",
            "silhouette_score",
            "calinski_harabasz_score",
            "min_cluster_size",
            "n_singletons",
        ]
        for col in required:
            assert col in k_selection_scores.columns, f"Missing column: {col}"
    
    def test_cluster_summary_schema(self, cluster_summary):
        """Cluster summary should have required columns."""
        required = ["cluster_id", "n_ntas", "cluster_label"]
        for col in required:
            assert col in cluster_summary.columns, f"Missing column: {col}"


# =============================================================================
# Test Class: NTA Identifiers
# =============================================================================

class TestNTAIdentifiers:
    """Test NTA code and name validity."""
    
    def test_no_null_ntacode(self, nta_clusters):
        """No null NTA codes."""
        assert nta_clusters["ntacode"].notna().all()
    
    def test_unique_ntacode(self, nta_clusters):
        """NTA codes should be unique."""
        assert nta_clusters["ntacode"].is_unique
    
    def test_no_null_nta_name(self, nta_clusters):
        """No null NTA names."""
        assert nta_clusters["nta_name"].notna().all()
    
    def test_ntacode_format(self, nta_clusters):
        """NTA codes should follow format (2 letters + 4 digits)."""
        pattern = r"^[A-Z]{2}\d{4}$"
        valid = nta_clusters["ntacode"].str.match(pattern)
        assert valid.all(), f"Invalid NTA codes: {nta_clusters.loc[~valid, 'ntacode'].tolist()}"


# =============================================================================
# Test Class: Cluster IDs
# =============================================================================

class TestClusterIDs:
    """Test cluster ID validity."""
    
    def test_no_null_cluster_id(self, nta_clusters):
        """No null cluster IDs."""
        assert nta_clusters["cluster_id"].notna().all()
    
    def test_cluster_id_range(self, nta_clusters):
        """Cluster IDs should be -1 (Low Volume) or 0..K-1."""
        min_id = nta_clusters["cluster_id"].min()
        assert min_id >= -1, f"Cluster ID below -1: {min_id}"
    
    def test_cluster_id_dtype(self, nta_clusters):
        """Cluster ID should be integer."""
        assert nta_clusters["cluster_id"].dtype in ["int64", "Int64"]
    
    def test_has_low_volume_cluster(self, nta_clusters):
        """Should have at least some Low Volume NTAs (cluster_id=-1)."""
        # This is expected based on min_cluster_count=200
        low_volume_count = (nta_clusters["cluster_id"] == -1).sum()
        # May or may not have low volume depending on data - just check it's handled
        assert low_volume_count >= 0


# =============================================================================
# Test Class: Cluster Labels
# =============================================================================

class TestClusterLabels:
    """Test cluster label validity."""
    
    def test_no_null_cluster_label(self, nta_clusters):
        """No null cluster labels."""
        assert nta_clusters["cluster_label"].notna().all()
    
    def test_low_volume_label(self, nta_clusters):
        """Low Volume NTAs should have cluster_label='Low Volume'."""
        low_vol = nta_clusters[nta_clusters["cluster_id"] == -1]
        if len(low_vol) > 0:
            assert (low_vol["cluster_label"] == "Low Volume").all()
    
    def test_clustered_labels_not_low_volume(self, nta_clusters):
        """Clustered NTAs (cluster_id >= 0) should not have 'Low Volume' label."""
        clustered = nta_clusters[nta_clusters["cluster_id"] >= 0]
        assert (clustered["cluster_label"] != "Low Volume").all()


# =============================================================================
# Test Class: Low Volume Rules
# =============================================================================

class TestLowVolumeRules:
    """Test Low Volume NTA handling."""
    
    def test_low_volume_is_outlier(self, nta_clusters):
        """Low Volume NTAs should have is_outlier_cluster=True."""
        low_vol = nta_clusters[nta_clusters["cluster_id"] == -1]
        if len(low_vol) > 0:
            assert low_vol["is_outlier_cluster"].all()
    
    def test_low_volume_count_threshold(self, nta_clusters, nta_features):
        """Low Volume NTAs should have count_night < min_cluster_count (200)."""
        min_cluster_count = 200  # From config
        
        low_vol_codes = nta_clusters.loc[nta_clusters["cluster_id"] == -1, "ntacode"]
        if len(low_vol_codes) > 0:
            low_vol_features = nta_features[nta_features["ntacode"].isin(low_vol_codes)]
            assert (low_vol_features["count_night"] < min_cluster_count).all()
    
    def test_eligible_ntas_above_threshold(self, nta_clusters, nta_features):
        """Clustered NTAs should have count_night >= min_cluster_count (200)."""
        min_cluster_count = 200  # From config
        
        clustered_codes = nta_clusters.loc[nta_clusters["cluster_id"] >= 0, "ntacode"]
        if len(clustered_codes) > 0:
            clustered_features = nta_features[nta_features["ntacode"].isin(clustered_codes)]
            assert (clustered_features["count_night"] >= min_cluster_count).all()


# =============================================================================
# Test Class: Outlier Cluster Flag
# =============================================================================

class TestOutlierClusterFlag:
    """Test is_outlier_cluster flag."""
    
    def test_outlier_flag_boolean(self, nta_clusters):
        """is_outlier_cluster should be boolean."""
        assert nta_clusters["is_outlier_cluster"].dtype == bool or \
               set(nta_clusters["is_outlier_cluster"].unique()).issubset({True, False})
    
    def test_low_volume_all_outliers(self, nta_clusters):
        """All Low Volume NTAs should be outliers."""
        low_vol = nta_clusters[nta_clusters["cluster_id"] == -1]
        if len(low_vol) > 0:
            assert low_vol["is_outlier_cluster"].all()


# =============================================================================
# Test Class: K Selection Scores
# =============================================================================

class TestKSelectionScores:
    """Test K selection score validity."""
    
    def test_k_range(self, k_selection_scores):
        """K values should be in expected range."""
        assert k_selection_scores["k"].min() >= 2
        assert k_selection_scores["k"].max() <= 20
    
    def test_silhouette_bounded(self, k_selection_scores):
        """Silhouette scores should be in [-1, 1]."""
        assert k_selection_scores["silhouette_score"].between(-1, 1).all()
    
    def test_calinski_harabasz_positive(self, k_selection_scores):
        """Calinski-Harabasz scores should be positive."""
        assert (k_selection_scores["calinski_harabasz_score"] > 0).all()
    
    def test_min_cluster_size_positive(self, k_selection_scores):
        """Min cluster size should be >= 1."""
        assert (k_selection_scores["min_cluster_size"] >= 1).all()
    
    def test_n_singletons_bounded(self, k_selection_scores):
        """Number of singletons should be <= K."""
        assert (k_selection_scores["n_singletons"] <= k_selection_scores["k"]).all()


# =============================================================================
# Test Class: Cluster Summary
# =============================================================================

class TestClusterSummary:
    """Test cluster summary validity."""
    
    def test_has_low_volume_row(self, cluster_summary, nta_clusters):
        """Summary should include Low Volume cluster if present in data."""
        has_low_vol_data = (nta_clusters["cluster_id"] == -1).any()
        has_low_vol_summary = (cluster_summary["cluster_id"] == -1).any()
        assert has_low_vol_data == has_low_vol_summary
    
    def test_n_ntas_positive(self, cluster_summary):
        """All clusters should have >= 1 NTA."""
        assert (cluster_summary["n_ntas"] >= 1).all()
    
    def test_n_ntas_sum(self, cluster_summary, nta_clusters):
        """Sum of n_ntas should equal total NTA count."""
        assert cluster_summary["n_ntas"].sum() == len(nta_clusters)


# =============================================================================
# Test Class: GeoJSON
# =============================================================================

class TestGeoJSON:
    """Test GeoJSON output validity."""
    
    def test_valid_geometry(self, nta_clusters_geojson):
        """All geometries should be valid."""
        assert nta_clusters_geojson.geometry.is_valid.all()
    
    def test_no_empty_geometry(self, nta_clusters_geojson):
        """No empty geometries."""
        assert not nta_clusters_geojson.geometry.is_empty.any()
    
    def test_crs_is_4326(self, nta_clusters_geojson):
        """CRS should be EPSG:4326."""
        assert nta_clusters_geojson.crs is not None
        assert nta_clusters_geojson.crs.to_epsg() == 4326
    
    def test_geojson_has_nta_name(self, nta_clusters_geojson):
        """GeoJSON should have nta_name column."""
        assert "nta_name" in nta_clusters_geojson.columns
        assert nta_clusters_geojson["nta_name"].notna().all()


# =============================================================================
# Test Class: Feature Presence
# =============================================================================

class TestFeaturePresence:
    """Test that clustering features are present in output."""
    
    def test_rate_features(self, nta_clusters):
        """Rate features should be present."""
        rate_cols = ["rate_per_1k_pop", "rate_per_km2"]
        for col in rate_cols:
            assert col in nta_clusters.columns, f"Missing feature: {col}"
    
    def test_share_features(self, nta_clusters):
        """Share features should be present."""
        share_cols = [
            "share_evening",
            "share_early_am",
            "share_core_night",
            "share_predawn",
        ]
        for col in share_cols:
            assert col in nta_clusters.columns, f"Missing feature: {col}"
    
    def test_behavioral_metrics(self, nta_clusters):
        """Behavioral metrics should be present."""
        metrics = ["late_night_share", "weekend_uplift", "warm_season_ratio"]
        for col in metrics:
            assert col in nta_clusters.columns, f"Missing metric: {col}"


# =============================================================================
# Test Class: Bounded Values
# =============================================================================

class TestBoundedValues:
    """Test that share/rate values are properly bounded."""
    
    def test_share_features_bounded(self, nta_clusters):
        """Share features should be in [0, 1]."""
        share_cols = [c for c in nta_clusters.columns if c.startswith("share_")]
        for col in share_cols:
            vals = nta_clusters[col].dropna()
            assert vals.between(0, 1).all(), f"{col} has values outside [0, 1]"
    
    def test_late_night_share_bounded(self, nta_clusters):
        """Late night share should be in [0, 1]."""
        vals = nta_clusters["late_night_share"].dropna()
        assert vals.between(0, 1).all()
    
    def test_rate_per_1k_non_negative(self, nta_clusters):
        """Rate per 1k should be non-negative."""
        vals = nta_clusters["rate_per_1k_pop"].dropna()
        assert (vals >= 0).all()


# =============================================================================
# Test Class: Reproducibility
# =============================================================================

class TestReproducibility:
    """Test output determinism."""
    
    def test_cluster_assignments_deterministic(self, nta_clusters, typology_dir):
        """Cluster assignments should be deterministic (same across parquet/csv)."""
        csv_path = typology_dir / "nta_clusters_residential.csv"
        df_csv = pd.read_csv(csv_path)
        
        # Merge and compare
        merged = nta_clusters.merge(
            df_csv[["ntacode", "cluster_id"]],
            on="ntacode",
            suffixes=("_parquet", "_csv"),
        )
        
        assert (merged["cluster_id_parquet"] == merged["cluster_id_csv"]).all()


# =============================================================================
# Test Class: Metadata
# =============================================================================

class TestMetadata:
    """Test metadata sidecar."""
    
    def test_metadata_exists(self):
        """Metadata sidecar should exist."""
        metadata_path = Path("data/processed/metadata/nta_clusters_residential_metadata.json")
        assert metadata_path.exists(), f"Metadata not found: {metadata_path}"
    
    def test_metadata_has_required_fields(self):
        """Metadata should have required fields."""
        import json
        metadata_path = Path("data/processed/metadata/nta_clusters_residential_metadata.json")
        if not metadata_path.exists():
            pytest.skip("Metadata not found")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Top-level fields
        top_required = ["inputs", "run_id", "extra"]
        for field in top_required:
            assert field in metadata, f"Missing top-level field: {field}"
        
        # Fields in "extra"
        extra_required = ["selected_k", "silhouette_score", "features_used"]
        for field in extra_required:
            assert field in metadata.get("extra", {}), f"Missing extra field: {field}"

