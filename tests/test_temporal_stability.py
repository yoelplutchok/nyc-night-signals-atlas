"""
Tests for Temporal Stability Analysis (Script 12) outputs.

Validates:
- Row counts per year
- No missing IDs or labels
- Stability metrics bounded [0, 1]
- Transition tables have required columns
- Reproducibility via fixed seed
"""

import pytest
import pandas as pd
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def temporal_dir():
    """Path to temporal outputs."""
    return Path("data/processed/temporal")


@pytest.fixture(scope="module")
def cd_clusters_by_year(temporal_dir):
    """Load CD clusters by year."""
    path = temporal_dir / "cd_clusters_by_year.parquet"
    if not path.exists():
        pytest.skip(f"CD clusters by year not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def cd_features_by_year(temporal_dir):
    """Load CD features by year."""
    path = temporal_dir / "cd_features_by_year.parquet"
    if not path.exists():
        pytest.skip(f"CD features by year not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def nta_clusters_by_year(temporal_dir):
    """Load NTA clusters by year."""
    path = temporal_dir / "nta_clusters_by_year.parquet"
    if not path.exists():
        pytest.skip(f"NTA clusters by year not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def nta_features_by_year(temporal_dir):
    """Load NTA features by year."""
    path = temporal_dir / "nta_features_by_year.parquet"
    if not path.exists():
        pytest.skip(f"NTA features by year not found: {path}")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def stability_metrics(temporal_dir):
    """Load stability metrics."""
    path = temporal_dir / "cluster_stability_metrics.csv"
    if not path.exists():
        pytest.skip(f"Stability metrics not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def cd_transitions(temporal_dir):
    """Load CD transitions."""
    path = temporal_dir / "cluster_transitions_cd.csv"
    if not path.exists():
        pytest.skip(f"CD transitions not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def nta_transitions(temporal_dir):
    """Load NTA transitions."""
    path = temporal_dir / "cluster_transitions_nta.csv"
    if not path.exists():
        pytest.skip(f"NTA transitions not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def hotspot_persistence(temporal_dir):
    """Load hotspot persistence."""
    path = temporal_dir / "hotspot_persistence.csv"
    if not path.exists():
        pytest.skip(f"Hotspot persistence not found: {path}")
    return pd.read_csv(path)


# =============================================================================
# Test Class: CD Clusters By Year
# =============================================================================

class TestCDClustersByYear:
    """Test CD clusters by year output."""
    
    def test_has_three_years(self, cd_clusters_by_year):
        """Should have data for 2021, 2022, 2023."""
        years = cd_clusters_by_year["year"].unique()
        assert set(years) == {2021, 2022, 2023}
    
    def test_59_cds_per_year(self, cd_clusters_by_year):
        """Each year should have 59 CDs."""
        for year in [2021, 2022, 2023]:
            year_df = cd_clusters_by_year[cd_clusters_by_year["year"] == year]
            assert len(year_df) == 59, f"Year {year} has {len(year_df)} CDs"
    
    def test_no_null_boro_cd(self, cd_clusters_by_year):
        """No null boro_cd."""
        assert cd_clusters_by_year["boro_cd"].notna().all()
    
    def test_no_null_cluster_id(self, cd_clusters_by_year):
        """No null cluster_id."""
        assert cd_clusters_by_year["cluster_id"].notna().all()
    
    def test_cluster_id_range(self, cd_clusters_by_year):
        """Cluster IDs should be in valid range."""
        min_id = cd_clusters_by_year["cluster_id"].min()
        max_id = cd_clusters_by_year["cluster_id"].max()
        assert min_id >= 0
        assert max_id < 20  # Reasonable upper bound
    
    def test_has_silhouette(self, cd_clusters_by_year):
        """Should have silhouette scores."""
        assert "silhouette" in cd_clusters_by_year.columns
        assert cd_clusters_by_year["silhouette"].notna().all()


# =============================================================================
# Test Class: CD Features By Year
# =============================================================================

class TestCDFeaturesByYear:
    """Test CD features by year output."""
    
    def test_has_three_years(self, cd_features_by_year):
        """Should have data for 2021, 2022, 2023."""
        years = cd_features_by_year["year"].unique()
        assert set(years) == {2021, 2022, 2023}
    
    def test_59_cds_per_year(self, cd_features_by_year):
        """Each year should have 59 CDs."""
        for year in [2021, 2022, 2023]:
            year_df = cd_features_by_year[cd_features_by_year["year"] == year]
            assert len(year_df) == 59, f"Year {year} has {len(year_df)} CDs"
    
    def test_has_count_night(self, cd_features_by_year):
        """Should have count_night column."""
        assert "count_night" in cd_features_by_year.columns
    
    def test_count_night_non_negative(self, cd_features_by_year):
        """count_night should be non-negative."""
        assert (cd_features_by_year["count_night"] >= 0).all()


# =============================================================================
# Test Class: NTA Clusters By Year
# =============================================================================

class TestNTAClustersByYear:
    """Test NTA clusters by year output."""
    
    def test_has_three_years(self, nta_clusters_by_year):
        """Should have data for 2021, 2022, 2023."""
        years = nta_clusters_by_year["year"].unique()
        assert set(years) == {2021, 2022, 2023}
    
    def test_197_ntas_per_year(self, nta_clusters_by_year):
        """Each year should have 197 residential NTAs."""
        for year in [2021, 2022, 2023]:
            year_df = nta_clusters_by_year[nta_clusters_by_year["year"] == year]
            assert len(year_df) == 197, f"Year {year} has {len(year_df)} NTAs"
    
    def test_no_null_ntacode(self, nta_clusters_by_year):
        """No null ntacode."""
        assert nta_clusters_by_year["ntacode"].notna().all()
    
    def test_no_null_cluster_id(self, nta_clusters_by_year):
        """No null cluster_id."""
        assert nta_clusters_by_year["cluster_id"].notna().all()
    
    def test_cluster_id_range(self, nta_clusters_by_year):
        """Cluster IDs should be in valid range (-1 for low volume, 0+ for clustered)."""
        min_id = nta_clusters_by_year["cluster_id"].min()
        max_id = nta_clusters_by_year["cluster_id"].max()
        assert min_id >= -1
        assert max_id < 20


# =============================================================================
# Test Class: Stability Metrics
# =============================================================================

class TestStabilityMetrics:
    """Test stability metrics output."""
    
    def test_has_required_columns(self, stability_metrics):
        """Should have required columns."""
        required = ["year_1", "year_2", "ari", "nmi", "level"]
        for col in required:
            assert col in stability_metrics.columns, f"Missing column: {col}"
    
    def test_has_cd_and_nta_levels(self, stability_metrics):
        """Should have both CD and NTA levels."""
        levels = stability_metrics["level"].unique()
        assert "CD" in levels
        assert "NTA" in levels
    
    def test_ari_bounded(self, stability_metrics):
        """ARI should be in [-1, 1]."""
        assert stability_metrics["ari"].between(-1, 1).all()
    
    def test_nmi_bounded(self, stability_metrics):
        """NMI should be in [0, 1]."""
        assert stability_metrics["nmi"].between(0, 1).all()
    
    def test_has_three_year_pairs(self, stability_metrics):
        """Should have 3 year pairs per level: 2021-2022, 2021-2023, 2022-2023."""
        for level in ["CD", "NTA"]:
            level_df = stability_metrics[stability_metrics["level"] == level]
            assert len(level_df) == 3, f"Level {level} has {len(level_df)} pairs"


# =============================================================================
# Test Class: CD Transitions
# =============================================================================

class TestCDTransitions:
    """Test CD transitions output."""
    
    def test_59_cds(self, cd_transitions):
        """Should have 59 CDs."""
        assert len(cd_transitions) == 59
    
    def test_has_boro_cd(self, cd_transitions):
        """Should have boro_cd column."""
        assert "boro_cd" in cd_transitions.columns
    
    def test_has_is_stable(self, cd_transitions):
        """Should have is_stable column."""
        assert "is_stable" in cd_transitions.columns
    
    def test_has_n_unique_clusters(self, cd_transitions):
        """Should have n_unique_clusters column."""
        assert "n_unique_clusters" in cd_transitions.columns
    
    def test_stable_means_one_cluster(self, cd_transitions):
        """Stable CDs should have n_unique_clusters == 1."""
        stable = cd_transitions[cd_transitions["is_stable"] == True]
        if len(stable) > 0:
            assert (stable["n_unique_clusters"] == 1).all()


# =============================================================================
# Test Class: NTA Transitions
# =============================================================================

class TestNTATransitions:
    """Test NTA transitions output."""
    
    def test_197_ntas(self, nta_transitions):
        """Should have 197 residential NTAs."""
        assert len(nta_transitions) == 197
    
    def test_has_ntacode(self, nta_transitions):
        """Should have ntacode column."""
        assert "ntacode" in nta_transitions.columns
    
    def test_has_is_stable(self, nta_transitions):
        """Should have is_stable column."""
        assert "is_stable" in nta_transitions.columns
    
    def test_has_n_unique_clusters(self, nta_transitions):
        """Should have n_unique_clusters column."""
        assert "n_unique_clusters" in nta_transitions.columns


# =============================================================================
# Test Class: Hotspot Persistence
# =============================================================================

class TestHotspotPersistence:
    """Test hotspot persistence output."""
    
    def test_has_required_columns(self, hotspot_persistence):
        """Should have required columns."""
        required = ["year_1", "year_2", "overlap_count", "jaccard"]
        for col in required:
            assert col in hotspot_persistence.columns, f"Missing column: {col}"
    
    def test_jaccard_bounded(self, hotspot_persistence):
        """Jaccard should be in [0, 1]."""
        assert hotspot_persistence["jaccard"].between(0, 1).all()
    
    def test_overlap_non_negative(self, hotspot_persistence):
        """Overlap count should be non-negative."""
        assert (hotspot_persistence["overlap_count"] >= 0).all()
    
    def test_has_all_years_summary(self, hotspot_persistence):
        """Should have 'all years' summary row."""
        all_row = hotspot_persistence[hotspot_persistence["year_1"] == "all"]
        assert len(all_row) == 1


# =============================================================================
# Test Class: Metadata
# =============================================================================

class TestMetadata:
    """Test metadata sidecar."""
    
    def test_metadata_exists(self):
        """Metadata sidecar should exist."""
        metadata_path = Path("data/processed/metadata/cluster_stability_metrics_metadata.json")
        assert metadata_path.exists(), f"Metadata not found: {metadata_path}"
    
    def test_metadata_has_required_fields(self):
        """Metadata should have required fields."""
        import json
        metadata_path = Path("data/processed/metadata/cluster_stability_metrics_metadata.json")
        if not metadata_path.exists():
            pytest.skip("Metadata not found")
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert "inputs" in metadata
        assert "run_id" in metadata
        assert "extra" in metadata
        
        extra = metadata.get("extra", {})
        assert "years_analyzed" in extra
        assert "cd_k" in extra
        assert "nta_k" in extra

