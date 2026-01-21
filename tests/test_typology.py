"""
Tests for Night Signatures Typology (Script 05 outputs).

Per NYC_Night_Signals_Plan.md:
- 59 rows, no null cluster_id
- cluster_id in [0..K-1]
- Cluster sizes not degenerate
- Reproducibility check
- cd_label present and non-null
"""

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
TYPOLOGY_DIR = PROCESSED_DIR / "typology"
CONFIG_DIR = Path(__file__).parent.parent / "configs"

CLUSTERS_PARQUET = TYPOLOGY_DIR / "cd_clusters.parquet"
CLUSTERS_CSV = TYPOLOGY_DIR / "cd_clusters.csv"
SUMMARY_CSV = TYPOLOGY_DIR / "cluster_summary.csv"
K_SCORES_CSV = TYPOLOGY_DIR / "k_selection_scores.csv"
FEATURE_MATRIX_CSV = TYPOLOGY_DIR / "feature_matrix_standardized.csv"
CLUSTERS_GEOJSON = TYPOLOGY_DIR / "cd_clusters.geojson"


class TestTypologyOutputsExist:
    """Tests for output file existence."""

    def test_clusters_parquet_exists(self):
        assert CLUSTERS_PARQUET.exists(), f"Missing: {CLUSTERS_PARQUET}"

    def test_clusters_csv_exists(self):
        assert CLUSTERS_CSV.exists(), f"Missing: {CLUSTERS_CSV}"

    def test_summary_csv_exists(self):
        assert SUMMARY_CSV.exists(), f"Missing: {SUMMARY_CSV}"

    def test_k_scores_csv_exists(self):
        assert K_SCORES_CSV.exists(), f"Missing: {K_SCORES_CSV}"

    def test_feature_matrix_csv_exists(self):
        assert FEATURE_MATRIX_CSV.exists(), f"Missing: {FEATURE_MATRIX_CSV}"

    def test_geojson_exists(self):
        assert CLUSTERS_GEOJSON.exists(), f"Missing: {CLUSTERS_GEOJSON}"


@pytest.fixture
def df_clusters():
    """Load cluster assignments."""
    if not CLUSTERS_PARQUET.exists():
        pytest.skip(f"Missing: {CLUSTERS_PARQUET}")
    return pd.read_parquet(CLUSTERS_PARQUET)


@pytest.fixture
def df_summary():
    """Load cluster summary."""
    if not SUMMARY_CSV.exists():
        pytest.skip(f"Missing: {SUMMARY_CSV}")
    return pd.read_csv(SUMMARY_CSV)


@pytest.fixture
def df_scores():
    """Load K selection scores."""
    if not K_SCORES_CSV.exists():
        pytest.skip(f"Missing: {K_SCORES_CSV}")
    return pd.read_csv(K_SCORES_CSV)


@pytest.fixture
def df_feature_matrix():
    """Load standardized feature matrix."""
    if not FEATURE_MATRIX_CSV.exists():
        pytest.skip(f"Missing: {FEATURE_MATRIX_CSV}")
    return pd.read_csv(FEATURE_MATRIX_CSV)


@pytest.fixture
def gdf_clusters():
    """Load GeoJSON clusters."""
    if not CLUSTERS_GEOJSON.exists():
        pytest.skip(f"Missing: {CLUSTERS_GEOJSON}")
    return gpd.read_file(CLUSTERS_GEOJSON)


class TestClusterAssignments:
    """Tests for cd_clusters.parquet."""

    def test_row_count_is_59(self, df_clusters):
        """Must have exactly 59 CDs."""
        assert len(df_clusters) == 59, f"Expected 59 rows, got {len(df_clusters)}"

    def test_boro_cd_unique(self, df_clusters):
        """boro_cd must be unique."""
        assert df_clusters["boro_cd"].is_unique

    def test_no_null_cluster_id(self, df_clusters):
        """cluster_id must have no nulls."""
        assert df_clusters["cluster_id"].notna().all(), "Found null cluster_id values"

    def test_cluster_id_non_negative(self, df_clusters):
        """cluster_id must be >= 0."""
        assert (df_clusters["cluster_id"] >= 0).all()

    def test_cluster_id_in_valid_range(self, df_clusters):
        """cluster_id must be in [0..K-1] where K is the number of unique clusters."""
        k = df_clusters["cluster_id"].nunique()
        max_id = df_clusters["cluster_id"].max()
        assert max_id == k - 1, f"Max cluster_id ({max_id}) != K-1 ({k-1})"

    def test_cd_label_present(self, df_clusters):
        """cd_label column must be present."""
        assert "cd_label" in df_clusters.columns

    def test_cd_label_no_nulls(self, df_clusters):
        """cd_label must have no nulls."""
        assert df_clusters["cd_label"].notna().all(), "Found null cd_label values"

    def test_cluster_label_present(self, df_clusters):
        """cluster_label must be present."""
        assert "cluster_label" in df_clusters.columns

    def test_cluster_label_no_nulls(self, df_clusters):
        """cluster_label must have no nulls."""
        assert df_clusters["cluster_label"].notna().all()

    def test_is_outlier_cluster_present(self, df_clusters):
        """is_outlier_cluster must be present."""
        assert "is_outlier_cluster" in df_clusters.columns

    def test_is_outlier_cluster_boolean(self, df_clusters):
        """is_outlier_cluster must be boolean."""
        assert df_clusters["is_outlier_cluster"].dtype == bool

    def test_cd_short_present(self, df_clusters):
        """cd_short column must be present."""
        assert "cd_short" in df_clusters.columns


class TestClusterSizes:
    """Tests for cluster size distribution."""

    def test_no_empty_clusters(self, df_clusters):
        """No cluster should have 0 members."""
        sizes = df_clusters["cluster_id"].value_counts()
        assert (sizes > 0).all(), "Found cluster with 0 members"

    def test_no_singleton_clusters(self, df_clusters):
        """Warn if any cluster has only 1 member (but don't fail)."""
        sizes = df_clusters["cluster_id"].value_counts()
        singletons = (sizes == 1).sum()
        if singletons > 0:
            pytest.warns(UserWarning, match=f"Found {singletons} singleton clusters")

    def test_all_cds_assigned(self, df_clusters):
        """All 59 CDs must be assigned to some cluster."""
        assigned = df_clusters["cluster_id"].notna().sum()
        assert assigned == 59


class TestKScores:
    """Tests for K selection scores."""

    def test_k_range_correct(self, df_scores):
        """K values should be in expected range [5..9]."""
        expected_ks = {5, 6, 7, 8, 9}
        actual_ks = set(df_scores["k"].values)
        assert actual_ks == expected_ks, f"Expected {expected_ks}, got {actual_ks}"

    def test_silhouette_positive(self, df_scores):
        """All silhouette scores should be positive."""
        assert (df_scores["silhouette_score"] > 0).all()

    def test_calinski_harabasz_positive(self, df_scores):
        """All Calinski-Harabasz scores should be positive."""
        assert (df_scores["calinski_harabasz_score"] > 0).all()

    def test_best_silhouette_reasonable(self, df_scores):
        """Best silhouette should be at least 0.1 (not random)."""
        best_sil = df_scores["silhouette_score"].max()
        assert best_sil >= 0.1, f"Best silhouette {best_sil} < 0.1 threshold"


class TestClusterSummary:
    """Tests for cluster_summary.csv."""

    def test_summary_has_n_cds(self, df_summary):
        """Summary must have n_cds column."""
        assert "n_cds" in df_summary.columns

    def test_summary_n_cds_sums_to_59(self, df_summary):
        """Total n_cds across clusters must equal 59."""
        total = df_summary["n_cds"].sum()
        assert total == 59, f"n_cds sum is {total}, expected 59"

    def test_summary_has_cluster_id(self, df_summary):
        """Summary must have cluster_id column."""
        assert "cluster_id" in df_summary.columns


class TestFeatureMatrix:
    """Tests for standardized feature matrix."""

    def test_feature_matrix_59_rows(self, df_feature_matrix):
        """Feature matrix must have 59 rows."""
        assert len(df_feature_matrix) == 59

    def test_feature_matrix_has_boro_cd(self, df_feature_matrix):
        """Feature matrix must have boro_cd column."""
        assert "boro_cd" in df_feature_matrix.columns

    def test_standardized_means_near_zero(self, df_feature_matrix):
        """Standardized feature means should be near 0."""
        feature_cols = [c for c in df_feature_matrix.columns if c != "boro_cd"]
        for col in feature_cols:
            mean = df_feature_matrix[col].mean()
            assert abs(mean) < 0.1, f"Feature {col} mean is {mean}, expected ~0"

    def test_standardized_stds_near_one(self, df_feature_matrix):
        """Standardized feature stds should be near 1."""
        feature_cols = [c for c in df_feature_matrix.columns if c != "boro_cd"]
        for col in feature_cols:
            std = df_feature_matrix[col].std()
            assert 0.9 < std < 1.1, f"Feature {col} std is {std}, expected ~1"


class TestGeoJSON:
    """Tests for map-ready GeoJSON."""

    def test_geojson_59_rows(self, gdf_clusters):
        """GeoJSON must have 59 features."""
        assert len(gdf_clusters) == 59

    def test_geojson_has_geometry(self, gdf_clusters):
        """GeoJSON must have valid geometry."""
        assert gdf_clusters.geometry.notna().all()

    def test_geojson_has_cluster_id(self, gdf_clusters):
        """GeoJSON must have cluster_id column."""
        assert "cluster_id" in gdf_clusters.columns

    def test_geojson_has_cd_label(self, gdf_clusters):
        """GeoJSON must have cd_label column."""
        assert "cd_label" in gdf_clusters.columns

    def test_geojson_has_cluster_label(self, gdf_clusters):
        """GeoJSON must have cluster_label column."""
        assert "cluster_label" in gdf_clusters.columns

    def test_geojson_has_is_outlier_cluster(self, gdf_clusters):
        """GeoJSON must have is_outlier_cluster column."""
        assert "is_outlier_cluster" in gdf_clusters.columns


class TestReproducibility:
    """Tests for clustering reproducibility."""

    def test_clustering_is_deterministic(self, df_feature_matrix, df_clusters, df_scores):
        """Re-running K-Means with same seed should give same results."""
        # Get the selected K and seed from scores
        best_k = df_scores.loc[df_scores["silhouette_score"].idxmax(), "k"]
        best_k = int(best_k)
        
        # Default seed from config
        random_seed = 12345
        
        # Extract feature matrix (exclude boro_cd)
        feature_cols = [c for c in df_feature_matrix.columns if c != "boro_cd"]
        X = df_feature_matrix[feature_cols].values
        
        # Fit K-Means
        kmeans = KMeans(
            n_clusters=best_k,
            random_state=random_seed,
            n_init=10,
            max_iter=300,
        )
        new_labels = kmeans.fit_predict(X)
        
        # Compare with stored labels
        original_labels = df_clusters.sort_values("boro_cd")["cluster_id"].values
        df_fm_sorted = df_feature_matrix.sort_values("boro_cd")
        X_sorted = df_fm_sorted[feature_cols].values
        
        kmeans2 = KMeans(
            n_clusters=best_k,
            random_state=random_seed,
            n_init=10,
            max_iter=300,
        )
        new_labels_sorted = kmeans2.fit_predict(X_sorted)
        
        # Labels should match
        assert np.array_equal(original_labels, new_labels_sorted), (
            "Clustering reproducibility check failed"
        )

