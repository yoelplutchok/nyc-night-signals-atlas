"""
Tests for 311 Night Features output.

Per NYC_Night_Signals_Plan.md:
- 59 rows (one per CD)
- Non-null cd_label
- Type shares sum ≈ 1
- Bin shares sum ≈ 1
- Coverage/assignment stats
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from sleep_esi.paths import PROCESSED_DIR

ATLAS_DIR = PROCESSED_DIR / "atlas"


class Test311FeaturesExists:
    """Tests that required output files exist."""

    def test_parquet_exists(self):
        """311_cd_features.parquet must exist."""
        path = ATLAS_DIR / "311_cd_features.parquet"
        assert path.exists(), f"Missing required file: {path}"

    def test_csv_exists(self):
        """311_cd_features.csv must exist."""
        path = ATLAS_DIR / "311_cd_features.csv"
        assert path.exists(), f"Missing required file: {path}"

    def test_geojson_exists(self):
        """311_cd_features.geojson must exist."""
        path = ATLAS_DIR / "311_cd_features.geojson"
        assert path.exists(), f"Missing required file: {path}"


class Test311FeaturesSchema:
    """Tests for 311 features schema correctness."""

    @pytest.fixture
    def features(self):
        """Load the features table."""
        return pd.read_parquet(ATLAS_DIR / "311_cd_features.parquet")

    def test_row_count(self, features):
        """Must have exactly 59 CDs."""
        assert len(features) == 59, f"Expected 59 rows, got {len(features)}"

    def test_boro_cd_unique(self, features):
        """boro_cd must be unique."""
        assert features["boro_cd"].nunique() == 59, "Duplicate boro_cd values found"

    def test_cd_label_present(self, features):
        """cd_label column must be present."""
        assert "cd_label" in features.columns, "Missing cd_label column"

    def test_cd_label_not_null(self, features):
        """cd_label must not have null values."""
        null_count = features["cd_label"].isna().sum()
        assert null_count == 0, f"Found {null_count} null cd_label values"

    def test_required_columns_present(self, features):
        """All required columns must be present."""
        required = [
            "boro_cd",
            "cd_label",
            "count_night",
            "rate_per_1k_pop",
            "rate_per_km2",
            "late_night_share",
            "weekend_uplift",
            "warm_season_ratio",
        ]
        for col in required:
            assert col in features.columns, f"Missing required column: {col}"


class Test311FeatureShares:
    """Tests for share columns summing to ~1."""

    @pytest.fixture
    def features(self):
        """Load the features table."""
        return pd.read_parquet(ATLAS_DIR / "311_cd_features.parquet")

    def test_type_shares_sum_to_one(self, features):
        """Complaint type shares should sum to ~1 for CDs with complaints."""
        type_share_cols = [c for c in features.columns if c.startswith("share_noise")]
        
        if not type_share_cols:
            pytest.skip("No type share columns found")
        
        # Only check CDs with complaints
        has_complaints = features["count_night"] > 0
        type_share_sum = features.loc[has_complaints, type_share_cols].sum(axis=1)
        
        # Allow some tolerance for floating point
        bad_sums = (type_share_sum < 0.99) | (type_share_sum > 1.01)
        assert not bad_sums.any(), \
            f"Type shares don't sum to ~1 for {bad_sums.sum()} CDs. " \
            f"Range: [{type_share_sum.min():.3f}, {type_share_sum.max():.3f}]"

    def test_bin_shares_sum_to_one(self, features):
        """Time-of-night bin shares should sum to ~1 for CDs with complaints."""
        # The 4 time bins should sum to 1
        bin_cols = ["share_evening", "share_early_am", "share_core_night", "share_predawn"]
        bin_cols = [c for c in bin_cols if c in features.columns]
        
        if not bin_cols:
            pytest.skip("No bin share columns found")
        
        # Only check CDs with complaints
        has_complaints = features["count_night"] > 0
        bin_share_sum = features.loc[has_complaints, bin_cols].sum(axis=1)
        
        # Allow some tolerance
        bad_sums = (bin_share_sum < 0.99) | (bin_share_sum > 1.01)
        assert not bad_sums.any(), \
            f"Bin shares don't sum to ~1 for {bad_sums.sum()} CDs. " \
            f"Range: [{bin_share_sum.min():.3f}, {bin_share_sum.max():.3f}]"

    def test_shares_non_negative(self, features):
        """All share columns should be non-negative."""
        share_cols = [c for c in features.columns if c.startswith("share_")]
        
        for col in share_cols:
            min_val = features[col].min()
            assert min_val >= 0, f"Negative values in {col}: min={min_val}"

    def test_shares_max_one(self, features):
        """All share columns should be <= 1."""
        share_cols = [c for c in features.columns if c.startswith("share_")]
        
        for col in share_cols:
            max_val = features[col].max()
            assert max_val <= 1.01, f"Values > 1 in {col}: max={max_val}"


class Test311FeatureCoverage:
    """Tests for data coverage and assignment."""

    @pytest.fixture
    def features(self):
        """Load the features table."""
        return pd.read_parquet(ATLAS_DIR / "311_cd_features.parquet")

    def test_all_cds_have_counts(self, features):
        """All CDs should have a count value (even if 0)."""
        assert features["count_night"].isna().sum() == 0, "Found null count values"

    def test_positive_total_complaints(self, features):
        """Total complaints should be positive."""
        total = features["count_night"].sum()
        assert total > 0, f"No complaints found: total={total}"

    def test_no_negative_counts(self, features):
        """No negative count values."""
        assert (features["count_night"] >= 0).all(), "Found negative counts"

    def test_population_present(self, features):
        """Population should be present for all CDs."""
        assert features["population"].isna().sum() == 0, "Missing population values"
        assert (features["population"] > 0).all(), "Found zero or negative population"

    def test_area_present(self, features):
        """Area should be present for all CDs."""
        assert features["area_km2"].isna().sum() == 0, "Missing area values"
        assert (features["area_km2"] > 0).all(), "Found zero or negative area"


class Test311FeatureValues:
    """Tests for reasonable feature values."""

    @pytest.fixture
    def features(self):
        """Load the features table."""
        return pd.read_parquet(ATLAS_DIR / "311_cd_features.parquet")

    def test_weekend_uplift_reasonable(self, features):
        """Weekend uplift should be in reasonable range."""
        valid = features["weekend_uplift"].notna()
        uplift = features.loc[valid, "weekend_uplift"]
        
        # Weekend uplift typically between 0.5 and 3.0
        assert uplift.min() > 0, f"Weekend uplift too low: {uplift.min()}"
        assert uplift.max() < 10, f"Weekend uplift too high: {uplift.max()}"

    def test_warm_season_ratio_reasonable(self, features):
        """Warm season ratio should be in reasonable range."""
        valid = features["warm_season_ratio"].notna()
        ratio = features.loc[valid, "warm_season_ratio"]
        
        # Warm season ratio typically between 0.5 and 5.0
        assert ratio.min() > 0, f"Warm season ratio too low: {ratio.min()}"
        assert ratio.max() < 20, f"Warm season ratio too high: {ratio.max()}"

    def test_late_night_share_reasonable(self, features):
        """Late-night share should be between 0 and 1."""
        share = features["late_night_share"]
        assert (share >= 0).all(), f"Negative late-night share"
        assert (share <= 1).all(), f"Late-night share > 1"


@pytest.mark.smoke
class Test311FeaturesSmoke:
    """Quick smoke tests for 311 features."""

    def test_can_load_features(self):
        """Basic load should work."""
        df = pd.read_parquet(ATLAS_DIR / "311_cd_features.parquet")
        assert len(df) > 0

    def test_has_cd_labels(self):
        """Should have cd_label column with values."""
        df = pd.read_parquet(ATLAS_DIR / "311_cd_features.parquet")
        assert "cd_label" in df.columns
        assert df["cd_label"].notna().all()

