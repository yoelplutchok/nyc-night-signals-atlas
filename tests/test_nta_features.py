"""
Tests for NTA-level 311 Night Features (Script 10 outputs).

Per NYC_Night_Signals_Plan.md Script 10:
- Feature schema mirrors Script 03
- TWO outputs: residential only (197) and all NTAs (262)
- Includes is_residential, ntatype_label, nta_name
"""

import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
ATLAS_DIR = PROCESSED_DIR / "atlas"

OUTPUT_PARQUET_ALL = ATLAS_DIR / "311_nta_features.parquet"
OUTPUT_CSV_ALL = ATLAS_DIR / "311_nta_features.csv"
OUTPUT_GEOJSON_ALL = ATLAS_DIR / "311_nta_features.geojson"
OUTPUT_PARQUET_RES = ATLAS_DIR / "311_nta_features_residential.parquet"
OUTPUT_CSV_RES = ATLAS_DIR / "311_nta_features_residential.csv"
OUTPUT_GEOJSON_RES = ATLAS_DIR / "311_nta_features_residential.geojson"


class TestOutputFilesExist:
    """Tests for output file existence."""

    def test_all_parquet_exists(self):
        assert OUTPUT_PARQUET_ALL.exists(), f"Missing: {OUTPUT_PARQUET_ALL}"

    def test_all_csv_exists(self):
        assert OUTPUT_CSV_ALL.exists(), f"Missing: {OUTPUT_CSV_ALL}"

    def test_all_geojson_exists(self):
        assert OUTPUT_GEOJSON_ALL.exists(), f"Missing: {OUTPUT_GEOJSON_ALL}"

    def test_res_parquet_exists(self):
        assert OUTPUT_PARQUET_RES.exists(), f"Missing: {OUTPUT_PARQUET_RES}"

    def test_res_csv_exists(self):
        assert OUTPUT_CSV_RES.exists(), f"Missing: {OUTPUT_CSV_RES}"

    def test_res_geojson_exists(self):
        assert OUTPUT_GEOJSON_RES.exists(), f"Missing: {OUTPUT_GEOJSON_RES}"


@pytest.fixture
def df_all():
    """Load all NTA features."""
    if not OUTPUT_PARQUET_ALL.exists():
        pytest.skip(f"Missing: {OUTPUT_PARQUET_ALL}")
    return pd.read_parquet(OUTPUT_PARQUET_ALL)


@pytest.fixture
def df_residential():
    """Load residential NTA features."""
    if not OUTPUT_PARQUET_RES.exists():
        pytest.skip(f"Missing: {OUTPUT_PARQUET_RES}")
    return pd.read_parquet(OUTPUT_PARQUET_RES)


@pytest.fixture
def gdf_all():
    """Load all NTA features GeoJSON."""
    if not OUTPUT_GEOJSON_ALL.exists():
        pytest.skip(f"Missing: {OUTPUT_GEOJSON_ALL}")
    return gpd.read_file(OUTPUT_GEOJSON_ALL)


@pytest.fixture
def gdf_residential():
    """Load residential NTA features GeoJSON."""
    if not OUTPUT_GEOJSON_RES.exists():
        pytest.skip(f"Missing: {OUTPUT_GEOJSON_RES}")
    return gpd.read_file(OUTPUT_GEOJSON_RES)


class TestAllNTARowCount:
    """Tests for all NTA output row counts."""

    def test_all_nta_count_262(self, df_all):
        """All NTAs should have 262 rows."""
        assert len(df_all) == 262, f"Expected 262 rows, got {len(df_all)}"

    def test_residential_nta_count_197(self, df_residential):
        """Residential NTAs should have 197 rows."""
        assert len(df_residential) == 197, f"Expected 197 rows, got {len(df_residential)}"


class TestRequiredColumns:
    """Tests for required columns."""

    def test_has_ntacode(self, df_all):
        assert "ntacode" in df_all.columns

    def test_ntacode_unique(self, df_all):
        assert df_all["ntacode"].is_unique

    def test_has_nta_name(self, df_all):
        assert "nta_name" in df_all.columns

    def test_nta_name_no_nulls(self, df_all):
        assert df_all["nta_name"].notna().all()

    def test_has_is_residential(self, df_all):
        assert "is_residential" in df_all.columns

    def test_has_ntatype_label(self, df_all):
        assert "ntatype_label" in df_all.columns

    def test_has_borough_name(self, df_all):
        assert "borough_name" in df_all.columns


class TestFeatureColumns:
    """Tests for expected feature columns."""

    def test_has_count_night(self, df_all):
        assert "count_night" in df_all.columns

    def test_has_rate_per_1k_pop(self, df_all):
        assert "rate_per_1k_pop" in df_all.columns

    def test_has_rate_per_km2(self, df_all):
        assert "rate_per_km2" in df_all.columns

    def test_has_late_night_share(self, df_all):
        assert "late_night_share" in df_all.columns

    def test_has_weekend_uplift(self, df_all):
        assert "weekend_uplift" in df_all.columns

    def test_has_warm_season_ratio(self, df_all):
        assert "warm_season_ratio" in df_all.columns

    def test_has_time_bin_shares(self, df_all):
        """Should have all 4 time-of-night bin shares."""
        expected_bins = ["share_evening", "share_early_am", "share_core_night", "share_predawn"]
        for col in expected_bins:
            assert col in df_all.columns, f"Missing {col}"


class TestTypeSharesSum:
    """Tests for complaint type shares summing to 1."""

    def test_type_shares_sum_approx_1(self, df_all):
        """Type shares should sum to ≈1 for NTAs with complaints."""
        type_share_cols = [c for c in df_all.columns if c.startswith("share_noise")]
        if not type_share_cols:
            pytest.skip("No type share columns found")
        
        has_complaints = df_all["count_night"] > 0
        if not has_complaints.any():
            pytest.skip("No NTAs with complaints")
        
        type_share_sum = df_all.loc[has_complaints, type_share_cols].sum(axis=1)
        
        # Check mean is close to 1
        assert 0.95 < type_share_sum.mean() < 1.05, f"Mean type share sum = {type_share_sum.mean()}"
        
        # Check individual sums
        bad = (type_share_sum < 0.95) | (type_share_sum > 1.05)
        assert bad.sum() < len(type_share_sum) * 0.05, f"{bad.sum()} NTAs have bad type share sums"


class TestBinSharesSum:
    """Tests for time-of-night bin shares summing to 1."""

    def test_bin_shares_sum_approx_1(self, df_all):
        """Bin shares should sum to ≈1 for NTAs with complaints."""
        bin_cols = ["share_evening", "share_early_am", "share_core_night", "share_predawn"]
        bin_cols = [c for c in bin_cols if c in df_all.columns]
        if not bin_cols:
            pytest.skip("No bin share columns found")
        
        has_complaints = df_all["count_night"] > 0
        if not has_complaints.any():
            pytest.skip("No NTAs with complaints")
        
        bin_share_sum = df_all.loc[has_complaints, bin_cols].sum(axis=1)
        
        # Check mean is close to 1
        assert 0.95 < bin_share_sum.mean() < 1.05, f"Mean bin share sum = {bin_share_sum.mean()}"
        
        # Check individual sums
        bad = (bin_share_sum < 0.95) | (bin_share_sum > 1.05)
        assert bad.sum() < len(bin_share_sum) * 0.05, f"{bad.sum()} NTAs have bad bin share sums"


class TestBoundedValues:
    """Tests for value ranges."""

    def test_count_night_non_negative(self, df_all):
        assert (df_all["count_night"] >= 0).all()

    def test_late_night_share_bounded(self, df_all):
        """late_night_share should be in [0, 1]."""
        assert (df_all["late_night_share"] >= 0).all()
        assert (df_all["late_night_share"] <= 1).all()

    def test_share_columns_bounded(self, df_all):
        """All share columns should be in [0, 1]."""
        share_cols = [c for c in df_all.columns if c.startswith("share_")]
        for col in share_cols:
            assert (df_all[col] >= 0).all(), f"{col} has negative values"
            assert (df_all[col] <= 1).all(), f"{col} > 1"

    def test_rate_per_1k_pop_positive(self, df_all):
        """rate_per_1k_pop should be non-negative where defined."""
        valid = df_all["rate_per_1k_pop"].notna()
        if valid.any():
            assert (df_all.loc[valid, "rate_per_1k_pop"] >= 0).all()

    def test_rate_per_km2_positive(self, df_all):
        """rate_per_km2 should be non-negative where defined."""
        valid = df_all["rate_per_km2"].notna()
        if valid.any():
            assert (df_all.loc[valid, "rate_per_km2"] >= 0).all()


class TestResidentialFilter:
    """Tests for residential filter consistency."""

    def test_residential_all_have_is_residential_true(self, df_residential):
        """All residential NTAs should have is_residential=True."""
        assert df_residential["is_residential"].all()

    def test_residential_subset_of_all(self, df_all, df_residential):
        """Residential NTAs should be a subset of all NTAs."""
        all_codes = set(df_all["ntacode"])
        res_codes = set(df_residential["ntacode"])
        assert res_codes.issubset(all_codes)

    def test_non_residential_in_all(self, df_all):
        """All NTAs should include non-residential types."""
        non_res = df_all[~df_all["is_residential"]]
        assert len(non_res) > 0, "Expected some non-residential NTAs in all output"


class TestGeoJSON:
    """Tests for GeoJSON outputs."""

    def test_all_geojson_has_geometry(self, gdf_all):
        assert gdf_all.geometry.notna().all()

    def test_all_geojson_row_count(self, gdf_all):
        assert len(gdf_all) == 262

    def test_residential_geojson_row_count(self, gdf_residential):
        assert len(gdf_residential) == 197

    def test_geojson_has_ntacode(self, gdf_all):
        assert "ntacode" in gdf_all.columns

    def test_geojson_has_nta_name(self, gdf_all):
        assert "nta_name" in gdf_all.columns


class TestDataReasonableness:
    """Tests for data reasonableness."""

    def test_total_complaints_reasonable(self, df_all):
        """Total complaints should be in reasonable range for NYC nighttime 2021-2023."""
        total = df_all["count_night"].sum()
        # From CD analysis we know ~400k total
        assert 300000 < total < 500000, f"Total complaints {total} seems off"

    def test_most_residential_have_complaints(self, df_residential):
        """Most residential NTAs should have some complaints."""
        has_complaints = (df_residential["count_night"] > 0).sum()
        pct_with = has_complaints / len(df_residential)
        assert pct_with > 0.9, f"Only {pct_with:.1%} residential NTAs have complaints"

    def test_weekend_uplift_reasonable(self, df_all):
        """Mean weekend uplift should be > 1 (more complaints on weekends)."""
        mean_uplift = df_all["weekend_uplift"].mean()
        assert mean_uplift > 1.0, f"Mean weekend uplift {mean_uplift} seems low"

