"""
Tests for Hotspot Investigation (Script 06b outputs).

Per NYC_Night_Signals_Plan.md Script 06b:
- Validate extreme hotspot cells
- Check for geocoding artifacts / proxy coordinates
- Examine temporal spread and address uniqueness
"""

import pytest
import pandas as pd
import json
from pathlib import Path

# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
HOTSPOTS_DIR = PROCESSED_DIR / "hotspots"

INVESTIGATION_CSV = HOTSPOTS_DIR / "hotspot_investigation_top_cells.csv"


class TestInvestigationOutputExists:
    """Tests for output file existence."""

    def test_investigation_csv_exists(self):
        assert INVESTIGATION_CSV.exists(), f"Missing: {INVESTIGATION_CSV}"


@pytest.fixture
def df_investigation():
    """Load investigation results."""
    if not INVESTIGATION_CSV.exists():
        pytest.skip(f"Missing: {INVESTIGATION_CSV}")
    return pd.read_csv(INVESTIGATION_CSV)


class TestInvestigationSchema:
    """Tests for required columns and schema."""

    def test_has_cell_id(self, df_investigation):
        """Must have cell_id column."""
        assert "cell_id" in df_investigation.columns

    def test_cell_id_unique(self, df_investigation):
        """cell_id must be unique."""
        assert df_investigation["cell_id"].is_unique

    def test_has_boro_cd(self, df_investigation):
        """Must have boro_cd column."""
        assert "boro_cd" in df_investigation.columns

    def test_has_cd_label(self, df_investigation):
        """Must have cd_label column."""
        assert "cd_label" in df_investigation.columns

    def test_has_complaint_count(self, df_investigation):
        """Must have complaint_count column."""
        assert "complaint_count" in df_investigation.columns

    def test_complaint_count_positive(self, df_investigation):
        """complaint_count must be positive."""
        assert (df_investigation["complaint_count"] > 0).all()

    def test_has_unique_latlon_rounded(self, df_investigation):
        """Must have unique_latlon_rounded column."""
        assert "unique_latlon_rounded" in df_investigation.columns

    def test_unique_latlon_positive(self, df_investigation):
        """unique_latlon_rounded must be positive."""
        assert (df_investigation["unique_latlon_rounded"] > 0).all()

    def test_has_top_latlon_share(self, df_investigation):
        """Must have top_latlon_share column."""
        assert "top_latlon_share" in df_investigation.columns

    def test_top_latlon_share_in_valid_range(self, df_investigation):
        """top_latlon_share must be in [0, 1]."""
        assert (df_investigation["top_latlon_share"] >= 0).all()
        assert (df_investigation["top_latlon_share"] <= 1).all()


class TestTemporalFields:
    """Tests for temporal spread fields."""

    def test_has_unique_dates(self, df_investigation):
        """Must have unique_dates column."""
        assert "unique_dates" in df_investigation.columns

    def test_unique_dates_positive(self, df_investigation):
        """unique_dates must be positive."""
        assert (df_investigation["unique_dates"] > 0).all()

    def test_has_unique_years(self, df_investigation):
        """Must have unique_years column."""
        assert "unique_years" in df_investigation.columns

    def test_unique_years_reasonable(self, df_investigation):
        """unique_years must be in range [1, 3] for 2021-2023."""
        assert (df_investigation["unique_years"] >= 1).all()
        assert (df_investigation["unique_years"] <= 3).all()

    def test_has_date_range(self, df_investigation):
        """Must have date_min and date_max columns."""
        assert "date_min" in df_investigation.columns
        assert "date_max" in df_investigation.columns

    def test_has_year_distribution(self, df_investigation):
        """Must have year_distribution column."""
        assert "year_distribution" in df_investigation.columns

    def test_year_distribution_is_valid_json(self, df_investigation):
        """year_distribution must be valid JSON."""
        for val in df_investigation["year_distribution"]:
            parsed = json.loads(val)
            assert isinstance(parsed, dict)

    def test_has_max_single_day_count(self, df_investigation):
        """Must have max_single_day_count column."""
        assert "max_single_day_count" in df_investigation.columns

    def test_max_single_day_count_positive(self, df_investigation):
        """max_single_day_count must be positive."""
        assert (df_investigation["max_single_day_count"] > 0).all()


class TestAddressFields:
    """Tests for address-related fields (may be nullable)."""

    def test_has_address_columns(self, df_investigation):
        """Should have address columns (values may be null if field missing)."""
        assert "unique_addresses" in df_investigation.columns
        assert "top_address_share" in df_investigation.columns

    def test_top_address_share_valid_when_present(self, df_investigation):
        """top_address_share must be in [0, 1] when not null."""
        valid = df_investigation["top_address_share"].notna()
        if valid.any():
            assert (df_investigation.loc[valid, "top_address_share"] >= 0).all()
            assert (df_investigation.loc[valid, "top_address_share"] <= 1).all()


class TestComplaintTypeFields:
    """Tests for complaint type fields."""

    def test_has_top_complaint_type(self, df_investigation):
        """Must have top_complaint_type column."""
        assert "top_complaint_type" in df_investigation.columns

    def test_has_top_complaint_type_share(self, df_investigation):
        """Must have top_complaint_type_share column."""
        assert "top_complaint_type_share" in df_investigation.columns

    def test_top_complaint_type_share_valid(self, df_investigation):
        """top_complaint_type_share must be in [0, 1] when not null."""
        valid = df_investigation["top_complaint_type_share"].notna()
        if valid.any():
            assert (df_investigation.loc[valid, "top_complaint_type_share"] >= 0).all()
            assert (df_investigation.loc[valid, "top_complaint_type_share"] <= 1).all()


class TestDescriptorFields:
    """Tests for descriptor fields."""

    def test_has_top_descriptor(self, df_investigation):
        """Must have top_descriptor column."""
        assert "top_descriptor" in df_investigation.columns

    def test_has_descriptor_top3(self, df_investigation):
        """Must have descriptor_top3 column."""
        assert "descriptor_top3" in df_investigation.columns

    def test_descriptor_top3_is_valid_json(self, df_investigation):
        """descriptor_top3 must be valid JSON when not null."""
        valid = df_investigation["descriptor_top3"].notna()
        for val in df_investigation.loc[valid, "descriptor_top3"]:
            parsed = json.loads(val)
            assert isinstance(parsed, dict)


class TestReasonableness:
    """Tests for reasonableness of investigation results."""

    def test_row_count_reasonable(self, df_investigation):
        """Should have 20 rows (default top_n_cells)."""
        # Default is 20, but could be configured differently
        assert len(df_investigation) >= 1
        assert len(df_investigation) <= 100

    def test_sorted_by_count(self, df_investigation):
        """Should be sorted by complaint_count descending."""
        assert df_investigation["complaint_count"].is_monotonic_decreasing

    def test_top_cell_matches_known_hotspot(self, df_investigation):
        """Top cell should be cell 33209 in BX 12 (per plan documentation)."""
        top_cell = df_investigation.iloc[0]
        # Cell 33209 has 31,207 complaints per plan
        assert top_cell["cell_id"] == 33209, f"Expected cell 33209, got {top_cell['cell_id']}"
        assert top_cell["boro_cd"] == 212, f"Expected BX 12 (212), got {top_cell['boro_cd']}"

    def test_unique_coords_less_than_complaints(self, df_investigation):
        """unique_latlon_rounded must be <= complaint_count."""
        assert (df_investigation["unique_latlon_rounded"] <= df_investigation["complaint_count"]).all()

    def test_unique_dates_less_than_complaint_count(self, df_investigation):
        """unique_dates must be <= complaint_count."""
        assert (df_investigation["unique_dates"] <= df_investigation["complaint_count"]).all()


class TestDiagnosticPower:
    """Tests to ensure investigation provides useful diagnostics."""

    def test_identifies_concentrated_locations(self, df_investigation):
        """At least one cell should have top_latlon_share indicating concentration."""
        # If all complaints were at random locations, share would be ~1/count
        # High share (>0.1) indicates geocoding artifact potential
        max_share = df_investigation["top_latlon_share"].max()
        assert max_share > 0.01, "Investigation should find some coordinate concentration"

    def test_temporal_spread_shows_persistence(self, df_investigation):
        """Top cells should span multiple dates (indicating persistent problems)."""
        top_cell = df_investigation.iloc[0]
        # A cell with 31K+ complaints should span many dates
        assert top_cell["unique_dates"] > 100, (
            f"Expected many unique dates for top cell, got {top_cell['unique_dates']}"
        )

    def test_year_coverage(self, df_investigation):
        """Most investigated cells should span multiple years."""
        multi_year = (df_investigation["unique_years"] >= 2).sum()
        assert multi_year >= len(df_investigation) * 0.5, (
            "Expected most cells to span multiple years"
        )

