"""
Tests for CD lookup table integrity.

Per NYC_Night_Signals_Plan.md Section 2 (Critical QA Gate):
- Verify boro_cd = borough_code * 100 + district_number
- Ensure 59 unique residential CDs
- Validate borough name/code mappings are authoritative
- Fail hard on any mismatch

These tests MUST pass before any maps, profile cards, or visual outputs are created.
"""

import pytest
import pandas as pd
from pathlib import Path

from sleep_esi.paths import GEO_DIR

# =============================================================================
# AUTHORITATIVE MAPPINGS - Fixed NYC standards
# =============================================================================

AUTHORITATIVE_BOROUGH_CODES = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island",
}

EXPECTED_CD_COUNTS = {
    "Manhattan": 12,
    "Bronx": 12,
    "Brooklyn": 18,
    "Queens": 14,
    "Staten Island": 3,
}

EXPECTED_TOTAL_CDS = 59


class TestCDLookupExists:
    """Tests that required lookup files exist."""

    def test_cd_lookup_parquet_exists(self):
        """cd_lookup.parquet must exist."""
        path = GEO_DIR / "cd_lookup.parquet"
        assert path.exists(), f"Missing required file: {path}"

    def test_cd_lookup_csv_exists(self):
        """cd_lookup.csv must exist for easy inspection."""
        path = GEO_DIR / "cd_lookup.csv"
        assert path.exists(), f"Missing required file: {path}"


class TestCDLookupSchema:
    """Tests for cd_lookup schema correctness."""

    @pytest.fixture
    def lookup(self):
        """Load the lookup table."""
        return pd.read_parquet(GEO_DIR / "cd_lookup.parquet")

    def test_required_columns_present(self, lookup):
        """All required columns must be present."""
        required = ["boro_cd", "borough_code", "borough_name", "district_number", "cd_label"]
        for col in required:
            assert col in lookup.columns, f"Missing required column: {col}"

    def test_row_count(self, lookup):
        """Must have exactly 59 residential CDs."""
        assert len(lookup) == EXPECTED_TOTAL_CDS, \
            f"Expected {EXPECTED_TOTAL_CDS} CDs, got {len(lookup)}"

    def test_boro_cd_unique(self, lookup):
        """boro_cd must be unique."""
        assert lookup["boro_cd"].nunique() == len(lookup), \
            "Duplicate boro_cd values found"

    def test_no_nulls(self, lookup):
        """No null values allowed in required columns."""
        required = ["boro_cd", "borough_code", "borough_name", "district_number", "cd_label"]
        for col in required:
            null_count = lookup[col].isna().sum()
            assert null_count == 0, f"Found {null_count} nulls in column {col}"


class TestBoroCDConsistency:
    """Critical QA tests for boro_cd consistency."""

    @pytest.fixture
    def lookup(self):
        """Load the lookup table."""
        return pd.read_parquet(GEO_DIR / "cd_lookup.parquet")

    def test_boro_cd_equals_computed(self, lookup):
        """
        CRITICAL: boro_cd must equal borough_code * 100 + district_number.
        
        This is the core consistency check from NYC_Night_Signals_Plan.md Section 2.
        Fails hard if any mismatch found.
        """
        computed = lookup["borough_code"] * 100 + lookup["district_number"]
        mismatches = lookup[lookup["boro_cd"] != computed]
        
        if len(mismatches) > 0:
            mismatch_info = mismatches[["boro_cd", "borough_code", "district_number"]].to_dict("records")
            pytest.fail(
                f"CRITICAL: {len(mismatches)} boro_cd mismatches found!\n"
                f"Mismatches: {mismatch_info}\n"
                f"Expected: boro_cd = borough_code * 100 + district_number"
            )

    def test_borough_codes_authoritative(self, lookup):
        """Borough codes must match authoritative NYC mapping."""
        for _, row in lookup.iterrows():
            code = row["borough_code"]
            name = row["borough_name"]
            
            assert code in AUTHORITATIVE_BOROUGH_CODES, \
                f"Unknown borough code: {code}"
            
            expected_name = AUTHORITATIVE_BOROUGH_CODES[code]
            assert name == expected_name, \
                f"Borough code {code} should be '{expected_name}', got '{name}'"

    def test_borough_cd_counts(self, lookup):
        """Each borough must have the expected number of CDs."""
        counts = lookup.groupby("borough_name").size().to_dict()
        
        for borough, expected_count in EXPECTED_CD_COUNTS.items():
            actual = counts.get(borough, 0)
            assert actual == expected_count, \
                f"{borough}: expected {expected_count} CDs, got {actual}"

    def test_district_numbers_valid(self, lookup):
        """District numbers must be positive integers within valid range."""
        for _, row in lookup.iterrows():
            dn = row["district_number"]
            borough = row["borough_name"]
            max_expected = EXPECTED_CD_COUNTS[borough]
            
            assert 1 <= dn <= max_expected, \
                f"Invalid district number {dn} for {borough} (expected 1-{max_expected})"


class TestCDLabels:
    """Tests for CD label formatting."""

    @pytest.fixture
    def lookup(self):
        """Load the lookup table."""
        return pd.read_parquet(GEO_DIR / "cd_lookup.parquet")

    def test_cd_label_format(self, lookup):
        """cd_label must follow '<Borough> Community District <n>' format."""
        for _, row in lookup.iterrows():
            label = row["cd_label"]
            borough = row["borough_name"]
            dn = row["district_number"]
            
            expected = f"{borough} Community District {dn}"
            assert label == expected, \
                f"Label mismatch for boro_cd {row['boro_cd']}: got '{label}', expected '{expected}'"

    def test_cd_short_format(self, lookup):
        """cd_short must follow '<ABBREV> <n>' format if present."""
        if "cd_short" not in lookup.columns:
            pytest.skip("cd_short column not present")
        
        abbrev_map = {
            "Manhattan": "MN",
            "Bronx": "BX",
            "Brooklyn": "BK",
            "Queens": "QN",
            "Staten Island": "SI",
        }
        
        for _, row in lookup.iterrows():
            short = row["cd_short"]
            borough = row["borough_name"]
            dn = row["district_number"]
            
            expected = f"{abbrev_map[borough]} {dn}"
            assert short == expected, \
                f"Short label mismatch: got '{short}', expected '{expected}'"


@pytest.mark.smoke
class TestCDLookupSmoke:
    """Quick smoke tests for CD lookup."""

    def test_can_load_lookup(self):
        """Basic load should work."""
        df = pd.read_parquet(GEO_DIR / "cd_lookup.parquet")
        assert len(df) > 0

    def test_known_cd_exists(self):
        """Known CDs should be present."""
        df = pd.read_parquet(GEO_DIR / "cd_lookup.parquet")
        known_cds = [101, 212, 301, 401, 501]  # One from each borough
        for cd in known_cds:
            assert cd in df["boro_cd"].values, f"Missing known CD: {cd}"


# =============================================================================
# HELPER FUNCTION FOR REPORTING QA
# =============================================================================

def assert_cd_labels_present(df: pd.DataFrame, context: str = "dataframe") -> None:
    """
    QA check to ensure cd_lookup has been joined into a reporting table.
    
    Per NYC_Night_Signals_Plan.md: Never print bare boro_cd without label.
    
    Args:
        df: DataFrame that should have CD labels
        context: Description for error messages
    
    Raises:
        AssertionError if cd_label column is missing or has nulls
    """
    if "boro_cd" in df.columns:
        assert "cd_label" in df.columns, \
            f"{context}: Has 'boro_cd' but missing 'cd_label'. " \
            f"Join cd_lookup.parquet before generating reports."
        
        null_labels = df["cd_label"].isna().sum()
        assert null_labels == 0, \
            f"{context}: Found {null_labels} null cd_label values after join."


class TestReportingQAHelper:
    """Tests for the reporting QA helper function."""
    
    def test_assert_cd_labels_passes_with_label(self):
        """Should pass when cd_label is present."""
        df = pd.DataFrame({
            "boro_cd": [101, 102],
            "cd_label": ["Manhattan Community District 1", "Manhattan Community District 2"],
            "value": [1.0, 2.0]
        })
        # Should not raise
        assert_cd_labels_present(df, "test")
    
    def test_assert_cd_labels_fails_without_label(self):
        """Should fail when boro_cd present but cd_label missing."""
        df = pd.DataFrame({
            "boro_cd": [101, 102],
            "value": [1.0, 2.0]
        })
        with pytest.raises(AssertionError, match="missing 'cd_label'"):
            assert_cd_labels_present(df, "test")
    
    def test_assert_cd_labels_fails_with_nulls(self):
        """Should fail when cd_label has null values."""
        df = pd.DataFrame({
            "boro_cd": [101, 102],
            "cd_label": ["Manhattan Community District 1", None],
            "value": [1.0, 2.0]
        })
        with pytest.raises(AssertionError, match="null cd_label"):
            assert_cd_labels_present(df, "test")
    
    def test_assert_cd_labels_passes_without_boro_cd(self):
        """Should pass if there's no boro_cd column (nothing to check)."""
        df = pd.DataFrame({
            "some_other_col": [1, 2, 3],
            "value": [1.0, 2.0, 3.0]
        })
        # Should not raise (no boro_cd to check)
        assert_cd_labels_present(df, "test")

