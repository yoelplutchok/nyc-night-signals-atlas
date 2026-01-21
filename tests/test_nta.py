"""
Tests for NTA (Neighborhood Tabulation Area) Geographies (Script 09 outputs).

Per NYC_Night_Signals_Plan.md Script 09:
- NTA geometry files with authoritative codes + names
- Lookup table for neighborhood-scale analysis
"""

import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path

# Paths
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
GEO_DIR = PROCESSED_DIR / "geo"

NTA_PARQUET = GEO_DIR / "nta.parquet"
NTA_2263_PARQUET = GEO_DIR / "nta_epsg2263.parquet"
NTA_LOOKUP_PARQUET = GEO_DIR / "nta_lookup.parquet"
NTA_LOOKUP_CSV = GEO_DIR / "nta_lookup.csv"


class TestNTAOutputsExist:
    """Tests for output file existence."""

    def test_nta_parquet_exists(self):
        assert NTA_PARQUET.exists(), f"Missing: {NTA_PARQUET}"

    def test_nta_2263_parquet_exists(self):
        assert NTA_2263_PARQUET.exists(), f"Missing: {NTA_2263_PARQUET}"

    def test_nta_lookup_parquet_exists(self):
        assert NTA_LOOKUP_PARQUET.exists(), f"Missing: {NTA_LOOKUP_PARQUET}"

    def test_nta_lookup_csv_exists(self):
        assert NTA_LOOKUP_CSV.exists(), f"Missing: {NTA_LOOKUP_CSV}"


@pytest.fixture
def gdf_nta():
    """Load NTA geometries (EPSG:4326)."""
    if not NTA_PARQUET.exists():
        pytest.skip(f"Missing: {NTA_PARQUET}")
    return gpd.read_parquet(NTA_PARQUET)


@pytest.fixture
def gdf_nta_2263():
    """Load NTA geometries (EPSG:2263)."""
    if not NTA_2263_PARQUET.exists():
        pytest.skip(f"Missing: {NTA_2263_PARQUET}")
    return gpd.read_parquet(NTA_2263_PARQUET)


@pytest.fixture
def df_lookup():
    """Load NTA lookup table."""
    if not NTA_LOOKUP_PARQUET.exists():
        pytest.skip(f"Missing: {NTA_LOOKUP_PARQUET}")
    return pd.read_parquet(NTA_LOOKUP_PARQUET)


class TestNTAGeometry4326:
    """Tests for NTA geometry in EPSG:4326."""

    def test_has_ntacode(self, gdf_nta):
        """Must have ntacode column."""
        assert "ntacode" in gdf_nta.columns

    def test_ntacode_unique(self, gdf_nta):
        """ntacode must be unique."""
        assert gdf_nta["ntacode"].is_unique, "Duplicate ntacode values found"

    def test_has_nta_name(self, gdf_nta):
        """Must have nta_name column."""
        assert "nta_name" in gdf_nta.columns

    def test_nta_name_no_nulls(self, gdf_nta):
        """nta_name must have no nulls."""
        assert gdf_nta["nta_name"].notna().all(), "Null nta_name values found"

    def test_has_geometry(self, gdf_nta):
        """Must have valid geometry."""
        assert gdf_nta.geometry.notna().all()

    def test_geometry_valid(self, gdf_nta):
        """All geometries must be valid."""
        assert gdf_nta.geometry.is_valid.all()

    def test_crs_is_4326(self, gdf_nta):
        """CRS must be EPSG:4326."""
        assert gdf_nta.crs is not None
        assert gdf_nta.crs.to_epsg() == 4326

    def test_bounds_nyc_region(self, gdf_nta):
        """Bounds must be within NYC region."""
        bounds = gdf_nta.total_bounds
        # NYC approximate bounds in WGS84
        assert bounds[0] > -75.0, f"minx {bounds[0]} too far west"
        assert bounds[2] < -73.0, f"maxx {bounds[2]} too far east"
        assert bounds[1] > 40.0, f"miny {bounds[1]} too far south"
        assert bounds[3] < 42.0, f"maxy {bounds[3]} too far north"


class TestNTAGeometry2263:
    """Tests for NTA geometry in EPSG:2263."""

    def test_has_ntacode(self, gdf_nta_2263):
        """Must have ntacode column."""
        assert "ntacode" in gdf_nta_2263.columns

    def test_ntacode_unique(self, gdf_nta_2263):
        """ntacode must be unique."""
        assert gdf_nta_2263["ntacode"].is_unique

    def test_crs_is_2263(self, gdf_nta_2263):
        """CRS must be EPSG:2263."""
        assert gdf_nta_2263.crs is not None
        assert gdf_nta_2263.crs.to_epsg() == 2263

    def test_bounds_nyc_stateplane(self, gdf_nta_2263):
        """Bounds must be within NYC StatePlane region."""
        bounds = gdf_nta_2263.total_bounds
        # NYC approximate bounds in StatePlane feet
        assert bounds[0] > 900000, f"minx {bounds[0]} too far west"
        assert bounds[2] < 1100000, f"maxx {bounds[2]} too far east"
        assert bounds[1] > 100000, f"miny {bounds[1]} too far south"
        assert bounds[3] < 300000, f"maxy {bounds[3]} too far north"

    def test_same_row_count_as_4326(self, gdf_nta, gdf_nta_2263):
        """EPSG:2263 should have same row count as EPSG:4326."""
        assert len(gdf_nta) == len(gdf_nta_2263)


class TestNTALookup:
    """Tests for NTA lookup table."""

    def test_has_ntacode(self, df_lookup):
        """Must have ntacode column."""
        assert "ntacode" in df_lookup.columns

    def test_ntacode_unique(self, df_lookup):
        """ntacode must be unique."""
        assert df_lookup["ntacode"].is_unique

    def test_has_nta_name(self, df_lookup):
        """Must have nta_name column."""
        assert "nta_name" in df_lookup.columns

    def test_nta_name_no_nulls(self, df_lookup):
        """nta_name must have no nulls."""
        assert df_lookup["nta_name"].notna().all()

    def test_has_borough_info(self, df_lookup):
        """Should have borough information."""
        has_boro_code = "borough_code" in df_lookup.columns
        has_boro_name = "borough_name" in df_lookup.columns
        assert has_boro_code or has_boro_name, "No borough information in lookup"

    def test_has_is_residential(self, df_lookup):
        """Should have is_residential flag."""
        assert "is_residential" in df_lookup.columns

    def test_is_residential_boolean(self, df_lookup):
        """is_residential should be boolean."""
        assert df_lookup["is_residential"].dtype == bool

    def test_same_row_count_as_geometry(self, gdf_nta, df_lookup):
        """Lookup should have same row count as geometry."""
        assert len(df_lookup) == len(gdf_nta)

    def test_all_ntacodes_match(self, gdf_nta, df_lookup):
        """All ntacodes in geometry should be in lookup."""
        geo_codes = set(gdf_nta["ntacode"])
        lookup_codes = set(df_lookup["ntacode"])
        assert geo_codes == lookup_codes


class TestNTAReasonableness:
    """Tests for reasonableness of NTA data."""

    def test_reasonable_nta_count(self, gdf_nta):
        """Should have a reasonable number of NTAs (200-300 for NYC)."""
        assert 200 <= len(gdf_nta) <= 300, f"Unexpected NTA count: {len(gdf_nta)}"

    def test_has_all_boroughs(self, df_lookup):
        """Should have NTAs in all 5 boroughs."""
        if "borough_name" in df_lookup.columns:
            boroughs = set(df_lookup["borough_name"].unique())
            expected = {"Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"}
            assert expected.issubset(boroughs), f"Missing boroughs: {expected - boroughs}"

    def test_has_residential_ntas(self, df_lookup):
        """Should have residential NTAs."""
        if "is_residential" in df_lookup.columns:
            n_residential = df_lookup["is_residential"].sum()
            assert n_residential > 100, f"Only {n_residential} residential NTAs"

    def test_known_neighborhoods_present(self, df_lookup):
        """Some well-known neighborhoods should be present."""
        known_names = [
            "Greenpoint",
            "Williamsburg", 
            "Bedford-Stuyvesant",
            "Astoria",
            "Harlem",
        ]
        nta_names = df_lookup["nta_name"].str.lower().tolist()
        found = sum(1 for name in known_names if any(name.lower() in n for n in nta_names))
        assert found >= 3, f"Only found {found} known neighborhoods"


class TestNTAConsistency:
    """Tests for consistency between outputs."""

    def test_ntacode_format_consistent(self, df_lookup):
        """NTA codes should follow consistent format (e.g., BK0101)."""
        # NTA 2020 codes are like BK0101, MN0101, etc.
        for code in df_lookup["ntacode"]:
            assert len(code) == 6, f"Unexpected ntacode format: {code}"
            assert code[:2] in ["BK", "BX", "MN", "QN", "SI"], f"Unknown borough prefix: {code}"

    def test_borough_code_valid(self, df_lookup):
        """Borough codes should be 1-5."""
        if "borough_code" in df_lookup.columns:
            valid_codes = {1, 2, 3, 4, 5}
            actual_codes = set(df_lookup["borough_code"].unique())
            assert actual_codes.issubset(valid_codes), f"Invalid borough codes: {actual_codes - valid_codes}"

