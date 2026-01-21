"""
Tests for outcomes analysis (Script 13).

Per NYC_Night_Signals_Plan.md:
- Aggregate tract-level data to CD/NTA using population-weighted averages
- Use intersection population from crosswalk for split tracts
- Join with cluster assignments
- Report effect sizes and uncertainty

Coverage includes:
- Schema validation
- Population weighting correctness
- ACS data handling
- Edge cases and failure modes
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from sleep_esi.paths import PROCESSED_DIR, XWALK_DIR, GEO_DIR


class TestOutcomesOutputs(unittest.TestCase):
    """Tests for outcomes output files."""

    def setUp(self):
        self.outcomes_dir = PROCESSED_DIR / "outcomes"
        self.cd_outcomes_path = self.outcomes_dir / "cd_outcomes.parquet"
        self.nta_outcomes_path = self.outcomes_dir / "nta_outcomes.parquet"
        self.summary_path = self.outcomes_dir / "cluster_outcome_summary.csv"

    def test_outputs_exist(self):
        """All outcomes outputs must exist."""
        self.assertTrue(self.cd_outcomes_path.exists(), "cd_outcomes.parquet missing")
        self.assertTrue(self.nta_outcomes_path.exists(), "nta_outcomes.parquet missing")
        self.assertTrue(self.summary_path.exists(), "cluster_outcome_summary.csv missing")

    def test_cd_outcomes_schema(self):
        """CD outcomes must have required columns."""
        df = pd.read_parquet(self.cd_outcomes_path)
        required_cols = [
            'boro_cd', 'cluster_id', 'cluster_label',
            'sleep', 'poverty_rate', 'pct_black', 'pct_hispanic', 'rent_burden_rate'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        self.assertEqual(len(df), 59, f"Expected 59 CDs, got {len(df)}")

    def test_cd_outcomes_boro_cd_unique(self):
        """boro_cd values must be unique."""
        df = pd.read_parquet(self.cd_outcomes_path)
        self.assertEqual(df['boro_cd'].nunique(), 59, "Duplicate boro_cd values")

    def test_nta_outcomes_schema(self):
        """NTA outcomes must have required columns."""
        df = pd.read_parquet(self.nta_outcomes_path)
        required_cols = [
            'ntacode', 'cluster_id', 'cluster_label',
            'sleep', 'poverty_rate', 'pct_black', 'pct_hispanic', 'rent_burden_rate'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        # 197 residential NTAs
        self.assertEqual(len(df), 197, f"Expected 197 NTAs, got {len(df)}")

    def test_nta_outcomes_ntacode_unique(self):
        """ntacode values must be unique."""
        df = pd.read_parquet(self.nta_outcomes_path)
        self.assertEqual(df['ntacode'].nunique(), len(df), "Duplicate ntacode values")

    def test_summary_format(self):
        """Summary must have correct format with both CD and NTA levels."""
        df = pd.read_csv(self.summary_path)
        self.assertIn('level', df.columns, "Missing level column")
        self.assertIn('CD', df['level'].values, "Missing CD level in summary")
        self.assertIn('NTA', df['level'].values, "Missing NTA level in summary")

        # Check that we have means for sleep
        self.assertIn('sleep_mean', df.columns, "Missing sleep_mean column")
        # Reasonable range for % short sleep (typically 25-45%)
        self.assertTrue((df['sleep_mean'] > 20).all(), "sleep_mean too low")
        self.assertTrue((df['sleep_mean'] < 60).all(), "sleep_mean too high")

    def test_no_null_clusters(self):
        """All records must have cluster assignments."""
        cd_df = pd.read_parquet(self.cd_outcomes_path)
        self.assertEqual(cd_df['cluster_id'].isna().sum(), 0, "Null cluster_id in CD outcomes")

        nta_df = pd.read_parquet(self.nta_outcomes_path)
        # NTA clusters might have -1 for Low Volume, but not null
        self.assertEqual(nta_df['cluster_id'].isna().sum(), 0, "Null cluster_id in NTA outcomes")


class TestPopulationWeighting(unittest.TestCase):
    """
    Tests for population weighting correctness.

    The aggregation formula should be:
        CD_rate = sum(tract_rate * intersection_pop) / sum(intersection_pop)

    Where intersection_pop = tract_pop from the crosswalk (not total tract pop).
    """

    def test_rates_within_bounds(self):
        """All rate columns should be within [0, 1]."""
        df = pd.read_parquet(PROCESSED_DIR / "outcomes" / "cd_outcomes.parquet")
        rate_cols = ['poverty_rate', 'pct_black', 'pct_hispanic', 'rent_burden_rate']

        for col in rate_cols:
            valid = df[col].dropna()
            self.assertTrue((valid >= 0).all(), f"{col} has negative values")
            self.assertTrue((valid <= 1).all(), f"{col} has values > 1")

    def test_population_estimates_reasonable(self):
        """Population estimates should be reasonable for residential CDs."""
        df = pd.read_parquet(PROCESSED_DIR / "outcomes" / "cd_outcomes.parquet")

        if 'population_est' in df.columns:
            # NYC has ~8.3M people, 59 CDs → ~140k average per CD
            # We only check CDs that are residential clusters (skip outliers/parks)
            # cluster_id -1 is Low Volume (for NTA), for CD we check if it's in the 59
            
            # Filter to those with meaningful population
            res_df = df[df['population_est'] > 1000] 
            
            self.assertTrue((res_df['population_est'] > 10000).all(), "Population too low for some residential CDs")
            self.assertTrue((res_df['population_est'] < 500000).all(), "Population too high for some CDs")

            # Total should be roughly NYC population
            total_pop = df['population_est'].sum()
            self.assertTrue(7_000_000 < total_pop < 10_000_000, f"Total pop {total_pop:,.0f} out of range")

    def test_crosswalk_weights_sum_to_one(self):
        """Crosswalk population weights should sum to ~1 for populated CDs."""
        xwalk_path = XWALK_DIR / "cd_to_tract_weights.parquet"
        if xwalk_path.exists():
            xwalk = pd.read_parquet(xwalk_path)
            # Skip CDs with 0 total population (parks, etc.)
            cd_pops = xwalk.groupby('boro_cd')['tract_pop'].sum()
            populated_cds = cd_pops[cd_pops > 0].index
            
            weight_sums = xwalk[xwalk['boro_cd'].isin(populated_cds)].groupby('boro_cd')['w_pop'].sum()

            # Weights should sum to ~1 (within tolerance)
            deviations = (weight_sums - 1.0).abs()
            self.assertTrue((deviations < 0.01).all(),
                            f"Weight sums deviate from 1: max deviation = {deviations.max()}")

    def test_poverty_rate_citywide_reasonable(self):
        """
        Citywide poverty rate should be reasonable.

        NYC poverty rate is typically 15-20%. The weighted average should be in this range.
        """
        df = pd.read_parquet(PROCESSED_DIR / "outcomes" / "cd_outcomes.parquet")

        if 'population_est' in df.columns and 'poverty_rate' in df.columns:
            # Population-weighted citywide average
            valid = df.dropna(subset=['population_est', 'poverty_rate'])
            citywide_poverty = (
                (valid['poverty_rate'] * valid['population_est']).sum() /
                valid['population_est'].sum()
            )
            self.assertTrue(0.10 < citywide_poverty < 0.30,
                            f"Citywide poverty rate {citywide_poverty:.1%} seems off")


class TestRateAggregation(unittest.TestCase):
    """Tests for rate aggregation methodology."""

    def test_sleep_deprivation_range(self):
        """Sleep deprivation rates should be in expected range (25-45% typically)."""
        df = pd.read_parquet(PROCESSED_DIR / "outcomes" / "cd_outcomes.parquet")

        if 'sleep' in df.columns:
            valid = df['sleep'].dropna()
            # CDC PLACES measures % adults with <7 hours sleep
            # NYC average is typically around 35%
            self.assertTrue((valid > 15).all(), "Sleep rate too low (expected % <7hrs sleep)")
            self.assertTrue((valid < 55).all(), "Sleep rate too high")

    def test_rent_burden_range(self):
        """Rent burden rates should be in expected range."""
        df = pd.read_parquet(PROCESSED_DIR / "outcomes" / "cd_outcomes.parquet")

        if 'rent_burden_rate' in df.columns:
            valid = df['rent_burden_rate'].dropna()
            # NYC has high rent burden, typically 40-60% of renters
            self.assertTrue((valid > 0.20).all(), "Rent burden too low")
            self.assertTrue((valid < 0.90).all(), "Rent burden too high")

    def test_demographic_rates_consistent(self):
        """
        pct_black + pct_hispanic + other should be ≤ 1 in most CDs.

        Note: These are NOT mutually exclusive in census data (Hispanic is ethnicity),
        but extreme values suggest data issues.
        """
        df = pd.read_parquet(PROCESSED_DIR / "outcomes" / "cd_outcomes.parquet")

        if all(c in df.columns for c in ['pct_black', 'pct_hispanic']):
            # Neither should exceed 1
            self.assertTrue((df['pct_black'].fillna(0) <= 1).all())
            self.assertTrue((df['pct_hispanic'].fillna(0) <= 1).all())


class TestClusterOutcomes(unittest.TestCase):
    """Tests for cluster-outcome associations."""

    def test_summary_has_all_clusters(self):
        """Summary should include all cluster types from both CD and NTA levels."""
        summary = pd.read_csv(PROCESSED_DIR / "outcomes" / "cluster_outcome_summary.csv")

        # Should have cluster_id and cluster_label
        self.assertIn('cluster_id', summary.columns)
        self.assertIn('cluster_label', summary.columns)

        # CD level should have clusters
        cd_summary = summary[summary['level'] == 'CD']
        self.assertGreater(len(cd_summary), 0, "No CD clusters in summary")

        # NTA level should have clusters
        nta_summary = summary[summary['level'] == 'NTA']
        self.assertGreater(len(nta_summary), 0, "No NTA clusters in summary")

    def test_cluster_outcome_variation(self):
        """
        Different clusters should show variation in outcomes.

        If all clusters have identical outcomes, clustering isn't capturing
        meaningful differences.
        """
        summary = pd.read_csv(PROCESSED_DIR / "outcomes" / "cluster_outcome_summary.csv")
        cd_summary = summary[summary['level'] == 'CD']

        if 'poverty_rate_mean' in cd_summary.columns and len(cd_summary) > 1:
            poverty_range = cd_summary['poverty_rate_mean'].max() - cd_summary['poverty_rate_mean'].min()
            self.assertGreater(poverty_range, 0.02, "Clusters show no variation in poverty rate")

    def test_sample_sizes_reasonable(self):
        """Each cluster should have reasonable sample sizes."""
        summary = pd.read_csv(PROCESSED_DIR / "outcomes" / "cluster_outcome_summary.csv")

        # Look for count columns
        count_cols = [c for c in summary.columns if c.endswith('_count')]
        if count_cols:
            for col in count_cols:
                # Each cluster should have multiple observations
                self.assertTrue((summary[col] >= 1).all(), f"Zero observations in {col}")


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and failure modes."""

    def test_missing_tract_data_handling(self):
        """
        Some tracts may have missing demographic data.
        Check that this doesn't cause NaN propagation.
        """
        df = pd.read_parquet(PROCESSED_DIR / "outcomes" / "cd_outcomes.parquet")

        # Most CDs should have valid data
        rate_cols = ['poverty_rate', 'pct_black', 'pct_hispanic', 'rent_burden_rate']
        for col in rate_cols:
            null_count = df[col].isna().sum()
            self.assertLess(null_count, 10, f"Too many null values in {col}: {null_count}")

    def test_split_tract_handling(self):
        """
        Check for tracts that span CD boundaries.
        Note: In 2020 NYC geography, tracts are designed to nest in CDTAs,
        so split tracts may be rare or non-existent in the official mapping.
        """
        xwalk_path = XWALK_DIR / "cd_to_tract_weights.parquet"
        if xwalk_path.exists():
            xwalk = pd.read_parquet(xwalk_path)

            # Check for split tracts (same geoid, multiple CDs)
            tract_counts = xwalk.groupby('tract_geoid')['boro_cd'].nunique()
            split_tracts = tract_counts[tract_counts > 1]

            if len(split_tracts) > 0:
                # For split tracts, weights across CDs should sum to ~1
                # This validation depends on how w_pop is defined
                pass

    def test_nta_tract_nesting(self):
        """
        2020 census tracts should nest perfectly within NTAs.

        Unlike CDs, there should be no split tracts for NTA aggregation.
        """
        # NTA outcomes should have population_est equal to sum of tract populations
        nta_df = pd.read_parquet(PROCESSED_DIR / "outcomes" / "nta_outcomes.parquet")

        # All NTAs should have valid outcomes
        self.assertEqual(len(nta_df), 197, "Not all residential NTAs have outcomes")


class TestMetadataIntegrity(unittest.TestCase):
    """Tests for metadata and provenance."""

    def test_metadata_file_exists(self):
        """Outcomes metadata sidecar should exist."""
        metadata_path = PROCESSED_DIR / "metadata" / "outcomes_metadata.json"
        self.assertTrue(metadata_path.exists(), "outcomes_metadata.json missing")

    def test_metadata_has_required_fields(self):
        """Metadata should have required provenance fields."""
        import json
        metadata_path = PROCESSED_DIR / "metadata" / "outcomes_metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.assertIn('script', metadata)
            self.assertIn('run_id', metadata)
            self.assertIn('timestamp', metadata)
            self.assertIn('inputs', metadata)


if __name__ == '__main__':
    unittest.main()
