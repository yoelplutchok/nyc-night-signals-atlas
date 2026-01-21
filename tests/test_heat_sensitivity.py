"""
Tests for heat sensitivity analysis (Script 07).

Per NYC_Night_Signals_Plan.md:
- Night date assignment: complaints at 22:00-23:59 assigned to NEXT day
- Complaints at 00:00-06:59 assigned to CURRENT day
- This matches the Tmin (minimum temperature) for the night ending that morning
- Poisson GLM with categorical controls (DOW, month, year)
- Slope-to-percent calculation: (exp(slope) - 1) * 100

Coverage includes:
- Schema validation
- Night date assignment logic
- Statistical model validation
- Edge cases and failure modes
"""

import unittest
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import numpy as np
import pytz

from sleep_esi.paths import PROCESSED_DIR
from sleep_esi.time_utils import ensure_nyc_timezone, is_nighttime, filter_nighttime


class TestHeatSensitivityOutputs(unittest.TestCase):
    """Tests for heat sensitivity output files."""

    def setUp(self):
        self.weather_dir = PROCESSED_DIR / "weather"
        self.cd_sens_path = self.weather_dir / "cd_heat_sensitivity.parquet"
        self.curve_path = self.weather_dir / "citywide_temp_curve.csv"

    def test_outputs_exist(self):
        """Heat sensitivity outputs must exist."""
        self.assertTrue(self.cd_sens_path.exists(), "cd_heat_sensitivity.parquet missing")
        self.assertTrue(self.curve_path.exists(), "citywide_temp_curve.csv missing")

    def test_cd_sens_schema(self):
        """CD sensitivity output must have required columns."""
        df = pd.read_parquet(self.cd_sens_path)
        required_cols = [
            'boro_cd', 'cd_label', 'slope', 'se', 'pvalue', 'pct_increase_per_c', 'n_obs'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")
        self.assertEqual(len(df), 59, f"Expected 59 CDs, got {len(df)}")

    def test_cd_sens_boro_cd_unique(self):
        """boro_cd values must be unique."""
        df = pd.read_parquet(self.cd_sens_path)
        self.assertEqual(df['boro_cd'].nunique(), 59, "Duplicate boro_cd values")

    def test_cd_sens_no_null_labels(self):
        """cd_label must not have null values."""
        df = pd.read_parquet(self.cd_sens_path)
        null_count = df['cd_label'].isna().sum()
        self.assertEqual(null_count, 0, f"Found {null_count} null cd_label values")

    def test_curve_format(self):
        """Citywide curve must have correct format."""
        df = pd.read_csv(self.curve_path)
        self.assertIn('tmin', df.columns, "Missing tmin column")
        self.assertIn('pred_count', df.columns, "Missing pred_count column")
        # Should be sorted by temperature
        self.assertTrue(df['tmin'].is_monotonic_increasing, "Curve not sorted by temperature")

    def test_slope_range(self):
        """Slopes should be within reasonable range."""
        df = pd.read_parquet(self.cd_sens_path)
        # Slopes should be reasonable (not extreme)
        self.assertTrue((df['slope'] > -0.5).all(), "Slope too negative")
        self.assertTrue((df['slope'] < 0.5).all(), "Slope too positive")

    def test_pct_increase_calculation(self):
        """Percent increase should be correctly calculated from slope."""
        df = pd.read_parquet(self.cd_sens_path)
        # pct_increase = (exp(slope) - 1) * 100
        expected_pct = (np.exp(df['slope']) - 1) * 100
        np.testing.assert_array_almost_equal(
            df['pct_increase_per_c'].values,
            expected_pct.values,
            decimal=4,
            err_msg="pct_increase_per_c calculation incorrect"
        )

    def test_n_obs_reasonable(self):
        """Observation counts should be reasonable."""
        df = pd.read_parquet(self.cd_sens_path)
        # Each CD should have observations (warm season days * years)
        # ~150 days * 3 years = ~450 observations per CD
        self.assertTrue((df['n_obs'] > 100).all(), "Too few observations for some CDs")
        self.assertTrue((df['n_obs'] < 1000).all(), "Unexpectedly high observation count")

    def test_standard_errors_positive(self):
        """Standard errors must be positive."""
        df = pd.read_parquet(self.cd_sens_path)
        self.assertTrue((df['se'] > 0).all(), "Standard errors must be positive")


class TestNightDateAssignment(unittest.TestCase):
    """
    Tests for night date assignment logic.

    The key rule: complaints are matched to the Tmin recorded the morning
    the night ENDS. So:
    - 22:00 on Jan 1 → assigned to Jan 2 (night ends morning of Jan 2)
    - 02:00 on Jan 2 → assigned to Jan 2 (night ends morning of Jan 2)
    - 06:59 on Jan 2 → assigned to Jan 2 (night ends morning of Jan 2)
    """

    def _assign_night_date(self, ts_nyc):
        """Replicate the night date assignment logic from Script 07."""
        hour = ts_nyc.hour
        if hour >= 22:
            # Before midnight: assign to next day
            return (ts_nyc + pd.Timedelta(days=1)).date()
        else:
            # After midnight: assign to current day
            return ts_nyc.date()

    def test_complaint_at_2200_assigned_to_next_day(self):
        """Complaint at 22:00 should be assigned to next day's Tmin."""
        nyc_tz = pytz.timezone("America/New_York")
        ts = nyc_tz.localize(datetime(2023, 7, 15, 22, 0))  # 10 PM on July 15
        night_date = self._assign_night_date(ts)
        self.assertEqual(night_date, date(2023, 7, 16), "22:00 should map to next day")

    def test_complaint_at_2359_assigned_to_next_day(self):
        """Complaint at 23:59 should be assigned to next day's Tmin."""
        nyc_tz = pytz.timezone("America/New_York")
        ts = nyc_tz.localize(datetime(2023, 7, 15, 23, 59))  # 11:59 PM on July 15
        night_date = self._assign_night_date(ts)
        self.assertEqual(night_date, date(2023, 7, 16), "23:59 should map to next day")

    def test_complaint_at_0000_assigned_to_current_day(self):
        """Complaint at midnight (00:00) should be assigned to current day."""
        nyc_tz = pytz.timezone("America/New_York")
        ts = nyc_tz.localize(datetime(2023, 7, 16, 0, 0))  # Midnight on July 16
        night_date = self._assign_night_date(ts)
        self.assertEqual(night_date, date(2023, 7, 16), "00:00 should map to current day")

    def test_complaint_at_0300_assigned_to_current_day(self):
        """Complaint at 03:00 should be assigned to current day."""
        nyc_tz = pytz.timezone("America/New_York")
        ts = nyc_tz.localize(datetime(2023, 7, 16, 3, 0))  # 3 AM on July 16
        night_date = self._assign_night_date(ts)
        self.assertEqual(night_date, date(2023, 7, 16), "03:00 should map to current day")

    def test_complaint_at_0659_assigned_to_current_day(self):
        """Complaint at 06:59 should be assigned to current day."""
        nyc_tz = pytz.timezone("America/New_York")
        ts = nyc_tz.localize(datetime(2023, 7, 16, 6, 59))  # 6:59 AM on July 16
        night_date = self._assign_night_date(ts)
        self.assertEqual(night_date, date(2023, 7, 16), "06:59 should map to current day")

    def test_month_boundary_dec31_to_jan1(self):
        """New Year's Eve 22:00 should map to January 1."""
        nyc_tz = pytz.timezone("America/New_York")
        ts = nyc_tz.localize(datetime(2022, 12, 31, 22, 30))  # 10:30 PM on Dec 31
        night_date = self._assign_night_date(ts)
        self.assertEqual(night_date, date(2023, 1, 1), "Dec 31 22:30 should map to Jan 1")

    def test_month_boundary_jan1_early_morning(self):
        """January 1 at 03:00 should map to January 1."""
        nyc_tz = pytz.timezone("America/New_York")
        ts = nyc_tz.localize(datetime(2023, 1, 1, 3, 0))  # 3 AM on Jan 1
        night_date = self._assign_night_date(ts)
        self.assertEqual(night_date, date(2023, 1, 1), "Jan 1 03:00 should map to Jan 1")

    def test_warm_season_boundary_june1(self):
        """May 31 at 23:00 should map to June 1 (warm season start)."""
        nyc_tz = pytz.timezone("America/New_York")
        ts = nyc_tz.localize(datetime(2023, 5, 31, 23, 0))  # 11 PM on May 31
        night_date = self._assign_night_date(ts)
        self.assertEqual(night_date, date(2023, 6, 1), "May 31 23:00 should map to June 1")


class TestNighttimeFiltering(unittest.TestCase):
    """Tests for nighttime filtering logic."""

    def test_is_nighttime_cross_midnight(self):
        """Test is_nighttime correctly handles cross-midnight window."""
        nyc_tz = pytz.timezone("America/New_York")

        # Create timestamps for different hours
        test_cases = [
            (22, True),   # 10 PM - in window
            (23, True),   # 11 PM - in window
            (0, True),    # Midnight - in window
            (1, True),    # 1 AM - in window
            (6, True),    # 6 AM - in window
            (7, False),   # 7 AM - NOT in window (end boundary)
            (8, False),   # 8 AM - NOT in window
            (12, False),  # Noon - NOT in window
            (21, False),  # 9 PM - NOT in window (before start)
        ]

        for hour, expected in test_cases:
            ts = pd.Series([nyc_tz.localize(datetime(2023, 7, 15, hour, 30))])
            result = is_nighttime(ts, start_hour=22, end_hour=7)
            self.assertEqual(
                result.iloc[0], expected,
                f"Hour {hour:02d}:30 should be nighttime={expected}"
            )

    def test_filter_nighttime_empty_input(self):
        """filter_nighttime should handle empty DataFrames."""
        nyc_tz = pytz.timezone("America/New_York")
        df = pd.DataFrame({'ts_nyc': pd.Series([], dtype='datetime64[ns, America/New_York]')})
        result = filter_nighttime(df, 'ts_nyc', start_hour=22, end_hour=7)
        self.assertEqual(len(result), 0)

    def test_filter_nighttime_all_daytime(self):
        """filter_nighttime should return empty when all records are daytime."""
        nyc_tz = pytz.timezone("America/New_York")
        daytime_hours = [8, 10, 12, 14, 16, 18]
        timestamps = [nyc_tz.localize(datetime(2023, 7, 15, h, 0)) for h in daytime_hours]
        df = pd.DataFrame({'ts_nyc': timestamps})
        result = filter_nighttime(df, 'ts_nyc', start_hour=22, end_hour=7, ensure_timezone=False)
        self.assertEqual(len(result), 0, "All daytime records should be filtered out")


class TestStatisticalModel(unittest.TestCase):
    """Tests for the Poisson GLM and statistical calculations."""

    def test_slope_to_percent_formula(self):
        """Verify slope-to-percent formula: (exp(slope) - 1) * 100."""
        # Test with known slopes
        test_cases = [
            (0.0, 0.0),           # Zero slope = 0% change
            (0.01, 1.005),        # Small positive
            (-0.01, -0.995),      # Small negative
            (0.05, 5.127),        # Moderate positive
            (np.log(1.1), 10.0),  # ln(1.1) should give exactly 10%
        ]

        for slope, expected_approx in test_cases:
            pct = (np.exp(slope) - 1) * 100
            self.assertAlmostEqual(
                pct, expected_approx, places=2,
                msg=f"slope={slope} should give ~{expected_approx}%"
            )

    def test_poisson_model_interpretation(self):
        """
        Verify understanding of Poisson GLM coefficients.

        In Poisson regression: log(E[Y]) = β₀ + β₁*X
        So: E[Y] = exp(β₀) * exp(β₁*X)

        For a 1-unit increase in X:
        E[Y|X+1] / E[Y|X] = exp(β₁)

        Percent change = (exp(β₁) - 1) * 100
        """
        # This is a conceptual test to document the formula
        slope = 0.02  # 2% increase in log-rate per degree
        rate_ratio = np.exp(slope)
        pct_change = (rate_ratio - 1) * 100

        # If slope is 0.02, rate ratio is exp(0.02) ≈ 1.0202
        # So complaints increase by ~2.02% per degree
        self.assertAlmostEqual(rate_ratio, 1.0202, places=3)
        self.assertAlmostEqual(pct_change, 2.02, places=1)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and failure modes."""

    def test_zero_complaint_night_handling(self):
        """
        Verify that zero-complaint nights are properly included.

        The model should include nights with zero complaints to avoid
        selection bias (only modeling nights when people complained).
        """
        df = pd.read_parquet(PROCESSED_DIR / "weather" / "cd_heat_sensitivity.parquet")

        # n_obs should equal the number of nights in the analysis period
        # If zero-complaint nights were excluded, n_obs would be much lower
        # for low-volume CDs

        # All CDs should have similar n_obs (within reason)
        n_obs_min = df['n_obs'].min()
        n_obs_max = df['n_obs'].max()
        n_obs_ratio = n_obs_max / n_obs_min

        # Ratio should be close to 1 if all CDs have same observation count
        self.assertLess(n_obs_ratio, 1.5, "n_obs varies too much across CDs - check zero handling")

    def test_missing_tmin_handling(self):
        """
        Check that model handles missing temperature data.

        The outputs should have valid slopes for all 59 CDs.
        """
        df = pd.read_parquet(PROCESSED_DIR / "weather" / "cd_heat_sensitivity.parquet")

        # All 59 CDs should have slopes (no NaN from model failures)
        self.assertEqual(df['slope'].isna().sum(), 0, "Some CDs have NaN slopes")
        self.assertEqual(df['se'].isna().sum(), 0, "Some CDs have NaN standard errors")


if __name__ == '__main__':
    unittest.main()
