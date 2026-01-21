"""
Acoustic utilities for noise data processing.

Per R9: dB noise aggregation is energy-averaged and unit-tested.
- Arithmetic mean of dB pixels is FORBIDDEN.
- Use energy_mean_db() and enforce via unit test.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd


class AcousticError(Exception):
    """Raised when acoustic calculation fails."""
    pass


def db_to_linear(db_values: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert decibel values to linear (power) scale.
    
    Formula: linear = 10^(dB/10)
    
    Args:
        db_values: Decibel value(s)
    
    Returns:
        Linear power value(s)
    """
    return 10 ** (np.asarray(db_values) / 10)


def linear_to_db(linear_values: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert linear (power) values to decibels.
    
    Formula: dB = 10 * log10(linear)
    
    Args:
        linear_values: Linear power value(s)
    
    Returns:
        Decibel value(s)
    """
    linear_arr = np.asarray(linear_values)
    
    # Handle zero/negative values
    if np.any(linear_arr <= 0):
        raise AcousticError("Cannot convert zero or negative values to dB")
    
    return 10 * np.log10(linear_arr)


def energy_mean_db(db_values: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Compute energy-averaged mean of decibel values.
    
    Per R9: This is the ONLY correct way to average dB values.
    Arithmetic mean of dB is physically incorrect.
    
    Formula:
        1. Convert dB to linear: linear = 10^(dB/10)
        2. Compute arithmetic mean of linear values
        3. Convert back to dB: dB = 10 * log10(mean_linear)
    
    Args:
        db_values: Array of decibel values
    
    Returns:
        Energy-averaged mean in dB
    
    Example:
        >>> energy_mean_db([60, 60])  # Two equal sources
        63.01  # NOT 60! Energy doubles.
        
        >>> energy_mean_db([70, 60])  # 70 dB dominates
        67.4   # NOT 65! High values dominate.
    """
    db_arr = np.asarray(db_values)
    
    # Remove NaN values
    db_arr = db_arr[~np.isnan(db_arr)]
    
    if len(db_arr) == 0:
        return np.nan
    
    if len(db_arr) == 1:
        return float(db_arr[0])
    
    # Convert to linear, mean, convert back
    linear = db_to_linear(db_arr)
    mean_linear = np.mean(linear)
    return float(linear_to_db(mean_linear))


def energy_sum_db(db_values: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Compute energy sum of decibel values (total combined level).
    
    This is the total sound level when multiple sources combine.
    
    Args:
        db_values: Array of decibel values
    
    Returns:
        Combined level in dB
    """
    db_arr = np.asarray(db_values)
    db_arr = db_arr[~np.isnan(db_arr)]
    
    if len(db_arr) == 0:
        return np.nan
    
    linear = db_to_linear(db_arr)
    sum_linear = np.sum(linear)
    return float(linear_to_db(sum_linear))


def arithmetic_mean_db_FORBIDDEN(db_values: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    THIS FUNCTION EXISTS ONLY TO DOCUMENT WHAT NOT TO DO.
    
    Per R9: Arithmetic mean of dB is FORBIDDEN.
    
    DO NOT USE THIS FUNCTION.
    It exists only so that unit tests can verify that:
    - energy_mean_db() gives different (correct) results
    - Tests can demonstrate why arithmetic mean is wrong
    
    Args:
        db_values: Array of decibel values
    
    Returns:
        INCORRECT arithmetic mean
    
    Raises:
        AcousticError: Always raises to prevent accidental use
    """
    raise AcousticError(
        "Arithmetic mean of dB values is FORBIDDEN. "
        "Use energy_mean_db() instead. See R9 in research plan."
    )


def percentile_db(db_values: Union[List[float], np.ndarray, pd.Series], percentile: float) -> float:
    """
    Compute percentile of dB values.
    
    Note: Percentiles of dB values are valid (unlike mean).
    Common acoustic metrics: L10, L50, L90, Lmax, Lmin
    
    Args:
        db_values: Array of decibel values
        percentile: Percentile to compute (0-100)
    
    Returns:
        Percentile value in dB
    """
    db_arr = np.asarray(db_values)
    db_arr = db_arr[~np.isnan(db_arr)]
    
    if len(db_arr) == 0:
        return np.nan
    
    return float(np.percentile(db_arr, percentile))


def leq_from_samples(
    db_values: Union[List[float], np.ndarray, pd.Series],
    duration_seconds: Optional[float] = None,
) -> float:
    """
    Compute equivalent continuous sound level (Leq) from samples.
    
    Leq is the energy-averaged level over a time period.
    This is equivalent to energy_mean_db for equal-duration samples.
    
    Args:
        db_values: Array of sound level measurements
        duration_seconds: Optional total duration (for reference only)
    
    Returns:
        Leq in dB
    """
    return energy_mean_db(db_values)


def day_night_level(
    day_db: float,
    night_db: float,
    day_hours: float = 15,  # 7 AM - 10 PM
    night_hours: float = 9,  # 10 PM - 7 AM
    night_penalty: float = 10,
) -> float:
    """
    Compute day-night average sound level (Ldn or DNL).
    
    Night levels receive a penalty (typically 10 dB) to account
    for increased sensitivity during sleep hours.
    
    Args:
        day_db: Daytime average level (dB)
        night_db: Nighttime average level (dB)
        day_hours: Number of daytime hours (default 15)
        night_hours: Number of nighttime hours (default 9)
        night_penalty: Penalty added to night levels (default 10 dB)
    
    Returns:
        Ldn in dB
    """
    total_hours = day_hours + night_hours
    
    # Weight by hours and apply night penalty
    day_linear = db_to_linear(day_db) * day_hours
    night_linear = db_to_linear(night_db + night_penalty) * night_hours
    
    # Energy average
    mean_linear = (day_linear + night_linear) / total_hours
    
    return float(linear_to_db(mean_linear))


# =============================================================================
# Validation helpers for unit tests
# =============================================================================

def demonstrate_energy_vs_arithmetic():
    """
    Demonstrate why energy averaging differs from arithmetic averaging.
    
    Returns dictionary with examples showing the difference.
    Used in unit tests to verify correct implementation.
    """
    examples = []
    
    # Example 1: Two equal sources
    values1 = [60, 60]
    arith1 = np.mean(values1)
    energy1 = energy_mean_db(values1)
    examples.append({
        "description": "Two equal 60 dB sources",
        "values": values1,
        "arithmetic_mean": arith1,  # 60
        "energy_mean": energy1,      # ~63 (energy doubles = +3 dB)
        "difference": energy1 - arith1,
    })
    
    # Example 2: Unequal sources (high dominates)
    values2 = [70, 60]
    arith2 = np.mean(values2)
    energy2 = energy_mean_db(values2)
    examples.append({
        "description": "70 dB and 60 dB sources",
        "values": values2,
        "arithmetic_mean": arith2,  # 65
        "energy_mean": energy2,      # ~67.4 (70 dB dominates)
        "difference": energy2 - arith2,
    })
    
    # Example 3: Very unequal (one dominates completely)
    values3 = [80, 50]
    arith3 = np.mean(values3)
    energy3 = energy_mean_db(values3)
    examples.append({
        "description": "80 dB and 50 dB sources",
        "values": values3,
        "arithmetic_mean": arith3,  # 65
        "energy_mean": energy3,      # ~80 (80 dB completely dominates)
        "difference": energy3 - arith3,
    })
    
    return examples

