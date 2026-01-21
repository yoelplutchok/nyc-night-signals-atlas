"""
Schema validation for canonical outputs.

Per R10: Schema validation is mandatory.
- Canonical outputs validated (columns, dtypes, NA rules) on read and write.
- Schema drift becomes an immediate local failure.

Per R4: Keys and dtypes are frozen.
- boro_cd is always pandas nullable Int64.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type, Union

import geopandas as gpd
import pandas as pd


# =============================================================================
# Schema Definition
# =============================================================================

@dataclass
class ColumnSpec:
    """Specification for a single column."""
    name: str
    dtype: Optional[str] = None  # e.g., "Int64", "float64", "object", "geometry"
    nullable: bool = True
    unique: bool = False
    allowed_values: Optional[Set[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class Schema:
    """Schema specification for a DataFrame or GeoDataFrame."""
    name: str
    columns: List[ColumnSpec]
    required_columns: List[str] = field(default_factory=list)
    row_count: Optional[int] = None  # Exact expected row count
    min_rows: int = 0
    
    def __post_init__(self):
        # Auto-populate required_columns if not specified
        if not self.required_columns:
            self.required_columns = [c.name for c in self.columns if not c.nullable]


class SchemaError(Exception):
    """Raised when schema validation fails."""
    pass


# =============================================================================
# Predefined Schemas (Per Research Plan)
# =============================================================================

# CD59 geometry schema (Section 3.1)
CD59_SCHEMA = Schema(
    name="cd59",
    columns=[
        ColumnSpec("boro_cd", dtype="Int64", nullable=False, unique=True),
        ColumnSpec("geometry", dtype="geometry", nullable=False),
    ],
    row_count=59,
)

# Crosswalk schema (Section 3.4)
CROSSWALK_SCHEMA = Schema(
    name="crosswalk",
    columns=[
        ColumnSpec("boro_cd", dtype="Int64", nullable=False),
        ColumnSpec("w_pop", dtype="float64", nullable=True, min_value=0, max_value=1),
        ColumnSpec("w_area", dtype="float64", nullable=True, min_value=0, max_value=1),
    ],
    min_rows=1,
)

# Domain output base schema (Section 4)
DOMAIN_BASE_SCHEMA = Schema(
    name="domain_base",
    columns=[
        ColumnSpec("boro_cd", dtype="Int64", nullable=False, unique=True),
        ColumnSpec("year_start", dtype="Int64", nullable=False),
        ColumnSpec("year_end", dtype="Int64", nullable=False),
    ],
    min_rows=1,
)

# 311 Noise domain schema
NOISE_311_SCHEMA = Schema(
    name="noise_311",
    columns=[
        ColumnSpec("boro_cd", dtype="Int64", nullable=False, unique=True),
        ColumnSpec("year_start", dtype="Int64", nullable=False),
        ColumnSpec("year_end", dtype="Int64", nullable=False),
        ColumnSpec("noise311_count", dtype="Int64", nullable=False, min_value=0),
        ColumnSpec("noise311_rate_per_1k_pop", dtype="float64", nullable=True, min_value=0),
        ColumnSpec("noise311_rate_per_km2", dtype="float64", nullable=True, min_value=0),
    ],
    min_rows=1,
)

# Index output schema
INDEX_SCHEMA = Schema(
    name="index",
    columns=[
        ColumnSpec("boro_cd", dtype="Int64", nullable=False, unique=True),
        ColumnSpec("year_start", dtype="Int64", nullable=False),
        ColumnSpec("year_end", dtype="Int64", nullable=False),
    ],
    min_rows=1,
)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_column(
    df: pd.DataFrame,
    spec: ColumnSpec,
    context: str = "",
) -> List[str]:
    """
    Validate a single column against its specification.
    
    Args:
        df: DataFrame containing the column
        spec: Column specification
        context: Optional context for error messages
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    col_name = spec.name
    
    # Check column exists
    if col_name not in df.columns:
        if col_name in ["geometry"] and isinstance(df, gpd.GeoDataFrame):
            pass  # geometry column handled specially
        else:
            errors.append(f"Missing column: {col_name}")
            return errors
    
    col = df[col_name]
    
    # Check dtype
    if spec.dtype is not None:
        if spec.dtype == "geometry":
            if not isinstance(df, gpd.GeoDataFrame):
                errors.append(f"Expected GeoDataFrame for geometry column {col_name}")
        elif spec.dtype == "Int64":
            if not pd.api.types.is_integer_dtype(col):
                errors.append(f"Column {col_name}: expected Int64, got {col.dtype}")
        elif spec.dtype == "float64":
            if not pd.api.types.is_float_dtype(col):
                errors.append(f"Column {col_name}: expected float64, got {col.dtype}")
    
    # Check nullable
    if not spec.nullable and col.isna().any():
        na_count = col.isna().sum()
        errors.append(f"Column {col_name}: {na_count} NA values not allowed")
    
    # Check unique
    if spec.unique and col.duplicated().any():
        dup_count = col.duplicated().sum()
        errors.append(f"Column {col_name}: {dup_count} duplicate values not allowed")
    
    # Check allowed values
    if spec.allowed_values is not None:
        invalid = ~col.isin(spec.allowed_values) & col.notna()
        if invalid.any():
            invalid_vals = col[invalid].unique()[:5]
            errors.append(f"Column {col_name}: invalid values {list(invalid_vals)}")
    
    # Check value range
    if spec.min_value is not None:
        below_min = (col < spec.min_value) & col.notna()
        if below_min.any():
            errors.append(f"Column {col_name}: values below min {spec.min_value}")
    
    if spec.max_value is not None:
        above_max = (col > spec.max_value) & col.notna()
        if above_max.any():
            errors.append(f"Column {col_name}: values above max {spec.max_value}")
    
    return errors


def validate_schema(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    schema: Schema,
    context: str = "",
    raise_on_error: bool = True,
) -> List[str]:
    """
    Validate a DataFrame against a schema.
    
    Per R10: Schema validation on read and write.
    
    Args:
        df: DataFrame to validate
        schema: Schema specification
        context: Optional context for error messages
        raise_on_error: If True, raise SchemaError on validation failure
    
    Returns:
        List of error messages (empty if valid)
    
    Raises:
        SchemaError: If raise_on_error=True and validation fails
    """
    errors = []
    ctx = f" ({context})" if context else ""
    
    # Check row count
    if schema.row_count is not None and len(df) != schema.row_count:
        errors.append(f"Expected {schema.row_count} rows, got {len(df)}{ctx}")
    
    if len(df) < schema.min_rows:
        errors.append(f"Expected at least {schema.min_rows} rows, got {len(df)}{ctx}")
    
    # Check required columns exist
    missing = set(schema.required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}{ctx}")
    
    # Validate each column
    for col_spec in schema.columns:
        col_errors = validate_column(df, col_spec, context)
        errors.extend(col_errors)
    
    if errors and raise_on_error:
        raise SchemaError(f"Schema validation failed for '{schema.name}':\n" + "\n".join(errors))
    
    return errors


def validate_boro_cd(
    df: pd.DataFrame,
    context: str = "",
) -> None:
    """
    Validate that boro_cd column meets requirements.
    
    Per R4: boro_cd is always pandas nullable Int64.
    
    Args:
        df: DataFrame with boro_cd column
        context: Optional context for error messages
    
    Raises:
        SchemaError: If boro_cd validation fails
    """
    if "boro_cd" not in df.columns:
        raise SchemaError(f"Missing boro_cd column ({context})")
    
    col = df["boro_cd"]
    
    # Check dtype is Int64 (nullable integer)
    if col.dtype != "Int64":
        raise SchemaError(
            f"boro_cd must be Int64, got {col.dtype} ({context}). "
            f"Convert with: df['boro_cd'] = df['boro_cd'].astype('Int64')"
        )
    
    # Check for NA values
    if col.isna().any():
        na_count = col.isna().sum()
        raise SchemaError(f"boro_cd has {na_count} NA values ({context})")


def ensure_boro_cd_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure boro_cd column is Int64 dtype.
    
    Args:
        df: DataFrame with boro_cd column
    
    Returns:
        DataFrame with boro_cd as Int64
    """
    if "boro_cd" in df.columns:
        df = df.copy()
        df["boro_cd"] = df["boro_cd"].astype("Int64")
    return df


# =============================================================================
# Merge Validation (R4)
# =============================================================================

def validate_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, List[str]],
    how: str = "left",
    validate: str = "one_to_one",
    context: str = "",
) -> pd.DataFrame:
    """
    Perform a merge with validation.
    
    Per R4: All merges use validate= to catch duplicates.
    
    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Column(s) to merge on
        how: Merge type
        validate: Merge validation ('one_to_one', 'one_to_many', 'many_to_one')
        context: Context for error messages
    
    Returns:
        Merged DataFrame
    
    Raises:
        ValueError: If merge validation fails
    """
    try:
        return pd.merge(left, right, on=on, how=how, validate=validate)
    except pd.errors.MergeError as e:
        raise ValueError(f"Merge validation failed ({context}): {e}")


# =============================================================================
# Schema Registry
# =============================================================================

SCHEMAS: Dict[str, Schema] = {
    "cd59": CD59_SCHEMA,
    "crosswalk": CROSSWALK_SCHEMA,
    "domain_base": DOMAIN_BASE_SCHEMA,
    "noise_311": NOISE_311_SCHEMA,
    "index": INDEX_SCHEMA,
}


def get_schema(name: str) -> Schema:
    """Get a registered schema by name."""
    if name not in SCHEMAS:
        raise KeyError(f"Unknown schema: {name}. Available: {list(SCHEMAS.keys())}")
    return SCHEMAS[name]


def register_schema(schema: Schema) -> None:
    """Register a new schema."""
    SCHEMAS[schema.name] = schema

