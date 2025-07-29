from feast import Field, FeatureView, FileSource, ValueType
from feast.types import Float64, String
from feast.data_format import ParquetFormat
from datetime import timedelta
from pathlib import Path
import os
from .entities import macro_asset_id

"""
Macro Calculated Features

This module defines calculated/derived macro economic features including:
- Yield curve regime classification
- Financial ratios and spreads
- Cross-asset relationships

These features are computed from other macro data sources and have custom schemas
depending on the specific calculation performed.

The data source points to the partitioned parquet files structure:
data_parquet/macro_data/{macro_asset_id}/data.parquet

Current calculated features:
- yieldCurveRegime: Classification of yield curve shape and direction
  - Fields: regime (string), spread_2s10s (float)
  - Regimes: bull_steepener, bull_flattener, bear_steepener, bear_flattener, neutral
- rtyYmRatio: Russell 2000 to Dow Jones ratio (small vs large cap performance)
  - Fields: ratio (float)
"""

# Get the project root directory (assuming this file is in soros_features/features/)
FEATURES_DIR = Path(__file__).parent
PROJECT_ROOT = FEATURES_DIR.parent.parent

# Build the relative path to the calculated macro data directory
DATA_PATH = str(PROJECT_ROOT / "data_parquet" / "macro_data" / "calculated" / "*" / "data.parquet")

# Define the data source pointing to our partitioned parquet structure
# Now points to calculated-specific subdirectory
macro_calculated_source = FileSource(
    path=DATA_PATH,
    timestamp_field="timestamp",
    file_format=ParquetFormat(),
    s3_endpoint_override=None  # Using local filesystem
)

# Create the feature view for yield curve regime data
yield_curve_regime_features = FeatureView(
    name="yield_curve_regime_features",
    entities=[macro_asset_id],
    ttl=timedelta(days=0),  # No expiration - we want all historical data for backtesting
    schema=[
        Field(name="regime", dtype=String),
        Field(name="spread_2s10s", dtype=Float64),
        Field(name="frequency", dtype=String),
    ],
    source=macro_calculated_source,
    tags={"type": "macro_data", "source": "calculated", "use_case": "backtesting", "calculation": "yield_curve_regime"}
)

# Create the feature view for RTY/YM ratio data
rty_ym_ratio_features = FeatureView(
    name="rty_ym_ratio_features", 
    entities=[macro_asset_id],
    ttl=timedelta(days=0),  # No expiration - we want all historical data for backtesting
    schema=[
        Field(name="ratio", dtype=Float64),
        Field(name="frequency", dtype=String),
    ],
    source=macro_calculated_source,
    tags={"type": "macro_data", "source": "calculated", "use_case": "backtesting", "calculation": "rty_ym_ratio"}
)