from feast import Field, FeatureView, FileSource, ValueType
from feast.types import Float64, String
from feast.data_format import ParquetFormat
from datetime import timedelta
from pathlib import Path
import os
from .entities import asset_id

"""
Crypto Market Features

This module defines the basic cryptocurrency market data features including:
- OHLCV (Open, High, Low, Close, Volume)
- Market capitalization
- BTC-relative pricing features

The data source points to the partitioned parquet files structure:
data_parquet/crypto_data/coingecko/{asset_id}/data.parquet
"""

# Get the project root directory (assuming this file is in soros_features/features/)
FEATURES_DIR = Path(__file__).parent
PROJECT_ROOT = FEATURES_DIR.parent.parent

# Build the relative path to the coingecko crypto data directory
# This will work regardless of where the project is cloned
# Using single * instead of ** for better DuckDB compatibility
DATA_PATH = str(PROJECT_ROOT / "data_parquet" / "crypto_data" / "coingecko" / "*" / "data.parquet")

# Alternative: Use environment variable if set, otherwise use relative path
# DATA_PATH = os.getenv("CRYPTO_DATA_PATH", DATA_PATH)

# Define the data source pointing to our partitioned parquet structure
crypto_market_source = FileSource(
    path=DATA_PATH,
    timestamp_field="timestamp",
    file_format=ParquetFormat(),  # Explicitly specify parquet format
    # Note: created_timestamp_column is optional and should be different from timestamp_field
    # We don't track when data was ingested, only when events occurred
    s3_endpoint_override=None  # Using local filesystem
)

# Create the feature view with all basic crypto market features
crypto_market_features = FeatureView(
    name="crypto_market_features",
    entities=[asset_id],
    ttl=timedelta(days=0),  # No expiration - we want all historical data for backtesting
    schema=[
        Field(name="open", dtype=Float64),
        Field(name="high", dtype=Float64), 
        Field(name="low", dtype=Float64),
        Field(name="close", dtype=Float64),
        Field(name="market_cap", dtype=Float64),
        Field(name="total_volume", dtype=Float64),
        Field(name="symbol", dtype=String),
        # BTC-relative features
        Field(name="open_btc", dtype=Float64),
        Field(name="high_btc", dtype=Float64),
        Field(name="low_btc", dtype=Float64),
        Field(name="close_btc", dtype=Float64),
        Field(name="volume_btc", dtype=Float64),
    ],
    source=crypto_market_source,
    tags={"type": "market_data", "use_case": "backtesting"}
)