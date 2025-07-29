from feast import Field, FeatureView, FileSource
from feast.types import Float64
from feast.data_format import ParquetFormat
from datetime import timedelta
from pathlib import Path

"""
Market Dominance Features

This module defines market-wide dominance metrics including:
- BTC dominance (BTC market cap / total crypto market cap)
- Stablecoin dominance (USDT + USDC market cap / total crypto market cap)
- Total market capitalization

The data source points to the market dominance parquet file:
data_parquet/market_data/dominance/data.parquet

These features provide important market context for trading strategies
and risk management.
"""

# Get the project root directory
FEATURES_DIR = Path(__file__).parent
PROJECT_ROOT = FEATURES_DIR.parent.parent

# Build the path to the dominance data
DATA_PATH = str(PROJECT_ROOT / "data_parquet" / "market_data" / "dominance" / "data.parquet")

# Define the data source for market dominance
market_dominance_source = FileSource(
    path=DATA_PATH,
    timestamp_field="timestamp",
    file_format=ParquetFormat(),
    s3_endpoint_override=None  # Using local filesystem
)

# Create the feature view for market dominance metrics
market_dominance_features = FeatureView(
    name="market_dominance_features",
    entities=[],  # No entities - these are global market metrics
    ttl=timedelta(days=0),  # No expiration - we want all historical data
    schema=[
        Field(name="btc_dominance", dtype=Float64),
        Field(name="stablecoin_dominance", dtype=Float64),
        Field(name="total_market_cap", dtype=Float64),
    ],
    source=market_dominance_source,
    tags={
        "type": "market_metrics",
        "use_case": "backtesting",
        "frequency": "daily",
        "scope": "global"
    }
)