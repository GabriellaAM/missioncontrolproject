from feast import Field, FeatureView, FileSource, ValueType
from feast.types import Float64, String
from feast.data_format import ParquetFormat
from datetime import timedelta
from pathlib import Path
import os
from .entities import macro_asset_id

"""
Macro Yahoo Finance Features

This module defines macro economic data features from Yahoo Finance including:
- OHLCV data (Open, High, Low, Close, Volume)
- Market indices (VIX, MOVE)
- Currency indices (DXY, USD/JPY, USD/EUR)
- Commodities (Gold, Copper, Oil)
- Futures (ES, YM, RTY)
- ETFs (HYG)

The data source points to the partitioned parquet files structure:
data_parquet/macro_data/{macro_asset_id}/data.parquet

Assets included:
- VIX (^VIX): Volatility Index
- MOVE (^MOVE): Treasury volatility
- DXY (DX-Y.NYB): Dollar Index
- USD/JPY (JPY=X): US Dollar / Japanese Yen
- USD/EUR (EUR=X): US Dollar / Euro
- Gold (GC=F): Gold futures
- Copper (HG=F): Copper futures  
- Oil (CL=F): Crude oil futures
- ES (ES=F): S&P 500 E-mini futures
- YM (YM=F): Dow Jones E-mini futures
- RTY (RTY=F): Russell 2000 E-mini futures
- HYG (HYG): High Yield Corporate Bond ETF
"""

# Get the project root directory (assuming this file is in soros_features/features/)
FEATURES_DIR = Path(__file__).parent
PROJECT_ROOT = FEATURES_DIR.parent.parent

# Build the relative path to the Yahoo macro data directory
# This will work regardless of where the project is cloned
DATA_PATH = str(PROJECT_ROOT / "data_parquet" / "macro_data" / "yahoo" / "*" / "data.parquet")

# Define the data source pointing to our partitioned parquet structure
# Now points to yahoo-specific subdirectory
macro_yahoo_source = FileSource(
    path=DATA_PATH,
    timestamp_field="timestamp",
    file_format=ParquetFormat(),
    s3_endpoint_override=None  # Using local filesystem
)

# Create the feature view for Yahoo Finance macro data (OHLCV format)
macro_yahoo_features = FeatureView(
    name="macro_yahoo_features",
    entities=[macro_asset_id],
    ttl=timedelta(days=0),  # No expiration - we want all historical data for backtesting
    schema=[
        Field(name="open", dtype=Float64),
        Field(name="high", dtype=Float64), 
        Field(name="low", dtype=Float64),
        Field(name="close", dtype=Float64),
        Field(name="volume", dtype=Float64),
    ],
    source=macro_yahoo_source,
    tags={"type": "macro_data", "source": "yahoo", "use_case": "backtesting"}
)