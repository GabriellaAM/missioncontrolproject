from feast import Field, FeatureView, FileSource, ValueType
from feast.types import Float64, String
from feast.data_format import ParquetFormat
from datetime import timedelta
from pathlib import Path
import os
from .entities import macro_asset_id

"""
Macro FRED Features

This module defines macro economic data features from FRED (Federal Reserve Economic Data) including:
- Treasury rates (1Y, 2Y, 3M, 6M, 10Y)
- Monetary policy rates (Fed funds rate, SOFR, IORB)
- Economic indicators (unemployment, CPI, GDP, etc.)
- Financial conditions (credit spreads, indices)
- Central bank data (Fed assets, repo agreements, etc.)

The data source points to the partitioned parquet files structure:
data_parquet/macro_data/{macro_asset_id}/data.parquet

Key assets included:
- Treasury rates: treasury1Y, treasury2Y, treasury3M, treasury6M, treasury10Y
- Policy rates: fedFundsRate, securedOvernightFinancingRate, interestOnReserves
- Employment: unemploymentRate, nonfarmPayrolls, initialClaims, jobOpenings, jobQuits
- Inflation: consumerPriceIndex, personalConsumptionExpenditures, treasury5YInflationExpectation
- Markets: sp500, nasdaq, dollarIndex, creditSpreads, creditSpreadsHighGrade
- Central banks: fedTotalAssets, reserveBalances, repoAgreements, bojAssets, ecbAssets
- Economy: realGDP, nominalGDP, m2, usTotalDebt, personalSavingsRate
"""

# Get the project root directory (assuming this file is in soros_features/features/)
FEATURES_DIR = Path(__file__).parent
PROJECT_ROOT = FEATURES_DIR.parent.parent

# Build the relative path to the FRED macro data directory
DATA_PATH = str(PROJECT_ROOT / "data_parquet" / "macro_data" / "fred" / "*" / "data.parquet")

# Define the data source pointing to our partitioned parquet structure
# Now points to fred-specific subdirectory
macro_fred_source = FileSource(
    path=DATA_PATH,
    timestamp_field="timestamp",
    file_format=ParquetFormat(),
    s3_endpoint_override=None  # Using local filesystem
)

# Create the feature view for FRED macro data (single value format)
macro_fred_features = FeatureView(
    name="macro_fred_features",
    entities=[macro_asset_id],
    ttl=timedelta(days=0),  # No expiration - we want all historical data for backtesting
    schema=[
        Field(name="value", dtype=Float64),
        Field(name="frequency", dtype=String),
    ],
    source=macro_fred_source,
    tags={"type": "macro_data", "source": "fred", "use_case": "backtesting"}
)