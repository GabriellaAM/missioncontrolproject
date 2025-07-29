from feast import Field, FeatureView, FileSource
from feast.types import Bool, String, Int32
from feast.data_format import ParquetFormat
from datetime import timedelta
from pathlib import Path
from .entities import asset_id

"""
Asset Category Features

This module defines cryptocurrency asset category features including:
- Primary category classification
- One-hot encoded boolean features for key categories (DeFi, Layer 1, Meme, etc.)
- Category count metrics
- Ecosystem membership flags

The data source points to the asset categories parquet file:
data_parquet/asset_categories/data.parquet

These features are designed for ML experiments and can be easily joined
with market data using the shared asset_id entity.
"""

# Get the project root directory
FEATURES_DIR = Path(__file__).parent
PROJECT_ROOT = FEATURES_DIR.parent.parent

# Build the path to the asset categories data
DATA_PATH = str(PROJECT_ROOT / "data_parquet" / "asset_categories" / "data.parquet")

# Define the data source for asset categories
asset_category_source = FileSource(
    path=DATA_PATH,
    timestamp_field="timestamp",
    file_format=ParquetFormat(),
    s3_endpoint_override=None  # Using local filesystem
)

# Create the feature view for asset categories
asset_category_features = FeatureView(
    name="asset_category_features",
    entities=[asset_id],
    ttl=timedelta(days=0),  # No expiration - categories don't change frequently
    schema=[
        # Basic category information
        Field(name="primary_category", dtype=String),
        Field(name="category_count", dtype=Int32),
        Field(name="all_categories", dtype=String),
        
        # Core category boolean features for ML
        Field(name="is_meme", dtype=Bool),
        Field(name="is_decentralized_finance_defi", dtype=Bool),
        Field(name="is_layer_1_l1", dtype=Bool),
        Field(name="is_layer_2_l2", dtype=Bool),
        Field(name="is_gaming", dtype=Bool),
        Field(name="is_nft", dtype=Bool),
        Field(name="is_oracle", dtype=Bool),
        Field(name="is_stablecoin", dtype=Bool),
        Field(name="is_infrastructure", dtype=Bool),
        Field(name="is_decentralized_exchange_dex", dtype=Bool),
        Field(name="is_smart_contract_platform", dtype=Bool),
        Field(name="is_artificial_intelligence_ai", dtype=Bool),
        
        # Ecosystem boolean features
        Field(name="is_ethereum_ecosystem", dtype=Bool),
        Field(name="is_solana_ecosystem", dtype=Bool),
        Field(name="is_bnb_chain_ecosystem", dtype=Bool),
        Field(name="is_base_ecosystem", dtype=Bool),
        Field(name="is_arbitrum_ecosystem", dtype=Bool),
        Field(name="is_polygon_ecosystem", dtype=Bool),
    ],
    source=asset_category_source,
    tags={
        "type": "metadata", 
        "use_case": "backtesting",
    }
)