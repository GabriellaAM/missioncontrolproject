"""
Soros Features Repository

This is the main feature repository file that Feast uses to discover
all entities, data sources, and feature views in the project.

Import all feature definitions here to make them available to Feast.
"""

# Import all entities
from features.entities import asset_id

# Import all feature views
from features.crypto_market_features import crypto_market_features, crypto_market_source
from features.asset_category_features import asset_category_features, asset_category_source
from features.market_dominance_features import market_dominance_features, market_dominance_source

# List all objects that should be registered with Feast
# Feast will automatically discover these when running feast commands
__all__ = [
    # Entities
    "asset_id",
    # Data sources
    "crypto_market_source",
    "asset_category_source",
    "market_dominance_source",
    # Feature views
    "crypto_market_features",
    "asset_category_features",
    "market_dominance_features",
]