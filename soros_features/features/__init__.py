"""
Feast features package

This package contains all feature definitions including:
- entities: Core entities like asset_id, macro_asset_id
- crypto_market_features: Basic market data features (OHLCV, market cap, etc.)
- macro_yahoo_features: Yahoo Finance macro data (OHLCV format)
- macro_fred_features: FRED economic data (single value format)
- macro_calculated_features: Calculated/derived macro features (custom formats)
"""

from .entities import asset_id, macro_asset_id
from .crypto_market_features import crypto_market_features, crypto_market_source
from .macro_yahoo_features import macro_yahoo_features, macro_yahoo_source
from .macro_fred_features import macro_fred_features, macro_fred_source
from .macro_calculated_features import yield_curve_regime_features, rty_ym_ratio_features, macro_calculated_source

__all__ = [
    # Entities
    "asset_id",
    "macro_asset_id",
    # Crypto features
    "crypto_market_features", 
    "crypto_market_source",
    # Macro features
    "macro_yahoo_features",
    "macro_yahoo_source", 
    "macro_fred_features",
    "macro_fred_source",
    "yield_curve_regime_features",
    "rty_ym_ratio_features",
    "macro_calculated_source",
]