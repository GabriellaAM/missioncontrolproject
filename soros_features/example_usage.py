"""
Example: How to use the Soros crypto feature store

This script demonstrates practical usage patterns for retrieving
historical crypto features for analysis and modeling.
"""

import pandas as pd
from datetime import datetime, timedelta
from feast import FeatureStore

def get_recent_crypto_features():
    """Get recent features for multiple assets"""
    fs = FeatureStore(repo_path=".")
    
    # Define assets and time range
    assets = ["bitcoin", "ethereum", "solana", "chainlink"]
    recent_date = datetime.now() - timedelta(days=1)
    
    # Create entity dataframe
    entity_df = pd.DataFrame({
        "asset_id": assets,
        "timestamp": [recent_date] * len(assets)
    })
    
    # Get basic market features
    features = [
        "crypto_market_features:open",
        "crypto_market_features:close",
        "crypto_market_features:market_cap",
        "crypto_market_features:total_volume",
        "crypto_market_features:close_btc",
    ]
    
    df = fs.get_historical_features(
        entity_df=entity_df, 
        features=features
    ).to_df()
    
    return df


def get_time_series_data(asset_id: str, start_date: str, end_date: str):
    """Get time series data for a single asset over a date range"""
    fs = FeatureStore(repo_path=".")
    
    # Create date range
    dates = pd.date_range(
        start=start_date, 
        end=end_date, 
        freq='D'
    )
    
    # Create entity dataframe
    entity_df = pd.DataFrame({
        "asset_id": [asset_id] * len(dates),
        "timestamp": dates
    })
    
    # Get OHLCV features
    features = [
        "crypto_market_features:open",
        "crypto_market_features:high", 
        "crypto_market_features:low",
        "crypto_market_features:close",
        "crypto_market_features:total_volume",
    ]
    
    df = fs.get_historical_features(
        entity_df=entity_df,
        features=features
    ).to_df()
    
    return df.sort_values('timestamp')


def main():
    print("🚀 Soros Crypto Feature Store Examples\n")
    
    # Example 1: Recent features for multiple assets
    print("1. Recent features for multiple assets:")
    recent_df = get_recent_crypto_features()
    print(recent_df[['asset_id', 'close', 'market_cap', 'close_btc']])
    
    # Example 2: Time series for Bitcoin in January 2024
    print("\n2. Bitcoin OHLCV for January 2024:")
    btc_series = get_time_series_data("bitcoin", "2024-01-01", "2024-01-31")
    print(f"Retrieved {len(btc_series)} days of data")
    print(btc_series[['timestamp', 'open', 'close']].head())
    
    print("\n✅ Feature store is working correctly!")


if __name__ == "__main__":
    main()