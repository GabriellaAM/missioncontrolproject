#!/usr/bin/env python3
"""
Test script to verify USD trend classification fixes.
This script tests trend classification specifically for altcoins to ensure
that USD trends are properly calculated.
"""

import pandas as pd
import numpy as np
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Import the TrendAnalyzer using absolute import
try:
    from main import TrendAnalyzer
except ImportError:
    # If we're being run from a different directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from soros_system.main import TrendAnalyzer

def test_altcoin_trend_classification():
    """Test trend classification for altcoins."""
    logger.info("Testing altcoin trend classification...")
    
    # Define asset IDs to test
    asset_ids = ['cardano', 'ethereum', 'binancecoin']
    
    # Initialize TrendAnalyzer with test configuration
    data_path = 'data/'
    btc_data_path = 'data/bitcoin_candles.csv'
    
    # Make sure the data path exists
    if not os.path.exists(data_path):
        logger.error(f"Data path does not exist: {data_path}")
        logger.info("This test needs actual crypto data to run.")
        sys.exit(1)
    
    analyzer = TrendAnalyzer(
        asset_ids=asset_ids,
        data_path=data_path,
        btc_data_path=btc_data_path,
        use_btc_adjusted=True,
        verbose=True,
        lookback_days=90
    )
    
    # Run analysis
    results = analyzer.analyze_multiple_assets()
    
    # Check results for each asset
    success = True
    for asset_id in asset_ids:
        logger.info(f"Checking {asset_id} trends...")
        
        if asset_id not in results:
            logger.error(f"No results found for {asset_id}")
            success = False
            continue
        
        df = results[asset_id]
        
        if df.empty:
            logger.error(f"Empty DataFrame returned for {asset_id}")
            success = False
            continue
        
        # Check if trend columns exist
        usd_trend_cols = [col for col in df.columns if col.startswith('Trend_') and col.endswith('_USD')]
        btc_trend_cols = [col for col in df.columns if col.startswith('Trend_') and col.endswith('_BTC')]
        
        logger.info(f"{asset_id} USD trend columns: {usd_trend_cols}")
        logger.info(f"{asset_id} BTC trend columns: {btc_trend_cols}")
        
        # Check if overall trends exist
        has_usd_overall = 'Overall_Trend_USD' in df.columns
        has_btc_overall = 'Overall_Trend_BTC' in df.columns
        
        logger.info(f"{asset_id} has USD overall trend: {has_usd_overall}")
        logger.info(f"{asset_id} has BTC overall trend: {has_btc_overall}")
        
        # Check for NaN values
        if has_usd_overall:
            usd_nan_count = df['Overall_Trend_USD'].isna().sum()
            usd_total_count = len(df)
            usd_nan_pct = usd_nan_count / usd_total_count * 100 if usd_total_count > 0 else 0
            
            logger.info(f"{asset_id} USD trend NaN values: {usd_nan_count}/{usd_total_count} ({usd_nan_pct:.2f}%)")
            
            if usd_nan_pct > 90:
                logger.warning(f"{asset_id} has too many NaN values in USD trend")
                success = False
        else:
            logger.error(f"{asset_id} is missing Overall_Trend_USD column")
            success = False
        
        # Print a sample of the data
        if not df.empty:
            sample_size = min(5, len(df))
            sample = df.tail(sample_size)
            
            logger.info(f"{asset_id} sample data (last {sample_size} rows):")
            
            trend_cols = ['Overall_Trend_USD', 'Overall_Trend_BTC']
            available_cols = [col for col in trend_cols if col in sample.columns]
            
            if available_cols:
                for _, row in sample.iterrows():
                    logger.info(f"Date: {row.get('date', 'N/A')}, " + 
                               ", ".join([f"{col}: {row.get(col, 'N/A')}" for col in available_cols]))
    
    return success

if __name__ == "__main__":
    success = test_altcoin_trend_classification()
    
    if success:
        logger.info("All tests passed! The trend classification fix is working.")
        sys.exit(0)
    else:
        logger.error("Some tests failed. Please check the logs for details.")
        sys.exit(1) 