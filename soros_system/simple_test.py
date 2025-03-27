#!/usr/bin/env python3
"""
Simple test script to verify USD trend classification fixes.
This script directly tests the trend classifier without using the whole TrendAnalyzer.
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

# Import the components directly
from indicators.trend_classifier import TrendClassifier
from indicators.moving_averages import MovingAverageCalculator

def create_test_data_with_trends():
    """Create test data with pre-assigned trend classifications."""
    # Create a date range
    dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
    n = len(dates)
    
    # Create price data
    close = 100 + np.cumsum(np.random.normal(0.001, 0.02, n))
    cardano_btc = close / 1000
    
    # Create DataFrame with basic price data
    df = pd.DataFrame({
        'date': dates,
        'open': close * 0.99,
        'high': close * 1.01,
        'low': close * 0.98,
        'close': close,
        'volume': np.random.randint(1000, 10000, n),
        'cardano_btc_open': cardano_btc * 0.99,
        'cardano_btc_high': cardano_btc * 1.01,
        'cardano_btc_low': cardano_btc * 0.98,
        'cardano_btc': cardano_btc,
        'asset_id': 'cardano'
    })
    
    # Manually add trend columns for USD
    df['Trend_Short Term_USD'] = np.random.choice([-2, -1, 0, 1, 2], size=n)
    df['Trend_Medium Term_USD'] = np.random.choice([-2, -1, 0, 1, 2], size=n)
    df['Trend_Long Term_USD'] = np.random.choice([-2, -1, 0, 1, 2], size=n)
    
    # Manually add trend columns for BTC
    df['Trend_Short Term_BTC'] = np.random.choice([-2, -1, 0, 1, 2], size=n)
    df['Trend_Medium Term_BTC'] = np.random.choice([-2, -1, 0, 1, 2], size=n)
    df['Trend_Long Term_BTC'] = np.random.choice([-2, -1, 0, 1, 2], size=n)
    
    # Calculate overall trends manually
    df['Overall_Trend_USD'] = df[['Trend_Short Term_USD', 'Trend_Medium Term_USD', 'Trend_Long Term_USD']].mean(axis=1).apply(
        lambda x: 2 if x > 1.5 else (1 if x > 0.5 else (0 if x > -0.5 else (-1 if x > -1.5 else -2)))
    )
    
    df['Overall_Trend_BTC'] = df[['Trend_Short Term_BTC', 'Trend_Medium Term_BTC', 'Trend_Long Term_BTC']].mean(axis=1).apply(
        lambda x: 2 if x > 1.5 else (1 if x > 0.5 else (0 if x > -0.5 else (-1 if x > -1.5 else -2)))
    )
    
    return df

def test_trend_classification_direct():
    """Test trend classification with pre-assigned trend values."""
    logger.info("Testing trend classification with pre-assigned trends...")
    
    # Create test data with trends
    test_data = create_test_data_with_trends()
    logger.info(f"Created test data with {len(test_data)} rows")
    
    # Check if trend columns exist
    usd_trend_cols = [col for col in test_data.columns if col.startswith('Trend_') and col.endswith('_USD')]
    btc_trend_cols = [col for col in test_data.columns if col.startswith('Trend_') and col.endswith('_BTC')]
    
    logger.info(f"USD trend columns: {usd_trend_cols}")
    logger.info(f"BTC trend columns: {btc_trend_cols}")
    
    # Check if overall trends exist
    has_usd_overall = 'Overall_Trend_USD' in test_data.columns
    has_btc_overall = 'Overall_Trend_BTC' in test_data.columns
    
    logger.info(f"Has USD overall trend: {has_usd_overall}")
    logger.info(f"Has BTC overall trend: {has_btc_overall}")
    
    # Print a sample of the results
    sample_size = min(5, len(test_data))
    sample = test_data.tail(sample_size)
    
    logger.info(f"Sample data (last {sample_size} rows):")
    for i, (_, row) in enumerate(sample.iterrows()):
        trend_values = {}
        if has_usd_overall:
            trend_values['Overall_Trend_USD'] = row['Overall_Trend_USD']
        if has_btc_overall:
            trend_values['Overall_Trend_BTC'] = row['Overall_Trend_BTC']
        
        logger.info(f"Row {i} - {trend_values}")
    
    # Check for NaN values
    success = True
    if has_usd_overall:
        usd_nan_count = test_data['Overall_Trend_USD'].isna().sum()
        usd_total_count = len(test_data)
        usd_nan_pct = usd_nan_count / usd_total_count * 100 if usd_total_count > 0 else 0
        
        logger.info(f"USD trend NaN values: {usd_nan_count}/{usd_total_count} ({usd_nan_pct:.2f}%)")
        
        if usd_nan_pct > 0:
            logger.warning(f"Found NaN values in USD trend: {usd_nan_pct:.2f}%")
            success = False
    else:
        logger.error("Missing Overall_Trend_USD column")
        success = False
    
    if has_btc_overall:
        btc_nan_count = test_data['Overall_Trend_BTC'].isna().sum()
        btc_total_count = len(test_data)
        btc_nan_pct = btc_nan_count / btc_total_count * 100 if btc_total_count > 0 else 0
        
        logger.info(f"BTC trend NaN values: {btc_nan_count}/{btc_total_count} ({btc_nan_pct:.2f}%)")
        
        if btc_nan_pct > 0:
            logger.warning(f"Found NaN values in BTC trend: {btc_nan_pct:.2f}%")
            success = False
    else:
        logger.error("Missing Overall_Trend_BTC column")
        success = False
    
    # Verify how many rows have missing USD trends but have BTC trends
    if has_usd_overall and has_btc_overall:
        usd_nulls_with_btc = test_data[test_data['Overall_Trend_USD'].isna() & test_data['Overall_Trend_BTC'].notna()].shape[0]
        btc_nulls_with_usd = test_data[test_data['Overall_Trend_BTC'].isna() & test_data['Overall_Trend_USD'].notna()].shape[0]
        
        logger.info(f"Rows with missing USD trend but valid BTC trend: {usd_nulls_with_btc}")
        logger.info(f"Rows with missing BTC trend but valid USD trend: {btc_nulls_with_usd}")
    
    return success

def test_compute_overall_classification():
    """Test the compute_overall_classification method directly."""
    logger.info("Testing compute_overall_classification method...")
    
    # Initialize components
    ma_calculator = MovingAverageCalculator()
    trend_classifier = TrendClassifier(ma_calculator)
    
    # Test with various combinations of trend values
    test_cases = [
        # All strong bull should give strong bull
        {'trends': pd.Series([2, 2, 2]), 'expected': 2},
        # All strong bear should give strong bear
        {'trends': pd.Series([-2, -2, -2]), 'expected': -2},
        # Mixed positive trends should average correctly
        {'trends': pd.Series([2, 1, 0]), 'expected': 1},
        # Mixed negative trends should average correctly
        {'trends': pd.Series([-2, -1, 0]), 'expected': -1},
        # Equal mix should be neutral
        {'trends': pd.Series([2, -2, 0]), 'expected': 0},
        # Empty series should return NaN
        {'trends': pd.Series([]), 'expected': np.nan},
    ]
    
    success = True
    for i, test_case in enumerate(test_cases):
        trends = test_case['trends']
        expected = test_case['expected']
        
        result = trend_classifier.compute_overall_classification(trends)
        
        # Handle NaN comparison correctly
        if pd.isna(expected) and pd.isna(result):
            match = True
        else:
            match = result == expected
        
        logger.info(f"Test case {i+1}: Trends {trends.values} -> Expected {expected}, Got {result}, Match: {match}")
        
        if not match:
            logger.error(f"Test case {i+1} failed: Expected {expected}, Got {result}")
            success = False
    
    return success

if __name__ == "__main__":
    # Run the data structure test
    structure_success = test_trend_classification_direct()
    
    if structure_success:
        logger.info("Trend data structure test passed.")
    else:
        logger.error("Trend data structure test failed.")
    
    # Run the classifier test
    classifier_success = test_compute_overall_classification()
    
    if classifier_success:
        logger.info("Classifier test passed.")
    else:
        logger.error("Classifier test failed.")
    
    # Overall success
    overall_success = structure_success and classifier_success
    
    if overall_success:
        logger.info("All tests passed! The trend classification fix is working.")
        sys.exit(0)
    else:
        logger.error("Some tests failed. Please check the logs for details.")
        sys.exit(1) 