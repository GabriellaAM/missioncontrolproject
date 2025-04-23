#!/usr/bin/env python3
"""
Simple example demonstrating the asset-specific backtesting through PortfolioAnalyzer.

This script provides a minimal example of how to:
1. Initialize the PortfolioAnalyzer
2. Load data for a single asset
3. Run backtest with default or custom signals
4. Access and display the backtest results
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary modules
from soros_system.core.portfolio_analyzer import PortfolioAnalyzer

def main():
    # Initialize PortfolioAnalyzer
    logger.info("Initializing PortfolioAnalyzer...")
    analyzer = PortfolioAnalyzer(verbose=True)
    
    # Load data for a single asset (Bitcoin)
    asset_id = "bitcoin"
    logger.info(f"Loading data for {asset_id}...")
    analyzer.load_data(asset_id)
    
    # Define backtest parameters
    start_date = "2021-01-01"
    end_date = "2022-12-31"
    initial_capital = 10000.0
    
    # Run a backtest with default signals
    logger.info(f"Running backtest with default signals...")
    default_result = analyzer.backtest_asset(
        asset_id=asset_id,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    # Display default backtest results
    if default_result:
        metrics = default_result['metrics_comparison']
        print("\nBacktest Results with Default Signals:")
        print(f"Total Return: Strategy={metrics.loc['Total Return', 'Strategy']:.2%}, Buy & Hold={metrics.loc['Total Return', 'Buy & Hold']:.2%}")
        print(f"Sharpe Ratio: Strategy={metrics.loc['Sharpe Ratio', 'Strategy']:.2f}, Buy & Hold={metrics.loc['Sharpe Ratio', 'Buy & Hold']:.2f}")
        print(f"Max Drawdown: Strategy={metrics.loc['Max Drawdown', 'Strategy']:.2%}, Buy & Hold={metrics.loc['Max Drawdown', 'Buy & Hold']:.2%}")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        results_df = default_result['results_df']
        plt.plot(results_df['portfolio_value'], label='Strategy')
        plt.plot(results_df['buy_hold_value'], label='Buy & Hold')
        plt.title(f'Backtest Results for {asset_id}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('default_backtest_results.png')
        print("\nSaved equity curve to 'default_backtest_results.png'")
    
    # Run a backtest with custom signals
    logger.info(f"Running backtest with custom signals...")
    custom_signals = [
        "BullLowVarianceSignalUSD",  # Use only low volatility bull market signals
        "RSI_Oversold_USD"           # And buy on RSI oversold signals
    ]
    
    custom_result = analyzer.backtest_asset(
        asset_id=asset_id,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        usd_signals=custom_signals
    )
    
    # Display custom backtest results
    if custom_result:
        metrics = custom_result['metrics_comparison']
        print("\nBacktest Results with Custom Signals:")
        print(f"Total Return: Strategy={metrics.loc['Total Return', 'Strategy']:.2%}, Buy & Hold={metrics.loc['Total Return', 'Buy & Hold']:.2%}")
        print(f"Sharpe Ratio: Strategy={metrics.loc['Sharpe Ratio', 'Strategy']:.2f}, Buy & Hold={metrics.loc['Sharpe Ratio', 'Buy & Hold']:.2f}")
        print(f"Max Drawdown: Strategy={metrics.loc['Max Drawdown', 'Strategy']:.2%}, Buy & Hold={metrics.loc['Max Drawdown', 'Buy & Hold']:.2%}")
        
        # Plot the metrics comparison
        plt.figure(figsize=(10, 6))
        metrics.iloc[:6].plot(kind='bar', figsize=(12, 6))
        plt.title(f'Strategy vs Buy & Hold Metrics for {asset_id}')
        plt.savefig('custom_backtest_metrics.png')
        print("\nSaved metrics comparison to 'custom_backtest_metrics.png'")
    
    # Access backtest through the asset
    asset = analyzer.get_asset(asset_id)
    if asset and hasattr(asset, 'get_backtest_names'):
        backtest_names = asset.get_backtest_names()
        logger.info(f"Available backtests for {asset_id}: {backtest_names}")
    
    logger.info("Example completed successfully")
    
if __name__ == "__main__":
    main() 