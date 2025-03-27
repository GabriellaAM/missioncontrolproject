#!/usr/bin/env python3
"""
SOROS System Example Script

This script demonstrates how to use the SOROS system with common use cases.
It includes examples of initializing the system, analyzing assets, creating
portfolios, backtesting, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import the refactored TrendAnalyzer
from soros_system.main import TrendAnalyzer

def main():
    """Main function demonstrating SOROS system usage."""
    print("SOROS System Example Script")
    print("===========================")
    
    # Step 1: Configuration
    print("\n1. Configuration")
    print("----------------")
    
    # Define asset IDs to analyze
    asset_ids = [
        'bitcoin', 'ethereum', 'binancecoin', 'solana', 
        'cardano', 'avalanche-2', 'polkadot', 'chainlink'
    ]
    
    # Set paths to your data files
    # Adjust these paths to match your data locations
    data_path = 'data/'
    btc_data_path = 'data/bitcoin_candles.csv'
    ssr_data_path = None  # Set to your SSR data path if you have one
    
    print(f"Asset IDs: {asset_ids}")
    print(f"Data path: {data_path}")
    print(f"BTC data path: {btc_data_path}")
    
    # Step 2: Initialize the SOROS System
    print("\n2. Initializing SOROS System")
    print("---------------------------")
    
    analyzer = TrendAnalyzer(
        asset_ids=asset_ids,
        data_path=data_path,
        btc_data_path=btc_data_path,
        use_btc_adjusted=True,
        verbose=True,
        lookback_days=90,
        ssr_data_path=ssr_data_path
    )
    
    print("TrendAnalyzer initialized successfully")
    
    # Step 3: Analyze Assets
    print("\n3. Analyzing Assets")
    print("------------------")
    
    try:
        results = analyzer.analyze_multiple_assets()
        print(f"Successfully analyzed {len(results)} assets")
        
        # Get the latest trend classifications
        latest_trends = analyzer.get_latest_trends()
        if not latest_trends.empty:
            print("\nLatest Trend Classifications:")
            print(latest_trends[['asset_id', 'Overall_Trend_USD', 'Overall_Trend_BTC']].head())
        
        # Get detailed asset information
        eth_info = analyzer.get_asset_information('ethereum')
        if eth_info:
            print("\nEthereum Trend Metrics (USD):")
            print(eth_info['trend_metrics_usd'][['Description', 'Avg_Return', 'Sharpe', 'Sortino']].head())
        
        # Compare trend performance
        sol_trends = analyzer.compare_trend_performance('solana', price_type='USD')
        if not sol_trends.empty:
            print("\nSolana Trend Performance Comparison:")
            print(sol_trends[['Description', 'Sharpe', 'Sortino', 'Avg_Return']].head())
    except Exception as e:
        print(f"Error analyzing assets: {e}")
        # Continue with the rest of the script even if analysis fails
    
    # Step 4: Create Portfolios
    print("\n4. Creating Portfolios")
    print("--------------------")
    
    # Create a BTC trend following portfolio
    try:
        analyzer.create_portfolio(
            portfolio_name='btc_trend_following',
            btc_trend_gating=0,  # Only active when BTC trend is neutral or bullish
            usd_conditions={'Short Term': [1, 2], 'Medium Term': [0, 1, 2]},  # Specify by trend term
            btc_conditions=[0, 1, 2],  # Include assets with neutral or bullish BTC trends
            max_assets=5,  # Maximum number of assets to include
            use_btc_rsi_signal=False
        )
        print("Created btc_trend_following portfolio")
        
        # Create portfolio with text trend descriptions
        analyzer.create_portfolio(
            portfolio_name='strong_bull_portfolio',
            btc_trend_gating=0,
            usd_conditions=['Strong Bull'],  # Use text description
            btc_conditions=['Strong Bull', 'Weak Bull'],  # Use text descriptions
            rsi_conditions_usd=True,
            rsi_conditions_btc=True,
            use_btc_rsi_signal=True,
            btc_only=False,
            use_volatility_filter=True,
            volatility_weight=1.0
        )
        print("Created strong_bull_portfolio")
        
        # List all portfolios
        portfolios = analyzer.list_portfolios()
        if not portfolios.empty:
            print("\nAvailable Portfolios:")
            print(portfolios)
    except Exception as e:
        print(f"Error creating portfolios: {e}")
    
    # Step 5: Backtest Portfolios
    print("\n5. Backtesting Portfolios")
    print("-----------------------")
    
    # Define period for backtesting
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    initial_capital = 10000
    
    print(f"Backtesting period: {start_date} to {end_date}")
    print(f"Initial capital: ${initial_capital}")
    
    try:
        # Backtest the BTC trend following portfolio
        btc_trend_results = analyzer.backtest_portfolio(
            portfolio_name='btc_trend_following',
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            show_plot=False  # Set to True to show the plot
        )
        print("Completed backtesting of btc_trend_following portfolio")
        
        # Get portfolio metrics
        if btc_trend_results and 'metrics' in btc_trend_results:
            print("\nBTC Trend Following Portfolio Metrics:")
            metrics = btc_trend_results['metrics']
            print(f"Total Return: {metrics.get('total_return')}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio')}")
            print(f"Max Drawdown: {metrics.get('max_drawdown')}")
            print(f"Win Rate: {metrics.get('win_rate')}")
        
        # Get portfolio signals
        signals_df = analyzer.get_portfolio_signals_df(btc_trend_results)
        if not signals_df.empty:
            print("\nPortfolio Signals (First 3 Days):")
            print(signals_df.head(3))
        
        # Get daily portfolio results
        daily_results = analyzer.get_portfolio_daily_results(btc_trend_results)
        if not daily_results.empty:
            print("\nDaily Portfolio Results (First 3 Days):")
            print(daily_results.head(3))
        
        # Get asset performance metrics
        asset_metrics = analyzer.get_portfolio_asset_metrics(btc_trend_results)
        if asset_metrics:
            print("\nAsset Performance Metrics:")
            for asset, metrics in asset_metrics.items():
                print(f"\n{asset}:")
                print(f"  Total Return: {metrics.get('total_return')}")
                print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio')}")
                print(f"  Win Rate: {metrics.get('win_rate')}")
                print(f"  Avg Holding Days: {metrics.get('avg_holding_days')}")
        
        # Get recommended assets table
        rec_assets = analyzer.get_recommended_assets_table(btc_trend_results)
        if not rec_assets.empty:
            print("\nRecommended Assets (First 3 Days):")
            print(rec_assets.head(3))
    except Exception as e:
        print(f"Error backtesting portfolios: {e}")
    
    # Step 6: Visualize Results
    print("\n6. Visualizing Results")
    print("--------------------")
    
    try:
        # Plot portfolio performance
        if btc_trend_results:
            print("Plotting portfolio performance...")
            fig = analyzer.plot_portfolio_performance(btc_trend_results, show_plot=False)
        
        # Plot asset performance
        if btc_trend_results:
            print("Plotting asset performance...")
            fig = analyzer.plot_asset_performance(
                btc_trend_results, 
                asset_filter=['ethereum', 'binancecoin'],
                show_plot=False
            )
        
        # Analyze trend distributions for BTC
        print("Plotting BTC trend distribution...")
        analyzer.plot_trend_distribution('bitcoin', trend_type='USD', show_plot=False)
        
        # Plot transition matrix
        print("Plotting BTC transition matrix...")
        analyzer.plot_transition_matrix('bitcoin', trend_type='USD', show_plot=False)
    except Exception as e:
        print(f"Error visualizing results: {e}")
    
    print("\nCompleted SOROS System Example")

if __name__ == "__main__":
    main() 