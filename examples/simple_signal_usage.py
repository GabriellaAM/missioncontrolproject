"""
Simple Signal Usage Example

This example demonstrates how to use the simplified signal-based approach
without statistical testing components.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath('..'))

# Import the PortfolioAnalyzer
from soros_system.core.portfolio_analyzer import PortfolioAnalyzer

def main():
    """Run the example."""
    print("Simple Signal Usage Example")
    print("--------------------------")
    
    # Define assets to use
    assets = ['bitcoin', 'ethereum']
    
    # Initialize the analyzer
    analyzer = PortfolioAnalyzer(
        data_dir='data',
        data_path='data/micro/candleData/',
        btc_data_path='data/micro/candleData/bitcoin_candles.csv',
        ssr_data_path='data/onchainData/BTC_SSR.csv',
        market_data_path='data/micro/assetData/',
        asset_ids=assets,
        verbose=True
    )
    
    # Configure signal parameters
    print("\nConfiguring signal parameters...")
    analyzer.set_signal_parameters('RSI_Oversold', weight=1.5, decay_period=14)
    analyzer.set_signal_parameters('RSI_Overbought', weight=-1.0, decay_period=7)
    analyzer.set_signal_parameters('Trend_Following', weight=2.0, decay_period=21)
    analyzer.set_signal_parameters('Volatility_Breakout', weight=1.0, decay_period=10)
    
    # Load data
    print("\nLoading data...")
    asset_data = analyzer.load_data()
    print(f"Loaded data for {len(asset_data)} assets")
    
    # Register signals
    print("\nRegistering signals...")
    signal_names = ['RSI_Oversold', 'RSI_Overbought', 'Trend_Following', 'Volatility_Breakout']
    
    for asset_id in analyzer.get_asset_ids():
        signals = analyzer.register_signals(
            asset_id=asset_id,
            signal_names=signal_names,
            calculate_values=True
        )
        print(f"Registered {len(signals)} signals for {asset_id}")
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Use 1 year of data
    
    # Track signal events
    print("\nTracking signal events...")
    for asset_id in analyzer.get_asset_ids():
        activations = analyzer.track_signal_events(
            asset_id=asset_id,
            start_date=start_date,
            end_date=end_date,
            signal_names=signal_names
        )
        
        total_activations = sum(len(dates) for dates in activations.values())
        print(f"Tracked signals for {asset_id}, found {total_activations} activations")
        
        # Print activation dates for each signal
        for signal_name, dates in activations.items():
            if dates:
                print(f"  - {signal_name}: {len(dates)} activations")
                for date in dates[:3]:  # Show first 3 activation dates
                    print(f"      {date.strftime('%Y-%m-%d')}")
                if len(dates) > 3:
                    print(f"      ... and {len(dates) - 3} more")
    
    # Get current recommendations
    print("\nGetting current recommendations...")
    recommendations = analyzer.get_current_recommendations()
    
    for asset_id, rec in recommendations.items():
        print(f"\nRecommendations for {asset_id} on {rec['date'].strftime('%Y-%m-%d')}:")
        print(f"Decision: {'EXPOSED' if rec['decision'] == 1 else 'CASH'}")
        print(f"Combined Weight: {rec['combined_weight']:.4f}")
        print(f"Active Signals: {len(rec['active_signals'])}")
        
        # Print active signals
        for signal_name, event in rec['active_signals'].items():
            print(f"  - {signal_name}: weight={event['weight']:.2f}, expires in {event['days_remaining']} days")
    
    # Run a backtest
    print("\nRunning backtest...")
    backtest_results = analyzer.run_backtest(
        backtest_name="simple_backtest",
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000.0,
        trade_cost=0.001
    )
    
    # Display backtest results
    if backtest_results.get('success', False):
        print("\nBacktest Results:")
        print(f"Total Return: {backtest_results.get('total_return', 0):.2%}")
        print(f"Annualized Return: {backtest_results.get('annualized_return', 0):.2%}")
        print(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {backtest_results.get('max_drawdown', 0):.2%}")
        print(f"Win Rate: {backtest_results.get('win_rate', 0):.2%}")
        
        # Plot performance
        if 'portfolio_value' in backtest_results:
            portfolio_values = backtest_results['portfolio_value']
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_values.index, portfolio_values.values)
            plt.title('Portfolio Performance')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.grid(True)
            plt.show()
    else:
        print(f"Backtest failed: {backtest_results.get('error', 'Unknown error')}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 