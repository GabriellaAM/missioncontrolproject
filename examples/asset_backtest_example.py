#!/usr/bin/env python3
"""
Example script demonstrating how to backtest individual assets with the new backtesting functionality.

This script shows how to:
1. Initialize the PortfolioAnalyzer
2. Load data for assets of interest
3. Register signals (USD and BTC quoted)
4. Run backtest for individual assets 
5. Access backtest results and metrics
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary modules
from soros_system.core.portfolio_analyzer import PortfolioAnalyzer
from soros_system.portfolio.backtest import PortfolioBacktester

def main():
    # Step 1: Initialize PortfolioAnalyzer
    logger.info("Initializing PortfolioAnalyzer...")
    analyzer = PortfolioAnalyzer(verbose=True)
    
    # Step 2: Load asset data
    # Define assets to analyze (including bitcoin and some altcoins)
    asset_ids = ["bitcoin", "ethereum", "solana", "cardano"]
    
    logger.info(f"Loading data for assets: {asset_ids}")
    for asset_id in asset_ids:
        analyzer.load_data(asset_id)
    
    # Step 3: Register signals
    # Define signals for testing
    # USD signals are valid for all assets
    usd_signals = [
        "BullHighVarianceSignalUSD",
        "BullLowVarianceSignalUSD",
        "RSI_Oversold_USD"
    ]
    
    # BTC signals only valid for altcoins
    btc_signals = [
        "BullHighVarianceSignalBTC",
        "BullLowVarianceSignalBTC", 
        "RSI_Oversold_BTC"
    ]
    
    # Register signals for all assets
    logger.info("Registering signals for all assets...")
    for asset_id in asset_ids:
        # Register USD signals for all assets
        analyzer.register_signals(
            asset_id=asset_id,
            signal_names=usd_signals,
            calculate_values=True
        )
        
        # Register BTC signals only for altcoins
        if asset_id != "bitcoin":
            analyzer.register_signals(
                asset_id=asset_id,
                signal_names=btc_signals,
                calculate_values=True
            )
    
    # Step 4: Create backtester
    logger.info("Creating backtester...")
    backtester = PortfolioBacktester(None, analyzer)
    
    # Step 5: Run backtest for each asset
    logger.info("Running backtests for each asset...")
    
    # Define backtest parameters
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    initial_capital = 10000.0
    
    # Store backtest results
    all_results = {}
    
    # Test different signal combinations
    for asset_id in asset_ids:
        logger.info(f"Backtesting {asset_id}...")
        
        # For bitcoin, test only with USD signals
        if asset_id == "bitcoin":
            try:
                # Test 1: All USD signals
                btc_result = backtester.backtest_asset(
                    asset_id=asset_id,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    trade_cost=0.001,  # Lower cost for bitcoin
                    usd_signals=usd_signals
                )
                
                all_results[f"{asset_id}_all_usd"] = btc_result
                
                logger.info(f"Bitcoin backtest complete with USD signals")
                
                # Print key metrics
                metrics = btc_result['metrics_comparison']
                logger.info(f"Bitcoin backtest metrics:")
                logger.info(f"  Total Return: Strategy={metrics.loc['Total Return', 'Strategy']:.2%}, Buy & Hold={metrics.loc['Total Return', 'Buy & Hold']:.2%}")
                logger.info(f"  Sharpe Ratio: Strategy={metrics.loc['Sharpe Ratio', 'Strategy']:.2f}, Buy & Hold={metrics.loc['Sharpe Ratio', 'Buy & Hold']:.2f}")
                logger.info(f"  Win Rate: Strategy={metrics.loc['Win Rate', 'Strategy']:.2%}")
                
            except Exception as e:
                logger.error(f"Error backtesting {asset_id}: {str(e)}")
        
        # For altcoins, test with USD signals, BTC signals, and both
        else:
            try:
                # Test 1: Only USD signals
                usd_result = backtester.backtest_asset(
                    asset_id=asset_id,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    trade_cost=0.002,  # Higher cost for altcoins
                    usd_signals=usd_signals
                )
                
                all_results[f"{asset_id}_usd_only"] = usd_result
                
                # Test 2: Only BTC signals
                btc_result = backtester.backtest_asset(
                    asset_id=asset_id,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    trade_cost=0.002,
                    btc_signals=btc_signals
                )
                
                all_results[f"{asset_id}_btc_only"] = btc_result
                
                # Test 3: Both USD and BTC signals
                combined_result = backtester.backtest_asset(
                    asset_id=asset_id,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    trade_cost=0.002,
                    usd_signals=usd_signals,
                    btc_signals=btc_signals
                )
                
                all_results[f"{asset_id}_combined"] = combined_result
                
                logger.info(f"{asset_id} backtests complete")
                
                # Print key metrics for combined strategy
                metrics = combined_result['metrics_comparison']
                logger.info(f"{asset_id} combined backtest metrics:")
                logger.info(f"  Total Return: Strategy={metrics.loc['Total Return', 'Strategy']:.2%}, Buy & Hold={metrics.loc['Total Return', 'Buy & Hold']:.2%}")
                logger.info(f"  Sharpe Ratio: Strategy={metrics.loc['Sharpe Ratio', 'Strategy']:.2f}, Buy & Hold={metrics.loc['Sharpe Ratio', 'Buy & Hold']:.2f}")
                logger.info(f"  Win Rate: Strategy={metrics.loc['Win Rate', 'Strategy']:.2%}")
                
            except Exception as e:
                logger.error(f"Error backtesting {asset_id}: {str(e)}")
    
    # Step 6: Compare results across assets and strategies
    logger.info("Comparing results across assets and strategies...")
    
    # Create summary dataframe
    summary_data = []
    
    for key, result in all_results.items():
        if result is None:
            continue
            
        strategy_metrics = result['strategy_metrics']
        buy_hold_metrics = result['buy_hold_metrics']
        
        summary_data.append({
            'Asset Strategy': key,
            'Asset': result['asset_id'],
            'Strategy Type': key.split('_')[-1],
            'Total Return': strategy_metrics['total_return'],
            'Buy & Hold Return': buy_hold_metrics['total_return'],
            'Outperformance': strategy_metrics['total_return'] - buy_hold_metrics['total_return'],
            'Sharpe Ratio': strategy_metrics['sharpe_ratio'],
            'Max Drawdown': strategy_metrics['max_drawdown'],
            'Win Rate': strategy_metrics['win_rate'],
            'Trade Count': len(result['trades']) if 'trades' in result else 0
        })
    
    # Create summary dataframe
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Display summary
        print("\nBacktest Results Summary:")
        print(summary_df[['Asset', 'Strategy Type', 'Total Return', 'Buy & Hold Return', 'Outperformance', 'Sharpe Ratio', 'Win Rate', 'Trade Count']])
        
        # Optional: Plot comparison
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Asset', y='Total Return', hue='Strategy Type', data=summary_df)
            plt.title('Strategy Returns by Asset and Signal Type')
            plt.ylabel('Total Return')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('asset_backtest_comparison.png')
            print("\nSaved comparison plot to 'asset_backtest_comparison.png'")
        except Exception as e:
            logger.error(f"Error creating comparison plot: {str(e)}")
    
    # Step 7: Demonstrate how to access backtest results through the asset
    logger.info("Accessing backtest results through the asset...")
    
    # Get an asset with backtests
    test_asset_id = asset_ids[1]  # use ethereum as test
    test_asset = analyzer.get_asset(test_asset_id)
    
    if test_asset:
        # Get list of backtest names
        if hasattr(test_asset, 'get_backtest_names'):
            backtest_names = test_asset.get_backtest_names()
            logger.info(f"Available backtests for {test_asset_id}: {backtest_names}")
            
            # Access one backtest
            if backtest_names:
                test_backtest = test_asset.get_backtest_result(backtest_names[0])
                if test_backtest:
                    logger.info(f"Successfully retrieved backtest {backtest_names[0]} for {test_asset_id}")
                    if 'metrics' in test_backtest:
                        print(f"\nMetrics for {test_asset_id} ({backtest_names[0]}):")
                        print(test_backtest['metrics'])
    
    logger.info("Example completed successfully")
    
if __name__ == "__main__":
    main() 