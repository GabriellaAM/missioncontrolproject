import os
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Add the parent directory to the path to allow importing the main package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from soros_system.core.portfolio_analyzer import PortfolioAnalyzer

def run_example():
    # Initialize the portfolio analyzer
    analyzer = PortfolioAnalyzer(
        data_dir='data',
        verbose=True
    )
    
    # Load data for a selection of assets
    assets = ['bitcoin', 'ethereum', 'solana', 'dogecoin', 'chainlink', 'pendle']
    loaded_assets = analyzer.load_data(assets=assets)
    
    print(f"Loaded {len(loaded_assets)} assets")
    
    # Register signals for all assets
    for asset_id in assets:
        analyzer.register_signals(
            asset_id=asset_id,
            signal_names=['RSI_Bullish_USD', 'RSI_Bullish_BTC'],
            calculate_values=True
        )
    
    # Define signals to use for backtesting
    usd_signals = ['RSI_Bullish_USD']
    btc_signals = ['RSI_Bullish_BTC']
    
    # Run backtests
    results = analyzer.backtest_assets(
        asset_ids=assets,
        start_date="2024-1-1",
        end_date="2025-12-31",
        initial_capital=10000.0,
        btc_cost=0.001,  # Lower cost for Bitcoin
        alt_cost=0.005,  # Higher cost for altcoins
        usd_signals=usd_signals,
        btc_signals=btc_signals,
        use_btc_filter=True
    )
    
    # Display the formatted summary table
    if 'formatted_summary' in results and not results['formatted_summary'].empty:
        print("\nFormatted Summary Table:")
        print(results['formatted_summary'])
    elif 'summary' in results and not results['summary'].empty:
        print("\nRegular Summary Table:")
        print(results['summary'])
    else:
        print("\nNo summary table available. Checking asset results...")
        # Try to manually build a summary from asset_results
        if 'asset_results' in results and results['asset_results']:
            print(f"Found {len(results['asset_results'])} asset results. Constructing summary...")
            summary_data = []
            
            for asset_id, asset_result in results['asset_results'].items():
                if 'results_df' in asset_result and 'metrics_comparison' in asset_result:
                    print(f"Processing {asset_id}...")
                    metrics = asset_result['metrics_comparison']
                    
                    # Calculate peak return
                    peak_return = 0.0
                    if 'portfolio_value' in asset_result['results_df'].columns:
                        initial_capital = results['parameters']['initial_capital']
                        max_val = asset_result['results_df']['portfolio_value'].max()
                        peak_return = (max_val / initial_capital - 1) * 100
                    
                    # Calculate buy & hold peak return
                    bh_peak_return = 0.0
                    if 'buy_hold_value' in asset_result['results_df'].columns:
                        initial_capital = results['parameters']['initial_capital']
                        max_val = asset_result['results_df']['buy_hold_value'].max()
                        bh_peak_return = (max_val / initial_capital - 1) * 100
                    
                    try:
                        summary_data.append({
                            'Asset': asset_id.upper(),
                            'Total Return (%)': metrics.loc['Total Return', 'Strategy'] * 100,
                            'Peak Return (%)': peak_return,
                            'Buy & Hold Return (%)': metrics.loc['Total Return', 'Buy & Hold'] * 100,
                            'Buy & Hold Peak (%)': bh_peak_return,
                            'Sharpe': metrics.loc['Sharpe Ratio', 'Strategy'],
                            'Buy & Hold Sharpe': metrics.loc['Sharpe Ratio', 'Buy & Hold'],
                            'Max Drawdown (%)': metrics.loc['Max Drawdown', 'Strategy'] * 100,
                            'Trade Count': len(asset_result['trades']) if 'trades' in asset_result else 0
                        })
                    except Exception as e:
                        print(f"Error extracting metrics for {asset_id}: {str(e)}")
            
            if summary_data:
                custom_summary = pd.DataFrame(summary_data)
                custom_summary.set_index('Asset', inplace=True)
                print("\nManually constructed summary table:")
                print(custom_summary)
            else:
                print("Could not construct summary table from asset results.")
        else:
            print("No asset results available.")
    
    # Plot the equity curves for comparison
    plt.figure(figsize=(12, 8))
    
    # Plot equity curves for each asset
    for asset_id, result in results['asset_results'].items():
        if 'results_df' in result and 'portfolio_value' in result['results_df'].columns:
            plt.plot(result['results_df'].index, result['results_df']['portfolio_value'], 
                    label=f"{asset_id.capitalize()} Strategy")
    
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_comparison.png')
    plt.close()
    
    print("\nBacktest comparison chart saved as 'backtest_comparison.png'")
    
    return results

if __name__ == "__main__":
    results = run_example() 