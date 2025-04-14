"""
Simple Notebook Example for Binary Signals

This is a simplified example using binary signals that can be directly copy-pasted into a Jupyter notebook.
"""

# Import necessary modules
from soros_system.core.portfolio_analyzer import PortfolioAnalyzer
from scripts.assetsRoster import carteira_AC  # Your asset list
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Define assets to use (or use your existing assets list)
assets = carteira_AC[:3]  # Use first 3 assets from your portfolio

# Initialize the analyzer with your data paths
analyzer = PortfolioAnalyzer(
    data_dir='/Users/valter.rebelo/MissionControl/data',
    data_path='/Users/valter.rebelo/MissionControl/data/micro/candleData/',
    btc_data_path='/Users/valter.rebelo/MissionControl/data/micro/candleData/bitcoin_candles.csv',
    ssr_data_path='/Users/valter.rebelo/MissionControl/data/onchainData/BTC_SSR.csv',
    market_data_path='/Users/valter.rebelo/MissionControl/data/micro/assetData/',
    asset_ids=assets
)

# Configure signal parameters with weights
print("Configuring signal parameters...")
analyzer.set_signal_parameters('RSI_Oversold', weight=1.5, decay_period=0)  # decay not used in binary approach
analyzer.set_signal_parameters('RSI_Overbought', weight=-1.0, decay_period=0)
analyzer.set_signal_parameters('Trend_Following', weight=2.0, decay_period=0)
analyzer.set_signal_parameters('Volatility_Breakout', weight=1.0, decay_period=0)

# Load data
print("Loading data...")
asset_data = analyzer.load_data()
print(f"Loaded data for {len(asset_data)} assets")

# Register signals
print("Registering signals...")
signal_names = ['RSI_Oversold', 'RSI_Overbought', 'Trend_Following', 'Volatility_Breakout']

for asset_id in analyzer.get_asset_ids():
    signals = analyzer.register_signals(
        asset_id=asset_id,
        signal_names=signal_names,
        calculate_values=True
    )
    print(f"Registered {len(signals)} signals for {asset_id}")

# Define date range (last 6 months)
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

# Calculate binary signal values
print("Calculating signal values...")
signal_values = {}

for asset_id in analyzer.get_asset_ids():
    # Get asset data
    asset = analyzer.assets.get(asset_id)
    if not asset:
        print(f"No data for {asset_id}")
        continue
        
    # Calculate signal values
    values_df = analyzer.calculate_signal_values(
        asset_id=asset_id, 
        start_date=start_date,
        end_date=end_date,
        force_recalculate=True
    )
    
    signal_values[asset_id] = values_df
    print(f"Calculated signal values for {asset_id}: {len(values_df)} data points")

# Generate trading decisions based on combined signals
print("Generating trading decisions...")
decisions = {}

for asset_id, values_df in signal_values.items():
    # Simple approach: weight the signals and apply threshold
    weighted_sum = pd.Series(0, index=values_df.index)
    
    for signal_name in signal_names:
        if signal_name in values_df.columns:
            weight = analyzer.signal_weights.get(signal_name, 1.0)
            weighted_sum += values_df[signal_name] * weight
    
    # Normalize by total weight
    total_weight = sum(abs(analyzer.signal_weights.get(signal_name, 1.0)) 
                     for signal_name in signal_names if signal_name in values_df.columns)
    
    if total_weight > 0:
        weighted_sum = weighted_sum / total_weight
    
    # Apply threshold (0.0 by default)
    threshold = 0.0
    decisions[asset_id] = (weighted_sum > threshold).astype(int)
    
    # Calculate number of trade signals
    transitions = decisions[asset_id].diff().abs()
    num_trades = transitions.sum()
    
    print(f"Generated {num_trades} trade signals for {asset_id}")

# Prepare price data for backtesting
price_data = {}
for asset_id in analyzer.get_asset_ids():
    asset = analyzer.assets.get(asset_id)
    if not asset:
        continue
    price_data[asset_id] = asset.get_price_data()

# Initialize and run backtester
from soros_system.portfolio.event_backtest import EventDrivenBacktester

backtester = EventDrivenBacktester(
    initial_capital=10000.0,
    trade_cost=0.001,
    slippage_pct=0.0
)

# Run backtest
print("Running backtest...")
backtest_results = backtester.run_backtest(
    price_data=price_data,
    decisions=decisions,
    start_date=start_date,
    end_date=end_date
)

# Display backtest results
if 'portfolio_value' in backtest_results:
    print("\nBacktest Results:")
    
    # Calculate metrics
    portfolio_value = backtest_results['portfolio_value']
    initial_value = portfolio_value.iloc[0]
    final_value = portfolio_value.iloc[-1]
    
    total_return = (final_value / initial_value) - 1
    days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    years = days / 365.0
    
    ann_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
    
    max_drawdown = backtest_results.get('max_drawdown', 0)
    sharpe_ratio = backtest_results.get('sharpe_ratio', 0)
    
    print(f"Initial Capital: ${initial_value:.2f}")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {ann_return:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Plot portfolio value
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value.index, portfolio_value.values)
    plt.title('Portfolio Performance with Binary Signals')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True)
    plt.show()
    
    # Optional: Plot trades for a single asset
    if len(decisions) > 0:
        asset_id = list(decisions.keys())[0]
        asset_decisions = decisions[asset_id]
        asset_prices = price_data[asset_id]['close']
        
        plt.figure(figsize=(12, 6))
        plt.plot(asset_prices.index, asset_prices.values, label='Price')
        
        # Find buy signals (0->1 transitions)
        buy_signals = asset_decisions.diff() == 1
        buy_dates = asset_decisions.index[buy_signals]
        buy_prices = asset_prices.loc[buy_dates]
        
        # Find sell signals (1->0 transitions)
        sell_signals = asset_decisions.diff() == -1
        sell_dates = asset_decisions.index[sell_signals]
        sell_prices = asset_prices.loc[sell_dates]
        
        # Plot buy and sell signals
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')
        plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')
        
        plt.title(f'Price and Trade Signals for {asset_id}')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.show()
else:
    print("Backtest failed to produce results.") 