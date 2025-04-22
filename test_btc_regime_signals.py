"""
Test script for BTC-quoted regime detection signals.

This script demonstrates how to use the new BTC-quoted regime detection signals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from soros_system.core.portfolio_analyzer import PortfolioAnalyzer
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Define the asset to analyze
asset_id = 'ethereum'  # Using Ethereum as example to test BTC-quoted signals

# Initialize the PortfolioAnalyzer
analyzer = PortfolioAnalyzer(
    data_dir='/Users/valter.rebelo/MissionControl/data',
    data_path='/Users/valter.rebelo/MissionControl/data/micro/candleData/',
    btc_data_path='/Users/valter.rebelo/MissionControl/data/micro/candleData/bitcoin_candles.csv',
    ssr_data_path='/Users/valter.rebelo/MissionControl/data/onchainData/BTC_SSR.csv',
    market_data_path='/Users/valter.rebelo/MissionControl/data/micro/assetData/',
    asset_ids=[asset_id, 'bitcoin']  # Include bitcoin for BTC quotes
)

# Load data for the assets
analyzer.load_data(assets=[asset_id, 'bitcoin'], start_date='2014-01-01', end_date='2025-12-31')

# Register only the new BTC-quoted regime detection signals for testing
signal_names = [
    'BullHighVarianceSignalBTC',
    'BullLowVarianceSignalBTC',
    'BearHighVarianceSignalBTC',
    'BearLowVarianceSignalBTC'
]

print(f"Registering BTC-quoted signals for {asset_id}...")
registered_signals = analyzer.register_signals(
    asset_id=asset_id,
    signal_names=signal_names,
    calculate_values=True
)

# Get the asset data and signal values
asset_data = analyzer.get_asset_data(asset_id)
if asset_data is None:
    print(f"Asset data not found for {asset_id}")
    exit(1)

# Get price data
price_data = asset_data.price_data
if price_data.empty:
    print(f"Price data is empty for {asset_id}")
    exit(1)

# Check BTC-quoted price column format
btc_col = f"{asset_id}_btc"
if btc_col not in price_data.columns and 'close_btc' not in price_data.columns:
    print(f"BTC-quoted price data not found for {asset_id}. Please ensure BTC prices are calculated.")
    print(f"Expected column names: '{btc_col}' or 'close_btc'")
    print(f"Available columns: {price_data.columns.tolist()}")
    exit(1)

# Print how many signal values we have for each signal
for signal_name in signal_names:
    signal_data = asset_data.get_signal(signal_name)
    if signal_data is not None and signal_data.values is not None:
        active_count = signal_data.values.sum()
        total_count = len(signal_data.values)
        print(f"{signal_name}: {active_count} active out of {total_count} data points ({active_count/total_count*100:.2f}%)")
    else:
        print(f"{signal_name}: No signal data")

# Plot the price with the signals (using BTC-quoted prices)
fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

# Set the title for the entire figure
fig.suptitle(f'{asset_id.upper()} BTC Price and Regime Detection Signals', fontsize=16)

# Determine which BTC price column to use
btc_price_col = btc_col if btc_col in price_data.columns else 'close_btc'
print(f"Using {btc_price_col} as BTC price column")

# Plot the BTC price in the first subplot
btc_price_series = price_data[btc_price_col]
axes[0].plot(btc_price_series.index, btc_price_series, color='orange', label='Price (BTC)')
axes[0].set_ylabel('Price (BTC)')
axes[0].set_title('BTC-Quoted Price')
axes[0].legend()
axes[0].grid(True)

# Plot each signal in a separate subplot
for i, signal_name in enumerate(signal_names, 1):
    signal_data = asset_data.get_signal(signal_name)
    if signal_data is not None and signal_data.values is not None:
        signal_values = signal_data.values
        
        # Create a masked price series where it shows the price only when the signal is active
        masked_price = btc_price_series.copy()
        masked_price[signal_values == 0] = np.nan
        
        axes[i].plot(btc_price_series.index, btc_price_series, color='gray', alpha=0.3, label='Price (BTC)')
        axes[i].plot(masked_price.index, masked_price, color='green', label=f'Active {signal_name}')
        axes[i].set_ylabel('Price (BTC)')
        axes[i].set_title(signal_name)
        axes[i].legend()
        axes[i].grid(True)

# Set the x-axis label for the bottom subplot
axes[-1].set_xlabel('Date')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# Save the figure
plt.savefig(f'{asset_id}_btc_regime_signals.png')
print(f"Plot saved as {asset_id}_btc_regime_signals.png")

# Display the plot
plt.show()

# Now test both USD and BTC signals side by side to compare
usd_signal_names = [
    'BullHighVarianceSignalUSD',
    'BullLowVarianceSignalUSD',
    'BearHighVarianceSignalUSD',
    'BearLowVarianceSignalUSD'
]

print(f"Registering USD-quoted signals for {asset_id} for comparison...")
usd_registered_signals = analyzer.register_signals(
    asset_id=asset_id,
    signal_names=usd_signal_names,
    calculate_values=True
)

# Create a comparison dataframe for USD vs BTC regime detection
result_df = pd.DataFrame(index=price_data.index)

# Add USD price and BTC price for reference
result_df['USD_Price'] = price_data['close']
result_df['BTC_Price'] = btc_price_series  # Using the determined BTC price column

# Add signals for both USD and BTC
for signal_name in usd_signal_names:
    signal_data = asset_data.get_signal(signal_name)
    if signal_data is not None and signal_data.values is not None:
        result_df[signal_name] = signal_data.values

for signal_name in signal_names:
    signal_data = asset_data.get_signal(signal_name)
    if signal_data is not None and signal_data.values is not None:
        result_df[signal_name] = signal_data.values

# Calculate agreement percentage between USD and BTC signals
for regime_type in ['BullHigh', 'BullLow', 'BearHigh', 'BearLow']:
    usd_col = f'{regime_type}VarianceSignalUSD'
    btc_col = f'{regime_type}VarianceSignalBTC'
    if usd_col in result_df.columns and btc_col in result_df.columns:
        agreement = (result_df[usd_col] == result_df[btc_col]).mean() * 100
        print(f"Agreement between {usd_col} and {btc_col}: {agreement:.2f}%")

# Save the results to CSV for further analysis
result_df.to_csv(f'{asset_id}_regime_signal_comparison.csv')
print(f"Comparison data saved to {asset_id}_regime_signal_comparison.csv") 