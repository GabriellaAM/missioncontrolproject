import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from soros_system.main import TrendAnalyzer
from soros_system.signals.rsi_signals import RSI_Oversold_USD, RSI_Overbought_USD
from soros_system.indicators.rsi import RSICalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rsi_signals():
    """Test the RSI signal calculation with the simplified implementation."""
    logger.info("Starting RSI signals test")
    
    # Set up data paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'micro', 'candleData')
    btc_data_path = os.path.join(data_path, 'bitcoin_candles.csv')
    market_data_path = os.path.join(base_dir, 'data', 'micro', 'assetData')
    
    # Load BTC data directly to test
    try:
        btc_data = pd.read_csv(os.path.join(data_path, 'bitcoin_candles.csv'))
        btc_data['date'] = pd.to_datetime(btc_data['date'])
        btc_data.set_index('date', inplace=True)
        logger.info(f"Loaded BTC data directly with shape: {btc_data.shape}")
    except Exception as e:
        logger.error(f"Error loading BTC data: {e}")
        return
    
    # Calculate RSI directly using RSICalculator
    rsi_calculator = RSICalculator()
    rsi_values = rsi_calculator.calculate_rsi(btc_data['close'], window=14)
    btc_data['RSI'] = rsi_values
    
    # Print RSI statistics
    print("\n========== RSI Statistics ==========")
    print(f"RSI Mean: {rsi_values.mean():.2f}")
    print(f"RSI Min: {rsi_values.min():.2f}")
    print(f"RSI Max: {rsi_values.max():.2f}")
    print(f"RSI Median: {rsi_values.median():.2f}")
    print(f"% of time RSI < 30: {(rsi_values < 30).mean() * 100:.2f}%")
    print(f"% of time RSI > 70: {(rsi_values > 70).mean() * 100:.2f}%")
    
    # Print recent RSI values
    print("\n========== Last 10 Days RSI Values ==========")
    recent_rsi = rsi_values.iloc[-10:]
    for date, rsi in recent_rsi.items():
        print(f"{date.strftime('%Y-%m-%d')}: {rsi:.2f}")
    
    # Print some extreme RSI values
    print("\n========== 5 Lowest RSI Values ==========")
    lowest_rsi = rsi_values.nsmallest(5)
    for date, rsi in lowest_rsi.items():
        print(f"{date.strftime('%Y-%m-%d')}: {rsi:.2f}")
    
    print("\n========== 5 Highest RSI Values ==========")
    highest_rsi = rsi_values.nlargest(5)
    for date, rsi in highest_rsi.items():
        print(f"{date.strftime('%Y-%m-%d')}: {rsi:.2f}")
    
    # Create RSI signals
    btc_data['RSI_Oversold'] = np.where(btc_data['RSI'] < 30, 1, 0)
    btc_data['RSI_Overbought'] = np.where(btc_data['RSI'] > 70, 1, 0)
    
    # Initialize TrendAnalyzer for testing with signal classes
    analyzer = TrendAnalyzer(
        asset_ids=['bitcoin'],
        data_path=data_path,
        btc_data_path=btc_data_path,
        market_data_path=market_data_path
    )
    
    # Analyze BTC
    btc_analyzed = analyzer.analyze_asset('bitcoin')
    logger.info(f"BTC analysis complete, data shape: {btc_analyzed.shape if btc_analyzed is not None else 'None'}")
    
    # Get the asset data
    btc_processed = analyzer.get_asset_processed_data('bitcoin')
    
    # Initialize signals
    rsi_oversold = RSI_Oversold_USD(params={'asset_id': 'bitcoin', 'rsi_length': 14, 'oversold': 30})
    rsi_overbought = RSI_Overbought_USD(params={'asset_id': 'bitcoin', 'rsi_length': 14, 'overbought': 70})
    
    # Calculate signals
    oversold_signal = rsi_oversold.calculate(btc_processed, asset_id='bitcoin')
    overbought_signal = rsi_overbought.calculate(btc_processed, asset_id='bitcoin')
    
    # Add calculated signals to the DataFrame
    btc_processed_with_signals = btc_processed.copy()
    btc_processed_with_signals['RSI_Direct'] = rsi_values.reindex(btc_processed.index)
    btc_processed_with_signals['Oversold_Signal'] = oversold_signal
    btc_processed_with_signals['Overbought_Signal'] = overbought_signal
    
    # Verify signal values match RSI thresholds
    print("\n========== Signal Verification ==========")
    oversold_matches = (btc_processed_with_signals['RSI_Direct'] < 30) == (btc_processed_with_signals['Oversold_Signal'] == 1)
    overbought_matches = (btc_processed_with_signals['RSI_Direct'] > 70) == (btc_processed_with_signals['Overbought_Signal'] == 1)
    
    print(f"Oversold signals match RSI < 30 threshold: {oversold_matches.mean() * 100:.2f}%")
    print(f"Overbought signals match RSI > 70 threshold: {overbought_matches.mean() * 100:.2f}%")
    
    # Validate results
    logger.info(f"RSI Oversold signal has {oversold_signal.sum()} buy signals")
    logger.info(f"RSI Overbought signal has {overbought_signal.sum()} sell signals")
    
    # Create a CSV with dates and RSI values for further analysis
    output_file = 'btc_rsi_values.csv'
    rsi_df = pd.DataFrame({
        'date': rsi_values.index,
        'close': btc_data.loc[rsi_values.index, 'close'],
        'rsi': rsi_values,
        'oversold': btc_data.loc[rsi_values.index, 'RSI_Oversold'],
        'overbought': btc_data.loc[rsi_values.index, 'RSI_Overbought']
    })
    rsi_df.to_csv(output_file)
    print(f"\nSaved all RSI values to {output_file} for further analysis")
    
    # Optionally, plot the results (last 200 days) to visually verify
    try:
        plt.figure(figsize=(12, 8))
        
        # Last 200 days for better visualization
        last_n_days = 200
        plot_data = btc_processed_with_signals.iloc[-last_n_days:]
        
        # Plot 1: BTC price
        plt.subplot(211)
        plt.title('BTC Price and RSI')
        plt.plot(plot_data.index, plot_data['close'], label='BTC Price')
        
        # Highlight oversold regions
        oversold_regions = plot_data[plot_data['Oversold_Signal'] == 1]
        plt.scatter(oversold_regions.index, oversold_regions['close'], color='green', marker='^', s=100, label='Oversold')
        
        # Highlight overbought regions
        overbought_regions = plot_data[plot_data['Overbought_Signal'] == 1]
        plt.scatter(overbought_regions.index, overbought_regions['close'], color='red', marker='v', s=100, label='Overbought')
        
        plt.legend()
        plt.grid(True)
        
        # Plot 2: RSI
        plt.subplot(212)
        plt.title('RSI(14)')
        plt.plot(plot_data.index, plot_data['RSI_Direct'], label='RSI', color='blue')
        
        # Add RSI thresholds
        plt.axhline(y=70, color='r', linestyle='-', alpha=0.5, label='Overbought (70)')
        plt.axhline(y=30, color='g', linestyle='-', alpha=0.5, label='Oversold (30)')
        
        # Color RSI based on signals
        plt.fill_between(plot_data.index, y1=30, y2=plot_data['RSI_Direct'], where=(plot_data['RSI_Direct'] < 30), 
                          color='green', alpha=0.3)
        plt.fill_between(plot_data.index, y1=70, y2=plot_data['RSI_Direct'], where=(plot_data['RSI_Direct'] > 70), 
                          color='red', alpha=0.3)
        
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('rsi_signals_test.png')
        logger.info("Saved RSI signals test plot to rsi_signals_test.png")
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
    
    logger.info("RSI signals test completed")

if __name__ == "__main__":
    test_rsi_signals() 