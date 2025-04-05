#!/usr/bin/env python
"""
Signal performance comparison example.

This script demonstrates signal performance vs buy-and-hold and uses
dot visualization for clearer signal weight representation.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mtick

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Soros System components
from soros_system.data.data_loader import DataLoader
from soros_system.signals.signal_base import SignalBase
from soros_system.signals.trend_signals import (
    ShortTermTrendSignal, 
    MediumTermTrendSignal, 
    LongTermTrendSignal,
    OverallTrendSignal
)
from soros_system.signals.rsi_signals import (
    RSISignal,
    RSIWithRoCSignal
)
from soros_system.signals.volatility_signals import (
    VolatilityTrendSignal
)
from soros_system.analysis.forward_returns.signal_evaluator import SignalEvaluator
from soros_system.portfolio.signal_combiner import SignalCombiner


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('signal_performance_comparison.log')
        ]
    )
    
    # Reduce verbosity of some modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def load_asset_data(asset_ids):
    """Load data for specific assets."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data for {len(asset_ids)} assets")
    
    # Create data loader with proper paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'micro', 'candleData')
    btc_data_path = os.path.join(base_dir, 'data', 'micro', 'candleData', 'bitcoin_candles.csv')
    market_data_path = os.path.join(base_dir, 'data', 'micro', 'assetData')
    
    data_loader = DataLoader(
        data_path=data_path,
        btc_data_path=btc_data_path,
        market_data_path=market_data_path
    )
    
    # Load data for each asset
    asset_data = {}
    for asset_id in asset_ids:
        try:
            data = data_loader.load_asset_data(asset_id)
            asset_data[asset_id] = data
            logger.info(f"Loaded {len(data)} rows for {asset_id}")
        except Exception as e:
            logger.error(f"Error loading data for {asset_id}: {e}")
    
    return asset_data


def create_signals_for_asset(quote_type='USD'):
    """Create signal instances for testing."""
    # Create signals
    signals = [
        ShortTermTrendSignal(params={
            'quote_type': quote_type,
            'trend_values': [1, 2]  # Bullish values
        }),
        MediumTermTrendSignal(params={
            'quote_type': quote_type,
            'trend_values': [1, 2]  # Bullish values
        }),
        LongTermTrendSignal(params={
            'quote_type': quote_type,
            'trend_values': [1, 2]  # Bullish values
        }),
        OverallTrendSignal(params={
            'quote_type': quote_type,
            'trend_values': [1, 2]  # Bullish values
        }),
        RSISignal(params={
            'quote_type': quote_type,
            'rsi_length': 28,
            'signal_mode': 'simple'  # Simple threshold at 50
        }),
        RSIWithRoCSignal(params={
            'quote_type': quote_type,
            'rsi_length': 28,
            'roc_length': 28
        }),
        VolatilityTrendSignal(params={
            'window': 14,
            'threshold': 0.1
        })
    ]
    
    return signals


def process_asset_signals(asset_id, data, signals, threshold=0.0):
    """Process signals for a specific asset."""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing signals for {asset_id}")
    
    # Create signal evaluator
    evaluator = SignalEvaluator(
        lookback_days=180,
        min_samples=30
    )
    
    # Create signal combiner
    combiner = SignalCombiner(
        signal_evaluator=evaluator,
        threshold=threshold
    )
    
    # Evaluate signals
    evaluations = evaluator.evaluate_multiple_signals(signals, data, asset_id)
    
    # Calculate weights
    weights = evaluator.calculate_normalized_weights(evaluations)
    
    # Print weights
    logger.info(f"Signal weights for {asset_id}:")
    for signal_name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {signal_name}: {weight:.4f}")
    
    # Select signals with positive weights
    selected_signals = [s for s in signals if s.name in weights and weights[s.name] > 0]
    
    # Combine signals
    combined_signal, final_decision, used_weights = combiner.combine_signals(
        selected_signals, data, asset_id, weights=weights
    )
    
    # Add signals to data
    result_df = data.copy()
    result_df['combined_signal'] = combined_signal
    result_df['final_decision'] = final_decision
    result_df['final_decision_shifted'] = final_decision.shift(1)
    
    # Add individual signal columns
    for signal in selected_signals:
        try:
            col_name = f"{signal.name}_signal"
            signal_values = signal.calculate(data, asset_id)
            result_df[col_name] = signal_values
        except Exception as e:
            logger.error(f"Error calculating {signal.name} for {asset_id}: {e}")
    
    return evaluations, weights, result_df


def calculate_performance(data, final_decision_col='final_decision_shifted'):
    """Calculate performance metrics for signal strategy vs buy and hold.
    
    Args:
        data: DataFrame with price data and signals
        final_decision_col: Column with final decision signals (shifted to avoid lookahead)
        
    Returns:
        dict: Performance metrics
    """
    logger = logging.getLogger(__name__)
    
    # Make sure we have all required columns
    required_cols = ['close', final_decision_col]
    if not all(col in data.columns for col in required_cols):
        logger.error(f"Missing required columns: {[col for col in required_cols if col not in data.columns]}")
        return {}
    
    # Create daily returns
    data['daily_return'] = data['close'].pct_change()
    
    # Strategy returns (only take positions when signal is 1)
    data['strategy_return'] = data['daily_return'] * data[final_decision_col].shift(1).fillna(0)
    
    # Prepare start and end values for both approaches
    buy_hold_start = data['close'].iloc[1]  # Skip the first row as it has no return
    buy_hold_end = data['close'].iloc[-1]
    
    # Calculate performance metrics
    perf = {}
    
    # Buy and hold metrics
    perf['buy_hold_return'] = (buy_hold_end / buy_hold_start) - 1.0
    perf['buy_hold_days'] = len(data) - 1
    
    # Signal strategy metrics
    # Calculate cumulative returns
    data['buy_hold_cum_return'] = (1 + data['daily_return']).cumprod() - 1
    data['strategy_cum_return'] = (1 + data['strategy_return']).cumprod() - 1
    
    perf['strategy_return'] = data['strategy_cum_return'].iloc[-1]
    
    # Calculate trading days and exposure
    perf['trading_days'] = data[final_decision_col].sum()
    perf['exposure'] = perf['trading_days'] / perf['buy_hold_days']
    
    # Calculate Sharpe Ratio (annual)
    perf['buy_hold_sharpe'] = (data['daily_return'].mean() / data['daily_return'].std()) * np.sqrt(252)
    perf['strategy_sharpe'] = (data['strategy_return'].mean() / data['strategy_return'].std()) * np.sqrt(252)
    
    # Calculate maximum drawdowns
    data['buy_hold_drawdown'] = data['buy_hold_cum_return'] - data['buy_hold_cum_return'].cummax()
    data['strategy_drawdown'] = data['strategy_cum_return'] - data['strategy_cum_return'].cummax()
    
    perf['buy_hold_max_drawdown'] = data['buy_hold_drawdown'].min()
    perf['strategy_max_drawdown'] = data['strategy_drawdown'].min()
    
    return perf, data


def plot_performance_comparison(asset_id, data, weights, output_file):
    """Plot performance comparison between signal strategy and buy-and-hold.
    
    Args:
        asset_id: ID of the asset
        data: DataFrame with performance data
        weights: Signal weights dictionary
        output_file: Output file for the plot
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get signal columns with positive weights
        signal_cols = [col for col in data.columns if col.endswith('_signal')]
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True, 
                                           gridspec_kw={'height_ratios': [2, 1, 2]})
        
        # Plot price
        ax1.plot(data.index, data['close'], label='Price', color='black')
        ax1.set_title(f"{asset_id.capitalize()} Price")
        ax1.set_ylabel("Price (USD)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot weights as dots when the signal is active (more sparse visualization)
        for i, col in enumerate(signal_cols):
            signal_name = col.replace('_signal', '')
            weight = weights.get(signal_name, 0)
            if weight > 0:
                # Only plot dots where the signal is active (=1)
                active_points = data.index[data[col] == 1]
                inactive_points = data.index[data[col] == -1]
                
                # Plot active signals with larger dots
                ax2.scatter(active_points, [weight] * len(active_points), 
                           label=f"{signal_name} ({weight:.2f})", 
                           s=40, alpha=0.7)
                
                # Plot inactive signals with smaller dots
                ax2.scatter(inactive_points, [weight] * len(inactive_points), 
                           color='lightgray', s=10, alpha=0.3)
        
        # Plot combined signal overlay
        ax2.plot(data.index, data['combined_signal'], 
                label='Combined Signal', color='blue', linewidth=1)
        
        # Add threshold line
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, 
                   label='Signal Threshold')
        
        ax2.set_title("Signal Weights (dots) and Combined Signal (line)")
        ax2.set_ylabel("Weight")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize='small')
        
        # Plot cumulative returns comparison
        if 'buy_hold_cum_return' in data.columns and 'strategy_cum_return' in data.columns:
            ax3.plot(data.index, data['buy_hold_cum_return'] * 100, 
                    label='Buy & Hold', color='gray')
            ax3.plot(data.index, data['strategy_cum_return'] * 100, 
                    label='Signal Strategy', color='green')
            
            # Add active period shading
            if 'final_decision_shifted' in data.columns:
                bottom, top = ax3.get_ylim()
                for i in range(len(data)-1):
                    if data['final_decision_shifted'].iloc[i] == 1:
                        ax3.axvspan(data.index[i], data.index[i+1], 
                                   alpha=0.1, color='green')
            
            # Format the y-axis as percentage
            ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
            
            ax3.set_title("Cumulative Returns")
            ax3.set_ylabel("Return (%)")
            ax3.set_xlabel("Date")
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left')
        
        # Format the date on the x-axis
        date_form = DateFormatter("%m/%y")
        ax3.xaxis.set_major_formatter(date_form)
        fig.autofmt_xdate()  # Rotate date labels
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        logger.info(f"Saved performance plot to {output_file}")
        
    except Exception as e:
        logger.error(f"Error plotting performance for {asset_id}: {e}")


def print_performance_metrics(asset_id, perf):
    """Print performance metrics in a readable format.
    
    Args:
        asset_id: ID of the asset
        perf: Dictionary with performance metrics
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Performance metrics for {asset_id}:")
    logger.info(f"  Buy & Hold Return: {perf['buy_hold_return']*100:.2f}%")
    logger.info(f"  Strategy Return: {perf['strategy_return']*100:.2f}%")
    logger.info(f"  Strategy Outperformance: {(perf['strategy_return'] - perf['buy_hold_return'])*100:.2f}%")
    logger.info(f"  Market Exposure: {perf['exposure']*100:.2f}% of days")
    logger.info(f"  Buy & Hold Sharpe: {perf['buy_hold_sharpe']:.2f}")
    logger.info(f"  Strategy Sharpe: {perf['strategy_sharpe']:.2f}")
    logger.info(f"  Buy & Hold Max Drawdown: {perf['buy_hold_max_drawdown']*100:.2f}%")
    logger.info(f"  Strategy Max Drawdown: {perf['strategy_max_drawdown']*100:.2f}%")


def main():
    """Main function."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting signal performance comparison")
    
    # Load asset data for Bitcoin and Ethereum
    asset_data = load_asset_data(['bitcoin', 'ethereum'])
    
    if not asset_data:
        logger.error("No asset data loaded. Exiting.")
        return
    
    # Process each asset
    for asset_id, data in asset_data.items():
        # Create signals for the asset
        signals = create_signals_for_asset(quote_type='USD')
        
        # Process the asset to get signals
        evaluations, weights, processed_data = process_asset_signals(
            asset_id, data, signals, threshold=0.0
        )
        
        # Calculate performance metrics
        perf, perf_data = calculate_performance(processed_data)
        
        # Print performance metrics
        print_performance_metrics(asset_id, perf)
        
        # Plot the performance comparison
        plot_performance_comparison(
            asset_id, 
            perf_data, 
            weights, 
            f"{asset_id}_performance_comparison.png"
        )
    
    logger.info("Signal performance comparison completed")


if __name__ == "__main__":
    main() 