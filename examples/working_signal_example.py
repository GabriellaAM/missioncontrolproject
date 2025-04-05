#!/usr/bin/env python
"""
Working example of asset-specific signal selection.

This script demonstrates how to use the signal framework without relying on
the register_signal decorator's return value.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import signal components directly
from soros_system.data.data_loader import DataLoader
from soros_system.signals.signal_base import SignalBase

# Import modules instead of classes to avoid decorator issues
import soros_system.signals.trend_signals as trend_signals
import soros_system.signals.rsi_signals as rsi_signals
import soros_system.signals.volatility_signals as vol_signals

from soros_system.analysis.forward_returns.signal_evaluator import SignalEvaluator
from soros_system.portfolio.signal_combiner import SignalCombiner


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('working_signal_example.log')
        ]
    )
    
    # Reduce verbosity of some modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def load_asset_data(asset_ids):
    """Load data for specific assets.
    
    Args:
        asset_ids (list): List of asset IDs to load
        
    Returns:
        dict: Dictionary mapping asset_ids to DataFrames
    """
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
    """Create signal instances for testing.
    
    Args:
        quote_type (str): Quote type to use ('USD' or 'BTC')
        
    Returns:
        list: List of signal instances
    """
    # Create signal instances manually by importing from modules
    signals = [
        # Trend signals
        trend_signals.ShortTermTrendSignal(params={
            'quote_type': quote_type,
            'trend_values': [1, 2]  # Bullish values
        }),
        trend_signals.MediumTermTrendSignal(params={
            'quote_type': quote_type,
            'trend_values': [1, 2]  # Bullish values
        }),
        trend_signals.LongTermTrendSignal(params={
            'quote_type': quote_type,
            'trend_values': [1, 2]  # Bullish values
        }),
        trend_signals.OverallTrendSignal(params={
            'quote_type': quote_type,
            'trend_values': [1, 2]  # Bullish values
        }),
        
        # RSI signals
        rsi_signals.RSISignal(params={
            'quote_type': quote_type,
            'rsi_length': 28,
            'signal_mode': 'simple'  # Simple threshold at 50
        }),
        rsi_signals.RSIWithRoCSignal(params={
            'quote_type': quote_type,
            'rsi_length': 28,
            'roc_length': 28
        }),
        
        # Volatility signals
        vol_signals.VolatilityTrendSignal(params={
            'window': 14,
            'threshold': 0.1
        })
    ]
    
    return signals


def process_asset_signals(asset_id, data, signals, threshold=0.0):
    """Process signals for a specific asset.
    
    Args:
        asset_id (str): ID of the asset
        data (pd.DataFrame): Asset data
        signals (list): List of signal instances
        threshold (float): Threshold for final decision
        
    Returns:
        tuple: (evaluations, weights, processed_data)
    """
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


def plot_weighted_signals(asset_id, processed_data, weights, output_file):
    """Plot the weighted signals for an asset.
    
    Args:
        asset_id (str): ID of the asset
        processed_data (pd.DataFrame): Processed data with signals
        weights (dict): Signal weights
        output_file (str): Output file for the plot
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get signal columns
        signal_cols = [col for col in processed_data.columns if col.endswith('_signal')]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot price
        ax1.plot(processed_data.index, processed_data['close'], label='Price', color='black')
        ax1.set_title(f"{asset_id.capitalize()} Price")
        ax1.set_ylabel("Price (USD)")
        ax1.grid(True)
        ax1.legend()
        
        # Plot individual signals
        cmap = plt.cm.get_cmap('tab10', len(signal_cols))
        for i, col in enumerate(signal_cols):
            signal_name = col.replace('_signal', '')
            weight = weights.get(signal_name, 0)
            if weight > 0:
                ax2.plot(processed_data.index, processed_data[col] * weight, 
                         label=f"{signal_name} ({weight:.2f})", 
                         color=cmap(i), alpha=0.7, linewidth=1)
        
        # Plot combined signal
        ax2.plot(processed_data.index, processed_data['combined_signal'], 
                 label='Combined Signal', color='blue', linewidth=2)
        
        # Plot decision points
        if 'final_decision_shifted' in processed_data.columns:
            decision_dates = processed_data.index[processed_data['final_decision_shifted'] == 1]
            if len(decision_dates) > 0:
                min_val, max_val = ax2.get_ylim()
                ax2.vlines(decision_dates, min_val, max_val * 0.9, 
                           colors='green', alpha=0.4, label='Buy Signal')
        
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_title("Weighted Signals")
        ax2.set_ylabel("Signal Value")
        ax2.set_xlabel("Date")
        ax2.grid(True)
        ax2.legend(loc='upper left', fontsize='small')
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Saved signal plot to {output_file}")
        
    except Exception as e:
        logger.error(f"Error plotting signals for {asset_id}: {e}")


def main():
    """Main function."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting working signal example")
    
    # Load asset data for Bitcoin and Ethereum
    asset_data = load_asset_data(['bitcoin', 'ethereum'])
    
    if not asset_data:
        logger.error("No asset data loaded. Exiting.")
        return
    
    # Process each asset
    for asset_id, data in asset_data.items():
        # Create signals for the asset
        signals = create_signals_for_asset(quote_type='USD')
        
        # Process the asset
        evaluations, weights, processed_data = process_asset_signals(
            asset_id, data, signals, threshold=0.0
        )
        
        # Plot the results
        plot_weighted_signals(
            asset_id, 
            processed_data, 
            weights, 
            f"{asset_id}_weighted_signals.png"
        )
    
    logger.info("Working signal example completed")


if __name__ == "__main__":
    main() 