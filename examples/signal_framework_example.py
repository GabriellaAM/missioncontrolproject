#!/usr/bin/env python
"""
Example demonstrating how to use the Soros System signal framework.

This script shows how to create signals, evaluate their effectiveness,
calculate optimal weights, and combine them into a final decision.
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

# Import Soros System components
from soros_system.data.data_loader import DataLoader
from soros_system.signals import (
    get_signal, 
    get_all_signals, 
    SignalBase
)
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
    MarkovVolatilitySignal,
    VolatilityTrendSignal
)
from soros_system.analysis.forward_returns import (
    ForwardReturnsCalculator,
    StatisticalTester,
    SignalEvaluator
)
from soros_system.portfolio import SignalCombiner


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('signal_framework_example.log')
        ]
    )
    
    # Reduce verbosity of some modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def load_asset_data(asset_id='bitcoin', data_path=None):
    """Load data for an asset.
    
    Args:
        asset_id (str): Asset ID to load
        data_path (str, optional): Path to data directory
        
    Returns:
        pd.DataFrame: DataFrame with asset data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data for {asset_id}")
    
    # Determine data path
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'data', 'micro', 'candleData')
    
    # Create data loader
    data_loader = DataLoader(data_path=data_path)
    
    # Load asset data
    data = data_loader.load_asset_data(asset_id)
    
    logger.info(f"Loaded {len(data)} rows for {asset_id}")
    return data


def create_signals():
    """Create signal instances.
    
    Returns:
        list: List of signal instances
    """
    logger = logging.getLogger(__name__)
    
    # Create trend signals
    st_trend_signal = ShortTermTrendSignal(params={
        'quote_type': 'USD',
        'trend_values': [1, 2]  # Bullish values
    })
    
    mt_trend_signal = MediumTermTrendSignal(params={
        'quote_type': 'USD',
        'trend_values': [1, 2]  # Bullish values
    })
    
    lt_trend_signal = LongTermTrendSignal(params={
        'quote_type': 'USD',
        'trend_values': [1, 2]  # Bullish values
    })
    
    overall_trend_signal = OverallTrendSignal(params={
        'quote_type': 'USD',
        'trend_values': [1, 2]  # Bullish values
    })
    
    # Create RSI signals
    rsi_signal = RSISignal(params={
        'quote_type': 'USD',
        'rsi_length': 28,
        'signal_mode': 'simple'  # Simple threshold at 50
    })
    
    rsi_roc_signal = RSIWithRoCSignal(params={
        'quote_type': 'USD',
        'rsi_length': 28,
        'roc_length': 28
    })
    
    # Create volatility signals
    vol_trend_signal = VolatilityTrendSignal(params={
        'window': 14,
        'threshold': 0.1
    })
    
    # Collect all signals
    signals = [
        st_trend_signal,
        mt_trend_signal,
        lt_trend_signal,
        overall_trend_signal,
        rsi_signal,
        rsi_roc_signal,
        vol_trend_signal
    ]
    
    logger.info(f"Created {len(signals)} signals")
    return signals


def evaluate_signals(signals, data, asset_id):
    """Evaluate signal effectiveness.
    
    Args:
        signals (list): List of signal instances
        data (pd.DataFrame): Asset data
        asset_id (str): Asset ID
        
    Returns:
        dict: Dictionary mapping signal names to evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {len(signals)} signals for {asset_id}")
    
    # Create signal evaluator
    evaluator = SignalEvaluator(
        periods=[1, 3, 5, 7, 14, 21, 28],
        min_samples=30,
        lookback_days=365  # Use last year of data
    )
    
    # Evaluate signals
    evaluations = evaluator.evaluate_multiple_signals(signals, data, asset_id)
    
    # Calculate weights
    weights = evaluator.calculate_normalized_weights(evaluations)
    
    logger.info(f"Signal weights: {weights}")
    return evaluations, weights


def combine_signals(signals, data, asset_id, weights=None):
    """Combine signals into a final decision.
    
    Args:
        signals (list): List of signal instances
        data (pd.DataFrame): Asset data
        asset_id (str): Asset ID
        weights (dict, optional): Dictionary mapping signal names to weights
        
    Returns:
        tuple: (combined_signal, final_decision, weights)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Combining signals for {asset_id}")
    
    # Create signal combiner
    combiner = SignalCombiner(threshold=0.0, auto_weights=(weights is None))
    
    # Combine signals
    combined_signal, final_decision, used_weights = combiner.combine_signals(
        signals, data, asset_id, weights=weights
    )
    
    logger.info(f"Combined {len(signals)} signals with {len(used_weights)} weights")
    return combined_signal, final_decision, used_weights


def plot_results(data, combined_signal, final_decision, asset_id):
    """Plot results.
    
    Args:
        data (pd.DataFrame): Asset data
        combined_signal (pd.Series): Combined signal values
        final_decision (pd.Series): Final decision values
        asset_id (str): Asset ID
    """
    logger = logging.getLogger(__name__)
    logger.info("Plotting results")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot price
    ax1.plot(data.index, data['close'], label='Price', color='black')
    ax1.set_title(f"{asset_id.capitalize()} Price and Signals")
    ax1.set_ylabel("Price (USD)")
    ax1.grid(True)
    
    # Plot combined signal
    ax2.plot(combined_signal.index, combined_signal, label='Combined Signal', color='blue')
    ax2.axhline(y=0, color='gray', linestyle='--')
    
    # Plot buy signals
    buy_dates = final_decision[final_decision == 1].index
    if not buy_dates.empty:
        min_val, max_val = ax2.get_ylim()
        ax2.vlines(buy_dates, min_val, max_val * 0.9, colors='green', alpha=0.3)
    
    ax2.set_title("Signal and Decisions")
    ax2.set_ylabel("Signal Value")
    ax2.set_xlabel("Date")
    ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{asset_id}_signals.png")
    logger.info(f"Saved plot to {asset_id}_signals.png")
    
    # Show figure
    plt.show()


def main():
    """Main function."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting signal framework example")
    
    # Load data
    asset_id = 'bitcoin'
    data = load_asset_data(asset_id)
    
    # Create signals
    signals = create_signals()
    
    # Evaluate signals
    evaluations, weights = evaluate_signals(signals, data, asset_id)
    
    # Combine signals
    combined_signal, final_decision, weights = combine_signals(signals, data, asset_id, weights)
    
    # Plot results
    plot_results(data, combined_signal, final_decision, asset_id)
    
    logger.info("Example completed")


if __name__ == "__main__":
    main() 