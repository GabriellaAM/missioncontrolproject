#!/usr/bin/env python
"""
Example demonstrating how to use asset-specific signal selection in Soros System.

This script shows how to create a portfolio that uses the SignalSelector
to dynamically select and weigh signals for each asset based on their effectiveness.
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
from soros_system.portfolio.portfolio_manager import PortfolioManager
from soros_system.portfolio.signal_selector import SignalSelector
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
    VolatilityTrendSignal
)
from soros_system.analysis.forward_returns import (
    ForwardReturnsCalculator,
    StatisticalTester,
    SignalEvaluator
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('asset_specific_signals_example.log')
        ]
    )
    
    # Reduce verbosity of some modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def load_asset_data(data_loader, asset_ids=None):
    """Load data for multiple assets.
    
    Args:
        data_loader: DataLoader instance
        asset_ids: List of asset IDs to load, if None loads all available assets
        
    Returns:
        dict: Dictionary mapping asset_ids to DataFrames
    """
    logger = logging.getLogger(__name__)
    
    if asset_ids is None:
        # List available asset files
        asset_ids = data_loader.list_available_assets()
        
    # Load a limited subset for example purposes
    if len(asset_ids) > 5:
        asset_ids = asset_ids[:5]  # Use only first 5 assets for this example
    
    logger.info(f"Loading data for {len(asset_ids)} assets")
    
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


def create_dynamic_signal_portfolio(portfolio_manager, asset_data):
    """Create a portfolio using dynamic signal selection.
    
    Args:
        portfolio_manager: PortfolioManager instance
        asset_data: Dictionary mapping asset_ids to DataFrames
        
    Returns:
        str: Name of the created portfolio
    """
    logger = logging.getLogger(__name__)
    
    # Define portfolio name
    portfolio_name = f"dynamic_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create portfolio with dynamic signal selection enabled
    portfolio_manager.create_portfolio(
        portfolio_name=portfolio_name,
        classified_data=asset_data,
        use_signal_selector=True,  # Enable dynamic signal selection
        usd_conditions=[1, 2],     # Bullish USD trend values
        btc_conditions=[1, 2],     # Bullish BTC trend values
        rsi_conditions_usd=True,   # Use USD RSI
        rsi_conditions_btc=True,   # Use BTC RSI
        use_volatility_filter=True,  # Include volatility filter
        signal_threshold=50,       # Threshold for positive signals
    )
    
    logger.info(f"Created portfolio '{portfolio_name}' with dynamic signal selection")
    return portfolio_name


def create_static_signal_portfolio(portfolio_manager, asset_data):
    """Create a portfolio using static signal selection (original method).
    
    Args:
        portfolio_manager: PortfolioManager instance
        asset_data: Dictionary mapping asset_ids to DataFrames
        
    Returns:
        str: Name of the created portfolio
    """
    logger = logging.getLogger(__name__)
    
    # Define portfolio name
    portfolio_name = f"static_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create portfolio with traditional static signal approach
    portfolio_manager.create_portfolio(
        portfolio_name=portfolio_name,
        classified_data=asset_data,
        use_signal_selector=False,  # Disable dynamic signal selection
        usd_conditions=[1, 2],      # Same criteria as dynamic portfolio
        btc_conditions=[1, 2], 
        rsi_conditions_usd=True,
        rsi_conditions_btc=True,
        use_volatility_filter=True,
        signal_threshold=50,
    )
    
    logger.info(f"Created portfolio '{portfolio_name}' with static signal selection")
    return portfolio_name


def compare_portfolios(portfolio_manager, dynamic_portfolio, static_portfolio):
    """Compare performance between dynamic and static signal portfolios.
    
    Args:
        portfolio_manager: PortfolioManager instance
        dynamic_portfolio: Name of the dynamic signal portfolio
        static_portfolio: Name of the static signal portfolio
    """
    logger = logging.getLogger(__name__)
    logger.info("Comparing dynamic vs static signal portfolios")
    
    # Get signal data for both portfolios
    dynamic_signals = portfolio_manager.get_portfolio_signals(dynamic_portfolio)
    static_signals = portfolio_manager.get_portfolio_signals(static_portfolio)
    
    if not dynamic_signals or not static_signals:
        logger.error("Failed to retrieve signal data for one or both portfolios")
        return
    
    # Compare signal differences for each asset
    for asset_id in dynamic_signals.keys():
        if asset_id not in static_signals:
            continue
            
        logger.info(f"Comparing signals for {asset_id}")
        
        dynamic_df = dynamic_signals[asset_id]
        static_df = static_signals[asset_id]
        
        # Ensure both have date indices for alignment
        if not isinstance(dynamic_df.index, pd.DatetimeIndex):
            dynamic_df = dynamic_df.set_index('date')
        if not isinstance(static_df.index, pd.DatetimeIndex):
            static_df = static_df.set_index('date')
        
        # Align indices
        common_index = dynamic_df.index.intersection(static_df.index)
        if len(common_index) == 0:
            logger.warning(f"No common dates found for {asset_id}")
            continue
            
        # Filter to common dates
        dynamic_aligned = dynamic_df.loc[common_index]
        static_aligned = static_df.loc[common_index]
        
        # Compare signals
        if 'final_decision_shifted' in dynamic_aligned.columns and 'final_decision_shifted' in static_aligned.columns:
            # Calculate agreement percentage
            agreement = (dynamic_aligned['final_decision_shifted'] == static_aligned['final_decision_shifted']).mean() * 100
            logger.info(f"Signal agreement for {asset_id}: {agreement:.2f}%")
            
            # Plot comparison
            plot_signal_comparison(
                asset_id, 
                dynamic_aligned, 
                static_aligned,
                f"signal_comparison_{asset_id}.png"
            )


def plot_signal_comparison(asset_id, dynamic_df, static_df, output_file):
    """Plot comparison of signals from dynamic and static portfolios.
    
    Args:
        asset_id: Asset ID
        dynamic_df: DataFrame with dynamic signals
        static_df: DataFrame with static signals
        output_file: File to save the plot
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot price
        ax1.plot(dynamic_df.index, dynamic_df['close'], label='Price', color='black')
        ax1.set_title(f"{asset_id.capitalize()} Price and Signals")
        ax1.set_ylabel("Price (USD)")
        ax1.grid(True)
        ax1.legend()
        
        # Plot dynamic signals
        ax2.plot(
            dynamic_df.index, 
            dynamic_df['combined_signal'] if 'combined_signal' in dynamic_df.columns else 0,
            label='Dynamic Combined Signal', 
            color='blue'
        )
        ax2.axhline(y=0, color='gray', linestyle='--')
        
        # Plot buy signals for dynamic portfolio
        if 'final_decision_shifted' in dynamic_df.columns:
            buy_dates = dynamic_df.index[dynamic_df['final_decision_shifted'] == 1]
            if len(buy_dates) > 0:
                min_val, max_val = ax2.get_ylim()
                ax2.vlines(buy_dates, min_val, max_val * 0.9, colors='green', alpha=0.3)
        
        ax2.set_title("Dynamic Signal Portfolio")
        ax2.set_ylabel("Signal Value")
        ax2.grid(True)
        ax2.legend()
        
        # Plot static signals
        ax3.plot(
            static_df.index,
            static_df['combined_signal_sum'] / len([c for c in static_df.columns if c.endswith('_signal')]) if 'combined_signal_sum' in static_df.columns else 0,
            label='Static Combined Signal', 
            color='red'
        )
        ax3.axhline(y=0, color='gray', linestyle='--')
        
        # Plot buy signals for static portfolio
        if 'final_decision_shifted' in static_df.columns:
            buy_dates = static_df.index[static_df['final_decision_shifted'] == 1]
            if len(buy_dates) > 0:
                min_val, max_val = ax3.get_ylim()
                ax3.vlines(buy_dates, min_val, max_val * 0.9, colors='orange', alpha=0.3)
        
        ax3.set_title("Static Signal Portfolio")
        ax3.set_ylabel("Signal Value")
        ax3.set_xlabel("Date")
        ax3.grid(True)
        ax3.legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Saved signal comparison plot to {output_file}")
        
    except Exception as e:
        logger.error(f"Error plotting signal comparison for {asset_id}: {e}")


def main():
    """Main function."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting asset-specific signals example")
    
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
    
    # Create signal selector with custom parameters
    signal_selector = SignalSelector(
        lookback_days=180,  # Use 6 months of data for evaluation
        min_samples=30,     # Minimum 30 samples for statistical validity
        recalculate_interval_days=30  # Recalculate weights every 30 days
    )
    
    # Create portfolio manager with the signal selector
    portfolio_manager = PortfolioManager(
        signal_selector=signal_selector
    )
    
    # Load asset data for Bitcoin and Ethereum only
    asset_data = load_asset_data(data_loader, asset_ids=['bitcoin', 'ethereum'])
    
    if not asset_data:
        logger.error("No asset data loaded. Exiting.")
        return
    
    # Create portfolios for comparison
    dynamic_portfolio = create_dynamic_signal_portfolio(
        portfolio_manager, asset_data
    )
    
    static_portfolio = create_static_signal_portfolio(
        portfolio_manager, asset_data
    )
    
    # Compare the two portfolios
    compare_portfolios(
        portfolio_manager, dynamic_portfolio, static_portfolio
    )
    
    logger.info("Asset-specific signals example completed")


if __name__ == "__main__":
    main() 