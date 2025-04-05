#!/usr/bin/env python
"""
Fixed example that avoids the register_signal decorator issue.

This script works around the issue by creating signal classes directly
without using the register_signal decorator.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import base components directly
from soros_system.data.data_loader import DataLoader
from soros_system.signals.signal_base import SignalBase
from soros_system.indicators.trend_classifier import TrendClassifier
from soros_system.indicators.moving_averages import MovingAverageCalculator
from soros_system.indicators.rsi import RSICalculator
from soros_system.analysis.forward_returns.signal_evaluator import SignalEvaluator
from soros_system.portfolio.signal_combiner import SignalCombiner


# Create custom classes by inheriting from SignalBase directly
class CustomShortTermTrendSignal(SignalBase):
    """Signal based on short-term trend classification."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = self.params.get('quote_type', 'USD')
        self.trend_values = self.params.get('trend_values', [1, 2])  # Bullish by default
        self.ma_calculator = MovingAverageCalculator()
        self.trend_classifier = TrendClassifier(self.ma_calculator)
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        return ['close'] if self.quote_type == 'USD' else [f'{self._get_asset_id()}_btc']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return 60  # Need at least 60 days for reliable trend classification
    
    def _get_asset_id(self) -> str:
        """Extract asset_id from parameters."""
        return self.params.get('asset_id', '')
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation."""
        if data is None or data.empty:
            self.logger.warning(f"Empty data provided for {asset_id}")
            return False
        
        # Check for minimum required samples
        if len(data) < self.get_min_required_samples():
            self.logger.warning(
                f"Insufficient data for {asset_id}: {len(data)} samples, "
                f"need at least {self.get_min_required_samples()}"
            )
            return False
        
        # Check for required columns
        required_cols = self.get_required_columns()
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            self.logger.warning(f"Missing required columns for {asset_id}: {missing_cols}")
            return False
        
        return True
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values."""
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Calculate trend classifications
        classified_data = self.trend_classifier._create_classified_data(data, asset_id)
        
        # Get the short-term trend column
        trend_col = f'short_term_trend_{self.quote_type}'
        
        if trend_col not in classified_data.columns:
            self.logger.warning(f"Trend column {trend_col} not found for {asset_id}")
            return pd.Series(index=data.index)
        
        # Create signal based on trend values
        signal = classified_data[trend_col].apply(
            lambda x: 1 if pd.notna(x) and x in self.trend_values else -1
        )
        
        return signal


class CustomRSISignal(SignalBase):
    """Signal based on RSI crossing specific thresholds."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = self.params.get('quote_type', 'USD')
        self.rsi_length = self.params.get('rsi_length', 28)
        self.oversold = self.params.get('oversold', 30)
        self.overbought = self.params.get('overbought', 70)
        self.signal_mode = self.params.get('signal_mode', 'simple')
        self.rsi_calculator = RSICalculator()
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        return ['close'] if self.quote_type == 'USD' else [f'{self._get_asset_id()}_btc']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return self.rsi_length + 10  # Need RSI length plus some extra samples
    
    def _get_asset_id(self) -> str:
        """Extract asset_id from parameters."""
        return self.params.get('asset_id', '')
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation."""
        if data is None or data.empty:
            self.logger.warning(f"Empty data provided for {asset_id}")
            return False
        
        # Check for minimum required samples
        if len(data) < self.get_min_required_samples():
            self.logger.warning(
                f"Insufficient data for {asset_id}: {len(data)} samples, "
                f"need at least {self.get_min_required_samples()}"
            )
            return False
        
        # Check for required columns
        required_cols = self.get_required_columns()
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            self.logger.warning(f"Missing required columns for {asset_id}: {missing_cols}")
            return False
        
        return True
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values."""
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Determine price column based on quote type
        price_col = 'close' if self.quote_type == 'USD' else f'{asset_id}_btc'
        
        # Calculate RSI using the RSI calculator
        try:
            # Use the existing RSI calculation logic
            rsi_data = self.rsi_calculator.calculate_smooth_rsi(
                data, price_col, rsi_length=self.rsi_length, roc_length=self.rsi_length
            )
            
            # Get the RSI column name
            rsi_col = f'RSI_{price_col}'
            
            if rsi_col not in rsi_data.columns:
                self.logger.warning(f"RSI column {rsi_col} not found for {asset_id}")
                return pd.Series(index=data.index)
            
            # Create signal based on RSI values
            if self.signal_mode == 'simple':
                # Simple mode: bullish when RSI > 50, bearish when RSI <= 50
                signal = rsi_data[rsi_col].apply(
                    lambda x: 1 if pd.notna(x) and x > 50 else -1
                )
            else:
                # Zone mode: bullish when RSI < oversold, bearish when RSI > overbought, neutral otherwise
                signal = rsi_data[rsi_col].apply(
                    lambda x: 1 if pd.notna(x) and x < self.oversold else 
                             (-1 if pd.notna(x) and x > self.overbought else 
                              (0 if pd.notna(x) else np.nan))
                )
                
                # Convert neutral (0) to -1 for binary signal requirement
                signal = signal.apply(lambda x: -1 if x == 0 else x)
            
            return signal
        
        except Exception as e:
            self.logger.error(f"Error calculating RSI signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


class CustomVolatilityTrendSignal(SignalBase):
    """Signal based on the trend of volatility changes."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.window = self.params.get('window', 14)
        self.threshold = self.params.get('threshold', 0.1)
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        return ['close']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return self.window * 2  # Need enough data to calculate volatility trend
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation."""
        if data is None or data.empty:
            self.logger.warning(f"Empty data provided for {asset_id}")
            return False
        
        # Check for minimum required samples
        if len(data) < self.get_min_required_samples():
            self.logger.warning(
                f"Insufficient data for {asset_id}: {len(data)} samples, "
                f"need at least {self.get_min_required_samples()}"
            )
            return False
        
        # Check for required columns
        required_cols = self.get_required_columns()
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            self.logger.warning(f"Missing required columns for {asset_id}: {missing_cols}")
            return False
        
        return True
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values."""
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.window).std()
        
        # Calculate volatility slope (trend)
        volatility_change = volatility.pct_change(self.window)
        
        # Generate signal based on volatility trend
        # +1 when volatility is decreasing, -1 when increasing
        signal = volatility_change.apply(
            lambda x: 1 if pd.notna(x) and x < -self.threshold else
                    (-1 if pd.notna(x) and x > self.threshold else -1)
        )
        
        # Fill initial NaN values
        signal = signal.fillna(-1)
        
        return signal


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fixed_signal_example.log')
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
    # Create signal instances using our custom classes
    signals = [
        CustomShortTermTrendSignal(params={
            'quote_type': quote_type,
            'trend_values': [1, 2]  # Bullish values
        }),
        CustomRSISignal(params={
            'quote_type': quote_type,
            'rsi_length': 28,
            'signal_mode': 'simple'
        }),
        CustomVolatilityTrendSignal(params={
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


def plot_weighted_signals(asset_id, processed_data, weights, output_file):
    """Plot the weighted signals for an asset."""
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
    logger.info("Starting fixed signal example")
    
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
    
    logger.info("Fixed signal example completed")


if __name__ == "__main__":
    main() 