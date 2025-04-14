"""
Signal combiner for the Soros System.

This module provides tools for combining multiple signals using weights.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

from ..signals.signal_base import SignalBase


class SignalCombiner:
    """
    Combine multiple signals using provided weights.
    
    This class aggregates signals from multiple sources and computes
    a combined signal value based on their weights.
    """
    
    def __init__(
        self, 
        threshold: float = 0.0,
        auto_weights: bool = False
    ):
        """Initialize the signal combiner.
        
        Args:
            threshold: Threshold for final decision (default: 0.0)
            auto_weights: Whether to automatically calculate weights (default: False)
        """
        self.logger = logging.getLogger(__name__)
        self.threshold = threshold
        self.auto_weights = auto_weights
        
        # Cache for calculated signals
        self.combined_signal_cache = {}
        
        # Default weights (equal weighting)
        self.default_weights = {}
    
    def set_default_weights(self, weights: Dict[str, float]) -> None:
        """Set default weights for signals.
        
        Args:
            weights: Dictionary mapping signal names to weights
        """
        self.default_weights = weights
        self.logger.info(f"Set default weights for {len(weights)} signals")
    
    def combine_signals(
        self, 
        signals: List[SignalBase], 
        data: pd.DataFrame, 
        asset_id: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
        """Combine multiple signals into a single signal.
        
        Args:
            signals: List of signal instances to combine
            data: Price data for the asset
            asset_id: ID of the asset
            weights: Dictionary mapping signal names to weights (optional)
            
        Returns:
            tuple: (combined_signal, final_decision, weights)
                combined_signal: Series with combined signal values (weighted average)
                final_decision: Series with binary decision values (0 or 1)
                weights: Dictionary mapping signal names to weights used
        """
        self.logger.info(f"Combining {len(signals)} signals for {asset_id}")
        
        try:
            # Use provided weights, default weights, or equal weights
            if weights is None:
                # Check if we have default weights for all signals
                signal_names = [s.name for s in signals]
                if all(name in self.default_weights for name in signal_names):
                    weights = {name: self.default_weights[name] for name in signal_names}
                    self.logger.debug(f"Using default weights for {asset_id}")
                else:
                    # Use equal weights
                    weights = {signal.name: 1.0 / len(signals) for signal in signals}
                    self.logger.debug(f"Using equal weights for {asset_id}")
            
            # Calculate individual signal values
            signal_values = {}
            for signal in signals:
                try:
                    # Calculate signal for all weights, regardless of sign
                    # We need negative weights for signals that predict negative returns
                    if signal.name in weights and weights[signal.name] != 0:
                        signal_series = signal.calculate(data, asset_id)
                        signal_values[signal.name] = signal_series
                except Exception as e:
                    self.logger.error(f"Error calculating signal {signal.name} for {asset_id}: {e}")
            
            # Create a DataFrame with all signal values aligned by date
            signal_df = pd.DataFrame(signal_values)
            
            # Check if we have any signals
            if signal_df.empty:
                self.logger.warning(f"No valid signals calculated for {asset_id}")
                # Return empty series with proper index
                return pd.Series(index=data.index), pd.Series(index=data.index), weights
            
            # Calculate combined signal for each date
            combined_signal = pd.Series(index=signal_df.index)
            
            # Separate positive and negative weights
            pos_weights = {k: v for k, v in weights.items() if v > 0}
            neg_weights = {k: v for k, v in weights.items() if v < 0}
            
            for date in signal_df.index:
                row = signal_df.loc[date]
                
                # Get valid signals for this date
                valid_signals = {}
                for name, value in row.items():
                    if pd.notna(value) and name in weights and weights[name] != 0:
                        # For signals with negative weights, invert the signal value
                        # If signal is 1 and weight is negative, we want it to reduce combined score
                        if weights[name] < 0:
                            # Use 1-value to invert binary signal (1->0, 0->1)
                            # Then multiply by the negative weight to make it positive
                            valid_signals[name] = (1-value) * abs(weights[name])
                        else:
                            valid_signals[name] = value * weights[name]
                
                # If no valid signals for this date, set to NaN
                if not valid_signals:
                    combined_signal.loc[date] = np.nan
                    continue
                
                # Calculate weighted sum
                weighted_sum = sum(valid_signals.values())
                
                # Calculate total weight used (absolute values)
                total_weight = sum(abs(weights[name]) for name in valid_signals.keys())
                
                # Normalize by total weight used
                if total_weight > 0:
                    combined_signal.loc[date] = weighted_sum / total_weight
                else:
                    combined_signal.loc[date] = np.nan
            
            # Calculate final decision
            final_decision = combined_signal.apply(
                lambda x: 1 if pd.notna(x) and x > self.threshold else 0
            )
            
            return combined_signal, final_decision, weights
        
        except Exception as e:
            self.logger.error(f"Error combining signals for {asset_id}: {e}")
            # Return empty series with proper index
            return pd.Series(index=data.index), pd.Series(index=data.index), {}
    
    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for final decision.
        
        Args:
            threshold: New threshold value
        """
        self.threshold = threshold
        self.logger.info(f"Set threshold to {threshold}")
        # Clear combined signal cache as threshold changed
        self.combined_signal_cache = {}
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self.combined_signal_cache = {}
        self.logger.info("Cleared cache") 