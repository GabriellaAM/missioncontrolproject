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
from ..analysis.forward_returns.signal_evaluator import SignalEvaluator


class SignalCombiner:
    """
    Combine multiple signals using calculated weights.
    
    This class aggregates signals from multiple sources and computes
    a combined signal value based on their weights.
    """
    
    def __init__(
        self, 
        signal_evaluator: Optional[SignalEvaluator] = None,
        threshold: float = 0.0,
        auto_weights: bool = True
    ):
        """Initialize the signal combiner.
        
        Args:
            signal_evaluator: Evaluator for calculating signal weights. If None, a new one is created.
            threshold: Threshold for final decision (default: 0.0)
            auto_weights: Whether to automatically calculate weights (default: True)
        """
        self.logger = logging.getLogger(__name__)
        self.signal_evaluator = signal_evaluator or SignalEvaluator()
        self.threshold = threshold
        self.auto_weights = auto_weights
        
        # Cache for weights and calculated signals
        self.weights_cache = {}
        self.combined_signal_cache = {}
    
    def combine_signals(
        self, 
        signals: List[SignalBase], 
        data: pd.DataFrame, 
        asset_id: str,
        weights: Optional[Dict[str, float]] = None,
        recalculate_weights: bool = False
    ) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
        """Combine multiple signals into a single signal.
        
        Args:
            signals: List of signal instances to combine
            data: Price data for the asset
            asset_id: ID of the asset
            weights: Dictionary mapping signal names to weights (optional)
            recalculate_weights: Whether to recalculate weights even if cached
            
        Returns:
            tuple: (combined_signal, final_decision, weights)
                combined_signal: Series with combined signal values (weighted average)
                final_decision: Series with binary decision values (0 or 1)
                weights: Dictionary mapping signal names to weights used
        """
        self.logger.info(f"Combining {len(signals)} signals for {asset_id}")
        
        try:
            # Check if we need to calculate weights
            if weights is None and self.auto_weights:
                # Check cache first if not forced to recalculate
                cache_key = f"{asset_id}_{'-'.join(sorted([s.name for s in signals]))}"
                
                if not recalculate_weights and cache_key in self.weights_cache:
                    weights = self.weights_cache[cache_key]
                    self.logger.debug(f"Using cached weights for {asset_id}")
                else:
                    # Calculate weights
                    weights = self._calculate_weights(signals, data, asset_id)
                    
                    # Cache weights
                    self.weights_cache[cache_key] = weights
            elif weights is None:
                # If auto_weights is False and no weights provided, use equal weights
                weights = {signal.name: 1.0 / len(signals) for signal in signals}
            
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
    
    def _calculate_weights(
        self, signals: List[SignalBase], data: pd.DataFrame, asset_id: str
    ) -> Dict[str, float]:
        """Calculate weights for signals based on their effectiveness.
        
        Args:
            signals: List of signal instances
            data: Price data for the asset
            asset_id: ID of the asset
            
        Returns:
            dict: Dictionary mapping signal names to weights
        """
        # Evaluate each signal
        evaluations = self.signal_evaluator.evaluate_multiple_signals(
            signals, data, asset_id
        )
        
        # Calculate normalized weights
        weights = self.signal_evaluator.calculate_normalized_weights(evaluations)
        
        # Log weights
        self.logger.info(f"Calculated weights for {asset_id}: {weights}")
        
        return weights
    
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
        self.weights_cache = {}
        self.combined_signal_cache = {}
        if self.signal_evaluator:
            self.signal_evaluator.clear_cache()
        self.logger.info("Cleared all caches") 