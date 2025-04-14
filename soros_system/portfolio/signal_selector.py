"""
Signal selector for Soros System.

This module provides the SignalSelector class, which selects signals 
for each asset with manually configured weights.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

from ..signals.signal_base import SignalBase
from ..signals.signal_registry import get_signal, get_all_signals
from .signal_combiner import SignalCombiner


class SignalSelector:
    """
    Select signals for assets with manually configured weights.
    
    This class serves as a bridge between the PortfolioManager and
    the signal calculation framework, applying user-configured weights.
    """
    
    def __init__(
        self,
        signal_combiner: Optional[SignalCombiner] = None,
        default_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the signal selector.
        
        Args:
            signal_combiner: Combiner for aggregating signals. If None, a new one is created.
            default_weights: Default weights for signals (if not specified elsewhere)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.signal_combiner = signal_combiner or SignalCombiner(auto_weights=False)
        
        # Set default weights or use empty dict
        self.default_weights = default_weights or {}
        
        if self.default_weights:
            self.signal_combiner.set_default_weights(self.default_weights)
            self.logger.info(f"Set default weights for {len(self.default_weights)} signals")
        
    def set_signal_weights(self, weights: Dict[str, float]) -> None:
        """Set weights for signals.
        
        Args:
            weights: Dictionary mapping signal names to weights
        """
        self.default_weights.update(weights)
        self.signal_combiner.set_default_weights(self.default_weights)
        self.logger.info(f"Updated weights for {len(weights)} signals")
    
    def select_signals_for_asset(
        self,
        asset_id: str,
        data: pd.DataFrame,
        available_signals: List[SignalBase],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[List[SignalBase], Dict[str, float]]:
        """Select signals for an asset using configured weights.
        
        Args:
            asset_id: ID of the asset
            data: Price data for the asset
            available_signals: List of available signal instances
            weights: Dictionary mapping signal names to weights (overrides default_weights)
            
        Returns:
            tuple: (selected_signals, weights)
                selected_signals: List of selected signal instances
                weights: Dictionary mapping signal names to weights
        """
        self.logger.info(f"Selecting signals for {asset_id} from {len(available_signals)} available signals")
        
        # Use provided weights or default weights
        signal_weights = weights or self.default_weights
        
        # If no weights defined, use equal weights
        if not signal_weights:
            signal_weights = {signal.name: 1.0 / len(available_signals) for signal in available_signals}
            self.logger.info(f"Using equal weights for {asset_id}: {signal_weights}")
        
        # Filter signals to only those with defined weights
        selected_signals = [
            signal for signal in available_signals 
            if signal.name in signal_weights and signal_weights[signal.name] != 0
        ]
        
        if not selected_signals:
            self.logger.warning(f"No signals with weights defined for {asset_id}")
            # Use all signals with equal weights as fallback
            selected_signals = available_signals
            signal_weights = {signal.name: 1.0 / len(available_signals) for signal in available_signals}
            
        return selected_signals, signal_weights
    
    def process_asset(
        self,
        asset_id: str,
        data: pd.DataFrame,
        available_signals: List[SignalBase],
        threshold: float = 0.0,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Process an asset with signals and weights.
        
        Args:
            asset_id: ID of the asset
            data: Price data for the asset
            available_signals: List of available signal instances
            threshold: Threshold for combined signal to generate a buy signal
            weights: Custom weights to use (overrides default_weights)
            
        Returns:
            pd.DataFrame: Original data with additional columns for signals and decisions
        """
        self.logger.info(f"Processing asset {asset_id}")
        
        try:
            # Select signals and weights
            selected_signals, signal_weights = self.select_signals_for_asset(
                asset_id, data, available_signals, weights
            )
            
            if not selected_signals:
                self.logger.warning(f"No signals selected for {asset_id}")
                # Add neutral signal columns and return
                result_df = data.copy()
                result_df['combined_signal'] = 0
                result_df['final_decision'] = 0
                result_df['final_decision_shifted'] = 0
                return result_df
            
            # Set threshold for signal combiner
            self.signal_combiner.set_threshold(threshold)
            
            # Combine signals using binary logic (no event tracking)
            combined_signal, final_decision, used_weights = self.signal_combiner.combine_signals(
                selected_signals, data, asset_id, weights=signal_weights
            )
            
            # Add signal columns to data
            result_df = data.copy()
            result_df['combined_signal'] = combined_signal
            result_df['final_decision'] = final_decision
            
            # Shift final decision to avoid lookahead bias (for execution on next period)
            result_df['final_decision_shifted'] = final_decision.shift(1)
            
            # Add individual signal columns for transparency
            for signal in selected_signals:
                try:
                    col_name = f"{signal.name}_signal"
                    # Always include asset_id when calculating the signal
                    signal_values = signal.calculate(data, asset_id)
                    result_df[col_name] = signal_values
                except Exception as e:
                    self.logger.error(f"Error calculating {signal.name} for {asset_id}: {e}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error processing asset {asset_id}: {e}")
            
            # Return original data with neutral signals
            result_df = data.copy()
            result_df['combined_signal'] = 0
            result_df['final_decision'] = 0
            result_df['final_decision_shifted'] = 0
            
            return result_df
    
    def clear_cache(self, asset_id: Optional[str] = None):
        """Clear any caches.
        
        Args:
            asset_id: ID of the asset to clear cache for (unused in simplified implementation)
        """
        if self.signal_combiner:
            self.signal_combiner.clear_cache()
        self.logger.info("Cleared signal combiner cache") 