"""
Signal selector for Soros System.

This module provides the SignalSelector class, which selects and weighs 
signals for each asset based on their effectiveness.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

from ..signals.signal_base import SignalBase
from ..signals.signal_registry import get_signal, get_all_signals
from ..analysis.forward_returns.signal_evaluator import SignalEvaluator
from .signal_combiner import SignalCombiner


class SignalSelector:
    """
    Select and weigh signals for assets based on effectiveness.
    
    This class serves as a bridge between the PortfolioManager and
    the signal evaluation framework, dynamically selecting the best
    signals for each asset.
    """
    
    def __init__(
        self,
        signal_evaluator: Optional[SignalEvaluator] = None,
        signal_combiner: Optional[SignalCombiner] = None,
        min_samples: int = 30,
        lookback_days: Optional[int] = 365,
        recalculate_interval_days: int = 30
    ):
        """Initialize the signal selector.
        
        Args:
            signal_evaluator: Evaluator for signal effectiveness. If None, a new one is created.
            signal_combiner: Combiner for aggregating signals. If None, a new one is created.
            min_samples: Minimum samples required for valid evaluation
            lookback_days: Days to look back for evaluation. None = all data
            recalculate_interval_days: Days between recalculations of weights
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.signal_evaluator = signal_evaluator or SignalEvaluator(
            min_samples=min_samples,
            lookback_days=lookback_days
        )
        
        self.signal_combiner = signal_combiner or SignalCombiner(
            signal_evaluator=self.signal_evaluator,
            auto_weights=True
        )
        
        self.min_samples = min_samples
        self.lookback_days = lookback_days
        self.recalculate_interval_days = recalculate_interval_days
        
        # Cache for weights and last recalculation dates
        self.weights_cache = {}
        self.last_recalculation = {}
        
        self.logger.info(
            f"Initialized SignalSelector with lookback={lookback_days}, "
            f"min_samples={min_samples}, recalc_interval={recalculate_interval_days}"
        )
    
    def select_signals_for_asset(
        self,
        asset_id: str,
        data: pd.DataFrame,
        available_signals: List[SignalBase],
        force_recalculate: bool = False
    ) -> Tuple[List[SignalBase], Dict[str, float]]:
        """Select the best signals for an asset and calculate weights.
        
        Args:
            asset_id: ID of the asset
            data: Price data for the asset
            available_signals: List of available signal instances
            force_recalculate: Whether to force recalculation of weights
            
        Returns:
            tuple: (selected_signals, weights)
                selected_signals: List of selected signal instances
                weights: Dictionary mapping signal names to weights
        """
        self.logger.info(f"Selecting signals for {asset_id} from {len(available_signals)} available signals")
        
        # Check if we need to recalculate weights
        should_recalculate = force_recalculate or self._should_recalculate(asset_id)
        
        if not should_recalculate and asset_id in self.weights_cache:
            # Use cached weights
            cached_weights = self.weights_cache[asset_id]
            self.logger.debug(f"Using cached weights for {asset_id}: {cached_weights}")
            
            # Filter signals based on cached weights
            selected_signals = [
                signal for signal in available_signals 
                if signal.name in cached_weights and cached_weights[signal.name] > 0
            ]
            
            return selected_signals, cached_weights
        
        # We need to evaluate all signals
        try:
            # Evaluate all signals
            evaluations = self.signal_evaluator.evaluate_multiple_signals(
                available_signals, data, asset_id
            )
            
            # Calculate weights
            weights = self.signal_evaluator.calculate_normalized_weights(evaluations)
            
            # Filter signals with positive weights
            selected_signals = [
                signal for signal in available_signals 
                if signal.name in weights and weights[signal.name] > 0
            ]
            
            # Update cache
            self.weights_cache[asset_id] = weights
            self.last_recalculation[asset_id] = datetime.now()
            
            self.logger.info(
                f"Selected {len(selected_signals)} signals for {asset_id} with weights: {weights}"
            )
            
            return selected_signals, weights
            
        except Exception as e:
            self.logger.error(f"Error selecting signals for {asset_id}: {e}")
            
            # Default to using all signals with equal weights
            equal_weights = {signal.name: 1.0 / len(available_signals) for signal in available_signals}
            
            return available_signals, equal_weights
    
    def process_asset(
        self,
        asset_id: str,
        data: pd.DataFrame,
        available_signals: List[SignalBase],
        threshold: float = 0.0,
        force_recalculate: bool = False
    ) -> pd.DataFrame:
        """Process an asset with the optimal signals and weights.
        
        Args:
            asset_id: ID of the asset
            data: Price data for the asset
            available_signals: List of available signal instances
            threshold: Threshold for combined signal to generate a buy signal
            force_recalculate: Whether to force recalculation
            
        Returns:
            pd.DataFrame: Original data with additional columns for signals and decisions
        """
        self.logger.info(f"Processing asset {asset_id}")
        
        try:
            # Select signals and weights
            selected_signals, weights = self.select_signals_for_asset(
                asset_id, data, available_signals, force_recalculate
            )
            
            if not selected_signals:
                self.logger.warning(f"No effective signals found for {asset_id}")
                # Add neutral signal columns and return
                result_df = data.copy()
                result_df['combined_signal'] = 0
                result_df['final_decision'] = 0
                result_df['final_decision_shifted'] = 0
                return result_df
            
            # Set threshold for signal combiner
            self.signal_combiner.set_threshold(threshold)
            
            # Combine signals
            combined_signal, final_decision, used_weights = self.signal_combiner.combine_signals(
                selected_signals, data, asset_id, weights=weights
            )
            
            # Add signal columns to data
            result_df = data.copy()
            result_df['combined_signal'] = combined_signal
            result_df['final_decision'] = final_decision
            
            # Shift final decision to avoid lookahead bias
            result_df['final_decision_shifted'] = final_decision.shift(1)
            
            # Add individual signal columns for transparency
            for signal in selected_signals:
                try:
                    col_name = f"{signal.name}_signal"
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
    
    def _should_recalculate(self, asset_id: str) -> bool:
        """Check if weights should be recalculated for an asset.
        
        Args:
            asset_id: ID of the asset
            
        Returns:
            bool: True if weights should be recalculated
        """
        # If never calculated, we should calculate
        if asset_id not in self.last_recalculation:
            return True
        
        # Check if enough time has passed since last recalculation
        last_time = self.last_recalculation[asset_id]
        time_diff = datetime.now() - last_time
        
        return time_diff.days >= self.recalculate_interval_days
    
    def clear_cache(self, asset_id: Optional[str] = None):
        """Clear cache for a specific asset or all assets.
        
        Args:
            asset_id: ID of the asset to clear cache for. If None, clear all.
        """
        if asset_id is None:
            self.weights_cache = {}
            self.last_recalculation = {}
            self.signal_evaluator.clear_cache()
            self.signal_combiner.clear_cache()
            self.logger.info("Cleared all caches")
        else:
            if asset_id in self.weights_cache:
                del self.weights_cache[asset_id]
            if asset_id in self.last_recalculation:
                del self.last_recalculation[asset_id]
            self.logger.info(f"Cleared cache for {asset_id}")
    
    # Placeholder for future meta-labeling integration
    def apply_meta_labeling(self, asset_id: str, data: pd.DataFrame) -> pd.DataFrame:
        """Apply meta-labeling to improve signal quality.
        
        Args:
            asset_id: ID of the asset
            data: DataFrame with signals and decisions
            
        Returns:
            pd.DataFrame: DataFrame with meta-labeled decisions
        """
        # TODO: Implement meta-labeling
        self.logger.info(f"Meta-labeling not yet implemented for {asset_id}")
        
        # For now, just return the original data
        return data 