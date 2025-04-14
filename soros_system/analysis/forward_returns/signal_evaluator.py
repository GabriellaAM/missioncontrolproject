"""
Signal evaluator for Soros System.

This module evaluates signal effectiveness and calculates weights
based on forward returns analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import json
from datetime import datetime
from scipy.stats import ttest_ind

from ..forward_returns.calculator import ForwardReturnsCalculator
from ..forward_returns.statistical_tests import StatisticalTester
from ...signals.signal_base import SignalBase


class SignalEvaluator:
    """
    Evaluate signal effectiveness using forward returns analysis.
    
    This class calculates weights for signals based on their effectiveness
    in shifting forward return distributions.
    """
    
    def __init__(
        self, 
        periods: List[int] = None,
        min_samples: int = 30,
        alpha: float = 0.05,
        lookback_days: Optional[int] = None
    ):
        """Initialize the signal evaluator.
        
        Args:
            periods (list): List of forward periods (in days) to evaluate. Default: [1, 3, 5, 7, 14, 21, 28]
            min_samples (int): Minimum number of samples required for statistical tests. Default: 30
            alpha (float): Significance level for statistical tests. Default: 0.05
            lookback_days (int, optional): Number of days to look back for evaluation. If None, use all available data.
        """
        self.logger = logging.getLogger(__name__)
        self.periods = periods or [1, 3, 5, 7, 14, 21, 28]
        self.min_samples = min_samples
        self.alpha = alpha
        self.lookback_days = lookback_days
        
        # Initialize components
        self.calculator = ForwardReturnsCalculator(periods=self.periods)
        self.tester = StatisticalTester(alpha=self.alpha)
        
        # Cache for evaluations
        self.evaluations_cache = {}
    
    def evaluate_signal(
        self, signal: SignalBase, data: pd.DataFrame, asset_id: str, price_col: str = 'close'
    ) -> Dict[str, Any]:
        """Evaluate a signal for a specific asset.
        
        This evaluates the effectiveness of a signal by comparing return distributions
        when the signal is active (1) vs. inactive (0).
        
        Args:
            signal: Signal instance to evaluate
            data: Price data for the asset
            asset_id: ID of the asset
            price_col: Column to use for price data
            
        Returns:
            dict: Evaluation results with metrics like effectiveness, weight, etc.
        """
        self.logger.info(f"Evaluating signal {signal.name} for {asset_id}")
        
        try:
            # Calculate signal values
            signal_values = self._get_signal_values(signal, data, asset_id)
            
            if signal_values is None or signal_values.empty:
                self.logger.error(f"Failed to get values for {signal.name} on {asset_id}")
                return {
                    'signal_name': signal.name,
                    'asset_id': asset_id,
                    'valid': False,
                    'error': "No signal values available",
                    'weight': 0.0,
                    'optimal_decay': 0
                }
            
            # Calculate forward returns for different periods
            returns_by_period = {}
            for period in self.periods:
                returns = self._calculate_forward_returns(data, period, price_col)
                returns_by_period[period] = returns
            
            # Prepare data for new effectiveness evaluation
            period_returns_data = {}
            for period, returns in returns_by_period.items():
                # Align signal values and returns
                aligned_data = pd.concat([signal_values, returns], axis=1).dropna()
                
                if len(aligned_data) < self.min_samples:
                    continue
                
                # Separate returns by signal value
                signal_col = aligned_data.columns[0]
                returns_col = aligned_data.columns[1]
                
                signal_1_returns = aligned_data[aligned_data[signal_col] == 1][returns_col]
                signal_0_returns = aligned_data[aligned_data[signal_col] == 0][returns_col]
                
                # Check if we have enough samples in each group
                if len(signal_1_returns) < self.min_samples / 2 or len(signal_0_returns) < self.min_samples / 2:
                    continue
                
                # Store returns for this period
                period_returns_data[period] = (signal_1_returns, signal_0_returns)
            
            # Evaluate effectiveness using the existing tester instance
            evaluation_results = self.tester.evaluate_signal_effectiveness_new(period_returns_data)
            
            # Extract key metrics
            overall_effective = evaluation_results['overall_effective']
            metrics = evaluation_results['metrics']
            
            if overall_effective:
                # Signal is effective, get optimal period and weight
                optimal_period = metrics['optimal_period']
                weight = metrics['weight']
                
                # For compatibility with Sortino ratio used elsewhere
                sortino = abs(weight)  # Using absolute weight as a proxy for Sortino
            else:
                # Signal is not effective
                optimal_period = 0
                weight = 0.0
                sortino = 0.0
                self.logger.info(f"Signal {signal.name} for {asset_id} is not effective, setting optimal decay to 0")
            
            # Compile final evaluation
            evaluation = {
                'signal_name': signal.name,
                'asset_id': asset_id,
                'valid': True,
                'metrics': metrics,
                'overall_effectiveness': overall_effective,
                'weight': weight,
                'optimal_decay': optimal_period,
                'sortino_ratio': sortino,
                'statistical_results': evaluation_results
            }
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating signal {signal.name} for {asset_id}: {e}")
            return {
                'signal_name': signal.name,
                'asset_id': asset_id,
                'valid': False,
                'error': str(e),
                'weight': 0.0,
                'optimal_decay': 0
            }
    
    def evaluate_multiple_signals(
        self, signals: List[SignalBase], data: pd.DataFrame, asset_id: str, price_col: str = 'close'
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple signals for a specific asset.
        
        Args:
            signals: List of signals to evaluate
            data: Price data for the asset
            asset_id: ID of the asset
            price_col: Column to use for price data
            
        Returns:
            dict: Dictionary mapping signal names to evaluation results
        """
        results = {}
        
        for signal in signals:
            try:
                # Check cache first
                cache_key = f"{signal.name}_{asset_id}"
                if cache_key in self.evaluations_cache:
                    self.logger.info(f"Using cached evaluation for {signal.name} on {asset_id}")
                    results[signal.name] = self.evaluations_cache[cache_key]
                    continue
                
                # Evaluate the signal
                evaluation = self.evaluate_signal(signal, data, asset_id, price_col)
                
                # Store in cache
                self.evaluations_cache[cache_key] = evaluation
                
                # Add to results
                results[signal.name] = evaluation
                
            except Exception as e:
                self.logger.error(f"Error evaluating signal {signal.name}: {e}")
                results[signal.name] = {
                    'signal_name': signal.name,
                    'asset_id': asset_id,
                    'valid': False,
                    'error': str(e),
                    'weight': 0.0,
                    'optimal_decay': 0
                }
        
        return results
    
    def calculate_normalized_weights(
        self, evaluations: Dict[str, Dict[str, Any]], min_weight: float = 0.0
    ) -> Dict[str, float]:
        """Calculate normalized weights for a set of signals.
        
        Args:
            evaluations: Dictionary mapping signal names to evaluation results
            min_weight: Minimum weight threshold (signals with weight below this are excluded)
            
        Returns:
            dict: Dictionary mapping signal names to normalized weights
        """
        # Extract weights
        weights = {}
        for signal_name, evaluation in evaluations.items():
            # Skip invalid evaluations
            if not evaluation.get('valid', False):
                continue
                
            # Skip ineffective signals
            if not evaluation.get('overall_effectiveness', False):
                continue
                
            # Get weight
            weight = evaluation.get('weight', 0.0)
            
            # Apply minimum weight threshold
            if abs(weight) < min_weight:
                continue
                
            weights[signal_name] = weight
            
        # If no valid weights, return empty dict
        if not weights:
            return {}
            
        # Normalize weights
        normalized_weights = self._normalize_weights(weights)
        
        return normalized_weights
        
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0.
        
        Args:
            weights: Dictionary mapping signal names to weights
            
        Returns:
            dict: Dictionary mapping signal names to normalized weights
        """
        total_weight = sum(abs(w) for w in weights.values())
        
        if total_weight == 0:
            return weights
        
        return {name: weight / total_weight for name, weight in weights.items()}
    
    def _get_signal_values(self, signal: SignalBase, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Get signal values for evaluation.
        
        Args:
            signal: Signal instance
            data: Price data
            asset_id: Asset ID
            
        Returns:
            pd.Series: Signal values (1 for active, 0 for inactive)
        """
        try:
            # Apply lookback filter if specified
            if self.lookback_days is not None:
                if isinstance(data.index, pd.DatetimeIndex):
                    cutoff_date = data.index[-1] - pd.Timedelta(days=self.lookback_days)
                    filtered_data = data[data.index >= cutoff_date]
                else:
                    # If no DatetimeIndex, use the most recent lookback_days rows
                    filtered_data = data.iloc[max(0, len(data) - self.lookback_days):]
            else:
                filtered_data = data
            
            # Calculate signal values
            signal_values = signal.calculate(filtered_data, asset_id)
            
            # Ensure signal values are binary (0 or 1)
            if not all(val in [0, 1] for val in signal_values.unique()):
                self.logger.warning(f"Signal {signal.name} contains non-binary values, normalizing")
                signal_values = (signal_values > 0).astype(int)
                
            return signal_values
        
        except Exception as e:
            self.logger.error(f"Error calculating signal values for {signal.name} on {asset_id}: {e}")
            return pd.Series()
    
    def _calculate_forward_returns(self, data: pd.DataFrame, period: int, price_col: str = 'close') -> pd.Series:
        """Calculate forward returns for a specific period.
        
        Args:
            data: Price data
            period: Number of days to look forward
            price_col: Column to use for price data
            
        Returns:
            pd.Series: Forward returns
        """
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                # Convert to DatetimeIndex if not already
                if 'date' in data.columns:
                    data = data.set_index('date')
                else:
                    # Cannot calculate returns without date index
                    self.logger.error("Cannot calculate forward returns: No date index available")
                    return pd.Series()
            
            # Calculate price changes
            returns = data[price_col].pct_change(period).shift(-period)
            
            # Label returns with the date where the signal would be generated
            return returns
            
        except Exception as e:
            self.logger.error(f"Error calculating forward returns for period {period}: {e}")
            return pd.Series()
    
    def clear_cache(self):
        """Clear the evaluations cache."""
        self.evaluations_cache = {} 