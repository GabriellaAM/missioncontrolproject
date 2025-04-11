"""
Signal evaluator for Soros System.

This module evaluates signal effectiveness and calculates weights
based on forward return distributions.
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
            
            # Calculate statistics for each period
            effectiveness_by_period = {}
            distribution_stats = {}
            
            for period, returns in returns_by_period.items():
                # Get statistics for signal=1 vs signal=0
                period_stats = self._calculate_signal_effectiveness(signal_values, returns)
                effectiveness_by_period[period] = period_stats
                
                # Get return distributions for different signal states
                dist_stats = self._calculate_return_distributions(signal_values, returns)
                distribution_stats[period] = dist_stats
            
            # Determine overall effectiveness and weight
            overall_effective, weight, metrics = self._calculate_overall_effectiveness(
                effectiveness_by_period, distribution_stats
            )
            
            # Only calculate optimal decay for effective signals
            if overall_effective:
                # Signal is effective, determine optimal decay
                optimal_period, sortino = self.determine_optimal_decay(
                    effectiveness_by_period, distribution_stats
                )
            else:
                # Signal is not effective, set optimal decay to 0
                optimal_period, sortino = 0, 0.0
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
                'sortino_ratio': sortino
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
    
    def _calculate_signal_effectiveness(
        self, signal_values: pd.Series, returns: pd.Series
    ) -> Dict[str, Any]:
        """Calculate the effectiveness of a signal by comparing return distributions.
        
        Args:
            signal_values: Binary signal values (1 for active, 0 for inactive)
            returns: Forward returns series
            
        Returns:
            dict: Effectiveness metrics
        """
        # Align signal values and returns
        aligned_data = pd.concat([signal_values, returns], axis=1).dropna()
        
        if len(aligned_data) < self.min_samples:
            return {
                'effectiveness': {
                    'overall_effective': False,
                    'effect_size': 0.0,
                    'confidence': 0.0,
                    'direction': 0,
                    'reason': f"Insufficient samples: {len(aligned_data)} < {self.min_samples}"
                }
            }
        
        # Separate returns by signal value
        signal_col = aligned_data.columns[0]
        returns_col = aligned_data.columns[1]
        
        signal_1_returns = aligned_data[aligned_data[signal_col] == 1][returns_col]
        signal_0_returns = aligned_data[aligned_data[signal_col] == 0][returns_col]
        
        # Check if we have enough samples in each group
        if len(signal_1_returns) < self.min_samples / 2 or len(signal_0_returns) < self.min_samples / 2:
            return {
                'effectiveness': {
                    'overall_effective': False,
                    'effect_size': 0.0,
                    'confidence': 0.0,
                    'direction': 0,
                    'reason': f"Imbalanced samples: signal=1 ({len(signal_1_returns)}) vs signal=0 ({len(signal_0_returns)})"
                }
            }
        
        # Calculate statistics
        mean_1 = signal_1_returns.mean()
        mean_0 = signal_0_returns.mean()
        mean_diff = mean_1 - mean_0
        
        # Perform t-test to determine if the difference is significant
        t_stat, p_value = ttest_ind(signal_1_returns, signal_0_returns, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(signal_1_returns) - 1) * signal_1_returns.std() ** 2 + 
             (len(signal_0_returns) - 1) * signal_0_returns.std() ** 2) / 
            (len(signal_1_returns) + len(signal_0_returns) - 2)
        )
        
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Determine if the signal is effective
        is_significant = p_value < self.alpha
        
        # Determine direction of effect (positive or negative)
        direction = 1 if mean_diff > 0 else -1 if mean_diff < 0 else 0
        
        # Calculate confidence level (1 - p_value)
        confidence = 1 - p_value
        
        # Result: signal is effective if it has a significant effect
        is_effective = is_significant and abs(effect_size) >= 0.1
        
        return {
            'statistics': {
                'signal_1_mean': mean_1,
                'signal_0_mean': mean_0,
                'mean_diff': mean_diff,
                'signal_1_std': signal_1_returns.std(),
                'signal_0_std': signal_0_returns.std(),
                'signal_1_count': len(signal_1_returns),
                'signal_0_count': len(signal_0_returns),
                't_statistic': t_stat,
                'p_value': p_value
            },
            'effectiveness': {
                'overall_effective': is_effective,
                'effect_size': effect_size,
                'confidence': confidence,
                'direction': direction,
                'reason': "Significant effect" if is_effective else 
                         "Non-significant effect" if not is_significant else
                         "Effect size too small"
            }
        }
        
    def _calculate_return_distributions(
        self, signal_values: pd.Series, returns: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Calculate return distributions for different signal states.
        
        Args:
            signal_values: Binary signal values
            returns: Forward returns series
            
        Returns:
            dict: Return distributions for different signal states
        """
        # Align signal values and returns
        aligned_data = pd.concat([signal_values, returns], axis=1).dropna()
        
        if len(aligned_data) < self.min_samples:
            return {
                'positive': {'mean': 0.0, 'std': 0.0, 'returns': []},
                'negative': {'mean': 0.0, 'std': 0.0, 'returns': []}
            }
        
        # Extract columns
        signal_col = aligned_data.columns[0]
        returns_col = aligned_data.columns[1]
        
        # Get returns for different signal states
        positive_returns = aligned_data[aligned_data[signal_col] == 1][returns_col]
        negative_returns = aligned_data[aligned_data[signal_col] == 0][returns_col]
        
        # Calculate statistics
        return {
            'positive': {
                'mean': positive_returns.mean() if not positive_returns.empty else 0.0,
                'std': positive_returns.std() if not positive_returns.empty else 0.0,
                'returns': positive_returns.tolist() if not positive_returns.empty else []
            },
            'negative': {
                'mean': negative_returns.mean() if not negative_returns.empty else 0.0,
                'std': negative_returns.std() if not negative_returns.empty else 0.0,
                'returns': negative_returns.tolist() if not negative_returns.empty else []
            }
        }
    
    def _calculate_overall_effectiveness(
        self, effectiveness_by_period: Dict[int, Dict[str, Any]], 
        distribution_stats: Dict[int, Dict[str, Dict[str, float]]]
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Calculate overall effectiveness and weight for a signal.
        
        Args:
            effectiveness_by_period: Dictionary mapping periods to effectiveness evaluations
            distribution_stats: Dictionary mapping periods to distribution statistics
            
        Returns:
            tuple: (overall_effectiveness, weight, metrics)
                overall_effectiveness (bool): Whether the signal is effective overall
                weight (float): Weight for the signal
                metrics (dict): Metrics used for weight calculation
        """
        if not effectiveness_by_period:
            return False, 0.0, {}
        
        # Count effective periods
        effective_periods = 0
        total_periods = len(effectiveness_by_period)
        
        # Collect metrics for weight calculation
        effect_sizes = []
        confidences = []
        mean_diffs = []
        directions = []  # Track direction of each period
        
        for period, period_data in effectiveness_by_period.items():
            effectiveness = period_data.get('effectiveness', {})
            
            # Check if effective for this period
            if effectiveness.get('overall_effective', False):
                effective_periods += 1
                
                # Collect metrics
                effect_sizes.append(abs(effectiveness.get('effect_size', 0.0)))
                confidences.append(effectiveness.get('confidence', 0.0))
                directions.append(effectiveness.get('direction', 0))
                
                # Get mean difference from stats
                if (period in distribution_stats and 
                    'positive' in distribution_stats[period] and 
                    'negative' in distribution_stats[period]):
                    pos_mean = distribution_stats[period]['positive'].get('mean', 0.0)
                    neg_mean = distribution_stats[period]['negative'].get('mean', 0.0)
                    mean_diffs.append(pos_mean - neg_mean)
        
        # Determine overall effectiveness (at least 25% of periods must be effective)
        ratio_effective = effective_periods / total_periods if total_periods > 0 else 0
        overall_effective = ratio_effective >= 0.25 and effective_periods > 0
        
        # Calculate weight using weighted combination of metrics
        if overall_effective and effect_sizes:
            # Calculate mean metrics
            avg_effect_size = np.mean(effect_sizes)
            avg_confidence = np.mean(confidences)
            
            # Determine dominant direction (-1, 0, or 1)
            neg_count = sum(1 for d in directions if d < 0)
            pos_count = sum(1 for d in directions if d > 0)
            dominant_direction = 1 if pos_count > neg_count else -1 if neg_count > pos_count else 0
            
            # Calculate weight (sign indicates direction)
            # Weight magnitude is based on effect size and confidence
            weight_magnitude = avg_effect_size * avg_confidence
            weight = weight_magnitude * dominant_direction
            
            metrics = {
                'avg_effect_size': avg_effect_size,
                'avg_confidence': avg_confidence,
                'dominant_direction': dominant_direction,
                'effective_periods': effective_periods,
                'total_periods': total_periods,
                'ratio_effective': ratio_effective
            }
            
            return overall_effective, weight, metrics
            
        return overall_effective, 0.0, {
            'effective_periods': effective_periods,
            'total_periods': total_periods,
            'ratio_effective': ratio_effective,
            'reason': "No effect sizes available" if not effect_sizes else "Not effective overall"
        }
    
    def evaluate_multiple_signals(
        self, signals: List[SignalBase], data: pd.DataFrame, asset_id: str, price_col: str = 'close'
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple signals for a specific asset.
        
        Args:
            signals: List of signal instances to evaluate
            data: Price data for the asset
            asset_id: ID of the asset
            price_col: Column to use for price data
            
        Returns:
            dict: Dictionary mapping signal names to evaluation results
        """
        results = {}
        
        for signal in signals:
            try:
                evaluation = self.evaluate_signal(signal, data, asset_id, price_col)
                results[signal.name] = evaluation
            except Exception as e:
                self.logger.error(f"Error evaluating signal {signal.name} for {asset_id}: {e}")
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
        """Calculate normalized weights for signals based on evaluations.
        
        Args:
            evaluations: Dictionary mapping signal names to evaluation results
            min_weight: Minimum weight to assign to a valid signal (default: 0.0)
            
        Returns:
            dict: Dictionary mapping signal names to normalized weights
        """
        if not evaluations:
            return {}
        
        # Extract weights and ensure validity
        weights = {}
        for signal_name, evaluation in evaluations.items():
            if evaluation.get('valid', False) and evaluation.get('overall_effectiveness', False):
                weights[signal_name] = max(min_weight, evaluation.get('weight', 0.0))
            else:
                weights[signal_name] = 0.0
        
        # Check if we have any positive weights
        if not weights or sum(weights.values()) == 0:
            self.logger.warning("No effective signals found, using equal weights for all valid signals")
            
            # Assign equal weights to all valid signals
            equal_weights = {}
            for signal_name, evaluation in evaluations.items():
                if evaluation.get('valid', False):
                    equal_weights[signal_name] = 1.0
                else:
                    equal_weights[signal_name] = 0.0
            
            # Normalize equal weights
            if sum(equal_weights.values()) > 0:
                equal_weights = self._normalize_weights(equal_weights)
            
            return equal_weights
        
        # Normalize the weights
        normalized_weights = self._normalize_weights(weights)
        
        return normalized_weights
    
    def determine_optimal_decay(
        self, 
        effectiveness_by_period: Dict[int, Dict[str, Any]], 
        distribution_stats: Dict[int, Dict[str, Dict[str, float]]],
        candidate_periods: List[int] = None
    ) -> Tuple[int, float]:
        """Determine the optimal decay period for a signal based on Sortino ratio.
        
        Args:
            effectiveness_by_period: Dict mapping periods to effectiveness metrics
            distribution_stats: Dict mapping periods to distribution statistics
            candidate_periods: List of periods to consider (defaults to all effective periods)
            
        Returns:
            tuple: (optimal_period, sortino_ratio)
        """
        # If candidate periods not specified, use all periods
        if candidate_periods is None:
            candidate_periods = self.periods
            
        # Filter to effective periods
        effective_periods = []
        for period in candidate_periods:
            if period in effectiveness_by_period:
                try:
                    # Check if this period has an effective signal
                    if effectiveness_by_period[period].get('effectiveness', {}).get('overall_effective', False):
                        effective_periods.append(period)
                except (KeyError, TypeError):
                    continue
        
        if not effective_periods:
            # No effective periods found
            self.logger.warning("No effective periods found for decay calculation")
            return 0, 0.0
            
        # Calculate Sortino ratio for each effective period
        sortino_by_period = {}
        for period in effective_periods:
            try:
                # Get distribution stats for signal=1
                dist = distribution_stats[period].get('signal_1', {})
                
                # Calculate downside deviation (using negative returns only)
                mean = dist.get('mean', 0)
                neg_returns = dist.get('neg_returns', [])
                
                if not neg_returns or mean <= 0:
                    sortino = 0.0
                else:
                    downside_deviation = np.sqrt(np.mean(np.square(neg_returns)))
                    sortino = mean / downside_deviation if downside_deviation > 0 else 0.0
                
                sortino_by_period[period] = sortino
                
            except (KeyError, TypeError, ZeroDivisionError) as e:
                self.logger.warning(f"Error calculating Sortino for period {period}: {e}")
                sortino_by_period[period] = 0.0
                
        # Find period with highest Sortino ratio
        if not sortino_by_period:
            # No valid Sortino ratios
            return 0, 0.0
            
        optimal_period = max(sortino_by_period.items(), key=lambda x: x[1])[0]
        optimal_sortino = sortino_by_period[optimal_period]
        
        self.logger.info(f"Optimal decay: {optimal_period} days (Sortino: {optimal_sortino:.4f})")
        
        return optimal_period, optimal_sortino
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0.
        
        Args:
            weights: Dictionary mapping signal names to weights
            
        Returns:
            dict: Dictionary mapping signal names to normalized weights
        """
        total_weight = sum(weights.values())
        
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