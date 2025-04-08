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
        """Evaluate the effectiveness of a signal for a specific asset.
        
        Args:
            signal: Signal instance to evaluate
            data: Price data for the asset
            asset_id: ID of the asset
            price_col: Column to use for price data
            
        Returns:
            dict: Evaluation results
        """
        self.logger.info(f"Evaluating signal {signal.name} for {asset_id}")
        
        try:
            # Generate cache key
            cache_key = f"{signal.name}_{asset_id}_{self.lookback_days}"
            
            # Check cache
            if cache_key in self.evaluations_cache:
                self.logger.debug(f"Using cached evaluation for {cache_key}")
                return self.evaluations_cache[cache_key]
            
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
            signal_series = signal.calculate(filtered_data, asset_id)
            
            # Calculate forward returns
            data_with_returns = self.calculator.calculate_forward_returns(filtered_data, price_col)
            
            # Get conditional returns
            conditional_returns, sample_counts = self.calculator.get_conditional_returns(
                data_with_returns, signal_series, min_samples=self.min_samples
            )
            
            # Check if we have enough samples
            if sample_counts['positive'] < self.min_samples or sample_counts['negative'] < self.min_samples:
                self.logger.warning(
                    f"Insufficient samples for signal {signal.name} on {asset_id}: "
                    f"positive={sample_counts['positive']}, negative={sample_counts['negative']}, "
                    f"need at least {self.min_samples}"
                )
                
                # Return limited evaluation
                evaluation = {
                    'signal_name': signal.name,
                    'asset_id': asset_id,
                    'sample_counts': sample_counts,
                    'valid': False,
                    'reason': f"Insufficient samples (need {self.min_samples})",
                    'effectiveness_by_period': {},
                    'overall_effectiveness': False,
                    'weight': 0.0
                }
                
                self.evaluations_cache[cache_key] = evaluation
                return evaluation
            
            # Calculate distribution statistics
            distribution_stats = self.calculator.calculate_distribution_stats(conditional_returns)
            
            # Run statistical tests for each period
            effectiveness_by_period = {}
            for period, returns_dict in conditional_returns.items():
                if 'positive' not in returns_dict or 'negative' not in returns_dict:
                    continue
                
                # Run tests
                test_results = self.tester.run_all_tests(
                    returns_dict['positive'], 
                    returns_dict['negative'],
                    returns_dict.get('all')
                )
                
                # Evaluate effectiveness
                effectiveness = self.tester.evaluate_signal_effectiveness(test_results)
                effectiveness_by_period[period] = {
                    'test_results': test_results,
                    'effectiveness': effectiveness
                }
            
            # Calculate overall effectiveness and weight
            overall_effectiveness, weight, metrics = self._calculate_overall_effectiveness(
                effectiveness_by_period, distribution_stats
            )
            
            # Determine optimal holding period
            optimal_period, optimal_sortino = self.determine_optimal_holding_period(
                effectiveness_by_period, distribution_stats
            )
            
            # Create evaluation result
            evaluation = {
                'signal_name': signal.name,
                'asset_id': asset_id,
                'sample_counts': sample_counts,
                'distribution_stats': distribution_stats,
                'effectiveness_by_period': effectiveness_by_period,
                'overall_effectiveness': overall_effectiveness,
                'metrics': metrics,
                'weight': weight,
                'optimal_holding_period': optimal_period,
                'optimal_sortino': optimal_sortino,
                'valid': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the result
            self.evaluations_cache[cache_key] = evaluation
            
            return evaluation
        
        except Exception as e:
            self.logger.error(f"Error evaluating signal {signal.name} for {asset_id}: {e}")
            return {
                'signal_name': signal.name,
                'asset_id': asset_id,
                'valid': False,
                'error': str(e),
                'weight': 0.0,
                'optimal_holding_period': 0
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
                    'optimal_holding_period': 0
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
    
    def determine_optimal_holding_period(
        self, 
        effectiveness_by_period: Dict[int, Dict[str, Any]], 
        distribution_stats: Dict[int, Dict[str, Dict[str, float]]],
        candidate_periods: List[int] = None
    ) -> Tuple[int, float]:
        """Determine the optimal holding period for a signal.
        
        Args:
            effectiveness_by_period: Effectiveness evaluations for different periods
            distribution_stats: Distribution statistics for returns
            candidate_periods: List of periods to consider (default: [7, 14, 30])
            
        Returns:
            tuple: (optimal_period, sortino_ratio)
        """
        # Default candidate periods if none provided
        candidate_periods = candidate_periods or [7, 14, 21, 28]
        
        # Filter to only include candidate periods that exist in effectiveness_by_period
        valid_periods = [p for p in candidate_periods if p in effectiveness_by_period]
        
        if not valid_periods:
            self.logger.warning("No valid periods found in evaluations")
            # Find the highest available period as a fallback
            available_periods = list(effectiveness_by_period.keys())
            if available_periods:
                default_period = max(available_periods)
                self.logger.info(f"Using highest available period ({default_period}) as fallback")
                return default_period, 0.0
            return 14, 0.0  # Default fallback if no periods available
        
        # Calculate Sortino ratio for each period
        sortino_ratios = {}
        for period in valid_periods:
            # Check if this period is effective
            period_data = effectiveness_by_period.get(period, {})
            effectiveness = period_data.get('effectiveness', {})
            is_effective = effectiveness.get('overall_effective', False)
            
            # Skip ineffective periods
            if not is_effective:
                continue
            
            # Get return statistics for positive signal
            if period not in distribution_stats or 'positive' not in distribution_stats[period]:
                continue
                
            pos_stats = distribution_stats[period]['positive']
            
            # Calculate Sortino ratio using mean and downside deviation
            mean_return = pos_stats.get('mean', 0.0)
            
            # Extract negative returns for downside deviation calculation
            neg_returns = []
            for val in pos_stats.values():
                if isinstance(val, (int, float)) and val < 0:
                    neg_returns.append(val)
            
            # If we have negative returns, calculate downside deviation
            if neg_returns:
                downside_std = np.std(neg_returns)
            else:
                # If no negative returns, use percentiles to estimate
                pct_10 = pos_stats.get('pct_10', 0.0)
                if pct_10 < 0:
                    # Approximate downside deviation using lowest percentile
                    downside_std = abs(pct_10)
                else:
                    # If even the 10th percentile is positive, use a small value
                    downside_std = 0.01
            
            # Calculate Sortino ratio (avoid division by zero)
            sortino = mean_return / downside_std if downside_std > 0 else 0.0
            
            sortino_ratios[period] = sortino
        
        # If no effective periods with valid Sortino ratios, use the longest candidate period
        if not sortino_ratios:
            self.logger.warning("No periods with significant effectiveness and valid Sortino ratios found")
            return max(valid_periods), 0.0
        
        # Find period with highest Sortino ratio
        optimal_period = max(sortino_ratios.items(), key=lambda x: x[1])[0]
        optimal_sortino = sortino_ratios[optimal_period]
        
        self.logger.info(f"Optimal holding period: {optimal_period} days (Sortino: {optimal_sortino:.4f})")
        
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
            
            # Determine majority direction
            mean_direction = np.mean(directions)
            direction = 1 if mean_direction >= 0 else -1
            
            # Calculate mean difference magnitude
            if mean_diffs:
                avg_mean_diff = np.mean(mean_diffs)
                avg_mean_diff_magnitude = abs(avg_mean_diff)
            else:
                avg_mean_diff_magnitude = 0.0
            
            # Calculate weighted sum (weighted by importance)
            # Formula: weight = effect_size × confidence × direction
            weight = avg_effect_size * avg_confidence * direction
            
            metrics = {
                'avg_effect_size': avg_effect_size,
                'avg_confidence': avg_confidence,
                'direction': direction,
                'avg_mean_diff': avg_mean_diff_magnitude,
                'ratio_effective': ratio_effective
            }
        else:
            weight = 0.0
            metrics = {
                'ratio_effective': ratio_effective
            }
        
        return overall_effective, weight, metrics
    
    def clear_cache(self):
        """Clear the evaluations cache."""
        self.evaluations_cache = {} 