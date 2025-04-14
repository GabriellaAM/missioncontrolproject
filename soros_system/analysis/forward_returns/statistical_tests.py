"""
Statistical tests for comparing forward return distributions.

This module provides tools for statistically comparing return distributions
to evaluate signal effectiveness.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple, Any, Union
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, ttest_1samp


class StatisticalTester:
    """Statistical tests for comparing return distributions."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize the statistical tester.
        
        Args:
            alpha (float): Significance level for statistical tests (default: 0.05)
        """
        self.logger = logging.getLogger(__name__)
        self.alpha = alpha
    
    def run_t_test(
        self, positive_returns: pd.Series, negative_returns: pd.Series, 
        all_returns: Optional[pd.Series] = None, equal_var: bool = False
    ) -> Dict[str, Any]:
        """Run t-test to compare means of return distributions.
        
        Args:
            positive_returns: Returns when signal is positive
            negative_returns: Returns when signal is negative
            all_returns: All returns (regardless of signal)
            equal_var: Whether to assume equal variance (default: False)
            
        Returns:
            dict: Dictionary with test results
        """
        results = {}
        
        try:
            # Check if we have enough samples
            if len(positive_returns) < 20 or len(negative_returns) < 20:
                self.logger.warning("Not enough samples for t-test")
                results['warning'] = "Not enough samples for t-test"
                results['valid'] = False
                return results
            
            # Run t-test comparing positive vs negative
            t_stat, p_value = ttest_ind(
                positive_returns, negative_returns, equal_var=equal_var, nan_policy='omit'
            )
            
            # Calculate effect size (Cohen's d)
            effect_size = self._cohens_d(positive_returns, negative_returns)
            
            # Store results
            results['test'] = 't-test'
            results['test_type'] = 'independent'
            results['comparison'] = 'positive_vs_negative'
            results['t_statistic'] = t_stat
            results['p_value'] = p_value
            results['significant'] = p_value < self.alpha
            results['effect_size'] = effect_size
            results['effect_magnitude'] = self._interpret_cohens_d(effect_size)
            results['mean_difference'] = positive_returns.mean() - negative_returns.mean()
            results['valid'] = True
            
            # Compare with all returns if provided
            if all_returns is not None and len(all_returns) > 20:
                # Compare positive returns to all returns
                t_stat_pos, p_value_pos = ttest_ind(
                    positive_returns, all_returns, equal_var=equal_var, nan_policy='omit'
                )
                effect_size_pos = self._cohens_d(positive_returns, all_returns)
                
                results['positive_vs_all'] = {
                    't_statistic': t_stat_pos,
                    'p_value': p_value_pos,
                    'significant': p_value_pos < self.alpha,
                    'effect_size': effect_size_pos,
                    'effect_magnitude': self._interpret_cohens_d(effect_size_pos),
                    'mean_difference': positive_returns.mean() - all_returns.mean()
                }
                
                # Compare negative returns to all returns
                t_stat_neg, p_value_neg = ttest_ind(
                    negative_returns, all_returns, equal_var=equal_var, nan_policy='omit'
                )
                effect_size_neg = self._cohens_d(negative_returns, all_returns)
                
                results['negative_vs_all'] = {
                    't_statistic': t_stat_neg,
                    'p_value': p_value_neg,
                    'significant': p_value_neg < self.alpha,
                    'effect_size': effect_size_neg,
                    'effect_magnitude': self._interpret_cohens_d(effect_size_neg),
                    'mean_difference': negative_returns.mean() - all_returns.mean()
                }
        
        except Exception as e:
            self.logger.error(f"Error running t-test: {e}")
            results['error'] = str(e)
            results['valid'] = False
        
        return results
    
    def run_mann_whitney_test(
        self, positive_returns: pd.Series, negative_returns: pd.Series,
        all_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Run Mann-Whitney U test to compare distributions (non-parametric).
        
        Args:
            positive_returns: Returns when signal is positive
            negative_returns: Returns when signal is negative
            all_returns: All returns (regardless of signal)
            
        Returns:
            dict: Dictionary with test results
        """
        results = {}
        
        try:
            # Check if we have enough samples
            if len(positive_returns) < 20 or len(negative_returns) < 20:
                self.logger.warning("Not enough samples for Mann-Whitney U test")
                results['warning'] = "Not enough samples for Mann-Whitney U test"
                results['valid'] = False
                return results
            
            # Run Mann-Whitney U test
            stat, p_value = mannwhitneyu(
                positive_returns, negative_returns, alternative='two-sided'
            )
            
            # Store results
            results['test'] = 'Mann-Whitney U'
            results['test_type'] = 'non-parametric'
            results['comparison'] = 'positive_vs_negative'
            results['statistic'] = stat
            results['p_value'] = p_value
            results['significant'] = p_value < self.alpha
            results['median_difference'] = positive_returns.median() - negative_returns.median()
            results['valid'] = True
            
            # Compare with all returns if provided
            if all_returns is not None and len(all_returns) > 20:
                # Compare positive returns to all returns
                stat_pos, p_value_pos = mannwhitneyu(
                    positive_returns, all_returns, alternative='two-sided'
                )
                
                results['positive_vs_all'] = {
                    'statistic': stat_pos,
                    'p_value': p_value_pos,
                    'significant': p_value_pos < self.alpha,
                    'median_difference': positive_returns.median() - all_returns.median()
                }
                
                # Compare negative returns to all returns
                stat_neg, p_value_neg = mannwhitneyu(
                    negative_returns, all_returns, alternative='two-sided'
                )
                
                results['negative_vs_all'] = {
                    'statistic': stat_neg,
                    'p_value': p_value_neg,
                    'significant': p_value_neg < self.alpha,
                    'median_difference': negative_returns.median() - all_returns.median()
                }
        
        except Exception as e:
            self.logger.error(f"Error running Mann-Whitney U test: {e}")
            results['error'] = str(e)
            results['valid'] = False
        
        return results
    
    def run_ks_test(
        self, positive_returns: pd.Series, negative_returns: pd.Series,
        all_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Run Kolmogorov-Smirnov test to compare distributions.
        
        Args:
            positive_returns: Returns when signal is positive
            negative_returns: Returns when signal is negative
            all_returns: All returns (regardless of signal)
            
        Returns:
            dict: Dictionary with test results
        """
        results = {}
        
        try:
            # Check if we have enough samples
            if len(positive_returns) < 20 or len(negative_returns) < 20:
                self.logger.warning("Not enough samples for KS test")
                results['warning'] = "Not enough samples for KS test"
                results['valid'] = False
                return results
            
            # Run KS test
            stat, p_value = ks_2samp(positive_returns, negative_returns)
            
            # Store results
            results['test'] = 'Kolmogorov-Smirnov'
            results['test_type'] = 'non-parametric'
            results['comparison'] = 'positive_vs_negative'
            results['statistic'] = stat
            results['p_value'] = p_value
            results['significant'] = p_value < self.alpha
            results['valid'] = True
            
            # Compare with all returns if provided
            if all_returns is not None and len(all_returns) > 20:
                # Compare positive returns to all returns
                stat_pos, p_value_pos = ks_2samp(positive_returns, all_returns)
                
                results['positive_vs_all'] = {
                    'statistic': stat_pos,
                    'p_value': p_value_pos,
                    'significant': p_value_pos < self.alpha
                }
                
                # Compare negative returns to all returns
                stat_neg, p_value_neg = ks_2samp(negative_returns, all_returns)
                
                results['negative_vs_all'] = {
                    'statistic': stat_neg,
                    'p_value': p_value_neg,
                    'significant': p_value_neg < self.alpha
                }
        
        except Exception as e:
            self.logger.error(f"Error running KS test: {e}")
            results['error'] = str(e)
            results['valid'] = False
        
        return results
    
    def run_skew_and_kurt_test(
        self, positive_returns: pd.Series, negative_returns: pd.Series,
        all_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Analyze skewness and kurtosis of return distributions.
        
        Args:
            positive_returns: Returns when signal is positive
            negative_returns: Returns when signal is negative
            all_returns: All returns (regardless of signal)
            
        Returns:
            dict: Dictionary with test results
        """
        results = {}
        
        try:
            # Check if we have enough samples
            if len(positive_returns) < 50 or len(negative_returns) < 50:
                self.logger.warning("Small sample for skew/kurtosis test")
                results['warning'] = "Small sample for skew/kurtosis test"
            
            # Calculate skewness and kurtosis
            pos_skew = stats.skew(positive_returns)
            neg_skew = stats.skew(negative_returns)
            # Use fisher=False to get true kurtosis rather than excess kurtosis (kurtosis - 3)
            pos_kurt = stats.kurtosis(positive_returns, fisher=False)
            neg_kurt = stats.kurtosis(negative_returns, fisher=False)
            
            # Store results
            results['test'] = 'Skew and Kurtosis'
            results['positive_skew'] = pos_skew
            results['negative_skew'] = neg_skew
            results['positive_kurtosis'] = pos_kurt
            results['negative_kurtosis'] = neg_kurt
            results['skew_difference'] = pos_skew - neg_skew
            results['kurt_difference'] = pos_kurt - neg_kurt
            results['valid'] = True
            
            # Compare with all returns if provided
            if all_returns is not None and len(all_returns) > 20:
                all_skew = stats.skew(all_returns)
                all_kurt = stats.kurtosis(all_returns, fisher=False)
                
                results['all_skew'] = all_skew
                results['all_kurtosis'] = all_kurt
                results['positive_vs_all_skew_diff'] = pos_skew - all_skew
                results['negative_vs_all_skew_diff'] = neg_skew - all_skew
                results['positive_vs_all_kurt_diff'] = pos_kurt - all_kurt
                results['negative_vs_all_kurt_diff'] = neg_kurt - all_kurt
        
        except Exception as e:
            self.logger.error(f"Error running skew/kurtosis test: {e}")
            results['error'] = str(e)
            results['valid'] = False
        
        return results
    
    def run_all_tests(
        self, positive_returns: pd.Series, negative_returns: pd.Series,
        all_returns: Optional[pd.Series] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Run all statistical tests for comparing distributions.
        
        Args:
            positive_returns: Returns when signal is positive
            negative_returns: Returns when signal is negative
            all_returns: All returns (regardless of signal)
            
        Returns:
            dict: Dictionary with all test results
        """
        results = {}
        
        # Run t-test
        results['t_test'] = self.run_t_test(
            positive_returns, negative_returns, all_returns
        )
        
        # Run Mann-Whitney U test
        results['mann_whitney'] = self.run_mann_whitney_test(
            positive_returns, negative_returns, all_returns
        )
        
        # Run KS test
        results['ks_test'] = self.run_ks_test(
            positive_returns, negative_returns, all_returns
        )
        
        # Run skew and kurtosis analysis
        results['skew_kurt'] = self.run_skew_and_kurt_test(
            positive_returns, negative_returns, all_returns
        )
        
        return results
    
    def evaluate_signal_effectiveness(
        self, test_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate overall signal effectiveness based on test results.
        
        Note: This method is maintained for backward compatibility. 
        For new code, use evaluate_signal_effectiveness_new directly.
        
        Args:
            test_results: Dictionary with test results from run_all_tests
            
        Returns:
            dict: Dictionary with overall effectiveness metrics
        """
        self.logger.warning(
            "The evaluate_signal_effectiveness method is deprecated. "
            "Use evaluate_signal_effectiveness_new instead."
        )
        
        # Extract the returns by period from test results
        # This is a bit hacky but necessary for backward compatibility
        returns_by_period = {}
        for test_name, test_result in test_results.items():
            if test_name == 'Mann-Whitney U':
                for period, period_data in test_result.items():
                    if isinstance(period_data, dict) and 'raw_data' in period_data:
                        raw_data = period_data['raw_data']
                        positive_returns = raw_data.get('positive_returns', pd.Series())
                        negative_returns = raw_data.get('negative_returns', pd.Series())
                        returns_by_period[period] = (positive_returns, negative_returns)
        
        # If we couldn't extract the returns, return a default result
        if not returns_by_period:
            return {
                'overall_effective': False,
                'weight': 0.0,
                'reason': "Could not extract return data from test results"
            }
        
        # Use the new implementation
        result = self.evaluate_signal_effectiveness_new(returns_by_period)
        
        # Return a simplified version for backward compatibility
        return {
            'overall_effective': result['overall_effective'],
            'weight': result['metrics'].get('weight', 0.0),
            'confidence': result['metrics'].get('confidence', 0.0),
            'direction': 1 if result['metrics'].get('delta', 0.0) > 0 else -1 if result['metrics'].get('delta', 0.0) < 0 else 0,
            'optimal_period': result['metrics'].get('optimal_period', 0),
            'metrics': result['metrics']
        }
    
    def _cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cohen's d effect size.
        
        Args:
            group1: First group data
            group2: Second group data
            
        Returns:
            float: Cohen's d effect size
        """
        # Calculate means
        mean1 = group1.mean()
        mean2 = group2.mean()
        
        # Calculate pooled standard deviation
        n1 = len(group1)
        n2 = len(group2)
        s1 = group1.std()
        s2 = group2.std()
        
        # Pooled standard deviation formula
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        # Calculate Cohen's d
        if pooled_std == 0:
            return 0  # Avoid division by zero
        
        d = (mean1 - mean2) / pooled_std
        return d
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size.
        
        Args:
            d: Cohen's d value
            
        Returns:
            str: Interpretation of effect size
        """
        d = abs(d)  # Use absolute value for interpretation
        
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def calculate_cliffs_delta(
        self, group1: pd.Series, group2: pd.Series
    ) -> float:
        """Calculate Cliff's Delta, a non-parametric effect size measure.
        
        Cliff's Delta measures the probability that a randomly chosen value
        from group1 is greater than a randomly chosen value from group2,
        minus the reverse probability. It ranges from -1 to 1, where:
        - -1: All values in group2 > all values in group1
        - 0: Groups are completely overlapping
        - 1: All values in group1 > all values in group2
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            float: Cliff's Delta effect size
        """
        # Convert to numpy arrays for faster computation
        x = group1.to_numpy()
        y = group2.to_numpy()
        
        # Count comparisons
        count1 = 0  # x > y
        count2 = 0  # x < y
        
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] > y[j]:
                    count1 += 1
                elif x[i] < y[j]:
                    count2 += 1
        
        # Calculate delta
        delta = (count1 - count2) / (len(x) * len(y))
        return delta
    
    def interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's Delta effect size.
        
        Args:
            delta: Cliff's Delta value
            
        Returns:
            str: Interpretation of effect size
        """
        abs_delta = abs(delta)
        
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.33:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def apply_fdr_correction(
        self, p_values: Dict[int, float], alpha: float = 0.05
    ) -> Dict[int, bool]:
        """Apply Benjamini-Hochberg FDR correction to p-values.
        
        Args:
            p_values: Dictionary mapping test identifiers to p-values
            alpha: Desired FDR level (default: 0.05)
            
        Returns:
            dict: Dictionary mapping test identifiers to significance (True/False)
        """
        if not p_values:
            return {}
        
        # Convert to list and sort
        items = list(p_values.items())
        items.sort(key=lambda x: x[1])  # Sort by p-value
        
        # Apply B-H procedure
        m = len(items)
        significance = {}
        
        # Find the largest k such that P(k) ≤ (k/m) * alpha
        for k, (identifier, p_value) in enumerate(items, start=1):
            if p_value <= (k / m) * alpha:
                significance[identifier] = True
            else:
                significance[identifier] = False
        
        # Convert to original identifier order
        result = {}
        for identifier in p_values:
            result[identifier] = significance.get(identifier, False)
        
        return result
        
    def run_mann_whitney_and_cliffs_delta(
        self, positive_returns: pd.Series, negative_returns: pd.Series
    ) -> Dict[str, Any]:
        """Run Mann-Whitney U test and calculate Cliff's Delta.
        
        This is the primary test for evaluating signal effectiveness.
        
        Args:
            positive_returns: Returns when signal is positive
            negative_returns: Returns when signal is negative
            
        Returns:
            dict: Dictionary with test results
        """
        results = {}
        
        try:
            # Check if we have enough samples
            if len(positive_returns) < 20 or len(negative_returns) < 20:
                self.logger.warning("Not enough samples for Mann-Whitney U test")
                results['warning'] = "Not enough samples for Mann-Whitney U test"
                results['valid'] = False
                return results
            
            # Run Mann-Whitney U test
            stat, p_value = mannwhitneyu(
                positive_returns, negative_returns, alternative='two-sided'
            )
            
            # Calculate Cliff's Delta
            delta = self.calculate_cliffs_delta(positive_returns, negative_returns)
            effect_magnitude = self.interpret_cliffs_delta(delta)
            
            # Store results
            results['test'] = 'Mann-Whitney U with Cliff\'s Delta'
            results['test_type'] = 'non-parametric'
            results['comparison'] = 'positive_vs_negative'
            results['statistic'] = stat
            results['p_value'] = p_value
            results['significant'] = p_value < self.alpha
            results['effect_size'] = delta
            results['effect_magnitude'] = effect_magnitude
            results['median_difference'] = positive_returns.median() - negative_returns.median()
            results['mean_difference'] = positive_returns.mean() - negative_returns.mean()
            results['valid'] = True
            
        except Exception as e:
            self.logger.error(f"Error running Mann-Whitney U test with Cliff's Delta: {e}")
            results['error'] = str(e)
            results['valid'] = False
        
        return results
        
    def evaluate_signal_effectiveness_new(
        self, returns_by_period: Dict[int, Tuple[pd.Series, pd.Series]]
    ) -> Dict[str, Any]:
        """Evaluate signal effectiveness using Mann-Whitney U and FDR correction.
        
        Args:
            returns_by_period: Dictionary mapping periods to (positive_returns, negative_returns)
            
        Returns:
            dict: Evaluation results with metrics
        """
        # Run tests for each period
        test_results = {}
        p_values = {}
        all_period_data = {}
        
        for period, (positive_returns, negative_returns) in returns_by_period.items():
            # Run Mann-Whitney U test and calculate Cliff's Delta
            results = self.run_mann_whitney_and_cliffs_delta(positive_returns, negative_returns)
            test_results[period] = results
            
            # Store p-value for FDR correction
            if results.get('valid', False):
                p_values[period] = results.get('p_value', 1.0)
                all_period_data[period] = {
                    'delta': results.get('effect_size', 0.0),
                    'mean_diff': results.get('mean_difference', 0.0),
                    'significant': results.get('significant', False),
                    'effect_magnitude': results.get('effect_magnitude', 'negligible')
                }
        
        # Apply FDR correction
        fdr_significant = self.apply_fdr_correction(p_values)
        
        # Calculate confidence and weight for each period
        confidence_by_period = {}
        weight_by_period = {}
        
        for period, is_significant in fdr_significant.items():
            delta = all_period_data[period]['delta']
            
            # Calculate confidence
            if is_significant and abs(delta) >= 0.147:  # Only non-negligible effects
                confidence = abs(delta)  # Confidence is the absolute effect size
            else:
                confidence = 0.0
            
            # Calculate weight
            weight = delta * confidence  # Weight includes direction (positive or negative)
            
            confidence_by_period[period] = confidence
            weight_by_period[period] = weight
        
        # Determine overall effectiveness
        effective_periods = [p for p, conf in confidence_by_period.items() if conf > 0]
        overall_effective = len(effective_periods) > 0
        
        # Calculate overall metrics
        if overall_effective:
            # Find period with highest confidence
            best_period = max(confidence_by_period.items(), key=lambda x: x[1])[0]
            optimal_period = best_period
            optimal_delta = all_period_data[best_period]['delta']
            optimal_confidence = confidence_by_period[best_period]
            optimal_weight = weight_by_period[best_period]
            
            # Summary metrics
            metrics = {
                'optimal_period': optimal_period,
                'delta': optimal_delta,
                'confidence': optimal_confidence,
                'weight': optimal_weight,
                'effect_magnitude': all_period_data[best_period]['effect_magnitude'],
                'mean_difference': all_period_data[best_period]['mean_diff'],
                'effective_periods': len(effective_periods),
                'total_periods': len(returns_by_period),
                'effective_ratio': len(effective_periods) / len(returns_by_period)
            }
        else:
            # No effective periods
            metrics = {
                'optimal_period': 0,
                'delta': 0.0,
                'confidence': 0.0,
                'weight': 0.0,
                'effect_magnitude': 'negligible',
                'mean_difference': 0.0,
                'effective_periods': 0,
                'total_periods': len(returns_by_period),
                'effective_ratio': 0.0
            }
        
        return {
            'overall_effective': overall_effective,
            'metrics': metrics,
            'confidence_by_period': confidence_by_period,
            'weight_by_period': weight_by_period,
            'fdr_significant': fdr_significant,
            'all_period_data': all_period_data
        } 