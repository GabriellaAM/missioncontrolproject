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
            pos_kurt = stats.kurtosis(positive_returns)
            neg_kurt = stats.kurtosis(negative_returns)
            
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
                all_kurt = stats.kurtosis(all_returns)
                
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
        
        Args:
            test_results: Results from run_all_tests
            
        Returns:
            dict: Dictionary with effectiveness evaluation
        """
        evaluation = {
            'valid_tests': 0,
            'significant_tests': 0,
            'mean_superiority': False,
            'median_superiority': False,
            'distribution_difference': False,
            'favorable_skew': False,
            'overall_effective': False,
            'confidence': 0.0,
            'effect_size': 0.0
        }
        
        # Count valid and significant tests
        if 't_test' in test_results and test_results['t_test'].get('valid', False):
            evaluation['valid_tests'] += 1
            if test_results['t_test'].get('significant', False):
                evaluation['significant_tests'] += 1
                
                # Check if positive returns have higher mean
                mean_diff = test_results['t_test'].get('mean_difference', 0)
                if mean_diff > 0:
                    evaluation['mean_superiority'] = True
                    evaluation['effect_size'] = test_results['t_test'].get('effect_size', 0)
        
        if 'mann_whitney' in test_results and test_results['mann_whitney'].get('valid', False):
            evaluation['valid_tests'] += 1
            if test_results['mann_whitney'].get('significant', False):
                evaluation['significant_tests'] += 1
                
                # Check if positive returns have higher median
                median_diff = test_results['mann_whitney'].get('median_difference', 0)
                if median_diff > 0:
                    evaluation['median_superiority'] = True
        
        if 'ks_test' in test_results and test_results['ks_test'].get('valid', False):
            evaluation['valid_tests'] += 1
            if test_results['ks_test'].get('significant', False):
                evaluation['significant_tests'] += 1
                evaluation['distribution_difference'] = True
        
        if 'skew_kurt' in test_results and test_results['skew_kurt'].get('valid', False):
            # Check if positive returns have more favorable skew
            pos_skew = test_results['skew_kurt'].get('positive_skew', 0)
            neg_skew = test_results['skew_kurt'].get('negative_skew', 0)
            
            # Positive skew is generally favorable for returns
            if pos_skew > neg_skew:
                evaluation['favorable_skew'] = True
        
        # Calculate confidence based on proportion of significant tests
        if evaluation['valid_tests'] > 0:
            evaluation['confidence'] = evaluation['significant_tests'] / evaluation['valid_tests']
        
        # Overall effectiveness criteria
        if (evaluation['mean_superiority'] or evaluation['median_superiority']) and evaluation['confidence'] >= 0.5:
            evaluation['overall_effective'] = True
        
        return evaluation
    
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