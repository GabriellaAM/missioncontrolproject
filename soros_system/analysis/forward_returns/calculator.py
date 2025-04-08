"""
Forward returns calculator for signal evaluation.

This module provides tools for calculating forward returns over multiple periods
to evaluate signal effectiveness.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union, Tuple


class ForwardReturnsCalculator:
    """Calculate forward returns for signal evaluation."""
    
    def __init__(self, periods: List[int] = None):
        """Initialize the forward returns calculator.
        
        Args:
            periods (list): List of forward periods (in days) for return calculation. 
                           Default: [1, 3, 5, 7, 14, 21, 28]
        """
        self.logger = logging.getLogger(__name__)
        self.periods = periods or [1, 3, 5, 7, 14, 21, 28]
    
    def calculate_forward_returns(self, data: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Calculate forward returns for multiple periods.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            price_col (str): Column to use for price data (default: 'close')
            
        Returns:
            pd.DataFrame: DataFrame with forward returns for each period
        """
        if data.empty:
            self.logger.warning("Empty data provided")
            return pd.DataFrame()
        
        # Check if price column exists
        if price_col not in data.columns:
            self.logger.error(f"Column '{price_col}' not found in data")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Ensure data is sorted by date
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        else:
            # If 'date' is a column, sort by it
            if 'date' in df.columns:
                df = df.sort_values('date')
        
        # Calculate forward returns for each period
        for period in self.periods:
            col_name = f'fwd_ret_{period}d'
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    # Use shift with negative period for forward returns
                    df[col_name] = (df[price_col].shift(-period) / df[price_col]) - 1
                else:
                    # If not DatetimeIndex, use iloc-based approach
                    df[col_name] = np.nan
                    for i in range(len(df) - period):
                        df.iloc[i, df.columns.get_loc(col_name)] = (
                            df.iloc[i + period][price_col] / df.iloc[i][price_col]
                        ) - 1
            except Exception as e:
                self.logger.error(f"Error calculating {period}-day forward returns: {e}")
                df[col_name] = np.nan
        
        return df
    
    def get_conditional_returns(
        self, data: pd.DataFrame, signal_series: pd.Series, 
        min_samples: int = 20
    ) -> Tuple[Dict[int, Dict[str, pd.Series]], Dict[str, int]]:
        """Get forward returns conditioned on signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with forward returns
            signal_series (pd.Series): Series with signal values (1 or 0)
            min_samples (int): Minimum samples required for each condition
            
        Returns:
            tuple: (
                conditional_returns_dict,
                sample_counts
            )
            where:
                conditional_returns_dict: Dictionary mapping periods to dictionaries of returns
                    {period: {'positive': Series, 'negative': Series, 'all': Series}}
                sample_counts: Dictionary with counts of samples for each condition
                    {'positive': int, 'negative': int, 'all': int}
        """
        if data.empty or signal_series.empty:
            self.logger.warning("Empty data or signal series provided")
            return {}, {'positive': 0, 'negative': 0, 'all': len(data)}
        
        # Create a copy of the data with the signal added
        df = data.copy()
        
        # Add signal column, aligning by index
        if isinstance(df.index, pd.DatetimeIndex) and isinstance(signal_series.index, pd.DatetimeIndex):
            # Align by DatetimeIndex
            df['signal'] = signal_series
        else:
            # If no DatetimeIndex, assume they're already aligned
            df['signal'] = signal_series.values
        
        # Get forward return columns
        fwd_cols = [col for col in df.columns if col.startswith('fwd_ret_')]
        if not fwd_cols:
            self.logger.warning("No forward return columns found in data")
            return {}, {'positive': 0, 'negative': 0, 'all': len(data)}
        
        # Initialize dictionaries for conditional returns
        conditional_returns = {}
        
        # Filter returns based on signal value
        positive_signal = df[df['signal'] == 1]  # Signal is active
        negative_signal = df[df['signal'] == 0]  # Signal is inactive
        all_signal = df  # All data, regardless of signal
        
        # Calculate sample counts
        sample_counts = {
            'positive': len(positive_signal),
            'negative': len(negative_signal),
            'all': len(all_signal)
        }
        
        # Log sample counts
        self.logger.info(f"Signal distribution: {sample_counts['positive']} positive (1), "
                        f"{sample_counts['negative']} negative (0), {sample_counts['all']} total")
        
        # Check if we have sufficient samples
        has_positive = sample_counts['positive'] >= min_samples
        has_negative = sample_counts['negative'] >= min_samples
        
        if not has_positive and not has_negative:
            self.logger.warning(f"Insufficient samples for both positive and negative signals. "
                              f"Need at least {min_samples}, got {sample_counts}")
        elif not has_positive:
            self.logger.warning(f"Insufficient samples for positive signal. "
                              f"Need at least {min_samples}, got {sample_counts['positive']}")
        elif not has_negative:
            self.logger.warning(f"Insufficient samples for negative signal. "
                              f"Need at least {min_samples}, got {sample_counts['negative']}")
        
        # Extract conditional returns for each period
        for col in fwd_cols:
            # Extract period number from column name
            period = int(col.split('_')[2].replace('d', ''))
            
            # Create dictionary for this period
            period_dict = {}
            
            # Extract returns for each condition
            if has_positive:
                period_dict['positive'] = positive_signal[col].dropna()
            
            if has_negative:
                period_dict['negative'] = negative_signal[col].dropna()
            
            period_dict['all'] = all_signal[col].dropna()
            
            # Add to conditional returns dictionary
            conditional_returns[period] = period_dict
        
        return conditional_returns, sample_counts
    
    def calculate_distribution_stats(
        self, conditional_returns: Dict[int, Dict[str, pd.Series]]
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Calculate distribution statistics for conditional returns.
        
        Args:
            conditional_returns: Dictionary mapping periods to dictionaries of returns
                {period: {'positive': Series, 'negative': Series, 'all': Series}}
                
        Returns:
            Dict: Dictionary with statistics for each period and condition
                {period: {'positive': stats_dict, 'negative': stats_dict, 'all': stats_dict}}
                where stats_dict has keys: mean, median, std, skew, kurt, etc.
        """
        if not conditional_returns:
            self.logger.warning("Empty conditional returns provided")
            return {}
        
        distribution_stats = {}
        
        for period, returns_dict in conditional_returns.items():
            period_stats = {}
            
            for condition, returns in returns_dict.items():
                if returns.empty:
                    self.logger.warning(f"No returns for {condition} condition in {period}-day period")
                    continue
                
                # Calculate statistics
                stats = {}
                try:
                    stats['count'] = len(returns)
                    stats['mean'] = returns.mean()
                    stats['median'] = returns.median()
                    stats['std'] = returns.std()
                    stats['min'] = returns.min()
                    stats['max'] = returns.max()
                    stats['skew'] = returns.skew()
                    stats['kurt'] = returns.kurt()
                    
                    # Calculate percentiles
                    stats['pct_10'] = returns.quantile(0.1)
                    stats['pct_25'] = returns.quantile(0.25)
                    stats['pct_75'] = returns.quantile(0.75)
                    stats['pct_90'] = returns.quantile(0.9)
                    
                    # Calculate win rate (% of positive returns)
                    stats['win_rate'] = (returns > 0).mean() * 100
                    
                    # Add to period stats
                    period_stats[condition] = stats
                    
                except Exception as e:
                    self.logger.error(f"Error calculating statistics for {condition} condition in {period}-day period: {e}")
            
            # Add to distribution stats
            distribution_stats[period] = period_stats
        
        return distribution_stats 