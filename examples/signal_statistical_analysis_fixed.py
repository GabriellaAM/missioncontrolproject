#!/usr/bin/env python
"""
Signal statistical analysis example with fixes for binary (0/1) signals.

This script demonstrates the statistical testing framework and signal selection process,
with fixes to handle binary signals correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from soros_system.main import TrendAnalyzer
from soros_system.analysis.markov_vol_model import MarkovVolModel
from soros_system.signals.signal_base import SignalBase
from soros_system.signals.trend_signals import (
    # USD Quote Trend Signals
    ShortTermStrongBearUSD, ShortTermWeakBearUSD, ShortTermNeutralUSD, ShortTermWeakBullUSD, ShortTermStrongBullUSD,
    MediumTermStrongBearUSD, MediumTermWeakBearUSD, MediumTermNeutralUSD, MediumTermWeakBullUSD, MediumTermStrongBullUSD,
    LongTermStrongBearUSD, LongTermWeakBearUSD, LongTermNeutralUSD, LongTermWeakBullUSD, LongTermStrongBullUSD,
    OverallStrongBearUSD, OverallWeakBearUSD, OverallNeutralUSD, OverallWeakBullUSD, OverallStrongBullUSD,
)
from soros_system.signals.rsi_signals import (
    RSI_Oversold_USD, RSI_Overbought_USD, RSI_Bullish_USD, RSI_Bearish_USD,
)
from soros_system.signals.volatility_signals import (
    MarkovLowVolatilitySignal, MarkovHighVolatilitySignal
)
from soros_system.signals.ssr_signals import SSR_RiskOn, SSR_RiskOff
from soros_system.analysis.forward_returns.calculator import ForwardReturnsCalculator
from soros_system.analysis.forward_returns.statistical_tests import StatisticalTester


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('signal_evaluation.log')
        ]
    )


# Create a fixed version of the get_conditional_returns method
class BinaryForwardReturnsCalculator(ForwardReturnsCalculator):
    """Forward returns calculator that handles binary (0/1) signals correctly."""
    
    def get_conditional_returns(
        self, data: pd.DataFrame, signal_series: pd.Series, 
        min_samples: int = 20
    ):
        """Get forward returns conditioned on binary signal values (0/1).
        
        Args:
            data (pd.DataFrame): DataFrame with forward returns
            signal_series (pd.Series): Series with signal values (1 for signal, 0 for no signal)
            min_samples (int): Minimum samples required for each condition
            
        Returns:
            tuple: (
                conditional_returns_dict,
                sample_counts
            )
        """
        if data.empty or signal_series.empty:
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
            return {}, {'positive': 0, 'negative': 0, 'all': len(data)}
        
        # Initialize dictionaries for conditional returns
        conditional_returns = {}
        
        # Filter returns based on signal value
        # FIXED: Use 1 for positive and 0 for negative instead of 1/-1
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
        print(f"Signal distribution: {sample_counts['positive']} positive (1), "
              f"{sample_counts['negative']} negative (0), {sample_counts['all']} total")
        
        # Check if we have sufficient samples
        has_positive = sample_counts['positive'] >= min_samples
        has_negative = sample_counts['negative'] >= min_samples
        
        if not has_positive and not has_negative:
            print(f"Insufficient samples for both positive and negative signals. "
                  f"Need at least {min_samples}, got {sample_counts}")
        elif not has_positive:
            print(f"Insufficient samples for positive signal. "
                  f"Need at least {min_samples}, got {sample_counts['positive']}")
        elif not has_negative:
            print(f"Insufficient samples for negative signal. "
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


# Create a fixed version of the signal evaluator
class BinarySignalEvaluator:
    """A simplified signal evaluator that handles binary (0/1) signals correctly."""
    
    def __init__(
        self, 
        periods: list = None,
        min_samples: int = 30,
        lookback_days: int = 365
    ):
        """Initialize the evaluator with modified calculator."""
        self.periods = periods or [1, 3, 5, 7, 14, 21, 28]
        self.min_samples = min_samples
        self.lookback_days = lookback_days
        
        # Initialize modified components
        self.calculator = BinaryForwardReturnsCalculator(periods=self.periods)
        self.tester = StatisticalTester(alpha=0.05)
        
        # Cache for evaluations
        self.evaluations_cache = {}
    
    def evaluate_signal(self, signal: SignalBase, data: pd.DataFrame, asset_id: str, price_col: str = 'close'):
        """Evaluate a binary signal with the fixed calculator."""
        print(f"Evaluating signal {signal.name} for {asset_id}")
        
        try:
            # Filter data if lookback is specified
            if self.lookback_days is not None:
                if isinstance(data.index, pd.DatetimeIndex):
                    cutoff_date = data.index[-1] - pd.Timedelta(days=self.lookback_days)
                    filtered_data = data[data.index >= cutoff_date]
                else:
                    if 'date' in data.columns:
                        data = data.sort_values('date')
                        filtered_data = data.iloc[max(0, len(data) - self.lookback_days):]
                    else:
                        filtered_data = data.iloc[max(0, len(data) - self.lookback_days):]
            else:
                filtered_data = data
            
            print(f"Using {len(filtered_data)} days of data for analysis")
            
            # Calculate signal values
            signal_series = signal.calculate(filtered_data, asset_id)
            print(f"Signal distribution: 1s: {sum(signal_series == 1)}, 0s: {sum(signal_series == 0)}")
            
            # Calculate forward returns
            data_with_returns = self.calculator.calculate_forward_returns(filtered_data, price_col)
            
            # Get conditional returns
            conditional_returns, sample_counts = self.calculator.get_conditional_returns(
                data_with_returns, signal_series, min_samples=self.min_samples
            )
            
            # Check if we have enough samples
            if sample_counts['positive'] < self.min_samples or sample_counts['negative'] < self.min_samples:
                return {
                    'signal_name': signal.name,
                    'asset_id': asset_id,
                    'sample_counts': sample_counts,
                    'valid': False,
                    'reason': f"Insufficient samples (need {self.min_samples})",
                    'effectiveness_by_period': {},
                    'overall_effectiveness': False,
                    'weight': 0.0
                }
            
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
                'valid': True
            }
            
            return evaluation
        
        except Exception as e:
            print(f"Error evaluating signal {signal.name}: {e}")
            return {
                'signal_name': signal.name,
                'asset_id': asset_id,
                'valid': False,
                'error': str(e),
                'weight': 0.0
            }
    
    def _calculate_overall_effectiveness(
        self, effectiveness_by_period, distribution_stats
    ):
        """Calculate overall effectiveness with relaxed threshold."""
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
        
        # Calculate average metrics
        avg_effect_size = np.mean(effect_sizes) if effect_sizes else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        avg_mean_diff = np.mean(mean_diffs) if mean_diffs else 0.0
        
        # Determine dominant direction (positive or negative signal)
        # If there are more periods with positive direction, the signal is bullish
        # Otherwise, it's bearish
        direction = 1 if sum(1 for d in directions if d > 0) >= sum(1 for d in directions if d < 0) else -1
        
        # Calculate effectiveness ratio
        effectiveness_ratio = effective_periods / total_periods if total_periods > 0 else 0.0
        
        # Store metrics
        metrics = {
            'effective_periods': effective_periods,
            'total_periods': total_periods,
            'effectiveness_ratio': effectiveness_ratio, 
            'avg_effect_size': avg_effect_size,
            'avg_confidence': avg_confidence, 
            'avg_mean_diff': avg_mean_diff,
            'direction': direction
        }
        
        # FIXED: Determine overall effectiveness with a lower effect size threshold
        # Original: overall_effective = effectiveness_ratio >= 0.5 and avg_effect_size > 0.2
        overall_effective = effectiveness_ratio >= 0.5 and avg_effect_size > 0.15
        
        # Calculate weight based on metrics
        if overall_effective:
            # Weight is based on effect size, confidence, and mean difference
            weight_magnitude = (avg_effect_size * 0.5) + (avg_confidence * 0.3) + (abs(avg_mean_diff) * 100 * 0.2)
            # Apply direction to weight
            weight = weight_magnitude * direction
        else:
            # FIXED: Still assign a small weight even if not "overall effective"
            weight_magnitude = (avg_effect_size * 0.25) + (avg_confidence * 0.15) + (abs(avg_mean_diff) * 100 * 0.1)
            # Apply direction to weight
            weight = weight_magnitude * direction
            if effectiveness_ratio < 0.3 or avg_effect_size < 0.1:
                weight = 0.0
        
        return overall_effective, weight, metrics


def plot_binary_return_distributions(positive_returns, negative_returns, signal_name, period):
    """Plot return distributions for positive and negative signals.
    
    Args:
        positive_returns (pd.Series): Returns when signal is 1
        negative_returns (pd.Series): Returns when signal is 0
        signal_name (str): Name of the signal
        period (int): Forward return period
    """
    plt.figure(figsize=(12, 6))
    
    # Plot distributions
    sns.kdeplot(data=positive_returns, label='Signal = 1', color='green', alpha=0.6)
    sns.kdeplot(data=negative_returns, label='Signal = 0', color='red', alpha=0.6)
    
    # Add vertical lines for means
    plt.axvline(positive_returns.mean(), color='green', linestyle='--', alpha=0.8)
    plt.axvline(negative_returns.mean(), color='red', linestyle='--', alpha=0.8)
    
    # Add title and labels
    plt.title(f'{signal_name} - {period}-Day Forward Return Distributions')
    plt.xlabel('Forward Returns')
    plt.ylabel('Density')
    plt.legend()
    
    # Add statistical annotations
    stats_text = f"""
    Signal=1 Mean: {positive_returns.mean():.4f}
    Signal=0 Mean: {negative_returns.mean():.4f}
    Signal=1 Std: {positive_returns.std():.4f}
    Signal=0 Std: {negative_returns.std():.4f}
    Signal=1 Skew: {stats.skew(positive_returns):.4f}
    Signal=0 Skew: {stats.skew(negative_returns):.4f}
    Signal=1 Kurt: {stats.kurtosis(positive_returns):.4f}
    Signal=0 Kurt: {stats.kurtosis(negative_returns):.4f}
    Signal=1 Count: {len(positive_returns)}
    Signal=0 Count: {len(negative_returns)}
    """
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def analyze_binary_signal(evaluator, signal, data, asset_id='ethereum'):
    """Analyze a binary signal using the fixed evaluator.
    
    Args:
        evaluator: The fixed BinarySignalEvaluator
        signal: Signal instance to analyze
        data: Asset price data
        asset_id: Asset identifier
    """
    print(f"\nAnalyzing {signal.name}...")
    
    # Evaluate signal
    evaluation = evaluator.evaluate_signal(signal, data, asset_id)
    
    if not evaluation['valid']:
        print(f"Invalid evaluation: {evaluation.get('reason', evaluation.get('error', 'Unknown reason'))}")
        return evaluation
    
    # Print evaluation metrics
    print(f"\nEvaluation results for {signal.name}:")
    print(f"Overall effectiveness: {evaluation['overall_effectiveness']}")
    print(f"Weight: {evaluation['weight']:.4f}")
    
    if 'metrics' in evaluation:
        metrics = evaluation['metrics']
        print("\nMetrics:")
        print(f"Effective periods: {metrics['effective_periods']}/{metrics['total_periods']}")
        print(f"Effectiveness ratio: {metrics['effectiveness_ratio']:.4f}")
        print(f"Average effect size: {metrics['avg_effect_size']:.4f}")
        print(f"Average confidence: {metrics['avg_confidence']:.4f}")
        print(f"Average mean difference: {metrics['avg_mean_diff']:.4f}")
        print(f"Direction: {'Bullish (positive)' if metrics.get('direction', 0) > 0 else 'Bearish (negative)'}")
    
    # Analyze each period
    for period, period_data in evaluation.get('effectiveness_by_period', {}).items():
        if 'test_results' in period_data:
            test_results = period_data['test_results']
            
            # Get conditional returns for this period
            signal_values = signal.calculate(data, asset_id)
            calculator = evaluator.calculator
            data_with_returns = calculator.calculate_forward_returns(data, 'close')
            conditional_returns, _ = calculator.get_conditional_returns(
                data_with_returns, signal_values
            )
            
            if period in conditional_returns:
                if 'positive' in conditional_returns[period] and 'negative' in conditional_returns[period]:
                    positive_returns = conditional_returns[period]['positive']
                    negative_returns = conditional_returns[period]['negative']
                    
                    # Plot distributions
                    plot_binary_return_distributions(
                        positive_returns, negative_returns, signal.name, period
                    )
                    
                    # Print test results
                    print(f"\nPeriod {period} test results:")
                    
                    if 't_test' in test_results:
                        t_test = test_results['t_test']
                        print(f"T-test (mean comparison):")
                        print(f"  Significant: {t_test.get('significant', False)}")
                        print(f"  P-value: {t_test.get('p_value', 'N/A'):.4f}")
                        print(f"  Mean difference: {t_test.get('mean_difference', 'N/A'):.4f}")
                        print(f"  Effect size: {t_test.get('effect_size', 'N/A'):.4f}")
                        print(f"  Direction: {'Positive' if t_test.get('mean_difference', 0) > 0 else 'Negative'}")
                    
                    if 'mann_whitney' in test_results:
                        mw_test = test_results['mann_whitney']
                        print(f"Mann-Whitney (distribution comparison):")
                        print(f"  Significant: {mw_test.get('significant', False)}")
                        print(f"  P-value: {mw_test.get('p_value', 'N/A'):.4f}")
                        print(f"  Median difference: {mw_test.get('median_difference', 'N/A'):.4f}")
                    
                    if 'ks_test' in test_results:
                        ks_test = test_results['ks_test']
                        print(f"Kolmogorov-Smirnov (distribution shape):")
                        print(f"  Significant: {ks_test.get('significant', False)}")
                        print(f"  P-value: {ks_test.get('p_value', 'N/A'):.4f}")
                        print(f"  Statistic: {ks_test.get('statistic', 'N/A'):.4f}")
                    
                    if 'skew_kurt' in test_results:
                        sk_test = test_results['skew_kurt']
                        print(f"Skewness & Kurtosis:")
                        print(f"  Signal=1 skew: {sk_test.get('positive_skew', 'N/A'):.4f}")
                        print(f"  Signal=0 skew: {sk_test.get('negative_skew', 'N/A'):.4f}")
                        
                        # Get and print raw kurtosis values for debugging
                        pos_kurt_raw = stats.kurtosis(positive_returns)
                        neg_kurt_raw = stats.kurtosis(negative_returns)
                        pos_kurt_fisher_false = stats.kurtosis(positive_returns, fisher=False)
                        neg_kurt_fisher_false = stats.kurtosis(negative_returns, fisher=False)
                        
                        print(f"  Signal=1 kurt: {sk_test.get('positive_kurtosis', 'N/A'):.4f}")
                        print(f"  Signal=0 kurt: {sk_test.get('negative_kurtosis', 'N/A'):.4f}")
                        print(f"  Raw calculation (Signal=1 kurt): {pos_kurt_raw}")
                        print(f"  Raw calculation (Signal=0 kurt): {neg_kurt_raw}")
                        print(f"  Raw with fisher=False (Signal=1 kurt): {pos_kurt_fisher_false}")
                        print(f"  Raw with fisher=False (Signal=0 kurt): {neg_kurt_fisher_false}")
                    
                    # Get effectiveness for this period
                    effectiveness = period_data.get('effectiveness', {})
                    print(f"\nEffectiveness Summary for {period}-day period:")
                    print(f"Overall effective: {effectiveness.get('overall_effective', False)}")
                    print(f"Confidence: {effectiveness.get('confidence', 0.0):.4f}")
                    print(f"Effect size: {effectiveness.get('effect_size', 0.0):.4f}")
                    print(f"Direction: {'Bullish' if effectiveness.get('direction', 0) > 0 else 'Bearish'}")
    
    return evaluation


def run_analysis(analyzer, asset_id='ethereum', min_samples=20, lookback_days=1000):
    """Run analysis on signals for a specific asset.
    
    Args:
        analyzer: The TrendAnalyzer instance
        asset_id: Asset to analyze
        min_samples: Minimum samples required for each signal state
        lookback_days: Number of days to look back for analysis
    """
    # Set up logging
    setup_logging()
    
    # Get data for the asset
    data = analyzer.get_asset_raw_data(asset_id)
    print(f"\n{asset_id.capitalize()} data available from {min(data['date'])} to {max(data['date'])}")
    print(f"Total data points: {len(data)}")
    
    # Create signals to test - use a more limited set for initial testing
    signals = [
        # Trend Signals - USD Quote
        ShortTermStrongBullUSD(),
        ShortTermWeakBullUSD(),
        ShortTermNeutralUSD(),
        ShortTermWeakBearUSD(),
        ShortTermStrongBearUSD(),
        
        # RSI Signals - USD Quote
        RSI_Oversold_USD(params={'rsi_length': 14, 'oversold': 30}),
        RSI_Overbought_USD(params={'rsi_length': 14, 'overbought': 70}),
        
        # Market Regime Signals
        SSR_RiskOn(params={'data_path': analyzer.ssr_data_path, 'handle_missing': 'zero'}),
        SSR_RiskOff(params={'data_path': analyzer.ssr_data_path, 'handle_missing': 'zero'}),
        
        # Volatility Regime Signals
        MarkovLowVolatilitySignal(),
        MarkovHighVolatilitySignal()
    ]
    
    # Create the fixed evaluator
    evaluator = BinarySignalEvaluator(
        periods=[1, 3, 5, 7, 14, 21, 28],
        min_samples=min_samples,
        lookback_days=lookback_days
    )
    
    # Analyze each signal
    evaluations = {}
    for signal in signals:
        evaluation = analyze_binary_signal(evaluator, signal, data, asset_id)
        evaluations[signal.name] = evaluation
        plt.show()  # Display distribution plots
    
    # Create a summary DataFrame of all evaluations
    summary_data = []
    for signal_name, eval_result in evaluations.items():
        if eval_result['valid']:
            metrics = eval_result.get('metrics', {})
            summary_data.append({
                'Signal': signal_name,
                'Effective': eval_result['overall_effectiveness'],
                'Weight': eval_result['weight'],
                'Effect Size': metrics.get('avg_effect_size', np.nan),
                'Confidence': metrics.get('avg_confidence', np.nan),
                'Mean Diff': metrics.get('avg_mean_diff', np.nan),
                'Effectiveness Ratio': metrics.get('effectiveness_ratio', np.nan),
                'Effective Periods': f"{metrics.get('effective_periods', 0)}/{metrics.get('total_periods', 0)}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Weight', ascending=False)
        print("\nSignal Evaluation Summary:")
        print(summary_df)
        return summary_df
    else:
        print("\nNo valid evaluations found.")
        return None


if __name__ == "__main__":
    # Set paths and initialize analyzer
    data_path='/Users/valter.rebelo/MissionControl/data/micro/candleData/'
    btc_data_path='/Users/valter.rebelo/MissionControl/data/micro/candleData/bitcoin_candles.csv'
    ssr_data_path='/Users/valter.rebelo/MissionControl/data/onchainData/BTC_SSR.csv'
    market_data_path="/Users/valter.rebelo/MissionControl/data/micro/assetData/"
    
    try:
        markov_vol_model = MarkovVolModel.load_model_by_timestamp('20250325_144631')
    except:
        print("Could not load Markov model, proceeding without it")
        markov_vol_model = None
    
    # Initialize analyzer with test assets
    test_assets = ['bitcoin', 'ethereum', 'solana']
    print(f"Testing with assets: {test_assets}")
    
    analyzer = TrendAnalyzer(
        asset_ids=test_assets,
        data_path=data_path,
        btc_data_path=btc_data_path,
        lookback_days='all',
        ssr_data_path=ssr_data_path,        
        markov_analyzer=markov_vol_model,
        market_data_path=market_data_path  
    )
    
    # Make sure the data is loaded
    analyzer.analyze_multiple_assets()
    
    # Run analysis with relaxed parameters
    run_analysis(analyzer, asset_id='ethereum', min_samples=20, lookback_days=1000) 