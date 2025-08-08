"""
Validation Utilities for Time Series Models
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
from .evaluation_metrics import calculate_sharpe_ratio, calculate_profit_factor, calculate_information_ratio


def create_ohlc_permutation(ohlc_df: pd.DataFrame, start_index: int = 0, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Create permuted OHLC data that preserves statistical properties while breaking temporal patterns.
    
    This implements the sophisticated OHLC permutation approach that:
    - Separates gaps (close-to-open) from intrabar moves (high-open, low-open, close-open)
    - Shuffles each component independently to break temporal patterns
    - Reconstructs coherent OHLC bars that maintain statistical properties
    - Preserves return distribution (mean, std, skew, kurtosis)
    
    Args:
        ohlc_df: DataFrame with OHLC columns ['open', 'high', 'low', 'close']
        start_index: Index to start permutation from (preserves early data)
        seed: Random seed for reproducibility
    
    Returns:
        Permuted OHLC DataFrame with same statistical properties
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in ohlc_df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    time_index = ohlc_df.index
    n_bars = len(ohlc_df)
    
    if start_index >= n_bars - 1:
        return ohlc_df.copy()
    
    perm_index = start_index + 1
    perm_n = n_bars - perm_index
    
    if perm_n < 2:
        return ohlc_df.copy()
    
    # Convert to log space
    log_bars = np.log(ohlc_df[required_cols])
    
    # Get start bar (the bar at start_index)
    start_bar = log_bars.iloc[start_index].to_numpy()
    
    # Calculate relative moves
    # Open relative to last close (gaps)
    r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
    
    # Intrabar moves relative to open
    r_h = (log_bars['high'] - log_bars['open']).to_numpy()
    r_l = (log_bars['low'] - log_bars['open']).to_numpy()
    r_c = (log_bars['close'] - log_bars['open']).to_numpy()
    
    # Extract the portions to be permuted
    relative_open = r_o[perm_index:]
    relative_high = r_h[perm_index:]
    relative_low = r_l[perm_index:]
    relative_close = r_c[perm_index:]
    
    # Create indices for shuffling
    idx = np.arange(perm_n)
    
    # Shuffle intrabar relative values (high/low/close) together
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[perm1]
    relative_low = relative_low[perm1]
    relative_close = relative_close[perm1]
    
    # Shuffle gaps (close-to-open) separately
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[perm2]
    
    # Create permuted OHLC bars
    perm_bars = np.zeros((n_bars, 4))
    
    # Copy original data before start_index
    perm_bars[:start_index] = log_bars.iloc[:start_index].to_numpy()
    
    # Set the start bar
    perm_bars[start_index] = start_bar
    
    # Reconstruct permuted bars
    for i in range(perm_index, n_bars):
        k = i - perm_index
        # Open = previous close + gap
        perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[k]
        # High = open + relative high
        perm_bars[i, 1] = perm_bars[i, 0] + relative_high[k]
        # Low = open + relative low  
        perm_bars[i, 2] = perm_bars[i, 0] + relative_low[k]
        # Close = open + relative close
        perm_bars[i, 3] = perm_bars[i, 0] + relative_close[k]
    
    # Convert back to price space
    perm_bars = np.exp(perm_bars)
    
    # Create DataFrame with same structure
    perm_df = pd.DataFrame(perm_bars, index=time_index, columns=required_cols)
    
    return perm_df


def validate_ohlc_permutation_quality(original_ohlc: pd.DataFrame, 
                                     permuted_ohlc: pd.DataFrame,
                                     tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Validate that permuted OHLC data preserves statistical properties of the original.
    
    Args:
        original_ohlc: Original OHLC DataFrame with columns ['open', 'high', 'low', 'close']
        permuted_ohlc: Permuted OHLC DataFrame 
        tolerance: Relative tolerance for statistical property preservation (default 10%)
    
    Returns:
        Dictionary with validation results
    """
    from scipy.stats import ks_2samp
    
    # Calculate returns for both series
    orig_returns = np.log(original_ohlc['close'] / original_ohlc['close'].shift(1)).dropna()
    perm_returns = np.log(permuted_ohlc['close'] / permuted_ohlc['close'].shift(1)).dropna()
    
    # Statistical moments comparison
    orig_stats = {
        'mean': np.mean(orig_returns),
        'std': np.std(orig_returns),
        'skew': float(orig_returns.skew()) if hasattr(orig_returns, 'skew') else 0.0,
        'kurtosis': float(orig_returns.kurtosis()) if hasattr(orig_returns, 'kurtosis') else 0.0
    }
    
    perm_stats = {
        'mean': np.mean(perm_returns),
        'std': np.std(perm_returns),
        'skew': float(perm_returns.skew()) if hasattr(perm_returns, 'skew') else 0.0,
        'kurtosis': float(perm_returns.kurtosis()) if hasattr(perm_returns, 'kurtosis') else 0.0
    }
    
    # Calculate relative differences
    stat_differences = {}
    stat_quality = {}
    for stat in ['mean', 'std', 'skew', 'kurtosis']:
        if orig_stats[stat] != 0:
            diff = abs((perm_stats[stat] - orig_stats[stat]) / orig_stats[stat])
        else:
            diff = abs(perm_stats[stat])
        stat_differences[stat] = diff
        stat_quality[stat] = diff <= tolerance
    
    # Kolmogorov-Smirnov test for distribution similarity
    ks_stat, ks_pvalue = ks_2samp(orig_returns, perm_returns)
    
    # OHLC coherence checks for permuted data
    coherence_checks = {
        'high_ge_open': np.all(permuted_ohlc['high'] >= permuted_ohlc['open']),
        'high_ge_close': np.all(permuted_ohlc['high'] >= permuted_ohlc['close']),
        'low_le_open': np.all(permuted_ohlc['low'] <= permuted_ohlc['open']),
        'low_le_close': np.all(permuted_ohlc['low'] <= permuted_ohlc['close']),
        'valid_ohlc_bars': True
    }
    coherence_checks['valid_ohlc_bars'] = all(coherence_checks.values())
    
    return {
        'original_stats': orig_stats,
        'permuted_stats': perm_stats,
        'stat_differences': stat_differences,
        'stat_quality': stat_quality,
        'overall_quality': all(stat_quality.values()),
        'ks_test': {
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'distributions_similar': ks_pvalue > 0.05
        },
        'ohlc_coherence': coherence_checks,
        'quality_score': sum(stat_quality.values()) / len(stat_quality)
    }


def in_sample_permutation_test(features_df: pd.DataFrame,
                             strategy_func: Callable,
                             strategy_params: Dict[str, Any],
                             price_col: str,
                             log_return_col: str,
                             train_start: str,
                             train_end: str,
                             n_permutations: int = 1000,
                             metrics: list = ['profit_factor', 'sharpe_ratio', 'information_ratio'],
                             random_seed: Optional[int] = 42,
                             return_permuted_returns: bool = False) -> Dict[str, Any]:
    """
    Test strategy performance against permuted OHLC series that maintain statistical properties.
    
    Args:
        features_df: DataFrame with OHLC price data and features
        strategy_func: Function that calculates strategy (e.g., calculate_ema_crossover_strategy)
        strategy_params: Parameters for the strategy function
        price_col: Column name for price data (typically close price)
        log_return_col: Column name for log returns
        train_start: Start date for evaluation period
        train_end: End date for evaluation period
        n_permutations: Number of permutation tests to run
        metrics: List of metrics to evaluate ['profit_factor', 'sharpe_ratio', 'information_ratio']
        random_seed: Random seed for reproducibility
        return_permuted_returns: Whether to return all permuted strategy returns
    
    Returns:
        Dictionary with permutation test results for each metric
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Extract asset prefix from price_col (e.g., 'bitcoin_close' -> 'bitcoin_')
    if '_' in price_col:
        asset_prefix = price_col.rsplit('_', 1)[0] + '_'
    else:
        raise ValueError(f"Price column '{price_col}' should follow pattern 'asset_close'")
    
    # Define OHLC column names
    ohlc_cols = {
        'open': f"{asset_prefix}open",
        'high': f"{asset_prefix}high", 
        'low': f"{asset_prefix}low",
        'close': price_col  # This is already the close column
    }
    
    # Validate OHLC columns exist
    missing_cols = [col for col in ohlc_cols.values() if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Missing OHLC columns in features_df: {missing_cols}")
    
    # Calculate original strategy performance
    strategy_only_params = {k: v for k, v in strategy_params.items() 
                          if k not in ['price_col', 'log_return_col']}
    original_strategy_data = strategy_func(
        features_df, 
        **strategy_only_params,
        price_col=strategy_params['price_col'],
        log_return_col=strategy_params['log_return_col']
    )
    train_data = original_strategy_data.loc[train_start:train_end].dropna()
    
    if len(train_data) == 0:
        raise ValueError("No data available in the specified training period")
    
    original_strategy_returns = train_data['strategy_returns'].values
    original_benchmark_returns = train_data[log_return_col].values
    
    # Calculate original metrics
    original_metrics = {}
    for metric in metrics:
        if metric == 'profit_factor':
            original_metrics[metric] = calculate_profit_factor(original_strategy_returns)
        elif metric == 'sharpe_ratio':
            original_metrics[metric] = calculate_sharpe_ratio(original_strategy_returns)
        elif metric == 'information_ratio':
            original_metrics[metric] = calculate_information_ratio(original_strategy_returns, original_benchmark_returns)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    # Run permutation tests using OHLC permutation
    permuted_results = {metric: [] for metric in metrics}
    permuted_returns_list = [] if return_permuted_returns else None
    
    print(f"Running {n_permutations} OHLC permutation tests...")
    
    valid_permutations = 0
    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_permutations} permutations ({valid_permutations} valid)")
        
        try:
            # Extract OHLC data for permutation
            ohlc_df = features_df[list(ohlc_cols.values())].copy()
            ohlc_df.columns = ['open', 'high', 'low', 'close']  # Standardize column names
            
            # Create permuted OHLC data
            permuted_ohlc = create_ohlc_permutation(
                ohlc_df, 
                start_index=0,
                seed=random_seed + i if random_seed else None
            )
            
            # Create permuted features dataframe
            permuted_df = features_df.copy()
            
            # Update all OHLC columns with permuted data
            permuted_df[ohlc_cols['open']] = permuted_ohlc['open']
            permuted_df[ohlc_cols['high']] = permuted_ohlc['high']
            permuted_df[ohlc_cols['low']] = permuted_ohlc['low']
            permuted_df[ohlc_cols['close']] = permuted_ohlc['close']
            
            # Recalculate log returns for permuted close prices
            permuted_df[log_return_col] = np.log(permuted_df[ohlc_cols['close']] / permuted_df[ohlc_cols['close']].shift(1))
            
            # Apply strategy to permuted data  
            permuted_strategy_data = strategy_func(
                permuted_df, 
                **strategy_only_params,
                price_col=strategy_params['price_col'],
                log_return_col=strategy_params['log_return_col']
            )
            permuted_train_data = permuted_strategy_data.loc[train_start:train_end].dropna()
            
            if len(permuted_train_data) == 0:
                continue
            
            permuted_strategy_returns = permuted_train_data['strategy_returns'].values
            permuted_benchmark_returns = permuted_train_data[log_return_col].values
            
            # Validate we have valid returns
            if len(permuted_strategy_returns) != len(original_strategy_returns):
                continue
            
            valid_permutations += 1
            
            # Store permuted returns if requested
            if return_permuted_returns:
                permuted_returns_list.append(permuted_strategy_returns.copy())
            
            # Calculate metrics for permuted data
            for metric in metrics:
                if metric == 'profit_factor':
                    value = calculate_profit_factor(permuted_strategy_returns)
                elif metric == 'sharpe_ratio':
                    value = calculate_sharpe_ratio(permuted_strategy_returns)
                elif metric == 'information_ratio':
                    value = calculate_information_ratio(permuted_strategy_returns, permuted_benchmark_returns)
                
                # Handle infinite or NaN values
                if np.isfinite(value):
                    permuted_results[metric].append(value)
        
        except Exception as e:
            # Skip failed permutations - add debug info for first few failures
            if i < 5:  # Only print first 5 errors to avoid spam
                print(f"    Warning: Permutation {i} failed: {str(e)}")
            continue
    
    print(f"Completed {valid_permutations} valid permutations out of {n_permutations} attempts")
    
    # Calculate p-values and statistics
    results = {}
    for metric in metrics:
        permuted_values = np.array(permuted_results[metric])
        original_value = original_metrics[metric]
        
        if len(permuted_values) == 0:
            p_value = 1.0
            percentile = 0.0
        else:
            p_value = np.mean(permuted_values >= original_value)
            percentile = (1 - p_value) * 100
        
        results[metric] = {
            'original_value': original_value,
            'permuted_values': permuted_values,
            'p_value': p_value,
            'percentile': percentile,
            'passes_test': p_value < 0.05,
            'n_permutations': len(permuted_values),
            'permuted_mean': np.mean(permuted_values) if len(permuted_values) > 0 else 0.0,
            'permuted_std': np.std(permuted_values) if len(permuted_values) > 0 else 0.0
        }
    
    # Add permuted returns if requested
    if return_permuted_returns:
        results['permuted_returns'] = permuted_returns_list
        results['original_returns'] = original_strategy_returns
    
    return results