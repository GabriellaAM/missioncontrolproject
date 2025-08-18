"""
Validation Utilities for Time Series Models
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
from .evaluation_metrics import calculate_sharpe_ratio, calculate_profit_factor, calculate_information_ratio, calculate_max_drawdown


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


def walk_forward_validation(features_df: pd.DataFrame,
                           strategy_func: Callable,
                           optimize_func: Callable,
                           train_test_split: float = 0.75,
                           n_folds: int = 12,
                           reoptimize_every: int = 3,
                           embargo_days: int = 7) -> Dict:
    """
    Perform walk-forward validation with automatic configuration based on train/test split.
    
    Args:
        features_df: Full dataset with features
        strategy_func: Function to calculate signals (signature: func(data, params) -> DataFrame)
        optimize_func: Function to optimize parameters (signature: func(data, train_start, train_end) -> Dict)
        train_test_split: Ratio of data to use for initial training (e.g., 0.75)
        n_folds: Number of folds to create in the test period (default: 12)
        reoptimize_every: Re-optimize every N folds (default: 3)
        embargo_days: Days to embargo after training to avoid label leakage (default: 7)
                     This accounts for triple barrier time horizon plus buffer
    
    Returns:
        Dict with walk-forward results including metrics and fold details
    """
    
    # Calculate split point based on train_test_split
    total_days = len(features_df)
    train_days = int(total_days * train_test_split)
    test_days = total_days - train_days
    
    # Derive dates from split
    train_end_date = features_df.index[train_days - 1]
    test_start_date = features_df.index[train_days]
    test_end_date = features_df.index[-1]
    
    # Calculate fold size
    test_window_days = test_days // n_folds
    
    # Initial optimization on training data
    train_start = features_df.index[0].strftime('%Y-%m-%d')
    train_end = train_end_date.strftime('%Y-%m-%d')
    
    initial_opt = optimize_func(features_df, train_start, train_end)
    current_params = initial_opt.get('best_params', {})
    
    # Initialize results
    wf_results = {
        'fold_dates': [],
        'fold_params': [],
        'fold_returns': [],
        'fold_metrics': [],
        'all_returns': [],
        'all_signals': [],
        'reoptimization_dates': [],
        'train_test_split': train_test_split,
        'n_folds': n_folds,
        'overall_metrics': None
    }
    
    # Walk-forward loop through test period
    current_idx = train_days
    fold_number = 0
    
    while current_idx < total_days:
        fold_number += 1
        
        # Define test window for this fold
        fold_start_idx = current_idx
        fold_end_idx = min(current_idx + test_window_days, total_days - 1)
        fold_start = features_df.index[fold_start_idx]
        fold_end = features_df.index[fold_end_idx]
        
        # Re-optimize if needed
        # For reoptimize_every=1: reoptimize before every fold after the first
        # For reoptimize_every=N: reoptimize before fold N+1, 2N+1, 3N+1, etc.
        should_reoptimize = False
        if fold_number > 1:  # Never reoptimize on first fold
            if reoptimize_every == 1:
                should_reoptimize = True  # Reoptimize before every fold after first
            elif (fold_number - 1) % reoptimize_every == 0:
                should_reoptimize = True  # Reoptimize every N folds
        
        if should_reoptimize:
            # Expanding window: use all data up to current fold minus embargo period
            # This ensures we don't use labels that would look into the test period
            embargo_idx = max(0, fold_start_idx - embargo_days - 1)
            retrain_end = features_df.index[embargo_idx].strftime('%Y-%m-%d')
            opt_result = optimize_func(features_df, train_start, retrain_end)
            current_params = opt_result.get('best_params', current_params)
            wf_results['reoptimization_dates'].append(fold_start)
        
        # Apply strategy to this fold
        strategy_data = strategy_func(features_df, **current_params)
        fold_data = strategy_data.iloc[fold_start_idx:fold_end_idx + 1].dropna()
        
        if len(fold_data) > 0:
            # Extract returns and signals
            fold_returns = fold_data.get('strategy_returns', pd.Series()).values
            fold_signals = fold_data.get('signal', pd.Series()).values
            
            # Store results
            wf_results['fold_dates'].append((fold_start, fold_end))
            wf_results['fold_params'].append(current_params.copy())
            wf_results['fold_returns'].append(fold_returns)
            wf_results['all_returns'].extend(fold_returns)
            wf_results['all_signals'].extend(fold_signals)
            
            # Calculate metrics for this fold
            if len(fold_returns) > 1:
                fold_metric = {
                    'fold': fold_number,
                    'sharpe': calculate_sharpe_ratio(fold_returns),
                    'profit_factor': calculate_profit_factor(fold_returns),
                    'total_return': (1 + fold_returns).prod() - 1,
                    'n_days': len(fold_returns)
                }
                wf_results['fold_metrics'].append(fold_metric)
        
        # Move to next fold
        current_idx = fold_end_idx + 1
    
    # Calculate overall metrics
    if len(wf_results['all_returns']) > 0:
        all_returns = np.array(wf_results['all_returns'])
        wf_results['overall_metrics'] = {
            'sharpe': calculate_sharpe_ratio(all_returns),
            'profit_factor': calculate_profit_factor(all_returns),
            'max_drawdown': calculate_max_drawdown(all_returns),
            'total_return': (1 + all_returns).prod() - 1,
            'n_days': len(all_returns),
            'n_folds': len(wf_results['fold_metrics'])
        }
    
    return wf_results


def walk_forward_permutation_test(features_df: pd.DataFrame,
                                 strategy_func: Callable,
                                 optimize_func: Callable,
                                 wf_results: Dict,
                                 n_permutations: int = 100,
                                 p_value_threshold: float = 0.05) -> Dict:
    """
    Test if walk-forward results are statistically significant using permutation.
    
    Args:
        features_df: Full dataset with features (could be strategy_data with signals)
        strategy_func: Function to calculate signals
        optimize_func: Function to optimize parameters
        wf_results: Results from walk_forward_validation to compare against
        n_permutations: Number of permutations to test
        p_value_threshold: Significance threshold
    
    Returns:
        Dict with permutation test results
    """
    
    if 'overall_metrics' not in wf_results or wf_results['overall_metrics'] is None:
        raise ValueError("No walk-forward results to test against")
    
    # Get original metrics
    original_sharpe = wf_results['overall_metrics']['sharpe']
    original_pf = wf_results['overall_metrics']['profit_factor']
    
    print(f"Original walk-forward metrics - Sharpe: {original_sharpe:.3f}, PF: {original_pf:.3f}")
    
    # Get configuration from original results
    train_test_split = wf_results.get('train_test_split', 0.75)
    n_folds = wf_results.get('n_folds', 12)
    
    # Use fewer folds for speed, but not too few
    perm_n_folds = max(3, n_folds // 3)  # At least 3 folds, but reduce for speed
    
    # Run permutations
    permuted_sharpes = []
    permuted_pfs = []
    
    print(f"Running {n_permutations} walk-forward permutation tests...")
    print(f"Using {perm_n_folds} folds per permutation (original: {n_folds})")
    
    for i in range(n_permutations):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  🎲 Running permutation {i + 1}/{n_permutations}...")
        
        # Extract asset prefix from column names
        price_cols = [col for col in features_df.columns if any(x in col for x in ['_close', '_open', '_high', '_low'])]
        
        if not price_cols:
            print("Warning: No price columns found for permutation")
            continue
            
        # Get asset prefix (e.g., 'bitcoin' from 'bitcoin_close')
        asset_prefix = price_cols[0].rsplit('_', 1)[0] if '_' in price_cols[0] else ''
        
        # Define OHLC column names
        ohlc_mapping = {
            'open': f'{asset_prefix}_open',
            'high': f'{asset_prefix}_high',
            'low': f'{asset_prefix}_low',
            'close': f'{asset_prefix}_close'
        }
        
        # Check if all OHLC columns exist
        if not all(col in features_df.columns for col in ohlc_mapping.values()):
            print(f"Warning: Missing OHLC columns for permutation. Found: {[c for c in ohlc_mapping.values() if c in features_df.columns]}")
            continue
        
        # Extract OHLC data
        ohlc_df = features_df[list(ohlc_mapping.values())].copy()
        ohlc_df.columns = ['open', 'high', 'low', 'close']
        
        # Create permuted OHLC
        permuted_ohlc = create_ohlc_permutation(ohlc_df, seed=42 + i)
        
        # Create permuted features dataframe
        permuted_df = features_df.copy()
        
        # Replace OHLC columns with permuted values
        for std_col, full_col in ohlc_mapping.items():
            permuted_df[full_col] = permuted_ohlc[std_col].values
        
        # Recalculate log returns from permuted close prices
        log_return_col = f'{asset_prefix}_log_return_1'
        if log_return_col in permuted_df.columns:
            permuted_df[log_return_col] = np.log(
                permuted_df[ohlc_mapping['close']] / permuted_df[ohlc_mapping['close']].shift(1)
            )
        
        # Run walk-forward on permuted data
        try:
            perm_wf = walk_forward_validation(
                permuted_df,
                strategy_func,
                optimize_func,
                train_test_split=train_test_split,
                n_folds=perm_n_folds,
                reoptimize_every=999  # No re-optimization for speed
            )
            
            if perm_wf['overall_metrics']:
                perm_sharpe = perm_wf['overall_metrics']['sharpe']
                perm_pf = perm_wf['overall_metrics']['profit_factor']
                
                # Only add finite values
                if np.isfinite(perm_sharpe):
                    permuted_sharpes.append(perm_sharpe)
                if np.isfinite(perm_pf):
                    permuted_pfs.append(perm_pf)
                
                # Debug output for first few permutations
                if i < 3:
                    print(f"    Permutation {i+1}: Sharpe={perm_sharpe:.3f}, PF={perm_pf:.3f}")
        except Exception as e:
            if i < 3:  # Only show first few errors
                print(f"    Warning: Permutation {i+1} failed: {str(e)}")
            continue
    
    print(f"✅ Completed {len(permuted_sharpes)} valid walk-forward permutations")
    
    # Calculate p-values
    permuted_sharpes = np.array([s for s in permuted_sharpes if np.isfinite(s)])
    permuted_pfs = np.array([pf for pf in permuted_pfs if np.isfinite(pf)])
    
    # Debug: Show distribution statistics
    if len(permuted_sharpes) > 0:
        print(f"Permuted Sharpe distribution: mean={np.mean(permuted_sharpes):.3f}, "
              f"std={np.std(permuted_sharpes):.3f}, min={np.min(permuted_sharpes):.3f}, "
              f"max={np.max(permuted_sharpes):.3f}")
        print(f"Original Sharpe ({original_sharpe:.3f}) vs Permuted mean ({np.mean(permuted_sharpes):.3f})")
    
    if len(permuted_pfs) > 0:
        print(f"Permuted PF distribution: mean={np.mean(permuted_pfs):.3f}, "
              f"std={np.std(permuted_pfs):.3f}, min={np.min(permuted_pfs):.3f}, "
              f"max={np.max(permuted_pfs):.3f}")
        print(f"Original PF ({original_pf:.3f}) vs Permuted mean ({np.mean(permuted_pfs):.3f})")
    
    sharpe_p_value = np.mean(permuted_sharpes >= original_sharpe) if len(permuted_sharpes) > 0 else 1.0
    pf_p_value = np.mean(permuted_pfs >= original_pf) if len(permuted_pfs) > 0 else 1.0
    
    return {
        'sharpe': {
            'original': original_sharpe,
            'permuted_values': permuted_sharpes,  # Store full distribution for histogram
            'permuted_mean': np.mean(permuted_sharpes) if len(permuted_sharpes) > 0 else 0,
            'permuted_std': np.std(permuted_sharpes) if len(permuted_sharpes) > 0 else 0,
            'p_value': sharpe_p_value,
            'passes_test': sharpe_p_value < p_value_threshold,
            'n_permutations': len(permuted_sharpes)
        },
        'profit_factor': {
            'original': original_pf,
            'permuted_values': permuted_pfs,  # Store full distribution for histogram
            'permuted_mean': np.mean(permuted_pfs) if len(permuted_pfs) > 0 else 0,
            'permuted_std': np.std(permuted_pfs) if len(permuted_pfs) > 0 else 0,
            'p_value': pf_p_value,
            'passes_test': pf_p_value < p_value_threshold,
            'n_permutations': len(permuted_pfs)
        }
    }