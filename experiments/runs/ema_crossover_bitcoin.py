# %%

########################################################
# Imports
########################################################

from config import setup_mlflow, create_experiment
import mlflow
import sys
import os
import pandas as pd
import numpy as np
import optuna
import logging
from typing import Dict, List
from utils.evaluation_metrics import calculate_profit_factor, calculate_sharpe_ratio, calculate_max_drawdown, calculate_sortino_ratio, calculate_calmar_ratio, calculate_information_ratio
from utils.validation import in_sample_permutation_test
from utils.feature_loader import FeatureLoader
from utils.feature_engineering import calculate_log_returns
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%

########################################################
# Setup
########################################################


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup MLflow
setup_mlflow()
create_experiment("bitcoin-models")
mlflow.set_experiment("bitcoin-models")

# %%

########################################################
# Load Features
########################################################

# You can change this to any asset(s) supported by your FeatureLoader
assets = ['bitcoin']  # e.g., ['ethereum'], ['aapl'], ['gold'], etc.

# Helper: get the asset prefix for column names (assume first asset for single-asset analysis)
asset = assets[0]
prefix = f"{asset}_"  # e.g., 'bitcoin_', 'ethereum_', etc.

# Initialize feature loader with global date range
loader = FeatureLoader(start_date='2020-01-01', end_date='2024-12-31')

# Build feature set with selected asset(s) + macro features
features_df = loader.build_feature_set(
    crypto_assets=assets,
    #fred_indicators={'creditSpreads': 'credit_spread', 'treasury5YInflationExpectation': 'treasury_5y_inflation_expectation'},
    #yahoo_tickers={'vix': 'vix'},
    #calculated_features={'rty_ym_ratio': 'rty_ym_ratio'}
)

print(f"Loaded {len(features_df)} rows of combined data")
print("\nColumns:", features_df.columns.tolist())
print("\nSample data:")
print(features_df.head(20))

# Set timestamp as index for time-based operations
if 'timestamp' in features_df.columns:
    features_df = features_df.set_index('timestamp')
    print(f"\nSet timestamp as index. Date range: {features_df.index.min()} to {features_df.index.max()}")


# %%

########################################################
# Feature Engineering Functions
########################################################

features_df = calculate_log_returns(features_df, periods=[1])
print("Available columns after log returns:", features_df.columns.tolist())
print("\nLog return columns:", [col for col in features_df.columns if 'log_return' in col])
features_df.head(20)

# %%

########################################################
# Strategy Definition and Parameter Optimization
########################################################

def calculate_ema_crossover_strategy(data, fast_period, slow_period, price_col, log_return_col):
    """
    Calculate EMA crossover signals and returns.
    
    Args:
        data: DataFrame with price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        price_col: Column name for price data
        log_return_col: Column name for log returns (explicit, no guessing)
    
    Returns:
        DataFrame with signals and strategy returns
    """
    df = data.copy()
    
    # Validate required columns exist
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataframe columns: {df.columns.tolist()}")
    if log_return_col not in df.columns:
        raise ValueError(f"Log return column '{log_return_col}' not found in dataframe columns: {df.columns.tolist()}")
    
    # Calculate EMAs
    df['ema_fast'] = df[price_col].ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = df[price_col].ewm(span=slow_period, adjust=False).mean()
    
    # Generate signals
    signals = pd.Series(0, index=df.index)
    signals[df['ema_fast'] >= df['ema_slow']] = 1
    signals[df['ema_fast'] < df['ema_slow']] = 0
    
    # Shift signals forward by 1 to apply at next day's open
    df['signal'] = signals.shift(1).fillna(0)
    
    # Calculate strategy returns
    df['strategy_returns'] = df[log_return_col] * df['signal']
    
    return df


def optimize_ema_with_optuna(
    data, train_start, train_end, price_col, log_return_col, n_trials=1000, random_seed=42
):
    """
    Optimize EMA parameters using Optuna.
    
    Args:
        data: DataFrame with price data and log returns
        train_start: Start date for training period
        train_end: End date for training period
        price_col: Column name for price data
        log_return_col: Column name for log returns
        n_trials: Number of optimization trials
        random_seed: Random seed for reproducibility
    
    Returns:
        optuna.Study object with optimization results
    """
    import random

    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    try:
        import torch
        torch.manual_seed(random_seed)
    except ImportError:
        pass

    def objective(trial):
        # Suggest parameters with constraints
        fast_period = trial.suggest_int('fast_period', 5, 20)
        slow_period = trial.suggest_int('slow_period', 21, 100)
        
        try:
            # Calculate strategy
            strategy_data = calculate_ema_crossover_strategy(
                data, fast_period, slow_period, price_col, log_return_col
            )
            
            # Evaluate on training period
            train_data = strategy_data.loc[train_start:train_end].dropna()
            
            # Minimum data requirement
            if len(train_data) < 100:
                return float('-inf')
            
            # Calculate profit factor
            strategy_returns = train_data['strategy_returns'].values
            
            # Ensure we have both positive and negative returns
            if np.all(strategy_returns >= 0) or np.all(strategy_returns <= 0):
                return float('-inf')
            
            profit_factor = calculate_profit_factor(strategy_returns)
            
            # Handle edge cases
            if np.isnan(profit_factor) or np.isinf(profit_factor):
                return float('-inf')
            
            return profit_factor
            
        except Exception as e:
            # Silent fail for invalid parameter combinations
            return float('-inf')
    
    # Create study with random seed
    study = optuna.create_study(
        direction='maximize',
        study_name=f"EMA_Optimization_{train_start}_{train_end}",
        sampler=optuna.samplers.TPESampler(seed=random_seed)
    )
    
    # Optimize
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=True,
        n_jobs=4  # Single-threaded for now
    )
    
    return study



# %%

########################################################
# Model Training and In-Sample Evaluation 
########################################################

# Define training period
train_start = '2020-01-01'
train_end = '2023-12-31'

# Define standard column names
price_col = f"{prefix}close"
log_return_col = f"{prefix}log_return_1"


# Check data in training period
train_period_data = features_df.loc[train_start:train_end]

print(f"\nTraining period: {train_start} to {train_end}")
print(f"Rows in training period: {len(train_period_data)}")

# Run Optuna optimization
print("\nStarting Optuna optimization...")
print("Objective: Maximize profit factor")
print("Parameter ranges: Fast EMA [5-20], Slow EMA [21-50]")

study = optimize_ema_with_optuna(
    features_df,
    train_start,
    train_end,
    price_col,
    log_return_col,
    n_trials=1000
)

# Extract best parameters
best_params = study.best_params
best_value = study.best_value

print(f"\nOptimization complete!")

# Apply best parameters to generate final signals
features_df = calculate_ema_crossover_strategy(
    features_df, 
    best_params['fast_period'], 
    best_params['slow_period'], 
    price_col,
    log_return_col
)

# Calculate in-sample metrics with optimized parameters
train_data = features_df.loc[train_start:train_end].dropna()
strategy_returns = train_data['strategy_returns'].values
benchmark_returns = train_data[f"{prefix}log_return_1"].values

# Calculate all metrics
profit_factor = calculate_profit_factor(strategy_returns)
sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
max_drawdown = calculate_max_drawdown(strategy_returns)
sortino_ratio = calculate_sortino_ratio(strategy_returns)
calmar_ratio = calculate_calmar_ratio(strategy_returns)
information_ratio = calculate_information_ratio(strategy_returns, benchmark_returns)
total_return = (1 + strategy_returns).prod() - 1
annualized_return = np.mean(strategy_returns) * 365


print(f"\n=== In-Sample Evaluation Results ===")
print(f"Optimized EMA Crossover ({best_params['fast_period']}/{best_params['slow_period']})")
print(f"Training period: {train_start} to {train_end}")
print(f"Number of trading days: {len(train_data)}")
print(f"\nPerformance Metrics:")
print(f"Profit Factor: {profit_factor:.3f}")
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
print(f"Sortino Ratio: {sortino_ratio:.3f}")
print(f"Calmar Ratio: {calmar_ratio:.3f}")
print(f"Information Ratio: {information_ratio:.3f}")
print(f"Max Drawdown: {max_drawdown:.1%}")
print(f"Total Return: {total_return:.1%}")
print(f"Annualized Return: {annualized_return:.1%}")

# %%

def plot_cumulative_returns(returns, benchmark_returns=None, title="Strategy vs Benchmark Cumulative Returns"):
    """
    Plots the cumulative returns of a strategy and optionally a benchmark.

    Args:
        returns (np.ndarray or pd.Series): Array or Series of strategy returns.
        benchmark_returns (np.ndarray or pd.Series, optional): Array or Series of benchmark returns.
        title (str): Title for the plot.
    """
    import matplotlib.pyplot as plt

    if isinstance(returns, np.ndarray):
        cum_returns = np.cumprod(1 + returns) - 1
    else:
        cum_returns = (1 + returns).cumprod() - 1

    plt.figure(figsize=(10, 5))
    plt.plot(cum_returns, label="Strategy Cumulative Return")

    if benchmark_returns is not None:
        if isinstance(benchmark_returns, np.ndarray):
            cum_benchmark = np.cumprod(1 + benchmark_returns) - 1
        else:
            cum_benchmark = (1 + benchmark_returns).cumprod() - 1
        plt.plot(cum_benchmark, label="Benchmark Cumulative Return", linestyle="--")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_cumulative_returns(strategy_returns, benchmark_returns=benchmark_returns)


# %%

########################################################
# In-Sample Permutation Test
########################################################

# Run permutation test to validate if the strategy exploits genuine patterns
# or just benefited from data mining bias
print("\n" + "="*60)
print("RUNNING IN-SAMPLE PERMUTATION TEST")
print("="*60)
print("Testing if the optimized EMA crossover strategy beats random...")
print("This test creates permuted price series with the same statistical properties")
print("but shuffled temporal patterns to test for data mining bias.")

# Prepare strategy parameters for permutation test
strategy_params = {
    'fast_period': best_params['fast_period'],
    'slow_period': best_params['slow_period'],
    'price_col': price_col,
    'log_return_col': log_return_col
}

# Run the permutation test
permutation_results = in_sample_permutation_test(
    features_df=features_df,
    strategy_func=calculate_ema_crossover_strategy,
    strategy_params=strategy_params,
    price_col=price_col,
    log_return_col=log_return_col,
    train_start=train_start,
    train_end=train_end,
    n_permutations=1000,
    metrics=['profit_factor', 'sharpe_ratio', 'information_ratio'],
    random_seed=42,
    return_permuted_returns=True
)

# Display results
print(f"\n=== PERMUTATION TEST RESULTS ===")
print(f"Strategy: EMA Crossover ({best_params['fast_period']}/{best_params['slow_period']})")
print(f"Test Period: {train_start} to {train_end}")
print(f"Number of Permutations: 1000")

for metric_name, results in permutation_results.items():
    # Skip non-metric keys
    if metric_name in ['permuted_returns', 'original_returns']:
        continue
        
    print(f"\n{metric_name.upper().replace('_', ' ')}:")
    print(f"  Original Value: {results['original_value']:.4f}")
    print(f"  Permuted Mean:  {results['permuted_mean']:.4f} ± {results['permuted_std']:.4f}")
    print(f"  P-value: {results['p_value']:.4f}")
    print(f"  Percentile: {results['percentile']:.1f}%")
    print(f"  Statistically Significant: {'YES' if results['passes_test'] else 'NO'}")
    print(f"  Valid Permutations: {results['n_permutations']}")

# Summary assessment
significant_metrics = [name for name, results in permutation_results.items() 
                      if name not in ['permuted_returns', 'original_returns'] and results['passes_test']]
print(f"\n=== OVERALL ASSESSMENT ===")

if len(significant_metrics) > 0:
    print(f"✅ STRATEGY PASSES PERMUTATION TEST")
    print(f"   Significant metrics: {', '.join(significant_metrics)}")
    print(f"   The strategy appears to exploit genuine market patterns.")
else:
    print(f"❌ STRATEGY FAILS PERMUTATION TEST")
    print(f"   No metrics show statistical significance (p < 0.05).")
    print(f"   Results may be due to data mining bias or random chance.")



print(f"\nPermutation test completed.")

# %%

########################################################
# Permutation Test Visualizations
########################################################

import matplotlib.pyplot as plt

def plot_permutation_cumulative_returns(original_returns, permuted_returns_list, title="Strategy vs Permuted Returns"):
    """
    Plot original strategy cumulative returns vs individual permuted return trajectories.
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate cumulative returns
    original_cumret = (1 + original_returns).cumprod() - 1
    
    # Plot each permuted trajectory with low opacity
    for i, perm_returns in enumerate(permuted_returns_list):
        if len(perm_returns) == len(original_returns):
            perm_cumret = (1 + perm_returns).cumprod() - 1
            
            # Plot with low alpha for transparency
            # Only label the first one for legend
            if i == 0:
                plt.plot(range(len(perm_cumret)), perm_cumret, 
                        color='gray', alpha=0.5, linewidth=0.5, 
                        label=f'Permuted ({len(permuted_returns_list)} trajectories)')
            else:
                plt.plot(range(len(perm_cumret)), perm_cumret, 
                        color='gray', alpha=0.5, linewidth=0.5)
    
    # Plot original strategy on top with bold color
    plt.plot(range(len(original_cumret)), original_cumret, 
            color='red', linewidth=2.5, label='Original Strategy', zorder=100)
    
    plt.title(title)
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_permutation_histogram(original_value, permuted_values, metric_name, p_value):
    """
    Plot histogram of permuted metric values with original value and p-value line.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of permuted values
    plt.hist(permuted_values, bins=50, alpha=0.7, color='lightblue', 
             edgecolor='black', density=True, label=f'Permuted {metric_name}')
    
    # Plot original value line
    plt.axvline(original_value, color='red', linestyle='-', linewidth=2, 
                label=f'Original {metric_name}: {original_value:.3f}')
    
    # Add p-value text
    plt.text(0.02, 0.98, f'P-value: {p_value:.4f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add percentile text
    percentile = (1 - p_value) * 100
    plt.text(0.02, 0.90, f'Percentile: {percentile:.1f}%', 
             transform=plt.gca().transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f'{metric_name.replace("_", " ").title()} Distribution - Permutation Test')
    plt.xlabel(metric_name.replace("_", " ").title())
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Generate visualizations
if 'permuted_returns' in permutation_results:
    print("\nGenerating permutation test visualizations...")
    
    # Plot cumulative returns comparison
    plot_permutation_cumulative_returns(
        permutation_results['original_returns'],
        permutation_results['permuted_returns'],
        title=f"EMA Crossover ({best_params['fast_period']}/{best_params['slow_period']}) vs Permuted Strategies"
    )
    
    # Plot histograms for each metric
    for metric_name, results in permutation_results.items():
        if metric_name in ['permuted_returns', 'original_returns']:
            continue
        
        plot_permutation_histogram(
            results['original_value'],
            results['permuted_values'],
            metric_name,
            results['p_value']
        )

print("Permutation test visualizations completed.")

# %%

########################################################
# Out-of-Sample Walk-Forward Validation
########################################################

print("\n" + "="*60)
print("OUT-OF-SAMPLE WALK-FORWARD VALIDATION")
print("="*60)
print("Testing strategy performance on unseen data with periodic re-optimization...")
print("This validates whether the strategy generalizes beyond the training period.")

# Define out-of-sample period (data NOT used in initial optimization)
oos_start = '2024-01-01'
oos_end = '2024-12-31'

# Walk-forward configuration
wf_config = {
    'initial_train_days': 730,  # 2 years of initial training data
    'test_window_days': 30,     # Test on 1 month at a time
    'reoptimize_frequency': 90, # Re-optimize every 3 months
    'n_optimization_trials': 200,  # Fewer trials for faster re-optimization
    'expanding_window': True     # Use all historical data for training
}

print(f"\nConfiguration:")
print(f"  Out-of-sample period: {oos_start} to {oos_end}")
print(f"  Initial training window: {wf_config['initial_train_days']} days")
print(f"  Test window: {wf_config['test_window_days']} days")
print(f"  Re-optimization frequency: Every {wf_config['reoptimize_frequency']} days")
print(f"  Window type: {'Expanding' if wf_config['expanding_window'] else 'Rolling'}")

# Get out-of-sample data
oos_data = features_df.loc[oos_start:oos_end]
print(f"\nOut-of-sample data points: {len(oos_data)}")

# Initialize storage for walk-forward results
wf_results = {
    'fold_dates': [],
    'fold_params': [],
    'fold_returns': [],
    'fold_metrics': [],
    'all_returns': [],
    'all_signals': [],
    'reoptimization_dates': []
}

# Convert to datetime for date arithmetic
oos_start_dt = pd.to_datetime(oos_start)
oos_end_dt = pd.to_datetime(oos_end)

# Initialize variables for walk-forward loop
current_date = oos_start_dt
fold_number = 0
days_since_reopt = 0
current_params = best_params.copy()  # Start with in-sample optimized params

print("\nStarting walk-forward validation...")

# Walk-forward loop
while current_date < oos_end_dt:
    fold_number += 1
    
    # Define test period for this fold
    test_start = current_date
    test_end = min(current_date + pd.Timedelta(days=wf_config['test_window_days']), oos_end_dt)
    
    # Check if we need to re-optimize
    if days_since_reopt >= wf_config['reoptimize_frequency'] or fold_number == 1:
        print(f"\n--- Fold {fold_number}: Re-optimization ---")
        
        # Define training period for re-optimization
        if wf_config['expanding_window']:
            # Use all data from beginning up to current date
            train_start_reopt = features_df.index[0].strftime('%Y-%m-%d')
        else:
            # Use rolling window
            train_start_reopt = (current_date - pd.Timedelta(days=wf_config['initial_train_days'])).strftime('%Y-%m-%d')
        
        train_end_reopt = (current_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"  Re-optimizing on data from {train_start_reopt} to {train_end_reopt}")
        
        # Run optimization with fewer trials for speed
        reopt_study = optimize_ema_with_optuna(
            features_df,
            train_start_reopt,
            train_end_reopt,
            price_col,
            log_return_col,
            n_trials=wf_config['n_optimization_trials'],
            random_seed=42 + fold_number  # Different seed for each re-optimization
        )
        
        # Update current parameters
        current_params = reopt_study.best_params
        print(f"  New parameters: Fast={current_params['fast_period']}, Slow={current_params['slow_period']}")
        print(f"  Optimization metric: {reopt_study.best_value:.4f}")
        
        # Store re-optimization info
        wf_results['reoptimization_dates'].append(current_date)
        days_since_reopt = 0
    else:
        print(f"\n--- Fold {fold_number}: Testing ---")
        print(f"  Using existing parameters: Fast={current_params['fast_period']}, Slow={current_params['slow_period']}")
    
    # Apply strategy to test period
    print(f"  Test period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    
    # Calculate strategy with current parameters
    test_strategy_data = calculate_ema_crossover_strategy(
        features_df,
        current_params['fast_period'],
        current_params['slow_period'],
        price_col,
        log_return_col
    )
    
    # Extract test period returns
    test_period_data = test_strategy_data.loc[test_start:test_end].dropna()
    
    if len(test_period_data) > 0:
        fold_returns = test_period_data['strategy_returns'].values
        fold_benchmark = test_period_data[log_return_col].values
        fold_signals = test_period_data['signal'].values
        
        # Calculate fold metrics
        fold_sharpe = calculate_sharpe_ratio(fold_returns) if len(fold_returns) > 1 else 0
        fold_profit_factor = calculate_profit_factor(fold_returns) if len(fold_returns) > 1 else 0
        fold_total_return = (1 + fold_returns).prod() - 1
        
        # Store results
        wf_results['fold_dates'].append((test_start, test_end))
        wf_results['fold_params'].append(current_params.copy())
        wf_results['fold_returns'].append(fold_returns)
        wf_results['all_returns'].extend(fold_returns)
        wf_results['all_signals'].extend(fold_signals)
        
        fold_metric = {
            'fold': fold_number,
            'sharpe': fold_sharpe,
            'profit_factor': fold_profit_factor,
            'total_return': fold_total_return,
            'n_days': len(fold_returns),
            'params': current_params.copy()
        }
        wf_results['fold_metrics'].append(fold_metric)
        
        print(f"  Fold performance: Sharpe={fold_sharpe:.3f}, PF={fold_profit_factor:.3f}, Return={fold_total_return:.1%}")
    else:
        print(f"  Warning: No data available for test period")
    
    # Move to next fold
    current_date = test_end + pd.Timedelta(days=1)
    days_since_reopt += wf_config['test_window_days']

print(f"\nWalk-forward validation completed. Total folds: {fold_number}")

# Calculate aggregate metrics
if len(wf_results['all_returns']) > 0:
    all_returns = np.array(wf_results['all_returns'])
    
    # Get benchmark returns for the same period
    oos_benchmark = features_df.loc[oos_start:oos_end][log_return_col].dropna().values
    
    # Calculate overall metrics
    overall_sharpe = calculate_sharpe_ratio(all_returns)
    overall_profit_factor = calculate_profit_factor(all_returns)
    overall_max_drawdown = calculate_max_drawdown(all_returns)
    overall_sortino = calculate_sortino_ratio(all_returns)
    overall_calmar = calculate_calmar_ratio(all_returns)
    overall_total_return = (1 + all_returns).prod() - 1
    overall_annualized_return = np.mean(all_returns) * 365
    
    # Calculate benchmark metrics
    benchmark_sharpe = calculate_sharpe_ratio(oos_benchmark)
    benchmark_total_return = (1 + oos_benchmark).prod() - 1
    benchmark_annualized = np.mean(oos_benchmark) * 365
    
    # Information ratio vs benchmark
    if len(all_returns) == len(oos_benchmark):
        overall_info_ratio = calculate_information_ratio(all_returns, oos_benchmark)
    else:
        overall_info_ratio = 0
    
    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Performance ({oos_start} to {oos_end}):")
    print(f"  Total Folds: {len(wf_results['fold_metrics'])}")
    print(f"  Re-optimizations: {len(wf_results['reoptimization_dates'])}")
    print(f"  Trading Days: {len(all_returns)}")
    
    print(f"\nStrategy Metrics:")
    print(f"  Sharpe Ratio: {overall_sharpe:.3f}")
    print(f"  Profit Factor: {overall_profit_factor:.3f}")
    print(f"  Sortino Ratio: {overall_sortino:.3f}")
    print(f"  Calmar Ratio: {overall_calmar:.3f}")
    print(f"  Information Ratio: {overall_info_ratio:.3f}")
    print(f"  Max Drawdown: {overall_max_drawdown:.1%}")
    print(f"  Total Return: {overall_total_return:.1%}")
    print(f"  Annualized Return: {overall_annualized_return:.1%}")
    
    print(f"\nBenchmark (Buy & Hold):")
    print(f"  Sharpe Ratio: {benchmark_sharpe:.3f}")
    print(f"  Total Return: {benchmark_total_return:.1%}")
    print(f"  Annualized Return: {benchmark_annualized:.1%}")
    
    print(f"\nStrategy vs Benchmark:")
    print(f"  Excess Return: {(overall_total_return - benchmark_total_return):.1%}")
    print(f"  Sharpe Difference: {(overall_sharpe - benchmark_sharpe):.3f}")
    
    # Analyze fold consistency
    fold_sharpes = [m['sharpe'] for m in wf_results['fold_metrics']]
    fold_pfs = [m['profit_factor'] for m in wf_results['fold_metrics']]
    fold_returns = [m['total_return'] for m in wf_results['fold_metrics']]
    
    print(f"\nFold Statistics:")
    print(f"  Average Fold Sharpe: {np.mean(fold_sharpes):.3f} ± {np.std(fold_sharpes):.3f}")
    print(f"  Average Fold PF: {np.mean(fold_pfs):.3f} ± {np.std(fold_pfs):.3f}")
    print(f"  Average Fold Return: {np.mean(fold_returns):.1%} ± {np.std(fold_returns):.1%}")
    print(f"  Positive Folds: {sum(1 for r in fold_returns if r > 0)}/{len(fold_returns)}")
    
    # Parameter evolution analysis
    fast_periods = [m['params']['fast_period'] for m in wf_results['fold_metrics']]
    slow_periods = [m['params']['slow_period'] for m in wf_results['fold_metrics']]
    
    print(f"\nParameter Evolution:")
    print(f"  Fast EMA range: {min(fast_periods)} - {max(fast_periods)}")
    print(f"  Slow EMA range: {min(slow_periods)} - {max(slow_periods)}")
    print(f"  Most common Fast: {max(set(fast_periods), key=fast_periods.count)}")
    print(f"  Most common Slow: {max(set(slow_periods), key=slow_periods.count)}")
    
    # Store aggregate results
    wf_results['overall_metrics'] = {
        'sharpe': overall_sharpe,
        'profit_factor': overall_profit_factor,
        'max_drawdown': overall_max_drawdown,
        'total_return': overall_total_return,
        'annualized_return': overall_annualized_return,
        'information_ratio': overall_info_ratio,
        'n_days': len(all_returns)
    }
    
    wf_results['benchmark_metrics'] = {
        'sharpe': benchmark_sharpe,
        'total_return': benchmark_total_return,
        'annualized_return': benchmark_annualized
    }
else:
    print("\nError: No returns generated during walk-forward validation")

# Visualization of walk-forward results
if len(wf_results['all_returns']) > 0:
    print("\n" + "="*60)
    print("WALK-FORWARD VISUALIZATION")
    print("="*60)
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Cumulative Returns Plot
    ax1 = axes[0]
    
    # Calculate cumulative returns
    strategy_cumret = (1 + np.array(wf_results['all_returns'])).cumprod() - 1
    benchmark_cumret = (1 + oos_benchmark).cumprod() - 1
    
    # Create date index for plotting
    date_index = pd.date_range(start=oos_start, periods=len(strategy_cumret), freq='D')[:len(strategy_cumret)]
    
    ax1.plot(date_index, strategy_cumret, label='Walk-Forward Strategy', color='blue', linewidth=2)
    ax1.plot(date_index[:len(benchmark_cumret)], benchmark_cumret, label='Buy & Hold', color='gray', linewidth=1.5, alpha=0.7)
    
    # Mark re-optimization points
    for reopt_date in wf_results['reoptimization_dates']:
        ax1.axvline(x=reopt_date, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_title('Walk-Forward Cumulative Returns', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 2. Parameter Evolution Plot
    ax2 = axes[1]
    
    # Extract fold dates and parameters
    fold_dates = [fold[0] for fold in wf_results['fold_dates']]
    fast_params = [m['params']['fast_period'] for m in wf_results['fold_metrics']]
    slow_params = [m['params']['slow_period'] for m in wf_results['fold_metrics']]
    
    ax2.plot(fold_dates, fast_params, 'o-', label='Fast EMA', color='green', markersize=6)
    ax2.plot(fold_dates, slow_params, 's-', label='Slow EMA', color='orange', markersize=6)
    
    # Mark re-optimization points
    for reopt_date in wf_results['reoptimization_dates']:
        ax2.axvline(x=reopt_date, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Re-optimization' if reopt_date == wf_results['reoptimization_dates'][0] else '')
    
    ax2.set_title('Parameter Evolution Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('EMA Period')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 3. Fold Performance Plot
    ax3 = axes[2]
    
    # Create bar plot of fold returns
    fold_indices = range(len(fold_returns))
    colors = ['green' if r > 0 else 'red' for r in fold_returns]
    
    bars = ax3.bar(fold_indices, np.array(fold_returns) * 100, color=colors, alpha=0.7, edgecolor='black')
    
    # Add horizontal line at zero
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Add average line
    avg_return = np.mean(fold_returns) * 100
    ax3.axhline(y=avg_return, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Average: {avg_return:.1f}%')
    
    ax3.set_title('Individual Fold Returns', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Fold Number')
    ax3.set_ylabel('Return (%)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Additional visualization: Performance heatmap by parameter combination
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create parameter combination performance matrix
    param_performance = {}
    for metric in wf_results['fold_metrics']:
        key = (metric['params']['fast_period'], metric['params']['slow_period'])
        if key not in param_performance:
            param_performance[key] = []
        param_performance[key].append(metric['total_return'])
    
    # Average performance for each parameter combination
    param_avg = {k: np.mean(v) for k, v in param_performance.items()}
    
    # Create scatter plot
    for (fast, slow), avg_return in param_avg.items():
        size = len(param_performance[(fast, slow)]) * 100  # Size based on frequency
        color = 'green' if avg_return > 0 else 'red'
        ax.scatter(fast, slow, s=size, c=[avg_return], cmap='RdYlGn', 
                  alpha=0.6, edgecolors='black', linewidth=1, vmin=-0.1, vmax=0.1)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=-0.1, vmax=0.1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average Return', rotation=270, labelpad=20)
    
    ax.set_title('Parameter Performance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fast EMA Period')
    ax.set_ylabel('Slow EMA Period')
    ax.grid(True, alpha=0.3)
    
    # Add text annotations for most used parameters
    most_used = max(param_performance.keys(), key=lambda x: len(param_performance[x]))
    ax.annotate(f'Most Used\n({len(param_performance[most_used])} times)', 
               xy=most_used, xytext=(most_used[0]+2, most_used[1]+5),
               arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
               fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.show()
    
    print("\nWalk-forward visualization completed.")

# %%

########################################################
# Out-of-Sample Permuted Walk-Forward Test
######################################################## 

print("\n" + "="*60)
print("OUT-OF-SAMPLE PERMUTED WALK-FORWARD TEST")
print("="*60)
print("Testing if walk-forward results are statistically significant...")
print("This validates that out-of-sample performance isn't due to random chance.")

# Only run if we have walk-forward results
if 'overall_metrics' not in wf_results or wf_results['overall_metrics'] is None:
    print("\nError: No walk-forward results available. Run walk-forward validation first.")
else:
    # Configuration
    n_permutations = 500  # Number of permutations
    metrics_to_test = ['sharpe_ratio', 'profit_factor']
    p_value_threshold = 0.05
    
    print(f"\nConfiguration:")
    print(f"  Number of permutations: {n_permutations}")
    print(f"  Metrics to test: {', '.join(metrics_to_test)}")
    print(f"  Significance threshold: {p_value_threshold}")
    print(f"  Test period: {oos_start} to {oos_end}")
    
    # Store original walk-forward metrics
    original_wf_sharpe = wf_results['overall_metrics']['sharpe']
    original_wf_pf = wf_results['overall_metrics']['profit_factor']
    
    print(f"\nOriginal Walk-Forward Metrics:")
    print(f"  Sharpe Ratio: {original_wf_sharpe:.4f}")
    print(f"  Profit Factor: {original_wf_pf:.4f}")
    
    # Function to run walk-forward on permuted data
    def run_permuted_walk_forward(permuted_df, fold_number_offset=0):
        """Run walk-forward validation on permuted OHLC data."""
        perm_results = {
            'all_returns': [],
            'fold_metrics': []
        }
        
        # Initialize walk-forward variables
        current_date = oos_start_dt
        fold_number = 0
        days_since_reopt = 0
        current_params = best_params.copy()
        
        while current_date < oos_end_dt:
            fold_number += 1
            
            # Define test period
            test_start = current_date
            test_end = min(current_date + pd.Timedelta(days=wf_config['test_window_days']), oos_end_dt)
            
            # Re-optimize if needed (same schedule as original)
            if days_since_reopt >= wf_config['reoptimize_frequency'] or fold_number == 1:
                # Define training period
                if wf_config['expanding_window']:
                    train_start_reopt = permuted_df.index[0].strftime('%Y-%m-%d')
                else:
                    train_start_reopt = (current_date - pd.Timedelta(days=wf_config['initial_train_days'])).strftime('%Y-%m-%d')
                
                train_end_reopt = (current_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                
                # Run optimization on permuted data (fewer trials for speed)
                try:
                    reopt_study = optimize_ema_with_optuna(
                        permuted_df,
                        train_start_reopt,
                        train_end_reopt,
                        price_col,
                        log_return_col,
                        n_trials=50,  # Even fewer trials for permutations
                        random_seed=42 + fold_number + fold_number_offset
                    )
                    current_params = reopt_study.best_params
                except:
                    # If optimization fails, keep existing params
                    pass
                
                days_since_reopt = 0
            
            # Apply strategy to test period
            try:
                test_strategy_data = calculate_ema_crossover_strategy(
                    permuted_df,
                    current_params['fast_period'],
                    current_params['slow_period'],
                    price_col,
                    log_return_col
                )
                
                test_period_data = test_strategy_data.loc[test_start:test_end].dropna()
                
                if len(test_period_data) > 0:
                    fold_returns = test_period_data['strategy_returns'].values
                    perm_results['all_returns'].extend(fold_returns)
            except:
                # Skip failed folds
                pass
            
            # Move to next fold
            current_date = test_end + pd.Timedelta(days=1)
            days_since_reopt += wf_config['test_window_days']
        
        return perm_results
    
    print(f"\nRunning {n_permutations} permuted walk-forward tests...")
    print("(This may take several minutes...)")
    
    # Run permutation tests
    permuted_sharpes = []
    permuted_pfs = []
    
    for i in range(n_permutations):
        if (i + 1) % 50 == 0:
            print(f"  Completed {i + 1}/{n_permutations} permutations...")
        
        # Create permuted OHLC data
        ohlc_cols = [f'{prefix}open', f'{prefix}high', f'{prefix}low', f'{prefix}close']
        ohlc_df = features_df[ohlc_cols].copy()
        ohlc_df.columns = ['open', 'high', 'low', 'close']
        
        # Use our sophisticated OHLC permutation
        from utils.validation import create_ohlc_permutation
        permuted_ohlc = create_ohlc_permutation(
            ohlc_df,
            start_index=0,
            seed=42 + i
        )
        
        # Create permuted features dataframe
        permuted_df = features_df.copy()
        permuted_df[f'{prefix}open'] = permuted_ohlc['open']
        permuted_df[f'{prefix}high'] = permuted_ohlc['high']
        permuted_df[f'{prefix}low'] = permuted_ohlc['low']
        permuted_df[f'{prefix}close'] = permuted_ohlc['close']
        
        # Recalculate log returns
        permuted_df[log_return_col] = np.log(
            permuted_df[f'{prefix}close'] / permuted_df[f'{prefix}close'].shift(1)
        )
        
        # Run walk-forward on permuted data
        try:
            perm_wf_results = run_permuted_walk_forward(permuted_df, fold_number_offset=i*100)
            
            if len(perm_wf_results['all_returns']) > 0:
                perm_returns = np.array(perm_wf_results['all_returns'])
                
                # Calculate metrics
                perm_sharpe = calculate_sharpe_ratio(perm_returns)
                perm_pf = calculate_profit_factor(perm_returns)
                
                if np.isfinite(perm_sharpe):
                    permuted_sharpes.append(perm_sharpe)
                if np.isfinite(perm_pf):
                    permuted_pfs.append(perm_pf)
        except Exception as e:
            # Skip failed permutations
            continue
    
    print(f"\nCompleted {len(permuted_sharpes)} valid permutations")
    
    # Calculate p-values
    permuted_sharpes = np.array(permuted_sharpes)
    permuted_pfs = np.array(permuted_pfs)
    
    sharpe_p_value = np.mean(permuted_sharpes >= original_wf_sharpe) if len(permuted_sharpes) > 0 else 1.0
    pf_p_value = np.mean(permuted_pfs >= original_wf_pf) if len(permuted_pfs) > 0 else 1.0
    
    sharpe_percentile = (1 - sharpe_p_value) * 100
    pf_percentile = (1 - pf_p_value) * 100
    
    # Display results
    print("\n" + "="*60)
    print("PERMUTED WALK-FORWARD TEST RESULTS")
    print("="*60)
    
    print(f"\nSHARPE RATIO:")
    print(f"  Original Walk-Forward: {original_wf_sharpe:.4f}")
    if len(permuted_sharpes) > 0:
        print(f"  Permuted Mean: {np.mean(permuted_sharpes):.4f} ± {np.std(permuted_sharpes):.4f}")
        print(f"  Permuted Range: [{np.min(permuted_sharpes):.4f}, {np.max(permuted_sharpes):.4f}]")
    print(f"  P-value: {sharpe_p_value:.4f}")
    print(f"  Percentile: {sharpe_percentile:.1f}%")
    print(f"  Statistically Significant: {'YES ✅' if sharpe_p_value < p_value_threshold else 'NO ❌'}")
    
    print(f"\nPROFIT FACTOR:")
    print(f"  Original Walk-Forward: {original_wf_pf:.4f}")
    if len(permuted_pfs) > 0:
        print(f"  Permuted Mean: {np.mean(permuted_pfs):.4f} ± {np.std(permuted_pfs):.4f}")
        print(f"  Permuted Range: [{np.min(permuted_pfs):.4f}, {np.max(permuted_pfs):.4f}]")
    print(f"  P-value: {pf_p_value:.4f}")
    print(f"  Percentile: {pf_percentile:.1f}%")
    print(f"  Statistically Significant: {'YES ✅' if pf_p_value < p_value_threshold else 'NO ❌'}")
    
    # Overall assessment
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    significant_metrics = []
    if sharpe_p_value < p_value_threshold:
        significant_metrics.append('Sharpe Ratio')
    if pf_p_value < p_value_threshold:
        significant_metrics.append('Profit Factor')
    
    if len(significant_metrics) > 0:
        print(f"✅ STRATEGY PASSES OUT-OF-SAMPLE PERMUTATION TEST")
        print(f"   Significant metrics: {', '.join(significant_metrics)}")
        print(f"   The walk-forward strategy shows genuine predictive ability.")
        print(f"   Results are unlikely to be due to random chance.")
    else:
        print(f"❌ STRATEGY FAILS OUT-OF-SAMPLE PERMUTATION TEST")
        print(f"   No metrics show statistical significance (p < {p_value_threshold}).")
        print(f"   Walk-forward results may be due to random market movements.")
        print(f"   Consider reviewing strategy logic or parameters.")
    
    # Visualization of permutation test results
    if len(permuted_sharpes) > 0 and len(permuted_pfs) > 0:
        print("\n" + "="*60)
        print("PERMUTATION TEST VISUALIZATION")
        print("="*60)
        
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Sharpe Ratio Histogram
        ax1 = axes[0, 0]
        ax1.hist(permuted_sharpes, bins=50, alpha=0.7, color='lightblue', 
                edgecolor='black', density=True, label='Permuted Sharpe')
        ax1.axvline(original_wf_sharpe, color='red', linestyle='-', linewidth=2,
                   label=f'Original: {original_wf_sharpe:.3f}')
        ax1.axvline(np.mean(permuted_sharpes), color='blue', linestyle='--', linewidth=1.5,
                   label=f'Permuted Mean: {np.mean(permuted_sharpes):.3f}')
        
        # Add p-value and percentile text
        ax1.text(0.02, 0.98, f'P-value: {sharpe_p_value:.4f}\nPercentile: {sharpe_percentile:.1f}%',
                transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title('Sharpe Ratio Distribution - Permuted Walk-Forward', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Sharpe Ratio')
        ax1.set_ylabel('Density')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Profit Factor Histogram
        ax2 = axes[0, 1]
        ax2.hist(permuted_pfs, bins=50, alpha=0.7, color='lightgreen',
                edgecolor='black', density=True, label='Permuted PF')
        ax2.axvline(original_wf_pf, color='red', linestyle='-', linewidth=2,
                   label=f'Original: {original_wf_pf:.3f}')
        ax2.axvline(np.mean(permuted_pfs), color='green', linestyle='--', linewidth=1.5,
                   label=f'Permuted Mean: {np.mean(permuted_pfs):.3f}')
        
        # Add p-value and percentile text
        ax2.text(0.02, 0.98, f'P-value: {pf_p_value:.4f}\nPercentile: {pf_percentile:.1f}%',
                transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_title('Profit Factor Distribution - Permuted Walk-Forward', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Profit Factor')
        ax2.set_ylabel('Density')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. Sharpe Ratio Cumulative Distribution
        ax3 = axes[1, 0]
        sorted_sharpes = np.sort(permuted_sharpes)
        cumulative = np.arange(1, len(sorted_sharpes) + 1) / len(sorted_sharpes)
        
        ax3.plot(sorted_sharpes, cumulative, 'b-', linewidth=2, label='Permuted CDF')
        ax3.axvline(original_wf_sharpe, color='red', linestyle='-', linewidth=2,
                   label=f'Original: {original_wf_sharpe:.3f}')
        ax3.axhline(1 - sharpe_p_value, color='red', linestyle='--', alpha=0.5,
                   label=f'Percentile: {sharpe_percentile:.1f}%')
        
        ax3.set_title('Sharpe Ratio Cumulative Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Sharpe Ratio')
        ax3.set_ylabel('Cumulative Probability')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # 4. Profit Factor Cumulative Distribution
        ax4 = axes[1, 1]
        sorted_pfs = np.sort(permuted_pfs)
        cumulative = np.arange(1, len(sorted_pfs) + 1) / len(sorted_pfs)
        
        ax4.plot(sorted_pfs, cumulative, 'g-', linewidth=2, label='Permuted CDF')
        ax4.axvline(original_wf_pf, color='red', linestyle='-', linewidth=2,
                   label=f'Original: {original_wf_pf:.3f}')
        ax4.axhline(1 - pf_p_value, color='red', linestyle='--', alpha=0.5,
                   label=f'Percentile: {pf_percentile:.1f}%')
        
        ax4.set_title('Profit Factor Cumulative Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Profit Factor')
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])
        
        # Add overall title
        fig.suptitle('Out-of-Sample Permuted Walk-Forward Test Results', fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualization: Joint scatter plot
        if len(permuted_sharpes) == len(permuted_pfs):
            fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Scatter plot of permuted results
            ax.scatter(permuted_sharpes, permuted_pfs, alpha=0.5, c='lightblue', 
                      edgecolors='blue', linewidth=0.5, s=30, label='Permuted Results')
            
            # Mark original result
            ax.scatter(original_wf_sharpe, original_wf_pf, color='red', s=200, 
                      marker='*', edgecolors='darkred', linewidth=2, zorder=100,
                      label=f'Original WF Result')
            
            # Add crosshairs at original point
            ax.axvline(original_wf_sharpe, color='red', alpha=0.3, linestyle='--', linewidth=1)
            ax.axhline(original_wf_pf, color='red', alpha=0.3, linestyle='--', linewidth=1)
            
            # Add percentile contours
            from scipy.stats import gaussian_kde
            try:
                # Create KDE for density estimation
                values = np.vstack([permuted_sharpes, permuted_pfs])
                kernel = gaussian_kde(values)
                
                # Create grid
                xx, yy = np.mgrid[permuted_sharpes.min():permuted_sharpes.max():.01,
                                  permuted_pfs.min():permuted_pfs.max():.01]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = np.reshape(kernel(positions).T, xx.shape)
                
                # Plot contours
                ax.contour(xx, yy, density, colors='gray', alpha=0.5, linewidths=1)
            except:
                pass  # Skip contours if KDE fails
            
            ax.set_xlabel('Sharpe Ratio', fontsize=12)
            ax.set_ylabel('Profit Factor', fontsize=12)
            ax.set_title('Joint Distribution: Sharpe vs Profit Factor\n(Out-of-Sample Walk-Forward)', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add text box with p-values
            textstr = f'Sharpe p-value: {sharpe_p_value:.4f}\nPF p-value: {pf_p_value:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.show()
        
        print("\nPermutation test visualization completed.")

# %%




