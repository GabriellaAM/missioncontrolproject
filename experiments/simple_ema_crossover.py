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
from utils.validation import permutation_test, walk_forward_validation, permuted_walk_forward_test, strategy_permutation_test
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


def optimize_ema_with_optuna(data, train_start, train_end, price_col, log_return_col, n_trials=1000):
    """
    Optimize EMA parameters using Optuna.
    
    Args:
        data: DataFrame with price data and log returns
        train_start: Start date for training period
        train_end: End date for training period
        price_col: Column name for price data
        log_return_col: Column name for log returns
        n_trials: Number of optimization trials
    
    Returns:
        optuna.Study object with optimization results
    """
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
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=f"EMA_Optimization_{train_start}_{train_end}"
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
permutation_results = strategy_permutation_test(
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
    Plot original strategy cumulative returns vs permuted returns with shaded area.
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate cumulative returns
    original_cumret = (1 + original_returns).cumprod() - 1
    
    # Calculate cumulative returns for all permutations
    permuted_cumrets = []
    for perm_returns in permuted_returns_list:
        if len(perm_returns) == len(original_returns):
            perm_cumret = (1 + perm_returns).cumprod() - 1
            permuted_cumrets.append(perm_cumret)
    
    if permuted_cumrets:
        permuted_cumrets = np.array(permuted_cumrets)
        
        # Calculate percentiles for shaded area
        percentile_5 = np.percentile(permuted_cumrets, 5, axis=0)
        percentile_95 = np.percentile(permuted_cumrets, 95, axis=0)
        percentile_25 = np.percentile(permuted_cumrets, 25, axis=0)
        percentile_75 = np.percentile(permuted_cumrets, 75, axis=0)
        median = np.median(permuted_cumrets, axis=0)
        
        # Create time index
        time_index = range(len(original_cumret))
        
        # Plot shaded areas for permuted returns
        plt.fill_between(time_index, percentile_5, percentile_95, 
                        alpha=0.2, color='gray', label='Permuted 5%-95%')
        plt.fill_between(time_index, percentile_25, percentile_75, 
                        alpha=0.3, color='gray', label='Permuted 25%-75%')
        
        # Plot median of permuted returns
        plt.plot(time_index, median, color='gray', linestyle='--', alpha=0.8, label='Permuted Median')
    
    # Plot original strategy
    plt.plot(range(len(original_cumret)), original_cumret, color='red', linewidth=2, label='Original Strategy')
    
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


# %%

########################################################
# Out-of-Sample Permuted Walk-Forward Test
######################################################## 





# %%

########################################################
# Training Pipeline
########################################################



