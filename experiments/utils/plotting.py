"""
Plotting utilities for strategy backtesting and validation results
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_price_and_signals(strategy_data: pd.DataFrame, 
                          asset_name: str,
                          price_col: str = None,
                          signal_col: str = 'signal',
                          title: str = None,
                          figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """
    Plot price data with strategy signals overlaid.
    
    Args:
        strategy_data: DataFrame with price and signal data
        asset_name: Name of the asset for labels
        price_col: Column name for price data (auto-detect if None)
        signal_col: Column name for signals
        title: Chart title (auto-generate if None)
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    if price_col is None:
        price_col = f"{asset_name}_close"
    
    if title is None:
        title = f"{asset_name.title()} Price with Strategy Signals"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Price plot
    ax1.plot(strategy_data.index, strategy_data[price_col], 
             color='black', linewidth=1, label='Price')
    
    # Signal overlays
    long_signals = strategy_data[strategy_data[signal_col] == 1]
    short_signals = strategy_data[strategy_data[signal_col] == -1]
    
    if len(long_signals) > 0:
        ax1.scatter(long_signals.index, long_signals[price_col], 
                   color='green', marker='^', s=50, alpha=0.7, label='Long Signal')
    
    if len(short_signals) > 0:
        ax1.scatter(short_signals.index, short_signals[price_col], 
                   color='red', marker='v', s=50, alpha=0.7, label='Short Signal')
    
    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Signals plot
    ax2.plot(strategy_data.index, strategy_data[signal_col], 
             color='blue', linewidth=1, drawstyle='steps-post')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Signal', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_returns_analysis(strategy_data: pd.DataFrame,
                         strategy_returns_col: str = 'strategy_returns',
                         benchmark_returns_col: str = None,
                         asset_name: str = 'Strategy',
                         title: str = None,
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot comprehensive returns analysis including cumulative returns, 
    drawdown, and return distribution (without rolling Sharpe).
    
    Args:
        strategy_data: DataFrame with returns data
        strategy_returns_col: Column name for strategy returns
        benchmark_returns_col: Column name for benchmark returns (optional)
        asset_name: Name for labeling
        title: Chart title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if title is None:
        title = f"{asset_name.title()} Returns Analysis"
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Calculate cumulative returns (convert log returns to simple returns first)
    strategy_returns = strategy_data[strategy_returns_col].fillna(0)
    strategy_simple_returns = np.exp(strategy_returns) - 1
    strategy_cum_returns = (1 + strategy_simple_returns).cumprod()
    
    # Cumulative returns plot
    ax = axes[0, 0]
    ax.plot(strategy_cum_returns.index, strategy_cum_returns, 
            color='blue', linewidth=2, label='Strategy')
    
    if benchmark_returns_col and benchmark_returns_col in strategy_data.columns:
        benchmark_returns = strategy_data[benchmark_returns_col].fillna(0)
        benchmark_simple_returns = np.exp(benchmark_returns) - 1
        benchmark_cum_returns = (1 + benchmark_simple_returns).cumprod()
        ax.plot(benchmark_cum_returns.index, benchmark_cum_returns, 
                color='gray', linewidth=2, label='Benchmark', alpha=0.7)
    
    ax.set_title('Cumulative Returns', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Drawdown plot
    ax = axes[0, 1]
    running_max = strategy_cum_returns.expanding().max()
    drawdown = (strategy_cum_returns - running_max) / running_max
    
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
    ax.set_title('Drawdown', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Returns distribution (use simple returns for histogram)
    ax = axes[1, 0]
    returns_clean = strategy_simple_returns.dropna()
    ax.hist(returns_clean, bins=50, alpha=0.7, color='blue', density=True)
    ax.axvline(returns_clean.mean(), color='red', linestyle='--', 
               label=f'Mean: {returns_clean.mean():.4f}')
    ax.set_title('Returns Distribution', fontsize=12)
    ax.set_xlabel('Daily Returns', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Monthly returns heatmap
    ax = axes[1, 1]
    if len(strategy_returns) > 30:  # Only if we have enough data
        try:
            # Convert to monthly returns (use simple returns)
            monthly_returns = strategy_simple_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns.index = monthly_returns.index.to_period('M')
            
            # Create pivot table for heatmap
            monthly_df = monthly_returns.to_frame('returns')
            monthly_df['year'] = monthly_df.index.year
            monthly_df['month'] = monthly_df.index.month
            
            pivot_table = monthly_df.pivot_table(values='returns', index='year', columns='month')
            
            # Create heatmap
            im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
            ax.set_xticks(range(len(pivot_table.columns)))
            ax.set_xticklabels([f'M{i}' for i in pivot_table.columns])
            ax.set_yticks(range(len(pivot_table.index)))
            ax.set_yticklabels(pivot_table.index)
            ax.set_title('Monthly Returns Heatmap', fontsize=12)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Monthly Return')
        except:
            # Fallback to simple text if heatmap fails
            ax.text(0.5, 0.5, 'Monthly Returns\nHeatmap\n(Insufficient Data)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Monthly Returns', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'Monthly Returns\nHeatmap\n(Insufficient Data)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Monthly Returns', fontsize=12)
    
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.tight_layout()
    return fig

def plot_permutation_histogram(original_value: float, 
                              permuted_values: np.ndarray, 
                              metric_name: str, 
                              p_value: float,
                              title: str = None,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot histogram of permuted metric values with original value and p-value line.
    
    Args:
        original_value: Original strategy metric value
        permuted_values: Array of permuted metric values
        metric_name: Name of the metric
        p_value: P-value from permutation test
        title: Chart title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if title is None:
        title = f'{metric_name.replace("_", " ").title()} Distribution - Permutation Test'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram of permuted values
    ax.hist(permuted_values, bins=50, alpha=0.7, color='lightblue', 
             edgecolor='black', density=True, label=f'Permuted {metric_name}')
    
    # Plot original value line
    ax.axvline(original_value, color='red', linestyle='-', linewidth=2, 
                label=f'Original {metric_name}: {original_value:.3f}')
    
    # Add p-value text
    ax.text(0.02, 0.98, f'P-value: {p_value:.4f}', 
             transform=ax.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add percentile text
    percentile = (1 - p_value) * 100
    ax.text(0.02, 0.90, f'Percentile: {percentile:.1f}%', 
             transform=ax.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(metric_name.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_validation_results(permutation_results: Dict,
                           wf_results: Dict = None,
                           wf_perm_results: Dict = None,
                           title: str = 'Permutation Test Results',
                           figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot permutation test histograms for in-sample and walk-forward validation results.
    
    Args:
        permutation_results: Results from in-sample permutation test
        wf_results: Walk-forward validation results (optional, not used)
        wf_perm_results: Walk-forward permutation test results
        title: Chart title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Prepare data for plotting
    plot_data = []
    
    # Add in-sample permutation results
    for metric_name, results in permutation_results.items():
        if metric_name not in ['permuted_returns', 'original_returns']:
            plot_data.append({
                'title': f'In-Sample {metric_name.replace("_", " ").title()}',
                'original': results['original_value'],
                'permuted': results['permuted_values'],
                'p_value': results['p_value']
            })
    
    # Add walk-forward permutation results if available
    if wf_perm_results:
        for metric in ['sharpe', 'profit_factor']:
            if metric in wf_perm_results:
                # Extract permuted values from walk-forward results
                wf_result = wf_perm_results[metric]
                plot_data.append({
                    'title': f'Walk-Forward {metric.replace("_", " ").title()}',
                    'original': wf_result['original'],
                    'permuted': wf_result.get('permuted_values', np.array([])),
                    'p_value': wf_result['p_value']
                })
    
    if len(plot_data) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No permutation test data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(title, fontsize=14)
        return fig
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each metric
    for i, data in enumerate(plot_data[:4]):  # Max 4 plots
        ax = axes[i]
        
        permuted_values = data['permuted']
        
        if isinstance(permuted_values, str):
            try:
                permuted_values = np.fromstring(permuted_values.strip('[]'), sep=' ')
            except:
                permuted_values = np.array([])
        
        if len(permuted_values) == 0:
            # No histogram data - show text summary
            ax.text(0.5, 0.5, 
                   f"{data['title']}\n\nOriginal: {data['original']:.3f}\nP-value: {data['p_value']:.4f}\nPasses: {'YES' if data['p_value'] < 0.05 else 'NO'}",
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgreen' if data['p_value'] < 0.05 else 'lightcoral', alpha=0.3))
            ax.set_title(data['title'], fontsize=12)
            ax.axis('off')
        else:
            # Plot histogram
            ax.hist(permuted_values, bins=30, alpha=0.7, color='lightblue', 
                    edgecolor='black', density=True, label='Permuted')
            
            # Plot original value line
            ax.axvline(data['original'], color='red', linestyle='-', linewidth=2, 
                      label=f'Original: {data["original"]:.3f}')
            
            # Add p-value text
            ax.text(0.02, 0.98, f'P-value: {data["p_value"]:.4f}', 
                    transform=ax.transAxes, fontsize=10, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add percentile text
            percentile = (1 - data['p_value']) * 100
            ax.text(0.02, 0.88, f'Percentile: {percentile:.1f}%', 
                    transform=ax.transAxes, fontsize=10, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(data['title'], fontsize=12)
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(plot_data), 4):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.tight_layout()
    return fig


def plot_walk_forward_analysis(strategy_data: pd.DataFrame,
                              wf_results: Dict,
                              asset_name: str,
                              strategy_name: str,
                              title: str = None,
                              figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
    """
    Create comprehensive walk-forward analysis visualization.
    
    Args:
        strategy_data: DataFrame with strategy results
        wf_results: Walk-forward validation results
        asset_name: Asset name for labeling
        strategy_name: Strategy name for labeling
        title: Chart title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    if title is None:
        title = f'{strategy_name} - {asset_name} Walk-Forward Analysis'
    
    # Check if we have walk-forward results
    if not wf_results or 'fold_dates' not in wf_results or len(wf_results['fold_dates']) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No walk-forward results available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(title, fontsize=14)
        return fig
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), height_ratios=[2, 1, 1, 1])
    
    # Extract data from walk-forward results
    fold_dates = wf_results['fold_dates']
    fold_params = wf_results.get('fold_params', [])
    fold_metrics = wf_results.get('fold_metrics', [])
    reopt_dates = wf_results.get('reoptimization_dates', [])
    
    # Get the walk-forward period (test period) using train_test_split
    # The test period starts right after training ends, not at the first fold
    train_test_split = wf_results.get('train_test_split', 0.75)
    split_idx = int(len(strategy_data) * train_test_split)
    
    # Test data spans from the split point to the end of available data.
    # Do NOT trim to the last recorded fold end, since later folds may have
    # been skipped (e.g., due to NaNs) while valid returns still exist.
    test_data = strategy_data.iloc[split_idx:].copy()
    
    # Only drop rows where strategy_returns is NaN (the critical column for plotting)
    if 'strategy_returns' in test_data.columns:
        test_data = test_data.dropna(subset=['strategy_returns'])
    else:
        test_data = test_data.dropna()

    # Clip reoptimization markers to the visible test period to avoid
    # extending the x-axis beyond the plotted returns
    if len(test_data) > 0 and reopt_dates:
        test_start_dt = test_data.index[0]
        test_end_dt = test_data.index[-1]
        reopt_dates = [d for d in reopt_dates if (d >= test_start_dt and d <= test_end_dt)]
    
    # 1. Cumulative Returns with Reoptimization Markers
    ax1 = axes[0]
    if len(test_data) > 0:
        # Strategy cumulative returns (convert log returns to simple returns first)
        strategy_log_returns = test_data['strategy_returns'].fillna(0)
        strategy_returns = np.exp(strategy_log_returns) - 1
        strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
        ax1.plot(strategy_cum_returns.index, strategy_cum_returns, 
                label="Strategy", color='blue', linewidth=2)
        
        # Benchmark cumulative returns (convert log returns to simple returns first)
        benchmark_col = f"{asset_name}_log_return_1"
        if benchmark_col in test_data.columns:
            benchmark_log_returns = test_data[benchmark_col].fillna(0)
            benchmark_returns = np.exp(benchmark_log_returns) - 1
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
            ax1.plot(benchmark_cum_returns.index, benchmark_cum_returns, 
                    label="Benchmark", linestyle="--", color='gray', linewidth=2)
        
        # Add reoptimization markers
        for reopt_date in reopt_dates:
            ax1.axvline(x=reopt_date, color='red', linestyle=':', alpha=0.7, linewidth=1)
        
        # Add legend entry for reoptimization lines
        if reopt_dates:
            ax1.axvline(x=reopt_dates[0], color='red', linestyle=':', alpha=0.7, 
                       linewidth=1, label='Reoptimization')
    
    ax1.set_title('Walk-Forward Cumulative Returns', fontsize=12)
    ax1.set_ylabel('Cumulative Return', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 2. Drawdown Analysis
    ax2 = axes[1]
    if len(test_data) > 0:
        # Calculate and plot strategy drawdown (already have simple returns)
        strategy_cumulative = (1 + strategy_returns).cumprod()
        strategy_running_max = strategy_cumulative.cummax()
        strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max
        
        ax2.fill_between(strategy_drawdown.index, 0, strategy_drawdown, 
                        color='blue', alpha=0.3, label='Strategy Drawdown')
        ax2.plot(strategy_drawdown.index, strategy_drawdown, color='blue', linewidth=1)
        
        # Benchmark drawdown if available (already have simple returns)
        if benchmark_col in test_data.columns:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_running_max = benchmark_cumulative.cummax()
            benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
            ax2.fill_between(benchmark_drawdown.index, 0, benchmark_drawdown, 
                            color='gray', alpha=0.2, label='Benchmark Drawdown')
            ax2.plot(benchmark_drawdown.index, benchmark_drawdown, color='gray', linewidth=1, linestyle='--')
        
        # Add reoptimization markers
        for reopt_date in reopt_dates:
            ax2.axvline(x=reopt_date, color='red', linestyle=':', alpha=0.7, linewidth=1)
    
    ax2.set_title('Walk-Forward Drawdown', fontsize=12)
    ax2.set_ylabel('Drawdown', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # 3. Parameter Evolution
    ax3 = axes[2]
    if fold_params and len(fold_params) > 0:
        # Extract parameter names dynamically (avoid hardcoding to EMA)
        param_names = list(fold_params[0].keys()) if fold_params else []
        
        # Plot evolution of each parameter
        colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))
        
        for i, param_name in enumerate(param_names):
            param_values = [params.get(param_name, np.nan) for params in fold_params]
            fold_numbers = range(1, len(param_values) + 1)
            
            # Only plot if parameter values are numeric
            if all(isinstance(v, (int, float)) and not np.isnan(v) for v in param_values):
                ax3.plot(fold_numbers, param_values, 
                        marker='o', color=colors[i], linewidth=2, 
                        label=param_name, markersize=6)
        
        ax3.set_title('Parameter Evolution', fontsize=12)
        ax3.set_xlabel('Fold Number', fontsize=10)
        ax3.set_ylabel('Parameter Value', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No parameter data available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Parameter Evolution', fontsize=12)
    
    # 4. Individual Fold Returns
    ax4 = axes[3]
    if fold_metrics and len(fold_metrics) > 0:
        fold_numbers = [metric['fold'] for metric in fold_metrics]
        fold_returns = [metric.get('total_return', 0) for metric in fold_metrics]
        
        # Color bars based on positive/negative returns
        colors = ['green' if ret >= 0 else 'red' for ret in fold_returns]
        
        bars = ax4.bar(fold_numbers, fold_returns, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax4.set_title('Individual Fold Returns', fontsize=12)
        ax4.set_xlabel('Fold Number', fontsize=10)
        ax4.set_ylabel('Total Return', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Format y-axis as percentage
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    else:
        ax4.text(0.5, 0.5, 'No fold metrics available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Individual Fold Returns', fontsize=12)
    
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    return fig


def save_plots_for_mlflow(strategy_data: pd.DataFrame,
                         optimization_results: Dict,
                         permutation_results: Dict,
                         wf_results: Dict,
                         wf_perm_results: Dict,
                         asset_name: str,
                         strategy_name: str,
                         output_dir: str = '.',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> List[str]:
    """
    Generate and save all plots for MLflow artifact logging.
    
    Args:
        strategy_data: DataFrame with strategy results
        optimization_results: Optimization results
        permutation_results: Permutation test results
        wf_results: Walk-forward results
        wf_perm_results: Walk-forward permutation results
        asset_name: Asset name
        strategy_name: Strategy name
        output_dir: Directory to save plots
        start_date: Optional start date to filter out warmup period (YYYY-MM-DD format)
        end_date: Optional end date to filter to training period only (YYYY-MM-DD format)
        
    Returns:
        List of saved plot filenames
    """
    saved_files = []
    
    # Filter data for in-sample plotting
    plot_data = strategy_data.copy()
    if start_date:
        print(f"   Filtering plots to exclude warmup period (from {start_date} onward)")
        plot_data = plot_data.loc[start_date:]
    if end_date:
        print(f"   Filtering plots to show training period only (until {end_date})")
        plot_data = plot_data.loc[:end_date]
    
    # Use filtered data ONLY for in-sample plots; keep original strategy_data for WFA
    # (do not reassign strategy_data)
    
    # Cumulative returns plot with drawdown
    try:
        # Create cumulative returns plot with drawdown subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # Strategy cumulative returns (convert log returns to simple returns first)
        strategy_log_returns = plot_data['strategy_returns'].fillna(0)
        strategy_returns = np.exp(strategy_log_returns) - 1
        strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
        ax1.plot(strategy_cum_returns.index, strategy_cum_returns, 
               label="Strategy Cumulative Return", color='blue', linewidth=2)
        
        # Benchmark cumulative returns (convert log returns to simple returns first)
        benchmark_col = f"{asset_name}_log_return_1"
        if benchmark_col in plot_data.columns:
            benchmark_log_returns = plot_data[benchmark_col].fillna(0)
            benchmark_returns = np.exp(benchmark_log_returns) - 1
            benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
            ax1.plot(benchmark_cum_returns.index, benchmark_cum_returns, 
                   label="Benchmark Cumulative Return", linestyle="--", color='gray', linewidth=2)
        
        ax1.set_title(f'{strategy_name} - {asset_name} Cumulative Returns', fontsize=14)
        ax1.set_ylabel("Cumulative Return", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Calculate and plot drawdowns
        strategy_cumulative = (1 + strategy_returns).cumprod()
        strategy_running_max = strategy_cumulative.cummax()
        strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max
        
        ax2.fill_between(strategy_drawdown.index, 0, strategy_drawdown, 
                        color='blue', alpha=0.3, label='Strategy Drawdown')
        ax2.plot(strategy_drawdown.index, strategy_drawdown, color='blue', linewidth=1)
        
        # Benchmark drawdown if available
        if benchmark_col in plot_data.columns:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_running_max = benchmark_cumulative.cummax()
            benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
            ax2.fill_between(benchmark_drawdown.index, 0, benchmark_drawdown, 
                            color='gray', alpha=0.2, label='Benchmark Drawdown')
            ax2.plot(benchmark_drawdown.index, benchmark_drawdown, color='gray', linewidth=1, linestyle='--')
        
        ax2.set_title('Drawdown', fontsize=12)
        ax2.set_xlabel("Time", fontsize=12)
        ax2.set_ylabel("Drawdown", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename1 = f'{output_dir}/cumulative_returns.png'
        fig.savefig(filename1, dpi=150, bbox_inches='tight')
        saved_files.append(filename1)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not create cumulative returns plot: {e}")
    
    
    # Permutation test histograms
    try:
        fig3 = plot_validation_results(permutation_results, wf_results, wf_perm_results,
                                     title=f'{strategy_name} - Permutation Test Results')
        filename3 = f'{output_dir}/permutation_tests.png'
        fig3.savefig(filename3, dpi=150, bbox_inches='tight')
        saved_files.append(filename3)
        plt.close(fig3)
    except Exception as e:
        print(f"Warning: Could not create permutation test plot: {e}")
    
    # Walk-forward analysis plot
    try:
        fig4 = plot_walk_forward_analysis(strategy_data, wf_results, asset_name, strategy_name,
                                        title=f'{strategy_name} - Walk-Forward Analysis')
        filename4 = f'{output_dir}/walk_forward_analysis.png'
        fig4.savefig(filename4, dpi=150, bbox_inches='tight')
        saved_files.append(filename4)
        plt.close(fig4)
    except Exception as e:
        print(f"Warning: Could not create walk-forward analysis plot: {e}")
    
    return saved_files


def plot_feature_importance(model, features: list, model_type: str,
                          title: str = None, max_features: int = 20,
                          save_path: str = None) -> str:
    """
    Create feature importance plot for any model type.

    Args:
        model: Trained model with feature importance capability
        features: List of feature names
        model_type: Type of model ('catboost', 'sklearn', 'lightgbm', etc.)
        title: Plot title (auto-generated if None)
        max_features: Maximum number of features to show
        save_path: Path to save plot (temp file if None)

    Returns:
        Path to saved plot file
    """
    import os
    import tempfile

    if title is None:
        title = f"{model_type.title()} Feature Importance"

    try:
        # Extract feature importance based on model type
        if model_type.lower() == 'catboost':
            importance_values = model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            # sklearn-style models
            importance_values = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models - use absolute coefficients
            importance_values = np.abs(model.coef_).flatten()
        else:
            raise ValueError(f"Unsupported model type for feature importance: {model_type}")

        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame({
            'feature': features[:len(importance_values)],  # Handle length mismatch
            'importance': importance_values
        }).sort_values('importance', ascending=False)

        # Limit to top features
        if len(importance_df) > max_features:
            importance_df = importance_df.head(max_features)

        # Create plot
        plt.figure(figsize=(12, max(6, len(importance_df) * 0.3)))

        # Use horizontal bar chart for better readability with long feature names
        sns.barplot(data=importance_df, x='importance', y='feature', orient='h')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)

        # Add value labels on bars
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            plt.text(row['importance'], i, f'{row["importance"]:.3f}',
                    va='center', ha='left', fontsize=9, alpha=0.8)

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = os.path.join(tempfile.gettempdir(), 'feature_importance.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    except Exception as e:
        print(f"Warning: Could not create feature importance plot: {e}")
        # Return empty file path on error
        return ""


def create_feature_importance_plot(importance_values, feature_names, save_path, title="Feature Importance"):
    """
    Create feature importance plot with color coding for different feature types.

    Args:
        importance_values: Array of feature importance values
        feature_names: List of feature names
        save_path: Path to save the plot
        title: Plot title

    Returns:
        str: Path to saved plot file
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Sort features by importance
    sorted_indices = np.argsort(importance_values)
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importances = importance_values[sorted_indices]

    # Define colors for different feature types
    colors = []
    for feature in sorted_features:
        if 'topo_' in feature:
            colors.append('skyblue')  # Topological features
        elif 'signal' in feature:
            colors.append('orange')   # Signal features
        elif 'return' in feature:
            colors.append('green')    # Return features
        else:
            colors.append('gray')     # Other features

    # Create horizontal bar plot
    plt.figure(figsize=(10, max(6, len(sorted_features) * 0.3)))
    bars = plt.barh(range(len(sorted_features)), sorted_importances, color=colors)

    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Feature Importance')
    plt.title(title)
    plt.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, sorted_importances)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontsize=8)

    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='skyblue', label='Topological'),
        plt.Rectangle((0,0),1,1, facecolor='orange', label='Signal'),
        plt.Rectangle((0,0),1,1, facecolor='green', label='Returns'),
        plt.Rectangle((0,0),1,1, facecolor='gray', label='Other')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def plot_signal_comparison(data: pd.DataFrame,
                          primary_signal_col: str = 'primary_signal',
                          meta_decision_col: str = 'meta_decision',
                          final_signal_col: str = 'signal',
                          title: str = "Signal Evolution Analysis",
                          save_path: str = None,
                          max_days: int = 500) -> str:
    """
    Create signal comparison plot showing primary, meta, and final signals.

    Args:
        data: DataFrame with signal columns
        primary_signal_col: Column name for primary signals
        meta_decision_col: Column name for meta decisions (0/1)
        final_signal_col: Column name for final combined signals
        title: Plot title
        save_path: Path to save plot (temp file if None)
        max_days: Maximum days to plot (for readability)

    Returns:
        Path to saved plot file
    """
    import os
    import tempfile

    try:
        # Limit data for readability
        plot_data = data.copy()
        if len(plot_data) > max_days:
            plot_data = plot_data.tail(max_days)

        # Check required columns exist
        required_cols = [primary_signal_col, meta_decision_col, final_signal_col]
        missing_cols = [col for col in required_cols if col not in plot_data.columns]
        if missing_cols:
            print(f"Warning: Missing columns for signal comparison: {missing_cols}")
            return ""

        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        # Plot 1: Primary signals
        axes[0].step(plot_data.index, plot_data[primary_signal_col],
                    where='post', color='blue', linewidth=1.5, label='Primary Signal')
        axes[0].set_title('Primary Strategy Signals', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Signal', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim(-1.2, 1.2)

        # Plot 2: Meta decisions (0/1)
        axes[1].step(plot_data.index, plot_data[meta_decision_col],
                    where='post', color='red', linewidth=1.5, label='Meta Decision')
        axes[1].fill_between(plot_data.index, 0, plot_data[meta_decision_col],
                           step='post', alpha=0.3, color='red')
        axes[1].set_title('Meta-Model Decisions (1=Take Signal, 0=Skip)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Decision', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        axes[1].set_ylim(-0.1, 1.2)

        # Plot 3: Final combined signals
        axes[2].step(plot_data.index, plot_data[final_signal_col],
                    where='post', color='green', linewidth=1.5, label='Final Signal')
        axes[2].set_title('Final Combined Signals (Primary × Meta)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Signal', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        axes[2].set_ylim(-1.2, 1.2)

        # Plot 4: Signal statistics over time (rolling counts)
        window = min(30, len(plot_data) // 10)  # Adaptive window
        if window >= 5:
            primary_nonzero = plot_data[primary_signal_col].abs().rolling(window).sum()
            final_nonzero = plot_data[final_signal_col].abs().rolling(window).sum()

            axes[3].plot(plot_data.index, primary_nonzero,
                        color='blue', alpha=0.7, label=f'Primary Signals ({window}d)')
            axes[3].plot(plot_data.index, final_nonzero,
                        color='green', alpha=0.7, label=f'Final Signals ({window}d)')
            axes[3].fill_between(plot_data.index, primary_nonzero, final_nonzero,
                               alpha=0.2, color='orange', label='Filtered Out')

        axes[3].set_title(f'Signal Activity (Rolling {window}-day Count)', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Count', fontsize=11)
        axes[3].set_xlabel('Time', fontsize=11)
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.xticks(rotation=45)
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = os.path.join(tempfile.gettempdir(), 'signal_comparison.png')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    except Exception as e:
        print(f"Warning: Could not create signal comparison plot: {e}")
        return ""