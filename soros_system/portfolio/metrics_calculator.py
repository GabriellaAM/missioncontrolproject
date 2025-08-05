import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

class PortfolioMetricsCalculator:
    """
    Specialized metrics calculator for portfolio performance analysis.
    """
    
    def __init__(self):
        """
        Initialize the PortfolioMetricsCalculator.
        """
        self.logger = logging.getLogger(__name__)
    
    def calculate_portfolio_metrics(self, portfolio_df, benchmark_df=None, risk_free_rate=0.0):
        """
        Calculate comprehensive performance metrics for a portfolio.
        
        Args:
            portfolio_df (pd.DataFrame): DataFrame with portfolio values and returns
                Required columns: 'date', 'portfolio_value', 'returns'
            benchmark_df (pd.DataFrame, optional): DataFrame with benchmark values and returns
                Required columns: 'date', 'value', 'returns'
            risk_free_rate (float, optional): Annual risk-free rate (default: 0.0)
            
        Returns:
            dict: Dictionary with performance metrics
        """
        if portfolio_df is None or portfolio_df.empty:
            self.logger.error("Portfolio DataFrame is empty or None")
            return {}
            
        if 'date' not in portfolio_df.columns or 'portfolio_value' not in portfolio_df.columns:
            self.logger.error("Portfolio DataFrame missing required columns")
            return {}
        
        # Ensure dates are in datetime format
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        
        # Calculate returns if not provided
        if 'returns' not in portfolio_df.columns:
            portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Calculate cumulative returns if not provided
        if 'cum_returns' not in portfolio_df.columns:
            portfolio_df['cum_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Get initial and final values
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        
        # Basic metrics
        metrics['initial_capital'] = initial_value
        metrics['final_capital'] = final_value
        metrics['total_return'] = ((final_value / initial_value) - 1) * 100
        
        # Calculate trading days and annualization factor
        trading_days = len(portfolio_df)
        days_between = (portfolio_df['date'].iloc[-1] - portfolio_df['date'].iloc[0]).days
        calendar_years = days_between / 365.25
        trading_days_per_year = trading_days / calendar_years if calendar_years > 0 else 252
        
        metrics['trading_days'] = trading_days
        metrics['calendar_days'] = days_between
        metrics['trading_days_per_year'] = trading_days_per_year
        
        # Calculate time-weighted return metrics
        returns = portfolio_df['returns'].fillna(0)
        
        # Annualized return
        annualized_return = ((1 + portfolio_df['cum_returns'].iloc[-1]) ** (1 / calendar_years)) - 1 if calendar_years > 0 else 0
        metrics['annualized_return'] = annualized_return * 100
        
        # Volatility metrics
        daily_std = returns.std()
        annualized_volatility = daily_std * np.sqrt(trading_days_per_year)
        metrics['daily_volatility'] = daily_std * 100
        metrics['annualized_volatility'] = annualized_volatility * 100
        
        # Downside deviation (for Sortino ratio)
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(trading_days_per_year) if len(negative_returns) > 0 else 0.0001
        
        # Risk-adjusted return metrics
        daily_risk_free = (1 + risk_free_rate) ** (1 / trading_days_per_year) - 1
        excess_return = annualized_return - risk_free_rate
        
        # Sharpe ratio
        metrics['sharpe_ratio'] = excess_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Sortino ratio
        metrics['sortino_ratio'] = excess_return / downside_deviation if downside_deviation != 0 else 0
        
        # Calmar ratio (return / max drawdown)
        max_drawdown = self.calculate_max_drawdown(portfolio_df)
        metrics['max_drawdown'] = max_drawdown * 100
        metrics['calmar_ratio'] = annualized_return / max_drawdown if max_drawdown != 0 else 0
        
        # Calculate metrics relative to benchmark if provided
        if benchmark_df is not None and not benchmark_df.empty:
            # Ensure benchmark has the same date range as portfolio
            benchmark_df = benchmark_df[benchmark_df['date'].isin(portfolio_df['date'])].copy()
            
            if not benchmark_df.empty:
                # Calculate benchmark returns if not provided
                if 'returns' not in benchmark_df.columns and 'value' in benchmark_df.columns:
                    benchmark_df['returns'] = benchmark_df['value'].pct_change()
                
                if 'returns' in benchmark_df.columns:
                    # Calculate beta and alpha
                    benchmark_returns = benchmark_df['returns'].fillna(0)
                    
                    # Beta calculation
                    if len(returns) == len(benchmark_returns):
                        covariance = np.cov(returns, benchmark_returns)[0, 1]
                        benchmark_variance = np.var(benchmark_returns)
                        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
                        metrics['beta'] = beta
                        
                        # Alpha calculation (annualized)
                        benchmark_annualized_return = ((1 + benchmark_returns.mean()) ** trading_days_per_year) - 1
                        alpha = annualized_return - (risk_free_rate + beta * (benchmark_annualized_return - risk_free_rate))
                        metrics['alpha'] = alpha * 100
                        
                        # Information ratio
                        tracking_error = (returns - beta * benchmark_returns).std() * np.sqrt(trading_days_per_year)
                        metrics['tracking_error'] = tracking_error * 100
                        metrics['information_ratio'] = alpha / tracking_error if tracking_error != 0 else 0
                        
                # Benchmark metrics
                if 'value' in benchmark_df.columns:
                    benchmark_initial = benchmark_df['value'].iloc[0]
                    benchmark_final = benchmark_df['value'].iloc[-1]
                    benchmark_return = ((benchmark_final / benchmark_initial) - 1) * 100
                    metrics['benchmark_return'] = benchmark_return
                    
                    # Outperformance
                    metrics['outperformance'] = metrics['total_return'] - benchmark_return
        
        return metrics
    
    def calculate_max_drawdown(self, portfolio_df):
        """
        Calculate maximum drawdown and related metrics.
        
        Args:
            portfolio_df (pd.DataFrame): DataFrame with portfolio values
                Required columns: 'portfolio_value' or 'cum_returns'
            
        Returns:
            float: Maximum drawdown as a decimal (not percentage)
        """
        if 'cum_returns' in portfolio_df.columns:
            # Use cumulative returns if available
            cum_returns = portfolio_df['cum_returns']
            running_max = np.maximum.accumulate(cum_returns + 1)
            drawdowns = (cum_returns + 1) / running_max - 1
        else:
            # Calculate drawdowns using portfolio values
            portfolio_value = portfolio_df['portfolio_value']
            running_max = np.maximum.accumulate(portfolio_value)
            drawdowns = portfolio_value / running_max - 1
            
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
        
        return max_drawdown
    
    def calculate_trade_metrics(self, trades_df):
        """
        Calculate comprehensive trade metrics.
        
        Args:
            trades_df (pd.DataFrame): DataFrame with trade data
                Required columns: entry_date, exit_date, entry_value, exit_value, etc.
            
        Returns:
            dict: Dictionary with trade metrics
        """
        if trades_df is None or trades_df.empty:
            return {
                'number_of_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_holding_days': 0,
                'total_costs': 0,
                'first_trade': None,
                'last_trade': None,
                'trading_days': 0
            }
        
        # Ensure all trades have required columns
        required_columns = ['entry_date', 'exit_date', 'entry_value', 'exit_value', 'trade_return']
        missing_columns = [col for col in required_columns if col not in trades_df.columns]
        
        if missing_columns:
            self.logger.error(f"Trades DataFrame missing required columns: {missing_columns}")
            # Try to calculate missing columns if possible
            if 'entry_price' in trades_df.columns and 'exit_price' in trades_df.columns:
                if 'trade_return' not in trades_df.columns:
                    trades_df['trade_return'] = (trades_df['exit_price'] / trades_df['entry_price']) - 1
        
        # Filter closed trades
        closed_trades = trades_df[trades_df['is_open'] == False] if 'is_open' in trades_df.columns else trades_df
        
        if closed_trades.empty:
            return {
                'number_of_trades': len(trades_df),
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_holding_days': 0,
                'total_costs': 0,
                'first_trade': None,
                'last_trade': None,
                'trading_days': 0
            }
        
        metrics = {}
        
        # Basic trade metrics
        metrics['number_of_trades'] = len(trades_df)
        metrics['number_of_closed_trades'] = len(closed_trades)
        
        # Win/loss metrics
        winning_trades = closed_trades[closed_trades['trade_return'] > 0]
        losing_trades = closed_trades[closed_trades['trade_return'] <= 0]
        
        metrics['win_rate'] = (len(winning_trades) / len(closed_trades)) * 100 if len(closed_trades) > 0 else 0
        metrics['number_of_winners'] = len(winning_trades)
        metrics['number_of_losers'] = len(losing_trades)
        
        # Average win/loss
        metrics['avg_win'] = winning_trades['trade_return'].mean() * 100 if not winning_trades.empty else 0
        metrics['avg_loss'] = losing_trades['trade_return'].mean() * 100 if not losing_trades.empty else 0
        
        # Profit factor
        total_profit = winning_trades['trade_return'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['trade_return'].sum()) if not losing_trades.empty else 0
        metrics['profit_factor'] = total_profit / total_loss if total_loss != 0 else 0
        
        # Maximum consecutive wins/losses
        if not closed_trades.empty and 'trade_return' in closed_trades.columns:
            trades_results = (closed_trades['trade_return'] > 0).astype(int)
            trade_changes = trades_results.diff().fillna(0) != 0
            trade_groups = trade_changes.cumsum()
            
            consecutive_wins = trades_results.groupby(trade_groups).sum()
            consecutive_losses = (1 - trades_results).groupby(trade_groups).sum()
            
            metrics['max_consecutive_wins'] = consecutive_wins.max() if not consecutive_wins.empty else 0
            metrics['max_consecutive_losses'] = consecutive_losses.max() if not consecutive_losses.empty else 0
        
        # Time metrics
        if 'entry_date' in closed_trades.columns and 'exit_date' in closed_trades.columns:
            # Convert to datetime if not already
            closed_trades['entry_date'] = pd.to_datetime(closed_trades['entry_date'])
            closed_trades['exit_date'] = pd.to_datetime(closed_trades['exit_date'])
            
            # Calculate holding days
            if 'holding_days' not in closed_trades.columns:
                closed_trades['holding_days'] = (closed_trades['exit_date'] - closed_trades['entry_date']).dt.days
            
            metrics['avg_holding_days'] = closed_trades['holding_days'].mean()
            metrics['min_holding_days'] = closed_trades['holding_days'].min()
            metrics['max_holding_days'] = closed_trades['holding_days'].max()
            
            # Trading period
            metrics['first_trade'] = closed_trades['entry_date'].min()
            metrics['last_trade'] = closed_trades['exit_date'].max()
            
            # Calculate trading days
            all_dates = pd.concat([
                closed_trades['entry_date'].dropna(),
                closed_trades['exit_date'].dropna()
            ]).drop_duplicates()
            
            metrics['trading_days'] = len(all_dates)
        
        # Cost metrics
        cost_columns = ['entry_cost', 'exit_cost']
        for col in cost_columns:
            if col in closed_trades.columns:
                if col == 'entry_cost':
                    metrics['total_entry_costs'] = closed_trades[col].sum()
                elif col == 'exit_cost':
                    metrics['total_exit_costs'] = closed_trades[col].sum()
        
        metrics['total_costs'] = metrics.get('total_entry_costs', 0) + metrics.get('total_exit_costs', 0)
        
        # Value metrics
        value_columns = ['entry_value', 'exit_value']
        for col in value_columns:
            if col in closed_trades.columns:
                if col == 'entry_value':
                    metrics['total_entry_value'] = closed_trades[col].sum()
                elif col == 'exit_value':
                    metrics['total_exit_value'] = closed_trades[col].sum()
        
        # Calculate average trade return
        if 'trade_return' in closed_trades.columns:
            metrics['avg_trade_return'] = closed_trades['trade_return'].mean() * 100
            metrics['median_trade_return'] = closed_trades['trade_return'].median() * 100
        
        return metrics
    
    def plot_portfolio_performance(self, results_df, benchmark_df=None, title='Portfolio Performance', figsize=(12, 8)):
        """
        Create a comprehensive performance chart.
        
        Args:
            results_df (pd.DataFrame): DataFrame with portfolio results
            benchmark_df (pd.DataFrame, optional): DataFrame with benchmark results
            title (str, optional): Chart title
            figsize (tuple, optional): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if results_df is None or results_df.empty:
            self.logger.error("Results DataFrame is empty or None")
            return None
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Ensure date is in datetime format
        results_df['date'] = pd.to_datetime(results_df['date'])
        
        # Plot portfolio value
        axes[0].plot(results_df['date'], results_df['portfolio_value'], label='Portfolio Value', linewidth=2)
        
        # Plot benchmark if provided
        if benchmark_df is not None and not benchmark_df.empty:
            # Ensure date is in datetime format
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            
            # Plot benchmark on same scale
            if 'value' in benchmark_df.columns:
                # Scale benchmark to start at same value as portfolio
                scale_factor = results_df['portfolio_value'].iloc[0] / benchmark_df['value'].iloc[0]
                axes[0].plot(benchmark_df['date'], benchmark_df['value'] * scale_factor, 
                         label='Benchmark (Scaled)', linewidth=2, linestyle='--')
        
        # Plot cumulative returns on second axis
        if 'cum_returns' in results_df.columns:
            axes[1].plot(results_df['date'], results_df['cum_returns'] * 100, label='Portfolio', linewidth=2)
            
            if benchmark_df is not None and 'cum_returns' in benchmark_df.columns:
                axes[1].plot(benchmark_df['date'], benchmark_df['cum_returns'] * 100, 
                          label='Benchmark', linewidth=2, linestyle='--')
        
        # Style the charts
        axes[0].set_title(title, fontsize=14)
        axes[0].set_ylabel('Value ($)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper left')
        
        axes[1].set_ylabel('Return (%)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper left')
        
        plt.tight_layout()
        
        return fig
    
    def plot_drawdowns(self, results_df, top_n=5, figsize=(12, 8)):
        """
        Plot drawdowns chart.
        
        Args:
            results_df (pd.DataFrame): DataFrame with portfolio results
            top_n (int, optional): Number of largest drawdowns to highlight
            figsize (tuple, optional): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if results_df is None or results_df.empty:
            self.logger.error("Results DataFrame is empty or None")
            return None
        
        # Ensure date is in datetime format
        results_df['date'] = pd.to_datetime(results_df['date'])
        
        # Calculate drawdowns
        if 'cum_returns' in results_df.columns:
            cum_returns = results_df['cum_returns']
        else:
            cum_returns = (results_df['portfolio_value'] / results_df['portfolio_value'].iloc[0]) - 1
        
        running_max = np.maximum.accumulate(cum_returns + 1)
        drawdowns = (cum_returns + 1) / running_max - 1
        
        # Create a copy of results with drawdowns
        results_with_dd = results_df.copy()
        results_with_dd['drawdown'] = drawdowns
        
        # Find the largest drawdowns
        underwater = drawdowns.copy()
        for i in range(1, len(underwater)):
            if underwater[i] == 0:
                underwater[i] = 0
            else:
                underwater[i] = min(underwater[i-1], underwater[i])
        
        # Find the start and end of each drawdown
        drawdown_info = []
        current_drawdown = 0
        drawdown_start = None
        
        for i, date in enumerate(results_with_dd['date']):
            dd = drawdowns.iloc[i]
            
            if dd < current_drawdown:
                current_drawdown = dd
                
            if dd == 0 and drawdown_start is not None:
                # End of a drawdown
                drawdown_info.append({
                    'start_date': drawdown_start,
                    'end_date': date,
                    'max_drawdown': abs(current_drawdown),
                    'duration': (date - drawdown_start).days
                })
                drawdown_start = None
                current_drawdown = 0
            elif dd < 0 and drawdown_start is None:
                # Start of a new drawdown
                drawdown_start = date
        
        # If we're still in a drawdown at the end
        if drawdown_start is not None:
            drawdown_info.append({
                'start_date': drawdown_start,
                'end_date': results_with_dd['date'].iloc[-1],
                'max_drawdown': abs(current_drawdown),
                'duration': (results_with_dd['date'].iloc[-1] - drawdown_start).days
            })
        
        # Sort drawdowns by severity
        drawdown_info = sorted(drawdown_info, key=lambda x: x['max_drawdown'], reverse=True)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
        
        # Plot cumulative returns
        axes[0].plot(results_with_dd['date'], cum_returns * 100, label='Cumulative Return (%)', linewidth=2)
        
        # Plot drawdowns
        axes[1].fill_between(results_with_dd['date'], drawdowns * 100, 0, color='red', alpha=0.3, label='Drawdowns')
        
        # Highlight top N drawdowns
        colors = plt.cm.tab10.colors
        for i, dd in enumerate(drawdown_info[:min(top_n, len(drawdown_info))]):
            mask = (results_with_dd['date'] >= dd['start_date']) & (results_with_dd['date'] <= dd['end_date'])
            axes[1].fill_between(
                results_with_dd.loc[mask, 'date'], 
                results_with_dd.loc[mask, 'drawdown'] * 100, 
                0, 
                color=colors[i % len(colors)], 
                alpha=0.5, 
                label=f"DD #{i+1}: {dd['max_drawdown']*100:.1f}%, {dd['duration']} days"
            )
        
        # Style the charts
        axes[0].set_title('Portfolio Cumulative Return and Drawdowns', fontsize=14)
        axes[0].set_ylabel('Return (%)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='lower right')
        
        # Invert y-axis for drawdowns
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        return fig
    
    def plot_monthly_returns_heatmap(self, results_df, figsize=(12, 8)):
        """
        Create a heatmap of monthly returns.
        
        Args:
            results_df (pd.DataFrame): DataFrame with portfolio results
            figsize (tuple, optional): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if results_df is None or results_df.empty or 'returns' not in results_df.columns:
            self.logger.error("Results DataFrame missing required data")
            return None
        
        # Ensure date is in datetime format
        results_df['date'] = pd.to_datetime(results_df['date'])
        
        # Extract month and year
        results_df['year'] = results_df['date'].dt.year
        results_df['month'] = results_df['date'].dt.month
        
        # Calculate monthly returns
        monthly_returns = results_df.groupby(['year', 'month'])['returns'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        
        # Pivot to create year x month matrix
        monthly_matrix = monthly_returns.pivot(index='year', columns='month', values='returns')
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(monthly_matrix.values, cmap='RdYlGn')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Return (%)')
        
        # Label axes
        ax.set_title('Monthly Returns Heatmap', fontsize=14)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        # Set tick labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(np.arange(len(month_names)))
        ax.set_xticklabels(month_names)
        
        ax.set_yticks(np.arange(len(monthly_matrix.index)))
        ax.set_yticklabels(monthly_matrix.index)
        
        # Add text annotations
        for i in range(len(monthly_matrix.index)):
            for j in range(len(month_names)):
                if j < monthly_matrix.shape[1] and not np.isnan(monthly_matrix.values[i, j]):
                    text = ax.text(j, i, f"{monthly_matrix.values[i, j]*100:.1f}%",
                                  ha="center", va="center", 
                                  color="black" if abs(monthly_matrix.values[i, j]) < 0.1 else "white")
        
        plt.tight_layout()
        
        return fig 