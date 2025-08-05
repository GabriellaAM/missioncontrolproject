"""
Evaluation Metrics for Trading Strategies
"""

import numpy as np
from typing import Union, Optional


def calculate_sharpe_ratio(returns: np.ndarray, 
                          risk_free_rate: float = 0,
                          periods_per_year: int = 365) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_sortino_ratio(returns: np.ndarray,
                           risk_free_rate: float = 0,
                           periods_per_year: int = 365) -> float:
    """
    Calculate annualized Sortino ratio (uses downside deviation).
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_deviation


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from returns.
    
    Args:
        returns: Array of returns
    
    Returns:
        Maximum drawdown (as negative percentage, e.g., -0.25 for 25% drawdown)
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    
    return np.min(drawdown)


def calculate_calmar_ratio(returns: np.ndarray,
                          periods_per_year: int = 365) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Calmar ratio
    """
    max_dd = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return np.inf if np.mean(returns) > 0 else 0.0
    
    annualized_return = np.mean(returns) * periods_per_year
    return -annualized_return / max_dd  # Negative because max_dd is negative


def calculate_annualized_volatility(returns: np.ndarray,
                                   periods_per_year: int = 365) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Annualized volatility
    """
    if len(returns) == 0:
        return 0.0
    
    return np.std(returns) * np.sqrt(periods_per_year)


def calculate_profit_factor(returns: np.ndarray, 
                           signals: Optional[np.ndarray] = None) -> float:
    """
    Calculate profit factor for a trading strategy.
    
    Args:
        returns: Array of returns
        signals: Array of trading signals (+1 for long, -1 for short, 0 for no position)
                If None, assumes all returns are strategy returns
    
    Returns:
        Profit factor (gross profit / gross loss)
    """
    if signals is not None:
        strategy_returns = returns * signals
    else:
        strategy_returns = returns
    
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_information_ratio(strategy_returns: np.ndarray, 
                               benchmark_returns: np.ndarray,
                               periods_per_year: int = 365) -> float:
    """
    Calculate Information Ratio (excess return / tracking error).
    
    Args:
        strategy_returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns (underlying asset)
        periods_per_year: Number of periods in a year (365 for daily)
    
    Returns:
        Annualized Information Ratio
    """
    if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("Strategy and benchmark returns must have the same length")
    
    # Calculate excess returns
    excess_returns = strategy_returns - benchmark_returns
    
    # Calculate tracking error (standard deviation of excess returns)
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    # Annualize both excess return and tracking error
    annualized_excess_return = np.mean(excess_returns) * periods_per_year
    annualized_tracking_error = tracking_error * np.sqrt(periods_per_year)
    
    return annualized_excess_return / annualized_tracking_error


def calculate_avg_time_underwater(returns: np.ndarray) -> float:
    """
    Calculate average time underwater (periods below peak).
    
    Args:
        returns: Array of returns
    
    Returns:
        Average number of periods underwater
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    
    # Identify underwater periods
    underwater = cumulative_returns < running_max
    
    # Count consecutive underwater periods
    underwater_periods = []
    current_period = 0
    
    for is_underwater in underwater:
        if is_underwater:
            current_period += 1
        else:
            if current_period > 0:
                underwater_periods.append(current_period)
                current_period = 0
    
    # Add last period if still underwater
    if current_period > 0:
        underwater_periods.append(current_period)
    
    if len(underwater_periods) == 0:
        return 0.0
    
    return np.mean(underwater_periods)


def calculate_all_metrics(returns: np.ndarray,
                         signals: Optional[np.ndarray] = None,
                         benchmark_returns: Optional[np.ndarray] = None,
                         risk_free_rate: float = 0,
                         periods_per_year: int = 365) -> dict:
    """
    Calculate all evaluation metrics at once.
    
    Args:
        returns: Array of returns
        signals: Optional array of trading signals
        benchmark_returns: Optional array of benchmark returns for Information Ratio
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
    
    Returns:
        Dictionary containing all metrics
    """
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(returns, periods_per_year),
        'max_drawdown': calculate_max_drawdown(returns),
        'annualized_volatility': calculate_annualized_volatility(returns, periods_per_year),
        'profit_factor': calculate_profit_factor(returns, signals),
        'avg_time_underwater': calculate_avg_time_underwater(returns),
        'total_return': (1 + returns).prod() - 1 if len(returns) > 0 else 0.0,
        'annualized_return': np.mean(returns) * periods_per_year if len(returns) > 0 else 0.0
    }
    
    # Add Information Ratio if benchmark returns are provided
    if benchmark_returns is not None:
        metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns, periods_per_year)
    
    return metrics