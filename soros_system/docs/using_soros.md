# Using the SOROS System

This guide provides comprehensive instructions on how to use the refactored SOROS system for cryptocurrency trend analysis, portfolio creation, and backtesting.

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Latest Enhancements](#latest-enhancements)
6. [Modules Deep Dive](#modules-deep-dive)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## System Overview

The SOROS system is a modular cryptocurrency analysis framework that integrates:

- Technical indicator calculation (RSI, Moving Averages)
- Trend classification (for both USD and BTC pairs)
- Markov chain analysis for state transitions
- Portfolio creation and management
- Performance backtesting
- Visualization tools

The system uses a modular architecture with clear separation of concerns:

```
soros_system/
├── data/                 # Data handling modules
├── indicators/           # Technical indicators
├── analysis/             # Trend classification and metrics
├── portfolio/            # Portfolio management and backtesting
├── visualization/        # Performance and trend visualization
└── main.py               # Main TrendAnalyzer class
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Required libraries: pandas, numpy, plotly, scipy, pycoingecko

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/missioncontrol.git
   cd missioncontrol
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

The main interface to the SOROS system is the `TrendAnalyzer` class. Here's how to get started:

```python
from soros_system.main import TrendAnalyzer

# Initialize the analyzer
analyzer = TrendAnalyzer(
    asset_ids=['bitcoin', 'ethereum', 'binancecoin'],
    data_path='data/',
    btc_data_path='data/bitcoin_candles.csv',
    use_btc_adjusted=True,
    verbose=True,
    lookback_days=90
)

# Analyze assets
results = analyzer.analyze_multiple_assets()

# Get the latest trend classifications
latest_trends = analyzer.get_latest_trends()
print(latest_trends[['asset_id', 'date', 'Overall_Trend_USD', 'Overall_Trend_BTC']])
```

## Advanced Usage

### Creating and Backtesting Portfolios

```python
# Create a portfolio with specific conditions
analyzer.create_portfolio(
    portfolio_name='my_portfolio',
    btc_trend_gating=0,           # Only active when BTC trend is neutral or bullish
    usd_conditions=[2],           # Only include assets with strong bullish USD trends
    btc_conditions=[1, 2],        # Include assets with weak or strong bullish BTC trends
    rsi_conditions_usd=True,      # Use RSI filter for USD
    rsi_conditions_btc=True,      # Use RSI filter for BTC
    use_btc_rsi_signal=True,      # Use BTC RSI as additional signal
    use_volatility_filter=True    # Filter by volatility state
)

# Backtest the portfolio
results = analyzer.backtest_portfolio(
    portfolio_name='my_portfolio',
    start_date='2022-01-01',
    end_date='2023-01-01',
    initial_capital=10000
)

# Visualize the results
analyzer.plot_portfolio_performance(results)
analyzer.plot_asset_performance(results)
```

### Analyzing Market Trends

```python
# Plot trend distribution
analyzer.plot_trend_distribution('bitcoin', trend_type='USD')

# Plot transition matrix
analyzer.plot_transition_matrix('bitcoin', trend_type='USD')
```

### Working with Markov Models

```python
# Train and save a Markov model
analyzer.create_markov_model('my_model')

# Load an existing model
analyzer.load_markov_model('my_model')

# Get volatility state probabilities
probs = analyzer.get_volatility_state('bitcoin')
print(f"Probability of high volatility: {probs.get('high', 0):.2f}")
```

## Latest Enhancements

The SOROS system has been enhanced with several new features designed to provide more comprehensive analysis capabilities and detailed portfolio reporting.

### Accessing Asset-Level Information

You can now access comprehensive information about each asset:

```python
# Get raw price data
btc_raw = analyzer.get_asset_raw_data('bitcoin')

# Get processed data with indicators and trend classifications
btc_processed = analyzer.get_asset_processed_data('bitcoin')

# Get trend metrics for USD
btc_metrics_usd = analyzer.get_asset_trend_metrics('bitcoin', 'USD')

# Get all information in one call
btc_info = analyzer.get_asset_information('bitcoin')
```

### Comparing Trend Performance

Compare performance metrics across different trend classifications:

```python
# Compare USD trend performance for Ethereum
eth_trends = analyzer.compare_trend_performance('ethereum', price_type='USD')
print(eth_trends[['Description', 'Avg_Return', 'Median_Return', 'Return_Skew', 'Sharpe', 'Sortino']])
```

### Enhanced Portfolio Creation

Create portfolios with specific trend term conditions using text descriptions or numeric values:

```python
# Using text descriptions
analyzer.create_portfolio(
    portfolio_name='text_based_portfolio',
    usd_conditions={'Short Term': ['Strong Bull'], 'Medium Term': ['Weak Bull', 'Strong Bull']},
    btc_conditions={'Short Term': ['Neutral', 'Weak Bull']}
)

# Using numeric values
analyzer.create_portfolio(
    portfolio_name='numeric_portfolio',
    usd_conditions={'Short Term': [2], 'Medium Term': [1, 2]},
    btc_conditions={'Short Term': [0, 1]}
)
```

### Comprehensive Backtest Results

Get detailed backtest results with structured access to signals, metrics, and performance:

```python
# Run backtest
results = analyzer.backtest_portfolio('my_portfolio', '2022-01-01', '2023-01-01', 10000)

# Get detailed performance metrics
metrics = results['metrics']
print(f"Total Return: {metrics['total_return']}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
print(f"Sortino Ratio: {metrics['sortino_ratio']}")
print(f"Max Drawdown: {metrics['max_drawdown']}")
print(f"Annualized Volatility: {metrics['annualized_volatility']}")
print(f"Win Rate: {metrics['win_rate']}")

# Get portfolio signals
signals_df = analyzer.get_portfolio_signals_df(results)

# Get daily portfolio results
daily_results = analyzer.get_portfolio_daily_results(results)

# Get asset performance metrics
asset_metrics = analyzer.get_portfolio_asset_metrics(results)

# Get recommended assets for each day
rec_assets = analyzer.get_recommended_assets_table(results)
```

### Trade History Analysis

Analyze detailed trade history for each asset:

```python
# Get asset trade history
asset_trades = results['asset_trade_history']['ethereum']

# Create a DataFrame of trades
import pandas as pd
trades_df = pd.DataFrame(asset_trades)
print(trades_df[['trade_id', 'entry_date', 'exit_date', 'holding_days', 'entry_price', 'exit_price', 'price_return', 'trade_return']])
```

### Asset Performance Metrics

Get comprehensive performance metrics for each asset in the portfolio:

```python
# Get asset performance metrics
for asset, metrics in asset_metrics.items():
    print(f"\n{asset} Performance:")
    print(f"Total Return: {metrics['total_return']}")
    print(f"Max Drawdown: {metrics['max_drawdown']}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']}")
    print(f"Number of Trades: {metrics['number_of_trades']}")
    print(f"Win Rate: {metrics['win_rate']}")
    print(f"Avg Holding Days: {metrics['avg_holding_days']}")
```

## Modules Deep Dive

### Data Module

The data module handles data loading and preprocessing:

```python
from soros_system.data.data_loader import DataLoader

# Create a data loader instance
data_loader = DataLoader(
    data_path='data/',
    asset_ids=['bitcoin', 'ethereum'],
    verbose=True
)

# Load data for a specific asset
btc_data = data_loader.load_asset_data('bitcoin')
```

### Indicators Module

The indicators module calculates technical indicators:

```python
from soros_system.indicators.rsi_calculator import RSICalculator
from soros_system.indicators.moving_average_calculator import MovingAverageCalculator

# Calculate RSI
rsi_calc = RSICalculator()
df_with_rsi = rsi_calc.calculate_rsi(df, window=14)

# Calculate Moving Averages
ma_calc = MovingAverageCalculator()
df_with_ma = ma_calc.calculate_moving_averages(df, windows=[20, 50, 200])
```

### Analysis Module

The analysis module provides trend classification and metrics calculation:

```python
from soros_system.analysis.trend_classifier import TrendClassifier
from soros_system.analysis.metrics_calculator import MetricsCalculator

# Classify trends
classifier = TrendClassifier()
classified_data = classifier.classify_trends(df)

# Calculate metrics
metrics_calc = MetricsCalculator()
sharpe = metrics_calc.calculate_sharpe_ratio(df, risk_free_rate=0.0)
```

### Portfolio Module

The portfolio module handles portfolio management and backtesting:

```python
from soros_system.portfolio.portfolio_manager import PortfolioManager
from soros_system.portfolio.portfolio_backtester import PortfolioBacktester

# Create a portfolio
portfolio_mgr = PortfolioManager()
portfolio_mgr.create_portfolio(
    name='test_portfolio',
    conditions={...},
    asset_data={...}
)

# Backtest a portfolio
backtester = PortfolioBacktester()
results = backtester.backtest_portfolio(
    portfolio='test_portfolio',
    data={...},
    start_date='2022-01-01',
    end_date='2023-01-01',
    initial_capital=10000
)
```

### Visualization Module

The visualization module provides tools for visualizing performance and trends:

```python
from soros_system.visualization.portfolio_visualizer import PortfolioVisualizer

# Create a visualizer
visualizer = PortfolioVisualizer()

# Plot portfolio performance
visualizer.plot_portfolio_performance(backtest_results)

# Plot trend distribution
visualizer.plot_trend_distribution(trend_data)
```

## Troubleshooting

### Common Issues

1. **Missing Data**
   
   If you encounter errors about missing columns or data:
   ```
   Check that your data files exist in the specified paths.
   Make sure data files have the expected column names.
   ```

2. **BTC-Adjusted Data Issues**
   
   If BTC-adjusted calculations fail:
   ```
   Ensure your Bitcoin data file is correctly formatted and accessible.
   Check that the timestamps align between your asset data and Bitcoin data.
   ```

3. **Portfolio Backtesting Errors**
   
   If backtesting fails:
   ```
   Verify that your date range exists in the historical data.
   Check that assets in your portfolio have data for the specified period.
   ```

4. **Trend Classification Warnings**
   
   If you see warnings about "No BTC price column found" or "Not enough data points":
   ```
   This typically occurs for assets with limited history or missing BTC-denominated data.
   The system will attempt to continue with available data.
   ```

### Debug Mode

Enable verbose mode for detailed logging:

```python
analyzer = TrendAnalyzer(
    # ... other parameters ...
    verbose=True
)
```

## Best Practices

1. **Data Preparation**
   - Ensure data is clean and properly formatted
   - Use consistent date ranges for all assets
   - Include sufficient historical data for lookback calculations

2. **Portfolio Creation**
   - Start with a base trend-following portfolio
   - Add additional filters incrementally to understand their impact
   - Test different combinations of conditions
   - Use trend term specifications to target specific time horizons

3. **Backtesting**
   - Test across different market conditions (bull/bear markets)
   - Use reasonable initial capital values
   - Compare against benchmark portfolios
   - Analyze asset-level performance to understand portfolio drivers

4. **Performance Analysis**
   - Look beyond total return to metrics like Sharpe, Sortino, and drawdown
   - Analyze individual trades to understand strategy strengths/weaknesses
   - Use signal analysis to identify which components drive performance
   - Compare trend performance to identify optimal trading conditions

5. **Performance Optimization**
   - Use caching for repeated calculations
   - Limit the number of assets analyzed if performance is an issue
   - Consider using a subset of indicators for initial screening

6. **Model Versioning**
   - Save and version Markov models
   - Document the conditions used for each model
   - Track performance metrics across model versions 