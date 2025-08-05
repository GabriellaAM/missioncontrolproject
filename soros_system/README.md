# SOROS System: Advanced Crypto Trend Analysis Framework

## Overview

The SOROS (Systematic, Objective Risk and Opportunity Signaling) System is a comprehensive framework for analyzing cryptocurrency market trends, creating trading portfolios, and backtesting investment strategies. It employs technical indicators, Markov models, and statistical methods to classify market regimes and generate investment signals.

## Latest Enhancements

The system has been significantly enhanced with the following new features:

- **Detailed Asset Information**: Get comprehensive asset-level data including raw data, processed indicators, and trend metrics
- **Trend Performance Comparison**: Compare performance metrics (Sharpe, Sortino, skew) across different trend types
- **Enhanced Portfolio Creation**: Create portfolios with specific short/medium/long term trend conditions using text descriptions or numeric values
- **Comprehensive Backtesting Reports**: Get detailed performance statistics, trade history, and signal analysis
- **Improved Data Structure Access**: Structured methods to access portfolio signals, asset metrics, and daily results
- **Advanced Performance Metrics**: Additional metrics like median returns, return skew, and detailed trade statistics

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
soros_system/
├── data/                  # Data handling components
│   ├── data_loader.py     # Asset data loading and preprocessing
│   └── ssr_data.py        # Stock-to-Flow Ratio data handling
├── indicators/            # Technical indicators
│   ├── rsi.py             # Relative Strength Index calculations
│   ├── moving_averages.py # Moving average calculations
│   └── trend_classifier.py # Trend classification logic
├── analysis/              # Analysis components
│   ├── metrics.py         # Performance metrics calculations
│   └── markov.py          # Markov chain analysis
├── portfolio/             # Portfolio components
│   ├── portfolio_manager.py # Portfolio creation and management
│   └── backtest.py        # Backtesting engine
├── visualization/         # Visualization components
│   └── plotters.py        # Performance and trend visualization
└── main.py                # Main coordinator class
```

## Key Features

- **Comprehensive Technical Analysis**: Calculate RSI, moving averages, and rate-of-change indicators to identify market trends
- **Trend Classification**: Classify market trends as Strong Bull, Weak Bull, Neutral, Weak Bear, or Strong Bear
- **BTC-Adjusted Analysis**: Analyze asset performance in both USD and BTC terms
- **Markov Analysis**: Use Markov chains to analyze transition probabilities between market states
- **Portfolio Management**: Create and manage portfolios with custom criteria
- **Backtesting**: Test portfolio performance against historical data
- **Visualization**: Generate interactive plots of portfolio performance, asset performance, and trend distributions

## How to Use

### Basic Usage

```python
from soros_system.main import TrendAnalyzer

# Initialize the system
analyzer = TrendAnalyzer(
    asset_ids=['bitcoin', 'ethereum', 'solana'],
    data_path='data/crypto/',
    btc_data_path='data/crypto/bitcoin_candles.csv'
)

# Analyze assets
results = analyzer.analyze_multiple_assets()

# Get latest trend classifications
latest_trends = analyzer.get_latest_trends()
print(latest_trends)

# Create a portfolio
analyzer.create_portfolio(
    'bull_portfolio',
    btc_trend_gating=0,  # Only invest when BTC trend is neutral or bullish
    usd_conditions=[1, 2],  # Only assets with bullish USD trends
    btc_conditions=[0, 1, 2],  # Assets with neutral or bullish BTC trends
    max_assets=5  # Maximum number of assets to include
)

# Backtest the portfolio
backtest_results = analyzer.backtest_portfolio(
    'bull_portfolio',
    start_date='2022-01-01',
    end_date='2023-01-01',
    initial_capital=10000
)

# Visualize results
analyzer.plot_portfolio_performance(backtest_results)
analyzer.plot_asset_performance(backtest_results)
```

### Advanced Usage

```python
# Create a portfolio with specific trend term conditions using text descriptions
analyzer.create_portfolio(
    'trend_term_portfolio',
    btc_trend_gating=1,
    usd_conditions={'Short Term': ['Strong Bull'], 'Medium Term': ['Weak Bull', 'Strong Bull']},
    btc_conditions={'Short Term': ['Neutral', 'Weak Bull', 'Strong Bull']},
    rsi_conditions_usd=True,
    use_volatility_filter=True,
    volatility_weight=0.7
)

# Get detailed asset information
eth_info = analyzer.get_asset_information('ethereum')
print(eth_info['trend_metrics_usd'])

# Compare trend performance across different trend types
eth_trend_comparison = analyzer.compare_trend_performance('ethereum', price_type='USD')
print(eth_trend_comparison[['Description', 'Sharpe', 'Sortino', 'Avg_Return']])

# Get detailed portfolio results
backtest_results = analyzer.backtest_portfolio('trend_term_portfolio', '2022-01-01', '2023-01-01', 10000)

# Get portfolio signal details
signals_df = analyzer.get_portfolio_signals_df(backtest_results)
print(signals_df.head())

# Get asset performance metrics
asset_metrics = analyzer.get_portfolio_asset_metrics(backtest_results)
for asset, metrics in asset_metrics.items():
    print(f"{asset}: {metrics['total_return']}, Sharpe: {metrics['sharpe_ratio']}")

# Get daily portfolio results
daily_results = analyzer.get_portfolio_daily_results(backtest_results)
print(daily_results.head())

# Get recommended assets on each day
rec_assets = analyzer.get_recommended_assets_table(backtest_results)
print(rec_assets.head())
```

## Requirements

- Python 3.8+
- pandas
- numpy
- plotly
- pycoingecko
- tabulate
- scipy

## Integration with Other Components

The SOROS system is designed to integrate with:

1. **MarkovVolatility Module**: For advanced volatility regime analysis
2. **FRED Data**: For macroeconomic overlay analysis
3. **On-chain Metrics**: For additional market insight

## Example Scripts and Notebooks

The system includes example scripts and Jupyter notebooks to help you get started:

- `soros_system_example.py`: A comprehensive example script demonstrating all major functionality
- `soros_system_example.ipynb`: A Jupyter notebook with interactive examples and visualizations

## Contributing

Contributions to the SOROS system are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your feature or bug fix
4. Add tests for your changes
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the work of George Soros on reflexivity in markets
- Utilizes concepts from Andrew Hyde's Ensemble Bayesian HMM approach
- Incorporates methods from Piotr Pomorski's work on Markov Switching with KAMA 