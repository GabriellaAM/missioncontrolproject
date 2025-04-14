# SOROS: Signal-Based Trading System

## Overview

SOROS is a decision support system that helps traders make data-driven decisions based on technical and fundamental signals. The system is designed to be modular and extensible, allowing traders to easily add new signals and strategies.

## Key Features

- **Data Loading**: Load price data, macro data, and on-chain data from various sources
- **Indicator Calculation**: Calculate technical indicators like trend, RSI, volatility, etc.
- **Signal Generation**: Generate binary signals (0/1) based on indicators and rules
- **Portfolio Management**: Create and backtest trading strategies based on signals
- **Event-Driven Architecture**: Track signal activations as events with appropriate holding periods

## Architecture

The system is composed of several modules:

- **Core**: Core components like `AssetData` and `SignalData` containers
- **Signals**: Signal definitions and rules for generating trading signals
- **Indicators**: Technical indicators and calculations
- **Portfolio**: Portfolio management and backtesting tools
- **Data**: Data loading and preprocessing utilities

## Usage

### Basic Workflow

```python
# Initialize portfolio analyzer
analyzer = PortfolioAnalyzer(
    data_dir='/path/to/data',
    data_path='/path/to/data/micro/candleData/',
    btc_data_path='/path/to/data/micro/candleData/bitcoin_candles.csv',
    ssr_data_path='/path/to/data/onchainData/BTC_SSR.csv',
    market_data_path='/path/to/data/micro/assetData/',
    asset_ids=['BTC', 'ETH', 'BNB']
)

# Set signal parameters
analyzer.set_signal_parameters('RSI_Oversold', weight=1.0, decay_period=14)
analyzer.set_signal_parameters('Trend_Following', weight=1.5, decay_period=21)
analyzer.set_signal_parameters('Volatility_Breakout', weight=0.8, decay_period=7)

# Load data
analyzer.load_data()

# Register signals
analyzer.register_signals('BTC', ['RSI_Oversold', 'Trend_Following', 'Volatility_Breakout'])
analyzer.register_signals('ETH', ['RSI_Oversold', 'Trend_Following'])

# Get current recommendations
recommendations = analyzer.get_current_recommendations()

# Run backtest
backtest_results = analyzer.run_backtest(
    backtest_name='test_backtest',
    start_date='2021-01-01',
    end_date='2022-01-01',
    initial_capital=10000.0
)
```

### Signal Configuration

Signals are configured with:

- **Weight**: The importance of the signal (higher weight = stronger influence)
- **Decay Period**: How long a signal remains active after triggering

### Extending with New Signals

To add a new signal:

1. Create a new signal class in the `signals` directory
2. Register the signal in the `signal_registry.py` file
3. Set appropriate weights and decay periods for the signal
4. Use it in your trading strategies

## Backtesting

The system supports backtesting through:

- **Event-Driven Backtest**: Tracks signal events and calculates performance
- **Portfolio Simulation**: Simulates portfolio allocation and rebalancing

## Getting Started

1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Set up your data directory
4. Set signal parameters for your preferred signals
5. Run backtests or get current recommendations 