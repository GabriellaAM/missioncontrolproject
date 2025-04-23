# Portfolio Management in MissionControl

This directory contains portfolio definitions and tools for portfolio management within the MissionControl system. The system provides a robust framework for creating, testing, and analyzing portfolios using various signal strategies.

## Portfolio Types

There are two main approaches to portfolio management in this system:

1. **Predefined Portfolios**: JSON files in this directory that define portfolio criteria and assets
2. **Dynamic Signal-Based Portfolios**: Built programmatically using the PortfolioAnalyzer

## Portfolio Structure

Each portfolio JSON file may contain:

```json
{
  "criteria": {
    "usd_conditions": ["bullish", "neutral"],
    "btc_conditions": ["bullish"],
    "use_volatility_filter": true,
    "signal_threshold": 60,
    "max_assets": 5
  },
  "assets": ["bitcoin", "ethereum", "solana", "..."],
  "created_at": "2023-04-01T12:00:00",
  "updated_at": "2023-04-01T12:00:00"
}
```

## Creating a Portfolio

### Using the PortfolioManager

```python
from soros_system.portfolio.portfolio_manager import PortfolioManager

pm = PortfolioManager()

# Create a portfolio that only trades in bullish BTC conditions
pm.create_portfolio(
    portfolio_name="bullish_btc_portfolio",
    classified_data=analyzer.data_cache,  # Pass data from an analyzer
    usd_conditions=["bullish", "neutral"],  # Accept bullish and neutral USD trends
    btc_conditions=["bullish"],  # Only accept bullish BTC trends
    btc_trend_gating=1,  # Minimum BTC trend value to allow trading
    use_volatility_filter=True,  # Filter by volatility
    volatility_weight=0.5,  # Weight for volatility
    rsi_conditions_usd=True,  # Apply RSI filter for USD trends
    rsi_conditions_btc=True,  # Apply RSI filter for BTC trends
    use_ssr_signal=True,  # Use SSR signal
    signal_threshold=60,  # Require 60% of signals to be positive
    max_assets=5,  # Maximum 5 assets in the portfolio
    use_signal_selector=True  # Use dynamic signal selection for optimal signals
)
```

### Using the PortfolioAnalyzer (Recommended)

The PortfolioAnalyzer provides a unified interface for both signal calculation and portfolio management:

```python
from soros_system.core.portfolio_analyzer import PortfolioAnalyzer

analyzer = PortfolioAnalyzer(
    data_dir='data',
    data_path='data/micro/candleData/',
    btc_data_path='data/micro/candleData/bitcoin_candles.csv',
    ssr_data_path='data/onchainData/BTC_SSR.csv',
    market_data_path='data/micro/assetData/',
    asset_ids=['bitcoin', 'ethereum', 'solana', 'cardano']
)

# Load data and register signals
analyzer.load_data(start_date='2020-01-01')
analyzer.register_signals_parallel(asset_ids=['bitcoin', 'ethereum', 'solana', 'cardano'])
```

## Signal-Based Backtesting

The system now includes a specialized backtesting method that focuses on specific signal types (Donchian, RSI, and Bull/Bear Variance signals) with dual USD/BTC evaluation:

```python
backtest_results = analyzer.run_signal_backtest(
    backtest_name="My_Signal_Backtest",
    assets=['bitcoin', 'ethereum', 'solana', 'cardano'],
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=10000.0,
    trade_cost=0.001,      # 0.1% trading fee
    slippage_pct=0.001,    # 0.1% slippage
    include_metrics=True,  # Calculate comprehensive metrics
    plot_results=True      # Generate performance charts
)
```

This backtest method:

1. Focuses specifically on Donchian, RSI, and Bull Variance signals
2. For non-Bitcoin assets, requires positive signals in BOTH USD and BTC terms
3. For Bitcoin, only evaluates USD signals
4. Provides comprehensive performance metrics and visualization
5. Makes the trades table available at the asset level inside the analyzer

## Key Features of Signal-Based Backtests

The `run_signal_backtest` method provides the following key features:

- **Dual Currency Evaluation**: For altcoins, positions are only entered when signals are positive in both USD and BTC terms
- **Comprehensive Metrics**: For each asset, calculates:
  - Return vs. buy & hold
  - Max drawdown
  - 365-day realized volatility
  - Sharpe ratio
  - Win rate
  - Accuracy, precision, recall, and F1 score (trading signal quality metrics)
- **Visualization**: Automatically generates:
  - Portfolio equity curve
  - Drawdown chart
  - Asset returns comparison
  - Detailed metrics table

## Example

See `examples/signal_backtest_example.py` for a complete example of how to use the signal-based portfolio backtesting functionality.

## Portfolio Recommendations

The system can also generate current portfolio recommendations based on signal analysis:

```python
# Get current trade recommendations
recommendations = analyzer.get_current_recommendations()

# Print recommendations
for asset_id, rec in recommendations.items():
    decision = "BUY" if rec['decision'] == 1 else "HOLD/SELL"
    confidence = rec.get('confidence', 0) * 100
    print(f"{asset_id}: {decision} (Confidence: {confidence:.1f}%)")
```

## Integration with Google Sheets (Future Development)

The system is designed to be extended to pull portfolio allocations from Google Sheets:

```python
from portfolios.getPositionsFromSheets import get_positions

positions = get_positions()
```

The imported positions could then be compared against recommended allocations from the system. 