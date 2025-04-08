from soros_system.core.portfolio_analyzer import PortfolioAnalyzer
from soros_system.indicators.markov_volatility import MarkovVolModel

# Set paths and initialize analyzer
data_path = '/Users/valter.rebelo/MissionControl/data/micro/candleData/'
btc_data_path = '/Users/valter.rebelo/MissionControl/data/micro/candleData/bitcoin_candles.csv'
ssr_data_path = '/Users/valter.rebelo/MissionControl/data/onchainData/BTC_SSR.csv'
market_data_path = '/Users/valter.rebelo/MissionControl/data/micro/assetData/'

try:
    markov_vol_model = MarkovVolModel.load_model_by_timestamp('20250325_144631')
except Exception as e:
    print(f"Could not load Markov model: {e}")
    markov_vol_model = None
    print("Continuing without Markov model")

# Define assets to analyze
assets = ['bitcoin', 'ethereum', 'solana']
print(f"Analyzing assets: {', '.join(assets)}")

# Initialize the analyzer with all required paths
analyzer = PortfolioAnalyzer(
    data_dir='/Users/valter.rebelo/MissionControl/data',
    data_path=data_path,
    btc_data_path=btc_data_path,
    ssr_data_path=ssr_data_path,
    market_data_path=market_data_path,
    markov_analyzer=markov_vol_model,
    signal_threshold=0.0,
    lookback_days='all',
    asset_ids=assets,
    verbose=False  # Set to True for more detailed logs
)

# Register signals
for asset_id in assets:
    analyzer.register_signals(asset_id=asset_id)

# Evaluate signals
analyzer.evaluate_signals()

# Run backtest
results = analyzer.run_backtest("my_backtest", assets=assets)

# Get today's recommendations
recommendations = analyzer.get_current_recommendations()
print("\nCurrent Asset Recommendations:")
for asset_id, rec in recommendations.items():
    decision = "BUY" if rec['decision'] == 1 else "HOLD/SELL"
    print(f"{asset_id}: {decision} - Active signals: {rec['active_signals_count']}")

# Get portfolio recommendations
weights = analyzer.get_portfolio_recommendations()
print("\nPortfolio Allocations:")
for asset_id, weight in weights.items():
    print(f"{asset_id}: {weight:.2%}") 