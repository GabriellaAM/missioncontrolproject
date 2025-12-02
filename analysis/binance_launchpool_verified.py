"""
Binance Launchpool Analysis - Verified tokens with real historical data
Uses yfinance for actual daily price data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import time

# VERIFIED Binance Launchpool tokens - only confirmed Launchpool projects
# Source: Binance official announcements
VERIFIED_LAUNCHPOOL = [
    # 2024 Projects
    {"ticker": "PIXEL", "symbol": "PIXEL-USD", "launch_date": "2024-02-19"},  # #46
    {"ticker": "PORTAL", "symbol": "PORTAL-USD", "launch_date": "2024-02-29"},  # #47
    {"ticker": "AEVO", "symbol": "AEVO-USD", "launch_date": "2024-03-13"},  # #48
    {"ticker": "ETHFI", "symbol": "ETHFI-USD", "launch_date": "2024-03-18"},  # #49
    {"ticker": "ENA", "symbol": "ENA-USD", "launch_date": "2024-04-02"},  # #50
    {"ticker": "SAGA", "symbol": "SAGA-USD", "launch_date": "2024-04-09"},  # #51
    {"ticker": "OMNI", "symbol": "OMNI-USD", "launch_date": "2024-04-17"},  # #52
    {"ticker": "REZ", "symbol": "REZ-USD", "launch_date": "2024-04-30"},  # #53
    {"ticker": "BB", "symbol": "BB-USD", "launch_date": "2024-05-13"},  # #54
    {"ticker": "NOT", "symbol": "NOT-USD", "launch_date": "2024-05-16"},  # #55
    {"ticker": "IO", "symbol": "IO-USD", "launch_date": "2024-06-11"},  # #56
    {"ticker": "ZK", "symbol": "ZK-USD", "launch_date": "2024-06-17"},  # #57
    {"ticker": "ZRO", "symbol": "ZRO-USD", "launch_date": "2024-06-20"},  # #58
    {"ticker": "LISTA", "symbol": "LISTA-USD", "launch_date": "2024-06-20"},  # #59
    {"ticker": "BANANA", "symbol": "BANANA-USD", "launch_date": "2024-07-18"},  # #60
    {"ticker": "DOGS", "symbol": "DOGS-USD", "launch_date": "2024-08-26"},  # #61
    {"ticker": "HMSTR", "symbol": "HMSTR-USD", "launch_date": "2024-09-26"},  # #62
    {"ticker": "CATI", "symbol": "CATI-USD", "launch_date": "2024-09-20"},  # #63
    {"ticker": "EIGEN", "symbol": "EIGEN-USD", "launch_date": "2024-10-01"},  # #64
    {"ticker": "SCR", "symbol": "SCR-USD", "launch_date": "2024-10-22"},  # #65
    {"ticker": "VANA", "symbol": "VANA-USD", "launch_date": "2024-12-16"},  # #66

    # 2025 Projects
    {"ticker": "BIO", "symbol": "BIO-USD", "launch_date": "2025-01-03"},  # #67
]

def fetch_crypto_prices_binance(symbol, start_date, end_date="2025-11-09"):
    """
    Fetch historical prices using Binance API directly
    """
    try:
        # Remove -USD suffix for Binance API
        base_symbol = symbol.replace("-USD", "")

        import requests
        url = "https://api.binance.com/api/v3/klines"

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        params = {
            "symbol": f"{base_symbol}USDT",
            "interval": "1d",
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000
        }

        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['price'] = df['close'].astype(float)
                return df[['date', 'price']].reset_index(drop=True)

        return None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def analyze_token_performance(end_date="2025-11-09"):
    """
    Analyze real historical performance
    """
    results = []
    price_trajectories = {}

    print(f"\nFetching real historical data for {len(VERIFIED_LAUNCHPOOL)} verified Binance Launchpool tokens...")
    print("="*80)

    for i, token in enumerate(VERIFIED_LAUNCHPOOL, 1):
        ticker = token["ticker"]
        symbol = token["symbol"]
        launch_date = token["launch_date"]

        print(f"[{i}/{len(VERIFIED_LAUNCHPOOL)}] {ticker}...", end=" ")

        # Fetch historical data
        df = fetch_crypto_prices_binance(symbol, launch_date, end_date)
        time.sleep(0.3)  # Rate limiting

        if df is not None and len(df) > 0:
            # Get first and last prices
            starting_price = df.iloc[0]['price']
            final_price = df.iloc[-1]['price']

            # Calculate returns
            cumulative_return = ((final_price - starting_price) / starting_price) * 100

            # Calculate day count
            df['days'] = (df['date'] - pd.to_datetime(launch_date)).dt.days
            df['cumulative_return'] = ((df['price'] - starting_price) / starting_price) * 100

            results.append({
                'ticker': ticker,
                'launch_date': launch_date,
                'starting_price': starting_price,
                'final_price': final_price,
                'cumulative_return': cumulative_return,
                'data_points': len(df)
            })

            price_trajectories[ticker] = df[['days', 'cumulative_return']].copy()

            print(f"✓ ${starting_price:.4f} → ${final_price:.4f} ({cumulative_return:+.2f}%) [{len(df)} days]")
        else:
            print(f"✗ No data")

    return pd.DataFrame(results), price_trajectories

def create_results_table(df):
    """Create and display results table"""
    df_sorted = df.sort_values('cumulative_return', ascending=False).reset_index(drop=True)

    print("\n" + "="*100)
    print("VERIFIED BINANCE LAUNCHPOOL TOKEN PERFORMANCE (Launch → November 9, 2025)")
    print("="*100)
    print(f"{'#':<4} {'Ticker':<8} {'Launch':<12} {'Start Price':<14} {'Final Price':<14} {'Return':<12} {'Days':<6}")
    print("-"*100)

    for idx, row in df_sorted.iterrows():
        print(f"{idx+1:<4} {row['ticker']:<8} {row['launch_date']:<12} "
              f"${row['starting_price']:<13.4f} ${row['final_price']:<13.4f} "
              f"{row['cumulative_return']:>+10.2f}%  {row['data_points']:<6.0f}")

    print("="*100)

    df_sorted.to_csv('binance_launchpool_verified_results.csv', index=False)
    print("\n✓ Results saved to: binance_launchpool_verified_results.csv")

    return df_sorted

def create_trajectory_chart(price_trajectories):
    """Create chart with real price trajectories"""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot each token's actual trajectory
    for ticker, df in price_trajectories.items():
        ax.plot(df['days'], df['cumulative_return'],
                label=ticker, alpha=0.75, linewidth=2)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.4)
    ax.set_xlabel('Days Since Launch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=13, fontweight='bold')
    ax.set_title('Binance Launchpool Tokens - Real Historical Performance\\n(Launch Date to November 9, 2025)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig('binance_launchpool_real_trajectories.png', dpi=300, bbox_inches='tight')
    print("✓ Chart saved to: binance_launchpool_real_trajectories.png")
    plt.close()

def print_statistics(df):
    """Print summary statistics"""
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"Total verified tokens: {len(df)}")
    print(f"Average return: {df['cumulative_return'].mean():.2f}%")
    print(f"Median return: {df['cumulative_return'].median():.2f}%")
    print(f"Std deviation: {df['cumulative_return'].std():.2f}%")
    print(f"Best performer: {df.iloc[0]['ticker']} ({df.iloc[0]['cumulative_return']:+.2f}%)")
    print(f"Worst performer: {df.iloc[-1]['ticker']} ({df.iloc[-1]['cumulative_return']:+.2f}%)")
    print(f"Positive returns: {len(df[df['cumulative_return'] > 0])}/{len(df)} "
          f"({len(df[df['cumulative_return'] > 0])/len(df)*100:.1f}%)")
    print(f"Returns < -50%: {len(df[df['cumulative_return'] < -50])}/{len(df)}")
    print("="*100)

# Run analysis
print("\\n" + "="*100)
print("BINANCE LAUNCHPOOL VERIFIED TOKEN ANALYSIS")
print("="*100)

results_df, trajectories = analyze_token_performance()

if not results_df.empty:
    sorted_df = create_results_table(results_df)
    print_statistics(sorted_df)
    create_trajectory_chart(trajectories)
    print("\\n✓ Analysis complete!")
else:
    print("\\n✗ No data retrieved")
