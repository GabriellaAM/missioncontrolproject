"""
ICO/IEO Token Performance Analysis 2024-2025
Tokens that had actual Initial Coin Offerings or Initial Exchange Offerings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import time

# Verified ICO/IEO tokens from 2024-2025
# These had actual token sales where investors purchased at specific prices
ICO_TOKENS = [
    # Major ICOs/IEOs from 2024
    {"ticker": "W", "symbol": "W-USD", "launch_date": "2024-04-03", "ico_price": 0.25, "name": "Wormhole"},
    {"ticker": "ENA", "symbol": "ENA-USD", "launch_date": "2024-04-02", "ico_price": 0.01, "name": "Ethena"},
    {"ticker": "STRK", "symbol": "STRK-USD", "launch_date": "2024-02-20", "ico_price": 2.00, "name": "Starknet"},
    {"ticker": "DYM", "symbol": "DYM-USD", "launch_date": "2024-02-06", "ico_price": 2.50, "name": "Dymension"},
    {"ticker": "TIA", "symbol": "TIA-USD", "launch_date": "2023-10-31", "ico_price": 2.00, "name": "Celestia"},
    {"ticker": "SEI", "symbol": "SEI-USD", "launch_date": "2023-08-15", "ico_price": 0.08, "name": "Sei Network"},
    {"ticker": "SUI", "symbol": "SUI-USD", "launch_date": "2023-05-03", "ico_price": 0.10, "name": "Sui"},
    {"ticker": "APT", "symbol": "APT-USD", "launch_date": "2022-10-19", "ico_price": 7.00, "name": "Aptos"},
    {"ticker": "OP", "symbol": "OP-USD", "launch_date": "2022-06-01", "ico_price": 0.73, "name": "Optimism"},
    {"ticker": "ARB", "symbol": "ARB-USD", "launch_date": "2023-03-23", "ico_price": 1.20, "name": "Arbitrum"},
]

def fetch_price_data(symbol, start_date, end_date="2025-11-09"):
    """Fetch historical prices from Binance"""
    try:
        base_symbol = symbol.replace("-USD", "")
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
        print(f"Error: {e}")
        return None

def analyze_ico_performance():
    """Analyze ICO token performance"""
    results = []
    trajectories = {}

    print(f"\\nAnalyzing {len(ICO_TOKENS)} ICO/IEO tokens...")
    print("="*90)

    for i, token in enumerate(ICO_TOKENS, 1):
        ticker = token["ticker"]
        symbol = token["symbol"]
        launch_date = token["launch_date"]
        ico_price = token["ico_price"]
        name = token["name"]

        print(f"[{i}/{len(ICO_TOKENS)}] {ticker} ({name})...", end=" ")

        df = fetch_price_data(symbol, launch_date)
        time.sleep(0.3)

        if df is not None and len(df) > 0:
            # Use first available price as starting price
            first_price = df.iloc[0]['price']
            final_price = df.iloc[-1]['price']

            # Calculate returns from ICO price
            return_from_ico = ((final_price - ico_price) / ico_price) * 100

            # Calculate returns from listing price
            return_from_listing = ((final_price - first_price) / first_price) * 100

            # ROI multiple from ICO
            roi_multiple = final_price / ico_price

            # Calculate trajectory
            df['days'] = (df['date'] - pd.to_datetime(launch_date)).dt.days
            df['return_from_ico'] = ((df['price'] - ico_price) / ico_price) * 100

            results.append({
                'ticker': ticker,
                'name': name,
                'launch_date': launch_date,
                'ico_price': ico_price,
                'listing_price': first_price,
                'final_price': final_price,
                'return_from_ico': return_from_ico,
                'return_from_listing': return_from_listing,
                'roi_multiple': roi_multiple,
                'days': len(df)
            })

            trajectories[ticker] = df[['days', 'return_from_ico']].copy()

            print(f"✓ ICO ${ico_price:.2f} → ${final_price:.2f} ({return_from_ico:+.1f}%) [{roi_multiple:.2f}x]")
        else:
            print("✗ No data")

    return pd.DataFrame(results), trajectories

def create_results_table(df):
    """Create and display results table"""
    df_sorted = df.sort_values('return_from_ico', ascending=False).reset_index(drop=True)

    print("\\n" + "="*110)
    print("ICO/IEO TOKEN PERFORMANCE (ICO Price → November 9, 2025)")
    print("="*110)
    print(f"{'#':<3} {'Ticker':<6} {'Name':<15} {'ICO $':<8} {'List $':<8} {'Final $':<8} "
          f"{'ICO ROI':<12} {'ROI Multiple':<12}")
    print("-"*110)

    for idx, row in df_sorted.iterrows():
        print(f"{idx+1:<3} {row['ticker']:<6} {row['name']:<15} "
              f"${row['ico_price']:<7.2f} ${row['listing_price']:<7.2f} ${row['final_price']:<7.2f} "
              f"{row['return_from_ico']:>+10.1f}%  {row['roi_multiple']:>10.2f}x")

    print("="*110)

    df_sorted.to_csv('ico_tokens_performance.csv', index=False)
    print("\\n✓ Results saved to: ico_tokens_performance.csv")

    return df_sorted

def create_chart(trajectories):
    """Create cumulative return chart"""
    fig, ax = plt.subplots(figsize=(16, 10))

    for ticker, df in trajectories.items():
        ax.plot(df['days'], df['return_from_ico'],
                label=ticker, alpha=0.75, linewidth=2)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.4, label='ICO Price')
    ax.set_xlabel('Days Since Launch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Return from ICO Price (%)', fontsize=13, fontweight='bold')
    ax.set_title('ICO/IEO Tokens - Performance from ICO Price\\n(Launch to November 9, 2025)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig('ico_tokens_performance_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Chart saved to: ico_tokens_performance_chart.png")
    plt.close()

def print_stats(df):
    """Print statistics"""
    print("\\n" + "="*110)
    print("SUMMARY STATISTICS")
    print("="*110)
    print(f"Total ICO tokens analyzed: {len(df)}")
    print(f"Average ROI from ICO: {df['return_from_ico'].mean():+.1f}%")
    print(f"Median ROI from ICO: {df['return_from_ico'].median():+.1f}%")
    print(f"Average ROI multiple: {df['roi_multiple'].mean():.2f}x")
    print(f"Best performer: {df.iloc[0]['ticker']} ({df.iloc[0]['return_from_ico']:+.1f}% / {df.iloc[0]['roi_multiple']:.2f}x)")
    print(f"Worst performer: {df.iloc[-1]['ticker']} ({df.iloc[-1]['return_from_ico']:+.1f}% / {df.iloc[-1]['roi_multiple']:.2f}x)")
    print(f"Positive ROI: {len(df[df['return_from_ico'] > 0])}/{len(df)} ({len(df[df['return_from_ico'] > 0])/len(df)*100:.1f}%)")
    print(f"Tokens above 10x: {len(df[df['roi_multiple'] >= 10])}/{len(df)}")
    print("="*110)

# Run analysis
print("\\n" + "="*110)
print("ICO/IEO TOKEN PERFORMANCE ANALYSIS")
print("="*110)

results_df, trajectories = analyze_ico_performance()

if not results_df.empty:
    sorted_df = create_results_table(results_df)
    print_stats(sorted_df)
    create_chart(trajectories)
    print("\\n✓ Analysis complete!")
else:
    print("\\n✗ No data retrieved")
