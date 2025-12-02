"""
ICO/IEO Token Performance - RETAIL INVESTOR PERSPECTIVE
Shows returns from Day 7 (when retail can actually buy, after initial hype)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import time

# ICO/IEO tokens with retail-accessible launches
ICO_TOKENS = [
    # 2024 Launches
    {"ticker": "W", "symbol": "W-USD", "launch_date": "2024-04-03", "ico_price": 0.25, "name": "Wormhole"},
    {"ticker": "ENA", "symbol": "ENA-USD", "launch_date": "2024-04-02", "ico_price": 0.01, "name": "Ethena"},
    {"ticker": "STRK", "symbol": "STRK-USD", "launch_date": "2024-02-20", "ico_price": 2.00, "name": "Starknet"},
    {"ticker": "DYM", "symbol": "DYM-USD", "launch_date": "2024-02-06", "ico_price": 2.50, "name": "Dymension"},
    {"ticker": "MANTA", "symbol": "MANTA-USD", "launch_date": "2024-01-18", "ico_price": 0.50, "name": "Manta Network"},
    {"ticker": "JUP", "symbol": "JUP-USD", "launch_date": "2024-01-31", "ico_price": 0.40, "name": "Jupiter"},
    {"ticker": "ZK", "symbol": "ZK-USD", "launch_date": "2024-06-17", "ico_price": 0.20, "name": "ZKsync"},
    {"ticker": "BLAST", "symbol": "BLAST-USD", "launch_date": "2024-06-26", "ico_price": 0.015, "name": "Blast"},

    # 2023 Launches
    {"ticker": "TIA", "symbol": "TIA-USD", "launch_date": "2023-10-31", "ico_price": 2.00, "name": "Celestia"},
    {"ticker": "SEI", "symbol": "SEI-USD", "launch_date": "2023-08-15", "ico_price": 0.08, "name": "Sei Network"},
    {"ticker": "SUI", "symbol": "SUI-USD", "launch_date": "2023-05-03", "ico_price": 0.10, "name": "Sui"},
    {"ticker": "ARB", "symbol": "ARB-USD", "launch_date": "2023-03-23", "ico_price": 1.20, "name": "Arbitrum"},
    {"ticker": "BLUR", "symbol": "BLUR-USD", "launch_date": "2023-02-14", "ico_price": 0.30, "name": "Blur"},
    {"ticker": "WLD", "symbol": "WLD-USD", "launch_date": "2023-07-24", "ico_price": 0.15, "name": "Worldcoin"},
    {"ticker": "MNT", "symbol": "MNT-USD", "launch_date": "2023-07-17", "ico_price": 0.50, "name": "Mantle"},
    {"ticker": "PYTH", "symbol": "PYTH-USD", "launch_date": "2023-11-20", "ico_price": 0.25, "name": "Pyth Network"},

    # 2022 Launches
    {"ticker": "APT", "symbol": "APT-USD", "launch_date": "2022-10-19", "ico_price": 7.00, "name": "Aptos"},
    {"ticker": "OP", "symbol": "OP-USD", "launch_date": "2022-06-01", "ico_price": 0.73, "name": "Optimism"},
]

RETAIL_ENTRY_DAY = 7  # Days after listing when retail can realistically buy

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

def analyze_retail_performance():
    """Analyze from retail entry point (Day 7)"""
    results = []
    trajectories = {}

    print(f"\\nICO/IEO RETAIL INVESTOR ANALYSIS")
    print(f"Entry point: Day {RETAIL_ENTRY_DAY} (after initial volatility settles)")
    print("="*105)

    for i, token in enumerate(ICO_TOKENS, 1):
        ticker = token["ticker"]
        symbol = token["symbol"]
        launch_date = token["launch_date"]
        ico_price = token["ico_price"]
        name = token["name"]

        print(f"[{i:>2}/{len(ICO_TOKENS)}] {ticker:<5} ({name:<16})...", end=" ")

        df = fetch_price_data(symbol, launch_date)
        time.sleep(0.3)

        if df is not None and len(df) > RETAIL_ENTRY_DAY:
            # Key prices
            listing_price = df.iloc[0]['price']
            retail_entry_price = df.iloc[RETAIL_ENTRY_DAY]['price']
            final_price = df.iloc[-1]['price']

            # Calculate returns from retail entry
            retail_return = ((final_price - retail_entry_price) / retail_entry_price) * 100

            # What retail missed (ICO to retail entry)
            missed_gains = ((retail_entry_price - ico_price) / ico_price) * 100

            # Create trajectory from retail entry
            df['days_from_retail'] = (df['date'] - df.iloc[RETAIL_ENTRY_DAY]['date']).dt.days
            df['return_from_retail'] = ((df['price'] - retail_entry_price) / retail_entry_price) * 100

            # Only keep data from day 7 onwards
            df_retail = df[df['days_from_retail'] >= 0].copy()

            results.append({
                'ticker': ticker,
                'name': name,
                'ico_price': ico_price,
                'listing_price': listing_price,
                'retail_entry_price': retail_entry_price,
                'final_price': final_price,
                'missed_gains': missed_gains,
                'retail_return': retail_return,
                'days_held': len(df_retail)
            })

            trajectories[ticker] = df_retail[['days_from_retail', 'return_from_retail']].copy()

            print(f"${retail_entry_price:>7.2f} → ${final_price:>7.2f} ({retail_return:>+7.1f}%)")
        else:
            print("✗ Insufficient data")

    return pd.DataFrame(results), trajectories

def create_results_table(df):
    """Create results table"""
    df_sorted = df.sort_values('retail_return', ascending=False).reset_index(drop=True)

    print("\\n" + "="*115)
    print("RETAIL INVESTOR PERFORMANCE (Day 7 Entry → November 9, 2025)")
    print("="*115)
    print(f"{'#':<3} {'Ticker':<6} {'Name':<17} {'ICO $':<8} {'Day7 $':<8} {'Final $':<8} "
          f"{'Missed':<11} {'Retail ROI':<11}")
    print("-"*115)

    for idx, row in df_sorted.iterrows():
        print(f"{idx+1:<3} {row['ticker']:<6} {row['name']:<17} "
              f"${row['ico_price']:<7.2f} ${row['retail_entry_price']:<7.2f} ${row['final_price']:<7.2f} "
              f"{row['missed_gains']:>+9.0f}%  {row['retail_return']:>+9.1f}%")

    print("="*115)

    df_sorted.to_csv('ico_retail_performance.csv', index=False)
    print("\\n✓ Results saved to: ico_retail_performance.csv")

    return df_sorted

def create_clean_chart(trajectories):
    """Create clean chart from retail entry"""
    fig, ax = plt.subplots(figsize=(16, 10))

    for ticker, df in trajectories.items():
        ax.plot(df['days_from_retail'], df['return_from_retail'],
                label=ticker, alpha=0.8, linewidth=2.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Days Since Retail Entry (Day 7)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=14, fontweight='bold')
    ax.set_title('ICO Tokens - RETAIL INVESTOR PERFORMANCE\\n(Returns from Day 7 Entry to November 9, 2025)',
                 fontsize=17, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig('ico_retail_performance_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Chart saved to: ico_retail_performance_chart.png")
    plt.close()

def print_stats(df):
    """Print statistics"""
    print("\\n" + "="*115)
    print("SUMMARY STATISTICS")
    print("="*115)
    print(f"Total tokens analyzed: {len(df)}")
    print(f"\\nWhat Retail MISSED (ICO → Day 7):")
    print(f"  Average: {df['missed_gains'].mean():+.0f}%")
    print(f"  Median: {df['missed_gains'].median():+.0f}%")
    print(f"\\nRetail Performance (Day 7 → Nov 9, 2025):")
    print(f"  Average return: {df['retail_return'].mean():+.1f}%")
    print(f"  Median return: {df['retail_return'].median():+.1f}%")
    print(f"  Best: {df.iloc[0]['ticker']} ({df.iloc[0]['retail_return']:+.1f}%)")
    print(f"  Worst: {df.iloc[-1]['ticker']} ({df.iloc[-1]['retail_return']:+.1f}%)")
    print(f"  Positive returns: {len(df[df['retail_return'] > 0])}/{len(df)} "
          f"({len(df[df['retail_return'] > 0])/len(df)*100:.1f}%)")
    print(f"  Returns > 50%: {len(df[df['retail_return'] > 50])}/{len(df)}")
    print(f"  Returns < -50%: {len(df[df['retail_return'] < -50])}/{len(df)}")
    print("="*115)

# Run
print("\\n" + "="*115)
print("ICO/IEO TOKEN ANALYSIS - RETAIL INVESTOR PERSPECTIVE")
print("="*115)

results_df, trajectories = analyze_retail_performance()

if not results_df.empty:
    sorted_df = create_results_table(results_df)
    print_stats(sorted_df)
    create_clean_chart(trajectories)
    print("\\n✓ Analysis complete!")
else:
    print("\\n✗ No data")
