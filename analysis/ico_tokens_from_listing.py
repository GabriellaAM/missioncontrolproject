"""
ICO/IEO Token Performance - From Listing Price
Shows performance after tokens became publicly tradable
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import time

# ICO tokens with listing dates
ICO_TOKENS = [
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

def analyze_from_listing():
    """Analyze performance from listing price"""
    results = []
    trajectories = {}

    print(f"\\nAnalyzing {len(ICO_TOKENS)} ICO tokens from listing price...")
    print("="*100)

    for i, token in enumerate(ICO_TOKENS, 1):
        ticker = token["ticker"]
        symbol = token["symbol"]
        launch_date = token["launch_date"]
        ico_price = token["ico_price"]
        name = token["name"]

        print(f"[{i}/{len(ICO_TOKENS)}] {ticker:>5} ({name:<15})...", end=" ")

        df = fetch_price_data(symbol, launch_date)
        time.sleep(0.3)

        if df is not None and len(df) > 0:
            # Use first available price as listing price
            listing_price = df.iloc[0]['price']
            final_price = df.iloc[-1]['price']

            # Calculate returns from listing
            return_from_listing = ((final_price - listing_price) / listing_price) * 100

            # Calculate ICO to listing return (initial pump/dump)
            ico_to_listing = ((listing_price - ico_price) / ico_price) * 100

            # Calculate trajectory from listing
            df['days'] = (df['date'] - pd.to_datetime(launch_date)).dt.days
            df['return_from_listing'] = ((df['price'] - listing_price) / listing_price) * 100

            results.append({
                'ticker': ticker,
                'name': name,
                'launch_date': launch_date,
                'ico_price': ico_price,
                'listing_price': listing_price,
                'final_price': final_price,
                'ico_to_listing': ico_to_listing,
                'return_from_listing': return_from_listing,
                'total_days': len(df)
            })

            trajectories[ticker] = df[['days', 'return_from_listing']].copy()

            print(f"List ${listing_price:>6.2f} → ${final_price:>6.2f} ({return_from_listing:>+7.1f}%)")
        else:
            print("✗ No data")

    return pd.DataFrame(results), trajectories

def create_results_table(df):
    """Create results table"""
    df_sorted = df.sort_values('return_from_listing', ascending=False).reset_index(drop=True)

    print("\\n" + "="*110)
    print("ICO TOKEN PERFORMANCE FROM LISTING PRICE (Listing → November 9, 2025)")
    print("="*110)
    print(f"{'#':<3} {'Ticker':<7} {'Name':<16} {'ICO $':<9} {'List $':<9} {'Final $':<9} "
          f"{'ICO→List':<12} {'List→Now':<12}")
    print("-"*110)

    for idx, row in df_sorted.iterrows():
        print(f"{idx+1:<3} {row['ticker']:<7} {row['name']:<16} "
              f"${row['ico_price']:<8.2f} ${row['listing_price']:<8.2f} ${row['final_price']:<8.2f} "
              f"{row['ico_to_listing']:>+10.1f}%  {row['return_from_listing']:>+10.1f}%")

    print("="*110)

    df_sorted.to_csv('ico_tokens_from_listing.csv', index=False)
    print("\\n✓ Results saved to: ico_tokens_from_listing.csv")

    return df_sorted

def create_clean_chart(trajectories):
    """Create clean cumulative return chart from listing"""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot each token's trajectory from listing
    for ticker, df in trajectories.items():
        ax.plot(df['days'], df['return_from_listing'],
                label=ticker, alpha=0.8, linewidth=2.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Listing Price')
    ax.set_xlabel('Days Since Listing', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return from Listing (%)', fontsize=14, fontweight='bold')
    ax.set_title('ICO/IEO Tokens - Performance from Listing Price\\n(Public Market Performance to November 9, 2025)',
                 fontsize=17, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_xlim(left=0)

    # Add shading
    ax.fill_between(ax.get_xlim(), 0, 100, alpha=0.05, color='green', label='Profit Zone')
    ax.fill_between(ax.get_xlim(), 0, -100, alpha=0.05, color='red', label='Loss Zone')

    plt.tight_layout()
    plt.savefig('ico_tokens_from_listing_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Chart saved to: ico_tokens_from_listing_chart.png")
    plt.close()

def print_stats(df):
    """Print statistics"""
    print("\\n" + "="*110)
    print("SUMMARY STATISTICS (Post-Listing Performance)")
    print("="*110)
    print(f"Total tokens: {len(df)}")
    print(f"\\nICO → Listing (Initial Pump):")
    print(f"  Average: {df['ico_to_listing'].mean():+.1f}%")
    print(f"  Median: {df['ico_to_listing'].median():+.1f}%")
    print(f"\\nListing → Nov 9, 2025 (Market Performance):")
    print(f"  Average: {df['return_from_listing'].mean():+.1f}%")
    print(f"  Median: {df['return_from_listing'].median():+.1f}%")
    print(f"  Best: {df.iloc[0]['ticker']} ({df.iloc[0]['return_from_listing']:+.1f}%)")
    print(f"  Worst: {df.iloc[-1]['ticker']} ({df.iloc[-1]['return_from_listing']:+.1f}%)")
    print(f"  Positive returns: {len(df[df['return_from_listing'] > 0])}/{len(df)} "
          f"({len(df[df['return_from_listing'] > 0])/len(df)*100:.1f}%)")
    print("="*110)

# Run analysis
print("\\n" + "="*110)
print("ICO/IEO TOKEN ANALYSIS - PERFORMANCE FROM LISTING PRICE")
print("="*110)

results_df, trajectories = analyze_from_listing()

if not results_df.empty:
    sorted_df = create_results_table(results_df)
    print_stats(sorted_df)
    create_clean_chart(trajectories)
    print("\\n✓ Analysis complete!")
else:
    print("\\n✗ No data retrieved")
