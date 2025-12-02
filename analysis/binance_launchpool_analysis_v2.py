"""
Binance Launchpool Token Performance Analysis - V2
Uses CoinGecko data via their API with proper rate limiting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import sys

# Comprehensive list of Binance Launchpool tokens with their launch dates
LAUNCHPOOL_TOKENS = [
    # Recent 2024-2025 tokens with confirmed dates
    {"ticker": "ETHFI", "coingecko_id": "ether-fi", "launch_date": "2024-03-18"},
    {"ticker": "SAGA", "coingecko_id": "saga", "launch_date": "2024-04-09"},
    {"ticker": "OMNI", "coingecko_id": "omni-network", "launch_date": "2024-04-17"},
    {"ticker": "REZ", "coingecko_id": "renzo", "launch_date": "2024-04-30"},
    {"ticker": "BB", "coingecko_id": "bouncebit", "launch_date": "2024-05-13"},
    {"ticker": "NOT", "coingecko_id": "notcoin", "launch_date": "2024-05-16"},
    {"ticker": "IO", "coingecko_id": "io", "launch_date": "2024-06-11"},
    {"ticker": "ZK", "coingecko_id": "zksync", "launch_date": "2024-06-17"},
    {"ticker": "ZRO", "coingecko_id": "layerzero", "launch_date": "2024-06-20"},
    {"ticker": "LISTA", "coingecko_id": "lista-dao", "launch_date": "2024-06-20"},
    {"ticker": "BANANA", "coingecko_id": "banana-gun", "launch_date": "2024-07-18"},
    {"ticker": "DOGS", "coingecko_id": "dogs", "launch_date": "2024-08-26"},
    {"ticker": "CATI", "coingecko_id": "catizen", "launch_date": "2024-09-20"},
    {"ticker": "HMSTR", "coingecko_id": "hamster-kombat", "launch_date": "2024-09-26"},
    {"ticker": "EIGEN", "coingecko_id": "eigenlayer", "launch_date": "2024-10-01"},
    {"ticker": "SCR", "coingecko_id": "scroll", "launch_date": "2024-10-22"},
    {"ticker": "USUAL", "coingecko_id": "usual", "launch_date": "2024-11-19"},
    {"ticker": "VANA", "coingecko_id": "vana", "launch_date": "2024-12-16"},
    {"ticker": "BIO", "coingecko_id": "bio-protocol", "launch_date": "2025-01-03"},
    {"ticker": "XAI", "coingecko_id": "xai-games", "launch_date": "2024-01-09"},
    {"ticker": "MANTA", "coingecko_id": "manta-network", "launch_date": "2024-01-18"},
    {"ticker": "ALT", "coingecko_id": "altlayer", "launch_date": "2024-01-25"},
    {"ticker": "JUP", "coingecko_id": "jupiter-exchange-solana", "launch_date": "2024-01-31"},
    {"ticker": "PYTH", "coingecko_id": "pyth-network", "launch_date": "2023-11-20"},
    {"ticker": "PIXEL", "coingecko_id": "pixels", "launch_date": "2024-02-19"},
    {"ticker": "PORTAL", "coingecko_id": "portal", "launch_date": "2024-02-29"},
    {"ticker": "AEVO", "coingecko_id": "aevo", "launch_date": "2024-03-13"},
    {"ticker": "W", "coingecko_id": "wormhole", "launch_date": "2024-04-03"},
    {"ticker": "ENA", "coingecko_id": "ethena", "launch_date": "2024-04-02"},
]

def fetch_price_from_coingecko(coin_id, date_str):
    """
    Fetch historical price from CoinGecko for a specific date
    """
    try:
        # Convert date string to DD-MM-YYYY format for CoinGecko
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%d-%m-%Y")

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/history"
        params = {"date": formatted_date, "localization": "false"}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if "market_data" in data and "current_price" in data["market_data"]:
                return data["market_data"]["current_price"].get("usd")
        return None
    except Exception as e:
        print(f"Error fetching price for {coin_id} on {date_str}: {str(e)}")
        return None

def get_current_price(coin_id):
    """
    Get current price from CoinGecko
    """
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin_id, "vs_currencies": "usd"}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return data.get(coin_id, {}).get("usd")
        return None
    except Exception as e:
        print(f"Error fetching current price for {coin_id}: {str(e)}")
        return None

def analyze_tokens(tokens, end_date="2025-11-09"):
    """
    Analyze token performance from launch to end date
    """
    results = []

    for i, token in enumerate(tokens):
        ticker = token["ticker"]
        coin_id = token["coingecko_id"]
        launch_date = token["launch_date"]

        print(f"[{i+1}/{len(tokens)}] Analyzing {ticker}...")

        # Get launch price (1 day after launch to allow for price stabilization)
        launch_date_obj = datetime.strptime(launch_date, "%Y-%m-%d")
        day_after_launch = (launch_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

        starting_price = fetch_price_from_coingecko(coin_id, day_after_launch)
        time.sleep(1.2)  # Rate limiting

        if starting_price is None:
            # Try launch date itself
            starting_price = fetch_price_from_coingecko(coin_id, launch_date)
            time.sleep(1.2)

        # Get end date price (November 9, 2025)
        final_price = fetch_price_from_coingecko(coin_id, end_date)
        time.sleep(1.2)

        if final_price is None:
            # Try getting current price if end date fails
            final_price = get_current_price(coin_id)
            time.sleep(1.2)

        if starting_price and final_price:
            cumulative_return = ((final_price - starting_price) / starting_price) * 100

            results.append({
                "ticker": ticker,
                "launch_date": launch_date,
                "starting_price": starting_price,
                "final_price": final_price,
                "cumulative_return": cumulative_return
            })

            print(f"  ✓ {ticker}: ${starting_price:.4f} → ${final_price:.4f} ({cumulative_return:+.2f}%)")
        else:
            print(f"  ✗ {ticker}: Could not fetch price data")

    return pd.DataFrame(results)

def create_results_table(df, output_file="binance_launchpool_results.csv"):
    """
    Create formatted results table
    """
    if df.empty:
        print("No data to create table")
        return

    # Sort by cumulative return (descending)
    df_sorted = df.sort_values("cumulative_return", ascending=False).reset_index(drop=True)

    # Save to CSV
    df_sorted.to_csv(output_file, index=False)

    # Create formatted display version
    display_df = df_sorted.copy()
    display_df["starting_price"] = display_df["starting_price"].apply(lambda x: f"${x:.4f}")
    display_df["final_price"] = display_df["final_price"].apply(lambda x: f"${x:.4f}")
    display_df["cumulative_return"] = display_df["cumulative_return"].apply(lambda x: f"{x:+.2f}%")

    print("\n" + "="*100)
    print("BINANCE LAUNCHPOOL TOKEN PERFORMANCE (Launch → November 9, 2025)")
    print("="*100)
    print(display_df.to_string(index=False))
    print("="*100)

    return df_sorted

def create_summary_stats(df):
    """
    Print summary statistics
    """
    if df.empty:
        return

    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"Tokens analyzed: {len(df)}")
    print(f"Average return: {df['cumulative_return'].mean():+.2f}%")
    print(f"Median return: {df['cumulative_return'].median():+.2f}%")
    print(f"Best performer: {df.iloc[0]['ticker']} ({df.iloc[0]['cumulative_return']:+.2f}%)")
    print(f"Worst performer: {df.iloc[-1]['ticker']} ({df.iloc[-1]['cumulative_return']:+.2f}%)")
    print(f"Positive returns: {len(df[df['cumulative_return'] > 0])}/{len(df)} ({len(df[df['cumulative_return'] > 0])/len(df)*100:.1f}%)")
    print("="*100)

def main():
    """
    Main execution
    """
    print("Starting Binance Launchpool Token Performance Analysis...")
    print(f"Analyzing {len(LAUNCHPOOL_TOKENS)} tokens from launch to November 9, 2025")
    print("This will take several minutes due to API rate limiting...\n")

    # Analyze tokens
    results_df = analyze_tokens(LAUNCHPOOL_TOKENS)

    if not results_df.empty:
        # Create and display results table
        sorted_df = create_results_table(results_df)

        # Display summary statistics
        create_summary_stats(sorted_df)

        print(f"\nResults saved to: binance_launchpool_results.csv")
    else:
        print("\nNo token data could be retrieved. Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
