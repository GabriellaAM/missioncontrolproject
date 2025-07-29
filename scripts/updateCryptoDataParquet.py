import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from datetime import datetime, timedelta
import os
import logging
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
import time
import concurrent.futures
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv('GECKO_API_KEY')
if api_key:
    cg = CoinGeckoAPI(api_key=api_key)
else:
    cg = CoinGeckoAPI()

# Import asset roster and utils
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from assetsRoster import ROSTER
from utils import parse_gecko_ohlcv, parse_gecko_prices
from assetCategoryManagerParquet import AssetCategoryManagerParquet
from marketDataParquet import MarketDominanceManager

# Output folder - using coingecko subfolder for future provider flexibility  
PARQUET_DATA_FOLDER = "./data_parquet/crypto_data/coingecko"

# Assets to process - use the full ROSTER
ASSETS = ROSTER  # Full roster (215 assets)

# Asset ID to ticker symbol mapping
ASSET_TICKERS = {}

# Create folder if it doesn't exist
os.makedirs(PARQUET_DATA_FOLDER, exist_ok=True)


class RateLimiter:
    def __init__(self, calls_per_second=2):
        self.calls_per_second = calls_per_second
        self.last_call = 0
        
    def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_call
        min_interval = 1.0 / self.calls_per_second
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_call = time.time()


rate_limiter = RateLimiter(calls_per_second=2)


def fetch_coin_tickers():
    """Fetch coin tickers from CoinGecko API and populate ASSET_TICKERS."""
    global ASSET_TICKERS
    
    def fetch():
        rate_limiter.wait()
        return cg.get_coins_list()
    
    try:
        coins_list = retry_with_backoff(fetch)
        if coins_list:
            # Create mapping for our assets
            for coin in coins_list:
                if coin['id'] in ASSETS:
                    ASSET_TICKERS[coin['id']] = coin['symbol'].upper()
            
            logger.info(f"Fetched tickers for {len(ASSET_TICKERS)} assets")
            logger.info(f"Ticker mapping: {ASSET_TICKERS}")
        else:
            logger.error("Failed to fetch coin tickers")
            # Fallback to known mappings
            ASSET_TICKERS = {
                'bitcoin': 'BTC',
                'ethereum': 'ETH'
            }
    except Exception as e:
        logger.error(f"Error fetching coin tickers: {e}")
        # Fallback to known mappings
        ASSET_TICKERS = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH'
        }


def retry_with_backoff(func, max_retries=5, initial_delay=1.0):
    """Retry a function with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise
            else:
                raise
    return None


def fetch_ohlc_data(coin_id, start_date, end_date):
    """Fetch OHLC data for a coin in chunks to ensure daily granularity."""
    all_chunks = []
    current_start = start_date
    
    while current_start <= end_date:
        chunk_end = min(current_start + timedelta(days=179), end_date)
        
        def fetch_chunk():
            rate_limiter.wait()
            
            data = cg.get_coin_ohlc_by_id_range(
                id=coin_id,
                vs_currency='usd',
                from_timestamp=int(current_start.timestamp()),
                to_timestamp=int(chunk_end.timestamp()),
                interval='daily'
            )
            return data
        
        try:
            data = retry_with_backoff(fetch_chunk)
            
            if data:
                df_chunk = parse_gecko_ohlcv(data)
                df_chunk.rename(columns={'date': 'timestamp'}, inplace=True)
                all_chunks.append(df_chunk)
            
            current_start = chunk_end + timedelta(days=1)
            
        except Exception as e:
            logger.error(f"Error fetching chunk for {coin_id}: {e}")
            return None
            
    if all_chunks:
        final_df = pd.concat(all_chunks, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['timestamp'])
        final_df = final_df.sort_values('timestamp')
        return final_df
    return None


def fetch_market_data(coin_id, days):
    """Fetch market data (close, market_cap, volume) for a coin."""
    def fetch():
        rate_limiter.wait()
        return cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency='usd',
            days=days,
            interval='daily'
        )
    
    data = retry_with_backoff(fetch)
    if data:
        df = parse_gecko_prices(data)
        df.rename(columns={'date': 'timestamp'}, inplace=True)
        return df
    return None


def merge_ohlc_and_market_data(ohlc_df, market_df):
    """Merge OHLC and market data on timestamp."""
    # Round timestamps to the nearest day for merging
    ohlc_df['date'] = ohlc_df['timestamp'].dt.date
    market_df['date'] = market_df['timestamp'].dt.date
    
    # Merge on date
    merged = pd.merge(
        ohlc_df,
        market_df[['date', 'market_cap', 'total_volume']],
        on='date',
        how='inner'
    )
    
    # Use OHLC timestamp and drop the date column
    merged = merged.drop('date', axis=1)
    
    # Drop close_market as we already have close from OHLC
    return merged


def calculate_btc_ratios(df, btc_df, symbol):
    """Calculate BTC-relative prices."""
    if symbol == 'bitcoin':
        # For Bitcoin, BTC ratios are just 1
        df['open_btc'] = 1.0
        df['high_btc'] = 1.0
        df['low_btc'] = 1.0
        df['close_btc'] = 1.0
        df['volume_btc'] = 1.0
    else:
        # Merge with BTC data on date
        df['date'] = df['timestamp'].dt.date
        btc_df['date'] = btc_df['timestamp'].dt.date
        
        # Select only needed BTC columns
        btc_cols = btc_df[['date', 'open', 'high', 'low', 'close', 'total_volume']].copy()
        btc_cols.columns = ['date', 'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume']
        
        # Merge
        df = pd.merge(df, btc_cols, on='date', how='left')
        
        # Calculate ratios
        df['open_btc'] = df['open'] / df['btc_open']
        df['high_btc'] = df['high'] / df['btc_high']
        df['low_btc'] = df['low'] / df['btc_low']
        df['close_btc'] = df['close'] / df['btc_close']
        df['volume_btc'] = df['total_volume'] / df['btc_volume']
        
        # Drop temporary columns
        df = df.drop(['date', 'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume'], axis=1)
    
    return df


def update_asset_data(coin_id, btc_df=None):
    """Update data for a single asset using partitioned storage."""
    # Create partition folder for this asset
    partition_folder = os.path.join(PARQUET_DATA_FOLDER, coin_id)
    parquet_file = os.path.join(partition_folder, "data.parquet")
    
    # Create partition folder if it doesn't exist
    os.makedirs(partition_folder, exist_ok=True)
    
    # Determine date range for data fetch
    if os.path.exists(parquet_file):
        # Read existing data to find the latest date
        existing_df = pd.read_parquet(parquet_file)
        if not existing_df.empty:
            latest_date = existing_df['timestamp'].max()
            days_since = (datetime.now() - latest_date).days
            
            if days_since <= 1:
                logger.info(f"{coin_id}: Data is up to date")
                return existing_df
            else:
                start_date = latest_date + timedelta(days=1)
                end_date = datetime.now()
                logger.info(f"{coin_id}: Fetching data from {start_date.date()} to {end_date.date()}")
        else:
            # Empty file, fetch all data
            start_date = datetime(2013, 1, 1)
            end_date = datetime.now()
    else:
        logger.info(f"{coin_id}: Creating new dataset")
        start_date = datetime(2013, 1, 1)
        end_date = datetime.now()
    
    # Calculate days for market data fetch
    days_needed = (end_date - start_date).days + 1
    
    # Fetch OHLC data with proper date range
    ohlc_df = fetch_ohlc_data(coin_id, start_date, end_date)
    if ohlc_df is None:
        logger.error(f"{coin_id}: Failed to fetch OHLC data")
        return None
    
    # Fetch market data
    market_df = fetch_market_data(coin_id, days_needed if days_needed < 10000 else 'max')
    if market_df is None:
        logger.error(f"{coin_id}: Failed to fetch market data")
        return None
    
    # Filter market data to match our date range
    market_df = market_df[market_df['timestamp'] >= start_date]
    
    # Merge OHLC and market data
    merged_df = merge_ohlc_and_market_data(ohlc_df, market_df)
    
    # Add asset_id and symbol columns
    merged_df['asset_id'] = coin_id
    merged_df['symbol'] = ASSET_TICKERS.get(coin_id, coin_id.upper())
    
    # Calculate BTC ratios
    if btc_df is not None or coin_id == 'bitcoin':
        merged_df = calculate_btc_ratios(merged_df, btc_df, coin_id)
    
    # If updating existing file, append new data
    if os.path.exists(parquet_file) and 'existing_df' in locals():
        # Combine and remove duplicates
        combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
        combined_df = combined_df.sort_values('timestamp')
        
        # Save updated data to partition
        combined_df.to_parquet(parquet_file, index=False, compression='snappy')
        logger.info(f"{coin_id}: Updated with {len(merged_df)} new records")
        
        return combined_df
    else:
        # Save new data to partition
        merged_df.to_parquet(parquet_file, index=False, compression='snappy')
        logger.info(f"{coin_id}: Saved {len(merged_df)} records")
        
        return merged_df


def process_single_asset_wrapper(asset, btc_df=None):
    """Wrapper function for parallel processing."""
    try:
        result_df = update_asset_data(asset, btc_df)
        if result_df is not None:
            return asset, True, f"Updated with {len(result_df)} records"
        else:
            return asset, False, "Failed to fetch data"
    except Exception as e:
        logger.error(f"Error processing {asset}: {e}")
        return asset, False, str(e)


def update_assets_parallel(btc_df, max_workers=4):
    """Update all assets in parallel (excluding Bitcoin)."""
    # Filter out Bitcoin since it's already processed
    other_assets = [asset for asset in ASSETS if asset != 'bitcoin']
    
    if not other_assets:
        return []
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_asset = {
            executor.submit(process_single_asset_wrapper, asset, btc_df): asset 
            for asset in other_assets
        }
        
        # Process results with progress bar
        with tqdm(total=len(other_assets), desc="Processing assets") as pbar:
            for future in concurrent.futures.as_completed(future_to_asset):
                asset = future_to_asset[future]
                try:
                    asset_name, success, message = future.result()
                    results.append((asset_name, success, message))
                    
                    if success:
                        pbar.set_postfix({"Status": "✅", "Asset": asset_name})
                        logger.info(f"✅ {asset_name}: {message}")
                    else:
                        pbar.set_postfix({"Status": "❌", "Asset": asset_name})
                        logger.error(f"❌ {asset_name}: {message}")
                        
                except Exception as e:
                    logger.error(f"Unexpected error processing {asset}: {e}")
                    results.append((asset, False, str(e)))
                    pbar.set_postfix({"Status": "❌", "Asset": asset})
                
                pbar.update(1)
    
    return results


def update_asset_categories():
    """Update asset categories for any new assets in the roster."""
    logger.info("Checking for new assets that need category data...")
    
    try:
        category_manager = AssetCategoryManagerParquet()
        missing_categories = category_manager._get_existing_assets()
        new_assets = [asset for asset in ASSETS if asset not in missing_categories]
        
        if new_assets:
            logger.info(f"Found {len(new_assets)} assets needing category updates: {new_assets}")
            updated_count = category_manager.update_missing_categories(new_assets)
            logger.info(f"✅ Updated categories for {updated_count} assets")
        else:
            logger.info("All assets already have category data")
            
    except Exception as e:
        logger.error(f"Error updating asset categories: {e}")
        # Don't fail the entire update if categories fail
        pass


def update_market_dominance():
    """Update market dominance metrics."""
    logger.info("Updating market dominance data...")
    
    try:
        dominance_manager = MarketDominanceManager()
        success = dominance_manager.update_dominance_data()
        
        if success:
            logger.info("✅ Market dominance data updated successfully")
        else:
            logger.error("❌ Failed to update market dominance data")
            
    except Exception as e:
        logger.error(f"Error updating market dominance: {e}")
        # Don't fail the entire update if dominance fails
        pass


def main():
    """Main function to update all assets."""
    start_time = datetime.now()
    logger.info(f"Starting crypto data update to Parquet at {start_time}")
    logger.info(f"Processing {len(ASSETS)} assets...")
    
    # First, update asset categories for any new assets
    update_asset_categories()
    
    # Then, fetch coin tickers
    fetch_coin_tickers()
    
    # First, update Bitcoin data (needed for BTC ratios)
    logger.info("Updating Bitcoin data...")
    btc_df = update_asset_data('bitcoin')
    
    if btc_df is None:
        logger.error("Failed to update Bitcoin data. Exiting.")
        return
    
    logger.info(f"✅ bitcoin: Successfully updated with {len(btc_df)} records")
    
    # Then update other assets in parallel
    if len(ASSETS) > 1:
        logger.info("Updating other assets in parallel...")
        results = update_assets_parallel(btc_df, max_workers=6)
        
        # Summary
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        logger.info(f"Parallel update completed: {len(successful)}/{len(results)} successful")
        
        if failed:
            logger.error("Failed assets:")
            for asset, _, error in failed:
                logger.error(f"  - {asset}: {error}")
    
    # Update market dominance data (after all asset data is updated)
    update_market_dominance()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"Crypto data update completed in {duration}")
    
    # Show summary
    logger.info("\nData files created/updated:")
    total_records = 0
    for asset in ASSETS:
        partition_folder = os.path.join(PARQUET_DATA_FOLDER, asset)
        file_path = os.path.join(partition_folder, "data.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            total_records += len(df)
            logger.info(f"  {asset}: {len(df)} records, latest: {df['timestamp'].max()}")
    
    logger.info(f"\nTotal records across all assets: {total_records:,}")


if __name__ == "__main__":
    main()