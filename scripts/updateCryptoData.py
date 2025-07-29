import subprocess
import logging
from datetime import datetime, timedelta
import os
import concurrent.futures
import time
import sys
import pandas as pd
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
from .assetsRoster import ROSTER
from .utils import parse_gecko_prices
from .assetCategoryManager import AssetCategoryManager
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_data_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv('GECKO_API_KEY')
if api_key:
    cg = CoinGeckoAPI(api_key=api_key)
else:
    cg = CoinGeckoAPI()

# Output folders
asset_data_folder = "./data/micro/assetData"
candle_data_folder = "./data/micro/candleData"

# Create folders if they don't exist
for folder in [asset_data_folder, candle_data_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize category manager
category_manager = AssetCategoryManager()

class RateLimiter:
    def __init__(self, calls_per_second=2):  # More conservative rate
        self.calls_per_second = calls_per_second
        self.last_call = 0
        
    def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_call
        min_interval = 1.0 / self.calls_per_second
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_call = time.time()

rate_limiter = RateLimiter(calls_per_second=2)  # Reduced from 5 to 2 calls/second

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

# Asset Data Functions
def process_single_asset(coin):
    """Process a single asset and return results."""
    try:
        csv_file_path = os.path.join(asset_data_folder, f"{coin}.csv")
        days_needed = 100000  # Default to max if no file exists

        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path)
            
            if not df.empty:
                latest_date_str = df['date'].max()
                latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
                yesterday = datetime.now() - timedelta(days=1)

                if latest_date < yesterday:
                    days_needed = (yesterday - latest_date).days
                else:
                    return coin, True, "Data is up to date"

        # Fetch data from the API with retry logic
        def fetch_asset_data():
            rate_limiter.wait()
            return cg.get_coin_market_chart_by_id(
                id=coin, 
                vs_currency='usd',
                days=days_needed,
                interval='daily'
            )
        
        data = retry_with_backoff(fetch_asset_data)

        # Parse and append new data
        new_data_df = parse_gecko_prices(data)
        
        if os.path.exists(csv_file_path):
            # Append new data to the existing CSV
            new_data_df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            # Save new data to a new CSV file
            new_data_df.to_csv(csv_file_path, index=False)

        return coin, True, f"Updated with {len(new_data_df)} records"

    except Exception as e:
        logger.error(f"Error fetching data for {coin}: {e}")
        return coin, False, str(e)

def update_assets_parallel(max_workers=6):
    """Update asset data for all coins in parallel."""
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_coin = {executor.submit(process_single_asset, coin): coin for coin in ROSTER}
        
        completed = 0
        total = len(ROSTER)
        
        for future in concurrent.futures.as_completed(future_to_coin):
            coin = future_to_coin[future]
            try:
                coin, success, message = future.result()
                results.append((coin, success, message))
                
                completed += 1
                if success:
                    logger.info(f"[{completed}/{total}] ✅ {coin}: {message}")
                else:
                    logger.error(f"[{completed}/{total}] ❌ {coin}: {message}")
                        
            except Exception as e:
                logger.error(f"Unexpected error processing {coin}: {e}")
                results.append((coin, False, str(e)))
    
    # Summary
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    logger.info(f"Asset data update completed: {len(successful)}/{len(results)} successful")
    
    if failed:
        logger.error("Failed assets:")
        for coin, _, error in failed:
            logger.error(f"  - {coin}: {error}")
    
    return len(failed) == 0

# Candle Data Functions
def fetch_data_in_chunks(coin_id, start_date, end_date):
    """Fetch data in chunks of 180 days to comply with API limits."""
    all_chunks = []
    current_start = start_date
    
    while current_start <= end_date:
        chunk_end = min(current_start + pd.Timedelta(days=179), end_date)
        
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
                df_chunk = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df_chunk['date'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
                
                # Don't shift dates - use the actual date from the API
                # df_chunk['date'] = df_chunk['date'] - pd.Timedelta(days=1)
                
                df_chunk = df_chunk[['date', 'open', 'high', 'low', 'close']]
                all_chunks.append(df_chunk)
            
            current_start = chunk_end + pd.Timedelta(days=1)
            
        except Exception as e:
            logger.error(f"Error fetching chunk for {coin_id}: {e}")
            return None
            
    if all_chunks:
        final_df = pd.concat(all_chunks, ignore_index=True)
        return final_df
    return None

def process_single_candle_asset(coin_id):
    """Process a single asset and return results."""
    try:
        # Check asset data
        asset_file = os.path.join(asset_data_folder, f"{coin_id}.csv")
        if not os.path.exists(asset_file):
            return coin_id, False, "No asset data found"
            
        asset_df = pd.read_csv(asset_file)
        asset_df['date'] = pd.to_datetime(asset_df['date'])
        asset_latest = asset_df['date'].max()
        
        # Check candle data
        candle_file = os.path.join(candle_data_folder, f"{coin_id}_candles.csv")
        
        if not os.path.exists(candle_file):
            new_data = fetch_data_in_chunks(coin_id, asset_df['date'].min(), asset_df['date'].max())
            if new_data is not None:
                new_data.to_csv(candle_file, index=False)
                return coin_id, True, f"Created with {len(new_data)} records"
            return coin_id, False, "Failed to fetch new data"
        
        # Check for missing data
        candle_df = pd.read_csv(candle_file)
        candle_df['date'] = pd.to_datetime(candle_df['date'])
        candle_latest = candle_df['date'].max()
        
        updates_made = False
        
        # Check for latest dates first
        if asset_latest > candle_latest:
            days_behind = (asset_latest - candle_latest).days
            
            new_data = fetch_data_in_chunks(
                coin_id, 
                candle_latest + pd.Timedelta(days=1), 
                asset_latest
            )
            
            if new_data is not None:
                updated_df = pd.concat([candle_df, new_data], ignore_index=True)
                updated_df.drop_duplicates(subset=['date'], inplace=True)
                updated_df.sort_values('date', inplace=True)
                updated_df.to_csv(candle_file, index=False)
                updates_made = True
        
        # Check for other missing dates
        asset_dates = set(asset_df['date'].dt.normalize())
        candle_dates = set(candle_df['date'].dt.normalize())
        missing_dates = sorted(list(asset_dates - candle_dates))
        
        if missing_dates:
            new_data = fetch_data_in_chunks(coin_id, min(missing_dates), max(missing_dates))
            
            if new_data is not None:
                if updates_made:
                    candle_df = pd.read_csv(candle_file)
                    candle_df['date'] = pd.to_datetime(candle_df['date'])
                
                updated_df = pd.concat([candle_df, new_data], ignore_index=True)
                updated_df.drop_duplicates(subset=['date'], inplace=True)
                updated_df.sort_values('date', inplace=True)
                updated_df.to_csv(candle_file, index=False)
                return coin_id, True, f"Updated with {len(new_data)} records"
        
        if updates_made:
            return coin_id, True, "Updated with latest data"
        else:
            return coin_id, True, "Data is complete"
            
    except Exception as e:
        logger.error(f"Error processing {coin_id}: {e}")
        return coin_id, False, str(e)

def update_candles_parallel(max_workers=4):
    """Fetch candle data for all assets in parallel."""
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_coin = {executor.submit(process_single_candle_asset, coin_id): coin_id for coin_id in ROSTER}
        
        with tqdm(total=len(ROSTER), desc="Processing candle data") as pbar:
            for future in concurrent.futures.as_completed(future_to_coin):
                coin_id = future_to_coin[future]
                try:
                    coin_id, success, message = future.result()
                    results.append((coin_id, success, message))
                    
                    if success:
                        pbar.set_postfix({"Status": "✅", "Asset": coin_id})
                    else:
                        pbar.set_postfix({"Status": "❌", "Asset": coin_id})
                        
                except Exception as e:
                    logger.error(f"Unexpected error processing {coin_id}: {e}")
                    results.append((coin_id, False, str(e)))
                    pbar.set_postfix({"Status": "❌", "Asset": coin_id})
                
                pbar.update(1)
    
    # Summary
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    logger.info(f"Candle data update completed: {len(successful)}/{len(results)} successful")
    
    if failed:
        logger.error("Failed candle assets:")
        for coin_id, _, error in failed:
            logger.error(f"  - {coin_id}: {error}")
    
    return len(failed) == 0

# Asset Category Functions
def update_asset_categories():
    """Update asset categories for any missing assets."""
    try:
        start_time = time.time()
        logger.info("Starting Asset Categories Update")
        
        # Check for assets that need categories
        missing_assets = category_manager._get_existing_assets()
        roster_set = set(ROSTER)
        missing_assets = roster_set - missing_assets
        
        if not missing_assets:
            logger.info("All assets already have categories")
            execution_time = time.time() - start_time
            return {
                'script': 'Asset Categories Update',
                'success': True,
                'execution_time': execution_time,
                'updated_count': 0,
                'error': None
            }
        
        logger.info(f"Found {len(missing_assets)} assets needing categories")
        updated_count = category_manager.update_missing_categories(list(missing_assets))
        
        execution_time = time.time() - start_time
        logger.info(f"Successfully updated categories for {updated_count} assets in {execution_time:.2f}s")
        
        return {
            'script': 'Asset Categories Update',
            'success': True,
            'execution_time': execution_time,
            'updated_count': updated_count,
            'error': None
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error in Asset Categories Update: {str(e)}")
        return {
            'script': 'Asset Categories Update',
            'success': False,
            'execution_time': execution_time,
            'updated_count': 0,
            'error': str(e)
        }

# Market Data Functions
def run_market_data_script():
    """Run the market data script."""
    try:
        start_time = time.time()
        logger.info("Starting Market Data Update")
        
        result = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "getMarketData.py")],
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Successfully completed Market Data Update in {execution_time:.2f}s")
        
        return {
            'script': 'Market Data Update',
            'success': True,
            'execution_time': execution_time,
            'error': None
        }
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        logger.error(f"Timeout running Market Data Update after {execution_time:.2f}s")
        return {
            'script': 'Market Data Update',
            'success': False,
            'execution_time': execution_time,
            'error': 'Timeout'
        }
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        logger.error(f"Error running Market Data Update: {str(e)}")
        return {
            'script': 'Market Data Update',
            'success': False,
            'execution_time': execution_time,
            'error': str(e)
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Unexpected error running Market Data Update: {str(e)}")
        return {
            'script': 'Market Data Update',
            'success': False,
            'execution_time': execution_time,
            'error': str(e)
        }

# Main Functions
def update_crypto_data_parallel():
    """Run all crypto data update tasks in parallel."""
    
    logger.info("Starting parallel crypto data update sequence")
    
    # Run all four tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_task = {
            executor.submit(update_assets_parallel, 6): "Asset Data Update",
            executor.submit(update_candles_parallel, 4): "Candle Data Update", 
            executor.submit(run_market_data_script): "Market Data Update",
            executor.submit(update_asset_categories): "Asset Categories Update"
        }
        
        results = []
        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
                
                if task_name in ["Market Data Update", "Asset Categories Update"]:
                    # Market data and categories return a dict
                    results.append(result)
                    if result['success']:
                        if task_name == "Asset Categories Update" and 'updated_count' in result:
                            logger.info(f"✅ {task_name} completed successfully (updated {result['updated_count']} assets)")
                        else:
                            logger.info(f"✅ {task_name} completed successfully")
                    else:
                        logger.error(f"❌ {task_name} failed: {result['error']}")
                else:
                    # Asset and candle data return boolean
                    results.append({
                        'script': task_name,
                        'success': result,
                        'execution_time': 0,
                        'error': None if result else 'Unknown error'
                    })
                    
                    if result:
                        logger.info(f"✅ {task_name} completed successfully")
                    else:
                        logger.error(f"❌ {task_name} failed")
                        
            except Exception as e:
                logger.error(f"❌ Unexpected error in {task_name}: {str(e)}")
                results.append({
                    'script': task_name,
                    'success': False,
                    'execution_time': 0,
                    'error': str(e)
                })
    
    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    total_time = sum(r['execution_time'] for r in results)
    
    logger.info(f"Summary: {len(successful)}/{len(results)} tasks completed successfully")
    logger.info(f"Total execution time: {total_time:.2f}s")
    
    if failed:
        logger.error("Failed tasks:")
        for result in failed:
            logger.error(f"  - {result['script']}: {result['error']}")
    
    return len(failed) == 0

def update_crypto_data_sequential():
    """Run all crypto data update tasks sequentially (fallback)."""
    
    logger.info("Starting sequential crypto data update sequence")
    
    tasks = [
        ("Asset Data", update_assets_parallel),
        ("Candle Data", update_candles_parallel),
        ("Market Data", run_market_data_script),
        ("Asset Categories", update_asset_categories)
    ]
    
    failed_tasks = []
    
    for task_name, task_func in tasks:
        try:
            if task_name in ["Market Data", "Asset Categories"]:
                result = task_func()
                success = result['success']
                if task_name == "Asset Categories" and success and 'updated_count' in result:
                    logger.info(f"Categories updated for {result['updated_count']} assets")
            else:
                success = task_func()
                
            if not success:
                failed_tasks.append(task_name)
                logger.warning(f"Task {task_name} failed but continuing with others")
                
        except Exception as e:
            logger.error(f"Error in {task_name}: {e}")
            failed_tasks.append(task_name)
    
    if failed_tasks:
        logger.error(f"Failed tasks: {failed_tasks}")
        return False
    
    logger.info("Successfully completed all crypto data updates")
    return True

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting crypto data update process at {start_time}")
    
    try:
        # Try parallel execution first
        logger.info("Attempting parallel execution...")
        success = update_crypto_data_parallel()
        
        # If parallel fails, fallback to sequential
        if not success:
            logger.warning("Parallel execution had failures, falling back to sequential...")
            success = update_crypto_data_sequential()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            logger.info(f"Crypto data update process completed successfully in {duration}")
        else:
            logger.error(f"Crypto data update process failed after {duration}")
            
    except Exception as e:
        logger.error(f"Unexpected error in crypto data update process: {str(e)}")
        end_time = datetime.now()
        duration = end_time - start_time
        logger.error(f"Process terminated after {duration}")