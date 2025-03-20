import os
import time
import pandas as pd
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
from assetsRoster import ROSTER
from tqdm import tqdm

load_dotenv()
cg = CoinGeckoAPI(api_key=os.getenv('GECKO_API_KEY'))
output_folder = "./data/micro/candleData"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def fetch_data_in_chunks(coin_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Fetch data in chunks of 180 days to comply with API limits."""
    all_chunks = []
    current_start = start_date
    
    while current_start <= end_date:
        # Calculate chunk end date (max 180 days from start)
        chunk_end = min(current_start + pd.Timedelta(days=179), end_date)
        
        try:
            print(f"🔄 Fetching {coin_id} chunk from {current_start.date()} to {chunk_end.date()}")
            data = cg.get_coin_ohlc_by_id_range(
                id=coin_id,
                vs_currency='usd',
                from_timestamp=int(current_start.timestamp()),
                to_timestamp=int(chunk_end.timestamp()),
                interval='daily'
            )
            
            if data:
                df_chunk = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df_chunk['date'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
                
                # Shift dates back by one day to match asset data convention
                df_chunk['date'] = df_chunk['date'] - pd.Timedelta(days=1)
                
                df_chunk = df_chunk[['date', 'open', 'high', 'low', 'close']]
                all_chunks.append(df_chunk)
                print(f"✅ Fetched {len(df_chunk)} records")
            
            time.sleep(0.2)  # Rate limiting
            current_start = chunk_end + pd.Timedelta(days=1)
            
        except Exception as e:
            print(f"❌ Error fetching chunk: {e}")
            return None
            
    if all_chunks:
        final_df = pd.concat(all_chunks)
        print(f"✅ Completed fetching {len(final_df)} total records for {coin_id}")
        return final_df
    return None

def fetch_daily():
    """Update candle data for all assets with priority on latest date."""
    pbar = tqdm(ROSTER, desc="Processing assets")
    for coin_id in pbar:
        try:
            pbar.set_description(f"Processing {coin_id}")
            
            # Check asset data
            asset_file = os.path.join('data/micro/assetData', f"{coin_id}.csv")
            if not os.path.exists(asset_file):
                print(f"\n❌ No asset data found for {coin_id}")
                continue
                
            asset_df = pd.read_csv(asset_file)
            asset_df['date'] = pd.to_datetime(asset_df['date'])
            asset_latest = asset_df['date'].max()
            
            # Check candle data
            candle_file = os.path.join(output_folder, f"{coin_id}_candles.csv")
            
            if not os.path.exists(candle_file):
                print(f"📈 Creating new candle data for {coin_id}")
                new_data = fetch_data_in_chunks(coin_id, min(asset_df['date']), max(asset_df['date']))
                if new_data is not None:
                    new_data.to_csv(candle_file, index=False)
                    print(f"✅ Created {coin_id} candle data with {len(new_data)} records")
                continue
            
            # Check for latest date first
            candle_df = pd.read_csv(candle_file)
            candle_df['date'] = pd.to_datetime(candle_df['date'])
            candle_latest = candle_df['date'].max()
            
            if asset_latest > candle_latest:
                days_behind = (asset_latest - candle_latest).days
                print(f"🔍 {coin_id} candle data is {days_behind} days behind (latest: {candle_latest.date()})")
                
                # Fetch just the missing latest dates
                new_data = fetch_data_in_chunks(
                    coin_id, 
                    candle_latest + pd.Timedelta(days=1), 
                    asset_latest
                )
                
                if new_data is not None:
                    # Merge new data with existing
                    updated_df = pd.concat([candle_df, new_data])
                    updated_df.drop_duplicates(subset=['date'], inplace=True)
                    updated_df.sort_values('date', inplace=True)
                    updated_df.to_csv(candle_file, index=False)
                    print(f"✅ Updated {coin_id} with {len(new_data)} latest records")
            
            # Then check for any other missing dates
            asset_dates = set(asset_df['date'].dt.normalize())
            candle_dates = set(candle_df['date'].dt.normalize())
            missing_dates = sorted(list(asset_dates - candle_dates))
            
            if missing_dates and missing_dates[0] != asset_latest:  # Skip if we already handled the latest date
                print(f"🔍 Found {len(missing_dates)} other missing dates in {coin_id}")
                new_data = fetch_data_in_chunks(coin_id, min(missing_dates), max(missing_dates))
                
                if new_data is not None:
                    # Merge new data with existing
                    updated_df = pd.concat([candle_df, new_data])
                    updated_df.drop_duplicates(subset=['date'], inplace=True)
                    updated_df.sort_values('date', inplace=True)
                    updated_df.to_csv(candle_file, index=False)
                    print(f"✅ Updated {coin_id} with {len(new_data)} records")
            else:
                print(f"✅ {coin_id} data is complete")
            
            time.sleep(0.2)  # API rate limit
            
        except Exception as e:
            print(f"❌ Error processing {coin_id}: {e}")
    
    # Get the latest date from all candle files
    latest_date = None
    for coin_id in ROSTER:
        candle_file = os.path.join(output_folder, f"{coin_id}_candles.csv")
        if os.path.exists(candle_file):
            df = pd.read_csv(candle_file)
            max_date = pd.to_datetime(df['date']).max()
            if latest_date is None or max_date > latest_date:
                latest_date = max_date
    
    if latest_date:
        print(f"\n✨ Full database updated to {latest_date.date()}")

def fetch_single_date(coin_id: str, date: pd.Timestamp) -> pd.DataFrame:
    """Fetch data for a specific date."""
    try:
        start_ts = int(date.timestamp())
        end_ts = int((date + pd.Timedelta(days=1)).timestamp())
        
        data = cg.get_coin_ohlc_by_id_range(
            id=coin_id,
            vs_currency='usd',
            from_timestamp=start_ts,
            to_timestamp=end_ts,
            interval='daily'
        )
        
        if data:
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Shift dates back by one day to match asset data convention
            df['date'] = df['date'] - pd.Timedelta(days=1)
            
            return df[['date', 'open', 'high', 'low', 'close']]
        return None
        
    except Exception as e:
        print(f"Error fetching single date: {e}")
        return None

def fill_gaps(coin_id: str):
    """Fill gaps in candle data."""
    asset_file = os.path.join('data/micro/assetData', f"{coin_id}.csv")
    candle_file = os.path.join('data/micro/candleData', f"{coin_id}_candles.csv")
    
    asset_df = pd.read_csv(asset_file)
    asset_df['date'] = pd.to_datetime(asset_df['date'])
    
    candle_df = pd.read_csv(candle_file)
    candle_df['date'] = pd.to_datetime(candle_df['date'])
    
    # Find missing dates
    asset_dates = set(asset_df['date'].dt.date)
    candle_dates = set(candle_df['date'].dt.date)
    missing_dates = asset_dates - candle_dates
    
    if missing_dates:
        print(f"Found {len(missing_dates)} missing dates")
        for date in sorted(missing_dates):
            print(f"Fetching {date}")
            new_data = fetch_single_date(coin_id, pd.Timestamp(date))
            if new_data is not None:
                candle_df = pd.concat([candle_df, new_data])
                time.sleep(0.2)  # Rate limiting
    
    candle_df = candle_df.sort_values('date')
    candle_df.to_csv(candle_file, index=False)

def fetch_latest_day():
    """
    Specifically fetch the most recent two days of data for all assets.
    This ensures the most recent data is always available.
    """
    today = pd.Timestamp.now().normalize()  # Today at midnight
    yesterday = today - pd.Timedelta(days=1)
    
    print(f"🔄 Fetching latest data through {today.date()}")
    
    for coin_id in ROSTER:
        try:
            # Check if we have asset data for this coin
            asset_file = os.path.join('data/micro/assetData', f"{coin_id}.csv")
            if not os.path.exists(asset_file):
                continue
                
            # Check if candle file exists
            candle_file = os.path.join(output_folder, f"{coin_id}_candles.csv")
            if not os.path.exists(candle_file):
                continue
                
            # Load candle data
            candle_df = pd.read_csv(candle_file)
            candle_df['date'] = pd.to_datetime(candle_df['date'])
            
            # Get the most recent dates we need to check
            dates_to_check = [yesterday, today]
            missing_dates = [date for date in dates_to_check 
                             if date not in set(candle_df['date'].dt.normalize())]
            
            if missing_dates:
                print(f"📊 Fetching {len(missing_dates)} latest dates for {coin_id}")
                
                # Fetch the missing dates
                for date in missing_dates:
                    new_data = fetch_single_date(coin_id, date)
                    
                    if new_data is not None and not new_data.empty:
                        # Merge new data with existing
                        candle_df = pd.concat([candle_df, new_data])
                        print(f"✅ Added data for {date.date()} to {coin_id}")
                    
                    time.sleep(0.2)  # API rate limit
                
                # Clean up and save
                candle_df.drop_duplicates(subset=['date'], inplace=True)
                candle_df.sort_values('date', inplace=True)
                candle_df.to_csv(candle_file, index=False)
            
        except Exception as e:
            print(f"❌ Error processing latest data for {coin_id}: {e}")

if __name__ == "__main__":
    fetch_daily()
    fetch_latest_day()  # Make sure we always have yesterday's data
