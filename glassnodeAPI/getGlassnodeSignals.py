import os
import requests
import pandas as pd
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import certifi

# SSL Certificate setup (as in getFredData.py)
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load environment variables
load_dotenv()

GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY")
BASE_URL = "https://api.glassnode.com/v1/metrics"

# Folder structure matching project convention
OUTPUT_FOLDER = "glassnodeAPI/onchainData"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Dictionary of endpoints mapping to a simpler slug
# If you prefer, you can import from apiEndpoints.py instead
GLASSNODE_ENDPOINTS = {
    "indicators/ssr_oscillator": "SSR",
    "market/price_usd_close": "BTC_PRICE",
    "supply/profit_relative": "SUPPLY_IN_PROFIT",
    "market/mvrv_z_score": "MVRV_Z_SCORE",
    "market/price_realized_usd": "BTC_REALIZED_PRICE",
    "market/spot_cvd_sum": "CVD",
    "mining/hash_rate_mean": "BTC_HASH_RATE",
    "indicators/net_unrealized_profit_loss_account_based": "ENTITY_ADJ_NUPL",
    "indicators/puell_multiple": "PUELL_MULTIPLE",
    "indicators/dormancy_flow": "ENTITY_ADJ_DORMANCY_FLOW",
    "metrics/indicators/sopr_less_155": "STH_SOPR",
    "metrics/derivatives/futures_funding_rate_perpetual": "FUTURES_FUNDING_RATE"
}

# volatility smile, 

def datetime_to_unix(dt):
    """Convert datetime object to Unix timestamp."""
    return int(dt.timestamp())

def fetch_glassnode_metric(endpoint, start_date, end_date, frequency="24h"):
    """Fetch a single Glassnode metric."""
    params = {
        'a': 'BTC',
        's': datetime_to_unix(start_date),
        'u': datetime_to_unix(end_date),
        'i': frequency,
        'f': 'JSON',
        'api_key': GLASSNODE_API_KEY
    }

    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(3):
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    logging.warning(f"No data returned for endpoint {endpoint}")
                    return None

                df = pd.DataFrame(data)
                df['t'] = pd.to_datetime(df['t'], unit='s')
                df.rename(columns={'v': endpoint.split('/')[-1], 't': 'date'}, inplace=True)
                df.set_index('date', inplace=True)
                return df
            else:
                logging.error(f"Failed to fetch {endpoint}: {response.status_code}")
                
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed for {endpoint}: {e}")
            if attempt == 2:  # Last attempt
                return None
            
def update_metric(endpoint, metric_name, frequency="24h"):
    """Update a single metric, handling both new files and updates."""
    file_path = os.path.join(OUTPUT_FOLDER, f"{metric_name}.csv")
    
    if os.path.exists(file_path):
        # Update existing file
        existing_data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
        last_date = existing_data.index.max()
        
        # Fetch only new data (with 1-day overlap for safety)
        start_date = last_date - timedelta(days=1)
        end_date = datetime.now()
        
        logging.info(f"Updating {metric_name} from {start_date}")
        
        new_data = fetch_glassnode_metric(endpoint, start_date, end_date, frequency)
        
        if new_data is not None and not new_data.empty:
            # Remove overlap and concatenate
            new_data = new_data[new_data.index > last_date]
            if not new_data.empty:
                updated_data = pd.concat([existing_data, new_data])
                updated_data.to_csv(file_path)
                logging.info(f"Updated {metric_name} with {len(new_data)} new records")
            else:
                logging.info(f"No new data for {metric_name}")
    else:
        # New file - fetch all historical data
        start_date = datetime(2010, 1, 1)  # Or your preferred start date
        end_date = datetime.now()
        
        logging.info(f"Fetching complete history for {metric_name}")
        
        data = fetch_glassnode_metric(endpoint, start_date, end_date, frequency)
        if data is not None and not data.empty:
            data.to_csv(file_path)
            logging.info(f"Created new file for {metric_name} with {len(data)} records")

def main():
    """Main function to update all Glassnode metrics."""
    logging.info("Starting Glassnode data update")
    
    for endpoint, metric_name in GLASSNODE_ENDPOINTS.items():
        try:
            update_metric(endpoint, metric_name)
        except Exception as e:
            logging.error(f"Failed to update {metric_name}: {e}")
            continue
    
    logging.info("Completed Glassnode data update")

if __name__ == "__main__":
    main()
