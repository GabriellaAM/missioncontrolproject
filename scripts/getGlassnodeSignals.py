import os
import requests
import pandas as pd
import logging
from dotenv import load_dotenv
from datetime import datetime, timedelta
import certifi
import ta

# SSL Certificate setup
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load environment variables
load_dotenv()

GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY")
BASE_URL = "https://api.glassnode.com/v1/metrics"

# Folder structure
OUTPUT_FOLDER = "./data/onchainData"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Dictionary of endpoints with asset availability
GLASSNODE_ENDPOINTS = {
    "indicators/ssr_oscillator": {"slug": "SSR", "assets": ["BTC"]},  # BTC only
    "supply/profit_relative": {"slug": "PCT_SUPPLY_IN_PROFIT", "assets": ["BTC", "ETH", "SOL"]},
    "market/mvrv": {"slug": "MVRV", "assets": ["BTC", "ETH", "SOL"]},
    "market/mvrv_more_155": {"slug": "MVRV_LTH", "assets": ["BTC"]},
    "market/mvrv_less_155": {"slug": "MVRV_STH", "assets": ["BTC"]},
    "market/price_realized_usd": {"slug": "REALIZED_PRICE", "assets": ["BTC", "ETH"]},
    "mining/hash_rate_mean": {"slug": "BTC_HASH_RATE", "assets": ["BTC"]},
    "indicators/net_unrealized_profit_loss_account_based": {"slug": "ENTITY_ADJ_NUPL", "assets": ["BTC"]},
    "indicators/puell_multiple": {"slug": "PUELL_MULTIPLE", "assets": ["BTC"]},
    "indicators/dormancy_flow": {"slug": "ENTITY_ADJ_DORMANCY_FLOW", "assets": ["BTC"]},
    "indicators/sopr_less_155": {"slug": "STH_SOPR", "assets": ["BTC"]},
    "supply/active_3m_6m": {"slug": "SUPPLY_ACTIVE_3M_6M", "assets": ["BTC", "ETH"]},
    "supply/active_1y_2y": {"slug": "SUPPLY_ACTIVE_1Y_2Y", "assets": ["BTC", "ETH"]},
    "transactions/transfers_volume_exchanges_net_pit": {"slug": "EXCHANGES_NET_PIT", "assets": ["BTC", "ETH"]},
    "indicators/cdd_account_based": {"slug": "CDD_ACCOUNT_BASED", "assets": ["BTC"]},
    "indicators/net_realized_profit_loss": {"slug": "NET_REALIZED_PROFIT_LOSS", "assets": ["BTC"]},
    "blockchain/utxo_profit_count": {"slug": "UTXO_PROFIT_COUNT", "assets": ["BTC"]},
    "blockchain/utxo_loss_count": {"slug": "UTXO_LOSS_COUNT", "assets": ["BTC"]},
    "indicators/realized_profits_to_value_ratio": {"slug": "REALIZED_PROFITS_TO_VALUE_RATIO", "assets": ["BTC"]},
    "indicators/realized_supply_density_more_155": {"slug": "LTH_REALIZED_SUPPLY_DENSITY", "assets": ["BTC"]},
    "indicators/realized_supply_density_less_155": {"slug": "STH_REALIZED_SUPPLY_DENSITY", "assets": ["BTC"]},
    "indicators/unrealized_profit": {"slug": "RELATIVE_UNREALIZED_PROFIT", "assets": ["BTC"]},
    "indicators/unrealized_loss": {"slug": "RELATIVE_UNREALIZED_LOSS", "assets": ["BTC"]},
    "indicators/reserve_risk": {"slug": "RESERVE_RISK", "assets": ["BTC"]},
    "market/price_drawdown_relative": {"slug": "PRICE_DRAWDOWN_RELATIVE", "assets": ["BTC", "ETH"]}
}

def process_ssr_signal(df):
    """Transform SSR signal according to specified procedure."""
    try:
        # Get the actual column name for the SSR value
        value_column = df.columns[0]  # Assuming it's the first column after the index
        
        # Replace empty strings with NaN and drop
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        df = df.dropna(subset=[value_column])
        
        # Ensure we have enough historical data for accurate calculations
        min_required_points = 50  # Need enough data for 14-day RSI and 30-day EMA
        if len(df) < min_required_points:
            logging.warning(f"Insufficient data points for SSR calculations: {len(df)}")
            return None
            
        # Calculate RSI-14 with enough warmup period
        rsi = ta.momentum.RSIIndicator(close=df[value_column], window=14).rsi()
        
        # Apply EMA-30 to RSI with enough warmup period
        ema = ta.trend.EMAIndicator(close=rsi, window=30).ema_indicator()
        
        # Calculate 7-day median
        median = ema.rolling(window=7, min_periods=1).median()
        
        # Calculate spread between EMA and median
        spread = ema - median
        
        # Create new dataframe with original date and transformed signal
        result_df = pd.DataFrame({
            'date': df.index,
            'ssr_oscillator': spread
        }).set_index('date')
        
        # Drop any remaining NaN values
        result_df = result_df.dropna()
        
        # Ensure we're not returning data with insufficient history
        if len(result_df) < min_required_points:
            logging.warning(f"Insufficient valid data points after processing: {len(result_df)}")
            return None
            
        return result_df
        
    except Exception as e:
        logging.error(f"Error processing SSR signal: {str(e)}")
        return None

def datetime_to_unix(dt):
    """Convert datetime object to Unix timestamp."""
    return int(dt.timestamp())

def fetch_glassnode_metric(endpoint, asset, start_date, end_date, frequency="24h"):
    """Fetch a single Glassnode metric for a specific asset."""
    params = {
        'a': asset,
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
                    logging.warning(f"No data returned for endpoint {endpoint} and asset {asset}")
                    return None

                df = pd.DataFrame(data)
                # Rename timestamp column to date
                df['date'] = pd.to_datetime(df['t'], unit='s')
                df = df.drop('t', axis=1)
                
                # Rename value column to 'value' for consistency
                if 'v' in df.columns:
                    df = df.rename(columns={'v': 'value'})
                
                # Set date as index
                df.set_index('date', inplace=True)
                
                return df
            else:
                logging.error(f"Failed to fetch {endpoint} for {asset}: {response.status_code}")
                
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed for {endpoint} and asset {asset}: {e}")
            if attempt == 2:  # Last attempt
                return None
    
    return None  # Return None if all attempts fail

def update_metric(endpoint, asset, metric_info, frequency="24h"):
    """Update a single Glassnode metric for a specific asset."""
    metric_name = metric_info["slug"]
    
    # Create a function to save the data to a CSV file
    def save_metric_data(df, column_name=None):
        if column_name:
            file_name = f"{asset}_{column_name}.csv"
        else:
            file_name = f"{asset}_{metric_name}.csv"
            
        # If the file exists, append new data
        if os.path.exists(os.path.join(OUTPUT_FOLDER, file_name)):
            existing_data = pd.read_csv(os.path.join(OUTPUT_FOLDER, file_name), 
                                      parse_dates=['date'], 
                                      index_col='date')
            
            # Combine existing and new data, drop duplicates
            combined_data = pd.concat([existing_data, df])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            combined_data = combined_data.sort_index()
            
            # Save the updated data
            combined_data.to_csv(os.path.join(OUTPUT_FOLDER, file_name))
            logging.info(f"Updated {file_name} with {len(df)} new records")
        else:
            # Save as new file
            df.to_csv(os.path.join(OUTPUT_FOLDER, file_name))
            logging.info(f"Created new file {file_name} with {len(df)} records")
    
    # Check if file exists to determine start date
    if os.path.exists(os.path.join(OUTPUT_FOLDER, f"{asset}_{metric_name}.csv")):
        # Update existing file
        existing_data = pd.read_csv(os.path.join(OUTPUT_FOLDER, f"{asset}_{metric_name}.csv"), 
                                   parse_dates=['date'], 
                                   index_col='date')
        last_date = existing_data.index.max()
        
        # For SSR, fetch 120 days of historical data to ensure accurate calculations
        if endpoint == "indicators/ssr_oscillator":
            start_date = last_date - timedelta(days=120)  # Fetch 120 days of history
        else:
            start_date = last_date - timedelta(days=1)
            
        end_date = datetime.now()
        logging.info(f"Updating {metric_name} for {asset} from {start_date}")
    else:
        # New file - fetch all historical data
        start_date = datetime(2010, 1, 1)
        end_date = datetime.now()
        logging.info(f"Fetching complete history for {asset}_{metric_name}")
    
    data = fetch_glassnode_metric(endpoint, asset, start_date, end_date, frequency)
    
    if data is not None and not data.empty:
        # Process SSR signal if applicable
        if endpoint == "indicators/ssr_oscillator":
            processed_data = process_ssr_signal(data)
            if processed_data is not None:
                # Only keep the new data points after the last existing date
                if os.path.exists(os.path.join(OUTPUT_FOLDER, f"{asset}_{metric_name}.csv")):
                    processed_data = processed_data[processed_data.index > last_date]
                    if not processed_data.empty:
                        save_metric_data(processed_data)
                else:
                    save_metric_data(processed_data)
            return
        
        # Check if any column contains dictionary-like data
        dict_columns = []
        for col in data.columns:
            # Find first non-null value to check if it's a dictionary
            non_null_values = data[col].dropna()
            if len(non_null_values) > 0:
                first_valid_value = non_null_values.iloc[0]
                if isinstance(first_valid_value, dict):
                    dict_columns.append(col)
        
        if dict_columns:
            for col in dict_columns:
                # Get all unique keys from the dictionaries
                all_keys = set()
                for _, value in data[col].dropna().items():
                    if isinstance(value, dict):
                        all_keys.update(value.keys())
                
                # Create separate dataframes for each key
                for key in all_keys:
                    # Extract values for this key
                    key_data = pd.DataFrame({
                        'date': data.index,
                        f"{metric_name.lower()}_{key.replace('%', 'pct').replace('.', '_')}": 
                            data[col].apply(lambda x: x.get(key) if isinstance(x, dict) else None)
                    }).set_index('date')
                    
                    # Clean key for filename
                    clean_key = key.replace('%', 'pct').replace('.', '_')
                    column_name = f"{metric_name.lower()}_{clean_key}"
                    
                    # Save as separate file with descriptive column name
                    save_metric_data(key_data, column_name)
        else:
            # Regular single-value metric
            # Rename the value column to match the metric name for consistency
            if 'value' in data.columns:
                data = data.rename(columns={'value': metric_name.lower()})
            save_metric_data(data)
    else:
        logging.warning(f"No data available for {metric_name} and asset {asset}")

def main():
    """Main function to update all Glassnode metrics."""
    logging.info("Starting Glassnode data update")
    
    for endpoint, info in GLASSNODE_ENDPOINTS.items():
        for asset in info["assets"]:
            try:
                update_metric(endpoint, asset, info)
            except Exception as e:
                logging.error(f"Failed to update {info['slug']} for {asset}: {e}")
                continue
    
    logging.info("Completed Glassnode data update")

if __name__ == "__main__":
    main()
