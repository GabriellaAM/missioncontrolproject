import pandas as pd
import requests
import os
from datetime import datetime
import time
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GECKO_API_KEY = os.getenv('GECKO_API_KEY')

def get_global_marketcap_data(days='max'):
    """Fetch global market cap data from CoinGecko Pro API"""
    url = f"https://pro-api.coingecko.com/api/v3/global/market_cap_chart"
    
    headers = {
        'x-cg-pro-api-key': GECKO_API_KEY
    }
    
    params = {
        'days': days,
        'vs_currency': 'usd'
    }
    
    try:
        logger.info(f"Fetching global market cap data for the last {days} days")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Correct JSON parsing
        market_cap_data = data['market_cap_chart']['market_cap']
        
        # Convert to DataFrame
        logger.debug("Converting response data to DataFrame")
        df = pd.DataFrame(market_cap_data, columns=['timestamp', 'market_cap'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop('timestamp', axis=1)
        df.set_index('date', inplace=True)
        
        logger.info(f"Successfully retrieved {len(df)} data points")
        return df
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return None
    except KeyError as e:
        logger.error(f"Failed to parse API response: {str(e)}")
        logger.debug(f"Received data structure: {data}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None

def calculate_btc_dominance():
    """Calculate BTC dominance using global market cap and BTC market cap"""
    logger.info("Starting BTC dominance calculation")
    
    # Get global market cap data
    global_mcap = get_global_marketcap_data()
    
    if global_mcap is None:
        logger.error("Failed to get global market cap data")
        return None
    
    try:
        # Load BTC historical data
        logger.info("Loading BTC historical data")
        btc_data = pd.read_csv('/Users/valter.rebelo/MissionControl/data/micro/assetData/bitcoin.csv')
        btc_data['date'] = pd.to_datetime(btc_data['date'])
        btc_data.set_index('date', inplace=True)
        
        # Calculate BTC dominance
        logger.info("Merging data and calculating BTC dominance")
        merged_data = pd.merge(
            global_mcap,
            btc_data[['market_cap']],
            left_index=True,
            right_index=True,
            how='inner',
            suffixes=('_total', '_btc')
        )
        
        merged_data['btc_dominance'] = (merged_data['market_cap_btc'] / merged_data['market_cap_total']) 
        
        # Create output directory if it doesn't exist
        output_dir = '/Users/valter.rebelo/MissionControl/data/micro/marketData'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        output_file = os.path.join(output_dir, 'btc_dominance.csv')
        merged_data['btc_dominance'].to_csv(output_file)
        
        logger.info(f"BTC dominance data successfully saved to {output_file}")
        logger.debug(f"Data range: {merged_data.index.min()} to {merged_data.index.max()}")
        return merged_data
        
    except Exception as e:
        logger.error(f"Error in calculate_btc_dominance: {str(e)}")
        return None
    
def calculate_stablecoin_dominance():
    """Calculate combined USDT and USDC dominance using global market cap"""
    logger.info("Starting stablecoin dominance calculation")
    
    # Get global market cap data
    global_mcap = get_global_marketcap_data()
    
    if global_mcap is None:
        logger.error("Failed to get global market cap data")
        return None
    
    try:
        # Load USDT historical data
        logger.info("Loading USDT historical data")
        usdt_data = pd.read_csv('/Users/valter.rebelo/MissionControl/data/micro/assetData/tether.csv')
        usdt_data['date'] = pd.to_datetime(usdt_data['date'])
        usdt_data.set_index('date', inplace=True)
        
        # Load USDC historical data
        logger.info("Loading USDC historical data")
        usdc_data = pd.read_csv('/Users/valter.rebelo/MissionControl/data/micro/assetData/usd-coin.csv')
        usdc_data['date'] = pd.to_datetime(usdc_data['date'])
        usdc_data.set_index('date', inplace=True)
        
        # Rename columns before merge to avoid conflicts
        global_mcap = global_mcap.rename(columns={'market_cap': 'market_cap_global'})
        usdt_data = usdt_data.rename(columns={'market_cap': 'market_cap_usdt'})
        usdc_data = usdc_data.rename(columns={'market_cap': 'market_cap_usdc'})
        
        # First merge - global market cap with USDT
        logger.info("Merging data and calculating stablecoin dominance")
        merged_data = pd.merge(
            global_mcap,
            usdt_data[['market_cap_usdt']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Second merge - add USDC data
        merged_data = pd.merge(
            merged_data,
            usdc_data[['market_cap_usdc']],
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        logger.debug(f"Columns after merges: {merged_data.columns.tolist()}")
        
        # Calculate combined stablecoin dominance
        merged_data['stablecoin_dominance'] = ((merged_data['market_cap_usdt'] + 
                                               merged_data['market_cap_usdc']) / 
                                              merged_data['market_cap_global'])
        
        # Create output directory if it doesn't exist
        output_dir = '/Users/valter.rebelo/MissionControl/data/micro/marketData'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        output_file = os.path.join(output_dir, 'stablecoin_dominance.csv')
        merged_data['stablecoin_dominance'].to_csv(output_file)
        
        logger.info(f"Stablecoin dominance data successfully saved to {output_file}")
        logger.debug(f"Data range: {merged_data.index.min()} to {merged_data.index.max()}")
        return merged_data
        
    except Exception as e:
        logger.error(f"Error in calculate_stablecoin_dominance: {str(e)}")
        logger.debug(f"Current merged_data columns: {merged_data.columns if 'merged_data' in locals() else 'Not created'}")
        return None

if __name__ == "__main__":
    try:
        # Calculate BTC dominance
        btc_result = calculate_btc_dominance()
        
        # Calculate stablecoin dominance
        stablecoin_result = calculate_stablecoin_dominance()
        
        if btc_result is None or stablecoin_result is None:
            logger.error("Failed to calculate either BTC or stablecoin dominance")
            
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")