import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import os
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GECKO_API_KEY = os.getenv('GECKO_API_KEY')

class MarketDominanceManager:
    """
    Manager for calculating and storing market dominance metrics in parquet format.
    
    Calculates:
    - BTC dominance (BTC market cap / total crypto market cap)
    - Stablecoin dominance (USDT + USDC market cap / total crypto market cap)
    - Total market cap
    """
    
    def __init__(self, parquet_path=None):
        if parquet_path is None:
            # Get the project root directory
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            parquet_path = project_root / "data_parquet" / "market_data" / "dominance" / "data.parquet"
        
        self.parquet_path = Path(parquet_path)
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Paths to crypto data
        self.crypto_data_path = Path(__file__).parent.parent / "data_parquet" / "crypto_data"
        
    def _get_existing_data(self):
        """Load existing dominance data if available."""
        if self.parquet_path.exists():
            try:
                df = pd.read_parquet(self.parquet_path)
                logger.info(f"Loaded {len(df)} existing dominance records")
                return df
            except Exception as e:
                logger.error(f"Error loading existing dominance data: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def _get_global_marketcap_data(self, days='max'):
        """Fetch global market cap data from CoinGecko Pro API."""
        url = "https://pro-api.coingecko.com/api/v3/global/market_cap_chart"
        
        headers = {
            'x-cg-pro-api-key': GECKO_API_KEY
        }
        
        params = {
            'days': days,
            'vs_currency': 'usd'
        }
        
        try:
            logger.info(f"Fetching global market cap data for the last {days} days")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse the market cap data
            market_cap_data = data['market_cap_chart']['market_cap']
            
            # Convert to DataFrame
            df = pd.DataFrame(market_cap_data, columns=['timestamp', 'market_cap'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            logger.info(f"Successfully retrieved {len(df)} global market cap data points")
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing global market cap data: {str(e)}")
            return None
    
    def _load_asset_data(self, asset_id):
        """Load asset data from parquet file."""
        asset_path = self.crypto_data_path / asset_id / "data.parquet"
        
        if not asset_path.exists():
            logger.error(f"Asset data not found for {asset_id} at {asset_path}")
            return None
        
        try:
            df = pd.read_parquet(asset_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            return df[['timestamp', 'market_cap']]
        except Exception as e:
            logger.error(f"Error loading {asset_id} data: {e}")
            return None
    
    def calculate_dominance(self, days_back=None):
        """
        Calculate BTC and stablecoin dominance metrics.
        
        Args:
            days_back: Number of days to fetch data for. None means all available data.
        
        Returns:
            DataFrame with dominance metrics or None if failed.
        """
        # Determine how many days of data we need
        existing_data = self._get_existing_data()
        
        if not existing_data.empty and days_back is None:
            # Calculate days since last update
            latest_date = pd.to_datetime(existing_data['timestamp']).max()
            days_since = (datetime.now(timezone.utc) - latest_date).days
            
            if days_since <= 1:
                logger.info("Dominance data is up to date")
                return existing_data
            
            # Fetch only missing days plus a small overlap
            days_back = days_since + 2
            logger.info(f"Fetching {days_back} days of new data")
        
        # Fetch global market cap data
        global_mcap = self._get_global_marketcap_data(days=days_back if days_back else 'max')
        if global_mcap is None:
            return None
        
        # Load asset data
        logger.info("Loading BTC data for dominance calculation")
        btc_data = self._load_asset_data('bitcoin')
        if btc_data is None:
            return None
        
        logger.info("Loading USDT data for stablecoin dominance")
        usdt_data = self._load_asset_data('tether')
        if usdt_data is None:
            return None
        
        logger.info("Loading USDC data for stablecoin dominance")
        usdc_data = self._load_asset_data('usd-coin')
        if usdc_data is None:
            return None
        
        # Merge all data on timestamp
        logger.info("Merging data and calculating dominance metrics")
        
        # Start with global market cap
        merged = global_mcap.copy()
        merged = merged.rename(columns={'market_cap': 'total_market_cap'})
        
        # Merge BTC data
        btc_data = btc_data.rename(columns={'market_cap': 'btc_market_cap'})
        merged = pd.merge(merged, btc_data, on='timestamp', how='inner')
        
        # Merge USDT data
        usdt_data = usdt_data.rename(columns={'market_cap': 'usdt_market_cap'})
        merged = pd.merge(merged, usdt_data, on='timestamp', how='inner')
        
        # Merge USDC data
        usdc_data = usdc_data.rename(columns={'market_cap': 'usdc_market_cap'})
        merged = pd.merge(merged, usdc_data, on='timestamp', how='inner')
        
        # Calculate dominance metrics
        merged['btc_dominance'] = merged['btc_market_cap'] / merged['total_market_cap']
        merged['stablecoin_dominance'] = (merged['usdt_market_cap'] + merged['usdc_market_cap']) / merged['total_market_cap']
        
        # Select final columns
        result = merged[['timestamp', 'btc_dominance', 'stablecoin_dominance', 'total_market_cap']]
        
        logger.info(f"Calculated dominance for {len(result)} data points")
        logger.info(f"Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
        
        return result
    
    def update_dominance_data(self):
        """Update dominance data with latest values."""
        try:
            # Calculate new dominance data
            new_data = self.calculate_dominance()
            if new_data is None:
                logger.error("Failed to calculate dominance metrics")
                return False
            
            # Load existing data
            existing_data = self._get_existing_data()
            
            if not existing_data.empty:
                # Combine with existing data
                combined = pd.concat([existing_data, new_data], ignore_index=True)
                
                # Remove duplicates, keeping the latest
                combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
                combined = combined.sort_values('timestamp')
                
                logger.info(f"Combined data has {len(combined)} records")
            else:
                combined = new_data
            
            # Save to parquet
            combined.to_parquet(self.parquet_path, index=False, compression='snappy')
            logger.info(f"Successfully saved dominance data to {self.parquet_path}")
            
            # Log some statistics
            latest_data = combined.iloc[-1]
            logger.info(f"Latest BTC dominance: {latest_data['btc_dominance']:.2%}")
            logger.info(f"Latest stablecoin dominance: {latest_data['stablecoin_dominance']:.2%}")
            logger.info(f"Latest total market cap: ${latest_data['total_market_cap']:,.0f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating dominance data: {e}")
            return False
    
    def migrate_csv_data(self):
        """
        One-time migration of existing CSV dominance data to parquet format.
        """
        try:
            csv_dir = Path(__file__).parent.parent / "data" / "micro" / "marketData"
            
            # Load BTC dominance CSV
            btc_csv = csv_dir / "btc_dominance.csv"
            if btc_csv.exists():
                btc_dom = pd.read_csv(btc_csv)
                btc_dom['date'] = pd.to_datetime(btc_dom['date'])
                btc_dom = btc_dom.rename(columns={'date': 'timestamp'})
                logger.info(f"Loaded {len(btc_dom)} BTC dominance records from CSV")
            else:
                logger.warning("BTC dominance CSV not found")
                btc_dom = pd.DataFrame()
            
            # Load stablecoin dominance CSV
            stable_csv = csv_dir / "stablecoin_dominance.csv"
            if stable_csv.exists():
                stable_dom = pd.read_csv(stable_csv)
                stable_dom['date'] = pd.to_datetime(stable_dom['date'])
                stable_dom = stable_dom.rename(columns={'date': 'timestamp'})
                logger.info(f"Loaded {len(stable_dom)} stablecoin dominance records from CSV")
            else:
                logger.warning("Stablecoin dominance CSV not found")
                stable_dom = pd.DataFrame()
            
            if not btc_dom.empty and not stable_dom.empty:
                # Merge the two datasets
                migrated = pd.merge(btc_dom, stable_dom, on='timestamp', how='outer')
                
                # We don't have historical total_market_cap in CSVs, so we'll need to fetch it
                logger.info("Fetching historical market cap data for migration...")
                global_mcap = self._get_global_marketcap_data(days='max')
                
                if global_mcap is not None:
                    global_mcap = global_mcap.rename(columns={'market_cap': 'total_market_cap'})
                    migrated = pd.merge(migrated, global_mcap, on='timestamp', how='left')
                
                # Sort and save
                migrated = migrated.sort_values('timestamp')
                migrated.to_parquet(self.parquet_path, index=False, compression='snappy')
                
                logger.info(f"Successfully migrated {len(migrated)} records to parquet format")
                return True
            else:
                logger.error("No CSV data found to migrate")
                return False
                
        except Exception as e:
            logger.error(f"Error during CSV migration: {e}")
            return False


# Convenience functions
def update_market_dominance():
    """Update market dominance data."""
    logging.basicConfig(level=logging.INFO)
    manager = MarketDominanceManager()
    return manager.update_dominance_data()


def migrate_dominance_csv_to_parquet():
    """One-time migration from CSV to parquet format."""
    logging.basicConfig(level=logging.INFO)
    manager = MarketDominanceManager()
    return manager.migrate_csv_data()


if __name__ == "__main__":
    # Run update
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    success = update_market_dominance()
    if success:
        logger.info("Market dominance update completed successfully")
    else:
        logger.error("Market dominance update failed")