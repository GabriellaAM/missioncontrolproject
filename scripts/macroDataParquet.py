import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
import requests
import os
import logging
import time
import certifi
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

# Set up logging
logger = logging.getLogger(__name__)

# Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

class MacroDataManager:
    """
    Base manager class for macro economic data in parquet format.
    """
    
    def __init__(self, base_path=None, source_type=None):
        if base_path is None:
            # Get the project root directory
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            base_path = project_root / "data_parquet" / "macro_data"
        
        self.base_path = Path(base_path)
        self.source_type = source_type  # 'yahoo', 'fred', or 'calculated'
        
        # If source_type is provided, use source-specific subdirectory
        if self.source_type:
            self.base_path = self.base_path / self.source_type
        
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def _get_asset_path(self, macro_asset_id):
        """Get the parquet file path for a specific macro asset."""
        asset_path = self.base_path / macro_asset_id
        asset_path.mkdir(parents=True, exist_ok=True)
        return asset_path / "data.parquet"
    
    def _get_existing_data(self, macro_asset_id):
        """Load existing data for a macro asset if available."""
        parquet_path = self._get_asset_path(macro_asset_id)
        
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                logger.info(f"Loaded {len(df)} existing records for {macro_asset_id}")
                return df
            except Exception as e:
                logger.error(f"Error loading existing data for {macro_asset_id}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def _save_data(self, macro_asset_id, df):
        """Save data to parquet file."""
        if df.empty:
            logger.warning(f"No data to save for {macro_asset_id}")
            return False
            
        parquet_path = self._get_asset_path(macro_asset_id)
        
        try:
            # Ensure timestamp is UTC datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Save to parquet
            df.to_parquet(parquet_path, index=False, compression='snappy')
            logger.info(f"Saved {len(df)} records for {macro_asset_id} to {parquet_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data for {macro_asset_id}: {e}")
            return False


class YahooDataManager(MacroDataManager):
    """
    Manager for Yahoo Finance data with OHLC support.
    """
    
    def __init__(self, base_path=None):
        super().__init__(base_path, source_type='yahoo')
        
        # Yahoo Finance tickers mapping
        self.tickers = {
            "DX-Y.NYB": "dxy",
            "JPY=X": "usdjpy", 
            "EUR=X": "usdeur",
            "^MOVE": "move",
            "^VIX": "vix",
            "HYG": "hyg",
            "RTY=F": "rty",
            "YM=F": "ym", 
            "ES=F": "es",
            "GC=F": "gold",
            "HG=F": "copper",
            "CL=F": "oil"
        }
    
    def _fetch_yahoo_data(self, symbol, start_date=None, end_date=None, period="max"):
        """
        Fetch data from Yahoo Finance using yf.download().
        
        Args:
            symbol: Yahoo Finance symbol
            start_date: Start date for data fetch
            end_date: End date for data fetch  
            period: Period string if not using start/end dates
            
        Returns:
            DataFrame with OHLC data or None if failed
        """
        try:
            # Always exclude today's data to avoid incomplete candles
            # Set end_date to yesterday if not specified or if it includes today
            if end_date is None or end_date.date() >= datetime.now().date():
                end_date = datetime.now() - timedelta(days=1)
                end_date = end_date.replace(hour=23, minute=59, second=59)
            
            if start_date and end_date:
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            else:
                # For period="max", we still need to exclude today
                yesterday = datetime.now() - timedelta(days=1)
                data = yf.download(symbol, period=period, end=yesterday, progress=False)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Handle multi-level columns if present
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten multi-level columns - take the first level (Price)
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            # Reset index to get timestamp as column
            data = data.reset_index()
            
            # Rename Date column to timestamp
            if 'Date' in data.columns:
                data = data.rename(columns={'Date': 'timestamp'})
            
            # Clean column names and convert to lowercase
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data = data.rename(columns={old_col: new_col})
            
            # Remove timezone info if present
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
                data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
            
            # Drop any NaN rows
            data = data.dropna()
            
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo data for {symbol}: {e}")
            return None
    
    def update_asset_data(self, symbol, macro_asset_id):
        """
        Update data for a single Yahoo Finance asset.
        
        Args:
            symbol: Yahoo Finance symbol (e.g., "^VIX")
            macro_asset_id: Internal macro asset ID (e.g., "vix")
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Updating {macro_asset_id} ({symbol})...")
        
        # Get existing data
        existing_data = self._get_existing_data(macro_asset_id)
        
        if not existing_data.empty:
            # Get the latest date and fetch only new data
            latest_date = existing_data['timestamp'].max()
            start_date = latest_date + timedelta(days=1)
            end_date = datetime.now()
            
            if start_date.date() >= end_date.date():
                logger.info(f"{macro_asset_id}: Data is up to date")
                return True
            
            logger.info(f"Fetching new data for {macro_asset_id} from {start_date.date()}")
            new_data = self._fetch_yahoo_data(symbol, start_date=start_date, end_date=end_date)
        else:
            # Fetch all available data
            logger.info(f"Fetching all available data for {macro_asset_id}")
            new_data = self._fetch_yahoo_data(symbol, period="max")
        
        if new_data is None or new_data.empty:
            logger.warning(f"No new data for {macro_asset_id}")
            return False
        
        # Add metadata columns
        new_data['macro_asset_id'] = macro_asset_id
        new_data['source'] = 'YAHOO'
        
        # Combine with existing data if present
        if not existing_data.empty:
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            # Remove duplicates based on timestamp
            combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='last')
        else:
            combined_data = new_data
        
        # Save updated data
        success = self._save_data(macro_asset_id, combined_data)
        
        if success:
            logger.info(f"✅ {macro_asset_id}: Updated with {len(new_data)} new records")
        else:
            logger.error(f"❌ {macro_asset_id}: Failed to save data")
            
        return success
    
    def update_all_assets(self):
        """Update all Yahoo Finance assets."""
        results = []
        
        for symbol, macro_asset_id in self.tickers.items():
            try:
                success = self.update_asset_data(symbol, macro_asset_id)
                results.append((macro_asset_id, success))
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error updating {macro_asset_id}: {e}")
                results.append((macro_asset_id, False))
        
        # Summary
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        logger.info(f"Yahoo Finance update completed: {len(successful)}/{len(results)} successful")
        
        if failed:
            logger.error("Failed assets:")
            for macro_asset_id, _ in failed:
                logger.error(f"  - {macro_asset_id}")
        
        return results


class FredDataManager(MacroDataManager):
    """
    Manager for FRED (Federal Reserve Economic Data) API data.
    """
    
    def __init__(self, base_path=None):
        super().__init__(base_path, source_type='fred')
        
        if not FRED_API_KEY:
            raise ValueError("FRED API key not found in environment variables")
        
        self.fred = Fred(api_key=FRED_API_KEY)
        
        # FRED series mapping (from original script)
        self.series_mapping = {
            'FF': 'fedFundsRate',
            'UNRATE': 'unemploymentRate',
            'CORESTICKM159SFRBATL': 'coreStickiness',
            'DGS2': 'treasury2Y',
            'DGS1': 'treasury1Y',
            'DTB3': 'treasury3M',
            'DTB6': 'treasury6M',
            'CPIAUCSL': 'consumerPriceIndex',
            'T5YIE': 'treasury5YInflationExpectation',
            'DGORDER': 'durableGoodsOrders',
            'JTSJOL': 'jobOpenings',
            'PAYEMS': 'nonfarmPayrolls',
            'CES0500000003': 'retailEmployment',
            'DGS10': 'treasury10Y',
            'JTSQUL': 'jobQuits',
            'JTSLDL': 'jobLayoffs',
            'ADXTNO': 'durableGoods',
            'ICSA': 'initialClaims',
            'PCEPI': 'personalConsumptionExpenditures',
            'T5YIFR': 'treasury5YInflationForwardRate',
            'PCETRIM12M159SFRBDAL': 'pceTrimmedMean12M',
            'MICH': 'consumerSentiment',
            'IORB': 'interestOnReserves',
            'SOFR': 'securedOvernightFinancingRate',
            'M2SL': 'm2',
            'RESPPANWW': 'fedTotalAssets',
            'WRESBAL': 'reserveBalances',
            'RRPONTSYD': 'repoAgreements',
            'GDPC1': 'realGDP',
            'GDP': 'nominalGDP',
            'NFCI': 'financialConditionsIndex',
            'DTWEXBGS': 'dollarIndex',
            'BAMLH0A0HYM2': 'creditSpreads',
            'BAMLC0A0CM': 'creditSpreadsHighGrade',
            'SP500': 'sp500',
            'NASDAQCOM': 'nasdaq',
            'GFDEBTN': 'usTotalDebt',
            'GFDEGDQ188S': 'usTotalDebt_GDP',
            'WTREGEN': 'tga',
            'JPNASSETS': 'bojAssets',
            'ECBASSETSW': 'ecbAssets',
            'PSAVERT': 'personalSavingsRate',
            'DSPIC96': 'realDisposableIncome'
        }
    
    def _retry_with_backoff(self, func, max_retries=3, initial_delay=1.0):
        """Retry a function with exponential backoff on rate limit errors."""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        raise
                else:
                    raise
        return None
    
    def _fetch_fred_series(self, series_id, start_date=None, end_date=None):
        """
        Fetch data from FRED API for a specific series using direct API calls.
        
        Args:
            series_id: FRED series ID
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame with timestamp and value columns
        """
        import requests
        
        def fetch():
            url = 'https://api.stlouisfed.org/fred/series/observations'
            params = {
                'series_id': series_id,
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'limit': 100000  # Get all available data
            }
            
            # Add date parameters if provided
            if start_date:
                if isinstance(start_date, datetime):
                    params['observation_start'] = start_date.strftime('%Y-%m-%d')
                else:
                    params['observation_start'] = str(start_date)
            
            if end_date:
                if isinstance(end_date, datetime):
                    params['observation_end'] = end_date.strftime('%Y-%m-%d')
                else:
                    params['observation_end'] = str(end_date)
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        
        try:
            json_data = self._retry_with_backoff(fetch)
            
            if not json_data or 'observations' not in json_data:
                logger.warning(f"No data returned for FRED series {series_id}")
                return None
            
            observations = json_data['observations']
            if not observations:
                logger.warning(f"No observations in FRED series {series_id}")
                return None
            
            # Convert to DataFrame
            data_list = []
            for obs in observations:
                if obs['value'] != '.':  # FRED uses '.' for missing values
                    data_list.append({
                        'timestamp': pd.to_datetime(obs['date'], utc=True),
                        'value': float(obs['value'])
                    })
            
            if not data_list:
                logger.warning(f"No valid observations for FRED series {series_id}")
                return None
            
            df = pd.DataFrame(data_list)
            
            logger.info(f"Fetched {len(df)} records for FRED series {series_id}")
            if not df.empty:
                logger.info(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {e}")
            return None
    
    def update_asset_data(self, series_id, macro_asset_id):
        """
        Update data for a single FRED series.
        
        Args:
            series_id: FRED series ID (e.g., "FF")
            macro_asset_id: Internal macro asset ID (e.g., "fedFundsRate")
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Updating {macro_asset_id} ({series_id})...")
        
        # Get existing data
        existing_data = self._get_existing_data(macro_asset_id)
        
        if not existing_data.empty:
            # Get the latest date and fetch only new data
            latest_date = existing_data['timestamp'].max()
            start_date = latest_date + timedelta(days=1)
            
            logger.info(f"Fetching new data for {macro_asset_id} from {start_date.date()}")
            new_data = self._fetch_fred_series(series_id, start_date=start_date)
        else:
            # Fetch all available data with explicit date range to avoid fredapi bug
            logger.info(f"Fetching all available data for {macro_asset_id}")
            start_date = datetime(2000, 1, 1)  # Start from 2000 for sufficient history
            end_date = datetime.now()
            new_data = self._fetch_fred_series(series_id, start_date=start_date, end_date=end_date)
        
        if new_data is None or new_data.empty:
            logger.info(f"No new data for {macro_asset_id}")
            return True  # Not an error, just no new data
        
        # Add metadata columns
        new_data['macro_asset_id'] = macro_asset_id
        new_data['source'] = 'FRED'
        
        # Determine frequency based on data pattern
        if len(new_data) > 1:
            time_diff = (new_data['timestamp'].iloc[1] - new_data['timestamp'].iloc[0]).days
            if time_diff <= 1:
                frequency = 'daily'
            elif time_diff <= 7:
                frequency = 'weekly'
            elif time_diff <= 31:
                frequency = 'monthly'
            else:
                frequency = 'quarterly'
        else:
            frequency = 'unknown'
        
        new_data['frequency'] = frequency
        
        # Combine with existing data if present
        if not existing_data.empty:
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            # Remove duplicates based on timestamp
            combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='last')
        else:
            combined_data = new_data
        
        # Save updated data
        success = self._save_data(macro_asset_id, combined_data)
        
        if success:
            logger.info(f"✅ {macro_asset_id}: Updated with {len(new_data)} new records")
        else:
            logger.error(f"❌ {macro_asset_id}: Failed to save data")
            
        return success
    
    def update_all_series(self):
        """Update all FRED data series."""
        results = []
        
        for series_id, macro_asset_id in self.series_mapping.items():
            try:
                success = self.update_asset_data(series_id, macro_asset_id)
                results.append((macro_asset_id, success))
                
                # Small delay to avoid rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error updating {macro_asset_id}: {e}")
                results.append((macro_asset_id, False))
        
        # Summary
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        logger.info(f"FRED update completed: {len(successful)}/{len(results)} successful")
        
        if failed:
            logger.error("Failed series:")
            for macro_asset_id, _ in failed:
                logger.error(f"  - {macro_asset_id}")
        
        return results


class MacroCalculationsManager(MacroDataManager):
    """
    Manager for calculated macro metrics (ratios, derived indicators).
    """
    
    def __init__(self, base_path=None):
        super().__init__(base_path, source_type='calculated')
    
    def _load_source_data(self, asset_id, source_type):
        """
        Load data from a specific source directory (yahoo, fred, etc).
        
        Args:
            asset_id: The macro asset identifier
            source_type: The source type ('yahoo', 'fred', 'calculated')
            
        Returns:
            DataFrame with the data or empty DataFrame if not found
        """
        try:
            # Use parent directory to access other source types
            macro_data_path = self.base_path.parent
            data_path = macro_data_path / source_type / asset_id / 'data.parquet'
            if data_path.exists():
                return pd.read_parquet(data_path)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {asset_id} from {source_type}: {e}")
            return pd.DataFrame()
    
    def calculate_rty_ym_ratio(self):
        """
        Calculate RTY/YM ratio from existing parquet data.
        
        Returns:
            Boolean indicating success
        """
        logger.info("Calculating RTY/YM ratio...")
        
        try:
            # Load RTY and YM data from Yahoo source
            rty_data = self._load_source_data('rty', 'yahoo')
            ym_data = self._load_source_data('ym', 'yahoo')
            
            if rty_data.empty or ym_data.empty:
                logger.error("RTY or YM data not available for ratio calculation")
                return False
            
            # Merge on timestamp using close prices
            rty_close = rty_data[['timestamp', 'close']].rename(columns={'close': 'rty_close'})
            ym_close = ym_data[['timestamp', 'close']].rename(columns={'close': 'ym_close'})
            
            merged = pd.merge(rty_close, ym_close, on='timestamp', how='inner')
            
            if merged.empty:
                logger.error("No matching timestamps between RTY and YM data")
                return False
            
            # Calculate ratio
            merged['value'] = merged['rty_close'] / merged['ym_close']
            
            # Prepare final data
            ratio_data = merged[['timestamp', 'value']].copy()
            ratio_data['macro_asset_id'] = 'rty_ym_ratio'
            ratio_data['source'] = 'CALCULATED'
            ratio_data['frequency'] = 'daily'
            
            # Save the ratio data
            success = self._save_data('rty_ym_ratio', ratio_data)
            
            if success:
                logger.info(f"✅ RTY/YM ratio: Calculated {len(ratio_data)} data points")
            else:
                logger.error("❌ RTY/YM ratio: Failed to save data")
            
            return success
            
        except Exception as e:
            logger.error(f"Error calculating RTY/YM ratio: {e}")
            return False
    
    def calculate_net_liquidity(self):
        """
        Calculate US Net Liquidity = Fed Assets - Repo - TGA
        
        Returns:
            Boolean indicating success
        """
        logger.info("Calculating US Net Liquidity...")
        
        try:
            # Load required components from FRED source
            fed_assets = self._load_source_data('fedTotalAssets', 'fred')
            repo = self._load_source_data('repoAgreements', 'fred') 
            tga = self._load_source_data('tga', 'fred')
            
            if fed_assets.empty or repo.empty or tga.empty:
                logger.error("Required data not available for net liquidity calculation")
                return False
            
            # Prepare data for merging
            fed_df = fed_assets[['timestamp', 'value']].rename(columns={'value': 'fed_assets'})
            repo_df = repo[['timestamp', 'value']].rename(columns={'value': 'repo'})
            tga_df = tga[['timestamp', 'value']].rename(columns={'value': 'tga'})
            
            # Convert Fed Assets from millions to billions
            fed_df['fed_assets'] = fed_df['fed_assets'] / 1000
            
            # Merge all components
            merged = fed_df.merge(repo_df, on='timestamp', how='outer')
            merged = merged.merge(tga_df, on='timestamp', how='outer')
            
            # Forward fill missing values
            merged = merged.sort_values('timestamp').fillna(method='ffill')
            
            # Calculate net liquidity (all in billions)
            merged['value'] = merged['fed_assets'] - merged['repo'] - merged['tga']
            
            # Prepare final data
            net_liquidity = merged[['timestamp', 'value']].copy()
            net_liquidity['macro_asset_id'] = 'usNetLiquidity'
            net_liquidity['source'] = 'CALCULATED'
            net_liquidity['frequency'] = 'daily'
            
            # Remove any remaining NaN values
            net_liquidity = net_liquidity.dropna()
            
            # Save the data
            success = self._save_data('usNetLiquidity', net_liquidity)
            
            if success:
                logger.info(f"✅ US Net Liquidity: Calculated {len(net_liquidity)} data points")
            else:
                logger.error("❌ US Net Liquidity: Failed to save data")
            
            return success
            
        except Exception as e:
            logger.error(f"Error calculating US Net Liquidity: {e}")
            return False
    
    def calculate_global_cb_liquidity(self):
        """
        Calculate Global CB Liquidity = US Net Liquidity + BOJ Assets (USD) + ECB Assets (USD)
        
        Returns:
            Boolean indicating success  
        """
        logger.info("Calculating Global CB Liquidity...")
        
        try:
            # Load required components from appropriate sources
            us_liq = self._load_source_data('usNetLiquidity', 'calculated')
            boj = self._load_source_data('bojAssets', 'fred')
            ecb = self._load_source_data('ecbAssets', 'fred')
            usdjpy = self._load_source_data('usdjpy', 'yahoo')
            usdeur = self._load_source_data('usdeur', 'yahoo')
            
            if any(df.empty for df in [us_liq, boj, ecb, usdjpy, usdeur]):
                logger.error("Required data not available for global CB liquidity calculation")
                return False
            
            # Prepare data for merging
            us_df = us_liq[['timestamp', 'value']].rename(columns={'value': 'us_liquidity'})
            boj_df = boj[['timestamp', 'value']].rename(columns={'value': 'boj_assets'})
            ecb_df = ecb[['timestamp', 'value']].rename(columns={'value': 'ecb_assets'})
            
            # Use close prices for FX rates
            jpy_df = usdjpy[['timestamp', 'close']].rename(columns={'close': 'usdjpy'})
            eur_df = usdeur[['timestamp', 'close']].rename(columns={'close': 'usdeur'})
            
            # Merge all components
            merged = us_df.merge(boj_df, on='timestamp', how='outer')
            merged = merged.merge(ecb_df, on='timestamp', how='outer')
            merged = merged.merge(jpy_df, on='timestamp', how='outer')
            merged = merged.merge(eur_df, on='timestamp', how='outer')
            
            # Forward fill missing values
            merged = merged.sort_values('timestamp').fillna(method='ffill')
            
            # Convert BOJ assets (100M yen -> USD billions)
            merged['boj_usd'] = (merged['boj_assets'] * 100) / merged['usdjpy'] / 1000
            
            # Convert ECB assets (M EUR -> USD billions)  
            merged['ecb_usd'] = merged['ecb_assets'] * merged['usdeur'] / 1000
            
            # Calculate global liquidity
            merged['value'] = merged['us_liquidity'] + merged['boj_usd'] + merged['ecb_usd']
            
            # Prepare final data
            global_liquidity = merged[['timestamp', 'value']].copy()
            global_liquidity['macro_asset_id'] = 'globalCbLiquidity'
            global_liquidity['source'] = 'CALCULATED'
            global_liquidity['frequency'] = 'daily'
            
            # Also save the USD-converted components
            boj_usd_data = merged[['timestamp', 'boj_usd']].rename(columns={'boj_usd': 'value'})
            boj_usd_data['macro_asset_id'] = 'bojAssetsUSD'
            boj_usd_data['source'] = 'CALCULATED'
            boj_usd_data['frequency'] = 'daily'
            
            ecb_usd_data = merged[['timestamp', 'ecb_usd']].rename(columns={'ecb_usd': 'value'})
            ecb_usd_data['macro_asset_id'] = 'ecbAssetsUSD'
            ecb_usd_data['source'] = 'CALCULATED'
            ecb_usd_data['frequency'] = 'daily'
            
            # Remove any remaining NaN values
            global_liquidity = global_liquidity.dropna()
            boj_usd_data = boj_usd_data.dropna()
            ecb_usd_data = ecb_usd_data.dropna()
            
            # Save all data
            results = []
            results.append(self._save_data('globalCbLiquidity', global_liquidity))
            results.append(self._save_data('bojAssetsUSD', boj_usd_data))
            results.append(self._save_data('ecbAssetsUSD', ecb_usd_data))
            
            success = all(results)
            
            if success:
                logger.info(f"✅ Global CB Liquidity: Calculated {len(global_liquidity)} data points")
                logger.info(f"✅ BOJ Assets USD: Calculated {len(boj_usd_data)} data points")
                logger.info(f"✅ ECB Assets USD: Calculated {len(ecb_usd_data)} data points")
            else:
                logger.error("❌ Global CB Liquidity: Failed to save some data")
            
            return success
            
        except Exception as e:
            logger.error(f"Error calculating Global CB Liquidity: {e}")
            return False
    
    def calculate_yield_curve_regime(self, window=20):
        """
        Calculate yield curve regime classification based on 2Y and 10Y treasury rates.
        
        Args:
            window: Rolling window for calculating rate changes (default 20 days)
            
        Returns:
            Boolean indicating success
        """
        logger.info("Calculating yield curve regime...")
        
        try:
            # Load treasury data from FRED source
            treasury_2y = self._load_source_data('treasury2Y', 'fred')
            treasury_10y = self._load_source_data('treasury10Y', 'fred')
            
            if treasury_2y.empty or treasury_10y.empty:
                logger.error("Treasury 2Y or 10Y data not available for regime calculation")
                return False
            
            # Merge treasury data on timestamp
            t2y = treasury_2y[['timestamp', 'value']].rename(columns={'value': 't2y_rate'})
            t10y = treasury_10y[['timestamp', 'value']].rename(columns={'value': 't10y_rate'})
            
            merged = pd.merge(t2y, t10y, on='timestamp', how='inner')
            
            if merged.empty:
                logger.error("No matching timestamps between 2Y and 10Y treasury data")
                return False
            
            # Sort by timestamp
            merged = merged.sort_values('timestamp')
            
            # Calculate 2s10s spread
            merged['spread_2s10s'] = merged['t10y_rate'] - merged['t2y_rate']
            
            # Calculate rolling changes
            merged['delta_2y'] = merged['t2y_rate'].diff(window)
            merged['delta_10y'] = merged['t10y_rate'].diff(window)  
            merged['delta_spread'] = merged['spread_2s10s'].diff(window)
            
            # Classify regime for each row
            def classify_regime(row):
                """Classify yield curve regime based on changes in rates."""
                delta_2y = row['delta_2y']
                delta_10y = row['delta_10y']
                delta_spread = row['delta_spread']
                
                # Check for NaN values
                if pd.isna(delta_2y) or pd.isna(delta_10y) or pd.isna(delta_spread):
                    return "neutral"
                
                tolerance = 1e-5
                if abs(delta_2y) < tolerance or abs(delta_10y) < tolerance or abs(delta_spread) < tolerance:
                    return "neutral"
                    
                # Determine overall direction if both rates move in the same way
                if delta_2y < 0 and delta_10y < 0:
                    direction = "bull"
                elif delta_2y > 0 and delta_10y > 0:
                    direction = "bear"
                # Twist scenarios when the rates move oppositely
                elif delta_2y > 0 and delta_10y < 0:
                    return "steepener_twist"
                elif delta_2y < 0 and delta_10y > 0:
                    return "flattener_twist"
                else:
                    direction = "neutral"
                
                # Decide if the yield curve is steepening or flattening
                if delta_spread > 0:
                    curve = "steepener"
                elif delta_spread < 0:
                    curve = "flattener"
                else:
                    curve = "neutral"
                
                # If we have a neutral component, classify as neutral overall
                if direction == "neutral" or curve == "neutral":
                    return "neutral"
                
                return f"{direction}_{curve}"
            
            # Apply regime classification
            merged['regime'] = merged.apply(classify_regime, axis=1)
            
            # Prepare final data with both regime and spread
            regime_data = merged[['timestamp', 'regime', 'spread_2s10s']].copy()
            regime_data['macro_asset_id'] = 'yieldCurveRegime'
            regime_data['source'] = 'CALCULATED'
            regime_data['frequency'] = 'daily'
            
            # Remove rows with NaN values (from rolling window)
            regime_data = regime_data.dropna()
            
            # Save the regime data
            success = self._save_data('yieldCurveRegime', regime_data)
            
            if success:
                logger.info(f"✅ Yield Curve Regime: Calculated {len(regime_data)} data points")
                # Log regime distribution
                regime_counts = regime_data['regime'].value_counts()
                logger.info(f"Regime distribution: {dict(regime_counts.head())}")
            else:
                logger.error("❌ Yield Curve Regime: Failed to save data")
            
            return success
            
        except Exception as e:
            logger.error(f"Error calculating yield curve regime: {e}")
            return False

    def update_all_calculations(self):
        """Update all calculated metrics."""
        logger.info("Updating all calculated metrics...")
        
        results = []
        
        # Calculate RTY/YM ratio
        results.append(('rty_ym_ratio', self.calculate_rty_ym_ratio()))
        
        # Calculate US Net Liquidity
        results.append(('usNetLiquidity', self.calculate_net_liquidity()))
        
        # Calculate Global CB Liquidity
        results.append(('globalCbLiquidity', self.calculate_global_cb_liquidity()))
        
        # Calculate Yield Curve Regime
        results.append(('yieldCurveRegime', self.calculate_yield_curve_regime()))
        
        # Summary
        successful = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        
        logger.info(f"Calculations completed: {len(successful)}/{len(results)} successful")
        
        if failed:
            logger.error("Failed calculations:")
            for calc_name, _ in failed:
                logger.error(f"  - {calc_name}")
        
        return results


# Convenience functions for external use
def update_yahoo_data():
    """Update all Yahoo Finance data."""
    logging.basicConfig(level=logging.INFO)
    manager = YahooDataManager()
    return manager.update_all_assets()

def update_fred_data():
    """Update all FRED data."""  
    logging.basicConfig(level=logging.INFO)
    manager = FredDataManager()
    return manager.update_all_series()

def update_macro_calculations():
    """Update all calculated metrics."""
    logging.basicConfig(level=logging.INFO)
    manager = MacroCalculationsManager()
    return manager.update_all_calculations()

def update_all_macro_data():
    """Update all macro data sources and calculations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting complete macro data update...")
    
    # Update Yahoo Finance data
    logger.info("=== UPDATING YAHOO FINANCE DATA ===")
    yahoo_results = update_yahoo_data()
    
    # Update FRED data
    logger.info("=== UPDATING FRED DATA ===")
    fred_results = update_fred_data()
    
    # Update calculations
    logger.info("=== UPDATING CALCULATIONS ===")
    calc_results = update_macro_calculations()
    
    # Overall summary
    total_yahoo = len(yahoo_results)
    successful_yahoo = len([r for r in yahoo_results if r[1]])
    
    total_fred = len(fred_results)
    successful_fred = len([r for r in fred_results if r[1]])
    
    total_calc = len(calc_results) 
    successful_calc = len([r for r in calc_results if r[1]])
    
    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"Yahoo Finance: {successful_yahoo}/{total_yahoo} successful")
    logger.info(f"FRED Data: {successful_fred}/{total_fred} successful")
    logger.info(f"Calculations: {successful_calc}/{total_calc} successful")
    logger.info(f"Overall: {successful_yahoo + successful_fred + successful_calc}/{total_yahoo + total_fred + total_calc} successful")
    
    return {
        'yahoo': yahoo_results,
        'fred': fred_results, 
        'calculations': calc_results
    }


if __name__ == "__main__":
    # Run complete update
    results = update_all_macro_data()