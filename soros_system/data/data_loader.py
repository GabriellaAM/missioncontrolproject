import pandas as pd
import numpy as np
import os
import logging
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv

load_dotenv()

class DataLoader:
    """
    Handles loading and preprocessing of asset price data from CSV files.
    """
    def __init__(self, data_path, btc_data_path, ssr_data_path=None, asset_ids=None, market_data_path=None):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (str): Path to the directory containing asset data files
            btc_data_path (str): Path to the Bitcoin data file
            ssr_data_path (str, optional): Path to the SSR data file
            asset_ids (list, optional): List of asset IDs to use for ticker mapping
            market_data_path (str, optional): Path to directory containing market data files
        """
        self.data_path = data_path
        self.btc_data_path = btc_data_path
        self.ssr_data_path = ssr_data_path
        self.market_data_path = market_data_path or "/Users/valter.rebelo/MissionControl/data/micro/assetData/"
        self.logger = logging.getLogger(__name__)
        self.asset_ids = asset_ids or []
        
        # Cache for loaded data
        self.asset_data_cache = {}
        self.btc_data_cache = None
        self.market_data_cache = {}
        
        # Initialize ticker mapping
        self.ticker_mapping = self._get_ticker_mapping()
        
    def _get_ticker_mapping(self):
        """
        Create a mapping between asset tickers and their IDs using CoinGecko API.
        Only includes mappings for assets in asset_ids.
        
        Returns:
            dict: Mapping of tickers to asset IDs
        """
        cg = CoinGeckoAPI(api_key=os.getenv('COINGECKO_API_KEY'))
        coins_list = cg.get_coins_list()
        coins_df = pd.DataFrame(coins_list)
        
        # Known mappings for assets that might have different tickers in CoinGecko
        known_mappings = {'VIRTUAL': 'virtual-protocol', 'HYPE': 'hyperliquid', 'YNE': 'yesnoerror'}
        
        # Only include mappings for assets in self.asset_ids
        mapping = {ticker: coin_id for ticker, coin_id in known_mappings.items() if coin_id in self.asset_ids}
        
        # Filter coins to only include those in asset_ids
        filtered_coins_df = coins_df[coins_df['id'].isin(self.asset_ids)]
        
        # Add mappings from filtered CoinGecko data
        for _, row in filtered_coins_df.iterrows():
            ticker = row['symbol'].upper()
            if ticker not in mapping:
                mapping[ticker] = row['id']
        
        return mapping
    
    def load_market_data(self, asset_id):
        """
        Load market data (market cap, volume) for a specific asset.
        
        Args:
            asset_id (str): ID of the asset to load market data for
            
        Returns:
            pd.DataFrame: DataFrame containing the market data or empty DataFrame if not found
        """
        # Return cached data if available
        if asset_id in self.market_data_cache:
            self.logger.debug(f"Returning cached market data for {asset_id}")
            return self.market_data_cache[asset_id]
        
        # Construct file path
        data_path = os.path.join(self.market_data_path, f"{asset_id}.csv")
        
        if not os.path.exists(data_path):
            self.logger.warning(f"Market data file not found for {asset_id}: {data_path}")
            return pd.DataFrame()
        
        try:
            self.logger.info(f"Loading market data from file: {data_path}")
            data = pd.read_csv(data_path)
            
            # Ensure date column is properly formatted
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            elif 'timestamp' in data.columns:
                data['date'] = pd.to_datetime(data['timestamp'])
                data = data.drop(columns=['timestamp'])
            
            # Cache the data
            self.market_data_cache[asset_id] = data
            
            return data
        except Exception as e:
            self.logger.error(f"Failed to load market data for {asset_id}: {e}")
            return pd.DataFrame()
        
    def load_asset_data(self, asset_id):
        """
        Load data for a specific asset from CSV file.
        
        Args:
            asset_id (str): ID of the asset to load data for
            
        Returns:
            pd.DataFrame: DataFrame containing the asset data
        """
        # Return cached data if available
        if asset_id in self.asset_data_cache:
            self.logger.debug(f"Returning cached data for {asset_id}")
            return self.asset_data_cache[asset_id]
        
        # Construct file path
        data_path = os.path.join(self.data_path, f"{asset_id}_candles.csv")
        
        if not os.path.exists(data_path):
            self.logger.warning(f"File not found for {asset_id}: {data_path}")
            return pd.DataFrame()
        
        try:
            self.logger.info(f"Loading data from file: {data_path}")
            data = pd.read_csv(data_path)
            data['date'] = pd.to_datetime(data['date'])
            data.dropna(subset=['close'], inplace=True)
            
            if data.empty:
                self.logger.warning(f"No valid data for {asset_id} after dropping NaN in 'close'.")
                return pd.DataFrame()
            
            # For non-Bitcoin assets, merge with BTC data for relative pricing
            if asset_id != 'bitcoin':
                try:
                    # Load BTC data
                    btc_data = self.load_btc_data()
                    
                    # Include OHLC from BTC data
                    btc_data_subset = btc_data[['date', 'open', 'high', 'low', 'close']].rename(
                        columns={'open': 'btc_open', 'high': 'btc_high', 'low': 'btc_low', 'close': 'btc_close'}
                    )
                    
                    # Merge asset data with BTC data
                    data = data.merge(btc_data_subset, on='date', how='inner')
                    if data.empty:
                        self.logger.warning(f"No overlapping dates between {asset_id} and Bitcoin data after merge.")
                        return pd.DataFrame()
                    
                    # Calculate BTC-adjusted OHLC prices
                    data[f'{asset_id}_btc_open'] = data['open'] / data['btc_open']
                    data[f'{asset_id}_btc_high'] = data['high'] / data['btc_high']
                    data[f'{asset_id}_btc_low'] = data['low'] / data['btc_low']
                    data[f'{asset_id}_btc'] = data['close'] / data['btc_close']
                    
                    # Drop BTC columns to keep DataFrame clean
                    data = data.drop(columns=['btc_open', 'btc_high', 'btc_low', 'btc_close'], errors='ignore')
                except Exception as e:
                    self.logger.error(f"Error merging Bitcoin data for {asset_id}: {e}")
                    return pd.DataFrame()
            
            # Try to add market data if available
            try:
                market_data = self.load_market_data(asset_id)
                if not market_data.empty and 'date' in market_data.columns:
                    # Only keep relevant columns to avoid duplicates
                    market_data_subset = market_data[['date']].copy()
                    
                    # Add available market data columns
                    market_columns = ['market_cap', 'total_volume']
                    for col in market_columns:
                        if col in market_data.columns:
                            market_data_subset[col] = market_data[col]
                        
                    # Merge with asset data
                    data = data.merge(market_data_subset, on='date', how='left')
                    self.logger.info(f"Added market data for {asset_id}")
            except Exception as e:
                self.logger.warning(f"Could not add market data for {asset_id}: {e}")
            
            self.logger.info(f"Successfully loaded data for {asset_id} with {len(data)} rows.")
            
            # Cache the data with asset_id as the key
            self.asset_data_cache[asset_id] = data
            
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data for {asset_id}: {e}")
            return pd.DataFrame()
    
    def load_btc_data(self):
        """
        Load Bitcoin data from CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing the Bitcoin data
        """
        # Return cached data if available
        if self.btc_data_cache is not None:
            return self.btc_data_cache
        
        try:
            btc_data = pd.read_csv(self.btc_data_path)
            btc_data['date'] = pd.to_datetime(btc_data['date'])
            btc_data.dropna(subset=['close'], inplace=True)
            
            # Cache the data
            self.btc_data_cache = btc_data
            return btc_data
        except Exception as e:
            self.logger.error(f"Failed to load Bitcoin data: {e}")
            return pd.DataFrame()
    
    def get_ticker_from_id(self, asset_id):
        """
        Get the ticker symbol for a given asset ID.
        
        Args:
            asset_id (str): Asset ID to find ticker for
            
        Returns:
            str: Ticker symbol for the asset, or uppercased asset_id if not found
        """
        # Create inverse mapping
        id_to_ticker = {v: k for k, v in self.ticker_mapping.items()}
        return id_to_ticker.get(asset_id, asset_id.upper())
    
    def clear_cache(self):
        """Clear all cached data."""
        self.asset_data_cache = {}
        self.btc_data_cache = None
        self.market_data_cache = {} 