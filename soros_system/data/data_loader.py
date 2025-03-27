import pandas as pd
import numpy as np
import os
import logging
from pycoingecko import CoinGeckoAPI

class DataLoader:
    """
    Handles loading and preprocessing of asset price data from CSV files.
    """
    def __init__(self, data_path, btc_data_path, ssr_data_path=None):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (str): Path to the directory containing asset data files
            btc_data_path (str): Path to the Bitcoin data file
            ssr_data_path (str, optional): Path to the SSR data file
        """
        self.data_path = data_path
        self.btc_data_path = btc_data_path
        self.ssr_data_path = ssr_data_path
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded data
        self.asset_data_cache = {}
        self.btc_data_cache = None
        
        # Initialize ticker mapping
        self.ticker_mapping = self._get_ticker_mapping()
        
    def _get_ticker_mapping(self):
        """
        Create a mapping between asset tickers and their IDs using CoinGecko API.
        
        Returns:
            dict: Mapping of tickers to asset IDs
        """
        try:
            cg = CoinGeckoAPI()
            coins_list = cg.get_coins_list()
            coins_df = pd.DataFrame(coins_list)
            
            # Known mappings for assets that might have different tickers in CoinGecko
            known_mappings = {
                # Special cases
                'VIRTUAL': 'virtual-protocol', 
                'HYPE': 'hyperliquid', 
                'YNE': 'yesnoerror',
                # Common cryptocurrencies with standard tickers
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'BNB': 'binancecoin',
                'XRP': 'ripple',
                'ADA': 'cardano',
                'SOL': 'solana',
                'DOGE': 'dogecoin',
                'DOT': 'polkadot',
                'AVAX': 'avalanche-2',
                'MATIC': 'polygon',
                'LINK': 'chainlink',
                'UNI': 'uniswap',
                'LTC': 'litecoin'
            }
            
            # Initialize with known mappings
            mapping = {ticker: coin_id for ticker, coin_id in known_mappings.items()}
            
            # Add mappings from CoinGecko data
            for _, row in coins_df.iterrows():
                ticker = row['symbol'].upper()
                if ticker not in mapping:
                    mapping[ticker] = row['id']
            
            return mapping
        except Exception as e:
            self.logger.error(f"Failed to get ticker mapping: {e}")
            # Return at least our hardcoded mappings to ensure we have basic functionality
            return {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'BNB': 'binancecoin',
                'XRP': 'ripple',
                'ADA': 'cardano'
            }
        
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
            return self.asset_data_cache[asset_id]
        
        data_path = f"{self.data_path}{asset_id}_candles.csv"
        if not os.path.exists(data_path):
            self.logger.warning(f"File not found for {asset_id}: {data_path}")
            return pd.DataFrame()
        
        try:
            data = pd.read_csv(data_path)
            data['date'] = pd.to_datetime(data['date'])
            data.dropna(subset=['close'], inplace=True)
            if data.empty:
                self.logger.warning(f"No valid data for {asset_id} after dropping NaN in 'close'.")
                return pd.DataFrame()
            
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
            
            self.logger.info(f"Successfully loaded data for {asset_id} with {len(data)} rows.")
            
            # Cache the data
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