"""
SSR (Stablecoin Supply Ratio) signals for the Soros System.

These signals are based on the relationship between stablecoin supply and Bitcoin market cap,
which can indicate market sentiment and capital flow dynamics.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Optional, Dict, Any, Union, List

from .signal_base import SignalBase
from .signal_registry import register_signal


class SSRSignalBase(SignalBase):
    """Base class for SSR signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal with parameters."""
        super().__init__(params)
        self.logger = logging.getLogger(__name__)
        self.ssr_data = None
        self.ssr_data_loaded = False
        self._load_ssr_data()
    
    def _load_ssr_data(self) -> bool:
        """Load SSR data from file.
        
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        # Try to get path from params
        ssr_path = self.params.get('ssr_data_path')
        self.logger.info(f"SSR path from params: {ssr_path}")
        
        # If no path in params, try environment variable
        if not ssr_path:
            ssr_path = os.environ.get('SSR_DATA_PATH')
            self.logger.info(f"SSR path from env var: {ssr_path}")
            
        # If still no path, use default location
        if not ssr_path:
            ssr_path = 'data/onchainData/BTC_SSR.csv'
            self.logger.info(f"Using default SSR path: {ssr_path}")
        
        # Check if the file exists
        if not os.path.exists(ssr_path):
            # Try a few common alternatives
            alt_paths = [
                'data/onchainData/BTC_SSR.csv',
                '/Users/valter.rebelo/MissionControl/data/onchainData/BTC_SSR.csv',
                'BTC_SSR.csv'
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    ssr_path = alt_path
                    self.logger.info(f"Found alternative SSR path: {ssr_path}")
                    break
            else:
                self.logger.error(f"SSR data file not found at {ssr_path} or any alternatives")
                return False
        
        # First try to use SSRDataHandler if available
        try:
            from ..data.ssr_data import SSRDataHandler
            self.logger.info(f"Using SSRDataHandler to load data from {ssr_path}")
            handler = SSRDataHandler(ssr_path)
            if handler.ssr_data is not None:
                self.ssr_data = handler.ssr_data
                self.ssr_data_loaded = True
                self.logger.info(f"Successfully loaded SSR data using SSRDataHandler: {len(self.ssr_data)} rows")
                return True
        except (ImportError, Exception) as e:
            self.logger.warning(f"Could not use SSRDataHandler: {e}, falling back to direct loading")
            
        # Check if file exists
        if not os.path.exists(ssr_path):
            self.logger.error(f"SSR data file not found at {ssr_path}")
            self.ssr_data_loaded = False
            return False
            
        try:
            # Load data directly
            self.logger.info(f"Loading SSR data directly from {ssr_path}")
            self.ssr_data = pd.read_csv(ssr_path)
            
            # Ensure date column is properly formatted
            if 'date' in self.ssr_data.columns:
                self.logger.info("Using 'date' column from SSR data")
                self.ssr_data['date'] = pd.to_datetime(self.ssr_data['date'])
                self.ssr_data.set_index('date', inplace=True)
            elif 'timestamp' in self.ssr_data.columns:
                self.logger.info("Using 'timestamp' column from SSR data")
                self.ssr_data['timestamp'] = pd.to_datetime(self.ssr_data['timestamp'])
                self.ssr_data.set_index('timestamp', inplace=True)
            else:
                self.logger.warning(f"No date/timestamp column found in SSR data. Columns: {self.ssr_data.columns.tolist()}")
                # Try to create a date index if there's none
                if self.ssr_data.index.dtype != 'datetime64[ns]':
                    self.logger.warning("Attempting to create a datetime index from numeric index")
                    try:
                        # Assume data is daily and starts from a reasonable date
                        start_date = pd.Timestamp('2010-01-01')
                        dates = [start_date + pd.Timedelta(days=i) for i in range(len(self.ssr_data))]
                        self.ssr_data.index = dates
                        self.logger.info("Created synthetic date index for SSR data")
                    except Exception as e:
                        self.logger.error(f"Failed to create synthetic date index: {e}")
                
            # Set flag to indicate data was loaded
            self.ssr_data_loaded = True
            self.logger.info(f"Successfully loaded SSR data directly: {len(self.ssr_data)} rows with columns {self.ssr_data.columns.tolist()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading SSR data: {e}")
            self.ssr_data_loaded = False
            return False
    
    def get_required_columns(self) -> List[str]:
        """Get the required columns for this signal."""
        return ['close']
    
    def get_min_required_samples(self) -> int:
        """Get the minimum required samples for this signal."""
        return 30
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate that the data contains the required columns."""
        # Check if price data has required columns
        if not super().validate(data, asset_id):
            return False
            
        # Check if SSR data is loaded
        if not self.ssr_data_loaded or self.ssr_data is None or self.ssr_data.empty:
            # Try loading again if not already loaded
            if not self.ssr_data_loaded:
                if not self._load_ssr_data():
                    self.logger.warning("SSR data not available, signal cannot be calculated")
                    return False
            else:
                self.logger.warning("SSR data not available, signal cannot be calculated")
                return False
                
        # Print some diagnostic info about the datasets
        self.logger.debug(f"Price data index: {data.index[0]} to {data.index[-1]} ({len(data)} rows)")
        self.logger.debug(f"SSR data index: {self.ssr_data.index[0]} to {self.ssr_data.index[-1]} ({len(self.ssr_data)} rows)")
        
        return True
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with signal values
        """
        # Validate data
        if not self.validate(data, asset_id):
            return pd.Series(index=data.index)
        
        # Add debug info about SSR data
        self.logger.info(f"SSR data loaded: {self.ssr_data_loaded}, rows: {len(self.ssr_data) if self.ssr_data is not None else 0}")
        if self.ssr_data is not None:
            self.logger.info(f"SSR data columns: {self.ssr_data.columns.tolist()}")
            self.logger.info(f"SSR data date range: {self.ssr_data.index[0]} to {self.ssr_data.index[-1]}")
            self.logger.info(f"SSR data index type: {self.ssr_data.index.dtype}")
            self.logger.info(f"Price data index type: {data.index.dtype}")
            
            # Print some sample dates to compare formats
            self.logger.info(f"SSR first 3 dates: {self.ssr_data.index[:3].tolist()}")
            self.logger.info(f"Price first 3 indices: {data.index[:3].tolist()}")
            
        # Get the SSR data as a simple series
        if 'ssr_oscillator' in self.ssr_data.columns:
            ssr_series = self.ssr_data['ssr_oscillator'].copy()
        else:
            self.logger.warning("ssr_oscillator column not found in SSR data")
            return pd.Series(index=data.index)
        
        # Initialize signal values array
        signal_values = pd.Series(0.0, index=data.index)
        
        try:
            # Check if we're dealing with a numeric index (no actual date information)
            if data.index.dtype == 'int64' or not hasattr(data.index, 'strftime'):
                self.logger.info("Price data has numeric index - using proper date mapping method")
                
                # IMPROVED APPROACH: Map numeric indices to actual dates
                # First, try to determine if the asset data has a date column we can use
                if hasattr(data, 'date') and isinstance(data.date, pd.Series):
                    self.logger.info("Using 'date' column from price data for mapping")
                    date_series = data.date
                elif 'date' in data.columns:
                    self.logger.info("Using 'date' column from price data columns for mapping")
                    date_series = data['date']
                else:
                    # If no date information, we need to estimate a date range
                    self.logger.info("No date column found, estimating dates based on asset ID and data size")
                    
                    # Get SSR data date range
                    ssr_start = self.ssr_data.index.min()
                    ssr_end = self.ssr_data.index.max()
                    
                    # Create date range for price data based on length and asset
                    if asset_id == 'bitcoin':
                        # Bitcoin typically starts from 2010
                        price_start = pd.Timestamp('2010-07-01')
                    elif asset_id == 'ethereum':
                        # Ethereum typically starts from 2015
                        price_start = pd.Timestamp('2015-08-01')
                    else:
                        # Default to a reasonable start date
                        price_start = pd.Timestamp('2017-01-01')
                    
                    # Create date range for the price data
                    date_series = pd.Series([price_start + pd.Timedelta(days=i) for i in range(len(data))], index=data.index)
                    self.logger.info(f"Created estimated date range from {date_series.iloc[0]} to {date_series.iloc[-1]}")
                
                # Now create the SSR values based on the dates
                first_ssr_date = self.ssr_data.index.min()
                self.logger.info(f"First SSR date: {first_ssr_date}")
                
                # For each price date, find the corresponding SSR value or the nearest one
                for idx in data.index:
                    price_date = date_series.loc[idx]
                    
                    # Make sure price_date is a Timestamp
                    if not isinstance(price_date, pd.Timestamp):
                        price_date = pd.Timestamp(price_date)
                    
                    # Skip dates before SSR data starts
                    if price_date < first_ssr_date:
                        # For dates before SSR data, set to 0
                        signal_values.loc[idx] = 0
                        continue
                    
                    # Find the closest SSR date to this price date
                    closest_ssr_date = self.ssr_data.index[self.ssr_data.index.get_indexer([price_date], method='nearest')[0]]
                    signal_values.loc[idx] = ssr_series.loc[closest_ssr_date]
                
                self.logger.info(f"Mapped {len(signal_values[signal_values != 0])} dates to SSR values")
                
                # Create the final signal based on the class type
                return self._create_signal(signal_values, data.index)
            
            # If we have dates in the index, proceed with date matching
            # Convert all dates to strings in 'YYYY-MM-DD' format to avoid any timezone or format issues
            ssr_dates_str = ssr_series.index.strftime('%Y-%m-%d')
            price_dates_str = data.index.strftime('%Y-%m-%d')
            
            # Create a simple lookup dictionary for the SSR values
            ssr_dict = dict(zip(ssr_dates_str, ssr_series.values))
            
            # Map values using direct string comparison
            common_dates = set(price_dates_str) & set(ssr_dates_str)
            self.logger.info(f"Common string-formatted dates found: {len(common_dates)}")
            
            # Direct mapping for common dates
            for date, date_str in zip(data.index, price_dates_str):
                if date_str in ssr_dict:
                    signal_values[date] = ssr_dict[date_str]
            
            # If no direct matches were found, try mapping to nearest available date
            if signal_values.notna().sum() == 0:
                self.logger.warning("No direct date matches found. Trying nearest date mapping...")
                
                # Convert ssr dates to proper datetime
                ssr_dates = pd.to_datetime(ssr_dates_str)
                
                # For each price date, find the nearest SSR date
                for date, date_str in zip(data.index, price_dates_str):
                    price_date = pd.to_datetime(date_str)  # Convert to datetime for comparison
                    
                    # Check if the price date is in range of SSR dates
                    if price_date < ssr_dates.min() or price_date > ssr_dates.max():
                        # Skip dates outside our SSR data range
                        continue
                    
                    # Find nearest date
                    # Convert to days since epoch for easy numeric comparison
                    price_days = (price_date - pd.Timestamp('1970-01-01')).days
                    ssr_days = [(d - pd.Timestamp('1970-01-01')).days for d in ssr_dates]
                    
                    # Find the closest date by absolute difference
                    closest_idx = min(range(len(ssr_days)), key=lambda i: abs(ssr_days[i] - price_days))
                    closest_date_str = ssr_dates_str[closest_idx]
                    
                    # Map the value
                    signal_values[date] = ssr_dict[closest_date_str]
            
            # Forward fill any remaining NaN values
            signal_values = signal_values.ffill()
            
            # Fill any remaining NaNs at the beginning with 0
            signal_values = signal_values.fillna(0)
            
            self.logger.info(f"Final aligned SSR data: non-null values: {signal_values.count()} out of {len(data)}")
            
        except Exception as e:
            self.logger.error(f"Error calculating values for {self.__class__.__name__} on {asset_id}: {e}")
            # Return a series of zeros for safety
            signal_values = pd.Series(0.0, index=data.index)
            
        # Create signal series (to be implemented by subclasses)
        return self._create_signal(signal_values, data.index)
    
    def _create_signal(self, ssr_values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Create the signal based on SSR values.
        
        Args:
            ssr_values: Series with SSR values
            index: DatetimeIndex for the output signal
            
        Returns:
            Series with signal values
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _create_signal")


@register_signal
class SSR_RiskOn(SSRSignalBase):
    """Signal indicating risk-on sentiment based on SSR being positive."""
    
    def _create_signal(self, ssr_values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Create risk-on signal based on SSR values.
        
        Risk-on is indicated by positive SSR values.
        
        Args:
            ssr_values: Series with SSR values
            index: DatetimeIndex for the output signal
            
        Returns:
            Series with signal values
        """
        try:
            # Check if the values are already properly formatted (all 0s or all 1s)
            unique_values = ssr_values.unique()
            if len(unique_values) == 1 and (unique_values[0] == 0 or unique_values[0] == 1):
                # Values are already properly formatted, just return them
                self.logger.info(f"SSR_RiskOn - values already properly formatted as {unique_values[0]}")
                signal = ssr_values
            else:
                # Generate signal - risk-on when SSR is positive
                signal = (ssr_values > 0).astype(int)
            
            # Verify and count active signals
            active_count = (signal == 1).sum()
            inactive_count = (signal == 0).sum()
            self.logger.info(f"SSR_RiskOn verification: active={active_count}, inactive={inactive_count}, total={len(signal)}")
            
            return signal
        except Exception as e:
            self.logger.error(f"Error creating risk-on signal: {e}")
            return pd.Series(0, index=index)


@register_signal
class SSR_RiskOff(SSRSignalBase):
    """Signal indicating risk-off sentiment based on SSR being negative."""
    
    def _create_signal(self, ssr_values: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Create risk-off signal based on SSR values.
        
        Risk-off is indicated by negative SSR values.
        
        Args:
            ssr_values: Series with SSR values
            index: DatetimeIndex for the output signal
            
        Returns:
            Series with signal values
        """
        try:
            # Check if the values are already properly formatted (all 0s or all 1s)
            unique_values = ssr_values.unique()
            if len(unique_values) == 1 and (unique_values[0] == 0 or unique_values[0] == 1):
                # Values are already properly formatted, just return them
                self.logger.info(f"SSR_RiskOff - values already properly formatted as {unique_values[0]}")
                signal = ssr_values
            else:
                # Generate signal - risk-off when SSR is negative
                signal = (ssr_values < 0).astype(int)
            
            # Verify and count active signals
            active_count = (signal == 1).sum()
            inactive_count = (signal == 0).sum()
            self.logger.info(f"SSR_RiskOff verification: active={active_count}, inactive={inactive_count}, total={len(signal)}")
            
            return signal
        except Exception as e:
            self.logger.error(f"Error creating risk-off signal: {e}")
            return pd.Series(0, index=index)


