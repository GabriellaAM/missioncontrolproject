"""
SSR-based signals for the Soros System.

These signals are based on the Stablecoin Supply Ratio (SSR) indicator which measures Bitcoin's scarcity.
Each signal returns 1 if the specified condition is detected, 0 otherwise.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from .signal_base import SignalBase
from .signal_registry import register_signal
from ..data.ssr_data import SSRDataHandler


class SSRSignalBase(SignalBase):
    """Base class for all SSR signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.ssr_handler = None
        
        # Set up the SSR data handler
        ssr_data_path = params.get('data_path', 'data/onchainData/BTC_SSR.csv')
        self._setup_ssr_handler(ssr_data_path)
    
    def _setup_ssr_handler(self, data_path):
        """Set up the SSR data handler."""
        try:
            self.ssr_handler = SSRDataHandler(data_path)
            if self.ssr_handler.ssr_data is not None:
                self.logger.info(f"Successfully set up SSR data handler with {len(self.ssr_handler.ssr_data)} rows")
                # Log the date range of available data
                min_date = self.ssr_handler.ssr_data.index.min()
                max_date = self.ssr_handler.ssr_data.index.max()
                self.logger.info(f"SSR data available from {min_date} to {max_date}")
            else:
                self.logger.error("Failed to set up SSR data handler")
        except Exception as e:
            self.logger.error(f"Error setting up SSR data handler: {str(e)}")
            self.ssr_handler = None
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        # SSR signals don't rely on price columns in the DataFrame itself
        return []
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        # We need at least one date to map to the SSR data
        return 1
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation."""
        if data is None or data.empty:
            self.logger.warning(f"Empty data provided for {asset_id}")
            return False
        
        # Check if we have SSR data handler
        if self.ssr_handler is None or self.ssr_handler.ssr_data is None:
            self.logger.warning("SSR data handler not properly initialized, cannot calculate signal.")
            return False
        
        # Check for minimum required samples
        if len(data) < self.get_min_required_samples():
            self.logger.warning(
                f"Insufficient data for {asset_id}: {len(data)} samples, "
                f"need at least {self.get_min_required_samples()}"
            )
            return False
        
        return True
    
    def get_ssr_value_for_date(self, date):
        """Get SSR value for a specific date using closest previous date if needed."""
        if self.ssr_handler is None or self.ssr_handler.ssr_data is None:
            return None
            
        # Convert to pandas datetime if needed
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
            
        # Find the closest date in the SSR data (same or earlier)
        closest_dates = self.ssr_handler.ssr_data.index[self.ssr_handler.ssr_data.index <= date]
        if len(closest_dates) == 0:
            return None
        
        closest_date = closest_dates[-1]  # Get the most recent date
        return self.ssr_handler.ssr_data.loc[closest_date, 'ssr_oscillator']


@register_signal
class SSR_RiskOn(SSRSignalBase):
    """Signal that indicates positive SSR (risk-on market conditions).
    
    Returns:
        1: When SSR is positive (> 0)
        0: When SSR is negative or not available
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                data_path (str): Path to the SSR CSV file
                handle_missing (str): How to handle missing data points: 'zero' or 'nan' (default: 'zero')
        """
        super().__init__(params)
        self.handle_missing = params.get('handle_missing', 'zero')
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data (used for dates).
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
                       1 when SSR is positive (risk-on)
                       0 when SSR is negative or not available
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        try:
            # Calculate SSR signal for each date in the DataFrame
            signals = pd.Series(index=data.index)
            
            # Get the range of dates in SSR data
            min_ssr_date = self.ssr_handler.ssr_data.index.min()
            max_ssr_date = self.ssr_handler.ssr_data.index.max()
            self.logger.info(f"Date range in SSR data: {min_ssr_date} to {max_ssr_date}")
            
            # Make sure we're working with datetime index or date column
            if not isinstance(data.index, pd.DatetimeIndex) and 'date' in data.columns:
                dates = pd.to_datetime(data['date'])
            else:
                dates = data.index
            
            for i, date in enumerate(dates):
                # Make sure date is a timestamp
                if not isinstance(date, pd.Timestamp):
                    date = pd.Timestamp(date)
                
                # Handle dates before SSR data starts
                if date < min_ssr_date:
                    signals.iloc[i] = np.nan if self.handle_missing == 'nan' else 0
                    continue
                
                # Get SSR value for this date
                ssr_value = self.get_ssr_value_for_date(date)
                
                # Set signal based on SSR value
                if ssr_value is None:
                    signals.iloc[i] = np.nan if self.handle_missing == 'nan' else 0
                else:
                    # 1 when SSR is positive, 0 otherwise
                    signals.iloc[i] = 1 if ssr_value > 0 else 0
            
            # Fill any remaining NaN values with 0
            if self.handle_missing == 'zero':
                signals = signals.fillna(0)
                
            return signals
            
        except Exception as e:
            self.logger.error(f"Error calculating SSR RiskOn signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


@register_signal
class SSR_RiskOff(SSRSignalBase):
    """Signal that indicates negative SSR (risk-off market conditions).
    
    Returns:
        1: When SSR is negative (< 0)
        0: When SSR is positive or not available
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                data_path (str): Path to the SSR CSV file
                handle_missing (str): How to handle missing data points: 'zero' or 'nan' (default: 'zero')
        """
        super().__init__(params)
        self.handle_missing = params.get('handle_missing', 'zero')
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data (used for dates).
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
                       1 when SSR is negative (risk-off)
                       0 when SSR is positive or not available
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        try:
            # Calculate SSR signal for each date in the DataFrame
            signals = pd.Series(index=data.index)
            
            # Get the range of dates in SSR data
            min_ssr_date = self.ssr_handler.ssr_data.index.min()
            max_ssr_date = self.ssr_handler.ssr_data.index.max()
            self.logger.info(f"Date range in SSR data: {min_ssr_date} to {max_ssr_date}")
            
            # Make sure we're working with datetime index or date column
            if not isinstance(data.index, pd.DatetimeIndex) and 'date' in data.columns:
                dates = pd.to_datetime(data['date'])
            else:
                dates = data.index
            
            for i, date in enumerate(dates):
                # Make sure date is a timestamp
                if not isinstance(date, pd.Timestamp):
                    date = pd.Timestamp(date)
                
                # Handle dates before SSR data starts
                if date < min_ssr_date:
                    signals.iloc[i] = np.nan if self.handle_missing == 'nan' else 0
                    continue
                
                # Get SSR value for this date
                ssr_value = self.get_ssr_value_for_date(date)
                
                # Set signal based on SSR value
                if ssr_value is None:
                    signals.iloc[i] = np.nan if self.handle_missing == 'nan' else 0
                else:
                    # 1 when SSR is negative, 0 otherwise
                    signals.iloc[i] = 1 if ssr_value < 0 else 0
            
            # Fill any remaining NaN values with 0
            if self.handle_missing == 'zero':
                signals = signals.fillna(0)
                
            return signals
            
        except Exception as e:
            self.logger.error(f"Error calculating SSR RiskOff signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


