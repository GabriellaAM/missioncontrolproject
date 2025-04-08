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
        
        # If no path in params, try environment variable
        if not ssr_path:
            ssr_path = os.environ.get('SSR_DATA_PATH')
            
        # If still no path, use default location
        if not ssr_path:
            ssr_path = 'data/onchainData/BTC_SSR.csv'
            
        # Check if file exists
        if not os.path.exists(ssr_path):
            self.logger.warning(f"SSR data file not found at {ssr_path}")
            self.ssr_data_loaded = False
            return False
            
        try:
            # Load data
            self.ssr_data = pd.read_csv(ssr_path)
            
            # Ensure date column is properly formatted
            if 'date' in self.ssr_data.columns:
                self.ssr_data['date'] = pd.to_datetime(self.ssr_data['date'])
                self.ssr_data.set_index('date', inplace=True)
            elif 'timestamp' in self.ssr_data.columns:
                self.ssr_data['timestamp'] = pd.to_datetime(self.ssr_data['timestamp'])
                self.ssr_data.set_index('timestamp', inplace=True)
                
            # Set flag to indicate data was loaded
            self.ssr_data_loaded = True
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
            
        # Only apply SSR signals to Bitcoin
        if asset_id.lower() != 'bitcoin':
            self.logger.debug(f"SSR signals only apply to Bitcoin, not {asset_id}")
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
            
        # Merge price data with SSR data
        merged_data = pd.merge(
            data, self.ssr_data, left_index=True, right_index=True, how='left'
        )
        
        # Fill missing values using forward fill
        if 'ssr' in merged_data.columns:
            merged_data['ssr'] = merged_data['ssr'].ffill()
        else:
            self.logger.warning("SSR column not found in data")
            return pd.Series(index=data.index)
            
        # Create signal series (to be implemented by subclasses)
        return self._create_signal(merged_data)
    
    def _create_signal(self, data: pd.DataFrame) -> pd.Series:
        """Create the signal based on SSR values.
        
        Args:
            data: DataFrame with price and SSR data
            
        Returns:
            Series with signal values
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement _create_signal")


@register_signal
class SSR_RiskOn(SSRSignalBase):
    """Signal indicating risk-on sentiment based on SSR decreasing."""
    
    def _create_signal(self, data: pd.DataFrame) -> pd.Series:
        """Create risk-on signal based on SSR values.
        
        Risk-on is indicated by decreasing SSR (stablecoins being deployed into crypto).
        
        Args:
            data: DataFrame with price and SSR data
            
        Returns:
            Series with signal values
        """
        if 'ssr' not in data.columns:
            return pd.Series(index=data.index)
            
        # Calculate SSR change
        data['ssr_change'] = data['ssr'].pct_change(periods=7)
        
        # Generate signal - risk-on when SSR is decreasing
        signal = ((data['ssr_change'] < -0.05) & (data['ssr'] < data['ssr'].rolling(30).mean())).astype(int)
        
        return signal


@register_signal
class SSR_RiskOff(SSRSignalBase):
    """Signal indicating risk-off sentiment based on SSR increasing."""
    
    def _create_signal(self, data: pd.DataFrame) -> pd.Series:
        """Create risk-off signal based on SSR values.
        
        Risk-off is indicated by increasing SSR (stablecoins accumulating).
        
        Args:
            data: DataFrame with price and SSR data
            
        Returns:
            Series with signal values
        """
        if 'ssr' not in data.columns:
            return pd.Series(index=data.index)
            
        # Calculate SSR change
        data['ssr_change'] = data['ssr'].pct_change(periods=7)
        
        # Generate signal - risk-off when SSR is increasing
        signal = ((data['ssr_change'] > 0.05) & (data['ssr'] > data['ssr'].rolling(30).mean())).astype(int)
        
        return signal


