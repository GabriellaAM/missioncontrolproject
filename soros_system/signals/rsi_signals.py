"""
RSI-based signals for the Soros System.

These signals are based on the Relative Strength Index (RSI) indicator.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from .signal_base import SignalBase
from .signal_registry import register_signal
from ..indicators.rsi import RSICalculator


# Base class for RSI signals to reduce code duplication
class RSISignalBase(SignalBase):
    """Base class for all RSI signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = self.params.get('quote_type', 'USD')
        self.rsi_length = self.params.get('rsi_length', 14)
        self.rsi_calculator = RSICalculator()
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        return ['close'] if self.quote_type == 'USD' else [f'{self._get_asset_id()}_btc']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return self.rsi_length + 10  # Need RSI length plus some extra samples
    
    def _get_asset_id(self) -> str:
        """Extract asset_id from parameters."""
        return self.params.get('asset_id', '')
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation."""
        if data is None or data.empty:
            self.logger.warning(f"Empty data provided for {asset_id}")
            return False
        
        # Check for minimum required samples
        if len(data) < self.get_min_required_samples():
            self.logger.warning(
                f"Insufficient data for {asset_id}: {len(data)} samples, "
                f"need at least {self.get_min_required_samples()}"
            )
            return False
        
        # Check for required columns
        required_cols = self.get_required_columns()
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            self.logger.warning(f"Missing required columns for {asset_id}: {missing_cols}")
            return False
        
        return True


@register_signal
class RSI_Bottom(RSISignalBase):
    """Signal for RSI oversold conditions."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                quote_type (str): 'USD' or 'BTC' (default: 'USD')
                rsi_length (int): Length of RSI calculation (default: 14)
                oversold (float): RSI level considered oversold (default: 30)
        """
        super().__init__(params)
        self.oversold = self.params.get('oversold', 30)
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (+1 or -1) indexed by date.
                       +1 when RSI is below oversold threshold (buy signal)
                       -1 when RSI is above oversold threshold (no signal)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Determine price column based on quote type
        price_col = 'close' if self.quote_type == 'USD' else f'{asset_id}_btc'
        
        try:
            # Calculate RSI using the RSI calculator
            rsi_data = self.rsi_calculator.calculate_smooth_rsi(
                data, price_col, rsi_length=self.rsi_length, roc_length=self.rsi_length
            )
            
            # Get the RSI column name
            rsi_col = f'RSI_{price_col}'
            
            if rsi_col not in rsi_data.columns:
                self.logger.warning(f"RSI column {rsi_col} not found for {asset_id}")
                return pd.Series(index=data.index)
            
            # Create signal based on oversold threshold
            # +1 when RSI is below the oversold threshold (buy signal)
            # -1 otherwise (no signal)
            signal = rsi_data[rsi_col].apply(
                lambda x: 1 if pd.notna(x) and x < self.oversold else -1
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI Bottom signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


@register_signal
class RSI_Top(RSISignalBase):
    """Signal for RSI overbought conditions."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                quote_type (str): 'USD' or 'BTC' (default: 'USD')
                rsi_length (int): Length of RSI calculation (default: 14)
                overbought (float): RSI level considered overbought (default: 70)
        """
        super().__init__(params)
        self.overbought = self.params.get('overbought', 70)
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (+1 or -1) indexed by date.
                       +1 when RSI is below overbought threshold (no signal)
                       -1 when RSI is above overbought threshold (sell signal)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Determine price column based on quote type
        price_col = 'close' if self.quote_type == 'USD' else f'{asset_id}_btc'
        
        try:
            # Calculate RSI using the RSI calculator
            rsi_data = self.rsi_calculator.calculate_smooth_rsi(
                data, price_col, rsi_length=self.rsi_length, roc_length=self.rsi_length
            )
            
            # Get the RSI column name
            rsi_col = f'RSI_{price_col}'
            
            if rsi_col not in rsi_data.columns:
                self.logger.warning(f"RSI column {rsi_col} not found for {asset_id}")
                return pd.Series(index=data.index)
            
            # Create signal based on overbought threshold
            # -1 when RSI is above the overbought threshold (sell signal)
            # +1 otherwise (no signal)
            signal = rsi_data[rsi_col].apply(
                lambda x: -1 if pd.notna(x) and x > self.overbought else 1
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI Top signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index) 