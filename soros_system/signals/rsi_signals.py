"""
RSI-based signals for the Soros System.

These signals are based on the Relative Strength Index (RSI) indicator.
Each signal returns 1 if the specified condition is detected, 0 otherwise.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple

from .signal_base import SignalBase
from .signal_registry import register_signal
from ..indicators.rsi import RSICalculator


# Base class for RSI signals to reduce code duplication
class RSISignalBase(SignalBase):
    """Base class for all RSI signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'USD'  # Will be overridden by subclasses
        self.rsi_length = self.params.get('rsi_length', 14)
        self.rsi_calculator = RSICalculator()
        # Store a reference to data to help with asset_id lookup
        self.data = None
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        if self.quote_type == 'USD':
            return ['close']
        else:
            # Store asset_id in params when available to ensure this works correctly
            asset_id = self._get_asset_id()
            # If we don't have an asset_id yet, return a generic column
            if not asset_id:
                return ['_btc']
            return [f'{asset_id}_btc']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return self.rsi_length + 10  # Need RSI length plus some extra samples
    
    def _get_asset_id(self) -> str:
        """Extract asset_id from parameters."""
        return self.params.get('asset_id', '')
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation."""
        # Store data reference for possible asset_id lookup
        self.data = data
        
        # Store asset_id in params if not already there
        if not self.params.get('asset_id'):
            self.params['asset_id'] = asset_id
            
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
        
        # For BTC quote type, check if the expected column exists
        if self.quote_type == 'BTC':
            # Try with asset_id prefix
            btc_col = f'{asset_id}_btc'
            if btc_col in data.columns:
                return True
                
            # Try finding any column ending with _btc (for flexibility)
            btc_cols = [col for col in data.columns if col.endswith('_btc') and col not in 
                        [c for c in data.columns if c.endswith(('_btc_open', '_btc_high', '_btc_low'))]]
            
            if btc_cols:
                # Update our internal reference to use this column
                self.params['_btc_column'] = btc_cols[0]
                return True
                
            # If no suitable column found, log and return False
            self.logger.warning(f"Missing required BTC column for {asset_id}")
            self.logger.debug(f"Available columns: {data.columns.tolist()}")
            return False
        
        # For USD quote type, require 'close' column
        elif 'close' not in data.columns:
            self.logger.warning(f"Missing 'close' column for {asset_id}")
            return False
            
        return True


# Base class for RSI Oversold signals
class RSIOversoldBase(RSISignalBase):
    """Base class for RSI oversold condition signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.oversold = self.params.get('oversold', 30)
        self.window = self.params.get('rsi_length', 14)
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
                       1 when RSI is below oversold threshold (buy signal)
                       0 when RSI is above oversold threshold (no signal)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Determine price column based on quote type
        price_col = 'close' if self.quote_type == 'USD' else f'{asset_id}_btc'
        
        try:
            # Use the simpler RSI calculation directly on the price column
            if price_col not in data.columns:
                self.logger.warning(f"Price column {price_col} not found for {asset_id}")
                return pd.Series(index=data.index)
                
            # Ensure data is numeric
            prices = pd.to_numeric(data[price_col], errors='coerce')
            
            # Calculate RSI directly
            rsi_values = self.rsi_calculator.calculate_rsi(prices, window=self.window)
            
            # Create signal based on oversold threshold (RSI < 30)
            signal = pd.Series(0, index=data.index)
            signal.loc[rsi_values < self.oversold] = 1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI Oversold signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


# Base class for RSI Overbought signals
class RSIOverboughtBase(RSISignalBase):
    """Base class for RSI overbought condition signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.overbought = self.params.get('overbought', 70)
        self.window = self.params.get('rsi_length', 14)
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
                       1 when RSI is above overbought threshold (sell signal)
                       0 when RSI is below overbought threshold (no signal)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Determine price column based on quote type
        price_col = 'close' if self.quote_type == 'USD' else f'{asset_id}_btc'
        
        try:
            # Use the simpler RSI calculation directly on the price column
            if price_col not in data.columns:
                self.logger.warning(f"Price column {price_col} not found for {asset_id}")
                return pd.Series(index=data.index)
                
            # Ensure data is numeric
            prices = pd.to_numeric(data[price_col], errors='coerce')
            
            # Calculate RSI directly
            rsi_values = self.rsi_calculator.calculate_rsi(prices, window=self.window)
            
            # Create signal based on overbought threshold (RSI > 70)
            signal = pd.Series(0, index=data.index)
            signal.loc[rsi_values > self.overbought] = 1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI Overbought signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


# Base class for RSI Bullish signals (RSI > 50 and rising)
class RSIBullishBase(RSISignalBase):
    """Base class for RSI bullish condition signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.rsi_length = self.params.get('rsi_length', 28)
        self.roc_length = self.params.get('roc_length', 28)
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return max(self.rsi_length, self.roc_length) + 10
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
                       1 when RSI > 50 AND RoC > 0 (bullish)
                       0 otherwise (not bullish)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Determine price column based on quote type
        price_col = 'close' if self.quote_type == 'USD' else f'{asset_id}_btc'
        
        try:
            # Use the existing smooth RSI calculation which also calculates RoC
            result_df = self.rsi_calculator.calculate_smooth_rsi(
                data, price_col, rsi_length=self.rsi_length, roc_length=self.roc_length
            )
            
            # Get RSI Signal column (already defined in calculator)
            signal_col = f'RSI_Signal_{price_col}'
            
            if signal_col in result_df.columns:
                # The RSI calculator already creates a combined RSI+RoC signal
                # Convert to our binary signal format (1/0)
                signal = result_df[signal_col].apply(
                    lambda x: 1 if pd.notna(x) and x > 0 else 0
                )
                return signal
            else:
                # If the signal column doesn't exist, calculate manually
                rsi_col = f'RSI_{price_col}'
                roc_col = f'RoC_{price_col}'
                
                if rsi_col in result_df.columns and roc_col in result_df.columns:
                    # Both RSI and RoC are available, create combined signal
                    # Signal is bullish if RSI > 50 AND RoC > 0
                    signal = result_df.apply(
                        lambda row: 1 if pd.notna(row[rsi_col]) and pd.notna(row[roc_col]) and 
                                        row[rsi_col] > 50 and row[roc_col] > 0 else 0,
                        axis=1
                    )
                    return signal
                else:
                    self.logger.warning(f"Required columns {rsi_col} and/or {roc_col} not found for {asset_id}")
                    return pd.Series(index=data.index)
        
        except Exception as e:
            self.logger.error(f"Error calculating RSI Bullish signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


# Base class for RSI Bearish signals (RSI < 50 and falling)
class RSIBearishBase(RSISignalBase):
    """Base class for RSI bearish condition signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.rsi_length = self.params.get('rsi_length', 28)
        self.roc_length = self.params.get('roc_length', 28)
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return max(self.rsi_length, self.roc_length) + 10
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
                       1 when RSI < 50 AND RoC < 0 (bearish)
                       0 otherwise (not bearish)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Determine price column based on quote type
        price_col = 'close' if self.quote_type == 'USD' else f'{asset_id}_btc'
        
        try:
            # Use the existing smooth RSI calculation which also calculates RoC
            result_df = self.rsi_calculator.calculate_smooth_rsi(
                data, price_col, rsi_length=self.rsi_length, roc_length=self.roc_length
            )
            
            # Get RSI Signal column (already defined in calculator)
            signal_col = f'RSI_Signal_{price_col}'
            
            if signal_col in result_df.columns:
                # The RSI calculator already creates a combined RSI+RoC signal
                # Convert to our binary signal format (1/0)
                signal = result_df[signal_col].apply(
                    lambda x: 1 if pd.notna(x) and x < 0 else 0
                )
                return signal
            else:
                # If the signal column doesn't exist, calculate manually
                rsi_col = f'RSI_{price_col}'
                roc_col = f'RoC_{price_col}'
                
                if rsi_col in result_df.columns and roc_col in result_df.columns:
                    # Both RSI and RoC are available, create combined signal
                    # Signal is bearish if RSI < 50 AND RoC < 0
                    signal = result_df.apply(
                        lambda row: 1 if pd.notna(row[rsi_col]) and pd.notna(row[roc_col]) and 
                                        row[rsi_col] < 50 and row[roc_col] < 0 else 0,
                        axis=1
                    )
                    return signal
                else:
                    self.logger.warning(f"Required columns {rsi_col} and/or {roc_col} not found for {asset_id}")
                    return pd.Series(index=data.index)
        
        except Exception as e:
            self.logger.error(f"Error calculating RSI Bearish signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


# ========== USD QUOTE SIGNALS ==========

@register_signal
class RSI_Oversold_USD(RSIOversoldBase):
    """Signal for RSI oversold conditions in USD terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                rsi_length (int): Length of RSI calculation (default: 14)
                oversold (float): RSI level considered oversold (default: 30)
        """
        super().__init__(params)
        self.quote_type = 'USD'


@register_signal
class RSI_Overbought_USD(RSIOverboughtBase):
    """Signal for RSI overbought conditions in USD terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                rsi_length (int): Length of RSI calculation (default: 14)
                overbought (float): RSI level considered overbought (default: 70)
        """
        super().__init__(params)
        self.quote_type = 'USD'


@register_signal
class RSI_Bullish_USD(RSIBullishBase):
    """Signal for RSI bullish conditions in USD terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                rsi_length (int): Length of RSI calculation (default: 28)
                roc_length (int): Length of RoC calculation (default: 28)
        """
        super().__init__(params)
        self.quote_type = 'USD'


@register_signal
class RSI_Bearish_USD(RSIBearishBase):
    """Signal for RSI bearish conditions in USD terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                rsi_length (int): Length of RSI calculation (default: 28)
                roc_length (int): Length of RoC calculation (default: 28)
        """
        super().__init__(params)
        self.quote_type = 'USD'


# ========== BTC QUOTE SIGNALS ==========

@register_signal
class RSI_Oversold_BTC(RSIOversoldBase):
    """Signal for RSI oversold conditions in BTC terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                rsi_length (int): Length of RSI calculation (default: 14)
                oversold (float): RSI level considered oversold (default: 30)
        """
        super().__init__(params)
        self.quote_type = 'BTC'


@register_signal
class RSI_Overbought_BTC(RSIOverboughtBase):
    """Signal for RSI overbought conditions in BTC terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                rsi_length (int): Length of RSI calculation (default: 14)
                overbought (float): RSI level considered overbought (default: 70)
        """
        super().__init__(params)
        self.quote_type = 'BTC'


@register_signal
class RSI_Bullish_BTC(RSIBullishBase):
    """Signal for RSI bullish conditions in BTC terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                rsi_length (int): Length of RSI calculation (default: 28)
                roc_length (int): Length of RoC calculation (default: 28)
        """
        super().__init__(params)
        self.quote_type = 'BTC'


@register_signal
class RSI_Bearish_BTC(RSIBearishBase):
    """Signal for RSI bearish conditions in BTC terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                rsi_length (int): Length of RSI calculation (default: 28)
                roc_length (int): Length of RoC calculation (default: 28)
        """
        super().__init__(params)
        self.quote_type = 'BTC'