"""
RSI-based signals for the Soros System.

These signals are based on the Relative Strength Index (RSI) indicator.
Each signal returns 1 if the specified condition is detected, 0 otherwise.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple
import ta

from .signal_base import SignalBase
from .signal_registry import register_signal
from ..indicators.rsi import RSICalculator


# Base class for RSI signals to reduce code duplication
class RSISignalBase(SignalBase):
    """Base class for RSI signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = self.params.get('quote_type', 'USD')
        self.rsi_calculator = RSICalculator()
        # Track if we've already warned about missing column
        self._warned_missing_column = False
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        if self.quote_type == 'USD':
            return ['close']
        else:  # BTC quote type
            # Skip BTC signals for bitcoin itself
            asset_id = self._get_asset_id()
            if asset_id == 'bitcoin':
                return ['close']
            # For other assets, require the asset's BTC price column
            # Only return a valid column name if we have an asset_id
            return [f'{asset_id}_btc'] if asset_id else ['close']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return self.rsi_length + 10  # Need RSI length plus some extra samples
    
    def _get_asset_id(self) -> str:
        """Extract asset_id from parameters."""
        return self.params.get('asset_id', '')
    
    def get_price_column(self, data: pd.DataFrame) -> Optional[str]:
        """Get the appropriate price column based on quote type."""
        asset_id = self._get_asset_id()
        
        if self.quote_type == 'USD':
            if 'close' in data.columns:
                return 'close'
            else:
                if not self._warned_missing_column:
                    self.logger.warning(f"Missing 'close' column for {asset_id}")
                    self._warned_missing_column = True
                return None
        
        # For BTC quotes
        if asset_id == 'bitcoin':
            return 'close'  # Bitcoin uses its own close price
        
        # For BTC quotes of other assets, use asset_id_btc column
        btc_col = f'{asset_id}_btc'
        if btc_col in data.columns:
            return btc_col
        
        # If no BTC column found, log warning and return None
        if not self._warned_missing_column:
            self.logger.warning(f"Missing required BTC column '{btc_col}' for {asset_id}")
            self._warned_missing_column = True
        return None
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation."""
        if not super().validate(data, asset_id):
            return False
            
        # Store asset_id in params if not already there
        if not self.params.get('asset_id'):
            self.params['asset_id'] = asset_id
            
        # Skip BTC signals for bitcoin itself
        if self.quote_type == 'BTC' and asset_id == 'bitcoin':
            return False
        
        # Get appropriate price column
        price_col = self.get_price_column(data)
        if price_col is None:
            return False
            
        return True


# Base class for RSI Oversold signals
class RSIOversoldBase(RSISignalBase):
    """Base class for RSI oversold signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.rsi_length = self.params.get('rsi_length', 14)
        self.window = self.rsi_length  # Use rsi_length for window size
        self.oversold_threshold = self.params.get('oversold_threshold', 30)
        self._warned_insufficient = False  # Track if we've warned about insufficient samples
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for RSI calculation."""
        return self.rsi_length + 1  # Need at least rsi_length + 1 samples for valid RSI
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate RSI oversold signal."""
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Get the appropriate price column
        price_col = self.get_price_column(data)
        if price_col is None:
            return pd.Series(index=data.index)
            
        try:
            # Calculate RSI
            rsi = ta.momentum.RSIIndicator(
                close=data[price_col],
                window=self.rsi_length
            ).rsi()
            
            # Generate signal
            signal = pd.Series(0, index=data.index)
            signal.loc[rsi < self.oversold_threshold] = 1
            
            # Log warning if no signals found
            if not signal.any() and not self._warned_insufficient:
                self.logger.warning(
                    f"No oversold signals found for {asset_id} "
                    f"(RSI never below {self.oversold_threshold})"
                )
                self._warned_insufficient = True
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI oversold signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


# Base class for RSI Overbought signals
class RSIOverboughtBase(RSISignalBase):
    """Base class for RSI overbought signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.rsi_length = self.params.get('rsi_length', 14)
        self.window = self.rsi_length  # Use rsi_length for window size
        self.overbought_threshold = self.params.get('overbought_threshold', 70)
        self._warned_insufficient = False  # Track if we've warned about insufficient samples
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for RSI calculation."""
        return self.rsi_length + 1  # Need at least rsi_length + 1 samples for valid RSI
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate RSI overbought signal."""
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Get the appropriate price column
        price_col = self.get_price_column(data)
        if price_col is None:
            return pd.Series(index=data.index)
            
        try:
            # Calculate RSI
            rsi = ta.momentum.RSIIndicator(
                close=data[price_col],
                window=self.rsi_length
            ).rsi()
            
            # Generate signal
            signal = pd.Series(0, index=data.index)
            signal.loc[rsi > self.overbought_threshold] = 1
            
            # Log warning if no signals found
            if not signal.any() and not self._warned_insufficient:
                self.logger.warning(
                    f"No overbought signals found for {asset_id} "
                    f"(RSI never above {self.overbought_threshold})"
                )
                self._warned_insufficient = True
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI overbought signal for {asset_id}: {str(e)}")
            return pd.Series(index=data.index)


# Base class for RSI Bullish signals (RSI > 50 and rising)
class RSIBullishBase(RSISignalBase):
    """Base class for RSI bullish condition signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.rsi_length = self.params.get('rsi_length', 28)  # Changed back to 28
        self.roc_length = self.params.get('roc_length', 28)  # Changed back to 28
        self._warned_insufficient = False
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        if self.quote_type == 'USD':
            return ['open', 'high', 'low', 'close']  # Need all OHLC columns
        else:  # BTC quote type
            # Skip BTC signals for bitcoin itself
            asset_id = self._get_asset_id()
            if asset_id == 'bitcoin':
                return ['open', 'high', 'low', 'close']
            # For other assets, we still need OHLC for the asset but BTC price for comparison
            return ['open', 'high', 'low', 'close', f'{asset_id}_btc'] if asset_id else ['open', 'high', 'low', 'close']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return max(self.rsi_length, self.roc_length) + 10
    
    def calculate_ohlc_rsi_roc(self, data: pd.DataFrame, price_suffix: str = '') -> Tuple[pd.Series, pd.Series]:
        """Calculate smoothed RSI and ROC from OHLC data.
        
        Args:
            data: DataFrame with OHLC data
            price_suffix: Suffix for column names (e.g., '_btc' for BTC quotes)
            
        Returns:
            Tuple of (smoothed_rsi, smoothed_roc)
        """
        import ta
        
        # Define column names
        open_col = f'open{price_suffix}'
        high_col = f'high{price_suffix}'
        low_col = f'low{price_suffix}'
        close_col = f'close{price_suffix}'
        
        # For BTC quotes of non-bitcoin assets, we use the asset's USD OHLC data
        # but calculate signals based on the BTC price column
        if price_suffix == '_btc':
            # Use the asset's OHLC data in USD terms
            open_col = 'open'
            high_col = 'high' 
            low_col = 'low'
            close_col = 'close'
        
        # Calculate RSI for each OHLC component
        rsi_open = ta.momentum.RSIIndicator(close=data[open_col], window=self.rsi_length).rsi()
        rsi_high = ta.momentum.RSIIndicator(close=data[high_col], window=self.rsi_length).rsi()
        rsi_low = ta.momentum.RSIIndicator(close=data[low_col], window=self.rsi_length).rsi()
        rsi_close = ta.momentum.RSIIndicator(close=data[close_col], window=self.rsi_length).rsi()
        
        # Calculate smoothed RSI (average of OHLC RSI values)
        rsi_smooth = (rsi_open + rsi_high + rsi_low + rsi_close) / 4
        
        # Calculate ROC for each OHLC component
        roc_open = ta.momentum.ROCIndicator(close=data[open_col], window=self.roc_length).roc()
        roc_high = ta.momentum.ROCIndicator(close=data[high_col], window=self.roc_length).roc()
        roc_low = ta.momentum.ROCIndicator(close=data[low_col], window=self.roc_length).roc()
        roc_close = ta.momentum.ROCIndicator(close=data[close_col], window=self.roc_length).roc()
        
        # Calculate smoothed ROC (average of OHLC ROC values)
        roc_smooth = (roc_open + roc_high + roc_low + roc_close) / 4
        
        return rsi_smooth, roc_smooth

    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values with state memory.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
                       1 when RSI > 50 AND RoC > 0 (bullish)
                       0 when RSI < 50 AND RoC < 0 (bearish)
                       Previous state otherwise (mixed conditions)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(0, index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        try:
            # Determine if we need BTC suffix
            price_suffix = ''
            if self.quote_type == 'BTC' and asset_id != 'bitcoin':
                price_suffix = '_btc'
                # Check if the BTC price column exists
                btc_col = f'{asset_id}_btc'
                if btc_col not in data.columns:
                    self.logger.warning(f"Missing required BTC column '{btc_col}' for {asset_id}")
                    return pd.Series(0, index=data.index)
            
            # Calculate smoothed RSI and ROC from OHLC data
            rsi_smooth, roc_smooth = self.calculate_ohlc_rsi_roc(data, price_suffix)
            
            # Initialize signal series and state tracking
            signal = pd.Series(0, index=data.index)
            last_active_mode = 0  # Track the last active signal state
            
            # Process each row with state memory
            for idx in data.index:
                if pd.isna(rsi_smooth.loc[idx]) or pd.isna(roc_smooth.loc[idx]):
                    signal.loc[idx] = last_active_mode
                    continue
                    
                # Bullish condition: RSI smooth > 50 AND smooth ROC > 0
                if rsi_smooth.loc[idx] > 50 and roc_smooth.loc[idx] > 0:
                    signal.loc[idx] = 1
                    last_active_mode = 1
                # Bearish condition: RSI smooth < 50 AND smooth ROC < 0  
                elif rsi_smooth.loc[idx] < 50 and roc_smooth.loc[idx] < 0:
                    signal.loc[idx] = 0
                    last_active_mode = 0
                # Mixed conditions: keep the last active mode
                else:
                    signal.loc[idx] = last_active_mode
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI Bullish signal for {asset_id}: {str(e)}")
            return pd.Series(0, index=data.index)


# Base class for RSI Bearish signals (RSI < 50 and falling)
class RSIBearishBase(RSISignalBase):
    """Base class for RSI bearish condition signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.rsi_length = self.params.get('rsi_length', 28)  # Changed back to 28
        self.roc_length = self.params.get('roc_length', 28)  # Changed back to 28
        self._warned_insufficient = False
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        if self.quote_type == 'USD':
            return ['open', 'high', 'low', 'close']  # Need all OHLC columns
        else:  # BTC quote type
            # Skip BTC signals for bitcoin itself
            asset_id = self._get_asset_id()
            if asset_id == 'bitcoin':
                return ['open', 'high', 'low', 'close']
            # For other assets, we still need OHLC for the asset but BTC price for comparison
            return ['open', 'high', 'low', 'close', f'{asset_id}_btc'] if asset_id else ['open', 'high', 'low', 'close']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return max(self.rsi_length, self.roc_length) + 10
    
    def calculate_ohlc_rsi_roc(self, data: pd.DataFrame, price_suffix: str = '') -> Tuple[pd.Series, pd.Series]:
        """Calculate smoothed RSI and ROC from OHLC data.
        
        Args:
            data: DataFrame with OHLC data
            price_suffix: Suffix for column names (e.g., '_btc' for BTC quotes)
            
        Returns:
            Tuple of (smoothed_rsi, smoothed_roc)
        """
        import ta
        
        # Define column names
        open_col = f'open{price_suffix}'
        high_col = f'high{price_suffix}'
        low_col = f'low{price_suffix}'
        close_col = f'close{price_suffix}'
        
        # For BTC quotes of non-bitcoin assets, we use the asset's USD OHLC data
        # but calculate signals based on the BTC price column
        if price_suffix == '_btc':
            # Use the asset's OHLC data in USD terms
            open_col = 'open'
            high_col = 'high' 
            low_col = 'low'
            close_col = 'close'
        
        # Calculate RSI for each OHLC component
        rsi_open = ta.momentum.RSIIndicator(close=data[open_col], window=self.rsi_length).rsi()
        rsi_high = ta.momentum.RSIIndicator(close=data[high_col], window=self.rsi_length).rsi()
        rsi_low = ta.momentum.RSIIndicator(close=data[low_col], window=self.rsi_length).rsi()
        rsi_close = ta.momentum.RSIIndicator(close=data[close_col], window=self.rsi_length).rsi()
        
        # Calculate smoothed RSI (average of OHLC RSI values)
        rsi_smooth = (rsi_open + rsi_high + rsi_low + rsi_close) / 4
        
        # Calculate ROC for each OHLC component
        roc_open = ta.momentum.ROCIndicator(close=data[open_col], window=self.roc_length).roc()
        roc_high = ta.momentum.ROCIndicator(close=data[high_col], window=self.roc_length).roc()
        roc_low = ta.momentum.ROCIndicator(close=data[low_col], window=self.roc_length).roc()
        roc_close = ta.momentum.ROCIndicator(close=data[close_col], window=self.roc_length).roc()
        
        # Calculate smoothed ROC (average of OHLC ROC values)
        roc_smooth = (roc_open + roc_high + roc_low + roc_close) / 4
        
        return rsi_smooth, roc_smooth

    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values with state memory.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
                       1 when RSI < 50 AND RoC < 0 (bearish)
                       0 when RSI > 50 AND RoC > 0 (bullish)
                       Previous state otherwise (mixed conditions)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(0, index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        try:
            # Determine if we need BTC suffix
            price_suffix = ''
            if self.quote_type == 'BTC' and asset_id != 'bitcoin':
                price_suffix = '_btc'
                # Check if the BTC price column exists
                btc_col = f'{asset_id}_btc'
                if btc_col not in data.columns:
                    self.logger.warning(f"Missing required BTC column '{btc_col}' for {asset_id}")
                    return pd.Series(0, index=data.index)
            
            # Calculate smoothed RSI and ROC from OHLC data
            rsi_smooth, roc_smooth = self.calculate_ohlc_rsi_roc(data, price_suffix)
            
            # Initialize signal series and state tracking
            signal = pd.Series(0, index=data.index)
            last_active_mode = 0  # Track the last active signal state
            
            # Process each row with state memory
            for idx in data.index:
                if pd.isna(rsi_smooth.loc[idx]) or pd.isna(roc_smooth.loc[idx]):
                    signal.loc[idx] = last_active_mode
                    continue
                    
                # Bearish condition: RSI smooth < 50 AND smooth ROC < 0
                if rsi_smooth.loc[idx] < 50 and roc_smooth.loc[idx] < 0:
                    signal.loc[idx] = 1
                    last_active_mode = 1
                # Bullish condition: RSI smooth > 50 AND smooth ROC > 0  
                elif rsi_smooth.loc[idx] > 50 and roc_smooth.loc[idx] > 0:
                    signal.loc[idx] = 0
                    last_active_mode = 0
                # Mixed conditions: keep the last active mode
                else:
                    signal.loc[idx] = last_active_mode
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI Bearish signal for {asset_id}: {str(e)}")
            return pd.Series(0, index=data.index)


# ========== USD QUOTE SIGNALS ==========

@register_signal
class RSI_Oversold_USD(RSIOversoldBase):
    """Signal for RSI oversold conditions in USD terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                rsi_length (int): Length of RSI calculation (default: 14)
                oversold_threshold (float): RSI level considered oversold (default: 30)
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
                overbought_threshold (float): RSI level considered overbought (default: 70)
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
                oversold_threshold (float): RSI level considered oversold (default: 30)
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
                overbought_threshold (float): RSI level considered overbought (default: 70)
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


class RSIEnsembleBase(RSISignalBase):
    """Base class for RSI ensemble signals that combine multiple window periods.
    
    This signal calculates RSI bullish conditions (RSI > 50 and rising) for multiple
    time windows and generates a signal when a threshold number of windows produce positive signals.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        # Standard window periods to use
        self.WINDOWS = [7, 14, 21, 28]
        # Threshold for how many windows must be positive to generate a signal
        self.signal_threshold = self.params.get('signal_threshold', 2)  # Default: at least 2 windows
        
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return max(self.WINDOWS) * 2  # Need enough data for the largest window
    
    def calculate_window_signal(self, data: pd.DataFrame, window: int, asset_id: str) -> pd.Series:
        """Calculate RSI bullish signal for a specific window period.
        
        Args:
            data: DataFrame with price data
            window: Window period for RSI calculation
            asset_id: ID of the asset
            
        Returns:
            Series with signal values (1 for bullish, 0 for not bullish)
        """
        # Get the appropriate price column
        price_col = self.get_price_column(data)
        if price_col is None:
            return pd.Series(0, index=data.index)
        
        try:
            # Calculate RSI and RoC for this window
            result_df = self.rsi_calculator.calculate_smooth_rsi(
                data, price_col, rsi_length=window, roc_length=window
            )
            
            # Get RSI and RoC columns
            rsi_col = f'RSI_{price_col}'
            roc_col = f'RoC_{price_col}'
            
            if rsi_col in result_df.columns and roc_col in result_df.columns:
                # Signal is bullish if RSI > 50 AND RoC > 0
                signal = pd.Series(0, index=data.index)  # Initialize with zeros using original index
                
                # Use loc to set values, avoiding reindexing issues
                mask = (result_df[rsi_col] > 50) & (result_df[roc_col] > 0) & result_df[rsi_col].notna() & result_df[roc_col].notna()
                if not mask.empty:
                    signal.loc[mask.index[mask]] = 1
                
                return signal
            else:
                self.logger.warning(f"Required columns {rsi_col} and/or {roc_col} not found for {asset_id}")
                return pd.Series(0, index=data.index)
        
        except Exception as e:
            self.logger.error(f"Error calculating RSI window signal for {asset_id} with window {window}: {str(e)}")
            return pd.Series(0, index=data.index)
            
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the ensemble signal by combining multiple window signals.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with ensemble signal values (1 for buy, 0 for no signal)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(0, index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        try:
            # Calculate signals for each window
            window_signals = {}
            for window in self.WINDOWS:
                window_signals[window] = self.calculate_window_signal(data, window, asset_id)
                
            # Combine signals into a DataFrame with the original data index
            signal_df = pd.DataFrame(index=data.index)
            
            # Ensure all window signals have the same index as the original data
            for window, signal in window_signals.items():
                if signal.index.equals(data.index):
                    signal_df[f'window_{window}'] = signal
                else:
                    # Safely reindex if needed (this should not happen with our improved implementation)
                    aligned_signal = pd.Series(0, index=data.index)
                    aligned_signal.loc[signal.index.intersection(data.index)] = signal.loc[signal.index.intersection(data.index)]
                    signal_df[f'window_{window}'] = aligned_signal
            
            # Count active signals for each date
            signal_df['active_count'] = signal_df.sum(axis=1)
            
            # Generate final signal if enough window signals are positive
            signal_df['final_signal'] = (signal_df['active_count'] >= self.signal_threshold).astype(int)
            
            # Handle initialization period
            # Set initial signals to 0 to avoid false positives at the start
            init_period = max(self.WINDOWS) + 5
            if len(signal_df) > init_period:
                signal_df['final_signal'].iloc[:init_period] = 0
            
            self.logger.info(f"RSI Ensemble for {asset_id}: Using {len(self.WINDOWS)} windows with threshold {self.signal_threshold}")
            
            # Ensure we return a Series with the same index as the input data
            return signal_df['final_signal']
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI ensemble signal for {asset_id}: {str(e)}")
            return pd.Series(0, index=data.index)


@register_signal
class RSI_Ensemble_Signal_USD(RSIEnsembleBase):
    """Ensemble signal combining multiple RSI window signals in USD terms.
    
    This signal calculates RSI bullish conditions (RSI > 50 and rising) for multiple
    time windows (7, 14, 21, 28 days) and generates a signal when a threshold 
    number of windows produce positive signals.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'USD'


@register_signal
class RSI_Ensemble_Signal_BTC(RSIEnsembleBase):
    """Ensemble signal combining multiple RSI window signals in BTC terms.
    
    This signal calculates RSI bullish conditions (RSI > 50 and rising) for multiple
    time windows (7, 14, 21, 28 days) and generates a signal when a threshold 
    number of windows produce positive signals.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'BTC'