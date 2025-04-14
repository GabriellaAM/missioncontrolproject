"""
Donchian Channel-based signals for the Soros System.

These signals are based on the Donchian Channel indicator and trailing stops.
Each signal returns 1 if the specified condition is detected, 0 otherwise.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable, Dict
from functools import lru_cache

from .signal_base import SignalBase
from .signal_registry import register_signal


# Base class for Donchian signals to reduce code duplication
class DonchianSignalBase(SignalBase):
    """Base class for Donchian Channel signals."""
    
    # Available window lengths for Donchian Channels
    WINDOWS = [5, 10, 20, 30, 60, 90, 150, 250, 360]
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = self.params.get('quote_type', 'USD')
        self.window = self.params.get('window', 20)  # Default to 20-day window
        
        # Ensure window is one of the allowed values
        if self.window not in self.WINDOWS:
            self.logger.warning(
                f"Window {self.window} not in allowed windows {self.WINDOWS}. "
                f"Using default window of 20."
            )
            self.window = 20
            
        # Track if we've already warned about missing column
        self._warned_missing_column = False
        
        # Cache for Donchian channels to avoid recalculation
        self._donchian_cache = {}
    
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
            # Only return valid column name if we have an asset_id
            if asset_id:
                # The required column is asset_id_btc, not asset_id_btc_close
                return [f'{asset_id}_btc']
            return ['close']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return max(self.WINDOWS) + 10  # Need max window length plus some extra samples
    
    def _get_asset_id(self) -> str:
        """Extract asset_id from parameters."""
        return self.params.get('asset_id', '')
    
    def get_price_columns(self, data: pd.DataFrame) -> Optional[Dict[str, str]]:
        """Get the appropriate price columns based on quote type."""
        asset_id = self._get_asset_id()
        
        if self.quote_type == 'USD':
            if 'close' in data.columns:
                return {'close': 'close'}
            else:
                if not self._warned_missing_column:
                    self.logger.warning(f"Missing close column for {asset_id}")
                    self._warned_missing_column = True
                return None
        
        # For BTC quotes
        if asset_id == 'bitcoin':
            return {'close': 'close'}  # Bitcoin uses its own price
        
        # For BTC quotes of other assets, use asset_id_btc column (not asset_id_btc_close)
        btc_col = f'{asset_id}_btc'
        
        if btc_col in data.columns:
            return {'close': btc_col}
        
        # If no BTC column found, log warning and return None
        if not self._warned_missing_column:
            self.logger.warning(f"Missing required BTC column {btc_col} for {asset_id}")
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
        
        # Get appropriate price columns
        price_cols = self.get_price_columns(data)
        if price_cols is None:
            return False
            
        return True
    
    def calculate_donchian_channels(self, data: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        """Calculate Donchian Channels for the given data.
        
        Args:
            data: DataFrame with price data
            window: Window length for Donchian Channel calculation
            
        Returns:
            Dictionary with Donchian Channels (high, low, mid) series
        """
        # Check cache first
        cache_key = (id(data), window)
        if cache_key in self._donchian_cache:
            return self._donchian_cache[cache_key]
        
        # Get appropriate price columns
        price_cols = self.get_price_columns(data)
        if price_cols is None:
            # Return empty dict if no price columns
            return {}
            
        close_col = price_cols['close']
        
        # Calculate Donchian Channels using close prices - without copying the entire dataframe
        # High line: Highest close in last X days
        up = data[close_col].rolling(window=window).max()
        
        # Low line: Lowest close in last X days
        down = data[close_col].rolling(window=window).min()
        
        # Middle line: (High line + Low line) / 2
        mid = (up + down) / 2
        
        # Store results in a dictionary
        result = {
            f'donchian_up_{window}': up,
            f'donchian_down_{window}': down,
            f'donchian_mid_{window}': mid
        }
        
        # Cache the result
        self._donchian_cache[cache_key] = result
        
        return result
    
    def calculate_trailing_stop(self, data: pd.DataFrame, signal: pd.Series, window: int) -> pd.Series:
        """Calculate trailing stop based on Donchian middle line.
        
        Args:
            data: DataFrame with price data and Donchian Channels
            signal: Series with signal values (1 or 0)
            window: Window length for Donchian Channel calculation
            
        Returns:
            Series with trailing stop values
        """
        # Get Donchian channels
        donchian = self.calculate_donchian_channels(data, window)
        if not donchian:
            return pd.Series(np.nan, index=data.index)
        
        # Get the Donchian middle line
        mid_series = donchian[f'donchian_mid_{window}']
        
        # Initialize trailing stop with NaN
        trailing_stop = pd.Series(np.nan, index=data.index)
        
        # Optimize the trailing stop calculation using vectorization where possible
        # We still need some iteration for the trailing stop, but we can optimize it
        
        # Pre-allocate an array for the trailing stop values
        stop_values = np.full(len(data), np.nan)
        
        # Find positions where signal is 1 (in a trade)
        in_trade_idx = np.where(signal == 1)[0]
        
        # Process each trade period more efficiently
        for i in in_trade_idx:
            if i == 0:  # Skip the first day
                continue
                
            # If this is the first day of a new trade
            if signal.iloc[i-1] == 0:
                stop_values[i] = mid_series.iloc[i]
            else:
                # Continue existing trade - take max of previous stop and current mid
                prev_stop = stop_values[i-1]
                if not np.isnan(prev_stop):
                    stop_values[i] = max(prev_stop, mid_series.iloc[i])
                else:
                    stop_values[i] = mid_series.iloc[i]
        
        # Convert back to pandas Series
        trailing_stop = pd.Series(stop_values, index=data.index)
        
        return trailing_stop


class DonchianBreakoutSignal(DonchianSignalBase):
    """Signal for Donchian Channel breakout strategy."""
    
    def calculate_window_signal(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate the signal values for a specific window.
        
        Implements the Donchian Channel breakout strategy:
        1. Entry: When price breaks above the highest high of the previous N days
        2. Exit: When price closes below the trailing stop
        3. Trailing stop: Updated daily as max(previous stop, current mid-line)
        4. Initial trailing stop: Set to midpoint of channel at entry
           
        Args:
            data: DataFrame with price data
            window: Window length for Donchian Channel calculation
            
        Returns:
            Series with signal values (1 for buy, 0 for no position)
        """
        # Get appropriate price columns
        price_cols = self.get_price_columns(data)
        if price_cols is None:
            return pd.Series(0, index=data.index)
        
        close_col = price_cols['close']
        
        # Initialize signal and trailing stop series
        signal = pd.Series(0, index=data.index)
        trailing_stop = pd.Series(np.nan, index=data.index)
        
        # Skip the first window periods where we don't have enough data
        for i in range(window, len(data)):
            # Calculate Donchian channels using ONLY past data
            past_n_days = data[close_col].iloc[i-window:i]  # Exclude today
            upper_band = past_n_days.max()
            lower_band = past_n_days.min()
            middle_band = (upper_band + lower_band) / 2
            
            # Get today's close and yesterday's values
            today_close = data[close_col].iloc[i]
            yesterdays_position = signal.iloc[i-1]
            yesterdays_stop = trailing_stop.iloc[i-1]
            
            # Apply trading rules:
            # Entry: Close breaks above upper band (buy)
            if yesterdays_position == 0 and today_close > upper_band:
                signal.iloc[i] = 1
                trailing_stop.iloc[i] = middle_band
                
            # Exit: Close falls below trailing stop (sell)
            elif yesterdays_position == 1 and today_close <= yesterdays_stop:
                signal.iloc[i] = 0
                trailing_stop.iloc[i] = np.nan
                
            # Hold: Update trailing stop if in position
            elif yesterdays_position == 1:
                signal.iloc[i] = 1
                # Only raise the stop, never lower it
                trailing_stop.iloc[i] = max(yesterdays_stop, middle_band) if not np.isnan(yesterdays_stop) else middle_band
                
            # No position, no stop
            else:
                signal.iloc[i] = 0
                trailing_stop.iloc[i] = np.nan
        
        return signal
        
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values using an ensemble of window signals.
        
        This calculates signals for all window lengths and combines them into a single signal.
        If threshold or more of the window signals are 1, the ensemble signal is 1, otherwise 0.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with ensemble signal values (1 for buy, 0 for no position)
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(0, index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Get signal threshold from params, default to 3 (at least 3 window signals must be active)
        signal_threshold = self.params.get('signal_threshold', 3)
        
        try:
            # Use all available window lengths from the base class
            windows = self.WINDOWS  # [5, 10, 20, 30, 60, 90, 150, 250, 360]
            
            # Calculate signals for window lengths
            window_signals = {}
            for window in windows:
                window_signals[window] = self.calculate_window_signal(data, window)
                
                # Ensure first window*2 days have zero signal (initialization period)
                window_signals[window].iloc[:window*2] = 0
            
            # Combine signals
            signal_df = pd.DataFrame(index=data.index)
            for window, signal in window_signals.items():
                signal_df[f'window_{window}'] = signal
            
            # Count active signals and check against threshold
            signal_df['active_count'] = signal_df.sum(axis=1)
            signal_df['final_signal'] = (signal_df['active_count'] >= signal_threshold).astype(int)
            
            # Ensure first max_window*2 days have zero signal - initialization period
            max_window = max(windows)
            signal_df['final_signal'].iloc[:max_window*2] = 0
            
            # Log final ensemble signal statistics
            total_days = len(signal_df)
            signal_days = signal_df['final_signal'].sum()
            signal_pct = (signal_days / total_days) * 100 if total_days > 0 else 0
            
            # Log window usage statistics
            window_usage = {}
            for window in windows:
                days_active = signal_df[f'window_{window}'].sum()
                pct_active = (days_active / total_days) * 100 if total_days > 0 else 0
                window_usage[window] = (days_active, pct_active)
                
            window_stats = ", ".join([f"W{w}: {d}/{total_days} ({p:.1f}%)" 
                                     for w, (d, p) in window_usage.items()])
            
            self.logger.info(
                f"Ensemble signal for {asset_id}: {signal_days}/{total_days} days in position ({signal_pct:.1f}%)"
            )
            self.logger.info(f"Window statistics for {asset_id}: {window_stats}")
            
            return signal_df['final_signal']
            
        except Exception as e:
            self.logger.error(f"Error calculating Donchian ensemble signal for {asset_id}: {str(e)}")
            self.logger.exception(e)  # Log full traceback
            return pd.Series(0, index=data.index)


# ========== USD QUOTE SIGNALS ==========

@register_signal
class DonchianEnsembleUSD(DonchianBreakoutSignal):
    """Ensemble signal combining all Donchian Channel breakout windows in USD."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'USD'


# ========== BTC QUOTE SIGNALS ==========

@register_signal
class DonchianEnsembleBTC(DonchianBreakoutSignal):
    """Ensemble signal combining all Donchian Channel breakout windows in BTC."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'BTC' 