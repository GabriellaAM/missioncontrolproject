"""
Trend Momentum signals based on 3 EMAs for the Soros System.

These signals use three exponential moving averages (fast, mid, slow) to determine
trend momentum and generate buy/sell signals based on the alignment of these EMAs.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List
import ta

from .signal_base import SignalBase
from .signal_registry import register_signal


class TrendMomentumBase(SignalBase):
    """Base class for trend momentum signals using 3 EMAs."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = self.params.get('quote_type', 'USD')
        
        # EMA lengths from the TradingView script
        self.fast_length = self.params.get('fast_length', 9)
        self.mid_length = self.params.get('mid_length', 21)
        self.slow_length = self.params.get('slow_length', 50)
        
        # Track if we've already warned about missing column
        self._warned_missing_column = False
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        if self.quote_type == 'USD':
            return ['open', 'high', 'low', 'close']
        else:  # BTC quote type
            # Skip BTC signals for bitcoin itself
            asset_id = self._get_asset_id()
            if asset_id == 'bitcoin':
                return ['open', 'high', 'low', 'close']
            # For other assets, require OHLC and BTC price columns
            return ['open', 'high', 'low', 'close', f'{asset_id}_btc']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        # Need enough samples for the slowest EMA plus some buffer
        return self.slow_length + 10
    
    def _get_asset_id(self) -> str:
        """Extract asset_id from parameters."""
        return self.params.get('asset_id', '')
    
    def get_price_data(self, data: pd.DataFrame) -> pd.Series:
        """Get the appropriate price data based on quote type.
        
        Uses OHLC4 (average of open, high, low, close) like the TradingView script.
        """
        asset_id = self._get_asset_id()
        
        if self.quote_type == 'USD':
            # Calculate OHLC4 for USD
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                return (data['open'] + data['high'] + data['low'] + data['close']) / 4
            else:
                if not self._warned_missing_column:
                    self.logger.warning(f"Missing OHLC columns for {asset_id}")
                    self._warned_missing_column = True
                return None
        
        # For BTC quotes
        if asset_id == 'bitcoin':
            # Bitcoin uses its own OHLC4 price
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                return (data['open'] + data['high'] + data['low'] + data['close']) / 4
            else:
                return None
        
        # For BTC quotes of other assets, calculate OHLC4 from BTC-adjusted columns
        btc_open = f'{asset_id}_btc_open'
        btc_high = f'{asset_id}_btc_high'
        btc_low = f'{asset_id}_btc_low'
        btc_close = f'{asset_id}_btc'
        
        if all(col in data.columns for col in [btc_open, btc_high, btc_low, btc_close]):
            return (data[btc_open] + data[btc_high] + data[btc_low] + data[btc_close]) / 4
        
        # If no BTC OHLC columns found, log warning and return None
        if not self._warned_missing_column:
            self.logger.warning(f"Missing required BTC OHLC columns for {asset_id}")
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
        
        # Get appropriate price data
        price_data = self.get_price_data(data)
        if price_data is None:
            return False
            
        return True
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the trend momentum signal.
        
        Signal logic:
        - Green (bullish stack): fast > mid AND fast > slow => Signal = 1
        - Red (bearish stack): fast < mid AND fast < slow => Signal = 0
        - Yellow (mixed): Previous state determines the signal
            - If previous was red => Signal = 1 (momentum shifting positive)
            - If previous was green => Signal = 0 (momentum shifting negative)
            - If previous was yellow => Keep previous signal state
        """
        if not self.validate(data, asset_id):
            return pd.Series(0, index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        try:
            # Get OHLC4 price data
            price_data = self.get_price_data(data)
            if price_data is None:
                return pd.Series(0, index=data.index)
            
            # Calculate the three EMAs
            ema_fast = ta.trend.EMAIndicator(close=price_data, window=self.fast_length).ema_indicator()
            ema_mid = ta.trend.EMAIndicator(close=price_data, window=self.mid_length).ema_indicator()
            ema_slow = ta.trend.EMAIndicator(close=price_data, window=self.slow_length).ema_indicator()
            
            # Initialize signal series
            signal = pd.Series(0, index=data.index)
            
            # Track candle colors for state logic
            # Green = 2, Yellow = 1, Red = 0
            candle_colors = pd.Series(1, index=data.index)  # Default to yellow
            
            # Determine candle colors based on EMA alignment
            bullish_mask = (ema_fast > ema_mid) & (ema_fast > ema_slow)
            bearish_mask = (ema_fast < ema_mid) & (ema_fast < ema_slow)
            
            candle_colors[bullish_mask] = 2  # Green
            candle_colors[bearish_mask] = 0  # Red
            # Everything else remains 1 (Yellow)
            
            # Apply signal logic with state memory
            last_signal = 0
            
            for idx in data.index:
                if pd.isna(ema_fast.loc[idx]) or pd.isna(ema_mid.loc[idx]) or pd.isna(ema_slow.loc[idx]):
                    signal.loc[idx] = last_signal
                    continue
                
                current_color = candle_colors.loc[idx]
                
                # Get previous color (if available)
                prev_idx = data.index.get_loc(idx) - 1
                if prev_idx >= 0:
                    prev_color = candle_colors.iloc[prev_idx]
                else:
                    prev_color = 1  # Assume yellow if no previous
                
                # Apply signal rules
                if current_color == 2:  # Green candle
                    signal.loc[idx] = 1
                    last_signal = 1
                elif current_color == 0:  # Red candle
                    signal.loc[idx] = 0
                    last_signal = 0
                else:  # Yellow candle (current_color == 1)
                    if prev_color == 0:  # Previous was red
                        signal.loc[idx] = 1  # Momentum shifting positive
                        last_signal = 1
                    elif prev_color == 2:  # Previous was green
                        signal.loc[idx] = 0  # Momentum shifting negative
                        last_signal = 0
                    else:  # Previous was also yellow
                        signal.loc[idx] = last_signal  # Keep previous signal state
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error calculating Trend Momentum signal for {asset_id}: {str(e)}")
            return pd.Series(0, index=data.index)


# ========== USD QUOTE SIGNALS ==========

@register_signal
class TrendMomentumUSD(TrendMomentumBase):
    """Trend momentum signal using 3 EMAs in USD terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                fast_length (int): Length of fast EMA (default: 9)
                mid_length (int): Length of mid EMA (default: 21)
                slow_length (int): Length of slow EMA (default: 50)
        """
        super().__init__(params)
        self.quote_type = 'USD'


# ========== BTC QUOTE SIGNALS ==========

@register_signal
class TrendMomentumBTC(TrendMomentumBase):
    """Trend momentum signal using 3 EMAs in BTC terms."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                fast_length (int): Length of fast EMA (default: 9)
                mid_length (int): Length of mid EMA (default: 21)
                slow_length (int): Length of slow EMA (default: 50)
        """
        super().__init__(params)
        self.quote_type = 'BTC'