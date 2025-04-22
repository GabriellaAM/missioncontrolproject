"""
Regime detection signals for the Soros System.

These signals are based on momentum and variance to classify market regimes.
Each signal returns 1 if the specified condition is detected, 0 otherwise.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List, Tuple

from .signal_base import SignalBase
from .signal_registry import register_signal


class RegimeDetectionBase(SignalBase):
    """Base class for regime detection signals using momentum and variance."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        # Default parameters
        self.price_col = self.params.get('price_col', 'close')
        self.quote_type = self.params.get('quote_type', 'USD')  # USD or BTC
        self.var_window = self.params.get('var_window', 5)  # 5-day log returns window
        self.var_ema_window = self.params.get('var_ema_window', 30)  # 30-day EMA window for variance
        self.var_norm_window = self.params.get('var_norm_window', 90)  # 90-day normalization window
        self.mom_ema_window = self.params.get('mom_ema_window', 5)  # 5-day EMA for momentum
        self.mom_norm_window = self.params.get('mom_norm_window', 14)  # 14-day normalization window
        
        # Cache for calculated components to avoid recalculation
        self._component_cache = {}
    
    def get_price_columns(self, data: pd.DataFrame, asset_id: str) -> Optional[Dict[str, str]]:
        """Get appropriate price columns based on quote type.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Dict with price column names or None if columns not found
        """
        if self.quote_type.upper() == 'USD':
            # Use standard USD price columns
            if 'close' in data.columns:
                return {
                    'open': 'open', 
                    'high': 'high', 
                    'low': 'low', 
                    'close': 'close'
                }
            return None
        
        elif self.quote_type.upper() == 'BTC':
            # Format for BTC-quoted price columns is {asset_id}_btc
            btc_col = f"{asset_id}_btc"
            if btc_col in data.columns:
                return {
                    'open': btc_col,  # Using the same column for all since we only need close
                    'high': btc_col,
                    'low': btc_col,
                    'close': btc_col
                }
            # Fallback to check the old format with _btc suffix
            elif 'close_btc' in data.columns:
                self.logger.warning(f"Using deprecated _btc suffix format for {asset_id}")
                return {
                    'open': 'open_btc', 
                    'high': 'high_btc', 
                    'low': 'low_btc', 
                    'close': 'close_btc'
                }
            return None
        
        # Unsupported quote type
        self.logger.warning(f"Unsupported quote type: {self.quote_type}")
        return None
    
    def calculate_components(self, data: pd.DataFrame, asset_id: str) -> Dict[str, pd.Series]:
        """Calculate variance and momentum components.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Dictionary with variance and momentum components
        """
        # Check cache first
        cache_key = (id(data), asset_id, self.quote_type)
        if cache_key in self._component_cache:
            return self._component_cache[cache_key]
            
        # Get appropriate price columns based on quote type
        price_cols = self.get_price_columns(data, asset_id)
        if not price_cols:
            self.logger.warning(f"Required price columns for {self.quote_type} not found in data for {asset_id}")
            return {}
            
        # Get close prices
        close_col = price_cols['close']
        close_prices = data[close_col]
        
        # 1. Variance component
        # Calculate log returns
        log_returns = np.log(close_prices / close_prices.shift(1))
        
        # Calculate rolling standard deviation of log returns
        rolling_std = log_returns.rolling(window=self.var_window).std()
        
        # Calculate the EMA of the standard deviation
        var_ema = rolling_std.ewm(span=self.var_ema_window, adjust=False).mean()
        
        # Calculate rolling normalized score of variance EMA
        var_mean = var_ema.rolling(window=self.var_norm_window).mean()
        var_std = var_ema.rolling(window=self.var_norm_window).std()
        var_score = (var_ema - var_mean) / var_std
        
        # 2. Momentum component
        # Calculate EMA of close price
        price_ema = close_prices.ewm(span=self.mom_ema_window, adjust=False).mean()
        
        # Calculate EMA of the EMA
        double_ema = price_ema.ewm(span=self.mom_ema_window, adjust=False).mean()
        
        # Calculate rolling normalized score
        mom_mean = double_ema.rolling(window=self.mom_norm_window).mean()
        mom_std = double_ema.rolling(window=self.mom_norm_window).std()
        mom_score = (double_ema - mom_mean) / mom_std
        
        # Store results
        result = {
            'var_score': var_score,
            'mom_score': mom_score
        }
        
        # Cache the result
        self._component_cache[cache_key] = result
        
        return result
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate input data."""
        # Check if we have enough data
        min_required = max(
            self.var_window + self.var_ema_window + self.var_norm_window,
            2 * self.mom_ema_window + self.mom_norm_window
        )
        
        if len(data) < min_required:
            self.logger.warning(f"Not enough data for regime detection on {asset_id}. "
                              f"Need at least {min_required} data points, got {len(data)}.")
            return False
            
        # Check if appropriate price columns exist based on quote type
        price_cols = self.get_price_columns(data, asset_id)
        if not price_cols:
            self.logger.warning(f"Required price columns for {self.quote_type} not found in data for {asset_id}")
            return False
            
        return True


# ========== USD QUOTE SIGNALS ==========

@register_signal
class BullHighVarianceSignalUSD(RegimeDetectionBase):
    """Signal for bull market with high variance regime in USD."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'USD'
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with signal values (1 for regime detected, 0 otherwise)
        """
        if not self.validate(data, asset_id):
            return pd.Series(0, index=data.index)
            
        # Calculate components
        components = self.calculate_components(data, asset_id)
        if not components:
            return pd.Series(0, index=data.index)
            
        var_score = components['var_score']
        mom_score = components['mom_score']
        
        # Bull High Variance: momentum score > 0 AND variance score > 0
        signal = ((mom_score > 0) & (var_score > 0)).astype(int)
        
        return signal


@register_signal
class BullLowVarianceSignalUSD(RegimeDetectionBase):
    """Signal for bull market with low variance regime in USD."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'USD'
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with signal values (1 for regime detected, 0 otherwise)
        """
        if not self.validate(data, asset_id):
            return pd.Series(0, index=data.index)
            
        # Calculate components
        components = self.calculate_components(data, asset_id)
        if not components:
            return pd.Series(0, index=data.index)
            
        var_score = components['var_score']
        mom_score = components['mom_score']
        
        # Bull Low Variance: momentum score > 0 AND variance score < 0
        signal = ((mom_score > 0) & (var_score < 0)).astype(int)
        
        return signal


@register_signal
class BearHighVarianceSignalUSD(RegimeDetectionBase):
    """Signal for bear market with high variance regime in USD."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'USD'
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with signal values (1 for regime detected, 0 otherwise)
        """
        if not self.validate(data, asset_id):
            return pd.Series(0, index=data.index)
            
        # Calculate components
        components = self.calculate_components(data, asset_id)
        if not components:
            return pd.Series(0, index=data.index)
            
        var_score = components['var_score']
        mom_score = components['mom_score']
        
        # Bear High Variance: momentum score < 0 AND variance score > 0
        signal = ((mom_score < 0) & (var_score > 0)).astype(int)
        
        return signal


@register_signal
class BearLowVarianceSignalUSD(RegimeDetectionBase):
    """Signal for bear market with low variance regime in USD."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'USD'
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with signal values (1 for regime detected, 0 otherwise)
        """
        if not self.validate(data, asset_id):
            return pd.Series(0, index=data.index)
            
        # Calculate components
        components = self.calculate_components(data, asset_id)
        if not components:
            return pd.Series(0, index=data.index)
            
        var_score = components['var_score']
        mom_score = components['mom_score']
        
        # Bear Low Variance: momentum score < 0 AND variance score < 0
        signal = ((mom_score < 0) & (var_score < 0)).astype(int)
        
        return signal


# ========== BTC QUOTE SIGNALS ==========

@register_signal
class BullHighVarianceSignalBTC(RegimeDetectionBase):
    """Signal for bull market with high variance regime in BTC."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'BTC'
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with signal values (1 for regime detected, 0 otherwise)
        """
        if not self.validate(data, asset_id):
            return pd.Series(0, index=data.index)
            
        # Calculate components
        components = self.calculate_components(data, asset_id)
        if not components:
            return pd.Series(0, index=data.index)
            
        var_score = components['var_score']
        mom_score = components['mom_score']
        
        # Bull High Variance: momentum score > 0 AND variance score > 0
        signal = ((mom_score > 0) & (var_score > 0)).astype(int)
        
        return signal


@register_signal
class BullLowVarianceSignalBTC(RegimeDetectionBase):
    """Signal for bull market with low variance regime in BTC."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'BTC'
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with signal values (1 for regime detected, 0 otherwise)
        """
        if not self.validate(data, asset_id):
            return pd.Series(0, index=data.index)
            
        # Calculate components
        components = self.calculate_components(data, asset_id)
        if not components:
            return pd.Series(0, index=data.index)
            
        var_score = components['var_score']
        mom_score = components['mom_score']
        
        # Bull Low Variance: momentum score > 0 AND variance score < 0
        signal = ((mom_score > 0) & (var_score < 0)).astype(int)
        
        return signal


@register_signal
class BearHighVarianceSignalBTC(RegimeDetectionBase):
    """Signal for bear market with high variance regime in BTC."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'BTC'
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with signal values (1 for regime detected, 0 otherwise)
        """
        if not self.validate(data, asset_id):
            return pd.Series(0, index=data.index)
            
        # Calculate components
        components = self.calculate_components(data, asset_id)
        if not components:
            return pd.Series(0, index=data.index)
            
        var_score = components['var_score']
        mom_score = components['mom_score']
        
        # Bear High Variance: momentum score < 0 AND variance score > 0
        signal = ((mom_score < 0) & (var_score > 0)).astype(int)
        
        return signal


@register_signal
class BearLowVarianceSignalBTC(RegimeDetectionBase):
    """Signal for bear market with low variance regime in BTC."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = 'BTC'
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data: DataFrame with price data
            asset_id: ID of the asset
            
        Returns:
            Series with signal values (1 for regime detected, 0 otherwise)
        """
        if not self.validate(data, asset_id):
            return pd.Series(0, index=data.index)
            
        # Calculate components
        components = self.calculate_components(data, asset_id)
        if not components:
            return pd.Series(0, index=data.index)
            
        var_score = components['var_score']
        mom_score = components['mom_score']
        
        # Bear Low Variance: momentum score < 0 AND variance score < 0
        signal = ((mom_score < 0) & (var_score < 0)).astype(int)
        
        return signal 