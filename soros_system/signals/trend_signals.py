"""
Trend-based signals for the Soros System.

These signals are based on trend classifications from moving averages and rate of change indicators.
"""

import pandas as pd
import numpy as np
import logging 
from typing import Optional, Dict, Any, List, Tuple 
from functools import lru_cache

from .signal_base import SignalBase # base class for all signals
from .signal_registry import register_signal # register signals in the system
from ..indicators.moving_averages import MovingAverageCalculator # calculate all MAs and crossovers
from ..indicators.trend_classifier import TrendClassifier # classify trends based on MAs and RoC


# Global trend classification cache to avoid redundant calculations
_TREND_CACHE = {}


# Base class for all trend signals to reduce code duplication
class TrendSignalBase(SignalBase):
    """Base class for all trend signals."""
    
    # Class-level MA calculator and trend classifier to avoid creating new instances for each signal
    _ma_calculator = MovingAverageCalculator()
    _trend_classifier = TrendClassifier(_ma_calculator)
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal."""
        super().__init__(params)
        self.quote_type = self.params.get('quote_type', 'USD')
        
        # Each subclass should define these:
        self.term = "short_term"  # short_term, medium_term, long_term, overall
        self.trend_value = None  # -2, -1, 0, 1, 2
        
        # Store a reference to data to help with asset_id lookup
        self.data = None
        
        # Track if we've already warned about missing BTC column
        self._warned_missing_btc = False
        
        # Whether to include 'period' column in required columns
        self.use_period = self.params.get('use_period', False)
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        # Get asset_id first to avoid duplicating logic
        asset_id = self._get_asset_id()
        
        if self.quote_type == 'USD':
            if asset_id == 'bitcoin':
                # For bitcoin itself, we only need the close column
                return ['close']
            else:
                # For non-bitcoin assets in USD, we need both close and optionally period
                return ['close', 'period'] if self.use_period else ['close']
        else:
            # For BTC quote type
            if asset_id == 'bitcoin':
                # For bitcoin itself in BTC quote, just close (though this is not a common use case)
                return ['close']
            elif asset_id:
                # For other assets in BTC quote with a valid asset_id, we need the BTC price column
                return [f'{asset_id}_btc']
            else:
                # Default to close if we don't have a valid asset_id
                return ['close']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        # Different terms require different minimum samples
        if self.term == "short_term":
            return 60
        elif self.term == "medium_term":
            return 120
        elif self.term == "long_term" or self.term == "overall":
            return 200
        else:
            return 200  # Default
    
    def _get_asset_id(self) -> str:
        """Extract asset_id from parameters."""
        return self.params.get('asset_id', '')
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation."""
        if not super().validate(data, asset_id):
            return False
            
        # Skip BTC signals for bitcoin itself
        if self.quote_type == 'BTC' and asset_id == 'bitcoin':
            return False
        
        # For BTC quote type, check if the expected BTC column exists
        if self.quote_type == 'BTC':
            # Get the BTC price column name
            btc_col = f'{asset_id}_btc'
            
            if btc_col in data.columns:
                # Store the BTC column name for use in calculate method
                self.params['_btc_column'] = btc_col
                return True
                
            # If no suitable column found, log and return False - but only once per instance
            if not self._warned_missing_btc:
                self.logger.warning(f"Missing required BTC column '{btc_col}' for {asset_id}")
                self.logger.debug(f"Available columns: {data.columns.tolist()}")
                self._warned_missing_btc = True
            return False
        
        # For USD quote type, require 'close' column
        elif 'close' not in data.columns:
            self.logger.warning(f"Missing 'close' column for {asset_id}")
            return False
            
        return True
    
    @staticmethod
    def get_trend_data(data: pd.DataFrame, asset_id: str, trend_classifier=None) -> pd.DataFrame:
        """Get the classified trend data for an asset, using cache if available.
        
        Args:
            data (pd.DataFrame): Price data
            asset_id (str): Asset ID
            trend_classifier: Optional trend classifier instance
            
        Returns:
            pd.DataFrame: Classified trend data
        """
        # Create a cache key based on asset_id and data shape/hash
        # This ensures we don't reuse stale data
        data_hash = hash(str(data.shape) + str(data.index[0]) + str(data.index[-1]))
        cache_key = f"{asset_id}_{data_hash}"
        
        # Check if we have cached results
        if cache_key in _TREND_CACHE:
            return _TREND_CACHE[cache_key]
        
        # No cache hit, calculate the trend data
        if trend_classifier is None:
            trend_classifier = TrendSignalBase._trend_classifier
            
        # Calculate trend classifications
        classified_data = trend_classifier._create_classified_data(data, asset_id)
        
        # Cache the result
        _TREND_CACHE[cache_key] = classified_data
        
        # Log cache status
        logger = logging.getLogger(__name__)
        logger.info(f"Calculated and cached trend data for {asset_id}, cache size: {len(_TREND_CACHE)}")
        
        return classified_data
    
    @staticmethod
    def clear_trend_cache():
        """Clear the trend data cache."""
        global _TREND_CACHE
        cache_size = len(_TREND_CACHE)
        _TREND_CACHE = {}
        logger = logging.getLogger(__name__)
        logger.info(f"Cleared trend cache, removed {cache_size} entries")
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Store the asset_id in params for later use
        self.params['asset_id'] = asset_id
        
        # Get the classified data from cache or calculate it
        classified_data = self.get_trend_data(data, asset_id)
        
        # Get the trend column
        trend_col = f'{self.term}_trend_{self.quote_type}'
        
        if trend_col not in classified_data.columns:
            self.logger.warning(f"Trend column {trend_col} not found for {asset_id}")
            return pd.Series(index=data.index)
        
        # Create signal based on exact trend value match
        signal = classified_data[trend_col].apply(
            lambda x: 1 if pd.notna(x) and x == self.trend_value else 0
        )
        
        return signal


# ========== USD QUOTE SIGNALS ==========

# Short-term trend signals (USD)

@register_signal
class ShortTermStrongBearUSD(TrendSignalBase):
    """Signal for short-term strong bear trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "short_term"
        self.trend_value = -2

@register_signal
class ShortTermWeakBearUSD(TrendSignalBase):
    """Signal for short-term weak bear trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "short_term"
        self.trend_value = -1

@register_signal
class ShortTermNeutralUSD(TrendSignalBase):
    """Signal for short-term neutral trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "short_term"
        self.trend_value = 0

@register_signal
class ShortTermWeakBullUSD(TrendSignalBase):
    """Signal for short-term weak bull trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "short_term"
        self.trend_value = 1

@register_signal
class ShortTermStrongBullUSD(TrendSignalBase):
    """Signal for short-term strong bull trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "short_term"
        self.trend_value = 2

# Medium-term trend signals (USD)

@register_signal
class MediumTermStrongBearUSD(TrendSignalBase):
    """Signal for medium-term strong bear trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "medium_term"
        self.trend_value = -2

@register_signal
class MediumTermWeakBearUSD(TrendSignalBase):
    """Signal for medium-term weak bear trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "medium_term"
        self.trend_value = -1

@register_signal
class MediumTermNeutralUSD(TrendSignalBase):
    """Signal for medium-term neutral trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "medium_term"
        self.trend_value = 0

@register_signal
class MediumTermWeakBullUSD(TrendSignalBase):
    """Signal for medium-term weak bull trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "medium_term"
        self.trend_value = 1

@register_signal
class MediumTermStrongBullUSD(TrendSignalBase):
    """Signal for medium-term strong bull trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "medium_term"
        self.trend_value = 2

# Long-term trend signals (USD)

@register_signal
class LongTermStrongBearUSD(TrendSignalBase):
    """Signal for long-term strong bear trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "long_term"
        self.trend_value = -2

@register_signal
class LongTermWeakBearUSD(TrendSignalBase):
    """Signal for long-term weak bear trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "long_term"
        self.trend_value = -1

@register_signal
class LongTermNeutralUSD(TrendSignalBase):
    """Signal for long-term neutral trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "long_term"
        self.trend_value = 0

@register_signal
class LongTermWeakBullUSD(TrendSignalBase):
    """Signal for long-term weak bull trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "long_term"
        self.trend_value = 1

@register_signal
class LongTermStrongBullUSD(TrendSignalBase):
    """Signal for long-term strong bull trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "long_term"
        self.trend_value = 2

# Overall trend signals (USD)

@register_signal
class OverallStrongBearUSD(TrendSignalBase):
    """Signal for overall strong bear trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "overall"
        self.trend_value = -2

@register_signal
class OverallWeakBearUSD(TrendSignalBase):
    """Signal for overall weak bear trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "overall"
        self.trend_value = -1

@register_signal
class OverallNeutralUSD(TrendSignalBase):
    """Signal for overall neutral trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "overall"
        self.trend_value = 0

@register_signal
class OverallWeakBullUSD(TrendSignalBase):
    """Signal for overall weak bull trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "overall"
        self.trend_value = 1

@register_signal
class OverallStrongBullUSD(TrendSignalBase):
    """Signal for overall strong bull trend in USD."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'USD'
        self.term = "overall"
        self.trend_value = 2


# ========== BTC QUOTE SIGNALS ==========

# Short-term trend signals (BTC)

@register_signal
class ShortTermStrongBearBTC(TrendSignalBase):
    """Signal for short-term strong bear trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "short_term"
        self.trend_value = -2

@register_signal
class ShortTermWeakBearBTC(TrendSignalBase):
    """Signal for short-term weak bear trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "short_term"
        self.trend_value = -1

@register_signal
class ShortTermNeutralBTC(TrendSignalBase):
    """Signal for short-term neutral trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "short_term"
        self.trend_value = 0

@register_signal
class ShortTermWeakBullBTC(TrendSignalBase):
    """Signal for short-term weak bull trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "short_term"
        self.trend_value = 1

@register_signal
class ShortTermStrongBullBTC(TrendSignalBase):
    """Signal for short-term strong bull trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "short_term"
        self.trend_value = 2

# Medium-term trend signals (BTC)

@register_signal
class MediumTermStrongBearBTC(TrendSignalBase):
    """Signal for medium-term strong bear trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "medium_term"
        self.trend_value = -2

@register_signal
class MediumTermWeakBearBTC(TrendSignalBase):
    """Signal for medium-term weak bear trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "medium_term"
        self.trend_value = -1

@register_signal
class MediumTermNeutralBTC(TrendSignalBase):
    """Signal for medium-term neutral trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "medium_term"
        self.trend_value = 0

@register_signal
class MediumTermWeakBullBTC(TrendSignalBase):
    """Signal for medium-term weak bull trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "medium_term"
        self.trend_value = 1

@register_signal
class MediumTermStrongBullBTC(TrendSignalBase):
    """Signal for medium-term strong bull trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "medium_term"
        self.trend_value = 2

# Long-term trend signals (BTC)

@register_signal
class LongTermStrongBearBTC(TrendSignalBase):
    """Signal for long-term strong bear trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "long_term"
        self.trend_value = -2

@register_signal
class LongTermWeakBearBTC(TrendSignalBase):
    """Signal for long-term weak bear trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "long_term"
        self.trend_value = -1

@register_signal
class LongTermNeutralBTC(TrendSignalBase):
    """Signal for long-term neutral trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "long_term"
        self.trend_value = 0

@register_signal
class LongTermWeakBullBTC(TrendSignalBase):
    """Signal for long-term weak bull trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "long_term"
        self.trend_value = 1

@register_signal
class LongTermStrongBullBTC(TrendSignalBase):
    """Signal for long-term strong bull trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "long_term"
        self.trend_value = 2

# Overall trend signals (BTC)

@register_signal
class OverallStrongBearBTC(TrendSignalBase):
    """Signal for overall strong bear trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "overall"
        self.trend_value = -2

@register_signal
class OverallWeakBearBTC(TrendSignalBase):
    """Signal for overall weak bear trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "overall"
        self.trend_value = -1

@register_signal
class OverallNeutralBTC(TrendSignalBase):
    """Signal for overall neutral trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "overall"
        self.trend_value = 0

@register_signal
class OverallWeakBullBTC(TrendSignalBase):
    """Signal for overall weak bull trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "overall"
        self.trend_value = 1

@register_signal
class OverallStrongBullBTC(TrendSignalBase):
    """Signal for overall strong bull trend in BTC."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.quote_type = 'BTC'
        self.term = "overall"
        self.trend_value = 2 