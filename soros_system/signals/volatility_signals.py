"""
Volatility-based signals for the Soros System.

These signals are based on volatility regimes detected by Markov models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .signal_base import SignalBase
from .signal_registry import register_signal
from ..analysis.markov_vol_model import MarkovVolModel


@register_signal
class MarkovVolatilitySignal(SignalBase):
    """Signal based on Markov volatility model regime detection."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                model_path (str): Path to the Markov model file (optional)
                inverse (bool): Whether to invert the signal (default: False)
        """
        super().__init__(params)
        self.model_path = self.params.get('model_path', None)
        self.inverse = self.params.get('inverse', False)
        self.markov_model = MarkovVolModel(model_path=self.model_path)
        
        # Initialize state cache
        self.vol_state_cache = {}
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        # This signal doesn't need specific columns as it uses the date for lookups
        return []
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return 1  # Only need date information
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation."""
        if data is None or data.empty:
            self.logger.warning(f"Empty data provided for {asset_id}")
            return False
        
        # Check if the model is loaded
        if self.markov_model.model is None:
            self.logger.warning("Markov volatility model not loaded.")
            return False
        
        # Check if index is datetime
        if not isinstance(data.index, pd.DatetimeIndex) and 'date' not in data.columns:
            self.logger.warning(f"No date information available for {asset_id}")
            return False
        
        return True
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (+1 or -1) indexed by date.
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Create a Series to hold the signal values
        signal = pd.Series(index=data.index)
        
        # Get dates for volatility lookup
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        else:
            # Use date column
            if 'date' in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    data['date'] = pd.to_datetime(data['date'])
                dates = data['date']
            else:
                self.logger.warning(f"No date information available for {asset_id}")
                return signal
        
        # Calculate volatility state for each date
        for idx, date in enumerate(dates):
            try:
                # Use cached value if available
                if date in self.vol_state_cache:
                    vol_state = self.vol_state_cache[date]
                else:
                    # Get volatility state from the model
                    state_name = self.markov_model.get_volatility_state(date)
                    
                    # Map state name to signal value
                    if state_name == 'low':
                        vol_state = 1  # Low volatility is bullish
                    elif state_name == 'high':
                        vol_state = -1  # High volatility is bearish
                    else:
                        vol_state = np.nan  # Unknown state
                    
                    # Cache the result
                    self.vol_state_cache[date] = vol_state
                
                # Apply inverse if specified
                if self.inverse:
                    vol_state = -vol_state
                
                # Set signal value
                signal.iloc[idx] = vol_state
                
            except Exception as e:
                self.logger.warning(f"Error getting volatility state for {date}: {e}")
                signal.iloc[idx] = np.nan
        
        # Fill NaN values with -1 (bearish) as a conservative approach
        signal = signal.fillna(-1)
        
        return signal


@register_signal
class VolatilityTrendSignal(SignalBase):
    """
    Signal based on the trend of volatility changes.
    
    This looks at whether volatility is increasing or decreasing
    over a specified window, rather than just the absolute level.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                window (int): Window size for volatility trend calculation (default: 14)
                threshold (float): Threshold for considering volatility changing (default: 0.1)
        """
        super().__init__(params)
        self.window = self.params.get('window', 14)
        self.threshold = self.params.get('threshold', 0.1)
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for the signal calculation."""
        return ['close']
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the signal calculation."""
        return self.window * 2  # Need enough data to calculate volatility trend
    
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
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (+1 or -1) indexed by date.
        """
        if not self.validate(data, asset_id):
            # Return empty series with same index as data
            return pd.Series(index=data.index)
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.window).std()
        
        # Calculate volatility slope (trend)
        volatility_change = volatility.pct_change(self.window)
        
        # Generate signal based on volatility trend
        # +1 when volatility is decreasing, -1 when increasing
        signal = volatility_change.apply(
            lambda x: 1 if pd.notna(x) and x < -self.threshold else
                    (-1 if pd.notna(x) and x > self.threshold else -1)
        )
        
        # Fill initial NaN values
        signal = signal.fillna(-1)
        
        return signal 