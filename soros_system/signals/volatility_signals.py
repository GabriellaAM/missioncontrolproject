"""
Volatility-based signals for the Soros System.

These signals are based on volatility regimes detected by Markov models.
Each signal returns 1 if the specified condition is detected, 0 otherwise.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .signal_base import SignalBase
from .signal_registry import register_signal
from ..analysis.markov_vol_model import MarkovVolModel


class MarkovVolatilitySignalBase(SignalBase):
    """Base class for all Markov volatility model signals."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal.
                model_path (str): Path to the Markov model file (optional)
        """
        super().__init__(params)
        self.model_path = self.params.get('model_path', None)
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


@register_signal
class MarkovLowVolatilitySignal(MarkovVolatilitySignalBase):
    """Signal that detects low volatility regimes using Markov model.
    
    Returns:
        1: When low volatility is detected
        0: When low volatility is not detected
    """
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
            1: When low volatility is detected
            0: When low volatility is not detected
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
                    state_name = self.vol_state_cache[date]
                else:
                    # Get volatility state from the model
                    state_name = self.markov_model.get_volatility_state(date)
                    
                    # Cache the result
                    self.vol_state_cache[date] = state_name
                
                # Set signal value - 1 if low volatility, 0 otherwise
                signal.iloc[idx] = 1 if state_name == 'low' else 0
                
            except Exception as e:
                self.logger.warning(f"Error getting volatility state for {date}: {e}")
                signal.iloc[idx] = np.nan
        
        # Fill NaN values with 0 (not detected) as a conservative approach
        signal = signal.fillna(0)
        
        return signal


@register_signal
class MarkovHighVolatilitySignal(MarkovVolatilitySignalBase):
    """Signal that detects high volatility regimes using Markov model.
    
    Returns:
        1: When high volatility is detected
        0: When high volatility is not detected
    """
    
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values.
        
        Args:
            data (pd.DataFrame): DataFrame with price data.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (1 or 0) indexed by date.
            1: When high volatility is detected
            0: When high volatility is not detected
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
                    state_name = self.vol_state_cache[date]
                else:
                    # Get volatility state from the model
                    state_name = self.markov_model.get_volatility_state(date)
                    
                    # Cache the result
                    self.vol_state_cache[date] = state_name
                
                # Set signal value - 1 if high volatility, 0 otherwise
                signal.iloc[idx] = 1 if state_name == 'high' else 0
                
            except Exception as e:
                self.logger.warning(f"Error getting volatility state for {date}: {e}")
                signal.iloc[idx] = np.nan
        
        # Fill NaN values with 0 (not detected) as a conservative approach
        signal = signal.fillna(0)
        
        return signal


