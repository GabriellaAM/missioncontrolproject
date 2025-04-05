"""
Base interface for signal classes in the Soros System.

This module defines the base interface that all signal implementations must adhere to.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union


class SignalBase(ABC):
    """Base class for all trading signals.
    
    All signal implementations should inherit from this class and implement
    the required methods. Signals generate binary values: +1 (bullish/buy)
    or 0 (bearish/sell).
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal calculation.
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.params = params or {} 
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, asset_id: str) -> pd.Series:
        """Calculate the signal values for the given data.
        
        Args:
            data (pd.DataFrame): Asset data with price information.
            asset_id (str): ID of the asset.
            
        Returns:
            pd.Series: Series with signal values (+1 or 0) indexed by date.
        """
        pass
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Check if the signal can be calculated with the available data.
        
        Args:
            data (pd.DataFrame): Asset data with price information.
            asset_id (str): ID of the asset.
            
        Returns:
            bool: True if the signal can be calculated, False otherwise.
        """
        if data is None or data.empty:
            self.logger.warning(f"Empty data provided for {asset_id}")
            return False
        
        # Basic validation - child classes should override with specific checks
        return True
    
    def get_required_columns(self) -> list:
        """Get the columns required for signal calculation.
        
        Returns:
            list: List of column names required for signal calculation.
        """
        # Child classes should override this method
        return ['close']
    
    def get_min_required_samples(self) -> int:
        """Get the minimum number of samples required for signal calculation.
        
        Returns:
            int: Minimum number of samples.
        """
        # Child classes should override this method
        return 30  # Default minimum of 30 data points
    
    def __str__(self) -> str:
        """String representation of the signal.
        
        Returns:
            str: Signal name and parameters.
        """
        return f"{self.name}(params={self.params})" 