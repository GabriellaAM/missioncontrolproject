"""
Base interface for signal classes in the Soros System.

This module defines the base interface that all signal implementations must adhere to.
"""

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List, Tuple


class SignalBase(ABC):
    """Base class for all trading signals.
    
    All signal implementations must inherit from this class and 
    implement the `calculate` method.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the signal.
        
        Args:
            params (dict, optional): Parameters for the signal. Defaults to None.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.params = params or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate the signal values for the given data.
        
        Args:
            data (pd.DataFrame): Input data for the signal calculation.
            **kwargs: Additional keyword arguments for the calculation.
            
        Returns:
            pd.Series: A pandas Series with binary signal values (1 for active signal, 0 for inactive)
        """
        pass
    
    def normalize(self, values: pd.Series) -> pd.Series:
        """Normalize signal values to binary 1/0 range.
        
        Args:
            values (pd.Series): Raw signal values.
            
        Returns:
            pd.Series: Normalized signal values (1 or 0).
        """
        # Default implementation just returns the original values
        return values
    
    def validate(self, data: pd.DataFrame, asset_id: str) -> bool:
        """Validate the data for signal calculation.
        
        Args:
            data (pd.DataFrame): Data to validate.
            asset_id (str): ID of the asset.
            
        Returns:
            bool: True if data is valid, False otherwise.
        """
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
    
    def _get_asset_id(self) -> str:
        """Extract asset_id from parameters or data.
        
        Returns:
            str: Asset ID or empty string if not available.
        """
        asset_id = self.params.get('asset_id', '')
        
        # If asset_id is empty but we have data with asset_id column, use that
        if not asset_id and hasattr(self, 'data') and self.data is not None and 'asset_id' in self.data.columns:
            if not self.data.empty:
                asset_id = self.data['asset_id'].iloc[0]
        
        return asset_id
    
    def __str__(self) -> str:
        """String representation of the signal.
        
        Returns:
            str: Signal name and parameters.
        """
        return f"{self.name}(params={self.params})" 