"""
Signal data container for Soros System.

This module provides a container for signal-specific information,
including signal values and activation dates.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

from ..signals.signal_base import SignalBase


class SignalData:
    """
    Container for signal-specific information.
    
    Stores signal values and activation dates.
    """
    
    def __init__(
        self, 
        signal_name: str, 
        asset_id: str, 
        values: Optional[pd.Series] = None
    ):
        """Initialize the signal data container.
        
        Args:
            signal_name: Name of the signal
            asset_id: ID of the asset this signal is for
            values: Series with signal values (index should be dates)
        """
        self.logger = logging.getLogger(__name__)
        self.signal_name = signal_name
        self.asset_id = asset_id
        self.values = values
        
        # Signal configuration
        self.weight = 1.0  # Default weight
        self.decay = 0     # No decay period
        
        # Activation tracking
        self.activation_dates = []
        
        # The signal instance (if registered)
        self.signal_instance = None
        
    @property
    def name(self) -> str:
        """Alias for signal_name for compatibility.
        
        Returns:
            str: The name of the signal
        """
        return self.signal_name
        
    def set_values(self, values: pd.Series) -> None:
        """Set or update the signal values.
        
        Args:
            values: Series with signal values (index should be dates)
        """
        self.values = values
        self.logger.debug(f"Updated values for signal {self.signal_name} on {self.asset_id}")
        
        # Extract activation dates (0->1 transitions)
        self.extract_activation_dates()
        
    def extract_activation_dates(self) -> List[datetime]:
        """Extract dates where the signal activates (transitions from 0 to 1).
        
        Returns:
            list: List of activation dates
        """
        if self.values is None or len(self.values) < 2:
            self.activation_dates = []
            return []
            
        # Find transitions from 0 to 1
        transitions = (self.values.shift(1) == 0) & (self.values == 1)
        
        # Get dates where transitions occur
        activation_dates = self.values.index[transitions].tolist()
        self.activation_dates = activation_dates
        
        self.logger.debug(
            f"Extracted {len(activation_dates)} activation dates for signal {self.signal_name} on {self.asset_id}"
        )
        
        return activation_dates
    
    def set_parameters(self, weight: float, decay: int) -> None:
        """Set parameters for this signal.
        
        Args:
            weight: Weight of the signal
            decay: Decay period (in days) for the signal - ignored, always set to 0
        """
        self.weight = weight
        self.decay = 0  # Always set to 0 - no decay
        self.logger.debug(
            f"Set parameters for signal {self.signal_name} on {self.asset_id}: "
            f"weight={weight:.4f}, decay=0"
        )
        
    def get_activations_in_range(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[datetime]:
        """Get activation dates within a specified range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            list: List of activation dates within the range
        """
        if not self.activation_dates:
            return []
            
        filtered_dates = self.activation_dates
        
        if start_date is not None:
            filtered_dates = [d for d in filtered_dates if d >= start_date]
            
        if end_date is not None:
            filtered_dates = [d for d in filtered_dates if d <= end_date]
            
        return filtered_dates
    
    def calculate_values(self, data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Calculate values for this signal.
        
        Args:
            data: DataFrame with price data (optional)
            
        Returns:
            Series with signal values
        """
        if self.signal_instance is None:
            self.logger.debug(f"No signal instance registered for {self.signal_name}")
            # Return empty series with proper index if data is provided
            if data is not None and not data.empty:
                return pd.Series(index=data.index)
            return pd.Series()
            
        try:
            # Calculate values
            if data is not None:
                # Pass both data and asset_id to the calculate method
                values = self.signal_instance.calculate(data, self.asset_id)
            else:
                # If no data provided, we don't try to recalculate
                # Just return the existing values if we have them
                if self.values is not None:
                    return self.values
                
                # Log that we need data for calculation
                self.logger.warning(f"No data provided for {self.signal_name} calculation on {self.asset_id}")
                return pd.Series()
                
            # Store values
            self.set_values(values)
            
            return values
            
        except Exception as e:
            self.logger.error(f"Error calculating values for {self.signal_name}: {e}")
            return pd.Series()
    
    def register_signal_instance(self, signal: SignalBase) -> None:
        """Register a signal instance with this SignalData.
        
        Args:
            signal: Signal instance
        """
        self.signal_instance = signal
        self.logger.debug(f"Registered signal instance for {self.signal_name}")
    
    def __str__(self) -> str:
        """String representation of the signal data."""
        return (f"SignalData(signal_name='{self.signal_name}', "
                f"asset_id='{self.asset_id}', "
                f"values_count={len(self.values) if self.values is not None else 0}, "
                f"activations={len(self.activation_dates)})")
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert this signal data to a dictionary.
        
        Returns:
            dict: Dictionary representation of this signal data
        """
        return {
            'signal_name': self.signal_name,
            'asset_id': self.asset_id,
            'weight': self.weight,
            'decay': self.decay,
            'activations_count': len(self.activation_dates)
        } 