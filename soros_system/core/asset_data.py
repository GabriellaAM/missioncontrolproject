"""
Asset data container for Soros System.

This module provides a container for all asset-specific data,
including price data, signals, and performance metrics.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from ..signals.signal_base import SignalBase
from .signal_data import SignalData


class AssetData:
    """
    Container for all asset-specific data.
    
    Stores price data, signals, and performance metrics for a specific asset.
    Provides helper methods for data access and signal management.
    """
    
    def __init__(self, asset_id: str, price_data: Optional[pd.DataFrame] = None):
        """Initialize the asset data container.
        
        Args:
            asset_id: Identifier for the asset
            price_data: DataFrame with price data for the asset (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.asset_id = asset_id
        self.price_data = price_data if price_data is not None else pd.DataFrame()
        
        # Storage for signal data
        self.signals = {}  # {signal_name: SignalData}
        
        # Storage for performance metrics
        self.metrics = {}
        
        # Storage for backtest results
        self.backtest_results = {}
        
    def set_price_data(self, price_data: pd.DataFrame) -> None:
        """Set or update the price data for this asset.
        
        Args:
            price_data: DataFrame with price data
        """
        self.price_data = price_data
        self.logger.debug(f"Updated price data for {self.asset_id}, {len(price_data)} rows")
        
    def add_signal(self, signal_name: str, signal_data: 'SignalData') -> None:
        """Add or update a signal for this asset.
        
        Args:
            signal_name: Name of the signal
            signal_data: SignalData object for this signal
        """
        self.signals[signal_name] = signal_data
        self.logger.debug(f"Added signal {signal_name} for {self.asset_id}")

    def register_signal(self, signal: SignalBase, params: Optional[Dict[str, Any]] = None) -> Optional[SignalData]:
        """Register a signal with this asset.
        
        This creates a SignalData object for the signal and calculates initial values.
        
        Args:
            signal: Signal instance to register
            params: Parameters for signal calculation
            
        Returns:
            SignalData object if successful, None otherwise
        """
        try:
            # Update signal parameters if provided
            if params:
                if hasattr(signal, 'params'):
                    signal.params.update(params)
                
            # Create SignalData object
            signal_data = SignalData(signal_name=signal.name, asset_id=self.asset_id)
            
            # Register the signal instance with the SignalData
            signal_data.register_signal_instance(signal)
            
            # Add signal to asset
            self.add_signal(signal.name, signal_data)
            
            # Calculate initial values if we have price data
            if self.has_data():
                try:
                    values = signal.calculate(self.price_data, self.asset_id)
                    signal_data.set_values(values)
                    self.logger.info(f"Calculated initial values for {signal.name} on {self.asset_id}: {len(values)} points")
                except Exception as e:
                    self.logger.error(f"Error calculating initial values for {signal.name} on {self.asset_id}: {e}")
            
            return signal_data
            
        except Exception as e:
            self.logger.error(f"Error registering signal {signal.name}: {e}")
            return None
        
    def get_signal(self, signal_name: str) -> Optional['SignalData']:
        """Get a signal by name.
        
        Args:
            signal_name: Name of the signal
            
        Returns:
            SignalData object if found, None otherwise
        """
        return self.signals.get(signal_name)
        
    def has_signal(self, signal_name: str) -> bool:
        """Check if a signal exists for this asset.
        
        Args:
            signal_name: Name of the signal
            
        Returns:
            bool: True if the signal exists, False otherwise
        """
        return signal_name in self.signals
        
    def get_signal_names(self) -> List[str]:
        """Get names of all signals for this asset.
        
        Returns:
            list: List of signal names
        """
        return list(self.signals.keys())
        
    def get_signals(self) -> Dict[str, 'SignalData']:
        """Get all signals for this asset.
        
        Returns:
            dict: Dictionary mapping signal names to SignalData objects
        """
        return self.signals
        
    def has_data(self) -> bool:
        """Check if the asset has price data loaded.
        
        Returns:
            bool: True if price data exists and is not empty, False otherwise
        """
        return not self.price_data.empty if hasattr(self.price_data, 'empty') else len(self.price_data) > 0
        
    def get_price_data(self) -> pd.DataFrame:
        """Get the price data for this asset.
        
        Returns:
            pd.DataFrame: The price data for this asset
        """
        return self.price_data
        
    def get_signal_values(self, start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get values for all signals within a date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame with signal values where columns are signal names
        """
        if not self.signals:
            return pd.DataFrame()
            
        # Create empty DataFrame with dates from price data
        if isinstance(self.price_data.index, pd.DatetimeIndex):
            dates = self.price_data.index
        elif 'date' in self.price_data.columns:
            dates = self.price_data['date']
        else:
            self.logger.warning(f"No dates found in price data for {self.asset_id}")
            return pd.DataFrame()
            
        # Apply date filtering
        if start_date is not None or end_date is not None:
            mask = pd.Series(True, index=dates)
            if start_date is not None:
                mask = mask & (dates >= start_date)
            if end_date is not None:
                mask = mask & (dates <= end_date)
            dates = dates[mask]
            
        # Create DataFrame with dates
        signal_values = pd.DataFrame(index=dates)
        
        # Add signal values
        for signal_name, signal_data in self.signals.items():
            if signal_data.values is not None:
                # Make sure we have values with the right index
                if isinstance(signal_data.values, pd.Series):
                    if len(signal_data.values) > 0:
                        # Reindex to match our dates
                        signal_values[signal_name] = signal_data.values.reindex(signal_values.index)
                else:
                    self.logger.warning(f"Signal {signal_name} has invalid values")
            
        return signal_values
        
    def add_backtest_result(self, backtest_name: str, result: Dict[str, Any]) -> None:
        """Add a backtest result for this asset.
        
        Args:
            backtest_name: Name of the backtest
            result: Dictionary with backtest results
        """
        self.backtest_results[backtest_name] = result
        self.logger.debug(f"Added backtest result {backtest_name} for {self.asset_id}")
        
    def get_backtest_result(self, backtest_name: str) -> Optional[Dict[str, Any]]:
        """Get a backtest result by name.
        
        Args:
            backtest_name: Name of the backtest
            
        Returns:
            Dict with backtest results if found, None otherwise
        """
        return self.backtest_results.get(backtest_name)
        
    def get_backtest_names(self) -> List[str]:
        """Get names of all backtests for this asset.
        
        Returns:
            list: List of backtest names
        """
        return list(self.backtest_results.keys())
        
    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        """Set or update the performance metrics for this asset.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        self.metrics = metrics
        self.logger.debug(f"Updated metrics for {self.asset_id}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this asset.
        
        Returns:
            dict: Dictionary with performance metrics
        """
        return self.metrics
        
    def __repr__(self) -> str:
        """String representation of the asset data container."""
        signals_str = ", ".join(self.get_signal_names())
        backtests_str = ", ".join(self.get_backtest_names())
        
        return (f"AssetData(asset_id='{self.asset_id}', "
                f"price_data_length={len(self.price_data)}, "
                f"signals=[{signals_str}], "
                f"backtests=[{backtests_str}])")
                
    def clear_cache(self) -> None:
        """Clear all cached data."""
        # Clear backtest results which can be regenerated
        self.backtest_results = {}
        
        # Keep primary data (price data, signals, metrics)
        self.logger.debug(f"Cleared cache for {self.asset_id}") 