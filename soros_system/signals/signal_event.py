"""
Signal event tracking for the Soros System.

This module implements signal activation event tracking for event-driven
signal logic, allowing signals to remain active for a fixed decay period
after activation.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any


class SignalEvent:
    """
    Represents a signal activation event.
    
    Tracks when a signal activates (0→1), the optimal decay period,
    and calculates the expiration date. If the signal reactivates during
    the decay period, the expiration will reset.
    """
    
    def __init__(
        self, 
        signal_name: str, 
        asset_id: str,
        activation_date: datetime,
        decay: int,  # in days
        weight: float,
        meta_approved: bool = True
    ):
        """Initialize a signal event.
        
        Args:
            signal_name: Name of the signal
            asset_id: ID of the asset
            activation_date: Date when the signal activated
            decay: Number of days to hold the signal (optimal period)
            weight: Weight of the signal based on its effectiveness
            meta_approved: Whether this signal was approved by meta-labeling
        """
        self.logger = logging.getLogger(__name__)
        self.signal_name = signal_name
        self.asset_id = asset_id
        self.activation_date = activation_date
        self.decay = decay
        self.expiration_date = activation_date + timedelta(days=decay)
        self.weight = weight
        self.meta_approved = meta_approved
        
    def is_active(self, current_date: datetime) -> bool:
        """Check if the signal event is still active.
        
        Args:
            current_date: Current date to check against
            
        Returns:
            bool: True if the signal is still active, False otherwise
        """
        return current_date <= self.expiration_date
        
    def reset_expiration(self, new_activation_date: datetime) -> None:
        """Reset the expiration date when the signal reactivates.
        
        Args:
            new_activation_date: New activation date to reset from
        """
        self.logger.debug(
            f"Resetting expiration for {self.signal_name} on {self.asset_id} "
            f"from {self.expiration_date} to {new_activation_date + timedelta(days=self.decay)}"
        )
        self.activation_date = new_activation_date
        self.expiration_date = new_activation_date + timedelta(days=self.decay)
        
    def get_remaining_days(self, current_date: datetime) -> int:
        """Get the number of days remaining until expiration.
        
        Args:
            current_date: Current date
            
        Returns:
            int: Number of days until expiration (0 if expired)
        """
        if not self.is_active(current_date):
            return 0
        
        days_remaining = (self.expiration_date - current_date).days
        return max(0, days_remaining)
        
    def __str__(self) -> str:
        """String representation of the signal event."""
        return (
            f"SignalEvent({self.signal_name}, {self.asset_id}, "
            f"activated={self.activation_date.strftime('%Y-%m-%d')}, "
            f"expires={self.expiration_date.strftime('%Y-%m-%d')}, "
            f"weight={self.weight:.4f})"
        ) 