"""
Signal event tracker for the Soros System.

This module provides tools for tracking and managing signal activation events
across multiple assets, implementing event-driven signal logic.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

from .signal_event import SignalEvent


class SignalEventTracker:
    """
    Tracks all signal activation events across assets.
    
    Manages the lifecycle of signal events, including activation, expiration,
    and reset. Calculates the combined weight of all active signals daily.
    """
    
    def __init__(self, threshold: float = 0.0):
        """Initialize the signal event tracker.
        
        Args:
            threshold: Threshold for final decision (default: 0.0)
        """
        self.logger = logging.getLogger(__name__)
        self.active_events = {}  # {asset_id: {signal_name: SignalEvent}}
        self.event_history = []  # List of all events for analysis
        self.rejected_events = []  # List of rejected events (meta-labeling)
        self.threshold = threshold
        
    def register_signal_event(
        self, 
        signal_name: str, 
        asset_id: str,
        activation_date: datetime,
        holding_period: int,
        weight: float,
        meta_approved: bool = True
    ) -> bool:
        """Register a new signal event or reset an existing one.
        
        Args:
            signal_name: Name of the signal
            asset_id: ID of the asset
            activation_date: Date when the signal activated
            holding_period: Number of days to hold the signal
            weight: Weight of the signal based on its effectiveness
            meta_approved: Whether this signal was approved by meta-labeling
            
        Returns:
            bool: True if the event was registered, False if rejected by meta-labeling
        """
        # Check meta-approval first
        if not meta_approved:
            # Log rejection
            rejection_info = {
                'type': 'rejection',
                'signal_name': signal_name,
                'asset_id': asset_id,
                'date': activation_date,
                'holding_period': holding_period,
                'weight': weight,
                'reason': 'meta_labeling'
            }
            self.rejected_events.append(rejection_info)
            
            self.logger.debug(
                f"Rejected by meta-labeling: {signal_name} for {asset_id} "
                f"on {activation_date}, weight: {weight:.4f}"
            )
            return False
            
        # Initialize asset dict if needed
        if asset_id not in self.active_events:
            self.active_events[asset_id] = {}
            
        # Create event
        event = SignalEvent(
            signal_name=signal_name,
            asset_id=asset_id,
            activation_date=activation_date,
            holding_period=holding_period,
            weight=weight,
            meta_approved=meta_approved
        )
        
        # If signal already exists, reset its expiration
        if signal_name in self.active_events[asset_id]:
            existing_event = self.active_events[asset_id][signal_name]
            existing_event.reset_expiration(activation_date)
            # Log the reset
            self.event_history.append({
                'type': 'reset',
                'signal_name': signal_name,
                'asset_id': asset_id,
                'date': activation_date,
                'new_expiration': existing_event.expiration_date,
                'weight': existing_event.weight,
                'meta_approved': existing_event.meta_approved
            })
            self.logger.debug(
                f"Reset signal event: {signal_name} for {asset_id} "
                f"on {activation_date}, new expiration: {existing_event.expiration_date}"
            )
        else:
            # Add new event
            self.active_events[asset_id][signal_name] = event
            # Log the activation
            self.event_history.append({
                'type': 'activation',
                'signal_name': signal_name,
                'asset_id': asset_id,
                'date': activation_date,
                'expiration': event.expiration_date,
                'weight': event.weight,
                'meta_approved': event.meta_approved
            })
            self.logger.debug(
                f"New signal event: {signal_name} for {asset_id} "
                f"on {activation_date}, expiration: {event.expiration_date}, "
                f"weight: {event.weight:.4f}"
            )
            
        return True
            
    def get_active_signals(self, asset_id: str, current_date: datetime) -> Dict[str, SignalEvent]:
        """Get all active signal events for an asset on a given date.
        
        Args:
            asset_id: ID of the asset
            current_date: Current date to check against
            
        Returns:
            dict: Dictionary mapping signal names to active SignalEvent objects
        """
        if asset_id not in self.active_events:
            return {}
            
        # Filter active events
        active = {}
        for signal_name, event in self.active_events[asset_id].items():
            if event.is_active(current_date) and event.meta_approved:
                active[signal_name] = event
                
        # Remove expired events
        for signal_name in list(self.active_events[asset_id].keys()):
            event = self.active_events[asset_id][signal_name]
            if not event.is_active(current_date):
                # Log expiration
                self.event_history.append({
                    'type': 'expiration',
                    'signal_name': signal_name,
                    'asset_id': asset_id,
                    'date': current_date,
                    'weight': event.weight
                })
                self.logger.debug(
                    f"Signal expired: {signal_name} for {asset_id} on {current_date}"
                )
                # Remove from active events
                del self.active_events[asset_id][signal_name]
                
        return active
        
    def calculate_combined_weight(self, asset_id: str, current_date: datetime) -> float:
        """Calculate the combined weight of all active signals for an asset.
        
        Weights are normalized across all active signals for this asset
        before being summed.
        
        Args:
            asset_id: ID of the asset
            current_date: Current date to check against
            
        Returns:
            float: Normalized combined weight of all active signals
        """
        active_signals = self.get_active_signals(asset_id, current_date)
        
        if not active_signals:
            return 0.0
            
        # First normalize weights across all active signals for this asset
        total_abs_weight = sum(abs(event.weight) for event in active_signals.values())
        
        if total_abs_weight == 0:
            return 0.0
        
        # Normalize each weight and then sum
        normalized_sum = sum(
            event.weight / total_abs_weight 
            for event in active_signals.values()
        )
        
        self.logger.debug(
            f"Asset {asset_id} on {current_date}: {len(active_signals)} active signals, "
            f"combined weight: {normalized_sum:.4f}"
        )
        
        return normalized_sum
        
    def get_final_decision(self, asset_id: str, current_date: datetime) -> int:
        """Get final decision (1 for exposed, 0 for cash) based on combined weight.
        
        Args:
            asset_id: ID of the asset
            current_date: Current date
            
        Returns:
            int: 1 if combined weight > threshold, 0 otherwise
        """
        combined_weight = self.calculate_combined_weight(asset_id, current_date)
        
        # 1 if positive weight above threshold, 0 otherwise
        decision = 1 if combined_weight > self.threshold else 0
        
        if decision == 1:
            self.logger.debug(
                f"Decision for {asset_id} on {current_date}: EXPOSED (1), "
                f"weight: {combined_weight:.4f}"
            )
        else:
            self.logger.debug(
                f"Decision for {asset_id} on {current_date}: CASH (0), "
                f"weight: {combined_weight:.4f}"
            )
        
        return decision
    
    def get_event_history_df(self) -> pd.DataFrame:
        """Get event history as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with event history
        """
        return pd.DataFrame(self.event_history)
    
    def get_rejected_events_df(self) -> pd.DataFrame:
        """Get rejected events as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with rejected events
        """
        return pd.DataFrame(self.rejected_events)
    
    def get_rejection_stats(self) -> Dict[str, Any]:
        """Get statistics about rejected events.
        
        Returns:
            dict: Dictionary with rejection statistics
        """
        if not self.rejected_events:
            return {
                'total_rejections': 0,
                'rejections_by_asset': {},
                'rejections_by_signal': {}
            }
            
        df = self.get_rejected_events_df()
        
        # Count rejections by asset
        asset_counts = df['asset_id'].value_counts().to_dict()
        
        # Count rejections by signal
        signal_counts = df['signal_name'].value_counts().to_dict()
        
        # Count rejections by reason if exists
        reason_counts = {}
        if 'reason' in df.columns:
            reason_counts = df['reason'].value_counts().to_dict()
        
        return {
            'total_rejections': len(df),
            'rejections_by_asset': asset_counts,
            'rejections_by_signal': signal_counts,
            'rejections_by_reason': reason_counts
        }
    
    def set_threshold(self, threshold: float) -> None:
        """Set threshold for final decision.
        
        Args:
            threshold: New threshold value
        """
        self.threshold = threshold
        self.logger.info(f"Set threshold to {threshold}")
    
    def clear_active_events(self, asset_id: Optional[str] = None):
        """Clear active events for a specific asset or all assets.
        
        Args:
            asset_id: ID of the asset to clear events for. If None, clear all.
        """
        if asset_id is None:
            self.active_events = {}
            self.logger.info("Cleared all active events")
        elif asset_id in self.active_events:
            self.active_events[asset_id] = {}
            self.logger.info(f"Cleared active events for {asset_id}")
    
    def clear_history(self):
        """Clear event history."""
        self.event_history = []
        self.rejected_events = []
        self.logger.info("Cleared event history and rejected events") 