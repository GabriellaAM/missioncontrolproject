"""
Triple-barrier labeling implementation based on López de Prado's methodology.

This module implements the triple-barrier labeling method for creating
meaningful training labels that account for:
1. Profit-taking barrier (upper limit)
2. Stop-loss barrier (lower limit) 
3. Time-based barrier (maximum holding period)
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


class TripleBarrierLabeler:
    """
    Implements triple-barrier labeling for financial time series.
    
    This creates labels based on which barrier is hit first:
    - +1: Price hits profit-taking barrier first
    - -1: Price hits stop-loss barrier first  
    - 0: Time barrier hit first (no clear direction)
    """
    
    def __init__(
        self,
        profit_taking_multiple: float = 2.0,
        stop_loss_multiple: float = 1.0,
        max_holding_period: int = 5,
        min_return_threshold: float = 0.001,
        num_workers: Optional[int] = None
    ):
        """
        Initialize the triple-barrier labeler.
        
        Args:
            profit_taking_multiple: Multiple of volatility for profit barrier
            stop_loss_multiple: Multiple of volatility for stop-loss barrier
            max_holding_period: Maximum holding period in days
            min_return_threshold: Minimum return threshold for labeling
            num_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.profit_taking_multiple = profit_taking_multiple
        self.stop_loss_multiple = stop_loss_multiple
        self.max_holding_period = max_holding_period
        self.min_return_threshold = min_return_threshold
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
    
    def calculate_barriers(
        self,
        prices: pd.Series,
        volatility: pd.Series,
        events: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Calculate dynamic profit-taking and stop-loss barriers.
        
        Args:
            prices: Price series
            volatility: Volatility series (e.g., rolling std)
            events: Event timestamps for barrier calculation
            
        Returns:
            DataFrame with columns: ['entry_price', 'profit_barrier', 'stop_barrier', 'time_barrier']
        """
        barriers = pd.DataFrame(index=events)
        
        # Get entry prices at event times
        barriers['entry_price'] = prices.reindex(events, method='ffill')
        
        # Get volatility at event times
        vol_at_events = volatility.reindex(events, method='ffill')
        
        # Calculate dynamic barriers based on volatility
        barriers['profit_barrier'] = (
            barriers['entry_price'] * (1 + self.profit_taking_multiple * vol_at_events)
        )
        barriers['stop_barrier'] = (
            barriers['entry_price'] * (1 - self.stop_loss_multiple * vol_at_events)
        )
        
        # Time barrier (end of holding period)
        barriers['time_barrier'] = events + pd.Timedelta(days=self.max_holding_period)
        
        return barriers
    
    def _label_single_event(
        self,
        event_idx: int,
        prices: pd.Series,
        barriers_row: pd.Series
    ) -> Tuple[int, int, float, str]:
        """
        Label a single event using triple-barrier method.
        
        Returns:
            Tuple of (event_index, label, return, barrier_hit)
        """
        entry_time = barriers_row.name
        entry_price = barriers_row['entry_price']
        profit_barrier = barriers_row['profit_barrier']
        stop_barrier = barriers_row['stop_barrier']
        time_barrier = barriers_row['time_barrier']
        
        # Get future prices until time barrier
        future_prices = prices.loc[entry_time:time_barrier].iloc[1:]  # Exclude entry price
        
        if future_prices.empty:
            return event_idx, 0, 0.0, 'no_data'
        
        # Check which barrier is hit first
        for timestamp, price in future_prices.items():
            if price >= profit_barrier:
                return_pct = (price - entry_price) / entry_price
                return event_idx, 1, return_pct, 'profit'
            elif price <= stop_barrier:
                return_pct = (price - entry_price) / entry_price
                return event_idx, -1, return_pct, 'stop_loss'
        
        # If no barrier hit, use final price at time barrier
        final_price = future_prices.iloc[-1]
        return_pct = (final_price - entry_price) / entry_price
        
        # Label based on return magnitude vs threshold
        if abs(return_pct) < self.min_return_threshold:
            label = 0  # Neutral
        else:
            label = 1 if return_pct > 0 else -1
            
        return event_idx, label, return_pct, 'time'
    
    def label_events(
        self,
        prices: pd.Series,
        events: pd.DatetimeIndex,
        volatility: Optional[pd.Series] = None,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Apply triple-barrier labeling to a series of events.
        
        Args:
            prices: Price series
            events: Event timestamps
            volatility: Volatility series (if None, uses rolling 20-day std)
            parallel: Whether to use parallel processing
            
        Returns:
            DataFrame with columns: ['label', 'return', 'barrier_hit', 'entry_price']
        """
        if volatility is None:
            volatility = prices.rolling(window=20, min_periods=10).std()
        
        # Calculate barriers for all events
        barriers = self.calculate_barriers(prices, volatility, events)
        
        # Initialize results
        results = []
        
        if parallel and len(events) > 100:
            # Parallel processing for large datasets
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}
                
                for idx, (event_time, barrier_row) in enumerate(barriers.iterrows()):
                    future = executor.submit(
                        self._label_single_event,
                        idx,
                        prices,
                        barrier_row
                    )
                    futures[future] = idx
                
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            # Sequential processing
            for idx, (event_time, barrier_row) in enumerate(barriers.iterrows()):
                result = self._label_single_event(idx, prices, barrier_row)
                results.append(result)
        
        # Sort results by event index and create DataFrame
        results.sort(key=lambda x: x[0])
        
        labels_df = pd.DataFrame([
            {
                'event_time': events[r[0]],
                'label': r[1],
                'return': r[2],
                'barrier_hit': r[3],
                'entry_price': barriers.iloc[r[0]]['entry_price']
            }
            for r in results
        ])
        
        labels_df.set_index('event_time', inplace=True)
        return labels_df
    
    def create_meta_labels(
        self,
        primary_predictions: pd.Series,
        actual_labels: pd.Series,
        market_features: pd.DataFrame,
        prediction_threshold: float = 0.55
    ) -> pd.DataFrame:
        """
        Create meta-labels for the meta-model training.
        
        The meta-model learns to predict the probability that the primary model
        prediction is correct, enabling intelligent position sizing.
        
        Args:
            primary_predictions: Primary model predictions (probabilities or signals)
            actual_labels: Actual triple-barrier labels
            market_features: Market features for meta-model training
            prediction_threshold: Threshold for considering primary prediction confident
            
        Returns:
            DataFrame with meta-labels and features for meta-model training
        """
        # Align all data on common index
        common_index = primary_predictions.index.intersection(
            actual_labels.index
        ).intersection(market_features.index)
        
        primary_aligned = primary_predictions.reindex(common_index)
        labels_aligned = actual_labels.reindex(common_index)
        features_aligned = market_features.reindex(common_index)
        
        # Create binary predictions from primary model
        primary_binary = np.where(primary_aligned > prediction_threshold, 1,
                                 np.where(primary_aligned < -prediction_threshold, -1, 0))
        
        # Meta-label: 1 if primary prediction matches actual label, 0 otherwise
        meta_labels = (primary_binary == labels_aligned).astype(int)
        
        # Create meta-training dataset
        meta_data = features_aligned.copy()
        meta_data['primary_prediction'] = primary_aligned
        meta_data['primary_confidence'] = np.abs(primary_aligned)
        meta_data['meta_label'] = meta_labels
        meta_data['actual_label'] = labels_aligned
        
        # Only use samples where primary model made a confident prediction
        confident_mask = np.abs(primary_aligned) > prediction_threshold
        meta_data_filtered = meta_data[confident_mask]
        
        return meta_data_filtered


def apply_triple_barrier_labeling(
    price_data: pd.Series,
    signal_events: pd.DatetimeIndex,
    profit_multiple: float = 2.0,
    stop_multiple: float = 1.0,
    holding_period: int = 5
) -> pd.DataFrame:
    """
    Convenience function to apply triple-barrier labeling.
    
    Args:
        price_data: Price time series
        signal_events: Timestamps of signal events
        profit_multiple: Profit-taking barrier multiple
        stop_multiple: Stop-loss barrier multiple
        holding_period: Maximum holding period in days
        
    Returns:
        Labeled events DataFrame
    """
    labeler = TripleBarrierLabeler(
        profit_taking_multiple=profit_multiple,
        stop_loss_multiple=stop_multiple,
        max_holding_period=holding_period
    )
    
    return labeler.label_events(price_data, signal_events)