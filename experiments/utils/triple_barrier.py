#!/usr/bin/env python

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


def calculate_ewma_volatility(
    returns: pd.Series,
    span: int = 20,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate exponentially weighted moving average volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Log returns series
    span : int, default 20
        Span for EWMA calculation
    min_periods : int, optional
        Minimum number of observations required for calculation
        
    Returns
    -------
    pd.Series
        EWMA volatility series
    """
    if min_periods is None:
        min_periods = span // 2
    
    return returns.ewm(span=span, min_periods=min_periods).std()


def triple_barrier_label(
    prices: pd.Series,
    events: Optional[pd.DatetimeIndex] = None,
    signals: Optional[pd.Series] = None,
    volatility_span: int = 20,
    time_barrier_days: int = 5,
    upper_barrier_mult: float = 2.0,
    lower_barrier_mult: float = 2.0,
    min_pct_move: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply triple barrier labeling method for establishing ground truth.
    
    The triple barrier method labels events based on which barrier is touched first,
    considering the direction of the strategy signal:
    
    For LONG signals (signal = 1):
    - Upper barrier hit: Label = 1 (profitable - price went up as expected)
    - Lower barrier hit: Label = 0 (unprofitable - price went down)
    - Time barrier: Label = 1 if return > 0, else 0
    
    For SHORT signals (signal = -1):
    - Upper barrier hit: Label = 0 (unprofitable - price went up against short)
    - Lower barrier hit: Label = 1 (profitable - price went down as expected)
    - Time barrier: Label = 1 if return < 0, else 0
    
    Note: Labels are binary (0/1) reflecting actual profitability given position direction.
    IMPORTANT: Only applies labeling to specified events, not every price point.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically close prices)
    events : pd.DatetimeIndex, optional
        Timestamps where strategy signals occur. If None, uses all price timestamps.
    signals : pd.Series, optional
        Strategy signals (-1=short, 0=neutral, 1=long) indexed by event timestamps.
        If None, assumes all events are long positions (signal=1).
    volatility_span : int, default 20
        Span for EWMA volatility calculation
    time_barrier_days : int, default 5
        Maximum holding period in days
    upper_barrier_mult : float, default 2.0
        Multiplier for upper barrier as multiple of volatility
    lower_barrier_mult : float, default 2.0
        Multiplier for lower barrier as multiple of volatility
    min_pct_move : float, optional
        Minimum percentage move to assign non-zero label at time barrier
        If None, any positive/negative move gets labeled based on signal direction
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - label: Binary profitability label (0=unprofitable, 1=profitable) 
                considering signal direction
        - barrier_touched: Which barrier was touched ('upper', 'lower', 'time')
        - days_to_barrier: Number of days until barrier was touched
        - return_at_barrier: Return at the point barrier was touched
        - volatility: EWMA volatility used for barriers
        - signal: Strategy signal used for labeling
    """
    # If no events specified, use all price timestamps (backward compatibility)
    if events is None:
        events = prices.index[:-1]  # Exclude last point as we need future prices
    
    # If signals provided, filter events to only include non-zero signals
    if signals is not None:
        # Validate signals are properly aligned with price index
        signals_missing = ~signals.index.isin(prices.index)
        if signals_missing.any():
            print(f"Warning: {signals_missing.sum()} signal timestamps not found in price data")
        
        # Only process events where we have non-zero signals
        event_signals = signals.reindex(events).fillna(0)
        valid_signal_events = event_signals != 0
        events = events[valid_signal_events]
        
        # Additional validation
        if len(events) == 0:
            print("Warning: No events with non-zero signals found")
            return pd.DataFrame(columns=['label', 'barrier_touched', 'days_to_barrier', 
                                       'return_at_barrier', 'volatility', 'signal'])
        
        print(f"Filtered to {len(events)} events with non-zero signals")
        print(f"Signal distribution: Long={np.sum(signals[signals.index.isin(events)] == 1)}, Short={np.sum(signals[signals.index.isin(events)] == -1)}")
    else:
        # If no signals provided, assume all events are long positions (backward compatibility)
        event_signals = pd.Series(1, index=events)
        print(f"No signals provided, treating all {len(events)} events as long positions")
    
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # Calculate EWMA volatility
    volatility = calculate_ewma_volatility(log_returns, span=volatility_span)
    
    # Initialize result lists
    event_labels = []
    event_barriers = []
    event_days = []
    event_returns = []
    event_volatilities = []
    event_signals = []
    valid_events = []
    
    # Process each event timestamp
    for event_time in events:
        if event_time not in prices.index:
            continue
            
        event_idx = prices.index.get_loc(event_time)
        
        # Skip if volatility is NaN or if we're at the end of the series
        if pd.isna(volatility.iloc[event_idx]) or event_idx >= len(prices) - 1:
            continue
            
        # Get signal for this event
        if signals is not None:
            current_signal = signals.loc[event_time] if event_time in signals.index else 0
        else:
            current_signal = 1  # Default to long position
            
        # Skip neutral signals
        if current_signal == 0:
            continue
            
        # Current price and volatility at event
        current_price = prices.iloc[event_idx]
        current_vol = volatility.iloc[event_idx]
        
        # Calculate barrier levels
        upper_barrier = current_price * np.exp(upper_barrier_mult * current_vol)
        lower_barrier = current_price * np.exp(-lower_barrier_mult * current_vol)
        
        # Check each future price up to time barrier
        max_days = min(time_barrier_days, len(prices) - event_idx - 1)
        label_assigned = False
        
        for j in range(1, max_days + 1):
            if event_idx + j >= len(prices):
                break
                
            future_price = prices.iloc[event_idx + j]
            
            # Check upper barrier
            if future_price >= upper_barrier:
                # Label based on signal direction:
                # Long (1): Upper barrier hit = profit (1)
                # Short (-1): Upper barrier hit = loss (0)
                label = 1 if current_signal == 1 else 0
                event_labels.append(label)
                event_barriers.append('upper')
                event_days.append(j)
                event_returns.append(np.log(future_price / current_price))
                event_volatilities.append(current_vol)
                event_signals.append(current_signal)
                valid_events.append(event_time)
                label_assigned = True
                break
                
            # Check lower barrier  
            elif future_price <= lower_barrier:
                # Label based on signal direction:
                # Long (1): Lower barrier hit = loss (0)
                # Short (-1): Lower barrier hit = profit (1)
                label = 0 if current_signal == 1 else 1
                event_labels.append(label)
                event_barriers.append('lower')
                event_days.append(j)
                event_returns.append(np.log(future_price / current_price))
                event_volatilities.append(current_vol)
                event_signals.append(current_signal)
                valid_events.append(event_time)
                label_assigned = True
                break
        
        # If no barrier hit and we have future prices, assign time barrier label
        if not label_assigned and max_days > 0:
            future_price = prices.iloc[event_idx + max_days]
            ret = np.log(future_price / current_price)
            
            # Assign label based on return at time barrier and signal direction
            if min_pct_move is not None:
                # Use minimum percentage move threshold
                if current_signal == 1:  # Long position
                    label = 1 if ret > min_pct_move / 100 else 0
                else:  # Short position (signal == -1)
                    label = 1 if ret < -min_pct_move / 100 else 0
            else:
                # No threshold - any profitable move
                if current_signal == 1:  # Long position
                    label = 1 if ret > 0 else 0
                else:  # Short position (signal == -1)
                    label = 1 if ret < 0 else 0
                
            event_labels.append(label)
            event_barriers.append('time')
            event_days.append(max_days)
            event_returns.append(ret)
            event_volatilities.append(current_vol)
            event_signals.append(current_signal)
            valid_events.append(event_time)
    
    # Create result DataFrame with only labeled events
    result = pd.DataFrame({
        'label': event_labels,
        'barrier_touched': event_barriers,
        'days_to_barrier': event_days,
        'return_at_barrier': np.round(event_returns, 4),
        'volatility': np.round(event_volatilities, 4),
        'signal': event_signals
    }, index=valid_events)
    
    return result


def add_triple_barrier_labels(
    data: pd.DataFrame,
    price_col: str,
    events: Optional[pd.DatetimeIndex] = None,
    signals: Optional[pd.Series] = None,
    volatility_span: int = 20,
    time_barrier_days: int = 5,
    upper_barrier_mult: float = 2.0,
    lower_barrier_mult: float = 2.0,
    min_pct_move: Optional[float] = None
) -> pd.DataFrame:
    """
    Add triple barrier labels to a DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing price data
    price_col : str
        Name of the price column
    events : pd.DatetimeIndex, optional
        Timestamps where strategy signals occur. If None, attempts to extract
        from 'signal' column in data, or uses all timestamps.
    signals : pd.Series, optional
        Strategy signals (-1=short, 0=neutral, 1=long). If None, attempts to
        extract from 'signal' column in data, or assumes all events are long.
    volatility_span : int, default 20
        Span for EWMA volatility calculation
    time_barrier_days : int, default 5
        Maximum holding period in days
    upper_barrier_mult : float, default 2.0
        Multiplier for upper barrier as multiple of volatility
    lower_barrier_mult : float, default 2.0
        Multiplier for lower barrier as multiple of volatility
    min_pct_move : float, optional
        Minimum percentage move to assign non-zero label at time barrier
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added triple barrier columns (only for event timestamps)
    """
    # If signals not provided, try to extract from signal column
    if signals is None and 'signal' in data.columns:
        signals = data['signal']
        print(f"Extracted signals from 'signal' column in data")
    
    # If events not provided, try to extract from signal column
    if events is None and signals is not None:
        # Extract timestamps where signal is non-zero (actual trading events)
        events = signals[signals != 0].index
        print(f"Extracted {len(events)} strategy events from signals")
    elif events is None:
        # Fall back to all timestamps if no signals available
        events = data.index
        print(f"Using all {len(events)} timestamps as events")
    
    # Apply triple barrier labeling to events only
    barrier_results = triple_barrier_label(
        prices=data[price_col],
        events=events,
        signals=signals,
        volatility_span=volatility_span,
        time_barrier_days=time_barrier_days,
        upper_barrier_mult=upper_barrier_mult,
        lower_barrier_mult=lower_barrier_mult,
        min_pct_move=min_pct_move
    )
    
    # Create result DataFrame - start with original data
    result = data.copy()
    
    # Initialize label columns with NaN for all rows
    result['label'] = np.nan
    result['barrier_touched'] = ''
    result['days_to_barrier'] = np.nan
    result['return_at_barrier'] = np.nan
    result['volatility'] = np.nan
    result['barrier_signal'] = np.nan  # Rename to avoid conflict with existing signal column
    
    # Only fill in labels for the events that were processed
    if len(barrier_results) > 0:
        common_idx = result.index.intersection(barrier_results.index)
        result.loc[common_idx, 'label'] = barrier_results.loc[common_idx, 'label']
        result.loc[common_idx, 'barrier_touched'] = barrier_results.loc[common_idx, 'barrier_touched']
        result.loc[common_idx, 'days_to_barrier'] = barrier_results.loc[common_idx, 'days_to_barrier']
        result.loc[common_idx, 'return_at_barrier'] = barrier_results.loc[common_idx, 'return_at_barrier']
        result.loc[common_idx, 'volatility'] = barrier_results.loc[common_idx, 'volatility']
        result.loc[common_idx, 'barrier_signal'] = barrier_results.loc[common_idx, 'signal']
        
        print(f"Applied triple barrier labels to {len(common_idx)} events")
    else:
        print("No events were labeled (empty barrier results)")
    
    return result






