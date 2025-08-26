#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
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
    volatility_span: int = 20,
    time_barrier_days: int = 5,
    upper_barrier_mult: float = 2.0,
    lower_barrier_mult: float = 2.0,
    min_pct_move: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply triple barrier labeling method for establishing ground truth.
    
    The triple barrier method labels events based on which barrier is touched first:
    - Upper barrier (profit target): Label = 1 (profitable)
    - Lower barrier (stop loss): Label = 0 (unprofitable)
    - Time barrier (max holding period): Label = 1 if positive return, 0 otherwise (unprofitable)
    
    Note: Labels are binary (0/1) to avoid directional bias for meta-model training.
    IMPORTANT: Only applies labeling to specified events, not every price point.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically close prices)
    events : pd.DatetimeIndex, optional
        Timestamps where strategy signals occur. If None, uses all price timestamps.
    volatility_span : int, default 20
        Span for EWMA volatility calculation
    time_barrier_days : int, default 5
        Maximum holding period in days
    upper_barrier_mult : float, default 2.0
        Multiplier for upper barrier (profit target) as multiple of volatility
    lower_barrier_mult : float, default 2.0
        Multiplier for lower barrier (stop loss) as multiple of volatility
    min_pct_move : float, optional
        Minimum percentage move to assign non-zero label at time barrier
        If None, any positive/negative move gets labeled
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - label: Binary profitability label (0=unprofitable, 1=profitable)
        - barrier_touched: Which barrier was touched ('upper', 'lower', 'time')
        - days_to_barrier: Number of days until barrier was touched
        - return_at_barrier: Return at the point barrier was touched
        - volatility: EWMA volatility used for barriers
    """
    # If no events specified, use all price timestamps (backward compatibility)
    if events is None:
        events = prices.index[:-1]  # Exclude last point as we need future prices
    
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
    valid_events = []
    
    # Process each event timestamp
    for event_time in events:
        if event_time not in prices.index:
            continue
            
        event_idx = prices.index.get_loc(event_time)
        
        # Skip if volatility is NaN or if we're at the end of the series
        if pd.isna(volatility.iloc[event_idx]) or event_idx >= len(prices) - 1:
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
                event_labels.append(1)
                event_barriers.append('upper')
                event_days.append(j)
                event_returns.append(np.log(future_price / current_price))
                event_volatilities.append(current_vol)
                valid_events.append(event_time)
                label_assigned = True
                break
                
            # Check lower barrier  
            elif future_price <= lower_barrier:
                event_labels.append(0)
                event_barriers.append('lower')
                event_days.append(j)
                event_returns.append(np.log(future_price / current_price))
                event_volatilities.append(current_vol)
                valid_events.append(event_time)
                label_assigned = True
                break
        
        # If no barrier hit and we have future prices, assign time barrier label
        if not label_assigned and max_days > 0:
            future_price = prices.iloc[event_idx + max_days]
            ret = np.log(future_price / current_price)
            
            # Assign label based on return at time barrier
            if min_pct_move is not None:
                label = 1 if ret > min_pct_move / 100 else 0
            else:
                label = 1 if ret > 0 else 0
                
            event_labels.append(label)
            event_barriers.append('time')
            event_days.append(max_days)
            event_returns.append(ret)
            event_volatilities.append(current_vol)
            valid_events.append(event_time)
    
    # Create result DataFrame with only labeled events
    result = pd.DataFrame({
        'label': event_labels,
        'barrier_touched': event_barriers,
        'days_to_barrier': event_days,
        'return_at_barrier': np.round(event_returns, 4),
        'volatility': np.round(event_volatilities, 4)
    }, index=valid_events)
    
    return result


def add_triple_barrier_labels(
    data: pd.DataFrame,
    price_col: str,
    events: Optional[pd.DatetimeIndex] = None,
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
    # If events not provided, try to extract from signal column
    if events is None and 'signal' in data.columns:
        # Extract timestamps where signal is non-zero (actual trading events)
        events = data[data['signal'] != 0].index
        print(f"Extracted {len(events)} strategy events from signal column")
    
    # Apply triple barrier labeling to events only
    barrier_results = triple_barrier_label(
        prices=data[price_col],
        events=events,
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
    
    # Only fill in labels for the events that were processed
    if len(barrier_results) > 0:
        common_idx = result.index.intersection(barrier_results.index)
        result.loc[common_idx, 'label'] = barrier_results.loc[common_idx, 'label']
        result.loc[common_idx, 'barrier_touched'] = barrier_results.loc[common_idx, 'barrier_touched']
        result.loc[common_idx, 'days_to_barrier'] = barrier_results.loc[common_idx, 'days_to_barrier']
        result.loc[common_idx, 'return_at_barrier'] = barrier_results.loc[common_idx, 'return_at_barrier']
        result.loc[common_idx, 'volatility'] = barrier_results.loc[common_idx, 'volatility']
        
        print(f"Applied triple barrier labels to {len(common_idx)} events")
    else:
        print("No events were labeled (empty barrier results)")
    
    return result


def create_confusion_matrix_plot(
    predictions: np.ndarray,
    labels: np.ndarray,
    title: str = "Triple Barrier Classification",
    save_path: Optional[str] = None
) -> str:
    """
    Create confusion matrix plot for triple barrier classification.
    
    Parameters
    ----------
    predictions : np.ndarray
        Strategy signals (-1, 0, 1)
    labels : np.ndarray
        Triple barrier labels (ground truth)
    title : str
        Plot title
    save_path : str, optional
        Path to save plot, if None creates temp file
        
    Returns
    -------
    str
        Path to saved plot file
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Filter out NaN values
    mask = ~(np.isnan(predictions) | np.isnan(labels))
    clean_predictions = predictions[mask]
    clean_labels = labels[mask]
    
    # For binary classification: 1 = long, 0 = not long (short or neutral)
    binary_predictions = (clean_predictions == 1).astype(int)
    binary_labels = (clean_labels == 1).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(binary_labels, binary_predictions, labels=[0, 1])
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Unprofitable', 'Predicted Profitable'],
                yticklabels=['Actual Unprofitable', 'Actual Profitable'])
    
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Add accuracy to plot
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figtext(0.5, 0.02, f'Total Positions: {np.sum(cm)}\nAccurate Predictions: {np.trace(cm)} ({accuracy:.1%})', 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, "insample_label_confusion_matrix.png")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path




def calculate_metrics_with_labels(
    predictions: np.ndarray,
    labels: np.ndarray,
    returns: Optional[np.ndarray] = None,
    plot_filename: str = "insample_label_confusion_matrix.png"
) -> dict:
    """
    Calculate classification metrics using triple barrier labels as ground truth.
    
    Parameters
    ----------
    predictions : np.ndarray
        Model predictions or strategy signals (-1, 0, 1)
    labels : np.ndarray
        Triple barrier labels (ground truth)
    returns : np.ndarray, optional
        Actual returns for additional metrics
        
    Returns
    -------
    dict
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision: Precision for binary classification
        - recall: Recall for binary classification
        - f1: F1 score for binary classification
        - mcc: Matthews Correlation Coefficient
        - confusion_matrix_plot: Path to confusion matrix plot
    """
    from sklearn.metrics import (precision_score, recall_score, f1_score, 
                               accuracy_score, matthews_corrcoef)
    
    # Filter out NaN values
    mask = ~(np.isnan(predictions) | np.isnan(labels))
    clean_predictions = predictions[mask]
    clean_labels = labels[mask]
    
    if len(clean_predictions) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mcc': 0.0,
            'confusion_matrix_plot': ''
        }
    
    # For binary classification: 1 = long, 0 = not long (short or neutral)
    binary_predictions = (clean_predictions == 1).astype(int)
    binary_labels = (clean_labels == 1).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(binary_labels, binary_predictions)
    precision = precision_score(binary_labels, binary_predictions, zero_division=0)
    recall = recall_score(binary_labels, binary_predictions, zero_division=0)
    f1 = f1_score(binary_labels, binary_predictions, zero_division=0)
    mcc = matthews_corrcoef(binary_labels, binary_predictions)
    
    # Create confusion matrix plot with custom filename
    temp_dir = tempfile.gettempdir()
    save_path = os.path.join(temp_dir, plot_filename)
    cm_plot_path = create_confusion_matrix_plot(clean_predictions, clean_labels, save_path=save_path)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'confusion_matrix_plot': cm_plot_path
    }
    
    # Add returns-based metrics if provided
    if returns is not None and len(returns) == len(clean_predictions):
        clean_returns = returns[mask]
        
        # Calculate returns for each predicted class
        long_returns = clean_returns[clean_predictions == 1]
        short_returns = -clean_returns[clean_predictions == -1]  # Invert for shorts
        
        metrics['avg_return_long'] = round(np.mean(long_returns), 4) if len(long_returns) > 0 else 0
        metrics['avg_return_short'] = round(np.mean(short_returns), 4) if len(short_returns) > 0 else 0
        
        # Hit rate: percentage of profitable trades
        if len(long_returns) > 0:
            metrics['hit_rate_long'] = round(np.sum(long_returns > 0) / len(long_returns), 4)
        else:
            metrics['hit_rate_long'] = 0
            
        if len(short_returns) > 0:
            metrics['hit_rate_short'] = round(np.sum(short_returns > 0) / len(short_returns), 4)
        else:
            metrics['hit_rate_short'] = 0
    
    return metrics