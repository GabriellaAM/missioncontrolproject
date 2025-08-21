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
    volatility_span: int = 20,
    time_barrier_days: int = 5,
    upper_barrier_mult: float = 2.0,
    lower_barrier_mult: float = 2.0,
    min_pct_move: Optional[float] = None
) -> pd.DataFrame:
    """
    Apply triple barrier labeling method for establishing ground truth.
    
    The triple barrier method labels each observation based on which barrier is touched first:
    - Upper barrier (profit target): Label = 1 (profitable)
    - Lower barrier (stop loss): Label = 0 (unprofitable)
    - Time barrier (max holding period): Label = 1 if positive return, 0 otherwise (unprofitable)
    
    Note: Labels are binary (0/1) to avoid directional bias for meta-model training.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (typically close prices)
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
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # Calculate EWMA volatility
    volatility = calculate_ewma_volatility(log_returns, span=volatility_span)
    
    # Initialize result arrays
    n = len(prices)
    labels = np.zeros(n)
    barrier_touched = [''] * n
    days_to_barrier = np.zeros(n)
    return_at_barrier = np.zeros(n)
    
    # Process each observation
    for i in range(n - 1):
        if pd.isna(volatility.iloc[i]):
            continue
            
        # Current price and volatility
        current_price = prices.iloc[i]
        current_vol = volatility.iloc[i]
        
        # Calculate barrier levels
        upper_barrier = current_price * np.exp(upper_barrier_mult * current_vol)
        lower_barrier = current_price * np.exp(-lower_barrier_mult * current_vol)
        
        # Check each future price up to time barrier
        max_days = min(time_barrier_days, n - i - 1)
        
        for j in range(1, max_days + 1):
            future_price = prices.iloc[i + j]
            
            # Check upper barrier
            if future_price >= upper_barrier:
                labels[i] = 1
                barrier_touched[i] = 'upper'
                days_to_barrier[i] = j
                return_at_barrier[i] = np.log(future_price / current_price)
                break
                
            # Check lower barrier  
            elif future_price <= lower_barrier:
                labels[i] = 0
                barrier_touched[i] = 'lower'
                days_to_barrier[i] = j
                return_at_barrier[i] = np.log(future_price / current_price)
                break
                
            # Time barrier reached
            elif j == max_days:
                ret = np.log(future_price / current_price)
                return_at_barrier[i] = ret
                days_to_barrier[i] = j
                barrier_touched[i] = 'time'
                
                # Assign label based on return at time barrier
                if min_pct_move is not None:
                    if ret > min_pct_move / 100:
                        labels[i] = 1
                    else:
                        labels[i] = 0
                else:
                    if ret > 0:
                        labels[i] = 1
                    else:
                        labels[i] = 0
    
    # Create result DataFrame with rounded returns
    result = pd.DataFrame({
        'label': labels,
        'barrier_touched': barrier_touched,
        'days_to_barrier': days_to_barrier,
        'return_at_barrier': np.round(return_at_barrier, 4),
        'volatility': np.round(volatility, 4)
    }, index=prices.index)
    
    return result


def add_triple_barrier_labels(
    data: pd.DataFrame,
    price_col: str,
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
        Original DataFrame with added triple barrier columns
    """
    # Apply triple barrier labeling
    barrier_results = triple_barrier_label(
        prices=data[price_col],
        volatility_span=volatility_span,
        time_barrier_days=time_barrier_days,
        upper_barrier_mult=upper_barrier_mult,
        lower_barrier_mult=lower_barrier_mult,
        min_pct_move=min_pct_move
    )
    
    # Add results to original DataFrame
    result = data.copy()
    result['label'] = barrier_results['label']
    result['barrier_touched'] = barrier_results['barrier_touched']
    result['days_to_barrier'] = barrier_results['days_to_barrier']
    result['return_at_barrier'] = barrier_results['return_at_barrier']
    result['volatility'] = barrier_results['volatility']
    
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