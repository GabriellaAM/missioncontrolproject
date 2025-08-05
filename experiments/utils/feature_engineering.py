"""
Feature Engineering Utilities
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def calculate_log_returns(df: pd.DataFrame, 
                          assets: Optional[List[str]] = None,
                          periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
    """
    Calculate log returns for asset close price columns.
    
    Args:
        df: Input dataframe
        assets: List of asset names (e.g., ['bitcoin', 'ethereum']). If None, detect from columns.
        periods: List of return periods to calculate
    
    Returns:
        DataFrame with added log return columns
    """
    df = df.copy()
    
    # Detect assets if not provided
    if assets is None:
        # Find all columns ending with _close to identify assets
        close_cols = [col for col in df.columns if col.endswith('_close')]
        assets = [col.replace('_close', '') for col in close_cols]
    
    for asset in assets:
        close_col = f"{asset}_close"
        if close_col in df.columns:
            for period in periods:
                return_col = f"{asset}_log_return_{period}"
                df[return_col] = np.log(df[close_col] / df[close_col].shift(period))
    
    return df


def shift_features_for_prediction(df: pd.DataFrame, 
                                 feature_cols: List[str], 
                                 target_cols: List[str], 
                                 shift_periods: int = 1) -> pd.DataFrame:
    """
    Shift feature columns backward in time to prevent look-ahead bias.
    Target columns remain unshifted for correct label alignment.
    
    Args:
        df: Input dataframe containing features and targets
        feature_cols: List of feature column names to shift
        target_cols: List of target column names (will not be shifted)
        shift_periods: Number of periods to shift features backward (default is 1)
    
    Returns:
        DataFrame with shifted features
    """
    df = df.copy()
    
    # Shift all features back to prevent look-ahead bias
    for col in feature_cols:
        if col not in target_cols:
            df[col] = df[col].shift(shift_periods)
    
    return df