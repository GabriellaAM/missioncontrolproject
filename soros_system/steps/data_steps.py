"""
ZenML data loading and preprocessing steps.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from zenml import step
import logging

from ..core.asset_data import AssetData
from ..data.data_loader import DataLoader


@step
def load_asset_data(
    asset_id: str,
    data_dir: str = 'data',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load asset price data for training/validation.
    
    Args:
        asset_id: Asset identifier
        data_dir: Base data directory
        start_date: Start date for data loading (YYYY-MM-DD)
        end_date: End date for data loading (YYYY-MM-DD)
        
    Returns:
        Asset price DataFrame with OHLCV columns
    """
    loader = DataLoader(data_dir=data_dir)
    
    # Load candle data
    candle_data = loader.load_candle_data([asset_id])
    asset_data = candle_data[asset_id]
    
    # Filter by date range if specified
    if start_date:
        asset_data = asset_data[asset_data.index >= start_date]
    if end_date:
        asset_data = asset_data[asset_data.index <= end_date]
    
    logging.info(f"Loaded {len(asset_data)} records for {asset_id}")
    
    return asset_data


@step
def prepare_training_data(
    asset_data: pd.DataFrame,
    train_split: float = 0.7,
    val_split: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets using time-based splits.
    
    Args:
        asset_data: Asset price data
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n_samples = len(asset_data)
    train_size = int(n_samples * train_split)
    val_size = int(n_samples * val_split)
    
    train_data = asset_data.iloc[:train_size]
    val_data = asset_data.iloc[train_size:train_size + val_size]
    test_data = asset_data.iloc[train_size + val_size:]
    
    logging.info(f"Split data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data


@step
def calculate_returns(
    price_data: pd.DataFrame,
    price_column: str = 'close',
    periods: List[int] = [1, 5, 10, 20]
) -> pd.DataFrame:
    """
    Calculate forward returns for different holding periods.
    
    Args:
        price_data: Price DataFrame
        price_column: Column to use for return calculation
        periods: List of periods for forward returns
        
    Returns:
        DataFrame with forward returns
    """
    returns_df = pd.DataFrame(index=price_data.index)
    prices = price_data[price_column]
    
    for period in periods:
        returns_df[f'forward_return_{period}d'] = (
            prices.shift(-period) / prices - 1
        )
    
    logging.info(f"Calculated forward returns for periods: {periods}")
    
    return returns_df


@step
def load_market_regime_data(
    data_dir: str = 'data',
    regime_indicators: List[str] = ['vix', 'treasury10Y', 'dxy']
) -> pd.DataFrame:
    """
    Load market regime indicators for meta-model features.
    
    Args:
        data_dir: Base data directory
        regime_indicators: List of macro indicators to load
        
    Returns:
        DataFrame with market regime features
    """
    loader = DataLoader(data_dir=data_dir)
    regime_data = pd.DataFrame()
    
    try:
        macro_data = loader.load_macro_data()
        
        for indicator in regime_indicators:
            if indicator in macro_data:
                regime_data[indicator] = macro_data[indicator]
                
                # Add momentum features
                regime_data[f'{indicator}_momentum_5d'] = (
                    macro_data[indicator].pct_change(5)
                )
                regime_data[f'{indicator}_momentum_20d'] = (
                    macro_data[indicator].pct_change(20)
                )
        
        logging.info(f"Loaded market regime data with {len(regime_data.columns)} features")
        
    except Exception as e:
        logging.warning(f"Failed to load market regime data: {e}")
        regime_data = pd.DataFrame()
    
    return regime_data