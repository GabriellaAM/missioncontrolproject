"""
EWMA Z-Score Normalization for strategy features.

Simple rolling normalization using exponentially weighted moving averages.
EWMA only uses past data, naturally preventing look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class ColumnNormalizer:
    """Simple EWMA z-score normalization."""
    
    def __init__(self, halflife: int = 90):
        self.halflife = halflife
        self.min_warmup = int(1.5 * halflife)  # 135 days for 90-day halflife
        self.exclude_cols = []
        
    def configure(self, config: Dict):
        """
        Configure which columns to exclude from normalization.
        
        Args:
            config: Dict with 'exclude' list of column names
        """
        self.exclude_cols = config.get('exclude', [])
    
    def get_warmup_days(self) -> int:
        """Return warmup days needed for EWMA normalization."""
        return self.min_warmup
    
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply EWMA z-score normalization to data.
        
        Args:
            data: DataFrame to normalize
            
        Returns:
            DataFrame with normalized columns
        """
        result = data.copy()
        
        for col in data.columns:
            if col in self.exclude_cols:
                continue
            
            # Calculate EWMA mean and std
            ewma_mean = data[col].ewm(
                halflife=self.halflife,
                min_periods=self.min_warmup
            ).mean()
            
            ewma_std = data[col].ewm(
                halflife=self.halflife,
                min_periods=self.min_warmup
            ).std()
            
            # Z-score normalization
            result[col] = (data[col] - ewma_mean) / (ewma_std + 1e-8)
        
        return result