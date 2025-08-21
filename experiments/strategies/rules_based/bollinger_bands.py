import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd
import numpy as np


class BollingerBandsStrategy(BaseStrategy):
    
    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("bollinger_bands")
        self.asset = asset
        self.period_range = [10, 50]
        self.std_dev_range = [1.5, 3.0]
        self.default_params = {'period': 30, 'std_multiplier': 2.0}
    
    def get_required_features(self) -> Dict[str, Any]:
        """Bollinger Bands only needs crypto asset price data."""
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': None,
            'yahoo_tickers': None,
            'calculated_features': None
        }
    
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate Bollinger Bands mean reversion signals."""
        df = data.copy()
        
        # Find close price column with asset prefix
        close_col = f"{self.asset}_close"
        if close_col not in df.columns:
            raise ValueError(f"Required column '{close_col}' not found in data")
        
        # Calculate Moving Average (MA)
        df['ma'] = df[close_col].rolling(window=params['period']).mean()
        
        # Calculate standard deviation (σt)
        df['std'] = df[close_col].rolling(window=params['period']).std()
        
        # Calculate Bollinger Bands
        df['banda_superior'] = df['ma'] + (params['std_multiplier'] * df['std'])
        df['banda_inferior'] = df['ma'] - (params['std_multiplier'] * df['std'])
        
        # Generate mean reversion signals (long-cash strategy)
        # Signal = 1 when current price < previous period's lower band (buy signal - expect mean reversion up)
        # Signal = 0 when current price >= previous period's lower band (cash/hold - no position)
        df['signal'] = np.where(df[close_col] < df['banda_inferior'].shift(1), 1, 0)
        
        return df
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
                 
        """No optimization - returns default parameters."""
        
        return {
            'best_params': self.default_params,
            'best_value': 0.0,
            'study': None
        }