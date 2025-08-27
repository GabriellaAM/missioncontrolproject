import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd


class BollingerBandsStrategy(BaseStrategy):
    
    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("bollinger_bands")
        self.asset = asset
        self.period_range = [10, 50]
        self.std_dev_range = [1.5, 3.0]
        self.default_params = {'period': 20, 'std_multiplier': 2.0}
    
    @property
    def strategy_type(self) -> str:
        """Bollinger Bands is a mean-reversion strategy"""
        return "mean_reversion"
    
    @property
    def implementation_type(self) -> str:
        """Bollinger Bands is a rules-based strategy"""
        return "rules_based"
    
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
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate Bollinger Bands signals
        for i in range(1, len(df)):
            current_price = df[close_col].iloc[i]
            prev_lower_band = df['banda_inferior'].iloc[i-1]
            prev_upper_band = df['banda_superior'].iloc[i-1]
            prev_signal = df['signal'].iloc[i-1]
            
            # LONG trigger: close[t] < lower_band[t-1]
            if current_price < prev_lower_band:
                df.loc[df.index[i], 'signal'] = 1
            
            # SHORT trigger: close[t] > upper_band[t-1]  
            elif current_price > prev_upper_band:
                df.loc[df.index[i], 'signal'] = -1
            
            # Exit long position: close[t] >= upper_band[t-1] and was long
            elif prev_signal == 1 and current_price >= prev_upper_band:
                df.loc[df.index[i], 'signal'] = 0
            
            # Exit short position: close[t] <= lower_band[t-1] and was short
            elif prev_signal == -1 and current_price <= prev_lower_band:
                df.loc[df.index[i], 'signal'] = 0
            
            # Otherwise, maintain previous signal
            else:
                df.loc[df.index[i], 'signal'] = prev_signal
        
        return df
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
                 
        """No optimization - returns default parameters."""
        
        return {
            'best_params': self.default_params,
            'best_value': 0.0,
            'study': None
        }