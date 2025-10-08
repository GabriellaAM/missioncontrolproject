import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any, List
import pandas as pd


class BollingerBandsStrategy(BaseStrategy):
    
    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("bollinger_bands")
        self.asset = asset
        self.period_range = [10, 50]
        self.std_dev_range = [1.5, 3.0]
        self.default_params = {'period': 50, 'std_multiplier': 2.5}
    
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
    
    # Bollinger Bands only uses close price
    used_crypto_features = ['close']
    
    def get_warmup_days(self, params: Dict = None) -> int:
        """Return warmup days needed for Bollinger Bands moving average."""
        if params:
            return params.get('period', self.default_params['period'])
        else:
            # Worst-case for optimization phase
            return self.period_range[1]  # 50 days
    
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

        # Initialize signal column - start with market neutral (flat position = -1 or 1 randomly)
        # For mean reversion, we'll start with -1 (short) until first signal
        df['signal'] = -1

        # Find first valid index where bands are available
        first_valid_idx = None

        # Generate Bollinger Bands signals only where bands are valid (not NaN)
        for i in range(1, len(df)):
            current_price = df[close_col].iloc[i]
            prev_lower_band = df['banda_inferior'].iloc[i-1]
            prev_upper_band = df['banda_superior'].iloc[i-1]
            prev_signal = df['signal'].iloc[i-1]

            # Skip if bands are not valid yet
            if pd.isna(prev_lower_band) or pd.isna(prev_upper_band) or pd.isna(current_price):
                continue

            # Mark first valid index
            if first_valid_idx is None:
                first_valid_idx = i

            # LONG trigger: close[t] < lower_band[t-1]
            if current_price < prev_lower_band:
                df.loc[df.index[i], 'signal'] = 1

            # SHORT trigger: close[t] > upper_band[t-1]
            elif current_price > prev_upper_band:
                df.loc[df.index[i], 'signal'] = -1

            # For mean reversion with exits: stay flat between bands after exit
            # Exit long position: close[t] >= upper_band[t-1] and was long
            elif prev_signal == 1 and current_price >= prev_upper_band:
                # Exit to short (mean reversion expects price to fall back)
                df.loc[df.index[i], 'signal'] = -1

            # Exit short position: close[t] <= lower_band[t-1] and was short
            elif prev_signal == -1 and current_price <= prev_lower_band:
                # Exit to long (mean reversion expects price to rise back)
                df.loc[df.index[i], 'signal'] = 1

            # Otherwise, maintain previous signal
            else:
                df.loc[df.index[i], 'signal'] = prev_signal

        # Only return data from where valid signals can be generated
        if first_valid_idx is not None:
            return df.iloc[first_valid_idx:]
        else:
            # If no valid signals, return empty DataFrame with same structure
            return df.iloc[0:0]
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
                 
        """No optimization - returns default parameters."""
        
        return {
            'best_params': self.default_params,
            'best_value': 0.0,
            'study': None
        }