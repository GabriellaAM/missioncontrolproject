import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any, List
import pandas as pd


class BollingerBandsCrossStrategy(BaseStrategy):
    
    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("bollinger_bands_cross")
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
        """Calculate Bollinger Bands crossover signals."""
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

        # Initialize signal column (flat = 0)
        df['signal'] = 0

        # Loop to generate crossover / crossunder signals
        for i in range(1, len(df)):
            price_prev = df[close_col].iloc[i-1]
            price_curr = df[close_col].iloc[i]

            lower_prev = df['banda_inferior'].iloc[i-1]
            lower_curr = df['banda_inferior'].iloc[i]

            upper_prev = df['banda_superior'].iloc[i-1]
            upper_curr = df['banda_superior'].iloc[i]

            prev_signal = df['signal'].iloc[i-1]

            # Skip if bands not available
            if pd.isna(lower_prev) or pd.isna(upper_prev) or pd.isna(price_curr):
                continue

            # Long entry: crossover(close, lower)
            if price_prev < lower_prev and price_curr > lower_curr:
                df.loc[df.index[i], 'signal'] = 1

            # Short entry: crossunder(close, upper)
            elif price_prev > upper_prev and price_curr < upper_curr:
                df.loc[df.index[i], 'signal'] = -1

            # Otherwise, maintain previous position
            else:
                df.loc[df.index[i], 'signal'] = prev_signal

        return df.dropna(subset=['ma', 'std', 'banda_superior', 'banda_inferior'])

    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
                 
        """No optimization - returns default parameters."""
        
        return {
            'best_params': self.default_params,
            'best_value': 0.0,
            'study': None
        }