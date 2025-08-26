import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd
import numpy as np


class EMACrossoverStrategy(BaseStrategy):
    
    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("ema_crossover")
        self.asset = asset
        self.fast_period_range = [5, 20]
        self.slow_period_range = [21, 100]
        self.default_params = {'fast_period': 10, 'slow_period': 30}
    
    @property
    def strategy_type(self) -> str:
        """EMA Crossover is a trend-following strategy"""
        return "trend_following"
    
    def get_required_features(self) -> Dict[str, Any]:
        """EMA crossover only needs crypto asset price data."""
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': None,
            'yahoo_tickers': None,
            'calculated_features': None
        }
    
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate EMA crossover signals."""
        df = data.copy()
        
        # Find close price column with asset prefix
        close_col = f"{self.asset}_close"
        if close_col not in df.columns:
            raise ValueError(f"Required column '{close_col}' not found in data")
        
        # Calculate EMAs
        df['ema_fast'] = df[close_col].ewm(span=params['fast_period'], adjust=False).mean()
        df['ema_slow'] = df[close_col].ewm(span=params['slow_period'], adjust=False).mean()
        
        # Generate signals: 1 when fast > slow, 0 otherwise
        df['signal'] = (df['ema_fast'] >= df['ema_slow']).astype(int)
        
        return df
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
                 
        """Optimize EMA periods using basic grid search."""
        
        best_params = None
        best_score = float('-inf')
        
        # Calculate log returns if not present
        close_col = f"{self.asset}_close"
        return_col = f"{self.asset}_log_return_1"
        
        if return_col not in data.columns:
            data[return_col] = np.log(data[close_col] / data[close_col].shift(1))
        
        # Simple grid search over parameter ranges
        for fast in range(self.fast_period_range[0], self.fast_period_range[1] + 1):
            for slow in range(fast + 1, self.slow_period_range[1] + 1):
                params = {'fast_period': fast, 'slow_period': slow}
                
                try:
                    # Calculate strategy
                    strategy_data = self.calculate_signals(data, params)
                    
                    # Evaluate on training period
                    train_data = strategy_data.loc[train_start:train_end].dropna()
                    if len(train_data) < 100:
                        continue
                    
                    # Calculate profit factor
                    strategy_returns = train_data[return_col] * train_data['signal'].shift(1)
                    gross_profit = strategy_returns[strategy_returns > 0].sum()
                    gross_loss = -strategy_returns[strategy_returns < 0].sum()
                    
                    if gross_loss > 0:
                        score = gross_profit / gross_loss
                        if score > best_score:
                            best_score = score
                            best_params = params
                
                except Exception:
                    continue
        
        return {
            'best_params': best_params or self.default_params,
            'best_value': best_score,
            'study': None
        }