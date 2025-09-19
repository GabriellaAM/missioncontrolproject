import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class SMACrossoverStrategy(BaseStrategy):
    
    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("sma_crossover")
        self.asset = asset
        self.fast_period_range = [5, 20]
        self.slow_period_range = [21, 100]
        self.default_params = {'fast_period': 10, 'slow_period': 30}
    
    @property
    def strategy_type(self) -> str:
        """SMA Crossover is a trend-following strategy"""
        return "trend_following"
    
    @property
    def implementation_type(self) -> str:
        """SMA Crossover is a rules-based strategy"""
        return "rules_based"
    
    def get_required_features(self) -> Dict[str, Any]:
        """SMA crossover only needs crypto asset price data."""
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': None,
            'yahoo_tickers': None,
            'calculated_features': None
        }
    
    # SMA crossover only uses close price
    used_crypto_features = ['close']
    
    def get_warmup_days(self, params: Dict = None) -> int:
        """Return warmup days needed for SMAs to be ready."""
        if params:
            # Use actual parameters - slow SMA needs the most warmup
            return params.get('slow_period', self.default_params['slow_period'])
        else:
            # Return worst-case scenario for optimization phase
            return self.slow_period_range[1]  # 100 days
    
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate SMA crossover signals."""
        df = data.copy()

        # Find close price column with asset prefix
        close_col = f"{self.asset}_close"
        if close_col not in df.columns:
            raise ValueError(f"Required column '{close_col}' not found in data")

        # Calculate SMAs
        df['sma_fast'] = df[close_col].rolling(window=params['fast_period']).mean()
        df['sma_slow'] = df[close_col].rolling(window=params['slow_period']).mean()

        # Generate signals where both SMAs are valid (not NaN)
        valid_mask = df['sma_fast'].notna() & df['sma_slow'].notna()
        df['signal'] = np.where(
            df['sma_fast'] >= df['sma_slow'],
            1, -1
        )

        # Only return data from where valid signals can be generated
        # This ensures no 0 signals are passed forward
        if valid_mask.any():
            first_valid_idx = df[valid_mask].index[0]
            return df.loc[first_valid_idx:]
        else:
            # If no valid signals, return empty DataFrame with same structure
            return df.iloc[0:0]
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
                 
        """Optimize SMA periods using basic grid search."""
        
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
                    
                    # Evaluate on training period only (no dropna to preserve warmup data)
                    train_data = strategy_data.loc[train_start:train_end]
                    
                    # Only check valid signal data within training period
                    valid_signals = train_data['signal'].notna()
                    if valid_signals.sum() < 100:
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