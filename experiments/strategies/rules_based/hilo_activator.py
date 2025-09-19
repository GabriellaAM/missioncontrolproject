import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class HiloActivatorStrategy(BaseStrategy):
    """
    Long/Short trend-following strategy using the Hilo Activator (MAs of shifted highs/lows).

    Convention:
      - +1 = long (in position)
      - -1 = short (in position)

    Rules:
      - Enter long when close > high_ma (start of bull trend)
      - Enter short when close < low_ma (start of bear trend)
      - Between the bands, maintain previous position
    """

    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("hilo_activator")
        self.asset = asset
        self.period_range = [5, 20]
        self.shift_range = [0, 3]
        self.default_params = {
            'period': 8,
            'shift': 1,
            'ma_type': 'sma'  # 'sma' or 'ema'
        }

    @property
    def strategy_type(self) -> str:
        """Hilo Activator is a trend-following strategy"""
        return "trend_following"
    
    @property
    def implementation_type(self) -> str:
        """Hilo Activator is a rules-based strategy"""
        return "rules_based"

    def get_required_features(self) -> Dict[str, Any]:
        """Hilo Activator needs OHLC data for the specified asset."""
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': None,
            'yahoo_tickers': None,
            'calculated_features': None
        }
    
    # Hilo Activator uses close, high, and low prices
    used_crypto_features = ['close', 'high', 'low']

    def get_warmup_days(self, params: Dict = None) -> int:
        """Return warmup days needed for Hilo Activator moving averages."""
        if params:
            period = params.get('period', self.default_params['period'])
            shift = params.get('shift', self.default_params['shift'])
            ma_type = params.get('ma_type', self.default_params['ma_type']).lower()
            
            if ma_type == 'ema':
                # EMA needs ~3x the period to converge, plus shift
                return (period * 3) + shift
            else:
                # SMA needs the period, plus shift
                return period + shift
        else:
            # Worst-case for optimization phase
            max_period = self.period_range[1]
            max_shift = self.shift_range[1]
            return (max_period * 3) + max_shift  # Assume EMA worst-case: 20*3 + 3 = 63 days

    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Calculate Hilo Activator signals.

        Steps:
        1) Calculate moving averages of shifted highs and lows
        2) Signal: +1 if close > high_ma, -1 if close < low_ma, NaN if in-between
        3) Forward-fill to maintain position between bands
        """
        df = data.copy()

        high_col = f"{self.asset}_high"
        low_col = f"{self.asset}_low"
        close_col = f"{self.asset}_close"

        required_cols = [high_col, low_col, close_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}")

        # Parameters
        period = params.get('period', self.default_params['period'])
        shift = params.get('shift', self.default_params['shift'])
        ma_type = params.get('ma_type', self.default_params['ma_type']).lower()

        # Shift highs and lows if needed
        shifted_high = df[high_col].shift(shift) if shift > 0 else df[high_col]
        shifted_low = df[low_col].shift(shift) if shift > 0 else df[low_col]

        # Moving averages
        if ma_type == 'ema':
            high_ma = shifted_high.ewm(span=period, adjust=False).mean()
            low_ma = shifted_low.ewm(span=period, adjust=False).mean()
        else:  # sma
            high_ma = shifted_high.rolling(window=period, min_periods=period).mean()
            low_ma = shifted_low.rolling(window=period, min_periods=period).mean()

        # Generate signals with explicit state tracking
        # This avoids forward-fill which causes issues with permutation tests

        # Initialize signal array - start with short position
        signal = pd.Series(-1, index=df.index)

        # Track current position state and first valid index
        in_position = -1  # Start short until we get first signal
        first_valid_idx = None

        for i in range(len(df)):
            if pd.isna(high_ma.iloc[i]) or pd.isna(low_ma.iloc[i]):
                # During warmup period, skip
                continue

            # Mark first valid index
            if first_valid_idx is None:
                first_valid_idx = i

            if df[close_col].iloc[i] > high_ma.iloc[i]:
                # Breakout above high MA - go long
                signal.iloc[i] = 1
                in_position = 1
            elif df[close_col].iloc[i] < low_ma.iloc[i]:
                # Break below low MA - go short
                signal.iloc[i] = -1
                in_position = -1
            else:
                # Between bands - maintain previous position
                signal.iloc[i] = in_position

        signal = signal.astype(int)

        # Event flags for backtesting (only calculate on valid data)
        if first_valid_idx is not None:
            valid_df = df.iloc[first_valid_idx:]
            valid_signal = signal.iloc[first_valid_idx:]
            valid_high_ma = high_ma.iloc[first_valid_idx:]
            valid_low_ma = low_ma.iloc[first_valid_idx:]

            enter_long = (valid_df[close_col] > valid_high_ma) & (valid_signal.shift(1).fillna(-1) != 1)
            enter_short = (valid_df[close_col] < valid_low_ma) & (valid_signal.shift(1).fillna(1) != -1)

            # Store results only for valid data
            df = df.iloc[first_valid_idx:].copy()
            df['hilo_high_ma'] = valid_high_ma
            df['hilo_low_ma'] = valid_low_ma
            df['signal'] = valid_signal
            df['enter_long'] = enter_long.astype(bool)
            df['enter_short'] = enter_short.astype(bool)

            return df
        else:
            # If no valid signals, return empty DataFrame
            return df.iloc[0:0]

    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str,
                 n_trials: int = 1000, **kwargs) -> Dict:
        """Optimize Hilo Activator parameters using Optuna to maximize Sharpe ratio."""
        import optuna
        from utils.evaluation_metrics import calculate_sharpe_ratio

        # Use full data for strategy calculation, evaluate only on training period
        return_col = f"{self.asset}_log_return_1"
        if return_col not in data.columns:
            data[return_col] = np.log(data[f"{self.asset}_close"] / data[f"{self.asset}_close"].shift(1))

        def objective(trial):
            period = trial.suggest_int('period', self.period_range[0], self.period_range[1])
            shift = trial.suggest_int('shift', self.shift_range[0], self.shift_range[1])
            ma_type = trial.suggest_categorical('ma_type', ['sma', 'ema'])

            params = {'period': period, 'shift': shift, 'ma_type': ma_type}

            try:
                # Calculate strategy on full data (including warmup)
                strat_df = self.calculate_signals(data, params)
                strat_df['strategy_returns'] = strat_df[return_col] * strat_df['signal'].shift(1)
                
                # Evaluate only on training period
                train_returns = strat_df.loc[train_start:train_end, 'strategy_returns']
                valid_returns = train_returns.dropna()
                
                if len(valid_returns) < 30:
                    return -np.inf
                
                sharpe = calculate_sharpe_ratio(valid_returns.values)
                return sharpe if np.isfinite(sharpe) else -np.inf
            except Exception:
                return -np.inf

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }

    def get_parameter_ranges(self) -> Dict[str, Any]:
        return {
            'period': {'type': 'int', 'range': self.period_range},
            'shift': {'type': 'int', 'range': self.shift_range},
            'ma_type': {'type': 'categorical', 'choices': ['sma', 'ema']}
        }

    def get_input_example(self) -> pd.DataFrame:
        """Return sample input for MLflow signature."""
        sample_data = {
            f"{self.asset}_high": [50000, 51000, 49000],
            f"{self.asset}_low": [48000, 49500, 47500], 
            f"{self.asset}_close": [49500, 50500, 48500]
        }
        return pd.DataFrame(sample_data)

    def describe(self) -> str:
        return f"""
        Hilo Activator Strategy (long/short) for {self.asset}:

        - Calculates moving averages of highs and lows shifted by N periods.
        - **Enter long** when close > high_ma (bull trend start).
        - **Enter short** when close < low_ma (bear trend start).
        - Maintain position when the price is between the bands.

        Parameters:
          - period: MA period (default: {self.default_params['period']})
          - shift: shift in periods (default: {self.default_params['shift']})
          - ma_type: 'sma' or 'ema' (default: {self.default_params['ma_type']})
        """
