import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class EMACrossoverWithTrailingStopStrategy(BaseStrategy):

    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("ema_crossover")
        self.asset = asset
        self.fast_period_range = [5, 20]
        self.slow_period_range = [21, 100]
        self.default_params = {'fast_period': 10, 'slow_period': 30, 'volatility_multiplier': 2.0}

    @property
    def strategy_type(self) -> str:
        """EMA Crossover is a trend-following strategy"""
        return "trend_following"

    @property
    def implementation_type(self) -> str:
        """EMA Crossover is a rules-based strategy"""
        return "rules_based"

    def get_required_features(self) -> Dict[str, Any]:
        """EMA crossover with trailing stop needs crypto asset OHLC data."""
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': None,
            'yahoo_tickers': None,
            'calculated_features': None
        }

    # EMA Crossover with Trailing Stop uses close, high, and low prices for ATR
    used_crypto_features = ['close', 'high', 'low']

    def get_warmup_days(self, params: Dict = None) -> int:
        """Return warmup days needed for EMAs to converge."""
        if params:
            # EMAs need ~3x the period to fully converge
            slow = params.get('slow_period', self.default_params['slow_period'])
            return slow * 3
        else:
            # Worst-case for optimization phase
            return self.slow_period_range[1]  # e.g., 100 * 3 = 300 days

    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate EMA crossover signals with trailing stop mechanism."""
        df = data.copy()

        # Find close price column with asset prefix
        close_col = f"{self.asset}_close"
        if close_col not in df.columns:
            raise ValueError(f"Required column '{close_col}' not found in data")

        # Calculate EMAs
        df['ema_fast'] = df[close_col].ewm(span=params['fast_period'], adjust=False).mean()
        df['ema_slow'] = df[close_col].ewm(span=params['slow_period'], adjust=False).mean()

        # Calculate ATR (Average True Range) for trailing stop
        high_col = f"{self.asset}_high"
        low_col = f"{self.asset}_low"

        if high_col in df.columns and low_col in df.columns:
            # Calculate true range
            df['tr1'] = df[high_col] - df[low_col]
            df['tr2'] = abs(df[high_col] - df[close_col].shift(1))
            df['tr3'] = abs(df[low_col] - df[close_col].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            # Calculate 14-period ATR
            df['atr'] = df['tr'].rolling(window=14).mean()
        else:
            raise ValueError(f"Required columns for ATR calculation not found in data")

        # Get volatility multiplier from params, default to 2 if not provided
        volatility_multiplier = params.get('volatility_multiplier', 2)

        # Generate signals where both EMAs are valid (not NaN)
        valid_mask = df['ema_fast'].notna() & df['ema_slow'].notna() & df['atr'].notna()

        # Initialize arrays for faster access
        signal = np.zeros(len(df), dtype=np.int8)
        trailing_stop = np.full(len(df), np.nan)
        raw_stop = np.full(len(df), np.nan)

        # Track if we were stopped out (to prevent immediate re-entry)
        was_stopped_out = False
        last_exit_direction = 0  # 1 for exited from long, -1 for exited from short

        close_prices = df[close_col].values
        atr_values = df['atr'].values
        ema_fast_values = df['ema_fast'].values
        ema_slow_values = df['ema_slow'].values
        valid_mask_values = valid_mask.values

        # State machine: Loop through and maintain position state
        for i in range(1, len(df)):
            if not valid_mask_values[i]:
                continue

            prev_signal = signal[i-1]
            close_price = close_prices[i]
            atr_value = atr_values[i]
            ema_fast = ema_fast_values[i]
            ema_slow = ema_slow_values[i]

            # Check for EMA crossover
            prev_ema_fast = ema_fast_values[i-1] if i > 0 and valid_mask_values[i-1] else np.nan
            prev_ema_slow = ema_slow_values[i-1] if i > 0 and valid_mask_values[i-1] else np.nan

            # Detect actual crossovers (not just relative positions)
            bullish_crossover = (not np.isnan(prev_ema_fast) and not np.isnan(prev_ema_slow) and
                                prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow)
            bearish_crossover = (not np.isnan(prev_ema_fast) and not np.isnan(prev_ema_slow) and
                                prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow)

            # State 1: Currently in LONG position
            if prev_signal == 1:
                # Update trailing stop (can only move up for long)
                prev_trailing_stop = trailing_stop[i-1]
                raw_stop_val = close_price - volatility_multiplier * atr_value
                raw_stop[i] = raw_stop_val

                # Handle NaN in previous trailing stop
                if np.isnan(prev_trailing_stop):
                    trailing_stop[i] = raw_stop_val
                else:
                    trailing_stop[i] = max(prev_trailing_stop, raw_stop_val)

                # Check exit condition: close <= trailing_stop (use current, not previous)
                # Also check for bearish crossover as an exit signal
                if (not np.isnan(trailing_stop[i]) and close_price <= trailing_stop[i]) or bearish_crossover:
                    signal[i] = 0  # Exit to neutral
                    was_stopped_out = True
                    last_exit_direction = 1  # Exited from long
                else:
                    signal[i] = 1  # Stay in long

            # State -1: Currently in SHORT position
            elif prev_signal == -1:
                # Update trailing stop (can only move down for short)
                prev_trailing_stop = trailing_stop[i-1]
                raw_stop_val = close_price + volatility_multiplier * atr_value
                raw_stop[i] = raw_stop_val

                # Handle NaN in previous trailing stop
                if np.isnan(prev_trailing_stop):
                    trailing_stop[i] = raw_stop_val
                else:
                    trailing_stop[i] = min(prev_trailing_stop, raw_stop_val)

                # Check exit condition: close >= trailing_stop (use current, not previous)
                # Also check for bullish crossover as an exit signal
                if (not np.isnan(trailing_stop[i]) and close_price >= trailing_stop[i]) or bullish_crossover:
                    signal[i] = 0  # Exit to neutral
                    was_stopped_out = True
                    last_exit_direction = -1  # Exited from short
                else:
                    signal[i] = -1  # Stay in short

            # State 0: Currently NEUTRAL - check for entry signals
            else:
                # Only enter on actual crossovers, not just EMA positions
                # Prevent immediate re-entry in same direction after stop out

                if bullish_crossover and last_exit_direction != 1:
                    signal[i] = 1
                    # Initialize trailing stop for new long position
                    initial_stop = close_price - volatility_multiplier * atr_value
                    raw_stop[i] = initial_stop
                    trailing_stop[i] = initial_stop
                    was_stopped_out = False
                    last_exit_direction = 0

                elif bearish_crossover and last_exit_direction != -1:
                    signal[i] = -1
                    # Initialize trailing stop for new short position
                    initial_stop = close_price + volatility_multiplier * atr_value
                    raw_stop[i] = initial_stop
                    trailing_stop[i] = initial_stop
                    was_stopped_out = False
                    last_exit_direction = 0
                else:
                    signal[i] = 0  # Stay neutral
                    # Carry forward the trailing stop value for visualization
                    if i > 0:
                        trailing_stop[i] = trailing_stop[i-1]

        # Assign back to dataframe
        df['signal'] = signal
        df['trailing_stop'] = trailing_stop
        df['raw_stop'] = raw_stop

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

        """Optimize EMA periods and volatility multiplier using Optuna with profit factor."""
        import optuna

        # Calculate log returns if not present
        return_col = f"{self.asset}_log_return_1"
        if return_col not in data.columns:
            data[return_col] = np.log(data[f"{self.asset}_close"] / data[f"{self.asset}_close"].shift(1))

        def objective(trial):
            # Suggest parameters
            fast_period = trial.suggest_int('fast_period', self.fast_period_range[0], self.fast_period_range[1])
            slow_period = trial.suggest_int('slow_period', fast_period + 1, self.slow_period_range[1])
            volatility_multiplier = trial.suggest_float('volatility_multiplier', 1.0, 3.0, step=0.5)

            params = {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'volatility_multiplier': volatility_multiplier
            }

            try:
                # Calculate strategy on full data (including warmup)
                strategy_data = self.calculate_signals(data, params)

                # Evaluate only on training period
                train_data = strategy_data.loc[train_start:train_end]

                # Only check valid signal data within training period
                valid_signals = train_data['signal'].notna()
                if valid_signals.sum() < 100:
                    return -np.inf

                # Calculate profit factor
                strategy_returns = train_data[return_col] * train_data['signal'].shift(1)
                gross_profit = strategy_returns[strategy_returns > 0].sum()
                gross_loss = -strategy_returns[strategy_returns < 0].sum()

                if gross_loss > 0:
                    profit_factor = gross_profit / gross_loss
                    return profit_factor if np.isfinite(profit_factor) else -np.inf
                else:
                    return -np.inf

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

    def get_input_example(self) -> pd.DataFrame:
        """Return sample input for MLflow signature."""
        sample_data = {
            f"{self.asset}_high": [50000, 51000, 49000],
            f"{self.asset}_low": [48000, 49500, 47500],
            f"{self.asset}_close": [49500, 50500, 48500]
        }
        return pd.DataFrame(sample_data)