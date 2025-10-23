import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class EMACrossoverWithTrailingStopStrategy(BaseStrategy):
    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("ema_crossover_trail_reentry")
        self.asset = asset
        self.default_params = {
            'fast_period': 9,
            'slow_period': 21,
            'volatility_multiplier': 2.0
        }

    @property
    def strategy_type(self): return "trend_following"
    @property
    def implementation_type(self): return "rules_based"

    used_crypto_features = ['close','high','low']

    def get_required_features(self, *args, **kwargs):
        return {'crypto_assets':[self.asset],'fred_indicators':None,'yahoo_tickers':None,'calculated_features':None}

    def get_warmup_days(self, params=None):
        slow = params.get('slow_period', self.default_params['slow_period']) if params else self.default_params['slow_period']
        return int(slow * 3)

    # === SIGNAL CALCULATION ===
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        df = data.copy()
        c, h, l = (f"{self.asset}_close", f"{self.asset}_high", f"{self.asset}_low")
        for col in (c,h,l):
            if col not in df.columns:
                raise ValueError(f"Missing {col}")

        # EMAs - Hardcoded values
        fast = 9
        slow = 21
        df['ema_fast'] = df[c].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df[c].ewm(span=slow, adjust=False).mean()

        # Step 2: ATR
        tr1, tr2, tr3 = df[h]-df[l], (df[h]-df[c].shift()).abs(), (df[l]-df[c].shift()).abs()
        df['atr'] = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1).rolling(14).mean()

        vol_mult = float(params.get('volatility_multiplier', self.default_params['volatility_multiplier']))

        ef, es, a, close = df['ema_fast'].values, df['ema_slow'].values, df['atr'].values, df[c].values
        n = len(df)

        # Initialize all intermediate signals as separate arrays (for debugging and clarity)
        ema_signal = np.zeros(n, dtype=np.int8)      # Step 1: EMA crossover signal
        long_atr = np.zeros(n)                        # Step 3: Lower band for long positions
        short_atr = np.zeros(n)                       # Step 4: Upper band for short positions
        long_trail_stop = np.zeros(n)                 # Step 5: Long trailing stop (max since position start)
        short_trail_stop = np.zeros(n)                # Step 6: Short trailing stop (min since position start)
        long_stop = np.zeros(n, dtype=np.int8)        # Step 7: Long stop triggered flag
        short_stop = np.zeros(n, dtype=np.int8)       # Step 8: Short stop triggered flag
        final_signal = np.zeros(n, dtype=np.int8)     # Step 9: Final combined signal

        val = (~np.isnan(ef)) & (~np.isnan(es)) & (~np.isnan(a))
        tol = 1e-12

        # State vars for position tracking
        long_position_start = -1
        short_position_start = -1
        long_stop_active = False
        short_stop_active = False

        for i in range(1, n):
            if not val[i]: continue

            # Step 1: ema_signal based on TODAY's EMA relationship (close[i])
            # The shift to next day's open is handled by generate_run.py
            if ef[i] > es[i] + tol:
                ema_signal[i] = 1
            elif ef[i] < es[i] - tol:
                ema_signal[i] = -1
            else:
                ema_signal[i] = 0

            # Detect crossovers for entry/exit
            bull_x = (ef[i-1] <= es[i-1] + tol) and (ef[i] > es[i] + tol)
            bear_x = (ef[i-1] >= es[i-1] - tol) and (ef[i] < es[i] - tol)

            # Track when positions start (for calculating max/min since start)
            if ema_signal[i] == 1 and ema_signal[i-1] != 1:
                long_position_start = i
            elif ema_signal[i] == -1 and ema_signal[i-1] != -1:
                short_position_start = i

            # Step 3: long_atr (RESET to 0 when not in long position)
            if ema_signal[i] == 1:
                long_atr[i] = close[i] - vol_mult * a[i]
            else:
                long_atr[i] = 0  # FIXED: Reset to 0 when not in position

            # Step 4: short_atr (RESET to 0 when not in short position)
            if ema_signal[i] == -1:
                short_atr[i] = close[i] + vol_mult * a[i]
            else:
                short_atr[i] = 0  # FIXED: Reset to 0 when not in position

            # Step 5: long_trail_stop (max since position start, RESET to 0 when not in long)
            if ema_signal[i] == 1 and long_atr[i] != 0:
                if long_position_start >= 0:
                    long_trail_stop[i] = np.max(long_atr[long_position_start:i+1])
                else:
                    long_trail_stop[i] = long_atr[i]
            else:
                long_trail_stop[i] = 0  # FIXED: Reset to 0 when not in position

            # Step 6: short_trail_stop (min since position start, RESET to 0 when not in short)
            if ema_signal[i] == -1 and short_atr[i] != 0:
                if short_position_start >= 0:
                    short_trail_stop[i] = np.min(short_atr[short_position_start:i+1])
                else:
                    short_trail_stop[i] = short_atr[i]
            else:
                short_trail_stop[i] = 0  # FIXED: Reset to 0 when not in position

            # Step 7: long_stop (matching EMATS.py logic with 1-period lag)
            # Reset stop flags when ema_signal changes
            if ema_signal[i] != ema_signal[i-1]:
                long_stop_active = False
                short_stop_active = False

            # Long stop logic
            if ema_signal[i] == 1:
                if long_stop_active:
                    long_stop[i] = 1
                    # Check if close goes above long_trail_stop to deactivate stop
                    if close[i] > long_trail_stop[i]:
                        long_stop_active = False
                else:
                    # Check if close drops below long_trail_stop to activate stop
                    if long_trail_stop[i] > 0 and close[i] < long_trail_stop[i]:
                        long_stop_active = True
                        long_stop[i] = 1

            # Step 8: short_stop (matching EMATS.py logic with 1-period lag)
            if ema_signal[i] == -1:
                if short_stop_active:
                    short_stop[i] = 1
                    # Check if close goes below short_trail_stop to deactivate stop
                    if close[i] < short_trail_stop[i]:
                        short_stop_active = False
                else:
                    # Check if close goes above short_trail_stop to activate stop
                    if short_trail_stop[i] > 0 and close[i] > short_trail_stop[i]:
                        short_stop_active = True
                        short_stop[i] = 1

            # Step 9: final_signal based on ema_signal and stops (matching EMATS.py)
            if ema_signal[i] == 1 and long_stop[i] != 1:
                final_signal[i] = 1
            elif ema_signal[i] == -1 and short_stop[i] != 1:
                final_signal[i] = -1
            else:
                final_signal[i] = 0

        # Assign all intermediate signals to dataframe for debugging and analysis
        df['ema_signal'] = ema_signal
        df['long_atr'] = long_atr
        df['short_atr'] = short_atr
        df['long_trail_stop'] = long_trail_stop
        df['short_trail_stop'] = short_trail_stop
        df['long_stop'] = long_stop
        df['short_stop'] = short_stop
        df['signal'] = final_signal

        # Keep legacy column names for backwards compatibility
        df['long_trailing_stop'] = long_trail_stop
        df['short_trailing_stop'] = short_trail_stop

        first_valid = df[val].index[0] if val.any() else 0
        return df.loc[first_valid:]

    # === OPTIMIZER ===
    def optimize(self, data, train_start, train_end, n_trials=500, **kwargs):
        # Since EMA periods are hardcoded, optimization only considers volatility multiplier
        import optuna
        return_col = f"{self.asset}_log_return_1"
        if return_col not in data:
            data[return_col] = np.log(data[f"{self.asset}_close"] / data[f"{self.asset}_close"].shift(1))

        def objective(trial):
            volm = trial.suggest_float('volatility_multiplier', 1.0, 3, step=0.5)

            params = dict(fast_period=9, slow_period=21,
                          volatility_multiplier=volm)

            try:
                strat = self.calculate_signals(data, params)
                train = strat.loc[train_start:train_end]
                strat_ret = train[return_col] * train['signal'].shift(1)
                mu, sd = strat_ret.mean(), strat_ret.std()
                return mu/sd if sd>0 else -np.inf
            except Exception:
                return -np.inf

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return {'best_params': study.best_params, 'best_value': study.best_value, 'n_trials': len(study.trials)}
