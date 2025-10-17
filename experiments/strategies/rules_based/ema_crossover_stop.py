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
        self.slow_period_range = [21, 50]
        self.default_params = {'fast_period': 10, 'slow_period': 30, 'volatility_multiplier': 2.0}

    @property
    def strategy_type(self) -> str:
        return "trend_following"

    @property
    def implementation_type(self) -> str:
        return "rules_based"

    def get_required_features(self) -> Dict[str, Any]:
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': None,
            'yahoo_tickers': None,
            'calculated_features': None
        }

    used_crypto_features = ['close', 'high', 'low']

    def get_warmup_days(self, params: Dict = None) -> int:
        if params:
            slow = params.get('slow_period', self.default_params['slow_period'])
            return slow * 3
        else:
            return self.slow_period_range[1]

    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        df = data.copy()

        close_col = f"{self.asset}_close"
        high_col  = f"{self.asset}_high"
        low_col   = f"{self.asset}_low"
        if close_col not in df.columns or high_col not in df.columns or low_col not in df.columns:
            raise ValueError("Required columns not found for EMA/ATR calculation")

        # EMAs
        fast = int(params.get('fast_period', self.default_params['fast_period']))
        slow = int(params.get('slow_period', self.default_params['slow_period']))
        df['ema_fast'] = df[close_col].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df[close_col].ewm(span=slow, adjust=False).mean()

        # ATR(14)
        df['tr1'] = df[high_col] - df[low_col]
        df['tr2'] = (df[high_col] - df[close_col].shift(1)).abs()
        df['tr3'] = (df[low_col]  - df[close_col].shift(1)).abs()
        df['tr']  = df[['tr1','tr2','tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        vol_mult = float(params.get('volatility_multiplier', self.default_params['volatility_multiplier']))
        valid_mask = df['ema_fast'].notna() & df['ema_slow'].notna() & df['atr'].notna()

        n = len(df)
        signal = np.zeros(n, dtype=np.int8)

        # Two trailing stops (NaN-initialized)
        long_ts  = np.full(n, np.nan, dtype=np.float64)   # non-decreasing when long
        short_ts = np.full(n, np.nan, dtype=np.float64)   # non-increasing when short

        # Vectors
        c  = df[close_col].values
        a  = df['atr'].values
        ef = df['ema_fast'].values
        es = df['ema_slow'].values
        val = valid_mask.values

        # State for price-recapture re-entry
        was_stopped_out   = False
        stop_trigger_close = np.nan
        ema_side_at_stop   = 0   # +1 if long at stop, -1 if short at stop

        # Tiny tolerance to avoid fence-sticking
        tol = 1e-12

        for i in range(1, n):
            if not val[i]:
                continue

            prev_sig = signal[i-1]
            pef = ef[i-1] if val[i-1] else np.nan
            pes = es[i-1] if val[i-1] else np.nan

            # strict cross detection with tolerance
            bull_x = (not np.isnan(pef) and not np.isnan(pes) and (pef - pes) <=  tol and (ef[i] - es[i]) >  tol)
            bear_x = (not np.isnan(pef) and not np.isnan(pes) and (pef - pes) >= -tol and (ef[i] - es[i]) < -tol)

            ema_pos = 1 if ef[i] > es[i] else (-1 if ef[i] < es[i] else 0)

            if prev_sig == 1:
                # ratchet ONLY long stop
                cand = c[i] - vol_mult * a[i]
                prev = long_ts[i-1]
                long_ts[i] = cand if np.isnan(prev) else max(prev, cand)
                short_ts[i] = short_ts[i-1]  # carry forward

                # exits
                if (not np.isnan(long_ts[i])) and (c[i] <= long_ts[i]):
                    signal[i] = 0
                    was_stopped_out = True
                    stop_trigger_close = c[i]
                    ema_side_at_stop = 1
                elif bear_x:
                    signal[i] = 0
                    was_stopped_out = False
                    stop_trigger_close = np.nan
                    ema_side_at_stop = 0
                else:
                    signal[i] = 1

            elif prev_sig == -1:
                # ratchet ONLY short stop
                cand = c[i] + vol_mult * a[i]
                prev = short_ts[i-1]
                short_ts[i] = cand if np.isnan(prev) else min(prev, cand)
                long_ts[i] = long_ts[i-1]  # carry forward

                # exits
                if (not np.isnan(short_ts[i])) and (c[i] >= short_ts[i]):
                    signal[i] = 0
                    was_stopped_out = True
                    stop_trigger_close = c[i]
                    ema_side_at_stop = -1
                elif bull_x:
                    signal[i] = 0
                    was_stopped_out = False
                    stop_trigger_close = np.nan
                    ema_side_at_stop = 0
                else:
                    signal[i] = -1

            else:
                # FLAT: do NOT ratchet; only carry forward for plotting
                long_ts[i]  = long_ts[i-1]
                short_ts[i] = short_ts[i-1]

                # 1) re-enter on any true crossover (always allowed)
                if bull_x:
                    signal[i] = 1
                    long_ts[i] = c[i] - vol_mult * a[i]  # initialize on entry
                    was_stopped_out = False
                    stop_trigger_close = np.nan
                    ema_side_at_stop = 0

                elif bear_x:
                    signal[i] = -1
                    short_ts[i] = c[i] + vol_mult * a[i]  # initialize on entry
                    was_stopped_out = False
                    stop_trigger_close = np.nan
                    ema_side_at_stop = 0

                # 2) price-recapture re-entry (ONLY after stop & same EMA side)
                elif was_stopped_out and ema_pos != 0 and ema_pos == ema_side_at_stop and not np.isnan(stop_trigger_close):
                    if (ema_pos == 1 and c[i] > stop_trigger_close):
                        signal[i] = 1
                        long_ts[i] = c[i] - vol_mult * a[i]
                        was_stopped_out = False
                        stop_trigger_close = np.nan
                        ema_side_at_stop = 0
                    elif (ema_pos == -1 and c[i] < stop_trigger_close):
                        signal[i] = -1
                        short_ts[i] = c[i] + vol_mult * a[i]
                        was_stopped_out = False
                        stop_trigger_close = np.nan
                        ema_side_at_stop = 0
                    else:
                        signal[i] = 0
                else:
                    signal[i] = 0

        # Attach outputs
        df['signal'] = signal
        df['long_trailing_stop']  = long_ts
        df['short_trailing_stop'] = short_ts

        if valid_mask.any():
            first_valid_idx = df[valid_mask].index[0]
            return df.loc[first_valid_idx:]
        else:
            return df.iloc[0:0]

    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str,
                 n_trials: int = 1000, **kwargs) -> Dict:
        import optuna

        return_col = f"{self.asset}_log_return_1"
        if return_col not in data.columns:
            data[return_col] = np.log(data[f"{self.asset}_close"] / data[f"{self.asset}_close"].shift(1))

        def objective(trial):
            fast_period = trial.suggest_int('fast_period', self.fast_period_range[0], self.fast_period_range[1])
            slow_period = trial.suggest_int('slow_period', fast_period + 1, self.slow_period_range[1])
            volatility_multiplier = trial.suggest_float('volatility_multiplier', 2.0, 3.0, step=0.1)

            params = {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'volatility_multiplier': volatility_multiplier
            }

            try:
                strat = self.calculate_signals(data, params)
                train = strat.loc[train_start:train_end]
                valid = train['signal'].notna()
                if valid.sum() < 100:
                    return -np.inf

                strat_ret = train[return_col] * train['signal'].shift(1)
                mu, sd = strat_ret.mean(), strat_ret.std()
                if sd > 0 and np.isfinite(mu / sd):
                    return mu / sd
                return -np.inf
            except Exception:
                return -np.inf

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }

    def get_required_artifacts(self) -> list:
        base_artifacts = super().get_required_artifacts()
        primary_artifacts = ['permutation_tests']
        return base_artifacts + primary_artifacts

    def get_input_example(self) -> pd.DataFrame:
        sample_data = {
            f"{self.asset}_high": [50000, 51000, 49000],
            f"{self.asset}_low":  [48000, 49500, 47500],
            f"{self.asset}_close":[49500, 50500, 48500]
        }
        return pd.DataFrame(sample_data)
