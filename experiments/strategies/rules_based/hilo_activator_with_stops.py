import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class HiloActivatorWithTrailingStopStrategy(BaseStrategy):
    """
    Hilo Activator strategy with trailing stop risk management.

    This strategy combines:
    1. Hilo Activator for primary signal generation (MAs of shifted highs/lows)
    2. ATR-based trailing stop system for risk management

    Primary Signal Logic (Hilo Activator):
      - Long signal when close > high_ma (moving average of shifted highs)
      - Short signal when close < low_ma (moving average of shifted lows)
      - Maintain position when price is between bands

    Risk Management Layer:
      - Uses ATR(14) to calculate dynamic stop levels
      - Long stop: close - (volatility_multiplier × ATR), tracks maximum since entry
      - Short stop: close + (volatility_multiplier × ATR), tracks minimum since entry
      - Stop flags prevent new positions when stops are triggered
      - Re-entry allowed when price recovers past the stop level

    Final Signal:
      - Long: hilo_signal = 1 AND long_stop not active
      - Short: hilo_signal = -1 AND short_stop not active
      - Neutral: when stop is active
    """

    def __init__(self, asset: str = 'bitcoin'):
        super().__init__("hilo_activator_with_stops")
        self.asset = asset

        # Hilo parameters (already optimized, so we'll use fixed values)
        self.hilo_period = 8
        self.hilo_shift = 1
        self.hilo_ma_type = 'sma'

        # Risk management parameters (to be optimized)
        self.default_params = {
            'period': 8,  # Fixed from Hilo optimization
            'shift': 1,   # Fixed from Hilo optimization
            'ma_type': 'sma',  # Fixed from Hilo optimization
            'volatility_multiplier': 2.0  # To be optimized
        }

    @property
    def strategy_type(self) -> str:
        """Hilo Activator with stops is a trend-following strategy"""
        return "trend_following"

    @property
    def implementation_type(self) -> str:
        """This is a rules-based strategy"""
        return "rules_based"

    def get_required_features(self) -> Dict[str, Any]:
        """Hilo Activator with stops needs OHLC data for the specified asset."""
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': None,
            'yahoo_tickers': None,
            'calculated_features': None
        }

    # Uses close, high, and low prices
    used_crypto_features = ['close', 'high', 'low']

    def get_warmup_days(self, params: Dict = None) -> int:
        """Return warmup days needed for Hilo Activator MAs and ATR calculation."""
        if params:
            period = params.get('period', self.default_params['period'])
            shift = params.get('shift', self.default_params['shift'])
            ma_type = params.get('ma_type', self.default_params['ma_type']).lower()

            # Need warmup for both Hilo MA and ATR(14)
            if ma_type == 'ema':
                hilo_warmup = (period * 3) + shift
            else:
                hilo_warmup = period + shift

            # ATR needs 14 days minimum
            atr_warmup = 14

            return max(hilo_warmup, atr_warmup)
        else:
            # Worst-case for optimization phase
            # Using fixed Hilo params, so just need their warmup + ATR
            return max(self.hilo_period + self.hilo_shift, 14)

    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Calculate Hilo Activator signals with trailing stop risk management.

        Steps:
        1) Calculate Hilo Activator primary signal (high_ma, low_ma bands)
        2) Calculate ATR(14) for volatility measurement
        3) Calculate dynamic stop bands: close ± (volatility_multiplier × ATR)
        4) Track trailing stops (max for long, min for short) since position entry
        5) Apply stop flags with activation/deactivation logic
        6) Combine: final_signal = hilo_signal AND (stop not active)
        """
        df = data.copy()

        high_col = f"{self.asset}_high"
        low_col = f"{self.asset}_low"
        close_col = f"{self.asset}_close"

        required_cols = [high_col, low_col, close_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found: {missing_cols}")

        # =========================
        # STEP 1: Hilo Activator Primary Signal
        # =========================

        # Use fixed Hilo parameters (already optimized)
        period = params.get('period', self.default_params['period'])
        shift = params.get('shift', self.default_params['shift'])
        ma_type = params.get('ma_type', self.default_params['ma_type']).lower()

        # Shift highs and lows if needed
        shifted_high = df[high_col].shift(shift) if shift > 0 else df[high_col]
        shifted_low = df[low_col].shift(shift) if shift > 0 else df[low_col]

        # Moving averages for Hilo bands
        if ma_type == 'ema':
            high_ma = shifted_high.ewm(span=period, adjust=False).mean()
            low_ma = shifted_low.ewm(span=period, adjust=False).mean()
        else:  # sma
            high_ma = shifted_high.rolling(window=period, min_periods=period).mean()
            low_ma = shifted_low.rolling(window=period, min_periods=period).mean()

        # =========================
        # STEP 2: ATR Calculation
        # =========================

        # Calculate True Range components
        tr1 = df[high_col] - df[low_col]
        tr2 = (df[high_col] - df[close_col].shift()).abs()
        tr3 = (df[low_col] - df[close_col].shift()).abs()

        # ATR as 14-period moving average of True Range
        df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

        # Get volatility multiplier parameter
        vol_mult = float(params.get('volatility_multiplier', self.default_params['volatility_multiplier']))

        # Convert to numpy arrays for efficient computation
        close = df[close_col].values
        atr = df['atr'].values
        n = len(df)

        # =========================
        # Initialize Signal Arrays (matching EMA crossover stop structure)
        # =========================

        # All intermediate signals stored separately for debugging
        hilo_signal = np.zeros(n, dtype=np.int8)      # Primary Hilo signal
        long_atr = np.zeros(n)                        # Lower band for long positions
        short_atr = np.zeros(n)                       # Upper band for short positions
        long_trail_stop = np.zeros(n)                 # Long trailing stop (max since entry)
        short_trail_stop = np.zeros(n)                # Short trailing stop (min since entry)
        long_stop = np.zeros(n, dtype=np.int8)        # Long stop triggered flag
        short_stop = np.zeros(n, dtype=np.int8)       # Short stop triggered flag
        final_signal = np.zeros(n, dtype=np.int8)     # Final combined signal

        # Check validity (need valid MAs and ATR)
        val = (~np.isnan(high_ma.values)) & (~np.isnan(low_ma.values)) & (~np.isnan(atr))
        tol = 1e-12

        # =========================
        # State Variables for Position Tracking
        # =========================

        hilo_position = -1  # Track Hilo position state (start short like original)
        long_position_start = -1
        short_position_start = -1
        long_stop_active = False
        short_stop_active = False
        first_valid_idx = None

        # =========================
        # Main Signal Calculation Loop
        # =========================

        for i in range(n):
            if not val[i]:
                continue

            # Mark first valid index
            if first_valid_idx is None:
                first_valid_idx = i

            # =========================
            # Step 1: Calculate Hilo primary signal
            # =========================

            if close[i] > high_ma.iloc[i]:
                # Breakout above high MA - go long
                hilo_signal[i] = 1
                hilo_position = 1
            elif close[i] < low_ma.iloc[i]:
                # Break below low MA - go short
                hilo_signal[i] = -1
                hilo_position = -1
            else:
                # Between bands - maintain previous position
                hilo_signal[i] = hilo_position

            # Detect position changes for tracking
            if i > 0:
                # Track when positions start (for calculating max/min since start)
                if hilo_signal[i] == 1 and hilo_signal[i-1] != 1:
                    long_position_start = i
                elif hilo_signal[i] == -1 and hilo_signal[i-1] != -1:
                    short_position_start = i

                # Reset stop flags when hilo_signal changes
                if hilo_signal[i] != hilo_signal[i-1]:
                    long_stop_active = False
                    short_stop_active = False

            # =========================
            # Step 2: Calculate ATR bands (only when in position)
            # =========================

            # Long ATR band (RESET to 0 when not in long position)
            if hilo_signal[i] == 1:
                long_atr[i] = close[i] - vol_mult * atr[i]
            else:
                long_atr[i] = 0

            # Short ATR band (RESET to 0 when not in short position)
            if hilo_signal[i] == -1:
                short_atr[i] = close[i] + vol_mult * atr[i]
            else:
                short_atr[i] = 0

            # =========================
            # Step 3: Calculate trailing stops (track max/min since entry)
            # =========================

            # Long trailing stop (max since position start, RESET to 0 when not long)
            if hilo_signal[i] == 1 and long_atr[i] != 0:
                if long_position_start >= 0:
                    long_trail_stop[i] = np.max(long_atr[long_position_start:i+1])
                else:
                    long_trail_stop[i] = long_atr[i]
            else:
                long_trail_stop[i] = 0

            # Short trailing stop (min since position start, RESET to 0 when not short)
            if hilo_signal[i] == -1 and short_atr[i] != 0:
                if short_position_start >= 0:
                    short_trail_stop[i] = np.min(short_atr[short_position_start:i+1])
                else:
                    short_trail_stop[i] = short_atr[i]
            else:
                short_trail_stop[i] = 0

            # =========================
            # Step 4: Apply stop flag logic (with re-entry capability)
            # =========================

            # Long stop logic (matching EMA crossover stop implementation)
            if hilo_signal[i] == 1:
                if long_stop_active:
                    long_stop[i] = 1
                    # Check if close goes above long_trail_stop to deactivate stop (re-entry)
                    if close[i] > long_trail_stop[i]:
                        long_stop_active = False
                else:
                    # Check if close drops below long_trail_stop to activate stop
                    if long_trail_stop[i] > 0 and close[i] < long_trail_stop[i]:
                        long_stop_active = True
                        long_stop[i] = 1

            # Short stop logic
            if hilo_signal[i] == -1:
                if short_stop_active:
                    short_stop[i] = 1
                    # Check if close goes below short_trail_stop to deactivate stop (re-entry)
                    if close[i] < short_trail_stop[i]:
                        short_stop_active = False
                else:
                    # Check if close goes above short_trail_stop to activate stop
                    if short_trail_stop[i] > 0 and close[i] > short_trail_stop[i]:
                        short_stop_active = True
                        short_stop[i] = 1

            # =========================
            # Step 5: Calculate final signal (hilo AND stop not active)
            # =========================

            if hilo_signal[i] == 1 and long_stop[i] != 1:
                final_signal[i] = 1
            elif hilo_signal[i] == -1 and short_stop[i] != 1:
                final_signal[i] = -1
            else:
                final_signal[i] = 0

        # =========================
        # Store All Signals in DataFrame
        # =========================

        # Store Hilo-specific columns
        df['hilo_high_ma'] = high_ma
        df['hilo_low_ma'] = low_ma
        df['hilo_signal'] = hilo_signal

        # Store risk management columns (matching EMA crossover stop naming)
        df['long_atr'] = long_atr
        df['short_atr'] = short_atr
        df['long_trail_stop'] = long_trail_stop
        df['short_trail_stop'] = short_trail_stop
        df['long_stop'] = long_stop
        df['short_stop'] = short_stop

        # Final signal
        df['signal'] = final_signal

        # Add legacy column names for compatibility
        df['long_trailing_stop'] = long_trail_stop
        df['short_trailing_stop'] = short_trail_stop

        # Event flags for analysis (matching original Hilo implementation)
        if first_valid_idx is not None:
            valid_df = df.iloc[first_valid_idx:]
            valid_hilo_signal = hilo_signal[first_valid_idx:]
            valid_final_signal = final_signal[first_valid_idx:]

            # Entry flags based on Hilo signal changes
            enter_long = (valid_hilo_signal == 1) & (pd.Series(valid_hilo_signal).shift(1).fillna(-1) != 1)
            enter_short = (valid_hilo_signal == -1) & (pd.Series(valid_hilo_signal).shift(1).fillna(1) != -1)

            # Store event flags
            df.loc[valid_df.index, 'enter_long'] = enter_long.values
            df.loc[valid_df.index, 'enter_short'] = enter_short.values

            return df.iloc[first_valid_idx:]
        else:
            # If no valid signals, return empty DataFrame
            return df.iloc[0:0]

    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str,
                 n_trials: int = 500, **kwargs) -> Dict:
        """
        Optimize the volatility_multiplier parameter using Optuna to maximize Sharpe ratio.

        Note: Hilo parameters (period, shift, ma_type) are fixed as they've been
        optimized separately. We only optimize the risk management parameter.
        """
        import optuna
        from utils.evaluation_metrics import calculate_sharpe_ratio

        # Calculate returns if not present
        return_col = f"{self.asset}_log_return_1"
        if return_col not in data.columns:
            data[return_col] = np.log(data[f"{self.asset}_close"] / data[f"{self.asset}_close"].shift(1))

        def objective(trial):
            # Only optimize volatility_multiplier
            volatility_multiplier = trial.suggest_float('volatility_multiplier', 1.0, 3.0, step=0.5)

            # Use fixed Hilo parameters
            params = {
                'period': self.hilo_period,
                'shift': self.hilo_shift,
                'ma_type': self.hilo_ma_type,
                'volatility_multiplier': volatility_multiplier
            }

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
            except Exception as e:
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
        """Return parameter ranges for optimization."""
        return {
            'volatility_multiplier': {'type': 'float', 'range': [1.0, 3.0], 'step': 0.5}
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
        Hilo Activator Strategy with Trailing Stops for {self.asset}:

        **Primary Signal (Hilo Activator):**
        - Calculates moving averages of highs and lows shifted by N periods
        - Enter long when close > high_ma (bull trend start)
        - Enter short when close < low_ma (bear trend start)
        - Maintain position when price is between bands

        **Risk Management (Trailing Stops):**
        - Uses ATR(14) to calculate dynamic stop levels
        - Long stop: tracks maximum of (close - volatility_multiplier × ATR) since entry
        - Short stop: tracks minimum of (close + volatility_multiplier × ATR) since entry
        - Stops prevent new positions when triggered
        - Re-entry allowed when price recovers past the stop level

        **Parameters:**
        - period: {self.hilo_period} (fixed from Hilo optimization)
        - shift: {self.hilo_shift} (fixed from Hilo optimization)
        - ma_type: {self.hilo_ma_type} (fixed from Hilo optimization)
        - volatility_multiplier: {self.default_params['volatility_multiplier']} (optimized for risk management)

        **Final Signal:**
        - Long: Hilo says long AND long stop not active
        - Short: Hilo says short AND short stop not active
        - Neutral: When stop is active (risk management override)
        """