"""
Signal framework for Soros System.

This package contains signal interfaces and implementations for generating trading signals.
"""

from .signal_base import SignalBase
from .signal_registry import (
    register_signal,
    get_signal,
    get_all_signals,
    get_signal_info,
    get_signal_names,
)

# Import signal implementations
from . import trend_signals
from . import rsi_signals
from . import ssr_signals
from . import volatility_signals
from . import donchian_signals  # Import the new Donchian signals module
from . import regime_signals  # Import the new Regime Detection signals module

# Import specific signals for easy access
from .trend_signals import (
    ShortTermNeutralUSD, MediumTermNeutralUSD, LongTermNeutralUSD,
    ShortTermStrongBullUSD, MediumTermStrongBullUSD, LongTermStrongBullUSD,
    ShortTermStrongBearUSD, MediumTermStrongBearUSD, LongTermStrongBearUSD
)
from .rsi_signals import (
    RSI_Oversold_USD, RSI_Overbought_USD, RSI_Bullish_USD, RSI_Bearish_USD,
    RSI_Oversold_BTC, RSI_Overbought_BTC, RSI_Bullish_BTC, RSI_Bearish_BTC
)
from .ssr_signals import SSR_RiskOn, SSR_RiskOff
from .volatility_signals import MarkovLowVolatilitySignal, MarkovHighVolatilitySignal
from .donchian_signals import (  # Import the Donchian ensemble signals
    DonchianEnsembleUSD, DonchianEnsembleBTC,
)
from .regime_signals import (  # Import the Regime Detection signals
    BullHighVarianceSignalUSD, BullLowVarianceSignalUSD,
    BearHighVarianceSignalUSD, BearLowVarianceSignalUSD,
    BullHighVarianceSignalBTC, BullLowVarianceSignalBTC,
    BearHighVarianceSignalBTC, BearLowVarianceSignalBTC,
)

# Ensure all signals are properly loaded and registered
from .ensure_signals_loaded import signals as registered_signals

# Log the number of registered signals
import logging
logger = logging.getLogger(__name__)
logger.info(f"Loaded a total of {len(registered_signals)} signals in the registry")

__all__ = [
    'SignalBase',
    'register_signal',
    'get_signal',
    'get_all_signals',
    'get_signal_info',
    'get_signal_names',
    
    # Trend signals - use existing specific classes instead
    'ShortTermNeutralUSD',
    'MediumTermNeutralUSD',
    'LongTermNeutralUSD',
    'ShortTermStrongBullUSD',
    'MediumTermStrongBullUSD',
    'LongTermStrongBullUSD',
    'ShortTermStrongBearUSD',
    'MediumTermStrongBearUSD',
    'LongTermStrongBearUSD',
    
    # RSI signals
    'RSI_Oversold_USD',
    'RSI_Overbought_USD',
    'RSI_Bullish_USD',
    'RSI_Bearish_USD',
    'RSI_Oversold_BTC',
    'RSI_Overbought_BTC',
    'RSI_Bullish_BTC',
    'RSI_Bearish_BTC',
    
    # SSR signals
    'SSR_RiskOn',
    'SSR_RiskOff',
    
    # Volatility signals
    'MarkovLowVolatilitySignal',
    'MarkovHighVolatilitySignal',
    
    # Donchian signals
    'DonchianEnsembleUSD',
    'DonchianEnsembleBTC',
    
    # Regime Detection signals (USD)
    'BullHighVarianceSignalUSD',
    'BullLowVarianceSignalUSD',
    'BearHighVarianceSignalUSD',
    'BearLowVarianceSignalUSD',
    
    # Regime Detection signals (BTC)
    'BullHighVarianceSignalBTC',
    'BullLowVarianceSignalBTC',
    'BearHighVarianceSignalBTC',
    'BearLowVarianceSignalBTC',
] 