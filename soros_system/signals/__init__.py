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
)

__all__ = [
    'SignalBase',
    'register_signal',
    'get_signal',
    'get_all_signals',
    'get_signal_info',
] 