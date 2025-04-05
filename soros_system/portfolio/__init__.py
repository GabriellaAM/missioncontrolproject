"""
Portfolio management package for Soros System.

This package contains tools for managing portfolios, combining signals,
and backtesting strategies.
"""

from .signal_combiner import SignalCombiner

__all__ = [
    'SignalCombiner',
]
