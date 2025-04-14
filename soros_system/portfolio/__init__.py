"""
Portfolio management package for Soros System.

This package contains tools for managing portfolios and backtesting strategies 
using a simplified binary signal approach.
"""

# Remove SignalCombiner import, which is not needed in the simplified approach
# from .signal_combiner import SignalCombiner
# Replace EventDrivenBacktester with the new Backtester
from .backtester import Backtester
from .portfolio_manager import PortfolioManager

__all__ = [
    # 'SignalCombiner',  # Remove from exports
    'Backtester',  # Add new class
    'PortfolioManager',
]
