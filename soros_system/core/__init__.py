"""
Core module for unified interfaces to Soros System.

This module provides simplified high-level interfaces to the
Soros System components for daily operations.
"""

# Import important components for direct access
from .asset_data import AssetData
from .signal_data import SignalData
from .portfolio_analyzer import PortfolioAnalyzer

__all__ = ['AssetData', 'SignalData', 'PortfolioAnalyzer'] 