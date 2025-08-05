"""
Soros System - Crypto Trading Decision Support System
"""

# Easy imports for notebooks
from .notebook_setup import setup_analyzer, get_portfolio_assets, quick_setup
from .config import load_config

__version__ = "0.1.0"
__all__ = ['setup_analyzer', 'get_portfolio_assets', 'quick_setup', 'load_config']