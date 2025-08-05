"""
Ensure that all signal classes are properly loaded and registered.

This module imports all signal modules and explicitly registers all signals
to make sure they are available in the registry.
"""

import logging
from importlib import import_module

# Set up logging
logger = logging.getLogger(__name__)

def ensure_signals_loaded():
    """
    Ensure all signal modules are loaded and their signals are registered.
    """
    # List of modules containing signal classes
    signal_modules = [
        'soros_system.signals.trend_signals',
        'soros_system.signals.rsi_signals',
        'soros_system.signals.ssr_signals',
        'soros_system.signals.volatility_signals'
    ]
    
    total_signals = 0
    
    # Import each module to trigger signal registration
    for module_name in signal_modules:
        try:
            module = import_module(module_name)
            # Count registered signals in this module
            if hasattr(module, '__all__'):
                signal_count = sum(1 for name in module.__all__ if not name.startswith('_'))
                total_signals += signal_count
                logger.info(f"Loaded {signal_count} signals from {module_name}")
            else:
                logger.info(f"Loaded module {module_name}")
        except ImportError as e:
            logger.error(f"Failed to import signal module {module_name}: {e}")
    
    # Import the signal registry to verify registration
    from .signal_registry import get_all_signals
    
    registered_signals = get_all_signals()
    logger.info(f"Total signals registered: {len(registered_signals)}")
    
    return registered_signals

# Run the function when this module is imported
signals = ensure_signals_loaded() 