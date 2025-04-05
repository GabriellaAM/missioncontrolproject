#!/usr/bin/env python
"""
Debug script to diagnose signal class import issue.
"""

import os
import sys
import logging
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_import():
    """Check if SignalBase and signal classes can be imported."""
    try:
        # Try importing SignalBase
        from soros_system.signals.signal_base import SignalBase
        logger.info("Successfully imported SignalBase")
        
        # Try importing register_signal
        from soros_system.signals.signal_registry import register_signal
        logger.info("Successfully imported register_signal")
        logger.info(f"Type of register_signal: {type(register_signal)}")
        
        # Try importing trend signals directly (not via decorator)
        import soros_system.signals.trend_signals as trend_module
        logger.info("Successfully imported trend_signals module")
        logger.info(f"ShortTermTrendSignal type: {type(trend_module.ShortTermTrendSignal)}")
        
        # Print out all attributes of ShortTermTrendSignal
        logger.info("ShortTermTrendSignal attributes:")
        for attr in dir(trend_module.ShortTermTrendSignal):
            if not attr.startswith('__'):
                logger.info(f"  {attr}: {type(getattr(trend_module.ShortTermTrendSignal, attr))}")
        
        # Try using ShortTermTrendSignal without params
        try:
            signal = trend_module.ShortTermTrendSignal()
            logger.info("Successfully created ShortTermTrendSignal instance without params")
        except Exception as e:
            logger.error(f"Error creating ShortTermTrendSignal without params: {str(e)}")
        
        # Try using ShortTermTrendSignal with params
        try:
            signal = trend_module.ShortTermTrendSignal(params={'quote_type': 'USD', 'trend_values': [1, 2]})
            logger.info("Successfully created ShortTermTrendSignal instance with params")
        except Exception as e:
            logger.error(f"Error creating ShortTermTrendSignal with params: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            
    except Exception as e:
        logger.error(f"Error importing signals: {str(e)}")

if __name__ == "__main__":
    debug_import() 