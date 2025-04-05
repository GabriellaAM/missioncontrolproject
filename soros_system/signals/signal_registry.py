"""
Signal registry for managing available signals in the Soros System.

This module provides a central registry for signal classes, allowing discovery
and instantiation of signals by name.
"""

import logging
from typing import Dict, Type, Optional, List

from .signal_base import SignalBase


class SignalRegistry:
    """Registry for available signal classes.
    
    This class maintains a registry of available signal implementations and
    provides methods to register, retrieve, and list signals.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(SignalRegistry, cls).__new__(cls)
            cls._instance._signals = {}
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    def register_signal(self, signal_class: Type[SignalBase]) -> bool:
        """Register a signal class.
        
        Args:
            signal_class: Signal class to register. Must inherit from SignalBase.
            
        Returns:
            bool: True if registration was successful, False otherwise.
        """
        if not issubclass(signal_class, SignalBase):
            self.logger.warning(
                f"Cannot register {signal_class.__name__}: Not a SignalBase subclass"
            )
            return False
        
        signal_name = signal_class.__name__
        
        if signal_name in self._signals:
            self.logger.warning(
                f"Signal {signal_name} already registered, overwriting"
            )
        
        self._signals[signal_name] = signal_class
        self.logger.info(f"Registered signal: {signal_name}")
        return True
    
    def get_signal_class(self, signal_name: str) -> Optional[Type[SignalBase]]:
        """Get a signal class by name.
        
        Args:
            signal_name: Name of the signal class to retrieve.
            
        Returns:
            Type[SignalBase] or None: Signal class if found, None otherwise.
        """
        if signal_name not in self._signals:
            self.logger.warning(f"Signal {signal_name} not found in registry")
            return None
        
        return self._signals.get(signal_name)
    
    def create_signal(self, signal_name: str, params: Optional[Dict] = None) -> Optional[SignalBase]:
        """Create a signal instance by name.
        
        Args:
            signal_name: Name of the signal class to instantiate.
            params: Parameters to pass to the signal constructor.
            
        Returns:
            SignalBase or None: Signal instance if creation was successful, None otherwise.
        """
        signal_class = self.get_signal_class(signal_name)
        
        if signal_class is None:
            return None
        
        try:
            signal_instance = signal_class(params=params)
            return signal_instance
        except Exception as e:
            self.logger.error(f"Error creating signal {signal_name}: {str(e)}")
            return None
    
    def get_all_signals(self) -> List[str]:
        """Get names of all registered signals.
        
        Returns:
            list: List of signal names.
        """
        return list(self._signals.keys())
    
    def get_signal_info(self) -> Dict:
        """Get information about all registered signals.
        
        Returns:
            dict: Dictionary mapping signal names to signal classes.
        """
        return {name: cls for name, cls in self._signals.items()}


# Create a singleton instance
registry = SignalRegistry()


def register_signal(signal_class: Type[SignalBase]) -> Type[SignalBase]:
    """Register a signal class with the registry.
    
    This can be used as a decorator:
    
    @register_signal
    class MySignal(SignalBase):
        ...
    
    Args:
        signal_class: Signal class to register.
        
    Returns:
        Type[SignalBase]: The original signal class (not a boolean).
    """
    registry.register_signal(signal_class)
    return signal_class  # Return the original class, not just True/False


def get_signal(signal_name: str, params: Optional[Dict] = None) -> Optional[SignalBase]:
    """Get a signal instance by name.
    
    Args:
        signal_name: Name of the signal to retrieve.
        params: Parameters to pass to the signal constructor.
        
    Returns:
        SignalBase or None: Signal instance if found, None otherwise.
    """
    return registry.create_signal(signal_name, params)


def get_all_signals() -> List[str]:
    """Get names of all registered signals.
    
    Returns:
        list: List of signal names.
    """
    return registry.get_all_signals()


def get_signal_info() -> Dict:
    """Get information about all registered signals.
    
    Returns:
        dict: Dictionary mapping signal names to signal classes.
    """
    return registry.get_signal_info() 