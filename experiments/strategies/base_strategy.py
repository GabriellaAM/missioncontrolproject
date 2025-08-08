from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseStrategy(ABC):
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate_signals(self, data, params):
        pass
    
    def optimize(self, data, train_start, train_end, **kwargs):
        pass