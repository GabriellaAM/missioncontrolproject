import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy


class EMACrossoverStrategy(BaseStrategy):
    
    def __init__(self):
        super().__init__("ema_crossover")
        self.fast_period_range = [5, 20]
        self.slow_period_range = [21, 100]
        self.default_params = {'fast_period': 10, 'slow_period': 30}
    
    def calculate_signals(self, data, params):
        pass
    
    def optimize(self, data, train_start, train_end, **kwargs):
        pass