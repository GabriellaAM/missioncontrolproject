import sys
import os
import pandas as pd
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.meta_models.base_meta_strategy import MetaStrategy

class TopologicalMRValidator(MetaStrategy):
    
    "Contextualizer for Primary Model Signals using TDA."
    
    def __init__(self, asset: str = 'bitcoin', primary_run_id: str = None):
        name = f"topological_mr_validator_{asset}"
        super().__init__(name, asset, primary_run_id)
        
    def get_required_features(self) -> Dict[str, Any]:
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': ['creditSpreads', 'treasury5YInflationExpectation'],
            'yahoo_tickers': ['vix', 'move'],
            'calculated_features': ['rty_ym_ratio', 'yieldCurveRegime']
        }
    
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return data