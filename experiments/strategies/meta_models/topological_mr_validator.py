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
            'calculated_features': {'yieldCurveRegime': ['regime', 'spread_2s10s'],
                                    'rty_ym_ratio': 'value'}
        }
    
    def get_normalization_config(self) -> Dict:
        """
        Configure EWMA normalization for contextual features.
        Primary signals and labels are excluded from normalization.
        """
        return {
            'exclude': [
                'signal',  # Primary model signal - never normalize
                'label',   # Triple barrier label - never normalize
                'timestamp',  # Time index
                'barrier_touched', 'days_to_barrier', 'return_at_barrier',  # Label metadata
                'yieldCurveRegime'  # Already categorical/binary
            ]
        }
    
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        return data
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str,
                 n_trials: int = 1000, **kwargs) -> Dict:
        """
        Placeholder optimization for meta-model.
        Real implementation would train TDA-based contextualizer.
        """
        # For now, return dummy results to maintain pipeline compatibility
        return {
            'best_params': {},
            'best_value': 0.0,
            'study': None
        }