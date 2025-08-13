from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseStrategy(ABC):
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_required_features(self) -> Dict[str, Any]:
        """
        Define what features this strategy needs for FeatureLoader.build_feature_set()
        
        Returns dict with keys:
            - crypto_assets: List of crypto assets needed
            - fred_indicators: Dict of FRED indicators {key: alias}
            - yahoo_tickers: Dict of Yahoo tickers {ticker: alias}
            - calculated_features: Dict of calculated features {feature: alias}
        """
        pass
    
    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate trading signals based on strategy logic."""
        pass
    
    @abstractmethod
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
        """Optimize strategy parameters."""
        pass