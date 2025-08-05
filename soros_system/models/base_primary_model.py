"""
Base class for all primary models in the meta-labeling framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
import logging


class BasePrimaryModel(ABC):
    """
    Abstract base class for all primary models.
    
    Primary models generate directional predictions that serve as input
    to the meta-labeling framework for position sizing and confidence estimation.
    """
    
    def __init__(self, model_type: str):
        """
        Initialize base primary model.
        
        Args:
            model_type: Type of model ('ml_based', 'rule_based', 'hybrid')
        """
        self.model_type = model_type
        self.is_trained = False
        self.training_metrics = {}
        self.model_version = "1.0.0"
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def predict(
        self, 
        asset_data: pd.DataFrame,
        return_confidence: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Generate predictions from the primary model.
        
        Args:
            asset_data: Asset price/volume data
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions series, optionally with confidence scores
        """
        pass
    
    @abstractmethod
    def get_signal_events(
        self,
        asset_data: pd.DataFrame,
        confidence_threshold: float = 0.5
    ) -> pd.DatetimeIndex:
        """
        Get signal events where the model generates confident predictions.
        
        Args:
            asset_data: Asset data
            confidence_threshold: Minimum confidence for signal generation
            
        Returns:
            DatetimeIndex of signal events
        """
        pass
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance or signal weights.
        
        Returns:
            Series with feature/signal importance scores
        """
        # Default implementation returns empty series
        return pd.Series(dtype=float, name='importance')
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'model_type': self.model_type,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'class_name': self.__class__.__name__
        }
    
    def validate_input(self, asset_data: pd.DataFrame) -> bool:
        """
        Validate input data for prediction.
        
        Args:
            asset_data: Input data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if asset_data is None or asset_data.empty:
            self.logger.warning("Empty asset data provided")
            return False
        
        if 'close' not in asset_data.columns:
            self.logger.warning("Missing 'close' column in asset data")
            return False
        
        if len(asset_data) < self.get_min_required_samples():
            self.logger.warning(
                f"Insufficient data samples: {len(asset_data)}, "
                f"need at least {self.get_min_required_samples()}"
            )
            return False
        
        return True
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for the model."""
        return 30  # Default minimum
    
    def set_version(self, version: str) -> None:
        """Set model version."""
        self.model_version = version
        self.logger.info(f"Model version updated to {version}")
    
    def increment_version(self, increment_type: str = 'patch') -> str:
        """
        Increment model version.
        
        Args:
            increment_type: Type of increment ('major', 'minor', 'patch')
            
        Returns:
            New version string
        """
        parts = self.model_version.split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        if increment_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif increment_type == 'minor':
            minor += 1
            patch = 0
        else:  # patch
            patch += 1
        
        new_version = f"{major}.{minor}.{patch}"
        self.set_version(new_version)
        return new_version