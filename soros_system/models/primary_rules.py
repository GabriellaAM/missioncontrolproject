"""
Rule-based primary model for meta-labeling framework.

This model uses technical signals directly as trading decisions without ML training,
then applies meta-labeling for position sizing and confidence estimation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
import logging
import json
from datetime import datetime

from ..signals.signal_registry import get_signal, get_all_signals
from .base_primary_model import BasePrimaryModel


class RuleBasedPrimaryModel(BasePrimaryModel):
    """
    Rule-based primary model that converts technical signals directly to trading decisions.
    
    This model doesn't require training - it applies rule logic to combine multiple
    technical signals into directional predictions for the meta-labeling framework.
    """
    
    def __init__(
        self,
        signal_names: Optional[List[str]] = None,
        combination_strategy: str = 'ensemble',
        signal_weights: Optional[Dict[str, float]] = None,
        signal_thresholds: Optional[Dict[str, float]] = None,
        confidence_mode: str = 'agreement',
        config_version: str = '1.0.0'
    ):
        """
        Initialize rule-based primary model.
        
        Args:
            signal_names: List of signal names to use
            combination_strategy: How to combine signals ('ensemble', 'majority_vote', 'weighted', 'single')
            signal_weights: Weights for each signal (for weighted strategy)
            signal_thresholds: Custom thresholds for each signal
            confidence_mode: How to calculate prediction confidence ('agreement', 'strength', 'custom')
            config_version: Version of the rule configuration
        """
        super().__init__(model_type='rule_based')
        
        self.signal_names = signal_names
        self.combination_strategy = combination_strategy
        self.signal_weights = signal_weights or {}
        self.signal_thresholds = signal_thresholds or {}
        self.confidence_mode = confidence_mode
        self.config_version = config_version
        
        # Rule-based models don't need training
        self.is_trained = True
        
        # Store signal instances
        self.signal_instances = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate rule configuration."""
        valid_strategies = ['ensemble', 'majority_vote', 'weighted', 'single']
        if self.combination_strategy not in valid_strategies:
            raise ValueError(f"Invalid combination strategy. Must be one of: {valid_strategies}")
        
        valid_confidence_modes = ['agreement', 'strength', 'custom']
        if self.confidence_mode not in valid_confidence_modes:
            raise ValueError(f"Invalid confidence mode. Must be one of: {valid_confidence_modes}")
        
        if self.combination_strategy == 'weighted' and not self.signal_weights:
            raise ValueError("Weighted strategy requires signal_weights parameter")
        
        if self.combination_strategy == 'single' and len(self.signal_names or []) != 1:
            raise ValueError("Single strategy requires exactly one signal name")
    
    def prepare_signals(
        self,
        asset_data: pd.DataFrame,
        signal_registry: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Generate all required signals for rule-based decisions.
        
        Args:
            asset_data: Asset price/volume data
            signal_registry: Registry of available signals
            
        Returns:
            DataFrame with signal values
        """
        if signal_registry is None:
            signal_registry = get_all_signals()
        
        signals_df = pd.DataFrame(index=asset_data.index)
        
        # Use specified signals or all available signals
        signals_to_use = self.signal_names or list(signal_registry.keys())
        
        for signal_name in signals_to_use:
            try:
                if signal_name in signal_registry:
                    if signal_name not in self.signal_instances:
                        signal_class = signal_registry[signal_name]
                        self.signal_instances[signal_name] = signal_class()
                    
                    signal_instance = self.signal_instances[signal_name]
                    
                    # Generate signal values
                    signal_data = signal_instance.generate_signal(asset_data)
                    
                    if isinstance(signal_data, pd.Series):
                        signals_df[signal_name] = signal_data
                    elif isinstance(signal_data, pd.DataFrame):
                        # If signal returns multiple columns, use the first or aggregate
                        if len(signal_data.columns) == 1:
                            signals_df[signal_name] = signal_data.iloc[:, 0]
                        else:
                            # For multi-column signals, take the mean or majority vote
                            signals_df[signal_name] = signal_data.mean(axis=1)
                    
                    # Apply custom thresholds if specified
                    if signal_name in self.signal_thresholds:
                        threshold = self.signal_thresholds[signal_name]
                        signals_df[signal_name] = (signals_df[signal_name] > threshold).astype(int)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate signal {signal_name}: {e}")
                continue
        
        # Forward fill and handle missing values
        signals_df = signals_df.ffill().fillna(0)
        
        return signals_df
    
    def predict(
        self,
        asset_data: pd.DataFrame,
        return_confidence: bool = True
    ) -> Union[pd.Series, tuple]:
        """
        Generate rule-based predictions from technical signals.
        
        Args:
            asset_data: Asset price/volume data
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions series, optionally with confidence scores
        """
        # Generate all signals
        signals_df = self.prepare_signals(asset_data)
        
        if signals_df.empty:
            raise ValueError("No valid signals generated for prediction")
        
        # Apply combination strategy
        if self.combination_strategy == 'single':
            predictions = signals_df.iloc[:, 0]  # Use first (and only) signal
        elif self.combination_strategy == 'majority_vote':
            predictions = self._apply_majority_vote(signals_df)
        elif self.combination_strategy == 'weighted':
            predictions = self._apply_weighted_combination(signals_df)
        else:  # ensemble (default)
            predictions = self._apply_ensemble_combination(signals_df)
        
        # Convert to directional predictions (-1, 0, 1)
        predictions = self._convert_to_directional(predictions)
        
        if return_confidence:
            confidence = self._calculate_confidence(signals_df, predictions)
            return predictions, confidence
        
        return predictions
    
    def _apply_majority_vote(self, signals_df: pd.DataFrame) -> pd.Series:
        """Apply majority vote combination strategy."""
        # Convert to binary signals if not already
        binary_signals = (signals_df > 0).astype(int)
        
        # Count positive votes
        positive_votes = binary_signals.sum(axis=1)
        total_signals = len(binary_signals.columns)
        
        # Majority vote: 1 if more than half are positive, -1 if less than half, 0 if tie
        predictions = pd.Series(0, index=signals_df.index)
        predictions[positive_votes > total_signals / 2] = 1
        predictions[positive_votes < total_signals / 2] = -1
        
        return predictions
    
    def _apply_weighted_combination(self, signals_df: pd.DataFrame) -> pd.Series:
        """Apply weighted combination strategy."""
        weighted_sum = pd.Series(0, index=signals_df.index)
        total_weight = 0
        
        for signal_name in signals_df.columns:
            weight = self.signal_weights.get(signal_name, 1.0)
            weighted_sum += signals_df[signal_name] * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            predictions = weighted_sum / total_weight
        else:
            predictions = weighted_sum
        
        return predictions
    
    def _apply_ensemble_combination(self, signals_df: pd.DataFrame) -> pd.Series:
        """Apply ensemble combination strategy."""
        # Simple ensemble: equal weight average
        predictions = signals_df.mean(axis=1)
        
        return predictions
    
    def _convert_to_directional(self, predictions: pd.Series) -> pd.Series:
        """Convert continuous predictions to directional (-1, 0, 1)."""
        directional = pd.Series(0, index=predictions.index, name='prediction')
        
        # Define thresholds for directional conversion
        buy_threshold = 0.5
        sell_threshold = -0.5
        
        directional[predictions > buy_threshold] = 1   # Buy
        directional[predictions < sell_threshold] = -1  # Sell
        # Everything else remains 0 (neutral)
        
        return directional
    
    def _calculate_confidence(
        self,
        signals_df: pd.DataFrame,
        predictions: pd.Series
    ) -> pd.Series:
        """Calculate confidence scores for predictions."""
        if self.confidence_mode == 'agreement':
            # Confidence based on signal agreement
            confidence = self._calculate_agreement_confidence(signals_df, predictions)
        elif self.confidence_mode == 'strength':
            # Confidence based on signal strength
            confidence = self._calculate_strength_confidence(signals_df)
        else:  # custom
            # Custom confidence calculation
            confidence = self._calculate_custom_confidence(signals_df, predictions)
        
        return confidence
    
    def _calculate_agreement_confidence(
        self,
        signals_df: pd.DataFrame,
        predictions: pd.Series
    ) -> pd.Series:
        """Calculate confidence based on signal agreement."""
        confidence = pd.Series(0.0, index=predictions.index, name='confidence')
        
        for idx in predictions.index:
            pred_value = predictions[idx]
            if pred_value == 0:
                confidence[idx] = 0.0
                continue
            
            # Count signals agreeing with prediction
            signal_values = signals_df.loc[idx]
            
            if pred_value > 0:  # Buy prediction
                agreeing = (signal_values > 0).sum()
            else:  # Sell prediction
                agreeing = (signal_values < 0).sum()
            
            total_signals = len(signal_values)
            confidence[idx] = agreeing / total_signals if total_signals > 0 else 0.0
        
        return confidence
    
    def _calculate_strength_confidence(self, signals_df: pd.DataFrame) -> pd.Series:
        """Calculate confidence based on signal strength."""
        # Use the average absolute signal strength as confidence
        confidence = signals_df.abs().mean(axis=1)
        confidence.name = 'confidence'
        
        return confidence
    
    def _calculate_custom_confidence(
        self,
        signals_df: pd.DataFrame,
        predictions: pd.Series
    ) -> pd.Series:
        """Calculate custom confidence scores."""
        # Combine agreement and strength
        agreement_conf = self._calculate_agreement_confidence(signals_df, predictions)
        strength_conf = self._calculate_strength_confidence(signals_df)
        
        # Weighted combination
        confidence = 0.7 * agreement_conf + 0.3 * strength_conf
        confidence.name = 'confidence'
        
        return confidence
    
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
        predictions, confidence = self.predict(asset_data, return_confidence=True)
        
        # Filter by confidence and non-neutral predictions
        confident_events = asset_data.index[
            (confidence >= confidence_threshold) & (predictions != 0)
        ]
        
        return pd.DatetimeIndex(confident_events)
    
    def save_config(self, filepath: str) -> None:
        """Save rule configuration to file."""
        config = {
            'model_type': self.model_type,
            'config_version': self.config_version,
            'signal_names': self.signal_names,
            'combination_strategy': self.combination_strategy,
            'signal_weights': self.signal_weights,
            'signal_thresholds': self.signal_thresholds,
            'confidence_mode': self.confidence_mode,
            'created_at': datetime.now().isoformat(),
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Rule configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'RuleBasedPrimaryModel':
        """Load rule configuration from file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Remove metadata fields
        config.pop('created_at', None)
        config.pop('model_type', None)
        config.pop('is_trained', None)
        
        return cls(**config)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance for rule-based model."""
        if not self.signal_weights:
            # Equal importance for unweighted models
            signals = self.signal_names or []
            importance = pd.Series(
                [1.0 / len(signals)] * len(signals),
                index=signals,
                name='importance'
            )
        else:
            # Use signal weights as importance
            total_weight = sum(abs(w) for w in self.signal_weights.values())
            importance = pd.Series({
                signal: abs(weight) / total_weight
                for signal, weight in self.signal_weights.items()
            }, name='importance')
        
        return importance.sort_values(ascending=False)
    
    def update_config(
        self,
        signal_weights: Optional[Dict[str, float]] = None,
        signal_thresholds: Optional[Dict[str, float]] = None,
        combination_strategy: Optional[str] = None
    ) -> None:
        """Update rule configuration."""
        if signal_weights is not None:
            self.signal_weights.update(signal_weights)
        
        if signal_thresholds is not None:
            self.signal_thresholds.update(signal_thresholds)
        
        if combination_strategy is not None:
            self.combination_strategy = combination_strategy
            self._validate_config()
        
        # Increment patch version
        version_parts = self.config_version.split('.')
        patch = int(version_parts[2]) + 1
        self.config_version = f"{version_parts[0]}.{version_parts[1]}.{patch}"
        
        self.logger.info(f"Configuration updated to version {self.config_version}")


# Factory functions for common rule-based configurations

def create_rsi_donchian_model(
    rsi_threshold: float = 30,
    donchian_periods: List[int] = [20, 55],
    combination: str = 'ensemble'
) -> RuleBasedPrimaryModel:
    """Create RSI + Donchian rule-based model."""
    return RuleBasedPrimaryModel(
        signal_names=['RSI_Bullish_USD', 'DonchianEnsembleUSD'],
        combination_strategy=combination,
        signal_thresholds={
            'RSI_Bullish_USD': 0.5,  # RSI signals are already binary
            'DonchianEnsembleUSD': 0.5
        },
        confidence_mode='agreement'
    )


def create_momentum_model(
    signals: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None
) -> RuleBasedPrimaryModel:
    """Create momentum-based rule model."""
    default_signals = ['RSI_Bullish_USD', 'TrendMomentumBullish', 'DonchianEnsembleUSD']
    default_weights = {'RSI_Bullish_USD': 0.3, 'TrendMomentumBullish': 0.4, 'DonchianEnsembleUSD': 0.3}
    
    return RuleBasedPrimaryModel(
        signal_names=signals or default_signals,
        combination_strategy='weighted',
        signal_weights=weights or default_weights,
        confidence_mode='strength'
    )


def create_single_signal_model(signal_name: str) -> RuleBasedPrimaryModel:
    """Create single signal rule-based model."""
    return RuleBasedPrimaryModel(
        signal_names=[signal_name],
        combination_strategy='single',
        confidence_mode='strength'
    )