"""
Hybrid primary model combining ML-based and rule-based approaches.

This model uses both machine learning and rule-based logic, allowing for
flexible combination of different prediction methodologies in the meta-labeling framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import json
import joblib
from datetime import datetime

from .base_primary_model import BasePrimaryModel
from .primary_ml import MLBasedPrimaryModel
from .primary_rules import RuleBasedPrimaryModel


class HybridPrimaryModel(BasePrimaryModel):
    """
    Hybrid primary model combining ML-based and rule-based predictions.
    
    This model combines predictions from both ML models (using signals as features)
    and rule-based models (using signals as direct decisions) to create a robust
    ensemble approach for the meta-labeling framework.
    """
    
    def __init__(
        self,
        ml_model_config: Optional[Dict[str, Any]] = None,
        rule_model_config: Optional[Dict[str, Any]] = None,
        combination_strategy: str = 'weighted',
        ml_weight: float = 0.6,
        rule_weight: float = 0.4,
        agreement_threshold: float = 0.5,
        confidence_mode: str = 'ensemble'
    ):
        """
        Initialize hybrid primary model.
        
        Args:
            ml_model_config: Configuration for ML-based model
            rule_model_config: Configuration for rule-based model
            combination_strategy: How to combine predictions ('weighted', 'majority', 'agreement', 'adaptive')
            ml_weight: Weight for ML model predictions (for weighted strategy)
            rule_weight: Weight for rule-based predictions (for weighted strategy)
            agreement_threshold: Minimum agreement required for prediction (for agreement strategy)
            confidence_mode: How to calculate confidence ('ensemble', 'max', 'agreement')
        """
        super().__init__(model_type='hybrid')
        
        self.combination_strategy = combination_strategy
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.agreement_threshold = agreement_threshold
        self.confidence_mode = confidence_mode
        
        # Initialize component models
        self.ml_model = MLBasedPrimaryModel(**(ml_model_config or {}))
        self.rule_model = RuleBasedPrimaryModel(**(rule_model_config or {}))
        
        # Normalize weights
        total_weight = ml_weight + rule_weight
        if total_weight > 0:
            self.ml_weight = ml_weight / total_weight
            self.rule_weight = rule_weight / total_weight
        
        self.logger = logging.getLogger(__name__)
        self._validate_config()
    
    def _validate_config(self):
        """Validate hybrid model configuration."""
        valid_strategies = ['weighted', 'majority', 'agreement', 'adaptive']
        if self.combination_strategy not in valid_strategies:
            raise ValueError(f"Invalid combination strategy. Must be one of: {valid_strategies}")
        
        valid_confidence_modes = ['ensemble', 'max', 'agreement']
        if self.confidence_mode not in valid_confidence_modes:
            raise ValueError(f"Invalid confidence mode. Must be one of: {valid_confidence_modes}")
        
        if not (0 <= self.agreement_threshold <= 1):
            raise ValueError("Agreement threshold must be between 0 and 1")
    
    def train(
        self,
        asset_data: pd.DataFrame,
        training_labels: Optional[pd.Series] = None,
        train_ml_model: bool = True,
        ml_training_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train the hybrid model (only ML component needs training).
        
        Args:
            asset_data: Asset price/volume data
            training_labels: Labels for ML model training
            train_ml_model: Whether to train the ML component
            ml_training_params: Parameters for ML model training
            
        Returns:
            Combined training metrics
        """
        training_results = {}
        
        if train_ml_model:
            # Train ML component if labels provided
            if training_labels is not None:
                features = self.ml_model.prepare_features(asset_data)
                ml_metrics = self.ml_model.train(
                    features,
                    training_labels,
                    **(ml_training_params or {})
                )
                training_results['ml_metrics'] = ml_metrics
                self.logger.info(f"ML model trained with accuracy: {ml_metrics.get('cv_accuracy_mean', 0):.4f}")
            else:
                self.logger.warning("No training labels provided for ML model")
        
        # Rule model doesn't need training (already configured)
        training_results['rule_model_ready'] = self.rule_model.is_trained
        
        # Update hybrid model status
        self.is_trained = self.ml_model.is_trained and self.rule_model.is_trained
        if self.is_trained:
            self.increment_version('minor')
        
        self.training_metrics = training_results
        return training_results
    
    def predict(
        self,
        asset_data: pd.DataFrame,
        return_confidence: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Generate hybrid predictions combining ML and rule-based approaches.
        
        Args:
            asset_data: Asset price/volume data
            return_confidence: Whether to return confidence scores
            
        Returns:
            Combined predictions, optionally with confidence scores
        """
        if not self.validate_input(asset_data):
            raise ValueError("Invalid input data")
        
        # Get predictions from both models
        try:
            ml_pred, ml_conf = self.ml_model.predict(asset_data, return_confidence=True)
        except Exception as e:
            self.logger.warning(f"ML model prediction failed: {e}")
            ml_pred = pd.Series(0, index=asset_data.index)
            ml_conf = pd.Series(0.0, index=asset_data.index)
        
        try:
            rule_pred, rule_conf = self.rule_model.predict(asset_data, return_confidence=True)
        except Exception as e:
            self.logger.warning(f"Rule model prediction failed: {e}")
            rule_pred = pd.Series(0, index=asset_data.index)
            rule_conf = pd.Series(0.0, index=asset_data.index)
        
        # Align predictions
        common_index = ml_pred.index.intersection(rule_pred.index)
        ml_pred_aligned = ml_pred.reindex(common_index, fill_value=0)
        rule_pred_aligned = rule_pred.reindex(common_index, fill_value=0)
        ml_conf_aligned = ml_conf.reindex(common_index, fill_value=0.0)
        rule_conf_aligned = rule_conf.reindex(common_index, fill_value=0.0)
        
        # Combine predictions based on strategy
        if self.combination_strategy == 'weighted':
            combined_pred = self._weighted_combination(ml_pred_aligned, rule_pred_aligned)
        elif self.combination_strategy == 'majority':
            combined_pred = self._majority_combination(ml_pred_aligned, rule_pred_aligned)
        elif self.combination_strategy == 'agreement':
            combined_pred = self._agreement_combination(
                ml_pred_aligned, rule_pred_aligned,
                ml_conf_aligned, rule_conf_aligned
            )
        else:  # adaptive
            combined_pred = self._adaptive_combination(
                ml_pred_aligned, rule_pred_aligned,
                ml_conf_aligned, rule_conf_aligned
            )
        
        if return_confidence:
            # Calculate combined confidence
            combined_conf = self._calculate_combined_confidence(
                ml_pred_aligned, rule_pred_aligned,
                ml_conf_aligned, rule_conf_aligned,
                combined_pred
            )
            return combined_pred, combined_conf
        
        return combined_pred
    
    def _weighted_combination(
        self,
        ml_pred: pd.Series,
        rule_pred: pd.Series
    ) -> pd.Series:
        """Combine predictions using weighted average."""
        combined = self.ml_weight * ml_pred + self.rule_weight * rule_pred
        
        # Convert to directional predictions
        directional = pd.Series(0, index=combined.index, name='hybrid_prediction')
        directional[combined > 0.33] = 1    # Buy threshold
        directional[combined < -0.33] = -1  # Sell threshold
        
        return directional
    
    def _majority_combination(
        self,
        ml_pred: pd.Series,
        rule_pred: pd.Series
    ) -> pd.Series:
        """Combine predictions using majority vote."""
        combined = pd.Series(0, index=ml_pred.index, name='hybrid_prediction')
        
        # Both agree on direction
        agree_buy = (ml_pred > 0) & (rule_pred > 0)
        agree_sell = (ml_pred < 0) & (rule_pred < 0)
        
        combined[agree_buy] = 1
        combined[agree_sell] = -1
        # Disagreement or neutral results in 0
        
        return combined
    
    def _agreement_combination(
        self,
        ml_pred: pd.Series,
        rule_pred: pd.Series,
        ml_conf: pd.Series,
        rule_conf: pd.Series
    ) -> pd.Series:
        """Combine predictions requiring minimum agreement threshold."""
        combined = pd.Series(0, index=ml_pred.index, name='hybrid_prediction')
        
        # Calculate agreement strength
        agreement_strength = self._calculate_agreement_strength(
            ml_pred, rule_pred, ml_conf, rule_conf
        )
        
        # Only make predictions where agreement is strong enough
        strong_agreement = agreement_strength >= self.agreement_threshold
        
        # For strong agreement, use weighted combination
        weighted_pred = self.ml_weight * ml_pred + self.rule_weight * rule_pred
        
        combined[strong_agreement & (weighted_pred > 0.25)] = 1
        combined[strong_agreement & (weighted_pred < -0.25)] = -1
        
        return combined
    
    def _adaptive_combination(
        self,
        ml_pred: pd.Series,
        rule_pred: pd.Series,
        ml_conf: pd.Series,
        rule_conf: pd.Series
    ) -> pd.Series:
        """Adaptive combination based on relative confidence."""
        combined = pd.Series(0, index=ml_pred.index, name='hybrid_prediction')
        
        # Dynamically weight based on confidence
        total_conf = ml_conf + rule_conf
        dynamic_ml_weight = ml_conf / (total_conf + 1e-8)  # Avoid division by zero
        dynamic_rule_weight = rule_conf / (total_conf + 1e-8)
        
        # Weighted combination with dynamic weights
        weighted_pred = dynamic_ml_weight * ml_pred + dynamic_rule_weight * rule_pred
        
        # Apply thresholds
        combined[weighted_pred > 0.4] = 1
        combined[weighted_pred < -0.4] = -1
        
        return combined
    
    def _calculate_agreement_strength(
        self,
        ml_pred: pd.Series,
        rule_pred: pd.Series,
        ml_conf: pd.Series,
        rule_conf: pd.Series
    ) -> pd.Series:
        """Calculate agreement strength between models."""
        agreement = pd.Series(0.0, index=ml_pred.index)
        
        # Direction agreement
        same_direction = (
            (ml_pred > 0) & (rule_pred > 0) |
            (ml_pred < 0) & (rule_pred < 0) |
            (ml_pred == 0) & (rule_pred == 0)
        )
        
        # Base agreement from direction
        agreement[same_direction] = 0.5
        
        # Boost agreement based on confidence
        confidence_boost = (ml_conf + rule_conf) / 2
        agreement += 0.5 * confidence_boost
        
        return np.clip(agreement, 0, 1)
    
    def _calculate_combined_confidence(
        self,
        ml_pred: pd.Series,
        rule_pred: pd.Series,
        ml_conf: pd.Series,
        rule_conf: pd.Series,
        combined_pred: pd.Series
    ) -> pd.Series:
        """Calculate confidence for combined predictions."""
        if self.confidence_mode == 'ensemble':
            # Average confidence weighted by model weights
            confidence = self.ml_weight * ml_conf + self.rule_weight * rule_conf
        elif self.confidence_mode == 'max':
            # Use maximum confidence
            confidence = pd.concat([ml_conf, rule_conf], axis=1).max(axis=1)
        else:  # agreement
            # Confidence based on model agreement
            agreement_strength = self._calculate_agreement_strength(
                ml_pred, rule_pred, ml_conf, rule_conf
            )
            base_confidence = (ml_conf + rule_conf) / 2
            confidence = agreement_strength * base_confidence
        
        confidence.name = 'hybrid_confidence'
        return confidence
    
    def get_signal_events(
        self,
        asset_data: pd.DataFrame,
        confidence_threshold: float = 0.6
    ) -> pd.DatetimeIndex:
        """
        Get signal events where the hybrid model is confident.
        
        Args:
            asset_data: Asset data
            confidence_threshold: Minimum confidence for signal generation
            
        Returns:
            DatetimeIndex of confident signal events
        """
        predictions, confidence = self.predict(asset_data, return_confidence=True)
        
        # Filter by confidence and non-neutral predictions
        confident_events = asset_data.index[
            (confidence >= confidence_threshold) & (predictions != 0)
        ]
        
        return pd.DatetimeIndex(confident_events)
    
    def get_feature_importance(self) -> pd.Series:
        """Get combined feature importance from both models."""
        ml_importance = self.ml_model.get_feature_importance()
        rule_importance = self.rule_model.get_feature_importance()
        
        # Combine importance scores
        all_features = set(ml_importance.index).union(set(rule_importance.index))
        combined_importance = pd.Series(0.0, index=list(all_features))
        
        for feature in all_features:
            ml_score = ml_importance.get(feature, 0.0) * self.ml_weight
            rule_score = rule_importance.get(feature, 0.0) * self.rule_weight
            combined_importance[feature] = ml_score + rule_score
        
        return combined_importance.sort_values(ascending=False)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive hybrid model information."""
        base_info = super().get_model_info()
        base_info.update({
            'combination_strategy': self.combination_strategy,
            'ml_weight': self.ml_weight,
            'rule_weight': self.rule_weight,
            'ml_model_info': self.ml_model.get_model_info(),
            'rule_model_info': self.rule_model.get_model_info(),
            'confidence_mode': self.confidence_mode
        })
        return base_info
    
    def save_model(self, filepath: str) -> None:
        """Save hybrid model configuration and trained components."""
        # Save ML model if trained
        if self.ml_model.is_trained:
            ml_model_path = filepath.replace('.pkl', '_ml.pkl')
            self.ml_model.save_model(ml_model_path)
        
        # Save rule model configuration
        rule_config_path = filepath.replace('.pkl', '_rules.json')
        self.rule_model.save_config(rule_config_path)
        
        # Save hybrid model metadata
        hybrid_data = {
            'model_info': self.get_model_info(),
            'combination_strategy': self.combination_strategy,
            'ml_weight': self.ml_weight,
            'rule_weight': self.rule_weight,
            'agreement_threshold': self.agreement_threshold,
            'confidence_mode': self.confidence_mode,
            'ml_model_path': ml_model_path if self.ml_model.is_trained else None,
            'rule_config_path': rule_config_path,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(hybrid_data, f, indent=2)
        
        self.logger.info(f"Hybrid model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'HybridPrimaryModel':
        """Load hybrid model from saved files."""
        with open(filepath, 'r') as f:
            hybrid_data = json.load(f)
        
        # Load ML model if available
        ml_model = None
        if hybrid_data.get('ml_model_path'):
            try:
                ml_model = MLBasedPrimaryModel.load_model(hybrid_data['ml_model_path'])
            except Exception as e:
                logging.warning(f"Failed to load ML model: {e}")
        
        # Load rule model
        rule_model = RuleBasedPrimaryModel.load_config(hybrid_data['rule_config_path'])
        
        # Create hybrid instance
        instance = cls(
            ml_model_config=None,  # Use loaded model
            rule_model_config=None,  # Use loaded model
            combination_strategy=hybrid_data['combination_strategy'],
            ml_weight=hybrid_data['ml_weight'],
            rule_weight=hybrid_data['rule_weight'],
            agreement_threshold=hybrid_data['agreement_threshold'],
            confidence_mode=hybrid_data['confidence_mode']
        )
        
        # Set loaded models
        if ml_model:
            instance.ml_model = ml_model
        instance.rule_model = rule_model
        
        # Update training status
        instance.is_trained = instance.ml_model.is_trained and instance.rule_model.is_trained
        instance.model_version = hybrid_data['model_info'].get('model_version', '1.0.0')
        
        return instance
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for hybrid model."""
        return max(
            self.ml_model.get_min_required_samples(),
            self.rule_model.get_min_required_samples()
        )


# Factory functions for common hybrid configurations

def create_balanced_hybrid_model(
    ml_model_type: str = 'random_forest',
    rule_combination: str = 'ensemble',
    signal_names: Optional[List[str]] = None
) -> HybridPrimaryModel:
    """Create balanced hybrid model with equal ML/rule weights."""
    return HybridPrimaryModel(
        ml_model_config={
            'model_type': ml_model_type,
            'signal_names': signal_names
        },
        rule_model_config={
            'signal_names': signal_names,
            'combination_strategy': rule_combination
        },
        combination_strategy='weighted',
        ml_weight=0.5,
        rule_weight=0.5
    )


def create_ml_heavy_hybrid_model(
    ml_model_type: str = 'gradient_boost',
    signal_names: Optional[List[str]] = None
) -> HybridPrimaryModel:
    """Create ML-heavy hybrid model."""
    return HybridPrimaryModel(
        ml_model_config={
            'model_type': ml_model_type,
            'signal_names': signal_names
        },
        rule_model_config={
            'signal_names': signal_names,
            'combination_strategy': 'majority_vote'
        },
        combination_strategy='weighted',
        ml_weight=0.75,
        rule_weight=0.25
    )


def create_adaptive_hybrid_model(
    signal_names: Optional[List[str]] = None
) -> HybridPrimaryModel:
    """Create adaptive hybrid model with confidence-based weighting."""
    return HybridPrimaryModel(
        ml_model_config={
            'model_type': 'random_forest',
            'signal_names': signal_names
        },
        rule_model_config={
            'signal_names': signal_names,
            'combination_strategy': 'weighted'
        },
        combination_strategy='adaptive',
        confidence_mode='agreement'
    )