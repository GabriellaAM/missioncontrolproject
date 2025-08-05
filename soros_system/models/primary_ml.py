"""
ML-based primary model for meta-labeling framework.

This model uses technical signals as features for machine learning models,
requiring training and providing the first stage of López de Prado's meta-labeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import logging
import joblib
from datetime import datetime

from ..signals.signal_registry import get_signal, get_all_signals
from ..meta_labeling.triple_barrier import TripleBarrierLabeler
from .base_primary_model import BasePrimaryModel


class MLBasedPrimaryModel(BasePrimaryModel):
    """
    ML-based primary model that uses technical signals as features.
    
    This model requires training and uses machine learning algorithms to predict
    price direction from technical signal features.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        signal_names: Optional[List[str]] = None,
        target_accuracy: float = 0.55,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize ML-based primary model.
        
        Args:
            model_type: Type of ML model ('random_forest', 'gradient_boost', 'logistic')
            signal_names: List of signal names to use as features
            target_accuracy: Target accuracy for direction prediction
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        super().__init__(model_type='ml_based')
        
        self.ml_model_type = model_type
        self.signal_names = signal_names
        self.target_accuracy = target_accuracy
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.model = None
        self.feature_importance = None
        self.scaler = None
        
        self.logger = logging.getLogger(__name__)
    
    def _get_base_model(self) -> Any:
        """Get the base ML model based on model_type."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            'logistic': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        }
        return models[self.ml_model_type]
    
    def prepare_features(
        self,
        asset_data: pd.DataFrame,
        signal_registry: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Prepare feature matrix from asset data and signals.
        
        Args:
            asset_data: Asset price/volume data
            signal_registry: Dictionary of available signals
            
        Returns:
            Feature matrix with technical indicators
        """
        if signal_registry is None:
            signal_registry = get_all_signals()
        
        features = pd.DataFrame(index=asset_data.index)
        
        # Use specified signals or all available signals
        signals_to_use = self.signal_names or list(signal_registry.keys())
        
        for signal_name in signals_to_use:
            try:
                if signal_name in signal_registry:
                    signal_class = signal_registry[signal_name]
                    signal_instance = signal_class()
                    
                    # Generate signal values
                    signal_data = signal_instance.generate_signal(asset_data)
                    
                    if isinstance(signal_data, pd.Series):
                        features[f'{signal_name}_value'] = signal_data
                    elif isinstance(signal_data, pd.DataFrame):
                        for col in signal_data.columns:
                            features[f'{signal_name}_{col}'] = signal_data[col]
                            
            except Exception as e:
                self.logger.warning(f"Failed to generate signal {signal_name}: {e}")
                continue
        
        # Add basic price features
        if 'close' in asset_data.columns:
            # Price momentum features
            for period in [5, 10, 20]:
                features[f'return_{period}d'] = asset_data['close'].pct_change(period)
                features[f'volatility_{period}d'] = (
                    asset_data['close'].rolling(period).std() / 
                    asset_data['close'].rolling(period).mean()
                )
        
        # Add volume features if available
        if 'volume' in asset_data.columns:
            features['volume_ma_ratio'] = (
                asset_data['volume'] / asset_data['volume'].rolling(20).mean()
            )
        
        # Forward fill and drop NaN values
        features = features.ffill().dropna()
        
        return features
    
    def create_training_labels(
        self,
        price_series: pd.Series,
        signal_events: pd.DatetimeIndex,
        labeling_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Create training labels using triple-barrier method.
        
        Args:
            price_series: Price time series
            signal_events: Event timestamps for labeling
            labeling_params: Parameters for triple-barrier labeling
            
        Returns:
            DataFrame with training labels
        """
        params = labeling_params or {
            'profit_taking_multiple': 2.0,
            'stop_loss_multiple': 1.0,
            'max_holding_period': 5
        }
        
        labeler = TripleBarrierLabeler(**params)
        return labeler.label_events(price_series, signal_events)
    
    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        optimize_params: bool = True,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the ML-based primary model.
        
        Args:
            features: Feature matrix
            labels: Target labels (-1, 0, 1)
            optimize_params: Whether to optimize hyperparameters
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics and results
        """
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        X = features.reindex(common_index)
        y = labels.reindex(common_index)
        
        # Remove neutral labels (focus on directional prediction)
        directional_mask = y != 0
        X_directional = X[directional_mask]
        y_directional = y[directional_mask]
        
        if len(X_directional) < 100:
            raise ValueError("Insufficient non-neutral samples for training")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        if optimize_params:
            self.model = self._optimize_hyperparameters(
                X_directional, y_directional, tscv
            )
        else:
            self.model = self._get_base_model()
            self.model.fit(X_directional, y_directional)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_directional.columns
            ).sort_values(ascending=False)
        
        # Evaluate model performance
        metrics = self._evaluate_model(X_directional, y_directional, tscv)
        self.training_metrics = metrics
        
        # Model is now trained
        self.is_trained = True
        self.increment_version('minor')  # Increment version after training
        
        return metrics
    
    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: TimeSeriesSplit
    ) -> Any:
        """Optimize model hyperparameters using grid search."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20]
            },
            'gradient_boost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0]
            },
            'logistic': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        base_model = self._get_base_model()
        param_grid = param_grids[self.ml_model_type]
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: TimeSeriesSplit
    ) -> Dict[str, Any]:
        """Evaluate model performance using time series cross-validation."""
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train fold model
            fold_model = self._get_base_model()
            fold_model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = fold_model.predict(X_test)
            
            accuracies.append(accuracy_score(y_test, y_pred))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted', zero_division='warn'
            )
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        return {
            'cv_accuracy_mean': np.mean(accuracies),
            'cv_accuracy_std': np.std(accuracies),
            'cv_precision_mean': np.mean(precisions),
            'cv_recall_mean': np.mean(recalls),
            'cv_f1_mean': np.mean(f1_scores),
            'target_accuracy_met': np.mean(accuracies) >= self.target_accuracy
        }
    
    def predict(
        self,
        asset_data: pd.DataFrame,
        return_confidence: bool = False
    ) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
        """
        Generate predictions from the ML model.
        
        Args:
            asset_data: Asset price/volume data
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Predictions series, optionally with confidence scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not self.validate_input(asset_data):
            raise ValueError("Invalid input data")
        
        # Prepare features
        features = self.prepare_features(asset_data)
        
        if features.empty:
            raise ValueError("No features generated from asset data")
        
        # Generate predictions
        predictions = self.model.predict(features)
        pred_series = pd.Series(predictions, index=features.index, name='ml_prediction')
        
        if return_confidence:
            # Use prediction probabilities as confidence
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)
                # Use max probability as confidence
                confidence = pd.Series(
                    probabilities.max(axis=1),
                    index=features.index,
                    name='ml_confidence'
                )
            else:
                # For models without probabilities, use constant confidence
                confidence = pd.Series(0.7, index=features.index, name='ml_confidence')
            
            return pred_series, confidence
        
        return pred_series
    
    def get_signal_events(
        self,
        asset_data: pd.DataFrame,
        confidence_threshold: float = 0.6
    ) -> pd.DatetimeIndex:
        """
        Get signal events where the model is confident about direction.
        
        Args:
            asset_data: Asset data
            confidence_threshold: Minimum confidence for signal generation
            
        Returns:
            DatetimeIndex of confident signal events
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals")
        
        predictions, confidence = self.predict(asset_data, return_confidence=True)
        
        # Filter by confidence and non-neutral predictions
        confident_events = asset_data.index[
            (confidence >= confidence_threshold) & (predictions != 0)
        ]
        
        return pd.DatetimeIndex(confident_events)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from trained model."""
        if self.feature_importance is not None:
            return self.feature_importance
        else:
            return pd.Series(dtype=float, name='importance')
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_info': self.get_model_info(),
            'feature_importance': self.feature_importance,
            'signal_names': self.signal_names,
            'ml_model_type': self.ml_model_type,
            'training_metrics': self.training_metrics,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"ML model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'MLBasedPrimaryModel':
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(
            model_type=model_data['ml_model_type'],
            signal_names=model_data['signal_names']
        )
        
        # Restore model state
        instance.model = model_data['model']
        instance.feature_importance = model_data.get('feature_importance')
        instance.training_metrics = model_data.get('training_metrics', {})
        instance.is_trained = True
        instance.model_version = model_data['model_info'].get('model_version', '1.0.0')
        
        return instance
    
    def get_min_required_samples(self) -> int:
        """Get minimum required samples for ML model."""
        return 100  # ML models need more data than rule-based models