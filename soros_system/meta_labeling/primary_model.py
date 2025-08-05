"""
Primary model for direction prediction in the meta-labeling framework.

This module implements the first stage of meta-labeling: predicting the direction
of price movements using existing technical signals, optimized for directional
accuracy rather than P&L.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import logging

from ..signals.signal_registry import get_signal, get_all_signals
from .triple_barrier import TripleBarrierLabeler


class PrimaryDirectionModel:
    """
    Primary model that predicts price direction using technical signals.
    
    This model is optimized for directional accuracy and serves as the first
    stage in López de Prado's meta-labeling framework.
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
        Initialize the primary direction model.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boost', 'logistic')
            signal_names: List of signal names to use (if None, uses all available)
            target_accuracy: Target accuracy for direction prediction
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.signal_names = signal_names
        self.target_accuracy = target_accuracy
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.model = None
        self.feature_importance = None
        self.training_metrics = {}
        self.scaler = None
        
        self.logger = logging.getLogger(__name__)
        
    def _get_base_model(self) -> Any:
        """Get the base model based on model_type."""
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
        return models[self.model_type]
    
    def prepare_features(
        self,
        asset_data: pd.DataFrame,
        signal_registry: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Prepare feature matrix from asset data and signals.
        
        Args:
            asset_data: Asset price/volume data
            signal_registry: Dictionary of available signals
            
        Returns:
            Feature matrix with technical indicators
        """
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
        Train the primary direction model.
        
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
        param_grid = param_grids[self.model_type]
        
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
        features: pd.DataFrame,
        return_probabilities: bool = True
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Generate predictions from the primary model.
        
        Args:
            features: Feature matrix
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predictions (and probabilities if requested)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(features)
        pred_series = pd.Series(predictions, index=features.index, name='primary_prediction')
        
        if return_probabilities:
            probabilities = self.model.predict_proba(features)
            prob_df = pd.DataFrame(
                probabilities,
                index=features.index,
                columns=[f'prob_class_{cls}' for cls in self.model.classes_]
            )
            return pred_series, prob_df
        
        return pred_series
    
    def get_signal_events(
        self,
        features: pd.DataFrame,
        confidence_threshold: float = 0.6
    ) -> pd.DatetimeIndex:
        """
        Get signal events where the model is confident about direction.
        
        Args:
            features: Feature matrix
            confidence_threshold: Minimum probability for signal generation
            
        Returns:
            DatetimeIndex of confident signal events
        """
        if self.model is None:
            raise ValueError("Model must be trained before generating signals")
        
        _, probabilities = self.predict(features, return_probabilities=True)
        
        # Find events where model is confident (high probability for any class)
        max_probs = probabilities.max(axis=1)
        confident_events = features.index[max_probs >= confidence_threshold]
        
        return pd.DatetimeIndex(confident_events)