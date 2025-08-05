"""
Meta-model for position sizing and confidence estimation.

This module implements the second stage of López de Prado's meta-labeling 
framework: predicting the probability that the primary model prediction is 
correct, enabling intelligent position sizing and signal filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import logging

from .triple_barrier import TripleBarrierLabeler


class MetaModel:
    """
    Meta-model that predicts the probability of primary model correctness.
    
    This model learns to estimate when the primary model predictions are
    likely to be correct, enabling dynamic position sizing and risk management.
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        target_precision: float = 0.65,
        cv_folds: int = 5,
        scale_features: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the meta-model.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boost', 'logistic')
            target_precision: Target precision for meta-predictions
            cv_folds: Number of cross-validation folds
            scale_features: Whether to scale features
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.target_precision = target_precision
        self.cv_folds = cv_folds
        self.scale_features = scale_features
        self.random_state = random_state
        
        self.model = None
        self.scaler = None if not scale_features else StandardScaler()
        self.feature_importance = None
        self.training_metrics = {}
        
        self.logger = logging.getLogger(__name__)
    
    def _get_base_model(self) -> Any:
        """Get the base model based on model_type."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
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
    
    def prepare_meta_features(
        self,
        asset_data: pd.DataFrame,
        primary_predictions: pd.Series,
        market_regime_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare feature matrix for meta-model training.
        
        Args:
            asset_data: Asset price/volume data
            primary_predictions: Primary model predictions
            market_regime_features: Additional market regime features
            
        Returns:
            Meta-feature matrix
        """
        features = pd.DataFrame(index=asset_data.index)
        
        # Primary model features
        features['primary_prediction'] = primary_predictions
        features['primary_confidence'] = np.abs(primary_predictions)
        
        # Market microstructure features
        if 'close' in asset_data.columns:
            close = asset_data['close']
            
            # Volatility features
            for window in [5, 10, 20]:
                returns = close.pct_change()
                features[f'volatility_{window}d'] = returns.rolling(window).std()
                features[f'skewness_{window}d'] = returns.rolling(window).skew()
                features[f'kurtosis_{window}d'] = returns.rolling(window).kurt()
            
            # Trend strength features
            for window in [10, 20, 50]:
                features[f'trend_strength_{window}d'] = (
                    close / close.rolling(window).mean() - 1
                )
            
            # Price momentum features
            for period in [3, 7, 14]:
                features[f'momentum_{period}d'] = close.pct_change(period)
        
        # Volume-based features (if available)
        if 'volume' in asset_data.columns:
            volume = asset_data['volume']
            
            # Volume trends
            for window in [5, 10, 20]:
                vol_ma = volume.rolling(window).mean()
                features[f'volume_ratio_{window}d'] = volume / vol_ma
                features[f'volume_volatility_{window}d'] = (
                    volume.rolling(window).std() / vol_ma
                )
        
        # Time-based features
        features['hour'] = asset_data.index.hour
        features['day_of_week'] = asset_data.index.dayofweek
        features['month'] = asset_data.index.month
        
        # Market regime features (if provided)
        if market_regime_features is not None:
            common_idx = features.index.intersection(market_regime_features.index)
            for col in market_regime_features.columns:
                features.loc[common_idx, f'regime_{col}'] = (
                    market_regime_features.loc[common_idx, col]
                )
        
        # Interaction features
        features['pred_vol_interaction'] = (
            features['primary_confidence'] * 
            features.get('volatility_20d', 0)
        )
        
        # Forward fill and handle missing values
        features = features.ffill().fillna(0)
        
        return features
    
    def create_meta_labels(
        self,
        primary_predictions: pd.Series,
        actual_returns: pd.Series,
        prediction_threshold: float = 0.1
    ) -> pd.Series:
        """
        Create meta-labels for training.
        
        Args:
            primary_predictions: Primary model predictions
            actual_returns: Actual forward returns
            prediction_threshold: Minimum prediction magnitude threshold
            
        Returns:
            Meta-labels (1 if primary prediction correct, 0 otherwise)
        """
        # Align data
        common_index = primary_predictions.index.intersection(actual_returns.index)
        preds = primary_predictions.reindex(common_index)
        returns = actual_returns.reindex(common_index)
        
        # Create binary labels based on prediction correctness
        # Only consider confident predictions (above threshold)
        confident_mask = np.abs(preds) > prediction_threshold
        
        # Meta-label: 1 if prediction direction matches return direction
        correct_predictions = (
            (preds > 0) & (returns > 0) |
            (preds < 0) & (returns < 0) |
            (np.abs(preds) <= prediction_threshold) & (np.abs(returns) < prediction_threshold)
        )
        
        meta_labels = correct_predictions.astype(int)
        
        # Only use confident predictions for training
        return meta_labels[confident_mask]
    
    def train(
        self,
        meta_features: pd.DataFrame,
        meta_labels: pd.Series,
        optimize_params: bool = True
    ) -> Dict[str, Any]:
        """
        Train the meta-model.
        
        Args:
            meta_features: Meta-feature matrix
            meta_labels: Meta-labels (0 or 1)
            optimize_params: Whether to optimize hyperparameters
            
        Returns:
            Training metrics and results
        """
        # Align features and labels
        common_index = meta_features.index.intersection(meta_labels.index)
        X = meta_features.reindex(common_index)
        y = meta_labels.reindex(common_index)
        
        if len(X) < 100:
            raise ValueError("Insufficient samples for meta-model training")
        
        # Scale features if required
        if self.scale_features:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
        else:
            X_scaled = X
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        if optimize_params:
            self.model = self._optimize_hyperparameters(X_scaled, y, tscv)
        else:
            self.model = self._get_base_model()
            self.model.fit(X_scaled, y)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_scaled.columns
            ).sort_values(ascending=False)
        
        # Evaluate model performance
        metrics = self._evaluate_model(X_scaled, y, tscv)
        self.training_metrics = metrics
        
        return metrics
    
    def _optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: TimeSeriesSplit
    ) -> Any:
        """Optimize meta-model hyperparameters using grid search."""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [4, 6, 8],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10]
            },
            'gradient_boost': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.7, 0.8, 0.9]
            },
            'logistic': {
                'C': [0.1, 1.0, 10.0, 100.0],
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
            scoring='average_precision',  # Better for imbalanced data
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        self.logger.info(f"Meta-model best parameters: {grid_search.best_params_}")
        self.logger.info(f"Meta-model best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: TimeSeriesSplit
    ) -> Dict[str, Any]:
        """Evaluate meta-model performance using time series cross-validation."""
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        aucs = []
        avg_precisions = []
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train fold model
            fold_model = self._get_base_model()
            fold_model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = fold_model.predict(X_test)
            y_prob = fold_model.predict_proba(X_test)[:, 1]
            
            accuracies.append(accuracy_score(y_test, y_pred))
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            
            if len(np.unique(y_test)) == 2:  # Only calculate AUC if both classes present
                aucs.append(roc_auc_score(y_test, y_prob))
                avg_precisions.append(average_precision_score(y_test, y_prob))
        
        return {
            'cv_accuracy_mean': np.mean(accuracies),
            'cv_accuracy_std': np.std(accuracies),
            'cv_precision_mean': np.mean(precisions),
            'cv_recall_mean': np.mean(recalls),
            'cv_f1_mean': np.mean(f1_scores),
            'cv_auc_mean': np.mean(aucs) if aucs else 0.0,
            'cv_avg_precision_mean': np.mean(avg_precisions) if avg_precisions else 0.0,
            'target_precision_met': np.mean(precisions) >= self.target_precision
        }
    
    def predict_confidence(
        self,
        meta_features: pd.DataFrame
    ) -> pd.Series:
        """
        Predict confidence scores for primary model predictions.
        
        Args:
            meta_features: Meta-feature matrix
            
        Returns:
            Confidence scores (0 to 1)
        """
        if self.model is None:
            raise ValueError("Meta-model must be trained before making predictions")
        
        # Scale features if required
        if self.scale_features and self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(meta_features),
                index=meta_features.index,
                columns=meta_features.columns
            )
        else:
            X_scaled = meta_features
        
        # Get probability of correct prediction (class 1)
        confidence_scores = self.model.predict_proba(X_scaled)[:, 1]
        
        return pd.Series(
            confidence_scores,
            index=meta_features.index,
            name='meta_confidence'
        )
    
    def calculate_position_sizes(
        self,
        primary_predictions: pd.Series,
        confidence_scores: pd.Series,
        base_position_size: float = 0.1,
        confidence_threshold: float = 0.55,
        max_position_size: float = 0.2
    ) -> pd.Series:
        """
        Calculate position sizes based on primary predictions and meta-confidence.
        
        Args:
            primary_predictions: Primary model predictions
            confidence_scores: Meta-model confidence scores
            base_position_size: Base position size for confident predictions
            confidence_threshold: Minimum confidence for position taking
            max_position_size: Maximum position size
            
        Returns:
            Position sizes (positive for long, negative for short)
        """
        # Align data
        common_index = primary_predictions.index.intersection(confidence_scores.index)
        preds = primary_predictions.reindex(common_index)
        confidence = confidence_scores.reindex(common_index)
        
        # Calculate position sizes
        position_sizes = pd.Series(0.0, index=common_index, name='position_size')
        
        # Only take positions where confidence exceeds threshold
        confident_mask = confidence >= confidence_threshold
        
        if confident_mask.any():
            # Scale position size by confidence
            scaled_sizes = base_position_size * (confidence[confident_mask] - confidence_threshold) / (1 - confidence_threshold)
            scaled_sizes = np.clip(scaled_sizes, 0, max_position_size)
            
            # Apply direction from primary model
            position_sizes.loc[confident_mask] = (
                np.sign(preds[confident_mask]) * scaled_sizes
            )
        
        return position_sizes
    
    def filter_signals(
        self,
        primary_signals: pd.Series,
        confidence_scores: pd.Series,
        confidence_threshold: float = 0.6
    ) -> pd.Series:
        """
        Filter primary model signals using meta-model confidence.
        
        Args:
            primary_signals: Primary model signals
            confidence_scores: Meta-model confidence scores
            confidence_threshold: Minimum confidence for signal acceptance
            
        Returns:
            Filtered signals
        """
        common_index = primary_signals.index.intersection(confidence_scores.index)
        signals = primary_signals.reindex(common_index)
        confidence = confidence_scores.reindex(common_index)
        
        # Filter signals by confidence
        filtered_signals = signals.copy()
        filtered_signals[confidence < confidence_threshold] = 0
        
        return filtered_signals