"""
ZenML training steps for primary and meta models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from zenml import step
import logging
import mlflow
import mlflow.sklearn

from ..meta_labeling.primary_model import PrimaryDirectionModel
from ..meta_labeling.meta_model import MetaModel
from ..meta_labeling.triple_barrier import TripleBarrierLabeler
from ..signals.signal_registry import get_all_signals


@step
def train_primary_model(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    model_type: str = 'random_forest',
    signal_names: Optional[List[str]] = None,
    labeling_params: Optional[Dict[str, Any]] = None
) -> Tuple[PrimaryDirectionModel, Dict[str, Any]]:
    """
    Train the primary direction prediction model.
    
    Args:
        train_data: Training data
        val_data: Validation data
        model_type: Type of model to train
        signal_names: List of signals to use (None for all)
        labeling_params: Triple-barrier labeling parameters
        
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    # Initialize primary model
    primary_model = PrimaryDirectionModel(
        model_type=model_type,
        signal_names=signal_names
    )
    
    # Get signal registry
    signal_registry = get_all_signals()
    
    # Prepare features
    train_features = primary_model.prepare_features(train_data, signal_registry)
    val_features = primary_model.prepare_features(val_data, signal_registry)
    
    # Create training labels using triple-barrier method
    labeling_config = labeling_params or {
        'profit_taking_multiple': 2.0,
        'stop_loss_multiple': 1.0,
        'max_holding_period': 5
    }
    
    # Get signal events (use all available data points for now)
    signal_events = train_features.index[::10]  # Sample every 10th point
    
    train_labels_df = primary_model.create_training_labels(
        train_data['close'], signal_events, labeling_config
    )
    
    # Train the model
    training_metrics = primary_model.train(
        train_features,
        train_labels_df['label'],
        optimize_params=True
    )
    
    # Validation performance
    val_predictions = primary_model.predict(val_features, return_probabilities=False)
    
    # Create validation labels
    val_signal_events = val_features.index[::10]
    val_labels_df = primary_model.create_training_labels(
        val_data['close'], val_signal_events, labeling_config
    )
    
    # Calculate validation accuracy
    common_idx = val_predictions.index.intersection(val_labels_df.index)
    val_accuracy = (
        val_predictions.reindex(common_idx) == 
        val_labels_df['label'].reindex(common_idx)
    ).mean()
    
    training_metrics['validation_accuracy'] = val_accuracy
    
    # Log metrics to MLflow
    with mlflow.start_run(run_name="primary_model_training", nested=True):
        mlflow.log_params({
            'model_type': model_type,
            'signal_names': signal_names or 'all',
            **labeling_config
        })
        mlflow.log_metrics(training_metrics)
        
        # Log feature importance
        if primary_model.feature_importance is not None:
            for feature, importance in primary_model.feature_importance.head(10).items():
                mlflow.log_metric(f"feature_importance_{feature}", importance)
        
        # Save model
        mlflow.sklearn.log_model(primary_model.model, "primary_model")
    
    logging.info(f"Primary model trained with accuracy: {training_metrics.get('cv_accuracy_mean', 0):.4f}")
    
    return primary_model, training_metrics


@step
def train_meta_model(
    primary_model: PrimaryDirectionModel,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    forward_returns: pd.DataFrame,
    market_regime_data: pd.DataFrame,
    model_type: str = 'random_forest'
) -> Tuple[MetaModel, Dict[str, Any]]:
    """
    Train the meta-model for position sizing and confidence estimation.
    
    Args:
        primary_model: Trained primary model
        train_data: Training data
        val_data: Validation data
        forward_returns: Forward return data
        market_regime_data: Market regime features
        model_type: Type of meta-model to train
        
    Returns:
        Tuple of (trained_meta_model, training_metrics)
    """
    # Initialize meta-model
    meta_model = MetaModel(model_type=model_type)
    
    # Get signal registry for features
    signal_registry = get_all_signals()
    
    # Generate primary model predictions on training data
    train_features = primary_model.prepare_features(train_data, signal_registry)
    train_predictions = primary_model.predict(train_features, return_probabilities=False)
    
    # Prepare meta-features
    train_meta_features = meta_model.prepare_meta_features(
        train_data,
        train_predictions,
        market_regime_data
    )
    
    # Create meta-labels (1 if primary prediction correct, 0 otherwise)
    train_meta_labels = meta_model.create_meta_labels(
        train_predictions,
        forward_returns['forward_return_5d'],  # Use 5-day forward returns
        prediction_threshold=0.1
    )
    
    # Train meta-model
    training_metrics = meta_model.train(
        train_meta_features,
        train_meta_labels,
        optimize_params=True
    )
    
    # Validation performance
    val_features = primary_model.prepare_features(val_data, signal_registry)
    val_predictions = primary_model.predict(val_features, return_probabilities=False)
    
    val_meta_features = meta_model.prepare_meta_features(
        val_data,
        val_predictions,
        market_regime_data
    )
    
    val_confidence = meta_model.predict_confidence(val_meta_features)
    
    # Calculate validation meta-labels
    val_forward_returns = forward_returns.reindex(val_data.index)
    val_meta_labels = meta_model.create_meta_labels(
        val_predictions,
        val_forward_returns['forward_return_5d'],
        prediction_threshold=0.1
    )
    
    # Validation accuracy
    common_idx = val_confidence.index.intersection(val_meta_labels.index)
    if len(common_idx) > 0:
        val_binary_predictions = (val_confidence.reindex(common_idx) > 0.5).astype(int)
        val_meta_accuracy = (
            val_binary_predictions == val_meta_labels.reindex(common_idx)
        ).mean()
        training_metrics['validation_meta_accuracy'] = val_meta_accuracy
    
    # Log metrics to MLflow
    with mlflow.start_run(run_name="meta_model_training", nested=True):
        mlflow.log_params({
            'meta_model_type': model_type,
            'primary_model_type': primary_model.model_type
        })
        mlflow.log_metrics(training_metrics)
        
        # Log feature importance
        if meta_model.feature_importance is not None:
            for feature, importance in meta_model.feature_importance.head(10).items():
                mlflow.log_metric(f"meta_feature_importance_{feature}", importance)
        
        # Save model
        mlflow.sklearn.log_model(meta_model.model, "meta_model")
    
    logging.info(f"Meta-model trained with precision: {training_metrics.get('cv_precision_mean', 0):.4f}")
    
    return meta_model, training_metrics


@step
def create_combined_predictions(
    primary_model: PrimaryDirectionModel,
    meta_model: MetaModel,
    test_data: pd.DataFrame,
    market_regime_data: pd.DataFrame,
    confidence_threshold: float = 0.6
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Create combined predictions from primary and meta models.
    
    Args:
        primary_model: Trained primary model
        meta_model: Trained meta model
        test_data: Test data
        market_regime_data: Market regime features
        confidence_threshold: Minimum confidence for signal generation
        
    Returns:
        Tuple of (primary_predictions, confidence_scores, position_sizes)
    """
    # Get signal registry
    signal_registry = get_all_signals()
    
    # Generate primary predictions
    test_features = primary_model.prepare_features(test_data, signal_registry)
    primary_predictions = primary_model.predict(test_features, return_probabilities=False)
    
    # Generate meta-features and confidence scores
    meta_features = meta_model.prepare_meta_features(
        test_data,
        primary_predictions,
        market_regime_data
    )
    
    confidence_scores = meta_model.predict_confidence(meta_features)
    
    # Calculate position sizes based on confidence
    position_sizes = meta_model.calculate_position_sizes(
        primary_predictions,
        confidence_scores,
        confidence_threshold=confidence_threshold
    )
    
    # Log prediction statistics
    with mlflow.start_run(run_name="combined_predictions", nested=True):
        mlflow.log_metrics({
            'total_predictions': len(primary_predictions),
            'confident_predictions': (confidence_scores >= confidence_threshold).sum(),
            'mean_confidence': confidence_scores.mean(),
            'mean_position_size': np.abs(position_sizes).mean(),
            'signal_rate': (position_sizes != 0).mean()
        })
    
    logging.info(f"Generated {len(primary_predictions)} predictions with {(confidence_scores >= confidence_threshold).sum()} confident signals")
    
    return primary_predictions, confidence_scores, position_sizes