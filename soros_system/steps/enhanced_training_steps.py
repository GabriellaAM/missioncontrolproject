"""
Enhanced ZenML training steps supporting multiple primary model types.

This module provides conditional training steps that handle ML-based, rule-based,
and hybrid primary models with proper model versioning and registry integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from zenml import step
import logging
import mlflow

from ..models.primary_ml import MLBasedPrimaryModel
from ..models.primary_rules import RuleBasedPrimaryModel
from ..models.primary_hybrid import HybridPrimaryModel
from ..models.model_registry import get_model_registry, register_model
from ..meta_labeling.meta_model import MetaModel
from ..meta_labeling.triple_barrier import TripleBarrierLabeler
from ..signals.signal_registry import get_all_signals


@step
def create_primary_model(
    model_mode: str = 'ml_based',
    model_config: Optional[Dict[str, Any]] = None,
    asset_id: str = 'default'
) -> Union[MLBasedPrimaryModel, RuleBasedPrimaryModel, HybridPrimaryModel]:
    """
    Create primary model based on specified mode.
    
    Args:
        model_mode: Type of primary model ('ml_based', 'rule_based', 'hybrid')
        model_config: Configuration parameters for the model
        asset_id: Asset identifier for logging
        
    Returns:
        Initialized primary model instance
    """
    config = model_config or {}
    
    with mlflow.start_run(run_name=f"create_primary_model_{asset_id}", nested=True):
        mlflow.log_params({
            'model_mode': model_mode,
            'asset_id': asset_id,
            **config
        })
        
        if model_mode == 'ml_based':
            model = MLBasedPrimaryModel(**config)
        elif model_mode == 'rule_based':
            model = RuleBasedPrimaryModel(**config)
        elif model_mode == 'hybrid':
            model = HybridPrimaryModel(**config)
        else:
            raise ValueError(f"Invalid model mode: {model_mode}")
        
        logging.info(f"Created {model_mode} primary model for {asset_id}")
        
        mlflow.log_param('model_version', model.model_version)
        mlflow.log_param('requires_training', model_mode in ['ml_based', 'hybrid'])
    
    return model


@step
def conditional_primary_training(
    primary_model: Union[MLBasedPrimaryModel, RuleBasedPrimaryModel, HybridPrimaryModel],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    training_labels: Optional[pd.DataFrame] = None,
    training_params: Optional[Dict[str, Any]] = None,
    force_retrain: bool = False
) -> Tuple[Union[MLBasedPrimaryModel, RuleBasedPrimaryModel, HybridPrimaryModel], Dict[str, Any]]:
    """
    Conditionally train primary model based on its type.
    
    Args:
        primary_model: Primary model instance
        train_data: Training data
        val_data: Validation data
        training_labels: Labels for ML/hybrid training
        training_params: Training parameters
        force_retrain: Force retraining even if model is already trained
        
    Returns:
        Tuple of (trained_model, training_metrics)
    """
    params = training_params or {}
    model_type = primary_model.model_type
    
    with mlflow.start_run(run_name=f"train_primary_{model_type}", nested=True):
        mlflow.log_params({
            'model_type': model_type,
            'force_retrain': force_retrain,
            'data_samples': len(train_data),
            **params
        })
        
        training_metrics = {}
        
        if model_type == 'rule_based':
            # Rule-based models don't need training
            logging.info("Rule-based model - no training required")
            training_metrics = {
                'model_ready': True,
                'training_required': False,
                'config_version': primary_model.config_version
            }
            
        elif model_type == 'ml_based':
            # ML models always need training
            if not primary_model.is_trained or force_retrain:
                if training_labels is None:
                    # Create training labels using triple-barrier method
                    training_labels = _create_training_labels(
                        train_data, params.get('labeling_params', {})
                    )
                
                # Prepare features
                features = primary_model.prepare_features(train_data)
                
                # Train model
                training_metrics = primary_model.train(
                    features,
                    training_labels['label'],
                    **params
                )
                
                logging.info(f"ML model trained with accuracy: {training_metrics.get('cv_accuracy_mean', 0):.4f}")
            else:
                logging.info("ML model already trained - skipping training")
                training_metrics = {'model_ready': True, 'training_skipped': True}
        
        elif model_type == 'hybrid':
            # Hybrid models need conditional training
            if not primary_model.is_trained or force_retrain:
                if training_labels is None:
                    training_labels = _create_training_labels(
                        train_data, params.get('labeling_params', {})
                    )
                
                # Train hybrid model (only ML component needs training)
                training_metrics = primary_model.train(
                    train_data,
                    training_labels['label'] if training_labels is not None else None,
                    train_ml_model=True,
                    ml_training_params=params
                )
                
                logging.info("Hybrid model trained successfully")
            else:
                logging.info("Hybrid model already trained - skipping training")
                training_metrics = {'model_ready': True, 'training_skipped': True}
        
        # Log training metrics
        mlflow.log_metrics(training_metrics)
        
        # Log model info
        model_info = primary_model.get_model_info()
        mlflow.log_params({
            'model_version': model_info['model_version'],
            'is_trained': model_info['is_trained']
        })
    
    return primary_model, training_metrics


@step
def register_primary_model(
    primary_model: Union[MLBasedPrimaryModel, RuleBasedPrimaryModel, HybridPrimaryModel],
    asset_id: str,
    performance_metrics: Dict[str, float],
    training_data_hash: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Register trained primary model in the model registry.
    
    Args:
        primary_model: Trained primary model
        asset_id: Asset identifier
        performance_metrics: Model performance metrics
        training_data_hash: Hash of training data
        tags: Additional tags for model
        
    Returns:
        Model ID in registry
    """
    registry = get_model_registry()
    
    with mlflow.start_run(run_name=f"register_model_{asset_id}", nested=True):
        model_id = registry.register_model(
            model=primary_model,
            asset_id=asset_id,
            performance_metrics=performance_metrics,
            model_subtype=getattr(primary_model, 'ml_model_type', primary_model.model_type),
            tags=tags,
            training_data_hash=training_data_hash
        )
        
        mlflow.log_params({
            'model_id': model_id,
            'asset_id': asset_id,
            'model_type': primary_model.model_type,
            'model_version': primary_model.model_version
        })
        
        mlflow.log_metrics(performance_metrics)
        
        logging.info(f"Model registered with ID: {model_id}")
    
    return model_id


@step
def enhanced_meta_model_training(
    primary_model: Union[MLBasedPrimaryModel, RuleBasedPrimaryModel, HybridPrimaryModel],
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    forward_returns: pd.DataFrame,
    market_regime_data: pd.DataFrame,
    meta_model_type: str = 'random_forest',
    meta_training_params: Optional[Dict[str, Any]] = None
) -> Tuple[MetaModel, Dict[str, Any]]:
    """
    Enhanced meta-model training that works with any primary model type.
    
    Args:
        primary_model: Trained primary model of any type
        train_data: Training data
        val_data: Validation data
        forward_returns: Forward return data
        market_regime_data: Market regime features
        meta_model_type: Type of meta-model
        meta_training_params: Meta-model training parameters
        
    Returns:
        Tuple of (trained_meta_model, training_metrics)
    """
    params = meta_training_params or {}
    
    with mlflow.start_run(run_name="train_meta_model_enhanced", nested=True):
        # Initialize meta-model
        meta_model = MetaModel(model_type=meta_model_type, **params)
        
        # Generate primary model predictions on training data
        train_predictions = primary_model.predict(train_data, return_confidence=False)
        
        # Prepare meta-features (same process regardless of primary model type)
        train_meta_features = meta_model.prepare_meta_features(
            train_data,
            train_predictions,
            market_regime_data
        )
        
        # Create meta-labels
        train_meta_labels = meta_model.create_meta_labels(
            train_predictions,
            forward_returns['forward_return_5d'],
            prediction_threshold=params.get('prediction_threshold', 0.1)
        )
        
        # Train meta-model
        training_metrics = meta_model.train(
            train_meta_features,
            train_meta_labels,
            optimize_params=params.get('optimize_params', True)
        )
        
        # Validation performance
        val_predictions = primary_model.predict(val_data, return_confidence=False)
        val_meta_features = meta_model.prepare_meta_features(
            val_data,
            val_predictions,
            market_regime_data
        )
        
        val_confidence = meta_model.predict_confidence(val_meta_features)
        
        # Log metrics
        mlflow.log_params({
            'primary_model_type': primary_model.model_type,
            'meta_model_type': meta_model_type,
            'training_samples': len(train_meta_labels)
        })
        mlflow.log_metrics(training_metrics)
        
        logging.info(f"Meta-model trained with precision: {training_metrics.get('cv_precision_mean', 0):.4f}")
    
    return meta_model, training_metrics


@step
def model_performance_evaluation(
    primary_model: Union[MLBasedPrimaryModel, RuleBasedPrimaryModel, HybridPrimaryModel],
    meta_model: MetaModel,
    test_data: pd.DataFrame,
    market_regime_data: pd.DataFrame,
    evaluation_params: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Comprehensive model performance evaluation.
    
    Args:
        primary_model: Trained primary model
        meta_model: Trained meta-model
        test_data: Test data
        market_regime_data: Market regime data
        evaluation_params: Evaluation parameters
        
    Returns:
        Performance metrics dictionary
    """
    params = evaluation_params or {}
    
    with mlflow.start_run(run_name="model_evaluation", nested=True):
        # Generate predictions
        primary_predictions = primary_model.predict(test_data, return_confidence=False)
        
        # Generate meta-features and confidence
        meta_features = meta_model.prepare_meta_features(
            test_data,
            primary_predictions,
            market_regime_data
        )
        
        confidence_scores = meta_model.predict_confidence(meta_features)
        
        # Calculate position sizes
        position_sizes = meta_model.calculate_position_sizes(
            primary_predictions,
            confidence_scores,
            confidence_threshold=params.get('confidence_threshold', 0.6)
        )
        
        # Calculate strategy returns
        if 'close' in test_data.columns:
            price_returns = test_data['close'].pct_change().shift(-1)
            strategy_returns = position_sizes.shift(1) * price_returns
            strategy_returns = strategy_returns.fillna(0).dropna()
        else:
            strategy_returns = pd.Series(0, index=position_sizes.index)
        
        # Calculate performance metrics
        performance_metrics = _calculate_comprehensive_metrics(
            strategy_returns, position_sizes, confidence_scores
        )
        
        # Add model-specific metrics
        performance_metrics.update({
            'primary_model_type': primary_model.model_type,
            'primary_model_version': primary_model.model_version,
            'meta_model_precision': meta_model.training_metrics.get('cv_precision_mean', 0),
            'total_signals': (position_sizes != 0).sum(),
            'data_points_evaluated': len(test_data)
        })
        
        # Log all metrics
        mlflow.log_metrics(performance_metrics)
        mlflow.log_params({
            'evaluation_period': f"{test_data.index[0]} to {test_data.index[-1]}",
            'primary_model_type': primary_model.model_type
        })
        
        logging.info(f"Model evaluation complete. Sharpe ratio: {performance_metrics.get('sharpe_ratio', 0):.4f}")
    
    return performance_metrics


@step
def select_best_model_version(
    asset_id: str,
    metric: str = 'oos_sharpe',
    model_type: Optional[str] = None,
    min_performance_threshold: float = 0.5
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Select the best model version for an asset.
    
    Args:
        asset_id: Asset identifier
        metric: Metric to optimize
        model_type: Filter by model type
        min_performance_threshold: Minimum performance threshold
        
    Returns:
        Tuple of (model_id, model_metadata) or None
    """
    registry = get_model_registry()
    
    with mlflow.start_run(run_name=f"select_best_model_{asset_id}", nested=True):
        best_metadata = registry.get_best_model(asset_id, metric, model_type)
        
        if best_metadata is None:
            logging.warning(f"No models found for {asset_id}")
            return None
        
        best_score = best_metadata.performance_metrics.get(metric, 0)
        
        # Check performance threshold
        if best_score < min_performance_threshold:
            logging.warning(
                f"Best model for {asset_id} below threshold: {best_score:.4f} < {min_performance_threshold}"
            )
            return None
        
        mlflow.log_params({
            'asset_id': asset_id,
            'best_model_id': best_metadata.model_id,
            'best_model_type': best_metadata.model_type,
            'selection_metric': metric
        })
        
        mlflow.log_metrics({
            'best_score': best_score,
            'performance_threshold': min_performance_threshold
        })
        
        logging.info(f"Selected best model for {asset_id}: {best_metadata.model_id} (score: {best_score:.4f})")
    
    return best_metadata.model_id, best_metadata.performance_metrics


# Helper functions

def _create_training_labels(
    data: pd.DataFrame,
    labeling_params: Dict[str, Any]
) -> pd.DataFrame:
    """Create training labels using triple-barrier method."""
    params = {
        'profit_taking_multiple': 2.0,
        'stop_loss_multiple': 1.0,
        'max_holding_period': 5,
        **labeling_params
    }
    
    labeler = TripleBarrierLabeler(**params)
    
    # Use every 10th data point as signal events for labeling
    signal_events = data.index[::10]
    
    return labeler.label_events(data['close'], signal_events)


def _calculate_comprehensive_metrics(
    returns: pd.Series,
    positions: pd.Series,
    confidence: pd.Series
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    if returns.empty or returns.isna().all():
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'signal_rate': 0.0,
            'avg_confidence': 0.0
        }
    
    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    mean_return = returns.mean() * 252  # Annualized
    volatility = returns.std() * np.sqrt(252)  # Annualized
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0
    
    # Drawdown calculation
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Signal metrics
    win_rate = (returns > 0).mean()
    signal_rate = (positions != 0).mean()
    avg_confidence = confidence.mean()
    
    # Risk metrics
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'volatility': float(volatility),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'signal_rate': float(signal_rate),
        'avg_confidence': float(avg_confidence),
        'mean_return': float(mean_return)
    }