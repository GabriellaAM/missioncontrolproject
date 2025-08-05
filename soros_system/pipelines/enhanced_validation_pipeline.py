"""
Enhanced ZenML validation pipeline supporting multiple primary model types.

This pipeline provides comprehensive asset diagnostics with support for:
- ML-based primary models (signals as features)
- Rule-based primary models (signals as direct decisions)
- Hybrid primary models (combination of both)
- Comprehensive model registry and versioning
"""

from typing import Dict, Any, List, Optional, Union
from zenml import pipeline
import mlflow
import logging
import numpy as np

from ..steps.data_steps import (
    load_asset_data,
    prepare_training_data,
    calculate_returns,
    load_market_regime_data
)
from ..steps.enhanced_training_steps import (
    create_primary_model,
    conditional_primary_training,
    register_primary_model,
    enhanced_meta_model_training,
    model_performance_evaluation,
    select_best_model_version
)
from ..steps.validation_steps import (
    run_insample_optimization,
    run_insample_permutation_test,
    run_walkforward_test,
    run_walkforward_permutation_test,
    generate_validation_report
)
from ..models.model_registry import get_model_registry


@pipeline
def enhanced_asset_diagnostic_pipeline(
    asset_id: str,
    primary_model_mode: str = 'ml_based',  # 'ml_based', 'rule_based', 'hybrid'
    primary_model_config: Optional[Dict[str, Any]] = None,
    meta_model_type: str = 'random_forest',
    data_dir: str = 'data',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    labeling_params: Optional[Dict[str, Any]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_permutations: int = 1000,
    train_split: float = 0.6,
    val_split: float = 0.2,
    register_models: bool = True,
    use_existing_model: bool = False
) -> Dict[str, Any]:
    """
    Enhanced asset diagnostic pipeline supporting multiple model types.
    
    Args:
        asset_id: Asset identifier
        primary_model_mode: Type of primary model ('ml_based', 'rule_based', 'hybrid')
        primary_model_config: Configuration for primary model
        meta_model_type: Type of meta-model
        data_dir: Base data directory
        start_date: Start date for analysis
        end_date: End date for analysis
        labeling_params: Triple-barrier labeling parameters
        param_grid: Parameter grid for optimization
        n_permutations: Number of permutations for significance testing
        train_split: Training data fraction
        val_split: Validation data fraction
        register_models: Whether to register models in registry
        use_existing_model: Whether to try loading existing models
        
    Returns:
        Comprehensive validation report
    """
    # Start MLflow experiment
    experiment_name = f"enhanced_diagnostics_{asset_id}_{primary_model_mode}"
    mlflow.set_experiment(experiment_name)
    
    # Default configurations
    if primary_model_config is None:
        primary_model_config = _get_default_model_config(primary_model_mode, asset_id)
    
    if param_grid is None:
        param_grid = {
            'confidence_threshold': [0.5, 0.6, 0.7],
            'base_position_size': [0.05, 0.1, 0.15],
            'max_position_size': [0.15, 0.2, 0.25]
        }
    
    with mlflow.start_run(run_name=f"enhanced_validation_{asset_id}_{primary_model_mode}"):
        # Log pipeline parameters
        mlflow.log_params({
            'asset_id': asset_id,
            'primary_model_mode': primary_model_mode,
            'meta_model_type': meta_model_type,
            'register_models': register_models,
            'use_existing_model': use_existing_model
        })
        
        # Step 1: Data Loading and Preparation
        asset_data = load_asset_data(
            asset_id=asset_id,
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date
        )
        
        train_data, val_data, test_data = prepare_training_data(
            asset_data=asset_data,
            train_split=train_split,
            val_split=val_split
        )
        
        forward_returns = calculate_returns(asset_data)
        market_regime_data = load_market_regime_data(data_dir=data_dir)
        
        # Step 2: Primary Model Creation and Training
        if use_existing_model:
            # Try to load existing model
            existing_model_id, _ = select_best_model_version(
                asset_id=asset_id,
                model_type=primary_model_mode
            )
            
            if existing_model_id:
                registry = get_model_registry()
                primary_model = registry.load_model(existing_model_id)
                primary_metrics = {'loaded_existing_model': True, 'model_id': existing_model_id}
            else:
                # Create new model if none exists
                primary_model = create_primary_model(
                    model_mode=primary_model_mode,
                    model_config=primary_model_config,
                    asset_id=asset_id
                )
                
                primary_model, primary_metrics = conditional_primary_training(
                    primary_model=primary_model,
                    train_data=train_data,
                    val_data=val_data,
                    training_params={'labeling_params': labeling_params}
                )
        else:
            # Always create new model
            primary_model = create_primary_model(
                model_mode=primary_model_mode,
                model_config=primary_model_config,
                asset_id=asset_id
            )
            
            primary_model, primary_metrics = conditional_primary_training(
                primary_model=primary_model,
                train_data=train_data,
                val_data=val_data,
                training_params={'labeling_params': labeling_params}
            )
        
        # Step 3: Meta-Model Training
        meta_model, meta_metrics = enhanced_meta_model_training(
            primary_model=primary_model,
            train_data=train_data,
            val_data=val_data,
            forward_returns=forward_returns,
            market_regime_data=market_regime_data,
            meta_model_type=meta_model_type
        )
        
        # Step 4: Model Performance Evaluation
        performance_metrics = model_performance_evaluation(
            primary_model=primary_model,
            meta_model=meta_model,
            test_data=test_data,
            market_regime_data=market_regime_data
        )
        
        # Step 5: Model Registration
        primary_model_id = None
        if register_models and primary_model.is_trained:
            primary_model_id = register_primary_model(
                primary_model=primary_model,
                asset_id=asset_id,
                performance_metrics=performance_metrics,
                tags={
                    'experiment_name': experiment_name,
                    'pipeline_version': '2.0.0',
                    'model_mode': primary_model_mode
                }
            )
        
        # Step 6: Enhanced 5-Step Validation Framework
        
        # Step 6.1: In-Sample Parameter Optimization
        best_params, optimization_results = run_insample_optimization(
            primary_model=primary_model,
            meta_model=meta_model,
            train_data=train_data,
            param_grid=param_grid,
            optimization_metric='sharpe_ratio'
        )
        
        # Step 6.2: In-Sample Permutation Test
        insample_permutation_results = run_insample_permutation_test(
            primary_model=primary_model,
            meta_model=meta_model,
            train_data=train_data,
            best_params=best_params,
            n_permutations=n_permutations
        )
        
        # Step 6.3: Walk-Forward Out-of-Sample Test
        walkforward_results = run_walkforward_test(
            primary_model=primary_model,
            meta_model=meta_model,
            test_data=test_data,
            best_params=best_params,
            market_regime_data=market_regime_data
        )
        
        # Generate out-of-sample returns for final permutation test
        test_predictions = primary_model.predict(test_data, return_confidence=False)
        test_meta_features = meta_model.prepare_meta_features(
            test_data, test_predictions, market_regime_data
        )
        test_confidence = meta_model.predict_confidence(test_meta_features)
        
        test_position_sizes = meta_model.calculate_position_sizes(
            test_predictions,
            test_confidence,
            confidence_threshold=best_params.get('confidence_threshold', 0.6)
        )
        
        if 'close' in test_data.columns:
            price_returns = test_data['close'].pct_change().shift(-1)
            oos_returns = test_position_sizes.shift(1) * price_returns
            oos_returns = oos_returns.fillna(0).dropna()
        else:
            oos_returns = test_position_sizes * 0
        
        # Step 6.4: Walk-Forward Permutation Test
        walkforward_permutation_results = run_walkforward_permutation_test(
            oos_returns=oos_returns,
            n_permutations=n_permutations
        )
        
        # Step 7: Generate Enhanced Validation Report
        validation_report = generate_validation_report(
            optimization_results=optimization_results,
            insample_permutation=insample_permutation_results,
            walkforward_results=walkforward_results,
            walkforward_permutation=walkforward_permutation_results,
            asset_id=asset_id
        )
        
        # Add enhanced information to report
        validation_report.update({
            'primary_model_type': primary_model_mode,
            'primary_model_version': primary_model.model_version,
            'primary_model_id': primary_model_id,
            'meta_model_type': meta_model_type,
            'primary_training_metrics': primary_metrics,
            'meta_training_metrics': meta_metrics,
            'model_registry_enabled': register_models,
            'feature_importance': primary_model.get_feature_importance().to_dict() if hasattr(primary_model.get_feature_importance(), 'to_dict') else {}
        })
        
        # Log final summary
        mlflow.log_metrics({
            'validation_passed': int(validation_report['validation_passed']),
            'final_sharpe': validation_report['oos_sharpe'],
            'primary_model_ready': int(primary_model.is_trained),
            'meta_model_precision': meta_metrics.get('cv_precision_mean', 0),
            'model_registered': int(primary_model_id is not None)
        })
        
        return validation_report


@pipeline
def model_comparison_pipeline(
    asset_id: str,
    model_modes: List[str] = ['ml_based', 'rule_based', 'hybrid'],
    data_dir: str = 'data',
    comparison_metric: str = 'oos_sharpe',
    **pipeline_kwargs
) -> Dict[str, Any]:
    """
    Compare different model types for the same asset.
    
    Args:
        asset_id: Asset identifier
        model_modes: List of model modes to compare
        data_dir: Data directory
        comparison_metric: Metric to use for comparison
        **pipeline_kwargs: Additional pipeline arguments
        
    Returns:
        Model comparison results
    """
    mlflow.set_experiment(f"model_comparison_{asset_id}")
    
    with mlflow.start_run(run_name=f"compare_models_{asset_id}"):
        comparison_results = {}
        
        # Run diagnostics for each model type
        for model_mode in model_modes:
            try:
                report = enhanced_asset_diagnostic_pipeline(
                    asset_id=asset_id,
                    primary_model_mode=model_mode,
                    data_dir=data_dir,
                    register_models=True,
                    **pipeline_kwargs
                )
                comparison_results[model_mode] = report
                
            except Exception as e:
                logging.error(f"Failed to run diagnostics for {model_mode}: {e}")
                comparison_results[model_mode] = {
                    'error': str(e),
                    'validation_passed': False,
                    comparison_metric: -np.inf
                }
        
        # Determine best model
        best_model_mode = None
        best_score = -np.inf
        
        for model_mode, results in comparison_results.items():
            score = results.get(comparison_metric, -np.inf)
            if score > best_score:
                best_score = score
                best_model_mode = model_mode
        
        # Create summary
        comparison_summary = {
            'asset_id': asset_id,
            'comparison_metric': comparison_metric,
            'best_model_mode': best_model_mode,
            'best_score': best_score,
            'model_results': comparison_results,
            'models_tested': len(model_modes),
            'successful_models': sum(1 for r in comparison_results.values() if not r.get('error')),
        }
        
        # Log comparison results
        mlflow.log_params({
            'asset_id': asset_id,
            'models_compared': model_modes,
            'comparison_metric': comparison_metric,
            'best_model': best_model_mode
        })
        
        mlflow.log_metrics({
            'best_score': best_score,
            'models_tested': len(model_modes),
            'success_rate': comparison_summary['successful_models'] / len(model_modes)
        })
        
        # Save comparison results
        mlflow.log_dict(comparison_summary, "model_comparison_results.json")
        
        return comparison_summary


@pipeline
def batch_enhanced_diagnostics(
    asset_list: List[str],
    primary_model_mode: str = 'hybrid',  # Default to hybrid for best of both worlds
    data_dir: str = 'data',
    parallel: bool = True,
    **pipeline_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Run enhanced diagnostics for multiple assets in batch.
    
    Args:
        asset_list: List of assets to analyze
        primary_model_mode: Primary model mode to use
        data_dir: Data directory
        parallel: Whether to run in parallel (placeholder for future implementation)
        **pipeline_kwargs: Additional pipeline arguments
        
    Returns:
        Batch diagnostic results
    """
    mlflow.set_experiment(f"batch_enhanced_diagnostics_{primary_model_mode}")
    
    with mlflow.start_run(run_name=f"batch_{len(asset_list)}_assets_{primary_model_mode}"):
        results = {}
        successful_count = 0
        
        for asset_id in asset_list:
            try:
                report = enhanced_asset_diagnostic_pipeline(
                    asset_id=asset_id,
                    primary_model_mode=primary_model_mode,
                    data_dir=data_dir,
                    register_models=True,
                    **pipeline_kwargs
                )
                results[asset_id] = report
                
                if report.get('validation_passed', False):
                    successful_count += 1
                    
            except Exception as e:
                logging.error(f"Failed to analyze {asset_id}: {e}")
                results[asset_id] = {
                    'error': str(e),
                    'validation_passed': False,
                    'asset_id': asset_id,
                    'primary_model_type': primary_model_mode
                }
        
        # Log batch summary
        mlflow.log_metrics({
            'total_assets': len(asset_list),
            'successful_validations': successful_count,
            'success_rate': successful_count / len(asset_list) if asset_list else 0,
            'failed_count': len(asset_list) - successful_count
        })
        
        mlflow.log_params({
            'primary_model_mode': primary_model_mode,
            'assets_analyzed': asset_list[:5]  # Log first 5 assets to avoid param size limits
        })
        
        # Save batch results
        mlflow.log_dict(results, "batch_diagnostic_results.json")
        
        return results


# Helper functions

def _get_default_model_config(model_mode: str, asset_id: str) -> Dict[str, Any]:
    """Get default configuration for model mode."""
    if model_mode == 'ml_based':
        return {
            'model_type': 'random_forest',
            'signal_names': None,  # Use all available signals
            'target_accuracy': 0.55
        }
    elif model_mode == 'rule_based':
        return {
            'signal_names': ['RSI_Bullish_USD', 'DonchianEnsembleUSD'],
            'combination_strategy': 'ensemble',
            'confidence_mode': 'agreement'
        }
    elif model_mode == 'hybrid':
        return {
            'ml_model_config': {
                'model_type': 'random_forest',
                'target_accuracy': 0.55
            },
            'rule_model_config': {
                'signal_names': ['RSI_Bullish_USD', 'DonchianEnsembleUSD'],
                'combination_strategy': 'ensemble'
            },
            'combination_strategy': 'weighted',
            'ml_weight': 0.6,
            'rule_weight': 0.4
        }
    else:
        return {}