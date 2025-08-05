"""
Main ZenML pipeline for 5-step asset diagnostics with meta-labeling.
"""

from typing import Dict, Any, List, Optional
from zenml import pipeline
import mlflow

from ..steps.data_steps import (
    load_asset_data,
    prepare_training_data,
    calculate_returns,
    load_market_regime_data
)
from ..steps.training_steps import (
    train_primary_model,
    train_meta_model,
    create_combined_predictions
)
from ..steps.validation_steps import (
    run_insample_optimization,
    run_insample_permutation_test,
    run_walkforward_test,
    run_walkforward_permutation_test,
    generate_validation_report
)


@pipeline
def asset_diagnostic_pipeline(
    asset_id: str,
    data_dir: str = 'data',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    primary_model_type: str = 'random_forest',
    meta_model_type: str = 'random_forest',
    signal_names: Optional[List[str]] = None,
    labeling_params: Optional[Dict[str, Any]] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_permutations: int = 1000,
    train_split: float = 0.6,
    val_split: float = 0.2
) -> Dict[str, Any]:
    """
    Complete asset diagnostic pipeline with meta-labeling and 5-step validation.
    
    Args:
        asset_id: Asset identifier
        data_dir: Base data directory
        start_date: Start date for analysis
        end_date: End date for analysis
        primary_model_type: Type of primary model
        meta_model_type: Type of meta model
        signal_names: List of signals to use
        labeling_params: Triple-barrier labeling parameters
        param_grid: Parameter grid for optimization
        n_permutations: Number of permutations for significance testing
        train_split: Training data fraction
        val_split: Validation data fraction
        
    Returns:
        Comprehensive validation report
    """
    # Start MLflow experiment
    mlflow.set_experiment(f"asset_diagnostic_{asset_id}")
    
    # Default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'confidence_threshold': [0.5, 0.6, 0.7],
            'base_position_size': [0.05, 0.1, 0.15],
            'max_position_size': [0.15, 0.2, 0.25]
        }
    
    with mlflow.start_run(run_name=f"full_validation_{asset_id}"):
        # Step 0: Data Loading and Preparation
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
        
        # Step 1: Train Primary Model (Direction Prediction)
        primary_model, primary_metrics = train_primary_model(
            train_data=train_data,
            val_data=val_data,
            model_type=primary_model_type,
            signal_names=signal_names,
            labeling_params=labeling_params
        )
        
        # Step 2: Train Meta Model (Position Sizing/Confidence)
        meta_model, meta_metrics = train_meta_model(
            primary_model=primary_model,
            train_data=train_data,
            val_data=val_data,
            forward_returns=forward_returns,
            market_regime_data=market_regime_data,
            model_type=meta_model_type
        )
        
        # Enhanced 5-Step Validation Framework
        
        # Step 1: In-Sample Parameter Optimization
        best_params, optimization_results = run_insample_optimization(
            primary_model=primary_model,
            meta_model=meta_model,
            train_data=train_data,
            param_grid=param_grid,
            optimization_metric='sharpe_ratio'
        )
        
        # Step 2: In-Sample Permutation Test
        insample_permutation_results = run_insample_permutation_test(
            primary_model=primary_model,
            meta_model=meta_model,
            train_data=train_data,
            best_params=best_params,
            n_permutations=n_permutations
        )
        
        # Step 3: Walk-Forward Out-of-Sample Test
        walkforward_results = run_walkforward_test(
            primary_model=primary_model,
            meta_model=meta_model,
            test_data=test_data,
            best_params=best_params,
            market_regime_data=market_regime_data
        )
        
        # Generate out-of-sample returns for final permutation test
        primary_predictions, confidence_scores, position_sizes = create_combined_predictions(
            primary_model=primary_model,
            meta_model=meta_model,
            test_data=test_data,
            market_regime_data=market_regime_data,
            confidence_threshold=best_params.get('confidence_threshold', 0.6)
        )
        
        # Calculate out-of-sample strategy returns
        if 'close' in test_data.columns:
            price_returns = test_data['close'].pct_change().shift(-1)
            oos_returns = position_sizes.shift(1) * price_returns
            oos_returns = oos_returns.fillna(0).dropna()
        else:
            oos_returns = position_sizes * 0  # Fallback
        
        # Step 4: Walk-Forward Permutation Test
        walkforward_permutation_results = run_walkforward_permutation_test(
            oos_returns=oos_returns,
            n_permutations=n_permutations
        )
        
        # Step 5: Generate Comprehensive Report
        validation_report = generate_validation_report(
            optimization_results=optimization_results,
            insample_permutation=insample_permutation_results,
            walkforward_results=walkforward_results,
            walkforward_permutation=walkforward_permutation_results,
            asset_id=asset_id
        )
        
        # Log final summary
        mlflow.log_metrics({
            'primary_model_accuracy': primary_metrics.get('cv_accuracy_mean', 0),
            'meta_model_precision': meta_metrics.get('cv_precision_mean', 0),
            'validation_passed': int(validation_report['validation_passed']),
            'final_sharpe': validation_report['oos_sharpe']
        })
        
        return validation_report


@pipeline
def batch_asset_diagnostics(
    asset_list: List[str],
    data_dir: str = 'data',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **pipeline_kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Run asset diagnostics for multiple assets in batch.
    
    Args:
        asset_list: List of asset IDs to analyze
        data_dir: Base data directory
        start_date: Start date for analysis
        end_date: End date for analysis
        **pipeline_kwargs: Additional arguments for asset_diagnostic_pipeline
        
    Returns:
        Dictionary mapping asset IDs to validation reports
    """
    results = {}
    
    mlflow.set_experiment("batch_asset_diagnostics")
    
    with mlflow.start_run(run_name=f"batch_analysis_{len(asset_list)}_assets"):
        for asset_id in asset_list:
            try:
                report = asset_diagnostic_pipeline(
                    asset_id=asset_id,
                    data_dir=data_dir,
                    start_date=start_date,
                    end_date=end_date,
                    **pipeline_kwargs
                )
                results[asset_id] = report
                
            except Exception as e:
                logging.error(f"Failed to analyze {asset_id}: {e}")
                results[asset_id] = {
                    'error': str(e),
                    'validation_passed': False,
                    'asset_id': asset_id
                }
        
        # Log batch summary
        passed_count = sum(1 for r in results.values() if r.get('validation_passed', False))
        mlflow.log_metrics({
            'total_assets': len(asset_list),
            'passed_validation': passed_count,
            'pass_rate': passed_count / len(asset_list) if asset_list else 0
        })
        
        # Save batch results as artifact
        mlflow.log_dict(results, "batch_diagnostic_results.json")
    
    return results


@pipeline
def asset_comparison_pipeline(
    primary_asset: str,
    benchmark_asset: str = "bitcoin",
    comparison_type: str = "relative_performance",
    data_dir: str = 'data',
    **analysis_kwargs
) -> Dict[str, Any]:
    """
    Compare asset performance against benchmark using meta-labeling framework.
    
    Args:
        primary_asset: Primary asset to analyze
        benchmark_asset: Benchmark asset for comparison
        comparison_type: Type of comparison ("relative_performance", "correlation", "both")
        data_dir: Base data directory
        **analysis_kwargs: Additional analysis parameters
        
    Returns:
        Comparative analysis report
    """
    mlflow.set_experiment(f"asset_comparison_{primary_asset}_vs_{benchmark_asset}")
    
    with mlflow.start_run(run_name=f"comparison_{primary_asset}_vs_{benchmark_asset}"):
        # Run diagnostics for both assets
        primary_report = asset_diagnostic_pipeline(
            asset_id=primary_asset,
            data_dir=data_dir,
            **analysis_kwargs
        )
        
        benchmark_report = asset_diagnostic_pipeline(
            asset_id=benchmark_asset,
            data_dir=data_dir,
            **analysis_kwargs
        )
        
        # Generate comparison metrics
        comparison_report = {
            'primary_asset': primary_asset,
            'benchmark_asset': benchmark_asset,
            'comparison_type': comparison_type,
            'primary_results': primary_report,
            'benchmark_results': benchmark_report,
            
            # Comparative metrics
            'relative_sharpe': (
                primary_report.get('oos_sharpe', 0) - 
                benchmark_report.get('oos_sharpe', 0)
            ),
            'relative_drawdown': (
                primary_report.get('max_drawdown', 0) - 
                benchmark_report.get('max_drawdown', 0)
            ),
            'both_passed_validation': (
                primary_report.get('validation_passed', False) and
                benchmark_report.get('validation_passed', False)
            )
        }
        
        # Log comparison metrics
        mlflow.log_metrics({
            'primary_sharpe': primary_report.get('oos_sharpe', 0),
            'benchmark_sharpe': benchmark_report.get('oos_sharpe', 0),
            'relative_sharpe': comparison_report['relative_sharpe'],
            'both_passed': int(comparison_report['both_passed_validation'])
        })
        
        # Save comparison report
        mlflow.log_dict(comparison_report, "asset_comparison_report.json")
        
        return comparison_report