"""
ZenML validation steps for the 5-step validation framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from zenml import step
import logging
import mlflow
from sklearn.utils import resample
from scipy import stats

from ..meta_labeling.primary_model import PrimaryDirectionModel
from ..meta_labeling.meta_model import MetaModel
from ..portfolio.backtester import Backtester


@step
def run_insample_optimization(
    primary_model: PrimaryDirectionModel,
    meta_model: MetaModel,
    train_data: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    optimization_metric: str = 'sharpe_ratio'
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Step 1: In-sample parameter optimization.
    
    Args:
        primary_model: Trained primary model
        meta_model: Trained meta model
        train_data: Training data
        param_grid: Parameter grid for optimization
        optimization_metric: Metric to optimize
        
    Returns:
        Tuple of (best_params, optimization_results)
    """
    best_score = -np.inf
    best_params = {}
    optimization_results = {}
    
    # Grid search over parameter combinations
    param_combinations = _generate_param_combinations(param_grid)
    
    for i, params in enumerate(param_combinations):
        try:
            # Generate signals with current parameters
            signals, returns = _generate_signals_and_returns(
                primary_model, meta_model, train_data, params
            )
            
            # Calculate performance metrics
            metrics = _calculate_performance_metrics(returns)
            score = metrics.get(optimization_metric, -np.inf)
            
            optimization_results[f'combination_{i}'] = {
                'params': params,
                'score': score,
                'metrics': metrics
            }
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
                
        except Exception as e:
            logging.warning(f"Parameter combination {i} failed: {e}")
            continue
    
    # Log results to MLflow
    with mlflow.start_run(run_name="insample_optimization", nested=True):
        mlflow.log_params(best_params)
        mlflow.log_metric(f'best_{optimization_metric}', best_score)
        mlflow.log_metric('total_combinations', len(param_combinations))
    
    logging.info(f"In-sample optimization completed. Best {optimization_metric}: {best_score:.4f}")
    
    return best_params, optimization_results


@step
def run_insample_permutation_test(
    primary_model: PrimaryDirectionModel,
    meta_model: MetaModel,
    train_data: pd.DataFrame,
    best_params: Dict[str, Any],
    n_permutations: int = 1000,
    test_metric: str = 'sharpe_ratio'
) -> Dict[str, float]:
    """
    Step 2: In-sample permutation test for statistical significance.
    
    Args:
        primary_model: Trained primary model
        meta_model: Trained meta model  
        train_data: Training data
        best_params: Best parameters from optimization
        n_permutations: Number of permutations to run
        test_metric: Metric to test for significance
        
    Returns:
        Permutation test results
    """
    # Generate actual strategy returns with best parameters
    signals, actual_returns = _generate_signals_and_returns(
        primary_model, meta_model, train_data, best_params
    )
    
    actual_metrics = _calculate_performance_metrics(actual_returns)
    actual_score = actual_metrics.get(test_metric, 0)
    
    # Run permutation tests
    permuted_scores = []
    
    for i in range(n_permutations):
        try:
            # Randomly permute the returns while keeping signal structure
            permuted_returns = actual_returns.copy()
            permuted_returns.values[:] = resample(
                actual_returns.values, 
                random_state=i,
                replace=False
            )
            
            # Calculate metrics for permuted returns
            perm_metrics = _calculate_performance_metrics(permuted_returns)
            perm_score = perm_metrics.get(test_metric, 0)
            permuted_scores.append(perm_score)
            
        except Exception as e:
            logging.warning(f"Permutation {i} failed: {e}")
            continue
    
    # Calculate statistical significance
    permuted_scores = np.array(permuted_scores)
    p_value = (permuted_scores >= actual_score).mean()
    
    results = {
        'actual_score': actual_score,
        'permutation_mean': permuted_scores.mean(),
        'permutation_std': permuted_scores.std(),
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'n_permutations': len(permuted_scores)
    }
    
    # Log results to MLflow
    with mlflow.start_run(run_name="insample_permutation", nested=True):
        mlflow.log_metrics(results)
    
    logging.info(f"In-sample permutation test completed. P-value: {p_value:.4f}")
    
    return results


@step
def run_walkforward_test(
    primary_model: PrimaryDirectionModel,
    meta_model: MetaModel,
    test_data: pd.DataFrame,
    best_params: Dict[str, Any],
    market_regime_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Step 3: Walk-forward out-of-sample test.
    
    Args:
        primary_model: Trained primary model
        meta_model: Trained meta model
        test_data: Out-of-sample test data
        best_params: Optimized parameters from in-sample
        market_regime_data: Market regime features
        
    Returns:
        Out-of-sample performance results
    """
    # Generate signals and returns on out-of-sample data
    signals, oos_returns = _generate_signals_and_returns(
        primary_model, meta_model, test_data, best_params, market_regime_data
    )
    
    # Calculate comprehensive performance metrics
    oos_metrics = _calculate_performance_metrics(oos_returns)
    
    # Additional out-of-sample specific metrics
    signal_count = (signals != 0).sum()
    signal_rate = signal_count / len(signals) if len(signals) > 0 else 0
    
    oos_metrics.update({
        'signal_count': signal_count,
        'signal_rate': signal_rate,
        'total_observations': len(test_data)
    })
    
    # Log results to MLflow
    with mlflow.start_run(run_name="walkforward_test", nested=True):
        mlflow.log_params(best_params)
        mlflow.log_metrics(oos_metrics)
    
    logging.info(f"Walk-forward test completed. OOS Sharpe: {oos_metrics.get('sharpe_ratio', 0):.4f}")
    
    return oos_metrics


@step
def run_walkforward_permutation_test(
    oos_returns: pd.Series,
    n_permutations: int = 1000,
    test_metric: str = 'sharpe_ratio'
) -> Dict[str, float]:
    """
    Step 4: Walk-forward permutation test for final validation.
    
    Args:
        oos_returns: Out-of-sample returns
        n_permutations: Number of permutations
        test_metric: Metric to test for significance
        
    Returns:
        Final permutation test results
    """
    # Calculate actual out-of-sample performance
    actual_metrics = _calculate_performance_metrics(oos_returns)
    actual_score = actual_metrics.get(test_metric, 0)
    
    # Run permutation tests on out-of-sample returns
    permuted_scores = []
    
    for i in range(n_permutations):
        try:
            # Randomly permute the out-of-sample returns
            permuted_returns = resample(
                oos_returns.values,
                random_state=i,
                replace=False
            )
            
            perm_series = pd.Series(permuted_returns, index=oos_returns.index)
            perm_metrics = _calculate_performance_metrics(perm_series)
            perm_score = perm_metrics.get(test_metric, 0)
            permuted_scores.append(perm_score)
            
        except Exception as e:
            logging.warning(f"OOS Permutation {i} failed: {e}")
            continue
    
    # Calculate final statistical significance
    permuted_scores = np.array(permuted_scores)
    p_value = (permuted_scores >= actual_score).mean()
    
    # Calculate percentile rank
    percentile_rank = stats.percentileofscore(permuted_scores, actual_score)
    
    results = {
        'actual_oos_score': actual_score,
        'permutation_mean': permuted_scores.mean(),
        'permutation_std': permuted_scores.std(),
        'p_value': p_value,
        'percentile_rank': percentile_rank,
        'is_significant': p_value < 0.05,
        'is_robust': p_value < 0.01,  # Stricter threshold for robustness
        'n_permutations': len(permuted_scores)
    }
    
    # Log results to MLflow
    with mlflow.start_run(run_name="walkforward_permutation", nested=True):
        mlflow.log_metrics(results)
    
    logging.info(f"Walk-forward permutation test completed. Final P-value: {p_value:.4f}")
    
    return results


@step
def generate_validation_report(
    optimization_results: Dict[str, Any],
    insample_permutation: Dict[str, float],
    walkforward_results: Dict[str, Any],
    walkforward_permutation: Dict[str, float],
    asset_id: str
) -> Dict[str, Any]:
    """
    Generate comprehensive validation report.
    
    Args:
        optimization_results: Results from in-sample optimization
        insample_permutation: In-sample permutation test results
        walkforward_results: Walk-forward test results
        walkforward_permutation: Walk-forward permutation results
        asset_id: Asset identifier
        
    Returns:
        Comprehensive validation report
    """
    # Determine overall validation status
    validation_passed = all([
        insample_permutation.get('is_significant', False),
        walkforward_permutation.get('is_significant', False),
        walkforward_results.get('sharpe_ratio', 0) > 0.5
    ])
    
    robustness_level = 'high' if walkforward_permutation.get('is_robust', False) else (
        'medium' if walkforward_permutation.get('is_significant', False) else 'low'
    )
    
    report = {
        'asset_id': asset_id,
        'validation_passed': validation_passed,
        'robustness_level': robustness_level,
        'timestamp': pd.Timestamp.now().isoformat(),
        
        # Summary metrics
        'insample_sharpe': optimization_results.get('best_sharpe_ratio', 0),
        'oos_sharpe': walkforward_results.get('sharpe_ratio', 0),
        'insample_p_value': insample_permutation.get('p_value', 1.0),
        'oos_p_value': walkforward_permutation.get('p_value', 1.0),
        
        # Detailed results
        'step1_optimization': optimization_results,
        'step2_insample_permutation': insample_permutation,
        'step3_walkforward': walkforward_results,
        'step4_oos_permutation': walkforward_permutation,
        
        # Risk metrics
        'max_drawdown': walkforward_results.get('max_drawdown', 0),
        'volatility': walkforward_results.get('volatility', 0),
        'signal_rate': walkforward_results.get('signal_rate', 0)
    }
    
    # Log final report to MLflow
    with mlflow.start_run(run_name="validation_report", nested=True):
        mlflow.log_params({'asset_id': asset_id})
        mlflow.log_metrics({
            'validation_passed': int(validation_passed),
            'insample_sharpe': report['insample_sharpe'],
            'oos_sharpe': report['oos_sharpe'],
            'insample_p_value': report['insample_p_value'],
            'oos_p_value': report['oos_p_value']
        })
        
        # Log report as artifact
        mlflow.log_dict(report, "validation_report.json")
    
    logging.info(f"Validation report generated for {asset_id}. Passed: {validation_passed}")
    
    return report


# Helper functions
def _generate_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from grid."""
    import itertools
    
    keys, values = zip(*param_grid.items())
    combinations = []
    
    for combination in itertools.product(*values):
        combinations.append(dict(zip(keys, combination)))
    
    return combinations


def _generate_signals_and_returns(
    primary_model: PrimaryDirectionModel,
    meta_model: MetaModel,
    data: pd.DataFrame,
    params: Dict[str, Any],
    market_regime_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.Series, pd.Series]:
    """Generate signals and returns for given parameters."""
    from ..signals.signal_registry import get_all_signals
    
    # Get signal registry
    signal_registry = get_all_signals()
    
    # Generate primary predictions
    features = primary_model.prepare_features(data, signal_registry)
    primary_predictions = primary_model.predict(features, return_probabilities=False)
    
    # Generate meta-features and confidence
    if market_regime_data is not None:
        meta_features = meta_model.prepare_meta_features(
            data, primary_predictions, market_regime_data
        )
    else:
        meta_features = meta_model.prepare_meta_features(data, primary_predictions)
    
    confidence_scores = meta_model.predict_confidence(meta_features)
    
    # Calculate position sizes with parameters
    position_sizes = meta_model.calculate_position_sizes(
        primary_predictions,
        confidence_scores,
        confidence_threshold=params.get('confidence_threshold', 0.6),
        base_position_size=params.get('base_position_size', 0.1),
        max_position_size=params.get('max_position_size', 0.2)
    )
    
    # Calculate returns (simplified - using next period return)
    if 'close' in data.columns:
        price_returns = data['close'].pct_change().shift(-1)  # Next period return
        strategy_returns = position_sizes.shift(1) * price_returns
        strategy_returns = strategy_returns.fillna(0)
    else:
        strategy_returns = pd.Series(0, index=position_sizes.index)
    
    return position_sizes, strategy_returns


def _calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive performance metrics."""
    if returns.empty or returns.isna().all():
        return {'sharpe_ratio': 0, 'total_return': 0, 'volatility': 0, 'max_drawdown': 0}
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    volatility = returns.std() * np.sqrt(252)  # Annualized
    mean_return = returns.mean() * 252  # Annualized
    
    # Sharpe ratio
    sharpe_ratio = mean_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).mean() if len(returns) > 0 else 0
    
    return {
        'total_return': float(total_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'mean_return': float(mean_return)
    }