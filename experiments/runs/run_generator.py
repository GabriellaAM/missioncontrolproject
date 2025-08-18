#!/usr/bin/env python

import argparse
import sys
import os
import importlib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import setup_mlflow, create_experiment
from utils.feature_loader import FeatureLoader
from utils.feature_engineering import calculate_log_returns
from utils.validation import in_sample_permutation_test, walk_forward_validation, walk_forward_permutation_test
from utils.evaluation_metrics import calculate_all_metrics
from utils.plotting import save_plots_for_mlflow
from utils.transaction_costs import apply_transaction_costs, adjust_strategy_returns_for_costs
from utils.triple_barrier import add_triple_barrier_labels, calculate_metrics_with_labels
import mlflow
import numpy as np
from datetime import datetime


class RunGenerator:
    
    def __init__(self, strategy_name: str, asset_name: str, start_date: str, end_date: str):
        self.strategy_name = strategy_name
        self.asset_name = asset_name
        self.start_date = start_date
        self.end_date = end_date
        
        self.strategy = self._load_strategy()
        self.features_df = self._load_features()
        self._setup_mlflow()
    
    def _load_strategy(self):
        """Dynamically load strategy class by scanning strategy directories."""
        # Strategy name format: "strategy_type:strategy_file"
        # e.g., "rules_based:ema_crossover" or "ml_based:persistent_homology_random_forest"
        
        if ':' not in self.strategy_name:
            raise ValueError(f"Strategy name must be in format 'type:filename' (e.g., 'rules_based:ema_crossover')")
        
        strategy_type, strategy_file = self.strategy_name.split(':', 1)
        
        # Build module path
        module_path = f"strategies.{strategy_type}.{strategy_file}"
        
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            raise ValueError(f"Strategy module '{module_path}' not found")
        
        # Find strategy class in the module (look for classes ending with 'Strategy')
        strategy_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                attr_name.endswith('Strategy') and 
                attr_name != 'BaseStrategy'):
                strategy_class = attr
                break
        
        if not strategy_class:
            raise ValueError(f"No strategy class found in '{module_path}' (looking for *Strategy class)")
        
        return strategy_class(asset=self.asset_name)
    
    def _load_features(self):
        """Load features based on strategy requirements."""
        # Get what features this strategy needs
        feature_spec = self.strategy.get_required_features()
        
        # Initialize feature loader 
        loader = FeatureLoader(start_date=self.start_date, end_date=self.end_date)
        
        # Load features using strategy specifications
        features_df = loader.build_feature_set(**feature_spec)
        
        print(f"Loaded {len(features_df)} rows with {len(features_df.columns)} features for {self.strategy_name}")
        return features_df
    
    def _setup_mlflow(self):
        setup_mlflow()
        
        # Extract clean strategy name (remove type prefix)
        if ':' in self.strategy_name:
            self.strategy_type, self.clean_strategy_name = self.strategy_name.split(':', 1)
        else:
            self.strategy_type = 'unknown'
            self.clean_strategy_name = self.strategy_name
        
        # Use clean strategy name for experiment
        experiment_id = create_experiment(self.clean_strategy_name)
        mlflow.set_experiment(self.clean_strategy_name)
    
    def run(self, 
            train_test_split: float = 0.75,
            n_optimization_trials: int = 1000,
            n_insample_permutations: int = 100,
            n_walkforward_permutations: int = 20,
            wf_n_folds: int = None,
            wf_reoptimize_every: int = None):
        """
        Execute the full training and validation pipeline.
        
        Reproduces the same pipeline as ema_crossover_bitcoin.py but using modular components:
        1. Feature Engineering (log returns)  
        2. Train/Test Split
        3. Strategy Optimization (in-sample)
        4. In-Sample Permutation Test
        5. Walk-Forward Validation 
        6. Walk-Forward Permutation Test
        7. Final Evaluation & MLflow Logging
        """
        
        # Create unique run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_run_name = f"{self.clean_strategy_name}_{self.asset_name}_{timestamp}"
        
        with mlflow.start_run(run_name=unique_run_name):
            
            # Set strategy type and asset as MLflow tags
            mlflow.set_tag("strategy_type", self.strategy_type)
            mlflow.set_tag("asset", self.asset_name)
            
            print(f"\n{'='*60}")
            print(f"🚀 RUNNING TRAINING PIPELINE")
            print(f"{'='*60}")
            print(f"Strategy: {self.strategy_name}")
            print(f"Asset: {self.asset_name}")
            print(f"Period: {self.start_date} to {self.end_date}")
            print(f"Train/Test Split: {train_test_split:.1%}")
            print(f"Optimization Trials: {n_optimization_trials}")
            print(f"In-Sample Permutations: {n_insample_permutations}")
            print(f"Walk-Forward Permutations: {n_walkforward_permutations}")
            
            # Log basic parameters  
            mlflow.log_param("asset", self.asset_name)
            mlflow.log_param("train_test_split", train_test_split)
            mlflow.log_param("start_date", self.start_date)
            mlflow.log_param("end_date", self.end_date)
            
            # Step 1: Feature Engineering
            print(f"\n--- Step 1: Feature Engineering ---")
            print(f"📊 Calculating log returns...")
            self.features_df = calculate_log_returns(self.features_df, periods=[1])
            
            # Set timestamp as index if not already
            if 'timestamp' in self.features_df.columns:
                self.features_df = self.features_df.set_index('timestamp')
            
            print(f"✅ Features after engineering: {len(self.features_df.columns)} columns")
            mlflow.log_metric("n_features", len(self.features_df.columns))
            mlflow.log_metric("n_samples", len(self.features_df))
            
            # Step 2: Train/Test Split with Embargo
            print(f"\n--- Step 2: Train/Test Split with Embargo ---")
            print(f"🔪 Splitting data at {train_test_split:.1%}...")
            
            # Apply embargo: training data ends 7 days before test to prevent label leakage
            embargo_days = 7  # 5 days for triple barrier + 2 days buffer
            split_idx = int(len(self.features_df) * train_test_split)
            
            # Adjust split for embargo
            train_split_idx = split_idx - embargo_days
            test_split_idx = split_idx
            
            train_data = self.features_df.iloc[:train_split_idx]
            test_data = self.features_df.iloc[test_split_idx:]
            
            train_start = train_data.index[0].strftime('%Y-%m-%d')
            train_end = train_data.index[-1].strftime('%Y-%m-%d')
            test_start = test_data.index[0].strftime('%Y-%m-%d')
            test_end = test_data.index[-1].strftime('%Y-%m-%d')
            
            print(f"Training period: {train_start} to {train_end} ({len(train_data)} days)")
            print(f"Testing period: {test_start} to {test_end} ({len(test_data)} days)")
            
            mlflow.log_metric("train_samples", len(train_data))
            mlflow.log_metric("test_samples", len(test_data))
            
            # Step 3: Strategy Optimization (In-Sample)
            print(f"\n--- Step 3: Strategy Optimization ---")
            print("🎯 Optimizing strategy parameters on training data...")
            print(f"⚙️  Running {n_optimization_trials} optimization trials...")
            optimization_result = self.strategy.optimize(
                self.features_df, 
                train_start, 
                train_end,
                n_trials=n_optimization_trials
            )
            
            best_params = optimization_result['best_params']
            best_value = optimization_result['best_value']
            
            print(f"✅ Best parameters: {best_params}")
            print(f"✅ Best optimization score: {best_value:.4f}")
            
            # Log optimization results
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            
            # Step 4: Calculate Strategy Signals & Returns
            print(f"\n--- Step 4: Calculate Strategy Signals ---")
            print(f"📈 Generating strategy signals...")
            strategy_data = self.strategy.calculate_signals(self.features_df, best_params)
            
            # Calculate strategy returns
            return_col = f"{self.asset_name}_log_return_1"
            if 'strategy_returns' not in strategy_data.columns:
                strategy_data['strategy_returns'] = (
                    strategy_data[return_col] * strategy_data['signal'].shift(1)
                )
            
            # Apply transaction costs
            strategy_data = apply_transaction_costs(strategy_data, 'signal', self.asset_name)
            
            # Calculate transaction cost rate and log to MLflow
            tx_cost_rate = 0.001 if self.asset_name.lower() == 'bitcoin' else 0.005
            mlflow.log_param("transaction_cost_rate", tx_cost_rate)
            
            # Keep gross returns and create net returns
            strategy_data['strategy_returns_gross'] = strategy_data['strategy_returns'].copy()
            strategy_data['strategy_returns'] = adjust_strategy_returns_for_costs(
                strategy_data['strategy_returns_gross'], 
                strategy_data['transaction_cost']
            )
            
            # Calculate in-sample performance metrics
            train_strategy_data = strategy_data.iloc[:train_split_idx].dropna()
            if len(train_strategy_data) > 0:
                train_returns = train_strategy_data['strategy_returns'].values
                train_signals = train_strategy_data['signal'].values
                train_benchmark = train_strategy_data[return_col].values
                
                # Calculate strategy metrics (including position-based classification metrics)
                strategy_metrics = calculate_all_metrics(train_returns, signals=train_signals, benchmark_returns=train_benchmark)
                
                # Calculate triple barrier labels ONLY for training data
                print(f"🎯 Generating triple barrier labels for training data...")
                price_col = f"{self.asset_name}_close"
                
                # Get training data up to embargo point
                train_features = self.features_df.iloc[:train_split_idx].copy()
                train_features_labeled = add_triple_barrier_labels(
                    train_features,
                    price_col=price_col,
                    volatility_span=20,
                    time_barrier_days=5,
                    upper_barrier_mult=2.0,
                    lower_barrier_mult=2.0
                )
                
                # Align labels with strategy data
                if 'label' in train_features_labeled.columns:
                    # Get labels for the same indices as train_strategy_data
                    common_idx = train_strategy_data.index.intersection(train_features_labeled.index)
                    labels = train_features_labeled.loc[common_idx, 'label'].values
                    aligned_signals = train_strategy_data.loc[common_idx, 'signal'].values
                    aligned_returns = train_strategy_data.loc[common_idx, 'strategy_returns'].values
                    
                    label_metrics = calculate_metrics_with_labels(
                        predictions=aligned_signals,
                        labels=labels,
                        returns=aligned_returns
                    )
                    
                    # Log label-based metrics
                    for metric, value in label_metrics.items():
                        if isinstance(value, np.ndarray):
                            # Skip confusion matrix arrays
                            continue
                        elif isinstance(value, str) and metric == 'confusion_matrix_plot':
                            # Log confusion matrix plot artifact
                            if value:  # Only if plot was created
                                mlflow.log_artifact(value)
                                # Clean up temp file
                                try:
                                    import os
                                    os.remove(value)
                                except:
                                    pass
                        elif not isinstance(value, str):
                            mlflow.log_metric(f"insample_label_{metric}", round(float(value), 4))
                
                # Also calculate and log total gross return for comparison
                train_returns_gross = train_strategy_data['strategy_returns_gross'].values
                total_gross_return = (1 + train_returns_gross).prod() - 1 if len(train_returns_gross) > 0 else 0.0
                mlflow.log_metric("insample_total_return_gross", round(float(total_gross_return), 4))
                
                for metric, value in strategy_metrics.items():
                    if isinstance(value, (np.ndarray, list, str)):
                        # Skip arrays, lists, and string values (no position-based plots anymore)
                        continue
                    else:
                        mlflow.log_metric(f"insample_{metric}", round(float(value), 4))
                
                # Calculate benchmark metrics
                benchmark_metrics = calculate_all_metrics(train_benchmark)
                for metric, value in benchmark_metrics.items():
                    if metric != 'information_ratio':  # Skip IR for benchmark vs itself
                        mlflow.log_metric(f"insample_benchmark_{metric}", round(value, 4))
            
            # Step 5: In-Sample Permutation Test
            print(f"\n--- Step 5: In-Sample Permutation Test ---")
            print(f"🎲 Testing if optimized strategy beats random chance...")
            print(f"⏳ Running {n_insample_permutations} permutation tests...")
            
            # Prepare strategy function for validation (compatible with validation.py signature)
            def strategy_func(data, **params):
                # Extract strategy params (exclude validation-specific params)
                strategy_params = {k: v for k, v in params.items() 
                                 if k not in ['price_col', 'log_return_col']}
                
                # Calculate signals
                result = self.strategy.calculate_signals(data, strategy_params)
                
                # Calculate strategy returns if not present
                if 'strategy_returns' not in result.columns:
                    result['strategy_returns'] = (
                        result[return_col] * result['signal'].shift(1)
                    )
                
                # Apply transaction costs
                result = apply_transaction_costs(result, 'signal', self.asset_name)
                result['strategy_returns_gross'] = result['strategy_returns'].copy()
                result['strategy_returns'] = adjust_strategy_returns_for_costs(
                    result['strategy_returns_gross'], 
                    result['transaction_cost']
                )
                
                return result
            
            # Prepare strategy params for permutation test
            strategy_params_for_perm = best_params.copy()
            strategy_params_for_perm.update({
                'price_col': f"{self.asset_name}_close",
                'log_return_col': return_col
            })
            
            permutation_results = in_sample_permutation_test(
                features_df=self.features_df,
                strategy_func=strategy_func,
                strategy_params=strategy_params_for_perm,
                price_col=f"{self.asset_name}_close",
                log_return_col=return_col,
                train_start=train_start,
                train_end=train_end,
                n_permutations=n_insample_permutations,
                metrics=['profit_factor', 'sharpe_ratio'],
                random_seed=42
            )
            
            # Log permutation results (only p-values)
            for metric_name, results in permutation_results.items():
                if metric_name not in ['permuted_returns', 'original_returns']:
                    mlflow.log_metric(f"insample_perm_{metric_name}_pvalue", round(results['p_value'], 4))
                    print(f"{metric_name}: p-value = {results['p_value']:.4f}, "
                          f"passes = {'YES' if results['passes_test'] else 'NO'}")
            
            # Step 6: Walk-Forward Validation
            print(f"\n--- Step 6: Walk-Forward Validation ---")
            print(f"🚶 Testing strategy on out-of-sample data with periodic re-optimization...")
            
            # Walk-forward parameters: use provided values or adaptive approach
            test_days = len(test_data)
            
            if wf_n_folds is not None and wf_reoptimize_every is not None:
                # Use user-provided values
                n_folds = wf_n_folds
                reoptimize_every = wf_reoptimize_every
                print(f"📊 Using user-specified walk-forward parameters")
            else:
                # Adaptive approach: aim for ~30-90 day folds based on test period
                if test_days <= 180:  # Less than 6 months of test data
                    n_folds = max(3, test_days // 30)  # Monthly folds, minimum 3
                    reoptimize_every = 1  # Reoptimize every fold for short periods
                elif test_days <= 365:  # 6-12 months of test data
                    n_folds = max(6, test_days // 45)  # ~45 day folds
                    reoptimize_every = 2  # Reoptimize every 2 folds (~3 months)
                elif test_days <= 730:  # 1-2 years of test data
                    n_folds = max(8, test_days // 60)  # ~2 month folds
                    reoptimize_every = 3  # Reoptimize quarterly
                else:  # More than 2 years
                    n_folds = max(12, test_days // 90)  # Quarterly folds
                    reoptimize_every = 4  # Reoptimize yearly
                
                # Override individual parameters if provided
                if wf_n_folds is not None:
                    n_folds = wf_n_folds
                if wf_reoptimize_every is not None:
                    reoptimize_every = wf_reoptimize_every
                
                # Cap at reasonable limits
                n_folds = min(n_folds, 24)  # Maximum 24 folds
                print(f"📊 Using adaptive walk-forward parameters based on {test_days} test days")
            
            print(f"⏳ Running {n_folds}-fold walk-forward validation (reoptimize every {reoptimize_every} folds)...")
            print(f"📊 Test period: {test_days} days, ~{test_days//n_folds} days per fold")
            
            def optimize_func(data, train_start, train_end):
                return self.strategy.optimize(data, train_start, train_end, n_trials=200)
            
            wf_results = walk_forward_validation(
                features_df=strategy_data,
                strategy_func=strategy_func,
                optimize_func=optimize_func,
                train_test_split=train_test_split,
                n_folds=n_folds,
                reoptimize_every=reoptimize_every,
                embargo_days=7  # 5 days for triple barrier + 2 days buffer
            )
            
            if wf_results['overall_metrics']:
                # Calculate comprehensive walk-forward metrics for strategy
                wf_strategy_returns = np.array(wf_results['all_returns'])
                if len(wf_strategy_returns) > 0:
                    # Get benchmark returns from the same periods as strategy returns
                    # We need to extract benchmark returns corresponding to the same dates/periods
                    # that were used in walk-forward validation
                    
                    # Get the fold dates and extract benchmark returns for those exact periods
                    wf_benchmark_returns = []
                    for fold_start, fold_end in wf_results['fold_dates']:
                        fold_data = strategy_data.loc[fold_start:fold_end]
                        fold_benchmark = fold_data[return_col].dropna().values
                        wf_benchmark_returns.extend(fold_benchmark)
                    
                    wf_benchmark_returns = np.array(wf_benchmark_returns)
                    
                    # Extract walk-forward signals and gross returns for classification metrics
                    wf_signals = []
                    wf_gross_returns = []
                    for fold_start, fold_end in wf_results['fold_dates']:
                        fold_data = strategy_data.loc[fold_start:fold_end]
                        fold_signals = fold_data['signal'].dropna().values
                        fold_gross = fold_data['strategy_returns_gross'].dropna().values
                        wf_signals.extend(fold_signals)
                        wf_gross_returns.extend(fold_gross)
                    wf_signals = np.array(wf_signals)
                    wf_gross_returns = np.array(wf_gross_returns)
                    
                    # Calculate and log total gross return for walk-forward
                    wf_total_gross_return = (1 + wf_gross_returns).prod() - 1 if len(wf_gross_returns) > 0 else 0.0
                    mlflow.log_metric("wf_total_return_gross", round(float(wf_total_gross_return), 4))
                    
                    # Only calculate IR if lengths match (they should now)
                    if len(wf_strategy_returns) == len(wf_benchmark_returns):
                        strategy_metrics = calculate_all_metrics(wf_strategy_returns, signals=wf_signals, benchmark_returns=wf_benchmark_returns)
                    else:
                        # Fallback without IR if there's still a mismatch
                        strategy_metrics = calculate_all_metrics(wf_strategy_returns, signals=wf_signals, benchmark_returns=None)
                        print(f"  Warning: Return length mismatch - strategy: {len(wf_strategy_returns)}, benchmark: {len(wf_benchmark_returns)}")
                    
                    # No label metrics for walk-forward (test data) - labels are only for training
                    
                    for metric, value in strategy_metrics.items():
                        if isinstance(value, (np.ndarray, list, str)):
                            # Skip arrays, lists, and string values (no position-based plots anymore)
                            continue
                        else:
                            mlflow.log_metric(f"wf_{metric}", round(float(value), 4))
                    
                    # Calculate benchmark metrics from test period
                    test_strategy_data = strategy_data.iloc[split_idx:]
                    test_benchmark = test_strategy_data[return_col].dropna().values
                    if len(test_benchmark) > 0:
                        benchmark_metrics = calculate_all_metrics(test_benchmark)
                        for metric, value in benchmark_metrics.items():
                            if metric != 'information_ratio':  # Skip IR for benchmark vs itself
                                mlflow.log_metric(f"wf_benchmark_{metric}", round(value, 4))
                    
                    print(f"Walk-forward results:")
                    print(f"  Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.3f}")
                    print(f"  Profit Factor: {strategy_metrics['profit_factor']:.3f}")
                    print(f"  Total Return: {strategy_metrics['total_return']:.1%}")
                    print(f"  Max Drawdown: {strategy_metrics['max_drawdown']:.1%}")
            
            # Step 7: Walk-Forward Permutation Test
            print(f"\n--- Step 7: Walk-Forward Permutation Test ---")
            print(f"🎲 Testing if walk-forward results are statistically significant...")
            print(f"⏳ Running {n_walkforward_permutations} walk-forward permutation tests...")
            
            wf_perm_results = walk_forward_permutation_test(
                features_df=self.features_df,  # Use raw features, not strategy_data with pre-calculated signals
                strategy_func=strategy_func,
                optimize_func=optimize_func,
                wf_results=wf_results,
                n_permutations=n_walkforward_permutations,
                p_value_threshold=0.05
            )
            
            for metric in ['sharpe', 'profit_factor']:
                result = wf_perm_results[metric]
                print(f"{metric}: p-value = {result['p_value']:.4f}, "
                      f"passes = {'YES' if result['passes_test'] else 'NO'}")
                
                # Log walk-forward permutation results (only p-values)
                mlflow.log_metric(f"wf_perm_{metric}_pvalue", round(result['p_value'], 4))
            
            # Note: Test metrics are the same as walk-forward metrics since
            # walk-forward covers the entire test period
            print(f"\n--- Step 8: Test Set Evaluation Complete ---")
            print(f"✅ Test metrics are captured in walk-forward results above.")
            
            # Generate and log plots as artifacts
            print(f"\n--- Generating Visualizations ---")
            print(f"📊 Creating performance charts...")
            try:
                plot_files = save_plots_for_mlflow(
                    strategy_data=strategy_data,
                    optimization_results=optimization_result,
                    permutation_results=permutation_results,
                    wf_results=wf_results,
                    wf_perm_results=wf_perm_results,
                    asset_name=self.asset_name,
                    strategy_name=self.clean_strategy_name
                )
                
                # Log plot files as MLflow artifacts
                for plot_file in plot_files:
                    mlflow.log_artifact(plot_file)
                    print(f"  📊 Logged plot: {plot_file}")
                
                # Clean up temporary files
                import os
                for plot_file in plot_files:
                    try:
                        os.remove(plot_file)
                    except:
                        pass
                        
            except Exception as e:
                print(f"  ⚠️  Warning: Could not generate all plots: {e}")
            
            # Create comprehensive summary and log as artifact
            import json
            summary = {
                'run_info': {
                    'run_id': mlflow.active_run().info.run_id,
                    'experiment_name': self.clean_strategy_name,
                    'strategy_type': self.strategy_type,
                    'asset': self.asset_name,
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'train_test_split': train_test_split
                },
                'optimization': {
                    'best_params': best_params,
                    'optimization_score': best_value
                },
                'validation': {
                    'permutation_results': {k: v for k, v in permutation_results.items() if k not in ['permuted_returns', 'original_returns']},
                    'walk_forward_permutation': wf_perm_results
                }
            }
            
            # Log summary as artifact
            summary_path = 'run_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            mlflow.log_artifact(summary_path)
            
            # Summary
            print(f"\n{'='*60}")
            print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"📋 MLflow Run ID: {mlflow.active_run().info.run_id}")
            
            return {
                'optimization_result': optimization_result,
                'permutation_results': permutation_results,
                'wf_results': wf_results,
                'wf_permutation_results': wf_perm_results,
                'strategy_data': strategy_data,
                'run_id': mlflow.active_run().info.run_id
            }
    
    @staticmethod
    def discover_strategies():
        """Discover all available strategies by scanning directories."""
        strategies_dir = Path(__file__).parent.parent / 'strategies'
        available_strategies = []
        
        for strategy_type_dir in strategies_dir.iterdir():
            if strategy_type_dir.is_dir() and not strategy_type_dir.name.startswith('__'):
                strategy_type = strategy_type_dir.name
                
                for strategy_file in strategy_type_dir.iterdir():
                    if (strategy_file.suffix == '.py' and 
                        not strategy_file.name.startswith('__') and
                        strategy_file.name != 'base_strategy.py'):
                        
                        strategy_name = strategy_file.stem
                        full_name = f"{strategy_type}:{strategy_name}"
                        available_strategies.append(full_name)
        
        return available_strategies


def main():
    parser = argparse.ArgumentParser(description='Run strategy backtest')
    parser.add_argument('--strategy', help='Strategy name (format: type:filename, e.g., rules_based:ema_crossover)')
    parser.add_argument('--asset', help='Asset name (e.g., bitcoin, ethereum)')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--train-test-split', type=float, default=0.75, help='Train/test split')
    parser.add_argument('--n-optimization-trials', type=int, default=1000, help='Number of optimization trials (default: 1000)')
    parser.add_argument('--n-insample-permutations', type=int, default=1000, help='Number of in-sample permutation tests (default: 100)')
    parser.add_argument('--n-walkforward-permutations', type=int, default=50, help='Number of walk-forward permutation tests (default: 20)')
    parser.add_argument('--wf-n-folds', type=int, default=None, help='Number of walk-forward folds (default: adaptive based on data length)')
    parser.add_argument('--wf-reoptimize-every', type=int, default=1, help='Reoptimize every N folds (default: adaptive based on data length)')
    parser.add_argument('--list-strategies', action='store_true', help='List all available strategies')
    
    args = parser.parse_args()
    
    # List available strategies if requested
    if args.list_strategies:
        strategies = RunGenerator.discover_strategies()
        print("Available strategies:")
        for strategy in sorted(strategies):
            print(f"  {strategy}")
        return
    
    # Validate required arguments
    if not all([args.strategy, args.asset, args.start_date, args.end_date]):
        parser.print_help()
        print("\nUse --list-strategies to see available strategies")
        return
    
    print(f"Running strategy: {args.strategy}")
    print(f"Asset: {args.asset}")
    
    runner = RunGenerator(args.strategy, args.asset, args.start_date, args.end_date)
    runner.run(
        train_test_split=args.train_test_split,
        n_optimization_trials=args.n_optimization_trials,
        n_insample_permutations=args.n_insample_permutations,
        n_walkforward_permutations=args.n_walkforward_permutations,
        wf_n_folds=args.wf_n_folds,
        wf_reoptimize_every=args.wf_reoptimize_every
    )


if __name__ == "__main__":
    main()