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
            n_walkforward_permutations: int = 20):
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
            
            # Step 2: Train/Test Split
            print(f"\n--- Step 2: Train/Test Split ---")
            print(f"🔪 Splitting data at {train_test_split:.1%}...")
            split_idx = int(len(self.features_df) * train_test_split)
            train_data = self.features_df.iloc[:split_idx]
            test_data = self.features_df.iloc[split_idx:]
            
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
            
            # Calculate in-sample performance metrics
            train_strategy_data = strategy_data.iloc[:split_idx].dropna()
            if len(train_strategy_data) > 0:
                train_returns = train_strategy_data['strategy_returns'].values
                train_benchmark = train_strategy_data[return_col].values
                
                # Calculate strategy metrics
                strategy_metrics = calculate_all_metrics(train_returns, benchmark_returns=train_benchmark)
                for metric, value in strategy_metrics.items():
                    mlflow.log_metric(f"insample_{metric}", round(value, 4))
                
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
            print(f"⏳ Running 12-fold walk-forward validation...")
            
            def optimize_func(data, train_start, train_end):
                return self.strategy.optimize(data, train_start, train_end, n_trials=200)
            
            wf_results = walk_forward_validation(
                features_df=strategy_data,
                strategy_func=strategy_func,
                optimize_func=optimize_func,
                train_test_split=train_test_split,
                n_folds=12,
                reoptimize_every=1
            )
            
            if wf_results['overall_metrics']:
                # Calculate comprehensive walk-forward metrics for strategy
                wf_strategy_returns = np.array(wf_results['all_returns'])
                if len(wf_strategy_returns) > 0:
                    # Get benchmark returns from the same period
                    test_strategy_data = strategy_data.iloc[split_idx:].dropna()
                    wf_benchmark_returns = test_strategy_data[return_col].values
                    
                    # Calculate strategy metrics
                    strategy_metrics = calculate_all_metrics(wf_strategy_returns, benchmark_returns=wf_benchmark_returns)
                    for metric, value in strategy_metrics.items():
                        mlflow.log_metric(f"wf_{metric}", round(value, 4))
                    
                    # Calculate benchmark metrics
                    benchmark_metrics = calculate_all_metrics(wf_benchmark_returns)
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
                features_df=strategy_data,
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
    parser.add_argument('--n-insample-permutations', type=int, default=100, help='Number of in-sample permutation tests (default: 100)')
    parser.add_argument('--n-walkforward-permutations', type=int, default=20, help='Number of walk-forward permutation tests (default: 20)')
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
        n_walkforward_permutations=args.n_walkforward_permutations
    )


if __name__ == "__main__":
    main()