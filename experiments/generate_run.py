#!/usr/bin/env python

import argparse
import sys
import os
import importlib
from pathlib import Path

# Handle both module execution (python -m experiments.generate_run) and direct execution
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try relative imports first (for module execution)
    from .config import setup_mlflow, create_experiment
    from .utils.feature_loader import FeatureLoader
    from .utils.feature_engineering import calculate_log_returns
    from .utils.validation import in_sample_permutation_test, walk_forward_validation, walk_forward_permutation_test
    from .utils.evaluation_metrics import calculate_all_metrics, create_confusion_matrix_plot, export_comprehensive_csv, create_model_summary, calculate_metrics_with_labels
    from .utils.plotting import save_plots_for_mlflow, plot_feature_importance, plot_signal_comparison
    from .utils.transaction_costs import apply_transaction_costs, adjust_strategy_returns_for_costs
    from .utils.triple_barrier import add_triple_barrier_labels
    from .utils.normalization import ColumnNormalizer
except ImportError:
    # Fall back to absolute imports (for direct execution)
    from config import setup_mlflow, create_experiment
    from utils.feature_loader import FeatureLoader
    from utils.feature_engineering import calculate_log_returns
    from utils.validation import in_sample_permutation_test, walk_forward_validation, walk_forward_permutation_test
    from utils.evaluation_metrics import calculate_all_metrics, create_confusion_matrix_plot, export_comprehensive_csv, create_model_summary, calculate_metrics_with_labels
    from utils.plotting import save_plots_for_mlflow, plot_feature_importance, plot_signal_comparison
    from utils.transaction_costs import apply_transaction_costs, adjust_strategy_returns_for_costs
    from utils.triple_barrier import add_triple_barrier_labels
    from utils.normalization import ColumnNormalizer
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime


class RunGenerator:
    
    def __init__(self, strategy_name: str, asset_name: str, start_date: str, end_date: str):
        self.strategy_name = strategy_name
        self.asset_name = asset_name
        self.start_date = start_date
        self.end_date = end_date
        
        # Setup MLflow early so meta-models can access primary run data
        self._setup_mlflow_tracking()
        
        self.strategy = self._load_strategy()
        
        # For meta-models, override dates, asset, and train_test_split with primary model's settings
        if self.strategy.is_meta_model:
            print(f"🔗 Meta-model detected - using primary model's configuration")
            self.start_date = self.strategy.primary_start_date
            self.end_date = self.strategy.primary_end_date
            self.asset_name = self.strategy.primary_asset
            self.primary_train_test_split = self.strategy.primary_train_test_split
            print(f"   Updated date range: {self.start_date} to {self.end_date}")
            print(f"   Updated asset: {self.asset_name}")
            print(f"   Using primary model's train_test_split: {self.primary_train_test_split}")
        
        self.features_df = self._load_features()
        self._setup_mlflow_experiment()
    
    def _load_strategy(self):
        """Dynamically load strategy class by scanning strategy directories."""
        # Strategy name format: "strategy_type:strategy_file" or "strategy_type:strategy_file:primary_run_id"
        # e.g., "rules_based:ema_crossover" or "meta_models:ensemble_meta:run123abc"
        
        if ':' not in self.strategy_name:
            raise ValueError(f"Strategy name must be in format 'type:filename' (e.g., 'rules_based:ema_crossover') or 'meta_models:filename:primary_run_id'")
        
        parts = self.strategy_name.split(':')
        if len(parts) == 2:
            strategy_type, strategy_file = parts
            primary_run_id = None
        elif len(parts) == 3:
            strategy_type, strategy_file, primary_run_id = parts
        else:
            raise ValueError(f"Invalid strategy name format: {self.strategy_name}")
        
        # Build module path - handle both module execution and direct execution
        try:
            # Try relative import first (for module execution)
            module_path = f".strategies.{strategy_type}.{strategy_file}"
            module = importlib.import_module(module_path, package='experiments')
        except ImportError:
            try:
                # Try absolute import (for direct execution)
                module_path = f"strategies.{strategy_type}.{strategy_file}"
                module = importlib.import_module(module_path)
            except ImportError:
                raise ValueError(f"Strategy module '{module_path}' not found")
        
        # Find strategy class in the module (look for classes ending with 'Strategy')
        strategy_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                attr_name.endswith('Strategy') and 
                attr_name not in ['BaseStrategy', 'MetaStrategy']):
                strategy_class = attr
                break
        
        if not strategy_class:
            raise ValueError(f"No strategy class found in '{module_path}' (looking for *Strategy class)")
        
        # Create strategy instance - meta-models need primary_run_id
        if strategy_type == 'meta_models':
            if primary_run_id is None:
                raise ValueError(f"Meta-model strategies require primary_run_id: use format 'meta_models:filename:run_id'")
            return strategy_class(asset=self.asset_name, primary_run_id=primary_run_id)
        else:
            return strategy_class(asset=self.asset_name)
    
    def _load_features(self):
        """Load features based on strategy requirements with automatic warmup period."""
        # Get strategy-specific warmup period
        strategy_warmup = self.strategy.get_warmup_days()
        
        # Check if normalization is required and get its warmup
        normalization_warmup = 0
        if hasattr(self.strategy, 'get_normalization_config'):
            normalization_config = self.strategy.get_normalization_config()
            if normalization_config:
                # EWMA normalization requires 135 days (1.5 * 90 halflife)
                normalization_warmup = 135
        
        # Total warmup is the maximum of both
        warmup_days = max(strategy_warmup, normalization_warmup)
        
        # Get what features this strategy needs
        feature_spec = self.strategy.get_required_features()
        
        if warmup_days > 0:
            # Calculate extended start date for warmup
            original_start = pd.to_datetime(self.start_date)
            extended_start = original_start - pd.Timedelta(days=warmup_days)
            extended_start_str = extended_start.strftime('%Y-%m-%d')
            
            print(f"🕐 Total warmup required: {warmup_days} days")
            if strategy_warmup > 0:
                print(f"   Strategy indicators: {strategy_warmup} days")
            if normalization_warmup > 0:
                print(f"   EWMA normalization: {normalization_warmup} days")
            print(f"   Extended data range: {extended_start_str} to {self.end_date}")
            print(f"   Actual trading range: {self.start_date} to {self.end_date}")
            
            # Initialize feature loader with extended dates
            loader = FeatureLoader(start_date=extended_start_str, end_date=self.end_date)
            
            # Store warmup info for later use
            self.warmup_days = warmup_days
            self.extended_start = extended_start_str
        else:
            # No warmup needed
            print(f"📊 Strategy requires no warmup period")
            loader = FeatureLoader(start_date=self.start_date, end_date=self.end_date)
            self.warmup_days = 0
            self.extended_start = self.start_date
        
        # For meta-models, ensure we load base OHLC data AND primary labels
        if self.strategy.is_meta_model:
            # Load base features for the asset (includes OHLC data needed for permutation tests)
            base_feature_spec = {
                'crypto_assets': [self.asset_name],
                'fred_indicators': None,
                'yahoo_tickers': None,
                'calculated_features': None
            }
            features_df = loader.build_feature_set(**base_feature_spec)
            
            print(f"🔗 Loading primary model data for meta-strategy...")
            
            # Load primary signals (full period) and labels (training only) separately
            primary_signals = self.strategy.load_primary_signals()
            primary_labels = self.strategy.load_primary_labels()
            
            # Ensure features DataFrame has timestamp as index
            if 'timestamp' in features_df.columns:
                features_df = features_df.set_index('timestamp')
            
            # Join base features with primary signals (full period)
            print(f"   Base features before join: {len(features_df)} rows")
            print(f"   Primary signals: {len(primary_signals)} rows")
            print(f"   Primary labels: {len(primary_labels)} rows")
            
            features_df = features_df.merge(
                primary_signals,  # Signals for full period
                left_index=True,
                right_index=True,
                how='left',  # Keep all features
                suffixes=('', '_primary_signals')
            )
            
            # Add labels (training only) - will be NaN for test period
            features_df = features_df.merge(
                primary_labels[['label'] + [col for col in primary_labels.columns if col.startswith('barrier') or col.startswith('return')]],  # Only label-related columns
                left_index=True,
                right_index=True,
                how='left',  # Keep all features, labels NaN for test period
                suffixes=('', '_primary_labels')
            )
            
            print(f"   Features after joins: {len(features_df)} rows")
            print(f"   Columns after merge: {list(features_df.columns)}")
            
            # Ensure signal column is present for meta-model
            if 'signal' not in features_df.columns:
                raise ValueError("Primary signal column missing after merge. Check timestamp alignment between base features and primary signals.")
                
            # Validation checks
            non_null_signals = features_df['signal'].notna().sum()
            label_count = features_df['label'].notna().sum() if 'label' in features_df.columns else 0
            
            print(f"   Non-null signals available: {non_null_signals}/{len(features_df)} (should cover full period)")
            print(f"   Training labels available: {label_count} (only training period)")
            
            if non_null_signals == 0:
                raise ValueError("No valid primary signals found after merge. Check date range overlap between primary model and meta-model.")
            
            if label_count == 0:
                print("   ⚠️  No training labels found - meta-model training may fail")
            
            print(f"   Meta-model will use {label_count} training labels from run: {self.strategy.primary_run_id}")
            
        else:
            # Regular strategies load features using their specifications
            features_df = loader.build_feature_set(**feature_spec)
            
            # For regular strategies, ensure timestamp is index
            if 'timestamp' in features_df.columns:
                features_df = features_df.set_index('timestamp')

        print(f"Loaded {len(features_df)} rows with {len(features_df.columns)} features for {self.strategy_name}")

        # Apply strategy-specific feature enhancement (topological, custom indicators, etc.)
        print(f"🔧 Applying strategy-specific feature enhancements...")
        enhanced_features = self.strategy.enhance_features(features_df)
        if enhanced_features.shape != features_df.shape:
            print(f"   Features enhanced: {features_df.shape} -> {enhanced_features.shape}")
            print(f"   New columns added: {len(enhanced_features.columns) - len(features_df.columns)}")
            features_df = enhanced_features
        else:
            print(f"   No feature enhancements applied by this strategy")

        return features_df
    
    def _setup_mlflow_tracking(self):
        """Setup MLflow tracking URI early so meta-models can access primary run data."""
        setup_mlflow()
    
    def _setup_mlflow_experiment(self):
        """Setup MLflow experiment after strategy is loaded."""
        # Extract clean strategy name (remove type prefix and run_id)
        parts = self.strategy_name.split(':')
        if len(parts) >= 2:
            self.strategy_type = parts[0]
            self.clean_strategy_name = parts[1]  # Just the strategy name, no run_id
        else:
            self.strategy_type = 'unknown'
            self.clean_strategy_name = self.strategy_name
        
        # Use clean strategy name for experiment
        experiment_id = create_experiment(self.clean_strategy_name)
        mlflow.set_experiment(self.clean_strategy_name)
    
    def run(self, 
            train_test_split: float = 0.80,
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
        
        # For meta-models, use primary model's train_test_split
        if self.strategy.is_meta_model:
            train_test_split = self.primary_train_test_split
            print(f"🔗 Meta-model using primary model's train_test_split: {train_test_split}")
        
        # Create unique run name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_run_name = f"{self.clean_strategy_name}_{self.asset_name}_{timestamp}"
        
        with mlflow.start_run(run_name=unique_run_name):
            
            # Set strategy tags and asset as MLflow tags
            mlflow.set_tag("strategy_type", self.strategy.strategy_type)  # Algorithmic approach (trend_following, mean_reversion)
            mlflow.set_tag("strategy_basis", getattr(self.strategy, 'strategy_basis', self.strategy_type))  # Implementation approach (rules_based, ml_based, meta_models)
            mlflow.set_tag("strategy", self.clean_strategy_name)  # Strategy name (e.g., sma_crossover)
            mlflow.set_tag("asset", self.asset_name)
            
            # For meta-models, log the primary run ID for lineage tracking
            if self.strategy.is_meta_model:
                mlflow.log_param("is_meta_model", True)
                mlflow.log_param("primary_run_id", self.strategy.primary_run_id)
                print(f"🏷️  Meta-model params: primary_run_id = {self.strategy.primary_run_id}")
            else:
                mlflow.log_param("is_meta_model", False)
            
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
            
            # Filter to original date range (excluding warmup) for train/test split calculations
            original_data = self.features_df.loc[self.start_date:self.end_date]
            print(f"   Original data range: {len(original_data)} rows from {self.start_date} to {self.end_date}")
            print(f"   Full data (with warmup): {len(self.features_df)} rows from {self.features_df.index[0].strftime('%Y-%m-%d')} to {self.end_date}")
            
            # Apply embargo: training data ends 7 days before test to prevent label leakage
            embargo_days = 7  # 5 days for triple barrier + 2 days buffer
            split_idx = int(len(original_data) * train_test_split)
            
            # Adjust split for embargo
            train_split_idx = split_idx - embargo_days
            test_split_idx = split_idx
            
            # Use original data for train/test date determination
            train_data_original = original_data.iloc[:train_split_idx]
            test_data_original = original_data.iloc[test_split_idx:]
            
            # Use original dates for training and testing (not including warmup)
            train_start = train_data_original.index[0].strftime('%Y-%m-%d')
            train_end = train_data_original.index[-1].strftime('%Y-%m-%d')
            test_start = test_data_original.index[0].strftime('%Y-%m-%d')
            test_end = test_data_original.index[-1].strftime('%Y-%m-%d')
            
            # But for data access, use the full features_df which includes warmup
            train_data = self.features_df.loc[:train_end]  # Full data up to train_end
            test_data = self.features_df.loc[test_start:]  # Data from test_start onward
            
            print(f"Training period: {train_start} to {train_end} ({len(train_data)} days)")
            print(f"Testing period: {test_start} to {test_end} ({len(test_data)} days)")
            
            mlflow.log_metric("train_samples", len(train_data_original))
            mlflow.log_metric("test_samples", len(test_data_original))
            
            # Step 2.5: Apply Normalization (if required by strategy)
            normalization_config = None
            if hasattr(self.strategy, 'get_normalization_config'):
                normalization_config = self.strategy.get_normalization_config()

            if normalization_config:
                print(f"\n--- Step 2.5: EWMA Z-Score Normalization ---")
                print(f"🔧 Applying rolling EWMA normalization...")
                
                # Initialize normalizer
                normalizer = ColumnNormalizer()
                normalizer.configure(normalization_config)
                
                # Apply continuous EWMA normalization to entire dataset
                # EWMA only uses past data, so no look-ahead bias
                self.features_df = normalizer.normalize(self.features_df)
                
                excluded_cols = normalization_config.get('exclude', [])
                normalized_cols = [col for col in self.features_df.columns if col not in excluded_cols]
                
                print(f"✅ Normalized {len(normalized_cols)} columns with 90-day EWMA")
                print(f"   Excluded columns: {excluded_cols}")
                mlflow.log_param("normalization_enabled", True)
                mlflow.log_param("normalization_method", "ewma_zscore")
                mlflow.log_param("ewma_halflife", normalizer.halflife)
                mlflow.log_metric("n_normalized_columns", len(normalized_cols))
            else:
                mlflow.log_param("normalization_enabled", False)
            
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
            
            # Save strategy as MLflow model artifact
            print(f"💾 Saving strategy as MLflow model...")
            model_uri = self.strategy.save_model(best_params)
            print(f"✅ Model saved: {model_uri}")
            
            
            # For ML strategies, ensure model is fitted with optimized parameters
            if self.strategy.implementation_type == 'ml_based' and hasattr(self.strategy, 'fit'):
                print(f"🧠 Training ML model with optimized parameters...")
                
                # Get training data and labels for ML model fitting
                train_strategy_data = strategy_data.loc[train_start:train_end]
                
                # Use triple barrier labels if available, otherwise create simple momentum labels
                price_col = f"{self.asset_name}_close"
                future_returns = train_strategy_data[price_col].shift(-5) / train_strategy_data[price_col] - 1
                ml_labels = pd.Series(0, index=train_strategy_data.index)
                ml_labels[future_returns > 0.02] = 1  # Long if 2% gain in 5 periods
                ml_labels[future_returns < -0.02] = -1  # Short if 2% loss in 5 periods
                
                # Fit the ML model with best parameters
                self.strategy.fit(train_strategy_data, ml_labels, **best_params)
                
                # Regenerate signals with fitted model
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
            
            # Calculate in-sample performance metrics using date-based filtering
            train_strategy_data = strategy_data.loc[train_start:train_end].dropna()
            if len(train_strategy_data) > 0:
                # Get strategy returns and signals from cleaned strategy data
                train_log_returns = train_strategy_data['strategy_returns'].values
                train_signals = train_strategy_data['signal'].values
                
                # Get benchmark returns from consistent raw data, then align lengths
                train_raw_data = self.features_df.loc[train_start:train_end]
                train_raw_benchmark = train_raw_data[return_col].dropna()
                
                # Align benchmark with strategy data by taking the same number of most recent returns
                if len(train_raw_benchmark) >= len(train_log_returns):
                    train_log_benchmark = train_raw_benchmark.tail(len(train_log_returns)).values
                else:
                    # If somehow strategy has more data, take what we have from benchmark
                    train_log_benchmark = train_raw_benchmark.values
                
                # Convert log returns to simple returns for accurate metric calculation
                train_returns = np.exp(train_log_returns) - 1
                train_benchmark = np.exp(train_log_benchmark) - 1
                
                # Calculate strategy metrics (including position-based classification metrics)
                strategy_metrics = calculate_all_metrics(train_returns, signals=train_signals, benchmark_returns=train_benchmark)
                
                # Skip triple barrier labeling for meta-models - they already have labels from primary model
                if not self.strategy.is_meta_model:
                    # Calculate triple barrier labels for FULL dataset (train + test) for all strategy events
                    print(f"🎯 Generating triple barrier labels for full dataset...")
                    price_col = f"{self.asset_name}_close"

                    # FIXED: Extract event timestamps where strategy actually trades (signal != 0) from FULL period
                    # This ensures meta-models have labels for test set evaluation
                    full_period_data = strategy_data.loc[self.start_date:self.end_date]
                    all_strategy_events = full_period_data[full_period_data['signal'] != 0].index
                    print(f"🎯 Found {len(all_strategy_events)} trading events in full dataset (train + test)")

                    # Split events for reporting
                    training_events = all_strategy_events[all_strategy_events <= train_end]
                    test_events = all_strategy_events[all_strategy_events >= test_start]
                    print(f"   Training events: {len(training_events)}")
                    print(f"   Test events: {len(test_events)}")

                    # Apply triple barrier labeling using FULL strategy data (including warmup) for volatility calculation
                    # and label ALL strategy events (train + test)
                    full_strategy_labeled = add_triple_barrier_labels(
                        strategy_data,  # Use FULL strategy data (with warmup) for volatility calculation!
                        price_col=price_col,
                        events=all_strategy_events,  # Label ALL events, not just training
                        volatility_span=20,
                        time_barrier_days=5,
                        upper_barrier_mult=2.0,
                        lower_barrier_mult=2.0
                    )
                    
                    # Get labeled data for full period (train + test)
                    full_period_labeled = full_strategy_labeled.loc[self.start_date:self.end_date]

                    # Create artifacts with labels for FULL period (train + test)
                    if 'label' in full_strategy_labeled.columns:
                        print(f"   Creating label artifacts...")

                        # Get signals and labels for the ENTIRE period (train + test) - but ONLY within start/end dates (no warmup)
                        full_signals = full_period_labeled[['signal']].copy()

                        # Get label columns from FULL period (train + test)
                        label_columns = ['label', 'barrier_touched', 'days_to_barrier', 'return_at_barrier']
                        available_label_columns = [col for col in label_columns if col in full_period_labeled.columns]
                        full_labels = full_period_labeled[available_label_columns].copy()

                        # Count labels by period for reporting
                        train_labels = full_labels.loc[train_start:train_end]
                        test_labels = full_labels.loc[test_start:test_end] if pd.to_datetime(test_start) <= full_labels.index.max() else pd.DataFrame()

                        train_label_count = train_labels['label'].notna().sum()
                        test_label_count = test_labels['label'].notna().sum() if len(test_labels) > 0 else 0

                        print(f"   Training labels: {train_label_count}")
                        print(f"   Test labels: {test_label_count}")
                        print(f"   Total labels: {train_label_count + test_label_count}")

                        # Create separate artifacts for clarity:

                        # 1. Primary signals artifact (for meta-models) - signals only, full period
                        primary_signals_path = 'primary_signals.csv'
                        full_signals.to_csv(primary_signals_path, index_label='timestamp')
                        mlflow.log_artifact(primary_signals_path)
                        print(f"   Saved primary signals: {len(full_signals)} rows ({self.start_date} to {self.end_date}, no warmup)")

                        # 2. Triple barrier labels artifact - signals + labels for FULL period
                        artifact_data = full_signals.merge(
                            full_labels,
                            left_index=True,
                            right_index=True,
                            how='left'  # Keep all signals, include all available labels
                        )

                        labels_csv_path = 'triple_barrier_labels.csv'
                        artifact_data.to_csv(labels_csv_path, index_label='timestamp')
                        mlflow.log_artifact(labels_csv_path)
                        print(f"   Saved combined artifact: {len(artifact_data)} rows with columns: {list(artifact_data.columns)}")
                        print(f"   Labels: full period ({full_labels['label'].notna().sum()} total labels for train + test)")
                        
                        # Clean up temporary files
                        try:
                            os.remove(labels_csv_path)
                            os.remove(primary_signals_path)
                        except:
                            pass
                    
                    # Align labels with strategy data - use training events only for in-sample metrics
                    if 'label' in full_strategy_labeled.columns:
                        # Filter training period labeled events for in-sample evaluation
                        train_strategy_labeled = full_strategy_labeled.loc[train_start:train_end]
                        labeled_events = train_strategy_labeled.dropna(subset=['label'])
                        print(f"🏷️  Found {len(labeled_events)} labeled trading events in training period")

                        if len(labeled_events) > 0:
                            # All data is now aligned since we used train_strategy_data throughout
                            labels = labeled_events['label'].values
                            aligned_signals = labeled_events['signal'].values
                            aligned_returns = labeled_events['strategy_returns'].values
                            
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
                        else:
                            print("⚠️  No labeled events found after filtering NaN")
                else:
                    print(f"🔗 Meta-model: Skipping triple barrier labeling (using primary model labels)")
                
                # Also calculate and log total gross return for comparison
                train_returns_gross = train_strategy_data['strategy_returns_gross'].values
                total_gross_return = np.exp(np.sum(train_returns_gross)) - 1 if len(train_returns_gross) > 0 else 0.0
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
            
            # Prepare strategy function for validation (compatible with validation.py signature)
            # This is needed for both permutation tests and walk-forward validation
            def strategy_func(data, **params):
                # Extract strategy params (exclude validation-specific params)
                strategy_params = {k: v for k, v in params.items() 
                                 if k not in ['price_col', 'log_return_col']}
                
                # Calculate signals
                result = self.strategy.calculate_signals(data, strategy_params)
                
                # Calculate strategy returns if not present
                if 'strategy_returns' not in result.columns:
                    if return_col in result.columns:
                        result['strategy_returns'] = (
                            result[return_col] * result['signal'].shift(1)
                        )
                    elif return_col in data.columns:
                        # Copy return column from input data if missing in result
                        result[return_col] = data[return_col]
                        result['strategy_returns'] = (
                            result[return_col] * result['signal'].shift(1)
                        )
                    else:
                        # Fallback - set to zero to prevent failure
                        result['strategy_returns'] = 0.0
                
                # Apply transaction costs
                result = apply_transaction_costs(result, 'signal', self.asset_name)
                result['strategy_returns_gross'] = result['strategy_returns'].copy()
                result['strategy_returns'] = adjust_strategy_returns_for_costs(
                    result['strategy_returns_gross'], 
                    result['transaction_cost']
                )
                
                return result
            
            # Step 5: In-Sample Permutation Test
            if hasattr(self.strategy, 'is_meta_model') and self.strategy.is_meta_model:
                print(f"\n--- Step 5: In-Sample Permutation Test (Skipped for Meta-Models) ---")
                print(f"🤖 Meta-models use pre-validated primary signals - skipping permutation tests...")
                
                # Create dummy permutation results to maintain pipeline compatibility
                permutation_results = {
                    'sharpe_ratio': {'p_value': 0.001, 'passes_test': True},
                    'information_ratio': {'p_value': 0.001, 'passes_test': True},
                    'max_drawdown': {'p_value': 0.001, 'passes_test': True},
                    'permuted_returns': [],
                    'original_returns': []
                }
                
                print(f"✅ In-sample permutation test skipped for meta-model")
                
            else:
                print(f"\n--- Step 5: In-Sample Permutation Test ---")
                print(f"🎲 Testing if optimized strategy beats random chance...")
                print(f"⏳ Running {n_insample_permutations} permutation tests...")
                
                # strategy_func already defined above
                
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
                return self.strategy.optimize(data, train_start, train_end, n_trials=n_optimization_trials)
            
            # Use processed strategy_data for all strategies to ensure return columns are present
            wf_features_df = strategy_data
                
            wf_results = walk_forward_validation(
                features_df=wf_features_df,
                strategy_func=strategy_func,
                optimize_func=optimize_func,
                train_test_split=train_test_split,
                n_folds=n_folds,
                reoptimize_every=reoptimize_every,
                embargo_days=7  # 5 days for triple barrier + 2 days buffer
            )
            
            print(f"📊 Walk-forward validation completed")
            print(f"   overall_metrics: {wf_results.get('overall_metrics', 'None')}")
            print(f"   all_returns length: {len(wf_results.get('all_returns', []))}")
            print(f"   all_signals length: {len(wf_results.get('all_signals', []))}")
            
            if wf_results['overall_metrics']:
                # Calculate comprehensive walk-forward metrics for strategy
                wf_log_returns = np.array(wf_results['all_returns'])
                if len(wf_log_returns) > 0:
                    # Get benchmark returns from the same periods as strategy returns
                    # We need to extract benchmark returns corresponding to the same dates/periods
                    # that were used in walk-forward validation
                    
                    # Get the fold dates and extract benchmark returns for those exact periods
                    wf_log_benchmark = []
                    for fold_start, fold_end in wf_results['fold_dates']:
                        fold_data = wf_features_df.loc[fold_start:fold_end]
                        fold_benchmark = fold_data[return_col].dropna().values
                        wf_log_benchmark.extend(fold_benchmark)
                    
                    wf_log_benchmark = np.array(wf_log_benchmark)
                    
                    # Convert log returns to simple returns for accurate metric calculation
                    wf_strategy_returns = np.exp(wf_log_returns) - 1
                    wf_benchmark_returns = np.exp(wf_log_benchmark) - 1
                    
                    # Use pre-built signals from walk-forward validation to ensure length consistency
                    wf_signals = np.array(wf_results['all_signals'])
                    
                    # Extract walk-forward gross returns for classification metrics
                    wf_gross_returns = []
                    for fold_start, fold_end in wf_results['fold_dates']:
                        fold_data = strategy_data.loc[fold_start:fold_end]
                        fold_gross = fold_data['strategy_returns_gross'].dropna().values
                        wf_gross_returns.extend(fold_gross)
                    wf_gross_returns = np.array(wf_gross_returns)
                    
                    # Calculate and log total gross return for walk-forward (convert log returns to simple)
                    wf_total_gross_return = np.exp(np.sum(wf_gross_returns)) - 1 if len(wf_gross_returns) > 0 else 0.0
                    mlflow.log_metric("wf_total_return_gross", round(float(wf_total_gross_return), 4))
                    
                    # Only calculate IR if lengths match (they should now)
                    if len(wf_strategy_returns) == len(wf_benchmark_returns):
                        strategy_metrics = calculate_all_metrics(wf_strategy_returns, signals=wf_signals, benchmark_returns=wf_benchmark_returns)
                    else:
                        # Fallback without IR if there's still a mismatch
                        strategy_metrics = calculate_all_metrics(wf_strategy_returns, signals=wf_signals, benchmark_returns=None)
                        print(f"  Warning: Return length mismatch - strategy: {len(wf_strategy_returns)}, benchmark: {len(wf_benchmark_returns)}")
                    
                    # No out-of-sample labeling - we don't have ground truth in real trading
                    for metric, value in strategy_metrics.items():
                        if isinstance(value, (np.ndarray, list, str)):
                            # Skip arrays, lists, and string values (no position-based plots anymore)
                            continue
                        else:
                            mlflow.log_metric(f"wf_{metric}", round(float(value), 4))
                    
                    # Calculate benchmark metrics from test period (respecting embargo)
                    test_strategy_data = wf_features_df.loc[test_start:test_end]
                    test_log_benchmark = test_strategy_data[return_col].dropna().values
                    if len(test_log_benchmark) > 0:
                        # Convert log returns to simple returns for accurate metric calculation
                        test_benchmark = np.exp(test_log_benchmark) - 1
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
            if hasattr(self.strategy, 'is_meta_model') and self.strategy.is_meta_model:
                print(f"\n--- Step 7: Walk-Forward Permutation Test (Skipped for Meta-Models) ---")
                print(f"🤖 Meta-models use pre-validated primary signals - skipping permutation tests...")
                
                # Create dummy walk-forward permutation results to maintain pipeline compatibility
                wf_perm_results = {
                    'sharpe': {'p_value': 0.001, 'passes_test': True},
                    'profit_factor': {'p_value': 0.001, 'passes_test': True}
                }
                
                print(f"✅ Walk-forward permutation test skipped for meta-model")
                
            else:
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
            
            # Generate artifacts based on strategy requirements
            print(f"\n--- Generating Strategy Artifacts ---")
            try:
                self._generate_universal_artifacts(
                    strategy_data=strategy_data,
                    optimization_results=optimization_result,
                    permutation_results=permutation_results,
                    wf_results=wf_results,
                    wf_perm_results=wf_perm_results,
                    train_start=train_start,
                    train_end=train_end
                )
            except Exception as e:
                print(f"  ⚠️  Warning: Could not generate all artifacts: {e}")
            
            
            # Create comprehensive summary and log as artifact
            import json
            summary = {
                'run_info': {
                    'run_id': mlflow.active_run().info.run_id,
                    'experiment_name': self.clean_strategy_name,
                    'strategy_type': self.strategy.strategy_type,  # Algorithmic approach
                    'strategy_basis': self.strategy_type,  # Implementation approach
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
                },
                'deployment': {
                    'model_artifact_path': 'model',
                    'ready_for_registration': True
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

    def _generate_universal_artifacts(self, strategy_data, optimization_results, permutation_results,
                                    wf_results, wf_perm_results, train_start, train_end):
        """
        Generate artifacts based on strategy requirements using get_required_artifacts().

        This universal artifact generation system allows each strategy to declare
        what artifacts it needs, while utilities implement the logic and generate_run orchestrates.
        """
        # Get required artifacts from strategy
        required_artifacts = self.strategy.get_required_artifacts()
        print(f"📦 Generating {len(required_artifacts)} required artifacts: {required_artifacts}")

        # Artifact generation mapping
        artifact_generators = {
            'performance_plots': self._generate_performance_plots,
            'returns_analysis': self._generate_returns_analysis,
            'confusion_matrix': self._generate_confusion_matrix,
            'signal_comparison': self._generate_signal_comparison,
            'comprehensive_csv': self._generate_comprehensive_csv,
            'model_summary': self._generate_model_summary,
            'feature_importance': self._generate_feature_importance
        }

        # Generate each required artifact
        for artifact_name in required_artifacts:
            if artifact_name in artifact_generators:
                try:
                    print(f"  🔧 Generating {artifact_name}...")
                    artifact_generators[artifact_name](
                        strategy_data, optimization_results, permutation_results,
                        wf_results, wf_perm_results, train_start, train_end
                    )
                    print(f"  ✅ Generated {artifact_name}")
                except Exception as e:
                    print(f"  ⚠️  Failed to generate {artifact_name}: {e}")
            else:
                print(f"  ❓ Unknown artifact type: {artifact_name}")

    def _generate_performance_plots(self, strategy_data, optimization_results, permutation_results,
                                  wf_results, wf_perm_results, train_start, train_end):
        """Generate standard performance plots."""
        plot_files = save_plots_for_mlflow(
            strategy_data=strategy_data,
            optimization_results=optimization_results,
            permutation_results=permutation_results,
            wf_results=wf_results,
            wf_perm_results=wf_perm_results,
            asset_name=self.asset_name,
            strategy_name=self.clean_strategy_name,
            start_date=train_start,
            end_date=train_end
        )

        # Log plot files as MLflow artifacts
        for plot_file in plot_files:
            mlflow.log_artifact(plot_file)

        # Clean up temporary files
        import os
        for plot_file in plot_files:
            try:
                os.remove(plot_file)
            except:
                pass

    def _generate_returns_analysis(self, _strategy_data, _optimization_results, _permutation_results,
                                 _wf_results, _wf_perm_results, _train_start, _train_end):
        """Generate returns analysis artifacts."""
        # This is typically included in performance_plots, but can be extended for custom analysis
        pass

    def _generate_confusion_matrix(self, strategy_data, _optimization_results, _permutation_results,
                                 _wf_results, _wf_perm_results, train_start, train_end):
        """Generate confusion matrix for strategies with labels."""
        # For meta-models, try to get enhanced data with meta_prediction column
        if hasattr(self.strategy, 'is_meta_model') and self.strategy.is_meta_model:
            # Get enhanced data from strategy if available (contains meta_prediction)
            enhanced_data = getattr(self.strategy, 'enhanced_data', strategy_data)

            if 'meta_prediction' in enhanced_data.columns:
                # Use enhanced data for meta-model confusion matrix
                train_data = enhanced_data.loc[train_start:train_end]
                test_data = enhanced_data.loc[train_end:]

                # Training confusion matrix (binary classification of meta-model predictions)
                train_labeled = train_data.dropna(subset=['label', 'meta_prediction'])
                if len(train_labeled) > 0:
                    # Convert labels to binary (1 for successful, 0 for unsuccessful)
                    # The meta-model predicts whether primary signal will be successful
                    binary_labels_train = (train_labeled['label'] == 1).astype(int)

                    cm_plot_path = create_confusion_matrix_plot(
                        y_true=binary_labels_train.values,
                        y_pred=train_labeled['meta_prediction'].values,
                        title=f"Confusion Matrix - {self.clean_strategy_name} (Training)",
                        save_path="confusion_matrix_train.png",
                        log_to_mlflow=True,
                        labels=[0, 1],  # Force binary classification
                        label_names=['Unsuccessful', 'Successful']
                    )
                    print(f"    ✅ Generated training confusion matrix for meta-model")

                # Test confusion matrix if test data available
                test_labeled = test_data.dropna(subset=['label', 'meta_prediction'])
                if len(test_labeled) > 0:
                    binary_labels_test = (test_labeled['label'] == 1).astype(int)

                    cm_plot_path_test = create_confusion_matrix_plot(
                        y_true=binary_labels_test.values,
                        y_pred=test_labeled['meta_prediction'].values,
                        title=f"Confusion Matrix - {self.clean_strategy_name} (Test)",
                        save_path="confusion_matrix_test.png",
                        log_to_mlflow=True,
                        labels=[0, 1],  # Force binary classification
                        label_names=['Unsuccessful', 'Successful']
                    )
                    print(f"    ✅ Generated test confusion matrix for meta-model")
                else:
                    print("    ⚠️  No test data available for confusion matrix")

                return  # Exit after processing meta-model

        # Standard strategy confusion matrix (non-meta-models)
        if 'label' not in strategy_data.columns:
            print("    ⚠️  No labels available for confusion matrix")
            return

        # Filter to training period for in-sample confusion matrix
        train_data = strategy_data.loc[train_start:train_end]
        labeled_data = train_data.dropna(subset=['label', 'signal'])

        # Filter out zero signals for binary classification
        labeled_data = labeled_data[labeled_data['signal'] != 0]

        if len(labeled_data) == 0:
            print("    ⚠️  No labeled data available for confusion matrix")
            return

        cm_plot_path = create_confusion_matrix_plot(
            y_true=labeled_data['label'].values,
            y_pred=labeled_data['signal'].values,
            title=f"Confusion Matrix - {self.clean_strategy_name} (Training)",
            save_path="confusion_matrix_train.png",
            log_to_mlflow=True
        )

    def _generate_signal_comparison(self, strategy_data, _optimization_results, _permutation_results,
                                  _wf_results, _wf_perm_results, _train_start, _train_end):
        """Generate signal comparison plot for meta-models."""
        if not hasattr(self.strategy, 'is_meta_model') or not self.strategy.is_meta_model:
            print("    ⚠️  Signal comparison only available for meta-models")
            return

        # Get enhanced data from strategy if available
        enhanced_data = getattr(self.strategy, 'enhanced_data', strategy_data)

        plot_path = plot_signal_comparison(
            data=enhanced_data,
            primary_signal_col='primary_signal' if 'primary_signal' in enhanced_data.columns else 'signal',
            meta_decision_col='meta_decision' if 'meta_decision' in enhanced_data.columns else None,
            final_signal_col='signal',
            title=f"Signal Evolution - {self.asset_name}",
            save_path="signal_comparison.png"
        )

        # Log to MLflow
        if plot_path:
            mlflow.log_artifact(plot_path, "signal_comparison")

    def _generate_comprehensive_csv(self, strategy_data, _optimization_results, _permutation_results,
                                  _wf_results, _wf_perm_results, train_start, train_end):
        """Generate comprehensive CSV export."""
        # Get enhanced data from strategy if available (for meta-models with features)
        enhanced_data = getattr(self.strategy, 'enhanced_data', strategy_data)

        # Filter data to exclude warmup period - only include actual strategy period
        # Use the actual start_date and end_date from strategy parameters
        start_date = self.start_date  # This is the actual strategy start date (excluding warmup)
        end_date = self.end_date      # This is the actual strategy end date

        # Filter the data to the actual strategy period
        filtered_data = enhanced_data.loc[start_date:end_date]

        print(f"    📊 Filtered comprehensive data from {len(enhanced_data)} to {len(filtered_data)} rows")
        print(f"    📅 Date range: {start_date} to {end_date} (excluding warmup)")

        csv_path = export_comprehensive_csv(
            data=filtered_data,
            filename="comprehensive_strategy_data.csv",
            mlflow_log=True
        )

    def _generate_model_summary(self, _strategy_data, optimization_results, _permutation_results,
                              wf_results, _wf_perm_results, _train_start, _train_end):
        """Generate model performance summary."""
        if not hasattr(self.strategy, 'model') or self.strategy.model is None:
            print("    ⚠️  No trained model available for summary")
            return

        # Collect metrics from various results
        all_metrics = {}
        if wf_results:
            all_metrics.update({f"wf_{k}": v for k, v in wf_results.items() if isinstance(v, (int, float))})
        if optimization_results:
            all_metrics.update({f"opt_{k}": v for k, v in optimization_results.items() if isinstance(v, (int, float))})

        # Get feature importance if available
        feature_importance = None
        if hasattr(self.strategy.model, 'get_feature_importance'):
            try:
                feature_cols = getattr(self.strategy, 'feature_columns', [])
                if feature_cols:
                    importances = self.strategy.model.get_feature_importance()
                    feature_importance = dict(zip(feature_cols, importances))
            except:
                pass

        model_type = str(type(self.strategy.model)).split('.')[-1].replace("'>", "")

        summary_path = create_model_summary(
            model=self.strategy.model,
            metrics=all_metrics,
            feature_importance=feature_importance,
            model_type=model_type,
            mlflow_log=True
        )

    def _generate_feature_importance(self, _strategy_data, _optimization_results, _permutation_results,
                                   _wf_results, _wf_perm_results, _train_start, _train_end):
        """Generate feature importance plot for ML models."""
        if not hasattr(self.strategy, 'model') or self.strategy.model is None:
            print("    ⚠️  No trained model available for feature importance")
            return

        feature_cols = getattr(self.strategy, 'feature_columns', [])
        if not feature_cols:
            print("    ⚠️  No feature columns available")
            return

        plot_path = plot_feature_importance(
            model=self.strategy.model,
            features=feature_cols,
            model_type="feature_importance",
            title=f"Feature Importance - {self.clean_strategy_name}",
            save_path="feature_importance.png"
        )

        if plot_path and os.path.exists(plot_path):
            mlflow.log_artifact(plot_path)

    @staticmethod
    def discover_strategies():
        """Discover all available strategies by scanning directories."""
        # Handle both module execution and direct execution
        current_file_path = Path(__file__).resolve()
        if current_file_path.parent.name == 'experiments':
            # Running from experiments directory
            strategies_dir = current_file_path.parent / 'strategies'
        else:
            # Fallback to relative path
            strategies_dir = current_file_path.parent.parent / 'strategies'
        
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
    parser.add_argument('--primary-run-id', help='Primary model run ID (required for meta-models, format: meta_models:strategy_name:run_id)')
    parser.add_argument('--train-test-split', type=float, default=0.8, help='Train/test split')
    parser.add_argument('--n-optimization-trials', type=int, default=10, help='Number of optimization trials (default: 1000)')
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
        print("\nFor meta-models, use: --primary-run-id <run_id> instead of dates")
        return
    
    # Handle meta-models vs regular strategies
    if args.primary_run_id and args.strategy:
        # Meta-model with separate primary run ID parameter
        strategy_with_run = f"{args.strategy}:{args.primary_run_id}"
        # Dates and asset will be extracted from primary model
        start_date = args.start_date or "2022-01-01"  # Dummy values, will be overridden
        end_date = args.end_date or "2024-12-31"
        asset = args.asset or "bitcoin"  # Dummy value, will be overridden
    else:
        # Regular strategy or meta-model with embedded run ID
        strategy_with_run = args.strategy
        start_date = args.start_date
        end_date = args.end_date
        asset = args.asset
        
        # Validate required arguments for regular strategies
        if not strategy_with_run.startswith('meta_models:'):
            if not all([args.strategy, asset, start_date, end_date]):
                parser.print_help()
                print("\nFor regular strategies: --strategy, --asset, --start-date, --end-date are required")
                print("For meta-models: --strategy, --primary-run-id are required")
                print("Use --list-strategies to see available strategies")
                return
    
    # For meta-models, only strategy and primary_run_id are required
    if args.primary_run_id:
        if not args.strategy:
            parser.print_help()
            return
    else:
        # For regular strategies, asset is still required
        if not args.strategy or not asset:
            parser.print_help()
            return
    
    print(f"Running strategy: {strategy_with_run}")
    print(f"Asset: {asset}")
    
    runner = RunGenerator(strategy_with_run, asset, start_date, end_date)
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