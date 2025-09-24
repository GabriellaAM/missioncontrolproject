import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.meta_models.base_meta_strategy import MetaStrategy
from utils.topological_features import extract_multi_series_topological_features
from utils.plotting import create_feature_importance_plot

class TopoCatBoostStrategy(MetaStrategy):

    "Meta-model using CatBoost with topological features and primary signals."

    def __init__(self, asset: str = 'bitcoin', primary_run_id: str = None):
        name = f"topo_catboost_{asset}"
        super().__init__(name, asset, primary_run_id)

        # Topological feature configuration
        self.window_length = 50
        self.tau = 3
        self.embedding_dim = 3
        self.max_dimension = 2

        # Model storage
        self.model = None
        self.feature_columns = None
        self.enhanced_data = None  # Store processed data for artifact generation

        # Define univariate series for topological analysis
        self.topological_series = [
            f'{self.asset}_close'
        ]

    def get_required_features(self) -> Dict[str, Any]:
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': [],
            'yahoo_tickers': [],
            'calculated_features': {}
        }

    def get_required_artifacts(self) -> list:
        """
        Return list of artifacts required for CatBoost meta-model.

        Inherits meta-model artifacts from base class and adds CatBoost-specific ones.
        """
        base_artifacts = super().get_required_artifacts()

        # Add CatBoost-specific artifacts
        catboost_artifacts = [
            'feature_importance'  # CatBoost feature importance visualization
        ]

        return base_artifacts + catboost_artifacts

    def get_normalization_config(self) -> Dict:
        """
        Configure normalization for raw features only.
        Primary signals, labels, and topological features are excluded from normalization.
        Topological features are calculated FROM normalized base features but are not normalized themselves.
        """
        return {
            'exclude': [
                'signal',  # Primary model signal - never normalize
                'label',   # Triple barrier label - never normalize
                'timestamp',  # Time index
                'barrier_touched', 'days_to_barrier', 'return_at_barrier',  # Label metadata
                'bitcoin_log_return', 'log_return',  # Any return-based labels
                'meta_decision', 'final_signal'  # Meta-model outputs
            ],
            'exclude_patterns': [
                '_topo_'  # Exclude all topological features from normalization
            ]
        }

    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standard Z-Score normalization to base features before topological analysis.
        """
        from sklearn.preprocessing import StandardScaler

        # Get normalization configuration
        normalization_config = self.get_normalization_config()

        if normalization_config:
            print(f"🔧 Applying standard normalization for topological analysis...")

            # Get columns to exclude from normalization
            excluded_cols = normalization_config.get('exclude', [])
            excluded_patterns = normalization_config.get('exclude_patterns', [])

            # Identify columns to normalize (exclude both explicit columns and pattern matches)
            cols_to_normalize = []
            for col in df.columns:
                if col in excluded_cols:
                    continue
                if any(pattern in col for pattern in excluded_patterns):
                    continue
                cols_to_normalize.append(col)

            if not cols_to_normalize:
                print(f"⚠️  No columns to normalize, using raw features")
                return df

            # Create a copy for normalization
            df_normalized = df.copy()

            # Apply standard scaling to each column individually
            for col in cols_to_normalize:
                if df[col].notna().sum() > 1:  # Need at least 2 non-NaN values
                    # Get non-NaN values for fitting
                    non_nan_mask = df[col].notna()
                    values = df.loc[non_nan_mask, col].values.reshape(-1, 1)

                    # Fit and transform
                    scaler = StandardScaler()
                    scaled_values = scaler.fit_transform(values)

                    # Update the normalized dataframe
                    df_normalized.loc[non_nan_mask, col] = scaled_values.flatten()

            print(f"✅ Normalized {len(cols_to_normalize)} columns for topological features")
            return df_normalized
        else:
            print(f"⚠️  No normalization config found, using raw features")
            return df

    def _add_topological_features_inplace(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add topological features to the DataFrame and return the enhanced DataFrame.
        Applies EMA transformation and normalization to base features before topological analysis.
        """
        # Transform asset_close using 10-period EMA before normalization
        df_transformed = df.copy()
        close_col = f'{self.asset}_close'

        if close_col in df_transformed.columns:
            # Apply 10-period EMA transformation
            df_transformed[close_col] = df_transformed[close_col].ewm(span=10, min_periods=1).mean()
            print(f"🔧 Applied 10-period EMA transformation to {close_col}")

        # Apply normalization to transformed features before topological analysis
        df_normalized = self._apply_normalization(df_transformed)

        # Create configuration for each series
        series_configs = {}

        for series_name in self.topological_series:
            if series_name in df_normalized.columns:
                series_configs[series_name] = {
                    'window_length': self.window_length,
                    'tau': self.tau,
                    'embedding_dim': self.embedding_dim,
                    'max_dimension': self.max_dimension
                }

        if series_configs:
            print(f"🔬 Extracting topological features for {len(series_configs)} normalized series...")

            # Extract topological features from normalized data
            topo_features = extract_multi_series_topological_features(
                data=df_normalized,
                series_configs=series_configs
            )

            if not topo_features.empty:
                # Merge topological features with the main DataFrame
                original_shape = df.shape

                # Debug: Show exact column names being added
                print(f"   Topological feature columns being added:")
                for col in topo_features.columns[:10]:  # Show first 10
                    print(f"      - {col}")
                if len(topo_features.columns) > 10:
                    print(f"      ... and {len(topo_features.columns) - 10} more")

                # CRITICAL FIX: Prevent duplicate columns by using proper suffixes
                # Check for potential column conflicts
                conflicting_cols = set(df.columns).intersection(set(topo_features.columns))
                if conflicting_cols:
                    print(f"   ⚠️  Found {len(conflicting_cols)} conflicting columns: {list(conflicting_cols)[:5]}...")
                    # Use suffixes to prevent automatic _x/_y naming
                    df = df.merge(
                        topo_features,
                        left_index=True,
                        right_index=True,
                        how='left',
                        suffixes=('_original', '_topo')  # Explicit control over naming
                    )
                else:
                    # No conflicts, merge normally
                    df = df.merge(
                        topo_features,
                        left_index=True,
                        right_index=True,
                        how='left'
                    )

                print(f"✅ Added topological features: {original_shape} -> {df.shape}")
                print(f"   New columns: {len(topo_features.columns)}")

                return df
            else:
                print("⚠️  No topological features generated")
                return df
        else:
            print(f"⚠️  No valid series found for topological analysis")
            print(f"   Available columns: {list(df.columns)}")
            print(f"   Looking for: {self.topological_series}")
            return df

    def get_warmup_days(self, params: Dict = None) -> int:
        """
        Calculate required warmup period for topological features.
        """
        # Base warmup for time delay embedding
        embedding_warmup = (self.embedding_dim - 1) * self.tau

        # Warmup for sliding window
        window_warmup = self.window_length

        # Additional warmup for moving averages on topological features
        ma_warmup = 10

        # Total warmup
        total_warmup = embedding_warmup + window_warmup + ma_warmup

        return total_warmup  # Approximately 37 days for default settings

    def get_input_example(self) -> pd.DataFrame:
        """
        Generate input example with topological features for MLflow signature.
        """
        # Create base features sample - only essentials
        sample_data = {
            # Basic crypto features
            f'{self.asset}_close': [50000.0],
            # Primary signal and label (for meta-model)
            'signal': [1],
            'label': [1],
            # Log returns
            f'{self.asset}_log_return_1': [0.01]
        }

        # Add sample topological features if we have trained feature columns
        if hasattr(self, 'feature_columns') and self.feature_columns:
            # Add topological features that the model expects
            for feature_col in self.feature_columns:
                if '_topo_' in feature_col and feature_col not in sample_data:
                    # Generate realistic sample values for topological features
                    if 'num_holes' in feature_col or 'betti' in feature_col:
                        sample_data[feature_col] = [2.0]  # Number of holes
                    elif 'lifetime' in feature_col:
                        sample_data[feature_col] = [0.15]  # Hole lifetime
                    elif 'norm' in feature_col:
                        sample_data[feature_col] = [1.5]   # Norm values
                    elif 'entropy' in feature_col:
                        sample_data[feature_col] = [0.8]   # Entropy values
                    elif 'wasserstein' in feature_col:
                        sample_data[feature_col] = [0.3]   # Distance values
                    elif '_ma' in feature_col:
                        sample_data[feature_col] = [1.0]   # Moving averages
                    else:
                        sample_data[feature_col] = [1.0]   # Default value
        else:
            # No fallback - fail if we don't have trained feature columns
            raise RuntimeError("Model must be trained before generating input example. No feature columns available.")

        return pd.DataFrame(sample_data)

    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Generate meta-model signals using trained CatBoost model.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Cannot generate meta-model signals.")

        # Add topological features
        enhanced_data = self._add_topological_features_inplace(data.copy())

        # Prepare features using the same columns as training
        if self.feature_columns is None:
            raise RuntimeError("Feature columns not defined. Model must be trained first.")

        # Extract features
        available_features = [col for col in self.feature_columns if col in enhanced_data.columns]
        if len(available_features) == 0:
            raise RuntimeError(f"No features available for prediction. Required features: {self.feature_columns}, Available columns: {list(enhanced_data.columns)}")

        # Prepare features - don't fillna(0) on signal column!
        X = enhanced_data[available_features].copy()

        # Only fill NaN for non-signal features
        for col in available_features:
            if col != 'signal' and X[col].isna().any():
                X[col] = X[col].fillna(0)

        # After filling NaN for non-signal features, check if any NaN remains
        if X.isna().any().any():
            # If signal column has NaN, we need to handle those rows separately
            nan_mask = X.isna().any(axis=1)
            print(f"   ⚠️ Found {nan_mask.sum()} rows with NaN values")

            # For rows with NaN, set predictions to 0 (no trade)
            X_clean = X[~nan_mask].copy()
        else:
            X_clean = X
            nan_mask = pd.Series([False] * len(X), index=X.index)

        # Generate predictions
        try:
            # Initialize final signals array
            final_signals = np.zeros(len(enhanced_data))

            # Get primary signals from the input data
            primary_signals = enhanced_data['signal'].values

            # Fix -0.0 vs 0.0 issue
            primary_signals = np.round(primary_signals, decimals=8)
            primary_signals[primary_signals == -0.0] = 0.0

            # Only process rows where we have clean features
            if len(X_clean) > 0:
                # Meta-model predicts {0, 1} directly on features - whether primary signal will be successful
                # Features already processed correctly: topological features from normalized data
                meta_predictions = self.model.predict(X_clean)

                # Get indices where we made predictions (no NaN)
                clean_indices = np.where(~nan_mask)[0]

                # Combine: final_signal = primary_signal * meta_prediction
                # This gives us {-1, 0, 1}:
                # - Primary = -1, Meta = 0 -> Final = 0 (skip bearish signal)
                # - Primary = -1, Meta = 1 -> Final = -1 (take bearish signal)
                # - Primary = 1, Meta = 0 -> Final = 0 (skip bullish signal)
                # - Primary = 1, Meta = 1 -> Final = 1 (take bullish signal)
                for i, idx in enumerate(clean_indices):
                    if not pd.isna(primary_signals[idx]):
                        final_signals[idx] = primary_signals[idx] * meta_predictions[i]

            # Log the signal combination
            print(f"   📊 Signal combination:")
            primary_dist = dict(zip(*np.unique(primary_signals[~pd.isna(primary_signals)], return_counts=True)))
            print(f"      Primary signals: {primary_dist}")
            if len(X_clean) > 0:
                meta_dist = dict(zip(*np.unique(meta_predictions, return_counts=True)))
                print(f"      Meta predictions: {meta_dist}")
            final_dist = dict(zip(*np.unique(final_signals[final_signals != 0], return_counts=True))) if any(final_signals != 0) else {0: len(final_signals)}
            print(f"      Final signals: {final_dist}")

            # Return enhanced data with combined signal
            result_df = enhanced_data.copy()
            result_df['signal'] = final_signals

            return result_df

        except Exception as e:
            raise RuntimeError(f"Error in signal generation: {e}") from e


    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str,
                 n_trials: int = 1000, **kwargs) -> Dict:
        """
        Train CatBoost meta-learner on topological features + primary signals.
        """
        try:
            import catboost as cb
            import optuna
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import balanced_accuracy_score
        except ImportError as e:
            print(f"⚠️  Missing required packages: {e}")
            print("   Install with: pip install catboost optuna")
            return {
                'best_params': {},
                'best_value': 0.0,
                'study': None
            }

        print(f"🌳 Training CatBoost meta-learner...")

        # Add topological features to the data
        enhanced_data = self._add_topological_features_inplace(data.copy())

        # FIX SIGNALS: Ensure primary signals are +1/-1 after warmup (no 0s)
        if 'signal' in enhanced_data.columns:
            # Store original signals first
            enhanced_data['primary_signal'] = enhanced_data['signal'].copy()

            # Convert to numpy array and handle NA values
            signals = enhanced_data['signal'].values.copy()

            # Handle pandas NA/NaN values
            na_mask = pd.isna(signals)
            signals_clean = np.where(na_mask, 0, signals)  # Replace NA with 0 temporarily

            # Round and fix -0.0 issue
            signals_clean = np.round(signals_clean, decimals=8)
            signals_clean[signals_clean == -0.0] = 0.0

            # Find first non-zero signal (end of warmup)
            non_zero_indices = np.where(signals_clean != 0)[0]
            if len(non_zero_indices) > 0:
                first_valid_idx = non_zero_indices[0]

                # After warmup, ensure all signals are strictly +1 or -1 (no 0s)
                for i in range(first_valid_idx, len(signals_clean)):
                    if not na_mask[i] and signals_clean[i] != 0:  # Non-NA, non-zero signals
                        signals_clean[i] = 1 if signals_clean[i] > 0 else -1

            # Restore NA values where they originally were
            signals_final = np.where(na_mask, np.nan, signals_clean)
            enhanced_data['signal'] = signals_final
            print(f"   ✅ Primary signals validated and fixed")

        # Filter to training period and get data with labels
        train_data = enhanced_data.loc[train_start:train_end]
        labeled_data = train_data.dropna(subset=['label'])

        if len(labeled_data) == 0:
            print("⚠️  No labeled training data found")
            return {
                'best_params': {},
                'best_value': 0.0,
                'study': None
            }

        print(f"   Training samples: {len(labeled_data)}")
        print(f"   Label distribution: {labeled_data['label'].value_counts().to_dict()}")

        # Prepare features - CRITICAL FIX: Prevent data leakage from labels
        feature_cols = []

        # EXCLUDE label-related columns to prevent data leakage
        label_leakage_cols = {
            'label', 'barrier_touched', 'return_at_barrier', 'days_to_barrier',
            'bitcoin_log_return', 'log_return'  # Any return-based labels
        }

        # Add ALL topological features (excluding any label-related columns)
        available_topo_cols = [col for col in labeled_data.columns
                              if '_topo_' in col and col not in label_leakage_cols]
        feature_cols.extend(available_topo_cols)
        print(f"   Using ALL {len(available_topo_cols)} topological features (label leakage excluded)")

        # Add primary signal as a feature (if not in exclusion list)
        if 'signal' in labeled_data.columns and 'signal' not in label_leakage_cols:
            feature_cols.append('signal')
            print(f"   ✓ Added primary signal as feature")

        # CRITICAL VALIDATION: Ensure no label-leakage columns are included
        actual_leakage = [col for col in feature_cols if col in label_leakage_cols]
        if actual_leakage:
            print(f"   🚨 DATA LEAKAGE DETECTED: {actual_leakage}")
            raise ValueError(f"Data leakage detected: {actual_leakage} columns should not be features")

        # Remove any columns with all NaN values
        feature_cols = [col for col in feature_cols
                       if col in labeled_data.columns and not labeled_data[col].isna().all()]

        print(f"   ✅ Data leakage check passed - no label metadata in features")

        if len(feature_cols) == 0:
            print("⚠️  No valid features found for training")
            return {
                'best_params': {},
                'best_value': 0.0,
                'study': None
            }

        topo_cols = [col for col in feature_cols if '_topo_' in col]
        print(f"   Features: {len(feature_cols)} ({len(topo_cols)} topological + signal)")

        # Prepare training data
        X_full = labeled_data[feature_cols].copy()

        # Only fill NaN for topological features, NOT for signal column
        for col in feature_cols:
            if col != 'signal' and X_full[col].isna().any():
                X_full[col] = X_full[col].fillna(0)

        # Verify signal column integrity
        if 'signal' in X_full.columns:
            # Fix -0.0 vs 0.0 issue by rounding
            X_full['signal'] = X_full['signal'].round(decimals=8)
            X_full.loc[X_full['signal'] == -0.0, 'signal'] = 0.0

            signal_dist = X_full['signal'].value_counts().to_dict()
            print(f"   Primary signal distribution in training: {signal_dist}")

        y_full = labeled_data['label']

        # Store feature columns for later use
        self.feature_columns = feature_cols

        # Create train/test split (80/20) while preserving temporal order
        split_idx = int(0.8 * len(X_full))
        X_train = X_full.iloc[:split_idx]
        y_train = y_full.iloc[:split_idx]
        X_test = X_full.iloc[split_idx:]
        y_test = y_full.iloc[split_idx:]

        print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")

        # Check if we have enough samples for cross-validation
        if len(X_train) < 30:
            print(f"⚠️  Limited training data ({len(X_train)} samples) - using simple training")
            # Simple training without hyperparameter optimization
            self.model = cb.CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                verbose=False,
                auto_class_weights='Balanced'
            )
            self.model.fit(X_train, y_train)

            # Calculate training and test accuracy
            train_pred = self.model.predict(X_train)
            train_score = balanced_accuracy_score(y_train, train_pred)

            test_pred = self.model.predict(X_test)
            test_score = balanced_accuracy_score(y_test, test_pred)
            print(f"   ✅ Test accuracy: {test_score:.4f}")

            # Log test performance
            mlflow.log_metric("test_accuracy", test_score)
            mlflow.log_metric("test_samples", len(X_test))

            # Create feature importance artifacts
            self._create_feature_importance_artifacts(self.model, feature_cols)

            return {
                'best_params': {
                    'iterations': 100,
                    'learning_rate': 0.1,
                    'depth': 6
                },
                'best_value': train_score,
                'test_score': test_score,
                'study': None
            }

        # Hyperparameter optimization with Optuna
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_seed': 42,
                'verbose': False,
                'auto_class_weights': 'Balanced'
            }

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=min(3, len(X_train) // 10))
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                try:
                    model = cb.CatBoostClassifier(**params)
                    model.fit(X_tr, y_tr)

                    # Use balanced accuracy for imbalanced classes
                    pred = model.predict(X_val)
                    scores.append(balanced_accuracy_score(y_val, pred))
                except Exception:
                    # If model fails, return poor score
                    scores.append(0.0)

            return np.mean(scores)

        # Create and optimize study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=min(n_trials, 200))  # Fewer trials for CatBoost

        best_params = study.best_params
        best_value = study.best_value

        print(f"   Best CV score: {best_value:.4f}")
        print(f"   Best params: {best_params}")

        # Train final model with best parameters
        final_params = best_params.copy()
        final_params['random_seed'] = 42
        final_params['verbose'] = False
        final_params['auto_class_weights'] = 'Balanced'

        self.model = cb.CatBoostClassifier(**final_params)
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        test_pred = self.model.predict(X_test)
        test_score = balanced_accuracy_score(y_test, test_pred)
        print(f"   ✅ Test accuracy: {test_score:.4f}")

        # Log test performance
        mlflow.log_metric("test_accuracy", test_score)
        mlflow.log_metric("test_samples", len(X_test))

        # Create feature importance artifacts
        self._create_feature_importance_artifacts(self.model, feature_cols)

        # Store enhanced data for artifact generation
        self.enhanced_data = enhanced_data
        print(f"   ✅ Stored enhanced data for artifacts: {enhanced_data.shape}")

        return {
            'best_params': best_params,
            'best_value': best_value,
            'test_score': test_score,
            'study': study
        }

    def _create_feature_importance_artifacts(self, model, feature_names):
        """Create feature importance visualization and log as MLflow artifact."""
        try:
            import tempfile
            import os

            # Get feature importance
            importance_values = model.get_feature_importance()

            # Create temporary file for the plot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name

            # Create the plot using utility function
            create_feature_importance_plot(
                importance_values=importance_values,
                feature_names=feature_names,
                save_path=temp_path,
                title="TopoCatBoost Feature Importance"
            )

            # Log as MLflow artifact
            mlflow.log_artifact(temp_path, "feature_importance")

            # Clean up temporary file
            os.unlink(temp_path)

            print("   ✅ Feature importance plot saved as MLflow artifact")

        except Exception as e:
            print(f"   ⚠️ Could not create feature importance artifacts: {e}")