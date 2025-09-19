import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.meta_models.base_meta_strategy import MetaStrategy
from utils.topological_features import extract_multi_series_topological_features

class TopologicalMRValidatorStrategy(MetaStrategy):

    "Barebones meta-model using only close price topological features and primary signals."

    def __init__(self, asset: str = 'bitcoin', primary_run_id: str = None):
        name = f"topological_mr_validator_{asset}"
        super().__init__(name, asset, primary_run_id)

        # Topological feature configuration
        self.window_length = 21
        self.tau = 3
        self.embedding_dim = 3
        self.max_dimension = 1

        # Model storage
        self.model = None
        self.feature_columns = None

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

    def get_normalization_config(self) -> Dict:
        """
        Configure EWMA normalization for contextual features.
        Primary signals and labels are excluded from normalization.
        Topological features will be calculated AFTER normalization using the normalized base features.
        """
        return {
            'exclude': [
                'signal',  # Primary model signal - never normalize
                'label',   # Triple barrier label - never normalize
                'timestamp',  # Time index
                'barrier_touched', 'days_to_barrier', 'return_at_barrier',  # Label metadata
            ]
        }

    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply EWMA Z-Score normalization to base features before topological analysis.
        """
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from utils.normalization import ColumnNormalizer

            # Get normalization configuration
            normalization_config = self.get_normalization_config()

            if normalization_config:
                print(f"🔧 Applying EWMA normalization for topological analysis...")

                # Initialize normalizer
                normalizer = ColumnNormalizer()
                normalizer.configure(normalization_config)

                # Apply normalization
                df_normalized = normalizer.normalize(df)

                excluded_cols = normalization_config.get('exclude', [])
                normalized_cols = [col for col in df.columns if col not in excluded_cols]

                print(f"✅ Normalized {len(normalized_cols)} columns for topological features")
                return df_normalized
            else:
                print(f"⚠️  No normalization config found, using raw features")
                return df

        except ImportError as e:
            print(f"⚠️  Could not import normalization utilities: {e}")
            print(f"   Using raw features for topological analysis")
            return df

    def _add_topological_features_inplace(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add topological features to the DataFrame and return the enhanced DataFrame.
        Applies EWMA normalization to base features before topological analysis.
        """
        # Apply normalization to base features before topological analysis
        df_normalized = self._apply_normalization(df.copy())

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

                # Ensure indices are aligned
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

        return total_warmup  # Approximately 33 days for default settings

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
            # Fallback: add common topological feature patterns
            topological_features = {
                f'{self.asset}_close_topo_num_holes_0': [2.0],
                f'{self.asset}_close_topo_num_holes_1': [1.0],
                f'{self.asset}_close_topo_max_hole_lifetime_0': [0.15],
                f'{self.asset}_close_topo_max_hole_lifetime_1': [0.12],
                f'{self.asset}_close_topo_avg_hole_lifetime_0': [0.10],
                f'{self.asset}_close_topo_avg_hole_lifetime_1': [0.08],
                f'{self.asset}_close_topo_l1_norm_0': [1.5],
                f'{self.asset}_close_topo_l1_norm_1': [1.2],
                f'{self.asset}_close_topo_persistence_entropy_0': [0.8],
                f'{self.asset}_close_topo_persistence_entropy_1': [0.6]
            }
            sample_data.update(topological_features)

        return pd.DataFrame(sample_data)

    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Generate meta-model signals using trained CatBoost model.
        """
        if self.model is None:
            print("⚠️  Model not trained. Using primary signals only.")
            # Return original data with primary signals
            result_df = data.copy()
            if 'signal' in data.columns:
                # Ensure signals are properly formatted (+1/-1, no 0s during trading)
                primary_signals = data['signal'].values.copy()
                # Fix -0.0 vs 0.0 issue
                primary_signals = np.round(primary_signals, decimals=8)
                primary_signals[primary_signals == -0.0] = 0.0
                result_df['signal'] = primary_signals
                result_df['meta_decision'] = np.nan  # No meta decision without model
            else:
                raise ValueError("Primary signals not found in data")
            return result_df

        # Add topological features
        enhanced_data = self._add_topological_features_inplace(data.copy())

        # Prepare features using the same columns as training
        if self.feature_columns is None:
            print("⚠️  Feature columns not defined. Using primary signals only.")
            result_df = enhanced_data.copy()
            if 'signal' in enhanced_data.columns:
                # Ensure signals are properly formatted
                primary_signals = enhanced_data['signal'].values.copy()
                primary_signals = np.round(primary_signals, decimals=8)
                primary_signals[primary_signals == -0.0] = 0.0
                result_df['signal'] = primary_signals
                result_df['meta_decision'] = np.nan
            else:
                raise ValueError("Primary signals not found in data")
            return result_df

        # Extract features
        available_features = [col for col in self.feature_columns if col in enhanced_data.columns]
        if len(available_features) == 0:
            print("⚠️  No features available for prediction. Using primary signals only.")
            result_df = enhanced_data.copy()
            if 'signal' in enhanced_data.columns:
                # Ensure signals are properly formatted
                primary_signals = enhanced_data['signal'].values.copy()
                primary_signals = np.round(primary_signals, decimals=8)
                primary_signals[primary_signals == -0.0] = 0.0
                result_df['signal'] = primary_signals
                result_df['meta_decision'] = np.nan
            else:
                raise ValueError("Primary signals not found in data")
            return result_df

        X = enhanced_data[available_features].fillna(0)

        # Generate predictions
        try:
            # Meta-model predicts {0, 1} - whether primary signal will be successful
            meta_predictions = self.model.predict(X)

            # Apply confidence threshold if specified
            if params.get('use_confidence', False):
                proba = self.model.predict_proba(X)
                max_proba = np.max(proba, axis=1)
                confidence_threshold = params.get('confidence_threshold', 0.6)
                meta_predictions = np.where(max_proba >= confidence_threshold, meta_predictions, 0)

            # Get primary signals from the input data
            if 'signal' not in enhanced_data.columns:
                raise ValueError("Primary signals not found in data")

            primary_signals = enhanced_data['signal'].values.copy()

            # Fix -0.0 vs 0.0 issue and validate primary signals
            primary_signals = np.round(primary_signals, decimals=8)
            primary_signals[primary_signals == -0.0] = 0.0

            # Store the original PRIMARY signals (not zeros)
            # During warmup, primary signals might be 0, but after warmup they should be +1/-1
            original_primary_signals = primary_signals.copy()

            # Meta predictions are {0, 1} - convert to {-1, +1} for multiplication
            meta_decisions = np.where(meta_predictions == 1, 1, -1)

            # For the final signal, we only trade when:
            # 1. Primary signal agrees with meta decision
            # 2. Primary signal is not 0 (not in warmup)
            final_signals = np.where(
                (primary_signals != 0) & (primary_signals == meta_decisions),
                primary_signals,  # Keep primary signal if meta agrees
                0  # Otherwise don't trade
            )

            # Log the signal combination
            non_warmup_mask = primary_signals != 0
            if non_warmup_mask.any():
                print(f"   📊 Signal combination (non-warmup):")
                print(f"      Primary signals: {dict(zip(*np.unique(original_primary_signals[non_warmup_mask], return_counts=True)))}")
                print(f"      Meta decisions: {dict(zip(*np.unique(meta_decisions[non_warmup_mask], return_counts=True)))}")
                print(f"      Final signals: {dict(zip(*np.unique(final_signals[non_warmup_mask], return_counts=True)))}")

            # Return enhanced data with both signals
            result_df = enhanced_data.copy()
            result_df['primary_signal'] = original_primary_signals  # Store original primary
            result_df['meta_decision'] = meta_decisions  # Store meta decision (+1/-1)
            result_df['signal'] = final_signals  # Final trading signal

            return result_df

        except Exception as e:
            print(f"⚠️  Error in signal generation: {e}")
            print("   Falling back to primary signals")
            result_df = enhanced_data.copy()
            if 'signal' in enhanced_data.columns:
                # Ensure signals are properly formatted
                primary_signals = enhanced_data['signal'].values.copy()
                primary_signals = np.round(primary_signals, decimals=8)
                primary_signals[primary_signals == -0.0] = 0.0
                result_df['signal'] = primary_signals
                result_df['primary_signal'] = primary_signals
                result_df['meta_decision'] = np.nan
            else:
                raise ValueError("Primary signals not found in data")
            return result_df
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str,
                 n_trials: int = 100, **kwargs) -> Dict:
        """
        Train CatBoost meta-learner on topological features + primary signals.
        """
        try:
            from catboost import CatBoostClassifier
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

        print(f"🧠 Training CatBoost meta-learner...")

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
                    # If it's 0, leave it as 0 (shouldn't happen after warmup)

                # Validate
                post_warmup_signals = signals_clean[first_valid_idx:]
                post_warmup_non_na = post_warmup_signals[~na_mask[first_valid_idx:]]
                unique_post_warmup = np.unique(post_warmup_non_na[post_warmup_non_na != 0])
                if len(unique_post_warmup) > 0 and not all(s in [1, -1] for s in unique_post_warmup):
                    print(f"   ⚠️ WARNING: Invalid signals found after warmup, fixing...")
                    # Force fix any remaining issues
                    for i in range(first_valid_idx, len(signals_clean)):
                        if not na_mask[i] and signals_clean[i] != 0:
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

        # Prepare features - only topological features and primary signal
        feature_cols = []

        # Add topological features
        topo_cols = [col for col in labeled_data.columns if '_topo_' in col]
        feature_cols.extend(topo_cols)

        # Add primary signal as a feature
        if 'signal' in labeled_data.columns:
            feature_cols.append('signal')

        # Remove any columns with all NaN values
        feature_cols = [col for col in feature_cols
                       if col in labeled_data.columns and not labeled_data[col].isna().all()]

        if len(feature_cols) == 0:
            print("⚠️  No valid features found for training")
            return {
                'best_params': {},
                'best_value': 0.0,
                'study': None
            }

        print(f"   Features: {len(feature_cols)} ({len(topo_cols)} topological)")

        # Prepare training data
        X_train = labeled_data[feature_cols].fillna(0)  # Fill NaN with 0
        y_train = labeled_data['label']

        # Store feature columns for later use
        self.feature_columns = feature_cols

        # Check if we have enough samples for cross-validation
        if len(X_train) < 30:
            print(f"⚠️  Limited training data ({len(X_train)} samples) - using simple training")
            # Simple training without hyperparameter optimization
            self.model = CatBoostClassifier(
                iterations=100,
                depth=4,
                learning_rate=0.1,
                verbose=False,
                random_state=42
            )
            self.model.fit(X_train, y_train)

            # Calculate training accuracy
            train_pred = self.model.predict(X_train)
            train_score = balanced_accuracy_score(y_train, train_pred)

            return {
                'best_params': {
                    'iterations': 100,
                    'depth': 4,
                    'learning_rate': 0.1
                },
                'best_value': train_score,
                'study': None
            }

        # Hyperparameter optimization with Optuna
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_strength': trial.suggest_float('random_strength', 0, 1),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'border_count': trial.suggest_int('border_count', 32, 128),
                'verbose': False,
                'random_state': 42
            }

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=min(3, len(X_train) // 10))
            scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = CatBoostClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=20)

                # Use balanced accuracy for imbalanced classes
                pred = model.predict(X_val)
                scores.append(balanced_accuracy_score(y_val, pred))

            return np.mean(scores)

        # Create and optimize study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=min(n_trials, 50))  # Limit trials for meta-models

        best_params = study.best_params
        best_value = study.best_value

        print(f"   Best CV score: {best_value:.4f}")
        print(f"   Best params: {best_params}")

        # Train final model with best parameters
        self.model = CatBoostClassifier(**best_params)
        self.model.fit(X_train, y_train)

        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
            # Sort by importance
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"   Top features: {[f'{name}: {imp:.3f}' for name, imp in top_features]}")

        # Note: Standard meta-model artifacts (training/test data) are now handled by generate_run.py
        # Only strategy-specific artifacts (if any) should be saved here
        print(f"   📝 Standard artifacts will be handled by generate_run.py")

        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }