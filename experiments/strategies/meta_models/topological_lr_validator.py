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

class TopologicalLRValidatorStrategy(MetaStrategy):

    "Barebones meta-model using logistic regression with close price topological features and primary signals."

    def __init__(self, asset: str = 'bitcoin', primary_run_id: str = None):
        name = f"topological_lr_validator_{asset}"
        super().__init__(name, asset, primary_run_id)

        # Topological feature configuration
        self.window_length = 21
        self.tau = 3
        self.embedding_dim = 3
        self.max_dimension = 1  # We only use dimension 1 features

        # Model storage
        self.model = None
        self.feature_columns = None
        self.scaler = None
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

                # Debug: Show exact column names being added
                print(f"   Topological feature columns being added:")
                for col in topo_features.columns[:10]:  # Show first 10
                    print(f"      - {col}")
                if len(topo_features.columns) > 10:
                    print(f"      ... and {len(topo_features.columns) - 10} more")

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
            # No fallback - fail if we don't have trained feature columns
            raise RuntimeError("Model must be trained before generating input example. No feature columns available.")

        return pd.DataFrame(sample_data)

    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Generate meta-model signals using trained logistic regression model.
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

        # Scale features if scaler exists
        if self.scaler is not None and len(X_clean) > 0:
            X_scaled = self.scaler.transform(X_clean)
        else:
            X_scaled = X_clean.values if len(X_clean) > 0 else np.array([])

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
                # Meta-model predicts {0, 1} - whether primary signal will be successful
                meta_predictions = self.model.predict(X_scaled)

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

    def _create_confusion_matrix_artifacts(self, X_train, y_train, X_test, y_test, model, scaler):
        """
        Create and save confusion matrix artifacts for train and test sets.
        """
        try:
            from sklearn.metrics import confusion_matrix, classification_report

            print(f"📊 Creating confusion matrix artifacts...")

            # Scale data
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Train confusion matrix
            cm_train = confusion_matrix(y_train, y_train_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Fail', 'Success'], yticklabels=['Fail', 'Success'])
            plt.title('Training Set Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix_train.png', dpi=300, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact('confusion_matrix_train.png')

            # Test confusion matrix
            cm_test = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Fail', 'Success'], yticklabels=['Fail', 'Success'])
            plt.title('Test Set Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact('confusion_matrix_test.png')

            # Classification reports
            train_report = classification_report(y_train, y_train_pred, output_dict=True)
            test_report = classification_report(y_test, y_test_pred, output_dict=True)

            # Save as DataFrames
            pd.DataFrame(train_report).transpose().to_csv('classification_report_train.csv')
            pd.DataFrame(test_report).transpose().to_csv('classification_report_test.csv')
            mlflow.log_artifact('classification_report_train.csv')
            mlflow.log_artifact('classification_report_test.csv')

            print(f"   ✅ Saved confusion matrices and classification reports")

            return {
                'train_accuracy': train_report['accuracy'],
                'test_accuracy': test_report['accuracy'],
                'train_f1': train_report['macro avg']['f1-score'],
                'test_f1': test_report['macro avg']['f1-score']
            }

        except Exception as e:
            print(f"⚠️  Error creating confusion matrix artifacts: {e}")
            return {}

    def _create_shap_artifacts(self, X_train, model, scaler, feature_cols):
        """
        Create and save SHAP analysis artifacts.
        """
        try:
            import shap

            print(f"🔍 Creating SHAP analysis artifacts...")

            # Scale training data
            X_train_scaled = scaler.transform(X_train)

            # Create SHAP explainer
            explainer = shap.LinearExplainer(model, X_train_scaled)
            shap_values = explainer.shap_values(X_train_scaled)

            # SHAP summary plot (bar chart)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_train_scaled, feature_names=feature_cols,
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('shap_summary_bar.png', dpi=300, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact('shap_summary_bar.png')

            # SHAP summary plot (beeswarm)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_train_scaled, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig('shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact('shap_summary_beeswarm.png')

            # SHAP waterfall plot for a sample prediction
            if len(X_train_scaled) > 0:
                sample_idx = 0
                plt.figure(figsize=(10, 6))
                # Create explanation object for waterfall plot
                explanation = shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=explainer.expected_value,
                    data=X_train_scaled[sample_idx],
                    feature_names=feature_cols
                )
                shap.waterfall_plot(explanation, show=False)
                plt.tight_layout()
                plt.savefig('shap_waterfall_sample.png', dpi=300, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact('shap_waterfall_sample.png')

            # Save SHAP values as CSV
            shap_df = pd.DataFrame(shap_values, columns=feature_cols)
            shap_df.to_csv('shap_values.csv', index=False)
            mlflow.log_artifact('shap_values.csv')

            # Feature importance summary
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'mean_abs_shap': np.abs(shap_values).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            feature_importance.to_csv('feature_importance_shap.csv', index=False)
            mlflow.log_artifact('feature_importance_shap.csv')

            print(f"   ✅ Saved SHAP analysis artifacts")

        except Exception as e:
            print(f"⚠️  Error creating SHAP artifacts: {e}")


    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str,
                 n_trials: int = 1000, **kwargs) -> Dict:
        """
        Train logistic regression meta-learner on topological features + primary signals.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            import optuna
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import balanced_accuracy_score
        except ImportError as e:
            print(f"⚠️  Missing required packages: {e}")
            print("   Install with: pip install scikit-learn optuna")
            return {
                'best_params': {},
                'best_value': 0.0,
                'study': None
            }

        print(f"🧠 Training logistic regression meta-learner...")

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

        # Prepare features - use ALL available topological features and let regularization handle selection
        feature_cols = []

        # Add ALL topological features (let the model select via regularization)
        available_topo_cols = [col for col in labeled_data.columns if '_topo_' in col]
        feature_cols.extend(available_topo_cols)
        print(f"   Using ALL {len(available_topo_cols)} topological features")

        # Add primary signal as a feature
        if 'signal' in labeled_data.columns:
            feature_cols.append('signal')
            print(f"   ✓ Added primary signal as feature")

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

        topo_cols = [col for col in feature_cols if '_topo_' in col]
        print(f"   Features: {len(feature_cols)} ({len(topo_cols)} topological + signal)")
        print(f"   Using regularization to automatically select most important features")

        # Prepare training data
        # CRITICAL FIX: Don't fillna(0) on signal column - it converts actual signals to 0!
        X_full = labeled_data[feature_cols].copy()

        # Only fill NaN for topological features, NOT for signal column
        for col in feature_cols:
            if col != 'signal' and X_full[col].isna().any():
                X_full[col] = X_full[col].fillna(0)

        # Verify signal column integrity
        if 'signal' in X_full.columns:
            # Fix -0.0 vs 0.0 issue by rounding to avoid floating point precision issues
            X_full['signal'] = X_full['signal'].round(decimals=8)
            # Convert -0.0 to 0.0
            X_full.loc[X_full['signal'] == -0.0, 'signal'] = 0.0

            signal_dist = X_full['signal'].value_counts().to_dict()
            print(f"   Primary signal distribution in training: {signal_dist}")
            if all(val == 0 for val in X_full['signal'].dropna().unique()):
                print("   ⚠️ WARNING: All signals are 0 - this will result in no trades!")

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

        # Note: Standard meta-model artifacts are now handled by generate_run.py
        print(f"   📝 Standard artifacts will be handled by generate_run.py")

        # Check if we have enough samples for cross-validation
        if len(X_train) < 30:
            print(f"⚠️  Limited training data ({len(X_train)} samples) - using simple training")
            # Simple training without hyperparameter optimization
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            self.model = LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_train_scaled, y_train)

            # Calculate training accuracy
            train_pred = self.model.predict(X_train_scaled)
            train_score = balanced_accuracy_score(y_train, train_pred)

            # Create enhanced artifacts even for simple training
            if len(X_test) > 0:
                # Confusion matrix artifacts
                metrics = self._create_confusion_matrix_artifacts(X_train, y_train, X_test, y_test, self.model, self.scaler)

                # SHAP analysis artifacts
                self._create_shap_artifacts(X_train, self.model, self.scaler, feature_cols)

                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

            return {
                'best_params': {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'lbfgs'
                },
                'best_value': train_score,
                'study': None
            }

        # Hyperparameter optimization with Optuna
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.1, 100, log=True),  # Increased lower bound to reduce over-regularization
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', 'none']),
                'solver': 'saga',  # Use saga to support all penalties
                'max_iter': 2000,
                'random_state': 42,
                'class_weight': 'balanced'
            }

            # Set l1_ratio for elasticnet
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)

            # Adjust solver for specific penalties
            if params['penalty'] == 'none':
                params['solver'] = 'lbfgs'
            elif params['penalty'] == 'l2':
                params['solver'] = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=min(3, len(X_train) // 10))
            scores = []

            # Fit scaler on full training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            for train_idx, val_idx in tscv.split(X_train_scaled):
                X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                try:
                    model = LogisticRegression(**params)
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
        study.optimize(objective, n_trials=min(n_trials, 1000))  # Increased trials for better optimization

        best_params = study.best_params
        best_value = study.best_value

        print(f"   Best CV score: {best_value:.4f}")
        print(f"   Best params: {best_params}")

        # Train final model with best parameters
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fix solver for final model based on penalty
        final_params = best_params.copy()
        if final_params.get('penalty') == 'l1':
            final_params['solver'] = 'liblinear'
        elif final_params.get('penalty') == 'elasticnet':
            final_params['solver'] = 'saga'
        elif final_params.get('penalty') == 'none':
            final_params['solver'] = 'lbfgs'
        elif final_params.get('penalty') == 'l2' and 'solver' not in final_params:
            final_params['solver'] = 'lbfgs'

        self.model = LogisticRegression(**final_params)
        self.model.fit(X_train_scaled, y_train)

        # Calculate feature importance (coefficients)
        if hasattr(self.model, 'coef_'):
            feature_importance = dict(zip(feature_cols, np.abs(self.model.coef_[0])))
            # Sort by importance
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"   Top features: {[f'{name}: {imp:.3f}' for name, imp in top_features]}")

        # Create enhanced artifacts
        if len(X_test) > 0:
            # Confusion matrix artifacts
            metrics = self._create_confusion_matrix_artifacts(X_train, y_train, X_test, y_test, self.model, self.scaler)

            # SHAP analysis artifacts
            self._create_shap_artifacts(X_train, self.model, self.scaler, feature_cols)

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        # Store enhanced data for artifact generation
        self.enhanced_data = enhanced_data
        print(f"   ✅ Stored enhanced data for artifacts: {enhanced_data.shape}")

        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }