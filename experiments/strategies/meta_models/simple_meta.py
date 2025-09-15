import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.meta_models.base_meta_strategy import MetaStrategy
from typing import Dict, Any
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class SimpleMetaStrategy(MetaStrategy):
    """Simple meta-model that refines primary model signals using price momentum and volume."""
    
    def __init__(self, asset: str = 'bitcoin', primary_run_id: str = None):
        name = f"simple_meta_{asset}"
        super().__init__(name, asset, primary_run_id)
        self.model = None
        self.feature_cols = []
        
    # strategy_type is inherited from MetaStrategy base class (returns "meta_model")
    
    def get_required_features(self) -> Dict[str, Any]:
        """Simple meta-model only needs crypto asset price and volume data."""
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': None,  # No external indicators needed
            'yahoo_tickers': None,
            'calculated_features': None
        }
    
    # Specify which crypto features we need
    #used_crypto_features = ['close', 'volume']
    
    def _create_meta_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create simple features for meta-model training."""
        df = data.copy()
        
        # Primary model features - fail fast if missing
        if 'signal' not in df.columns:
            raise ValueError("Required 'signal' column missing from input data. Meta-model requires primary model signals.")
        
        df['primary_signal'] = df['signal'].fillna(0).astype(float)
        
        # Validation: ensure we have meaningful signals
        non_zero_signals = (df['signal'] != 0).sum()
        if non_zero_signals == 0:
            raise ValueError("No non-zero primary signals found in data. Meta-model requires actual primary model decisions.")
        
        # Asset price features - fail fast if missing required data
        close_col = f"{self.asset}_close"
        if close_col not in df.columns:
            raise ValueError(f"Required price column '{close_col}' missing from input data. Check asset name and data loading.")
        
        if df[close_col].isna().all():
            raise ValueError(f"Price column '{close_col}' contains only NaN values. Check data quality.")
        
        # Recent price momentum
        df['momentum_3'] = df[close_col].pct_change(3)
        df['momentum_5'] = df[close_col].pct_change(5)
        df['momentum_10'] = df[close_col].pct_change(10)
        
        # Simple moving averages
        df['sma_5'] = df[close_col].rolling(5).mean()
        df['sma_10'] = df[close_col].rolling(10).mean()
        
        # Price position relative to moving averages
        df['price_vs_sma5'] = (df[close_col] / df['sma_5']) - 1
        df['price_vs_sma10'] = (df[close_col] / df['sma_10']) - 1
        
        # Price volatility
        df['volatility_5'] = df[close_col].pct_change().rolling(5).std()
        
        # Volume features - fail fast if missing
        volume_col = f"{self.asset}_total_volume"
        if volume_col not in df.columns:
            raise ValueError(f"Required volume column '{volume_col}' missing from input data. Check asset name and data loading.")
        
        if df[volume_col].isna().all():
            raise ValueError(f"Volume column '{volume_col}' contains only NaN values. Check data quality.")
        
        df['volume_sma_5'] = df[volume_col].rolling(5).mean()
        df['volume_ratio'] = df[volume_col] / df['volume_sma_5']
        
        # Interaction features: signal performance in different momentum regimes
        df['signal_x_momentum3'] = df['primary_signal'] * df['momentum_3'].fillna(0)
        df['signal_x_momentum5'] = df['primary_signal'] * df['momentum_5'].fillna(0)
        
        return df
    
    def _add_meta_features_inplace(self, df: pd.DataFrame) -> None:
        """Add meta features directly to the DataFrame in-place to preserve index."""
        # Primary model features - fail fast if missing
        if 'signal' not in df.columns:
            raise ValueError("Required 'signal' column missing from input data. Meta-model requires primary model signals.")
        
        df['primary_signal'] = df['signal'].fillna(0).astype(float)
        
        # Validation: ensure we have meaningful signals
        non_zero_signals = (df['signal'] != 0).sum()
        if non_zero_signals == 0:
            raise ValueError("No non-zero primary signals found in data. Meta-model requires actual primary model decisions.")
        
        # Asset price features - fail fast if missing required data
        close_col = f"{self.asset}_close"
        if close_col not in df.columns:
            raise ValueError(f"Required price column '{close_col}' missing from input data. Check asset name and data loading.")
        
        if df[close_col].isna().all():
            raise ValueError(f"Price column '{close_col}' contains only NaN values. Check data quality.")
        
        # Recent price momentum
        df['momentum_3'] = df[close_col].pct_change(3)
        df['momentum_5'] = df[close_col].pct_change(5)
        df['momentum_10'] = df[close_col].pct_change(10)
        
        # Simple moving averages
        df['sma_5'] = df[close_col].rolling(5).mean()
        df['sma_10'] = df[close_col].rolling(10).mean()
        
        # Price position relative to moving averages
        df['price_vs_sma5'] = (df[close_col] / df['sma_5']) - 1
        df['price_vs_sma10'] = (df[close_col] / df['sma_10']) - 1
        
        # Price volatility
        df['volatility_5'] = df[close_col].pct_change().rolling(5).std()
        
        # Volume features - fail fast if missing
        volume_col = f"{self.asset}_total_volume"
        if volume_col not in df.columns:
            raise ValueError(f"Required volume column '{volume_col}' missing from input data. Check asset name and data loading.")
        
        if df[volume_col].isna().all():
            raise ValueError(f"Volume column '{volume_col}' contains only NaN values. Check data quality.")
        
        df['volume_sma_5'] = df[volume_col].rolling(5).mean()
        df['volume_ratio'] = df[volume_col] / df['volume_sma_5']
        
        # Interaction features: signal performance in different momentum regimes
        df['signal_x_momentum3'] = df['primary_signal'] * df['momentum_3'].fillna(0)
        df['signal_x_momentum5'] = df['primary_signal'] * df['momentum_5'].fillna(0)
    
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate meta-model signals using trained model."""
        if self.model is None:
            raise ValueError("Meta-model not trained yet! Call optimize() first.")
        
        # Work directly with the input data to preserve index
        # This is the key fix - don't create new DataFrames that lose the index
        df = data.copy()
        
        # Add meta features directly to the original data structure
        # This preserves the timestamp index throughout
        self._add_meta_features_inplace(df)
        
        # Prepare features for prediction - shift to prevent look-ahead bias
        X = df[self.feature_cols].shift(1).fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Get primary signal (handle NaN values properly)
        primary_signal = df.get('primary_signal', 0).fillna(0).astype(int)
        
        # Meta-model directly predicts {0, 1} as a filter
        meta_filter = predictions  # Binary predictions from RandomForest
        
        # Combine: primary_signal × meta_filter
        # Results in {-1, 0, +1} where 0 means filtered out
        df['signal'] = primary_signal * meta_filter
        
        return df
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 500, **kwargs) -> Dict:
        """Optimize meta-model parameters using Optuna."""
        
        # Create meta features using in-place method to preserve index
        feature_data = data.copy()
        self._add_meta_features_inplace(feature_data)
        
        # Filter training period (preserve warmup data by not using dropna)
        train_data = feature_data.loc[train_start:train_end]
        
        if len(train_data) < 20:
            raise ValueError(f"Not enough training data for meta-model: {len(train_data)} samples")
        
        # Define potential features (exclude NaN-prone ones)
        potential_features = [
            'primary_signal', 'momentum_3', 'momentum_5', 'momentum_10',
            'price_vs_sma5', 'price_vs_sma10', 'volatility_5', 'volume_ratio',
            'signal_x_momentum3', 'signal_x_momentum5'
        ]
        
        # Only use features that exist and have sufficient non-null values
        available_features = []
        for f in potential_features:
            if f in train_data.columns:
                non_null_ratio = train_data[f].notna().mean()
                if non_null_ratio > 0.8:  # At least 80% non-null values
                    available_features.append(f)
        
        self.feature_cols = available_features
        
        if len(self.feature_cols) < 2:
            raise ValueError(f"Not enough valid features for meta-model: {self.feature_cols}")
        
        # Prepare training data - IMPORTANT: We need to shift labels to avoid look-ahead bias
        # The label at time t tells us the outcome of a trade entered at time t
        # So we need to align features at time t-1 with label at time t
        X_train = train_data[self.feature_cols].shift(1).fillna(0)
        
        # Filter for valid labels BEFORE converting to int (avoids NaN conversion error)
        valid_label_mask = train_data['label'].notna()
        X_train = X_train[valid_label_mask]
        # No shift needed for labels - they already represent future outcomes
        y_train = train_data.loc[valid_label_mask, 'label'].astype(int)
        
        # Remove only the first row due to shifting (but keep warmup data)
        if len(X_train) > 0:
            X_train = X_train.iloc[1:]
            y_train = y_train.iloc[1:]
        
        if len(X_train) < 10:
            raise ValueError(f"Not enough clean training samples: {len(X_train)}")
        
        # Optimization with Optuna
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 50)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_train)
                score = accuracy_score(y_train, y_pred)
                return score
            except Exception:
                return 0.0
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"Best meta-model parameters: {best_params}")
        print(f"Best meta-model score: {best_score:.4f}")
        
        # Train final model
        self.model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Final evaluation
        y_pred = self.model.predict(X_train)
        final_accuracy = accuracy_score(y_train, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        
        print(f"Final meta-model accuracy: {final_accuracy:.4f}")
        print(f"Top features: {sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        return {
            'best_params': best_params,
            'best_value': final_accuracy,
            'study': study,
            'model': self.model,
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'final_accuracy': final_accuracy
        }
    
    def get_input_example(self) -> pd.DataFrame:
        """Generate input example with meta-features for MLflow signature."""
        # Create a sample with meta-features that the model actually uses
        sample_data = {
            'primary_signal': [1.0],
            'momentum_3': [0.01],
            'momentum_5': [0.02],
            'momentum_10': [0.03],
            'price_vs_sma5': [0.01],
            'price_vs_sma10': [0.02],
            'volatility_5': [0.015],
            'volume_ratio': [1.1],
            'signal_x_momentum3': [0.01],
            'signal_x_momentum5': [0.02]
        }
        
        # Only include features that are actually in self.feature_cols
        if hasattr(self, 'feature_cols') and self.feature_cols:
            sample_data = {k: v for k, v in sample_data.items() if k in self.feature_cols}
        
        return pd.DataFrame(sample_data)