import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.meta_models.base_meta_strategy import MetaStrategy
from typing import Dict, Any
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


class EnsembleMetaStrategy(MetaStrategy):
    """Meta-model that refines primary model signals using labels + macro context."""
    
    def __init__(self, asset: str = 'bitcoin', primary_run_id: str = None):
        name = f"ensemble_meta_{asset}"
        super().__init__(name, asset, primary_run_id)
        self.model = None
        self.feature_cols = []  # Will be populated during optimization
        
    @property
    def strategy_type(self) -> str:
        """Meta-models can be trend-following (following primary model's nature)"""
        return "trend_following"
    
    def get_required_features(self) -> Dict[str, Any]:
        """Meta-model uses crypto + macro features to augment primary model signals."""
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': {'VIXCLS': 'vix'},  # Add VIX for market volatility context
            'yahoo_tickers': None,
            'calculated_features': None
        }
    
    # Specify which crypto features we need
    used_crypto_features = ['close', 'volume']
    
    def _create_meta_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for meta-model training."""
        df = data.copy()
        
        # Primary model features
        if 'signal' in df.columns:
            df['primary_signal'] = df['signal'].astype(float)
            df['signal_strength'] = df['signal'].abs()  # How confident was primary model
        
        # Asset price features
        close_col = f"{self.asset}_close"
        if close_col in df.columns:
            df['price_momentum_5'] = df[close_col].pct_change(5)
            df['price_momentum_10'] = df[close_col].pct_change(10)
            
            # Price vs moving averages
            df['sma_10'] = df[close_col].rolling(10).mean()
            df['sma_20'] = df[close_col].rolling(20).mean()
            df['price_vs_sma10'] = (df[close_col] / df['sma_10']) - 1
            df['price_vs_sma20'] = (df[close_col] / df['sma_20']) - 1
        
        # Volume features
        volume_col = f"{self.asset}_total_volume"
        if volume_col in df.columns:
            df['volume_sma_10'] = df[volume_col].rolling(10).mean()
            df['volume_ratio'] = df[volume_col] / df['volume_sma_10']
        else:
            df['volume_ratio'] = 1.0
        
        # Macro context (VIX)
        if 'vix' in df.columns:
            df['vix_level'] = df['vix']
            df['vix_change'] = df['vix'].pct_change(5)  # 5-day VIX change
            df['high_vol_regime'] = (df['vix'] > 25).astype(float)  # High volatility periods
        else:
            df['vix_level'] = 20.0  # Default VIX level
            df['vix_change'] = 0.0
            df['high_vol_regime'] = 0.0
        
        # Interaction features: How primary signal performs in different regimes
        if 'signal' in df.columns:
            df['signal_x_momentum'] = df['primary_signal'] * df.get('price_momentum_5', 0)
            df['signal_x_vol_regime'] = df['primary_signal'] * df['high_vol_regime']
        
        return df
    
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate meta-model signals using trained model."""
        if self.model is None:
            raise ValueError("Meta-model not trained yet! Call optimize() first.")
        
        # Create meta features
        feature_data = self._create_meta_features(data)
        
        # Prepare features for prediction - shift to prevent look-ahead bias
        X = feature_data[self.feature_cols].shift(1).fillna(method='ffill').fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Create signals DataFrame
        df = feature_data.copy()
        
        # Use probability-based thresholds for more nuanced signals
        confidence_threshold = params.get('confidence_threshold', 0.6)
        
        # Default to primary signal, but override if meta-model is confident
        df['signal'] = df.get('primary_signal', 0).astype(int)
        
        # Get probabilities for each class (-1, 0, 1)
        prob_short = probabilities[:, 0] if probabilities.shape[1] > 0 else np.zeros(len(df))
        prob_hold = probabilities[:, 1] if probabilities.shape[1] > 1 else np.zeros(len(df))
        prob_long = probabilities[:, 2] if probabilities.shape[1] > 2 else np.zeros(len(df))
        
        # Override primary signal where meta-model is confident
        high_confidence_long = prob_long > confidence_threshold
        high_confidence_short = prob_short > confidence_threshold
        high_confidence_hold = prob_hold > confidence_threshold
        
        df.loc[high_confidence_long, 'signal'] = 1
        df.loc[high_confidence_short, 'signal'] = -1
        df.loc[high_confidence_hold, 'signal'] = 0
        
        # Store confidence for analysis
        df['meta_confidence'] = np.max(probabilities, axis=1)
        df['refined_by_meta'] = (
            high_confidence_long | high_confidence_short | high_confidence_hold
        ).astype(int)
        
        return df
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
        """Optimize meta-model parameters using Optuna."""
        
        # Create meta features
        feature_data = self._create_meta_features(data)
        
        # Filter training period
        train_data = feature_data.loc[train_start:train_end].dropna()
        
        if len(train_data) < 50:
            raise ValueError(f"Not enough training data for meta-model: {len(train_data)} samples")
        
        # Define potential features
        potential_features = [
            'primary_signal', 'signal_strength', 'price_momentum_5', 'price_momentum_10',
            'price_vs_sma10', 'price_vs_sma20', 'volume_ratio', 'vix_level', 'vix_change', 
            'high_vol_regime', 'signal_x_momentum', 'signal_x_vol_regime'
        ]
        
        # Only use features that exist and have non-null values
        available_features = [f for f in potential_features if f in train_data.columns]
        self.feature_cols = available_features
        
        print(f"Meta-model features: {self.feature_cols}")
        
        # Prepare training data with look-ahead bias prevention
        X_train = train_data[self.feature_cols].shift(1).fillna(method='ffill').fillna(0)
        y_train = train_data['label'].astype(int)  # Use ground truth labels
        
        # Remove first row due to shifting and any remaining NaN labels
        mask = ~y_train.isna()
        X_train = X_train[mask].iloc[1:]
        y_train = y_train[mask].iloc[1:]
        
        if len(X_train) < 30:
            raise ValueError(f"Not enough clean training samples: {len(X_train)}")
        
        print(f"Training samples: {len(X_train)}")
        print(f"Label distribution: {y_train.value_counts().to_dict()}")
        
        # Optuna optimization
        def objective(trial):
            # Sample hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_train)
                
                # Use accuracy as optimization metric
                score = accuracy_score(y_train, y_pred)
                return score
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=min(n_trials, 100))  # Limit trials for meta-model
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"Best meta-model parameters: {best_params}")
        print(f"Best meta-model score: {best_score:.4f}")
        
        # Train final model with best parameters
        self.model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate final model
        y_pred = self.model.predict(X_train)
        final_accuracy = accuracy_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_train, y_pred, average='macro', zero_division=0)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        
        print(f"Final meta-model performance:")
        print(f"  Accuracy: {final_accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        # Add confidence threshold parameter
        best_params['confidence_threshold'] = 0.6  # Default threshold
        
        return {
            'best_params': best_params,
            'best_value': final_accuracy,
            'study': study,
            'model': self.model,
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'final_accuracy': final_accuracy,
            'precision': precision,
            'recall': recall
        }