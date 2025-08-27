import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.base_strategy import BaseStrategy
from typing import Dict, Any
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


class SimpleCatBoostStrategy(BaseStrategy):
    
    def __init__(self, asset: str = 'bitcoin'):
        super().__init__(f"simple_catboost_{asset}")
        self.asset = asset
        self.model = None
        self.input_example = None  # Store training sample for MLflow signature
        self.feature_cols = [
            'returns_1d', 'returns_7d', 'price_vs_sma7', 'price_vs_sma21',
            'volume_ratio', 'volatility_7d'
        ]
        self.default_params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 4,
            'random_seed': 42
        }
    
    @property
    def strategy_type(self) -> str:
        """ML-based strategies can be trend-following"""
        return "trend_following"
    
    @property 
    def implementation_type(self) -> str:
        """This is an ML-based strategy"""
        return "ml_based"
    
    def get_required_features(self) -> Dict[str, Any]:
        """Simple ML strategy only needs crypto asset price data."""
        return {
            'crypto_assets': [self.asset],
            'fred_indicators': None,
            'yahoo_tickers': None,
            'calculated_features': None
        }
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model"""
        df = data.copy()
        
        # Find close price column
        close_col = f"{self.asset}_close"
        volume_col = f"{self.asset}_total_volume"
        
        if close_col not in df.columns:
            raise ValueError(f"Required column '{close_col}' not found in data")
        
        # Basic price features
        df['returns_1d'] = df[close_col].pct_change(1)
        df['returns_7d'] = df[close_col].pct_change(7)
        
        # Moving averages
        df['sma_7'] = df[close_col].rolling(7).mean()
        df['sma_21'] = df[close_col].rolling(21).mean()
        
        # Price ratios
        df['price_vs_sma7'] = df[close_col] / df['sma_7'] - 1
        df['price_vs_sma21'] = df[close_col] / df['sma_21'] - 1
        
        # Volume features (if available)
        if volume_col in df.columns:
            df['volume_sma_7'] = df[volume_col].rolling(7).mean()
            df['volume_ratio'] = df[volume_col] / df['volume_sma_7']
        else:
            df['volume_ratio'] = 1.0  # Default value
        
        # Volatility
        df['volatility_7d'] = df['returns_1d'].rolling(7).std()
        
        return df
    
    def _create_target(self, data: pd.DataFrame, horizon_days: int = 3) -> pd.DataFrame:
        """Create simple binary target: 1 if price goes up >1% in next N days"""
        df = data.copy()
        close_col = f"{self.asset}_close"
        
        # Look forward N days
        df['future_price'] = df[close_col].shift(-horizon_days)
        df['future_return'] = (df['future_price'] / df[close_col]) - 1
        
        # Binary target: 1 if return > 1%, 0 otherwise
        df['target'] = (df['future_return'] > 0.01).astype(int)
        
        # Remove last N rows (no future data)
        df = df[:-horizon_days]
        
        return df
    
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate ML-based signals"""
        if self.model is None:
            raise ValueError("Model not trained yet! Call optimize() first.")
        
        # Create features
        feature_data = self._create_features(data)
        
        # Prepare features for prediction - shift to prevent look-ahead bias
        X = feature_data[self.feature_cols].shift(1).fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Create signals DataFrame
        df = feature_data.copy()
        df['signal'] = np.where(predictions == 1, 1, -1)  # 1 for buy, -1 for sell
        df['confidence'] = probabilities
        
        return df
    
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
        """Train the ML model (optimization for ML means training)"""
        
        # Create features and target
        feature_data = self._create_features(data)
        target_data = self._create_target(feature_data)
        
        # Filter training period and ensure proper time shifting
        train_data = target_data.loc[train_start:train_end].dropna()
        
        if len(train_data) < 100:
            raise ValueError(f"Not enough training data: {len(train_data)} samples")
        
        # Prepare features - shift them to prevent look-ahead bias
        X_train = train_data[self.feature_cols].shift(1).fillna(0)  # Shift features by 1 period
        y_train = train_data['target']
        
        # Remove first row due to shifting
        X_train = X_train.iloc[1:]
        y_train = y_train.iloc[1:]
        
        # Train model
        self.model = CatBoostClassifier(
            iterations=self.default_params['iterations'],
            learning_rate=self.default_params['learning_rate'],
            depth=self.default_params['depth'],
            random_seed=self.default_params['random_seed'],
            verbose=False,
            allow_writing_files=False  # Prevent catboost_info directory creation
        )
        
        self.model.fit(X_train, y_train)
        
        # Store a sample for MLflow signature (take first 5 rows)
        self.input_example = X_train.head(5).copy()
        
        # Evaluate on training data
        y_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        
        # Calculate feature importance
        feature_importance = dict(zip(self.feature_cols, self.model.get_feature_importance()))
        
        print(f"Model trained successfully!")
        print(f"Training samples: {len(X_train)}")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Target distribution: {y_train.value_counts(normalize=True).to_dict()}")
        
        return {
            'best_params': self.default_params,
            'best_value': train_accuracy,
            'study': None,
            'model': self.model,
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'training_accuracy': train_accuracy
        }
    
    def fit(self, data: pd.DataFrame, labels: pd.Series, **params) -> 'SimpleCatBoostStrategy':
        """Fit method for ML pipeline compatibility"""
        
        # Create features
        feature_data = self._create_features(data)
        
        # Prepare features - shift them to prevent look-ahead bias
        X = feature_data[self.feature_cols].shift(1).fillna(0)
        
        # Align features and labels
        min_len = min(len(X), len(labels))
        X = X.iloc[1:min_len]  # Skip first row due to shifting
        y = labels.iloc[1:min_len]
        
        # Use provided params or defaults
        model_params = {**self.default_params, **params}
        
        # Train model
        self.model = CatBoostClassifier(
            iterations=model_params.get('iterations', 500),
            learning_rate=model_params.get('learning_rate', 0.1),
            depth=model_params.get('depth', 4),
            random_seed=model_params.get('random_seed', 42),
            verbose=False,
            allow_writing_files=False  # Prevent catboost_info directory creation
        )
        
        self.model.fit(X, y)
        
        # Store a sample for MLflow signature (take first 5 rows)
        self.input_example = X.head(5).copy()
        
        return self
    
    def get_parameter_ranges(self) -> Dict:
        """Return parameter ranges for this strategy"""
        return {
            'iterations': [100, 1000],
            'learning_rate': [0.01, 0.3],
            'depth': [3, 8]
        }
    
    def get_input_example(self) -> pd.DataFrame:
        """Return a sample input for MLflow signature inference"""
        if self.input_example is not None:
            return self.input_example
        
        # Create a minimal example if none stored
        sample_data = {
            'returns_1d': [0.01, -0.02, 0.005],
            'returns_7d': [0.05, -0.03, 0.02], 
            'price_vs_sma7': [0.02, -0.01, 0.03],
            'price_vs_sma21': [0.01, -0.02, 0.015],
            'volume_ratio': [1.2, 0.8, 1.5],
            'volatility_7d': [0.02, 0.03, 0.025]
        }
        return pd.DataFrame(sample_data)