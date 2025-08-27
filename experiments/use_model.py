#!/usr/bin/env python
"""
Unified model consumption script for all registered strategies.
Works with rules-based, ML-based, and hybrid models.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlflow
import mlflow.pyfunc
import pandas as pd
from config import setup_mlflow


class StrategyPredictor:
    """Unified interface for loading and using any registered strategy model."""
    
    def __init__(self, model_name: str, version: str = "latest"):
        """
        Initialize predictor with a registered model.
        
        Args:
            model_name: Name of the registered model
            version: Version to load ("latest", "1", "2", etc.)
        """
        self.model_name = model_name
        self.version = version
        self.model = None
        self.model_info = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and its metadata."""
        setup_mlflow()
        
        model_uri = f"models:/{self.model_name}/{self.version}"
        print(f"🔄 Loading model: {model_uri}")
        
        try:
            # Load model
            self.model = mlflow.pyfunc.load_model(model_uri)
            
            # Get model version info for metadata
            client = mlflow.MlflowClient()
            if self.version == "latest":
                versions = client.search_model_versions(f"name='{self.model_name}'")
                latest_version = max(versions, key=lambda x: int(x.version))
                self.model_info = latest_version
            else:
                self.model_info = client.get_model_version(self.model_name, self.version)
            
            print(f"✅ Model loaded successfully!")
            print(f"   Model: {self.model_name}")
            print(f"   Version: {self.model_info.version}")
            print(f"   Tags: {self.model_info.tags}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions from input data.
        
        Args:
            data: DataFrame with required features for the strategy
            
        Returns:
            DataFrame with signals and predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print(f"📊 Making predictions on {len(data)} samples...")
        
        # Prepare data based on model signature
        prepared_data = self._prepare_data_for_model(data)
        
        try:
            predictions = self.model.predict(prepared_data)
            
            # Convert to DataFrame if numpy array
            if hasattr(predictions, 'shape') and len(predictions.shape) == 1:
                predictions = pd.DataFrame({'signal': predictions}, index=data.index)
            elif isinstance(predictions, pd.DataFrame):
                predictions.index = data.index
            
            print(f"✅ Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise
    
    def _prepare_data_for_model(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data based on model's expected signature."""
        
        # Get model signature from metadata
        signature = None
        if hasattr(self.model, 'metadata') and hasattr(self.model.metadata, 'signature'):
            signature = self.model.metadata.signature
        
        if signature is None:
            print("⚠️  No model signature found, using data as-is")
            return data
        
        # Extract column names from schema  
        expected_columns = [col.name for col in signature.inputs.inputs]
        print(f"📋 Model expects columns: {expected_columns}")
        print(f"📋 Data has columns: {list(data.columns)}")
        
        # Check if this looks like an ML model (needs engineered features)
        ml_features = ['returns_1d', 'returns_7d', 'price_vs_sma7', 'price_vs_sma21', 'volume_ratio', 'volatility_7d']
        needs_feature_engineering = any(feat in expected_columns for feat in ml_features)
        
        if needs_feature_engineering:
            print("🔧 Creating ML features from raw data...")
            return self._create_ml_features(data)
        else:
            print("📊 Using raw OHLC data for rules-based model")
            # Filter to only expected columns
            available_columns = [col for col in expected_columns if col in data.columns]
            return data[available_columns]
    
    def _create_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from raw OHLC data."""
        df = data.copy()
        
        # Extract asset name from column names  
        asset = None
        for col in df.columns:
            if '_close' in col:
                asset = col.replace('_close', '')
                break
        
        if asset is None:
            raise ValueError("Cannot determine asset from column names")
        
        close_col = f"{asset}_close"
        volume_col = f"{asset}_total_volume"
        
        # Basic price features
        df['returns_1d'] = df[close_col].pct_change(1).fillna(0)
        df['returns_7d'] = df[close_col].pct_change(7).fillna(0)
        
        # Moving averages  
        df['sma_7'] = df[close_col].rolling(7, min_periods=1).mean()
        df['sma_21'] = df[close_col].rolling(21, min_periods=1).mean()
        
        # Price ratios
        df['price_vs_sma7'] = (df[close_col] / df['sma_7'] - 1).fillna(0)
        df['price_vs_sma21'] = (df[close_col] / df['sma_21'] - 1).fillna(0)
        
        # Volume features
        if volume_col in df.columns:
            df['volume_sma_7'] = df[volume_col].rolling(7, min_periods=1).mean()
            df['volume_ratio'] = (df[volume_col] / df['volume_sma_7']).fillna(1.0)
        else:
            df['volume_ratio'] = 1.0
        
        # Volatility
        df['volatility_7d'] = df['returns_1d'].rolling(7, min_periods=1).std().fillna(0)
        
        # Return only the ML features
        ml_features = ['returns_1d', 'returns_7d', 'price_vs_sma7', 'price_vs_sma21', 'volume_ratio', 'volatility_7d']
        return df[ml_features]
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        if self.model_info is None:
            return {}
        
        return {
            'name': self.model_name,
            'version': self.model_info.version,
            'tags': self.model_info.tags,
            'creation_timestamp': self.model_info.creation_timestamp,
            'run_id': self.model_info.run_id
        }


def create_sample_data(asset: str = 'bitcoin', samples: int = 10) -> pd.DataFrame:
    """Create sample data for testing predictions."""
    import numpy as np
    np.random.seed(42)
    
    # Create realistic sample data (using integers to match model signature)
    base_price = 50000
    prices_high = (base_price + np.random.uniform(0, 2000, samples)).astype(int)
    prices_low = (base_price - np.random.uniform(0, 2000, samples)).astype(int)
    prices_close = (base_price + np.random.uniform(-1000, 1000, samples)).astype(int)
    
    data = {
        f'{asset}_high': prices_high,
        f'{asset}_low': prices_low,
        f'{asset}_close': prices_close,
        f'{asset}_total_volume': np.random.uniform(800, 1500, samples).astype(int)
    }
    
    dates = pd.date_range('2024-01-01', periods=samples, freq='D')
    return pd.DataFrame(data, index=dates)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Use registered MLflow models for predictions')
    parser.add_argument('--model-name', required=True, help='Name of registered model')
    parser.add_argument('--version', default='latest', help='Model version (default: latest)')
    parser.add_argument('--data-file', help='Path to input data CSV file')
    parser.add_argument('--sample-data', action='store_true', help='Use sample data for testing')
    parser.add_argument('--asset', default='bitcoin', help='Asset for sample data')
    parser.add_argument('--samples', type=int, default=10, help='Number of sample data points')
    
    args = parser.parse_args()
    
    print(f"🎯 Using model: {args.model_name} (version: {args.version})")
    
    # Initialize predictor
    predictor = StrategyPredictor(args.model_name, args.version)
    
    # Load data
    if args.data_file:
        print(f"📂 Loading data from: {args.data_file}")
        data = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
    elif args.sample_data:
        print(f"🧪 Creating sample data for {args.asset}...")
        data = create_sample_data(args.asset, args.samples)
    else:
        print("❌ Please provide --data-file or use --sample-data")
        sys.exit(1)
    
    print(f"📊 Input data shape: {data.shape}")
    print(f"📊 Input columns: {list(data.columns)}")
    
    # Make predictions
    predictions = predictor.predict(data)
    
    print(f"\n📈 PREDICTIONS")
    print("=" * 50)
    print(predictions.tail(min(10, len(predictions))))
    
    # Show model info
    info = predictor.get_model_info()
    print(f"\n📋 MODEL INFO")
    print("=" * 50)
    for key, value in info.items():
        if key != 'tags':
            print(f"{key}: {value}")
    
    print(f"\n🎉 Prediction complete!")


if __name__ == "__main__":
    main()

