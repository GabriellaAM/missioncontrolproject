# %%
"""
Model Usage Script

This script helps you load registered models and use them for predictions/diagnosis
on historical data, particularly yesterday's market data.

Usage:
1. Load a registered model by asset-strategy combination
2. Get yesterday's (or any historical) data
3. Run diagnosis/predictions
4. Analyze signals
"""

import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import setup_mlflow
from utils.feature_loader import FeatureLoader

# %%
########################################################
# Setup and Configuration
########################################################

# Setup MLflow
setup_mlflow()
client = mlflow.MlflowClient()

print("✅ MLflow setup complete")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# %%
########################################################
# Model Loading Class
########################################################

class ModelUser:
    """Class for loading and using registered MLflow models"""
    
    def __init__(self):
        self.client = mlflow.MlflowClient()
        
    def list_registered_models(self) -> pd.DataFrame:
        """List all registered models with their info"""
        try:
            models = self.client.search_registered_models()
            
            model_data = []
            for model in models:
                # Get latest version
                latest_versions = self.client.get_latest_versions(model.name, stages=["None"])
                if latest_versions:
                    version = latest_versions[0]
                    model_data.append({
                        'name': model.name,
                        'version': version.version,
                        'stage': version.current_stage,
                        'creation_time': version.creation_timestamp,
                        'description': model.description or 'No description'
                    })
            
            return pd.DataFrame(model_data)
            
        except Exception as e:
            print(f"Error listing models: {e}")
            return pd.DataFrame()
    
    def load_model(self, asset: str, strategy: str, version: str = "latest") -> mlflow.pyfunc.PyFuncModel:
        """
        Load a registered model by asset-strategy combination
        
        Args:
            asset: Asset name (e.g., 'bitcoin')
            strategy: Strategy name (e.g., 'sma_crossover')
            version: Model version ('latest', 'staging', 'production', or specific number)
        
        Returns:
            Loaded MLflow model
        """
        model_name = f"{asset}_{strategy}"
        
        try:
            if version in ['latest', 'staging', 'production']:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{version}"
            
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✅ Loaded model: {model_name} (version: {version})")
            
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            return None
    
    def get_model_info(self, asset: str, strategy: str) -> Dict:
        """Get detailed information about a registered model"""
        model_name = f"{asset}_{strategy}"
        
        try:
            # Get model versions
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            if not versions:
                return {"error": f"Model {model_name} not found"}
            
            # Get latest version details
            latest = versions[0]
            
            # Get the original run info
            run = self.client.get_run(latest.run_id)
            
            return {
                'model_name': model_name,
                'latest_version': latest.version,
                'current_stage': latest.current_stage,
                'creation_time': latest.creation_timestamp,
                'run_id': latest.run_id,
                'metrics': dict(run.data.metrics),
                'params': dict(run.data.params),
                'tags': dict(run.data.tags)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_historical_data(self, asset: str, lookback_days: int = 30, end_date: str = None) -> pd.DataFrame:
        """
        Get historical data for an asset
        
        Args:
            asset: Asset name
            lookback_days: How many days back to load
            end_date: End date (default: yesterday)
        
        Returns:
            DataFrame with historical data
        """
        try:
            # Default to yesterday if no end date specified
            if end_date is None:
                end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Calculate start date
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime('%Y-%m-%d')
            
            # Load data using FeatureLoader
            loader = FeatureLoader(start_date=start_date, end_date=end_date)
            features_df = loader.build_feature_set(crypto_assets=[asset])
            
            print(f"✅ Loaded {len(features_df)} days of data for {asset}")
            print(f"   Date range: {start_date} to {end_date}")
            
            return features_df
            
        except Exception as e:
            print(f"❌ Failed to load historical data: {e}")
            return pd.DataFrame()
    
    def get_yesterday_data(self, asset: str) -> pd.DataFrame:
        """Get yesterday's data for diagnosis"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        return self.get_historical_data(asset, lookback_days=100, end_date=yesterday)
    
    def diagnose(self, model: mlflow.pyfunc.PyFuncModel, data: pd.DataFrame, 
                asset: str) -> Dict:
        """
        Run model diagnosis on historical data
        
        Args:
            model: Loaded MLflow model
            data: Historical data DataFrame
            asset: Asset name for column identification
        
        Returns:
            Dictionary with diagnosis results
        """
        try:
            # Get the latest available data point
            if data.empty:
                return {"error": "No data available"}
            
            # For rule-based models, we might need to recreate the strategy signals
            # This is a simplified approach - you may need to adapt based on your model
            
            # Get the last available data point(s)
            recent_data = data.tail(50)  # Get last 50 days for signal calculation
            
            if hasattr(model, 'predict'):
                # Try to get predictions
                try:
                    predictions = model.predict(recent_data)
                    latest_signal = predictions[-1] if len(predictions) > 0 else None
                except Exception as e:
                    return {"error": f"Prediction failed: {e}"}
            else:
                return {"error": "Model does not support predictions"}
            
            # Get latest price data
            price_col = f"{asset}_close"
            latest_price = recent_data[price_col].iloc[-1] if price_col in recent_data.columns else None
            latest_date = recent_data.index[-1] if hasattr(recent_data.index, '__getitem__') else None
            
            return {
                'asset': asset,
                'date': str(latest_date),
                'price': float(latest_price) if latest_price is not None else None,
                'signal': int(latest_signal) if latest_signal is not None else None,
                'signal_interpretation': self._interpret_signal(latest_signal),
                'data_points_used': len(recent_data)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _interpret_signal(self, signal) -> str:
        """Interpret numerical signal into readable format"""
        if signal is None:
            return "No signal"
        elif signal > 0:
            return "BUY/LONG"
        elif signal < 0:
            return "SELL/SHORT"
        else:
            return "HOLD/FLAT"
    
    def get_signal_for_date(self, asset: str, strategy: str, date: str) -> Dict:
        """
        Get trading signal for a specific date
        
        Args:
            asset: Asset name
            strategy: Strategy name
            date: Date in YYYY-MM-DD format
        
        Returns:
            Dictionary with signal information
        """
        try:
            # Load model
            model = self.load_model(asset, strategy)
            if model is None:
                return {"error": "Failed to load model"}
            
            # Get historical data up to that date (with some lookback)
            end_dt = datetime.strptime(date, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=100)  # 100 days lookback
            
            data = self.get_historical_data(
                asset, 
                lookback_days=100, 
                end_date=date
            )
            
            # Run diagnosis
            result = self.diagnose(model, data, asset)
            result['requested_date'] = date
            
            return result
            
        except Exception as e:
            return {"error": str(e)}

# %%
########################################################
# Usage Examples
########################################################

# Initialize the model user
model_user = ModelUser()

# Configuration - Change these as needed
TARGET_ASSET = 'bitcoin'        # Change this
TARGET_STRATEGY = 'sma_crossover'   # Change this

print(f"🎯 Target: {TARGET_ASSET} + {TARGET_STRATEGY}")

# %%
########################################################
# List Available Models
########################################################

print("📋 Available registered models:")
registered_models = model_user.list_registered_models()

if len(registered_models) > 0:
    print(registered_models.to_string(index=False))
else:
    print("❌ No registered models found")

# %%
########################################################
# Load and Inspect Model
########################################################

# Get model information
model_info = model_user.get_model_info(TARGET_ASSET, TARGET_STRATEGY)

if 'error' not in model_info:
    print(f"\n📊 Model Information:")
    print(f"Name: {model_info['model_name']}")
    print(f"Version: {model_info['latest_version']}")
    print(f"Stage: {model_info['current_stage']}")
    print(f"Strategy Type: {model_info['tags'].get('strategy_type', 'Unknown')}")
    
    # Show key performance metrics
    print(f"\n📈 Performance Metrics:")
    key_metrics = ['wf_sharpe_ratio', 'wf_profit_factor', 'wf_total_return', 'wf_max_drawdown']
    for metric in key_metrics:
        value = model_info['metrics'].get(metric)
        if value is not None:
            print(f"  {metric}: {value:.4f}")
else:
    print(f"❌ Model info error: {model_info['error']}")

# %%
########################################################
# Load Model and Get Yesterday's Signal
########################################################

# Load the model
model = model_user.load_model(TARGET_ASSET, TARGET_STRATEGY)

if model is not None:
    print(f"\n🔍 Running diagnosis on yesterday's data...")
    
    # Get yesterday's data
    yesterday_data = model_user.get_yesterday_data(TARGET_ASSET)
    
    if not yesterday_data.empty:
        # Run diagnosis
        diagnosis = model_user.diagnose(model, yesterday_data, TARGET_ASSET)
        
        if 'error' not in diagnosis:
            print(f"📅 Diagnosis Results:")
            print(f"  Date: {diagnosis['date']}")
            print(f"  Asset: {diagnosis['asset']}")
            print(f"  Price: ${diagnosis['price']:.2f}" if diagnosis['price'] else "  Price: N/A")
            print(f"  Signal: {diagnosis['signal']}")
            print(f"  Interpretation: {diagnosis['signal_interpretation']}")
            print(f"  Data points used: {diagnosis['data_points_used']}")
        else:
            print(f"❌ Diagnosis error: {diagnosis['error']}")
    else:
        print("❌ No yesterday data available")
else:
    print("❌ Could not load model")

# %%
########################################################
# Historical Signal Analysis
########################################################

if model is not None:
    print(f"\n📊 Historical Signal Analysis (Last 10 days):")
    
    # Get signals for last 10 days
    for i in range(10):
        days_back = i + 1
        target_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        result = model_user.get_signal_for_date(TARGET_ASSET, TARGET_STRATEGY, target_date)
        
        if 'error' not in result:
            signal_str = f"{result['signal']} ({result['signal_interpretation']})"
            price_str = f"${result['price']:.2f}" if result['price'] else "N/A"
            print(f"  {target_date}: {price_str} -> {signal_str}")
        else:
            print(f"  {target_date}: Error - {result['error']}")

# %%
########################################################
# Summary and Next Steps
########################################################

print("\n" + "="*60)
print("📋 USAGE SUMMARY")
print("="*60)

print(f"Target Asset: {TARGET_ASSET}")
print(f"Target Strategy: {TARGET_STRATEGY}")

if 'error' not in model_info:
    print(f"Model Loaded: ✅ {model_info['model_name']} v{model_info['latest_version']}")
else:
    print(f"Model Status: ❌ {model_info['error']}")

print(f"\n💡 Key Functions:")
print(f"1. Load model: model_user.load_model('{TARGET_ASSET}', '{TARGET_STRATEGY}')")
print(f"2. Get yesterday data: model_user.get_yesterday_data('{TARGET_ASSET}')")
print(f"3. Run diagnosis: model_user.diagnose(model, data, '{TARGET_ASSET}')")
print(f"4. Get specific date signal: model_user.get_signal_for_date('{TARGET_ASSET}', '{TARGET_STRATEGY}', '2024-01-15')")

print(f"\n🎯 To use different models:")
print(f"   - Change TARGET_ASSET and TARGET_STRATEGY variables")
print(f"   - Re-run the cells")

# %%