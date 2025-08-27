from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import mlflow
import mlflow.pyfunc


class BaseStrategy(ABC):
    
    def __init__(self, name: str):
        self.name = name
    
    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """Return strategy type: 'mean_reversion' or 'trend_following'"""
        pass
    
    @abstractmethod
    def get_required_features(self) -> Dict[str, Any]:
        """
        Define what features this strategy needs for FeatureLoader.build_feature_set()
        
        Returns dict with keys:
            - crypto_assets: List of crypto assets needed
            - fred_indicators: Dict of FRED indicators {key: alias}
            - yahoo_tickers: Dict of Yahoo tickers {ticker: alias}
            - calculated_features: Dict of calculated features {feature: alias}
        """
        pass
    
    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Calculate trading signals based on strategy logic."""
        pass
    
    @abstractmethod
    def optimize(self, data: pd.DataFrame, train_start: str, train_end: str, 
                 n_trials: int = 1000, **kwargs) -> Dict:
        """Optimize strategy parameters."""
        pass
    
    def get_input_example(self) -> pd.DataFrame:
        """Automatically generate sample input based on strategy requirements."""
        asset = getattr(self, 'asset', 'bitcoin')
        
        # Start with basic price data (most strategies need this)
        sample_data = {f"{asset}_close": [50000, 51000, 49000]}
        
        # Detect if strategy needs OHLC data
        if self._needs_ohlc_data():
            sample_data[f"{asset}_high"] = [51000, 52000, 50000]
            sample_data[f"{asset}_low"] = [49000, 50000, 48000]
        
        # Detect if strategy needs volume data
        if self._needs_volume_data():
            sample_data[f"{asset}_total_volume"] = [1000, 1100, 900]
        
        return pd.DataFrame(sample_data)
    
    def _needs_ohlc_data(self) -> bool:
        """Check if strategy needs high/low data."""
        # Check class name and strategy name for indicators
        class_name = self.__class__.__name__.lower()
        strategy_name = getattr(self, 'name', '').lower()
        
        ohlc_indicators = ['hilo', 'bollinger', 'donchian', 'high', 'low']
        return any(indicator in class_name or indicator in strategy_name for indicator in ohlc_indicators)
    
    def _needs_volume_data(self) -> bool:
        """Check if strategy needs volume data."""
        # Check if calculate_signals method references volume
        if hasattr(self, 'calculate_signals'):
            import inspect
            try:
                source = inspect.getsource(self.calculate_signals)
                return 'volume' in source.lower()
            except:
                return False
        return False
    
    def save_model(self, params: Dict) -> str:
        """Save model using the most appropriate MLflow flavor."""
        
        # Smart flavor selection
        if hasattr(self, 'model') and self.model is not None:
            model_type = str(type(self.model)).lower()
            
            if 'catboost' in model_type:
                return self._save_catboost_model(params)
            elif 'sklearn' in model_type:
                return self._save_sklearn_model(params)
        
        # Default to PyFunc wrapper (for rules-based or unknown models)
        return self._save_pyfunc_model(params)
    
    def _save_catboost_model(self, params: Dict) -> str:
        """Save CatBoost model using native flavor."""
        import mlflow.catboost
        from mlflow.models import infer_signature
        
        # Get input example and signature
        input_example = self.get_input_example() if hasattr(self, 'get_input_example') else None
        signature = None
        
        if input_example is not None:
            try:
                predictions = self.model.predict(input_example)
                signature = infer_signature(input_example, predictions)
                print(f"✅ Generated MLflow signature with {len(input_example)} samples")
            except Exception as e:
                print(f"⚠️  Could not generate signature: {e}")
        
        # Save CatBoost model
        model_info = mlflow.catboost.log_model(
            self.model,
            "model",
            signature=signature,
            input_example=input_example
        )
        
        # Save strategy metadata
        self._save_strategy_config(params)
        
        current_run = mlflow.active_run()
        return f"runs:/{current_run.info.run_id}/model" if current_run else model_info.model_uri
    
    def _save_sklearn_model(self, params: Dict) -> str:
        """Save sklearn model using native flavor."""
        import mlflow.sklearn
        from mlflow.models import infer_signature
        
        # Get input example and signature
        input_example = self.get_input_example() if hasattr(self, 'get_input_example') else None
        signature = None
        
        if input_example is not None:
            try:
                predictions = self.model.predict(input_example)
                signature = infer_signature(input_example, predictions)
                print(f"✅ Generated MLflow signature with {len(input_example)} samples")
            except Exception as e:
                print(f"⚠️  Could not generate signature: {e}")
        
        # Save sklearn model
        model_info = mlflow.sklearn.log_model(
            self.model,
            "model",
            signature=signature,
            input_example=input_example
        )
        
        # Save strategy metadata
        self._save_strategy_config(params)
        
        current_run = mlflow.active_run()
        return f"runs:/{current_run.info.run_id}/model" if current_run else model_info.model_uri
    
    def _save_pyfunc_model(self, params: Dict) -> str:
        """Save as PyFunc model (for rules-based or fallback)."""
        
        class StrategyModel(mlflow.pyfunc.PythonModel):
            def __init__(self, strategy_instance, strategy_params):
                self.strategy = strategy_instance
                self.strategy_params = strategy_params
            
            def predict(self, context, model_input):
                # Return only the signal column for consistency
                result = self.strategy.calculate_signals(model_input, self.strategy_params)
                return result[['signal']] if 'signal' in result.columns else result
        
        # Create wrapper
        model = StrategyModel(self, params)
        
        # Try to get input example for signature
        input_example = None
        signature = None
        
        if hasattr(self, 'get_input_example'):
            try:
                input_example = self.get_input_example()
                sample_output = model.predict(None, input_example)
                signature = mlflow.models.infer_signature(input_example, sample_output)
                print(f"✅ Generated PyFunc signature")
            except Exception as e:
                print(f"⚠️  Could not generate signature: {e}")
        
        # Save PyFunc model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            signature=signature,
            input_example=input_example
        )
        
        # Save strategy metadata
        self._save_strategy_config(params)
        
        current_run = mlflow.active_run()
        return f"runs:/{current_run.info.run_id}/model" if current_run else model_info.model_uri
    
    def _save_strategy_config(self, params: Dict):
        """Save strategy configuration as artifact."""
        import json
        import tempfile
        import os
        
        strategy_config = {
            'strategy_name': self.name,
            'strategy_type': self.strategy_type,
            'implementation_type': getattr(self, 'implementation_type', 'unknown'),
            'asset': getattr(self, 'asset', 'bitcoin'),
            'parameters': params,
            'feature_cols': getattr(self, 'feature_cols', []),
            'strategy_class': self.__class__.__module__ + '.' + self.__class__.__name__
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(strategy_config, f, indent=2, default=str)
            temp_path = f.name
        
        try:
            # Just log the artifact without specifying artifact_path to avoid directory creation
            mlflow.log_artifact(temp_path)
        finally:
            os.unlink(temp_path)