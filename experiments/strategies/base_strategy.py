from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
import mlflow
import mlflow.pyfunc


class BaseStrategy(ABC):
    
    # Default flags for strategy types
    is_meta_model = False
    primary_run_id = None
    
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
    
    def get_warmup_days(self, params: Dict = None) -> int:
        """
        Return number of days needed before start_date for indicator warmup.
        
        Override this in strategies that use lagged indicators (SMA, EMA, etc.).
        
        Args:
            params: Strategy parameters (optional). If provided, calculate
                    based on actual parameters. If None, return worst-case.
        
        Returns:
            Number of days needed for warmup. Default is 0 for strategies
            without lagged indicators.
        """
        return 0  # Default: no warmup needed
    
    def get_normalization_config(self) -> Dict:
        """
        Return normalization configuration for this strategy.
        
        Override this in strategies that require feature normalization.
        
        Returns:
            Dict with normalization config or None if no normalization needed.
            Format:
            {
                'columns': {
                    'column_name': 'scaler_type',  # standard, minmax, robust
                    'pattern:*': 'scaler_type',    # patterns supported
                },
                'exclude': ['signal', 'label'],  # columns to never normalize
                'default': 'standard'  # default scaler if not specified
            }
        """
        return None  # Default: no normalization needed
    
    # Default crypto features used by most strategies
    used_crypto_features = ['close']
    
    def get_input_schema(self):
        """Generate MLflow input schema using explicit used_crypto_features property."""
        from mlflow.types import DataType, ColSpec, Schema
        
        # Get strategy's feature requirements 
        try:
            required_features = self.get_required_features()
        except:
            # Fallback to basic detection if method fails
            return self._get_basic_input_schema()
        
        columns = []
        
        # 1. Crypto Assets - Add only the features specified in used_crypto_features
        crypto_assets = required_features.get('crypto_assets', [])
        crypto_features = self.used_crypto_features
        
        if crypto_assets and crypto_features:
            for asset in crypto_assets:
                for feature in crypto_features:
                    if feature == 'volume':
                        # Map 'volume' to 'total_volume' as that's what the FeatureLoader uses
                        columns.append(ColSpec(DataType.double, f"{asset}_total_volume"))
                    else:
                        columns.append(ColSpec(DataType.double, f"{asset}_{feature}"))
                
                # Add BTC-denominated features for non-BTC assets if specified
                if asset != 'bitcoin':
                    for feature in crypto_features:
                        if feature in ['close', 'market_cap']:  # Only these have BTC versions
                            columns.append(ColSpec(DataType.double, f"{asset}_{feature}_btc"))
        
        # 2. FRED Indicators - Add all FRED features (they're typically single-column)
        fred_indicators = required_features.get('fred_indicators')
        if fred_indicators:
            for indicator_key, alias in fred_indicators.items():
                columns.append(ColSpec(DataType.double, alias))
        
        # 3. Yahoo Tickers - Add all yahoo features (for now, can be made configurable later)
        yahoo_tickers = required_features.get('yahoo_tickers')
        if yahoo_tickers:
            for ticker, prefix in yahoo_tickers.items():
                # Add all OHLCV features unless it's an index like VIX
                columns.append(ColSpec(DataType.double, f"{prefix}_close"))
                if not any(x in ticker.lower() for x in ['vix', 'index']):
                    columns.extend([
                        ColSpec(DataType.double, f"{prefix}_open"),
                        ColSpec(DataType.double, f"{prefix}_high"), 
                        ColSpec(DataType.double, f"{prefix}_low"),
                        ColSpec(DataType.double, f"{prefix}_volume")
                    ])
        
        # 4. Calculated Features - Add all calculated features (they're typically single-column)
        calculated_features = required_features.get('calculated_features')
        if calculated_features:
            for feature_key, alias in calculated_features.items():
                columns.append(ColSpec(DataType.double, alias))
        
        # If no features specified, fall back to basic detection
        if not columns:
            return self._get_basic_input_schema()
        
        return Schema(columns)
    
    def _get_basic_input_schema(self):
        """Fallback method using pattern-based detection."""
        from mlflow.types import DataType, ColSpec, Schema
        
        asset = getattr(self, 'asset', 'bitcoin')
        columns = [ColSpec(DataType.double, f"{asset}_close")]
        
        # Detect if strategy needs OHLC data
        if self._needs_ohlc_data():
            columns.extend([
                ColSpec(DataType.double, f"{asset}_high"),
                ColSpec(DataType.double, f"{asset}_low")
            ])
        
        # Detect if strategy needs volume data
        if self._needs_volume_data():
            columns.append(ColSpec(DataType.double, f"{asset}_total_volume"))
        
        return Schema(columns)
    
    def get_input_example(self) -> pd.DataFrame:
        """Generate minimal input example for MLflow signature validation."""
        # Create a minimal sample with just 1 row for signature validation
        schema = self.get_input_schema()
        sample_data = {}
        
        for col_spec in schema.inputs:
            # Generate minimal realistic values
            col_name = col_spec.name
            if 'close' in col_name and 'btc' not in col_name:
                sample_data[col_name] = [50000.0]
            elif '_btc' in col_name:
                sample_data[col_name] = [0.02]
            elif any(x in col_name.lower() for x in ['volume', 'market_cap']):
                sample_data[col_name] = [1000000.0]
            elif any(x in col_name.lower() for x in ['vix', 'volatility']):
                sample_data[col_name] = [25.0]
            elif any(x in col_name.lower() for x in ['rate', 'fed']):
                sample_data[col_name] = [0.05]
            else:
                sample_data[col_name] = [100.0]  # Generic value
        
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
        from mlflow.models import ModelSignature
        
        # Get schema and input example
        input_schema = self.get_input_schema()
        input_example = self.get_input_example()
        signature = None
        
        try:
            predictions = self.model.predict(input_example)
            # Create signature from schema and predictions
            from mlflow.models import infer_signature
            signature = infer_signature(input_example, predictions)
            print(f"✅ Generated MLflow signature from schema")
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
        from mlflow.models import ModelSignature
        
        # Get schema and input example
        input_schema = self.get_input_schema()
        input_example = self.get_input_example()
        signature = None
        
        try:
            predictions = self.model.predict(input_example)
            # Create signature from schema and predictions
            from mlflow.models import infer_signature
            signature = infer_signature(input_example, predictions)
            print(f"✅ Generated MLflow signature from schema")
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
        
        # Get schema and input example
        input_schema = self.get_input_schema()
        input_example = self.get_input_example()
        signature = None
        
        try:
            sample_output = model.predict(None, input_example)
            signature = mlflow.models.infer_signature(input_example, sample_output)
            print(f"✅ Generated PyFunc signature from schema")
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
            # Log with proper artifact name to avoid temporary filename being used
            mlflow.log_artifact(temp_path, artifact_path="strategy_config.json")
        finally:
            os.unlink(temp_path)