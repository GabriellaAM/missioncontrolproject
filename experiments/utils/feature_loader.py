"""
Feature Loader - Personalized feature loading system for SOROS Lab
"""

import pandas as pd
import json
import mlflow
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable
from datetime import datetime

class FeatureLoader:
    """Lightweight feature loading system for crypto and macro data"""
    
    def __init__(self, base_path: str = "../data_parquet", 
                 start_date: Optional[Union[str, datetime]] = None, 
                 end_date: Optional[Union[str, datetime]] = None):
        self.base_path = Path(base_path)
        self.start_date = start_date
        self.end_date = end_date
        
        # Metadata tracking
        self._metadata = {
            'config': {
                'base_path': str(base_path),
                'start_date': str(start_date) if start_date else None,
                'end_date': str(end_date) if end_date else None,
                'load_timestamp': datetime.now().isoformat()
            },
            'features_loaded': {},
            'data_quality': {},
            'sources': []
        }
    
    def _apply_date_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply global date filters to dataframe"""
        if 'timestamp' not in df.columns:
            return df
        
        result = df.copy()
        
        # Since we normalize all timestamps to timezone-naive, use simple comparison
        if self.start_date:
            start_dt = pd.to_datetime(self.start_date)
            result = result[result['timestamp'] >= start_dt]
            
        if self.end_date:
            end_dt = pd.to_datetime(self.end_date)
            result = result[result['timestamp'] <= end_dt]
        
        return result.sort_values('timestamp').reset_index(drop=True)
    
    def _load_single_crypto(self, asset: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load single crypto asset"""
        default_columns = ['timestamp', 'open', 'close', 'high', 'low', 'total_volume', 'market_cap']
        
        # Load full dataset to check available columns
        path = self.base_path / "crypto_data" / "coingecko" / asset / "data.parquet"
        full_df = pd.read_parquet(path)
        
        # Handle BTC quote columns logic
        if columns is None:
            if asset == 'bitcoin':
                # For Bitcoin, exclude _btc columns (self-referential)
                available_btc_cols = [col for col in full_df.columns if col.endswith('_btc')]
                columns = [col for col in default_columns if col in full_df.columns]
            else:
                # For other assets, include _btc columns
                btc_columns = [col for col in full_df.columns if col.endswith('_btc')]
                columns = default_columns + btc_columns
                # Filter to only existing columns
                columns = [col for col in columns if col in full_df.columns]
        
        df = full_df[columns]
        
        # Ensure timestamp is timezone-naive for consistent merging
        if hasattr(df['timestamp'].dtype, 'tz') and df['timestamp'].dtype.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Track metadata
        self._metadata['sources'].append({
            'type': 'crypto',
            'asset': asset,
            'path': str(path),
            'columns': columns,
            'shape': df.shape,
            'date_range': [str(df['timestamp'].min()), str(df['timestamp'].max())]
        })
        
        return df
    
    def _load_single_fred(self, indicator: str, rename_as: str) -> pd.DataFrame:
        """Load single FRED indicator"""
        path = self.base_path / "macro_data" / "fred" / indicator / "data.parquet"
        df = pd.read_parquet(path)[['timestamp', 'value']]
        
        # Ensure timestamp is timezone-naive for consistent merging
        if hasattr(df['timestamp'].dtype, 'tz') and df['timestamp'].dtype.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        result = df.rename(columns={'value': rename_as})
        
        # Track metadata
        self._metadata['sources'].append({
            'type': 'fred',
            'indicator': indicator,
            'renamed_as': rename_as,
            'path': str(path),
            'shape': result.shape,
            'date_range': [str(result['timestamp'].min()), str(result['timestamp'].max())]
        })
        
        return result
    
    def _load_single_yahoo(self, ticker: str, columns: Optional[List[str]] = None, prefix: str = None) -> pd.DataFrame:
        """Load single Yahoo ticker"""
        default_columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume']
        columns = columns or default_columns
        
        path = self.base_path / "macro_data" / "yahoo" / ticker / "data.parquet"
        df = pd.read_parquet(path)[columns]
        
        # Ensure timestamp is timezone-naive for consistent merging
        if hasattr(df['timestamp'].dtype, 'tz') and df['timestamp'].dtype.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Add prefix to columns (except timestamp)
        prefix = prefix or ticker
        rename_dict = {col: f"{prefix}_{col}" for col in columns if col != 'timestamp'}
        result = df.rename(columns=rename_dict)
        
        # Track metadata
        self._metadata['sources'].append({
            'type': 'yahoo',
            'ticker': ticker,
            'prefix': prefix,
            'path': str(path),
            'columns': columns,
            'renamed_columns': list(rename_dict.values()),
            'shape': result.shape,
            'date_range': [str(result['timestamp'].min()), str(result['timestamp'].max())]
        })
        
        return result
    
    def _load_single_calculated(self, feature_name: str, rename_as: str) -> pd.DataFrame:
        """Load single calculated feature"""
        path = self.base_path / "macro_data" / "calculated" / feature_name / "data.parquet"
        df = pd.read_parquet(path)[['timestamp', 'value']]
        
        # Ensure timestamp is timezone-naive for consistent merging
        if hasattr(df['timestamp'].dtype, 'tz') and df['timestamp'].dtype.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        result = df.rename(columns={'value': rename_as})
        
        # Track metadata
        self._metadata['sources'].append({
            'type': 'calculated',
            'feature_name': feature_name,
            'renamed_as': rename_as,
            'path': str(path),
            'shape': result.shape,
            'date_range': [str(result['timestamp'].min()), str(result['timestamp'].max())]
        })
        
        return result
    
    def load_crypto_features(self, 
                           assets: Union[str, List[str]], 
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load multiple crypto assets at once
        
        Args:
            assets: Single asset name or list of asset names
            columns: Columns to load (default: timestamp, OHLCV, market_cap)
        
        Returns:
            DataFrame with prefixed columns (e.g., bitcoin_close, ethereum_open)
        """
        
        if isinstance(assets, str):
            assets = [assets]
        
        dfs = []
        for asset in assets:
            df = self._load_single_crypto(asset, columns)
            # Add asset prefix to columns (except timestamp)
            rename_dict = {col: f"{asset}_{col}" for col in df.columns if col != 'timestamp'}
            df = df.rename(columns=rename_dict)
            dfs.append(df)
        
        # Merge all crypto assets
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on='timestamp', how='outer')
        
        return self._apply_date_filter(result)
    
    def load_fred_features(self, 
                          indicators: Union[str, List[str], Dict[str, str]]) -> pd.DataFrame:
        """
        Load multiple FRED indicators
        
        Args:
            indicators: Can be:
                - str: single indicator 
                - List[str]: multiple indicators (use original names lowercased)
                - Dict[str, str]: {indicator: rename_as} for custom naming
        
        Returns:
            DataFrame with FRED features
        
        Examples:
            load_fred_features('creditSpreads')
            load_fred_features(['creditSpreads', 'fedFundsRate'])
            load_fred_features({'creditSpreads': 'credit_spread', 'fedFundsRate': 'fed_rate'})
        """
        
        if isinstance(indicators, str):
            indicators = {indicators: indicators.lower()}
        elif isinstance(indicators, list):
            indicators = {ind: ind.lower() for ind in indicators}
        
        dfs = []
        for indicator, rename_as in indicators.items():
            df = self._load_single_fred(indicator, rename_as)
            dfs.append(df)
        
        # Merge all FRED features
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on='timestamp', how='outer')
        
        return self._apply_date_filter(result)
    
    def load_yahoo_features(self, 
                           tickers: Union[str, List[str], Dict[str, str]],
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load multiple Yahoo tickers
        
        Args:
            tickers: Can be:
                - str: single ticker
                - List[str]: multiple tickers (use ticker name as prefix)
                - Dict[str, str]: {ticker: prefix} for custom prefixes
            columns: Columns to load (default: timestamp, OHLCV)
        
        Returns:
            DataFrame with prefixed columns (e.g., vix_close, dxy_open)
        
        Examples:
            load_yahoo_features('vix')
            load_yahoo_features(['vix', 'dxy'])
            load_yahoo_features({'vix': 'volatility', 'dxy': 'dollar'})
        """
        
        if isinstance(tickers, str):
            tickers = {tickers: tickers}
        elif isinstance(tickers, list):
            tickers = {ticker: ticker for ticker in tickers}
        
        dfs = []
        for ticker, prefix in tickers.items():
            df = self._load_single_yahoo(ticker, columns, prefix)
            dfs.append(df)
        
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on='timestamp', how='outer')
        
        return self._apply_date_filter(result)
    
    def load_calculated_features(self,
                               features: Union[str, List[str], Dict[str, str]]) -> pd.DataFrame:
        """
        Load multiple calculated features
        
        Args:
            features: Can be:
                - str: single feature
                - List[str]: multiple features (use original names)
                - Dict[str, str]: {feature_name: rename_as} for custom naming
        
        Returns:
            DataFrame with calculated features
        """
        
        if isinstance(features, str):
            features = {features: features}
        elif isinstance(features, list):
            features = {feat: feat for feat in features}
        
        dfs = []
        for feature_name, rename_as in features.items():
            df = self._load_single_calculated(feature_name, rename_as)
            dfs.append(df)
        
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on='timestamp', how='outer')
        
        return self._apply_date_filter(result)
    
    def custom_query(self, query_func: Callable, *args, **kwargs) -> pd.DataFrame:
        """
        Execute custom query function for edge cases
        
        Args:
            query_func: Function that takes base_path as first argument and returns DataFrame
            *args, **kwargs: Additional arguments passed to query_func
        
        Returns:
            DataFrame with date filters applied
        
        Example:
            def load_special_data(base_path, some_param):
                # Your custom loading logic
                return pd.DataFrame(...)
            
            df = loader.custom_query(load_special_data, some_param='value')
        """
        result = query_func(self.base_path, *args, **kwargs)
        return self._apply_date_filter(result)
    
    def build_feature_set(self, 
                         crypto_assets: Optional[Union[str, List[str]]] = None,
                         fred_indicators: Optional[Union[str, List[str], Dict[str, str]]] = None,
                         yahoo_tickers: Optional[Union[str, List[str], Dict[str, str]]] = None,
                         calculated_features: Optional[Union[str, List[str], Dict[str, str]]] = None,
                         fillna_method: str = 'ffill') -> pd.DataFrame:
        """
        Build a complete feature set by combining multiple data sources
        
        Args:
            crypto_assets: Crypto assets to load
            fred_indicators: FRED indicators to load
            yahoo_tickers: Yahoo tickers to load
            calculated_features: Calculated features to load
            fillna_method: Method to handle missing values ('ffill', 'bfill', None)
        
        Returns:
            Combined DataFrame with all features
        
        Example:
            df = loader.build_feature_set(
                crypto_assets=['bitcoin', 'ethereum'],
                fred_indicators={'creditSpreads': 'credit_spread'},
                yahoo_tickers=['vix', 'dxy'],
                calculated_features='rty_ym_ratio'
            )
        """
        
        dfs = []
        
        if crypto_assets:
            crypto_df = self.load_crypto_features(crypto_assets)
            dfs.append(crypto_df)
        
        if fred_indicators:
            fred_df = self.load_fred_features(fred_indicators)
            dfs.append(fred_df)
        
        if yahoo_tickers:
            yahoo_df = self.load_yahoo_features(yahoo_tickers)
            dfs.append(yahoo_df)
        
        if calculated_features:
            calc_df = self.load_calculated_features(calculated_features)
            dfs.append(calc_df)
        
        if not dfs:
            raise ValueError("At least one feature type must be specified")
        
        # Merge all dataframes
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on='timestamp', how='outer')
        
        # Handle missing values
        if fillna_method == 'ffill':
            result = result.fillna(method='ffill')
        elif fillna_method == 'bfill':
            result = result.fillna(method='bfill')
        
        # Track data quality before cleaning
        rows_before_cleaning = len(result)
        missing_data_before = result.isnull().sum().to_dict()
        
        # Drop initial NaN rows (typically at the beginning of the dataset)
        result = result.dropna(axis=0, how='any').reset_index(drop=True)
        
        # Track data quality after cleaning
        rows_after_cleaning = len(result)
        
        # Update metadata with final dataset info
        self._metadata['data_quality'] = {
            'final_shape': result.shape,
            'final_date_range': [str(result['timestamp'].min()), str(result['timestamp'].max())] if len(result) > 0 else [None, None],
            'rows_before_cleaning': rows_before_cleaning,
            'rows_after_cleaning': rows_after_cleaning,
            'rows_dropped': rows_before_cleaning - rows_after_cleaning,
            'missing_data_before_cleaning': missing_data_before,
            'final_columns': result.columns.tolist()
        }
        
        return result
    
    def get_metadata(self) -> Dict:
        """Get complete metadata about loaded features"""
        return self._metadata.copy()
    
    def log_feature_metadata_to_mlflow(self):
        """Log feature metadata as MLflow artifacts"""
        
        # Log feature metadata as JSON
        metadata_str = json.dumps(self._metadata, indent=2, default=str)
        mlflow.log_text(metadata_str, "feature_metadata.json")
        
        # Log configuration as parameters
        mlflow.log_param("feature_loader_start_date", self._metadata['config']['start_date'])
        mlflow.log_param("feature_loader_end_date", self._metadata['config']['end_date'])
        mlflow.log_param("feature_loader_base_path", self._metadata['config']['base_path'])
        
        # Log data quality metrics
        if 'data_quality' in self._metadata:
            dq = self._metadata['data_quality']
            mlflow.log_metric("final_dataset_rows", dq['rows_after_cleaning'])
            mlflow.log_metric("final_dataset_columns", len(dq['final_columns']))
            mlflow.log_metric("rows_dropped_cleaning", dq['rows_dropped'])
        
        # Log feature sources summary
        source_summary = {}
        for source in self._metadata['sources']:
            source_type = source['type']
            if source_type not in source_summary:
                source_summary[source_type] = []
            source_summary[source_type].append(source.get('asset', source.get('indicator', source.get('ticker', source.get('feature_name')))))
        
        for source_type, assets in source_summary.items():
            mlflow.log_param(f"features_{source_type}_assets", ','.join(assets))
    
    def generate_feature_stats_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive feature statistics report"""
        
        stats_data = []
        for col in df.columns:
            if col == 'timestamp':
                continue
                
            col_data = df[col]
            stats = {
                'feature': col,
                'count': len(col_data),
                'missing': col_data.isnull().sum(),
                'missing_pct': (col_data.isnull().sum() / len(col_data)) * 100,
                'dtype': str(col_data.dtype),
                'mean': col_data.mean() if col_data.dtype.kind in 'biufc' else None,
                'std': col_data.std() if col_data.dtype.kind in 'biufc' else None,
                'min': col_data.min() if col_data.dtype.kind in 'biufc' else None,
                'max': col_data.max() if col_data.dtype.kind in 'biufc' else None,
                'q25': col_data.quantile(0.25) if col_data.dtype.kind in 'biufc' else None,
                'q50': col_data.quantile(0.50) if col_data.dtype.kind in 'biufc' else None,
                'q75': col_data.quantile(0.75) if col_data.dtype.kind in 'biufc' else None,
            }
            stats_data.append(stats)
        
        return pd.DataFrame(stats_data)
    
    def log_feature_stats_to_mlflow(self, df: pd.DataFrame):
        """Log feature statistics as MLflow artifacts"""
        
        # Generate stats report
        stats_df = self.generate_feature_stats_report(df)
        
        # Save as CSV artifact
        stats_csv = stats_df.to_csv(index=False)
        mlflow.log_text(stats_csv, "feature_statistics.csv")
        
        # Log key metrics
        mlflow.log_metric("total_features", len(stats_df))
        mlflow.log_metric("features_with_missing_data", (stats_df['missing'] > 0).sum())
        mlflow.log_metric("avg_missing_data_pct", stats_df['missing_pct'].mean())
    
    def log_all_to_mlflow(self, df: pd.DataFrame):
        """Log all feature metadata and statistics to MLflow"""
        self.log_feature_metadata_to_mlflow()
        self.log_feature_stats_to_mlflow(df)
        
        print("✅ Feature metadata and statistics logged to MLflow")
        print(f"📊 Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        if 'data_quality' in self._metadata:
            dropped = self._metadata['data_quality']['rows_dropped']
            if dropped > 0:
                print(f"🧹 Dropped {dropped} rows during cleaning")


# Example usage functions
def example_bitcoin_macro_features():
    """Example: Bitcoin with macro features"""
    
    loader = FeatureLoader(start_date='2020-01-01', end_date='2024-01-01')
    
    features = loader.build_feature_set(
        crypto_assets='bitcoin',
        fred_indicators={'creditSpreads': 'credit_spread', 'fedFundsRate': 'fed_rate'},
        yahoo_tickers={'vix': 'volatility'},
        calculated_features='rty_ym_ratio'
    )
    
    return features

def example_multi_crypto_comparison():
    """Example: Multiple crypto assets for comparison"""
    
    loader = FeatureLoader(start_date='2021-01-01')
    
    features = loader.build_feature_set(
        crypto_assets=['bitcoin', 'ethereum', 'solana'],
        yahoo_tickers=['vix', 'dxy']
    )
    
    return features