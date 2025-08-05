"""
Simple runner for SOROS models without full ZenML pipeline
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import os
import logging

from ..models.primary_rules import RuleBasedPrimaryModel
from ..models.primary_ml import MLBasedPrimaryModel
from ..models.primary_hybrid import HybridPrimaryModel
from ..models.model_registry import get_model_registry
from ..meta_labeling.triple_barrier import TripleBarrierLabeler
from ..meta_labeling.meta_model import MetaModel


class SimpleRunner:
    """Run SOROS models without ZenML pipeline complexity"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
    def load_asset_data(self, asset_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load asset data from CSV"""
        # Map asset ID to filename
        asset_map = {
            'BTC-USD': 'btc_dominance.csv',
            'ETH-USD': 'eth_dominance.csv',
            'bitcoin': 'btc_dominance.csv',
            'ethereum': 'eth_dominance.csv'
        }
        
        filename = asset_map.get(asset_id, f'{asset_id.lower()}_dominance.csv')
        filepath = os.path.join(self.data_dir, 'micro', 'marketData', filename)
        
        if not os.path.exists(filepath):
            # Try alternative path
            filepath = os.path.join(self.data_dir, 'micro', 'marketData', f'{asset_id.lower()}.csv')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found for {asset_id}")
            
        # Load data
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Filter by date if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic signals from price data"""
        signals = pd.DataFrame(index=data.index)
        
        # RSI Signal
        if 'close' in data.columns:
            # Simple RSI approximation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            signals['RSI_Bullish_USD'] = (rsi < 30).astype(int) - (rsi > 70).astype(int)
        
        # Donchian Channel Signal
        if 'high' in data.columns and 'low' in data.columns:
            upper = data['high'].rolling(window=20).max()
            lower = data['low'].rolling(window=20).min()
            signals['DonchianEnsembleUSD'] = (data['close'] > upper.shift(1)).astype(int) - \
                                            (data['close'] < lower.shift(1)).astype(int)
        
        # MACD Signal
        if 'close' in data.columns:
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            signals['MACDSignalUSD'] = (macd > signal).astype(int) - (macd < signal).astype(int)
        
        # Volume Signal (if available)
        if 'volume' in data.columns:
            vol_sma = data['volume'].rolling(window=20).mean()
            signals['VolumeSignal'] = (data['volume'] > vol_sma * 1.5).astype(int)
        
        # Fill NaN values
        signals = signals.fillna(0)
        
        return signals
    
    def run_rule_based_model(
        self, 
        asset_id: str,
        signal_names: list = None,
        combination_strategy: str = 'ensemble',
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Run rule-based model on asset"""
        try:
            # Load data
            data = self.load_asset_data(asset_id, start_date, end_date)
            
            # Generate signals
            signals = self.generate_signals(data)
            
            # Default signals if none specified
            if signal_names is None:
                signal_names = ['RSI_Bullish_USD', 'DonchianEnsembleUSD']
            
            # Filter available signals
            available_signals = [s for s in signal_names if s in signals.columns]
            if not available_signals:
                raise ValueError(f"No valid signals found. Available: {list(signals.columns)}")
            
            # Create model
            model = RuleBasedPrimaryModel(
                signal_names=available_signals,
                combination_strategy=combination_strategy,
                confidence_mode='agreement'
            )
            
            # Get predictions
            predictions = model.predict(signals[available_signals])
            
            # Calculate simple performance
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                strategy_returns = predictions.shift(1) * returns
                strategy_returns = strategy_returns.dropna()
                
                # Calculate metrics
                total_return = (1 + strategy_returns).prod() - 1
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
                max_dd = (strategy_returns.cumsum().cummax() - strategy_returns.cumsum()).max()
                signal_rate = (predictions != 0).mean()
                
                return {
                    'success': True,
                    'model_type': 'rule_based',
                    'model_version': model.model_version,
                    'signals_used': available_signals,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'signal_rate': signal_rate,
                    'num_trades': (predictions.diff() != 0).sum(),
                    'model': model
                }
            else:
                return {
                    'success': True,
                    'model_type': 'rule_based', 
                    'model_version': model.model_version,
                    'signals_used': available_signals,
                    'signal_rate': (predictions != 0).mean(),
                    'model': model,
                    'error': 'No price data for performance calculation'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def prepare_ml_training_data(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        labeling_params: Dict[str, Any] = None
    ) -> tuple:
        """Prepare training data for ML model"""
        if labeling_params is None:
            labeling_params = {
                'profit_taking_multiple': 2.0,
                'stop_loss_multiple': 1.0,
                'max_holding_period': 5
            }
        
        # Create labels using triple barrier
        labeler = TripleBarrierLabeler(**labeling_params)
        
        # Use every 10th point as events
        events = data.index[::10]
        labels = labeler.label_events(data['close'], events)
        
        # Align signals with labels
        features = signals.loc[labels.index]
        
        return features, labels['label']
    
    def simple_backtest(self, predictions: pd.Series, prices: pd.Series) -> Dict[str, float]:
        """Simple backtest calculation"""
        returns = prices.pct_change()
        strategy_returns = predictions.shift(1) * returns
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        mean_return = strategy_returns.mean() * 252
        vol = strategy_returns.std() * np.sqrt(252)
        sharpe = mean_return / vol if vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        win_rate = (strategy_returns > 0).mean()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        }