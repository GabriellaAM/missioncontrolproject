import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from datetime import datetime

class MarkovVolatility:
    """
    Implementation of a two-state Markov-Switching Regression (MSR) model 
    for volatility regime detection.
    
    This model detects two volatility states: high and low.
    """
    
    def __init__(self, data_path, train_test_split=0.80, init_date='2014-01-01', embargo_pct=0.01):
        self.data_path = data_path
        self.train_test_split = train_test_split
        self.init_date = init_date
        self.embargo_pct = embargo_pct
        self.load_data()
        
    def load_data(self):
        """Load and prepare the price data."""
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
            
            self.data = self.data.dropna()
            self.data = self.data[self.data.index >= self.init_date]
            
            split_idx = int(len(self.data) * self.train_test_split)
            self.train_data = self.data.iloc[:split_idx]
            self.test_data = self.data.iloc[split_idx:]
            
            self.asset_data = self.data['close']
            self.train_asset_data = self.train_data['close']
            self.test_asset_data = self.test_data['close']
            
            print(f"Data loaded successfully: {len(self.data)} records")
            print(f"Training set: {len(self.train_data)} records")
            print(f"Test set: {len(self.test_data)} records")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def fit_markov(self, no_regimes=2):
        """Fit Markov-Switching Regression to detect volatility regimes."""
        try:
            log_returns_full = np.log(self.asset_data).diff().dropna()
            
            split_idx = int(len(self.data) * self.train_test_split)
            embargo_size = int(len(self.data) * self.embargo_pct)
            
            log_returns_full.index = self.data.index[1:]
            
            train_end_idx = split_idx - embargo_size
            log_returns_train = log_returns_full.iloc[:train_end_idx]
            
            markov_model = sm.tsa.MarkovRegression(
                log_returns_train.iloc[1:], 
                k_regimes=no_regimes, 
                switching_variance=True, 
                exog=log_returns_train.iloc[:-1]
            )
            
            self.msreg_results = markov_model.fit()
            
            params = self.msreg_results.params
            
            full_model = sm.tsa.MarkovRegression(
                log_returns_full.iloc[1:],
                k_regimes=no_regimes,
                switching_variance=True,
                exog=log_returns_full.iloc[:-1]
            )
            
            full_results = full_model.smooth(params)
            
            low_var = full_results.smoothed_marginal_probabilities[0]
            high_var = full_results.smoothed_marginal_probabilities[1]
            
            markov_data = pd.concat([low_var, high_var], axis=1)
            markov_data.columns = ['Low_var', 'High_var']
            
            markov_data['dataset'] = 1
            markov_data.iloc[train_end_idx:split_idx, -1] = 0
            markov_data.iloc[split_idx:, -1] = 2
            
            self.markov = markov_data
            
            print("Markov Switching Regression model fitted successfully.")
            print(f"Training data: {train_end_idx} points")
            print(f"Embargo data: {embargo_size} points")
            print(f"Test data: {len(markov_data) - split_idx} points")
                
        except Exception as e:
            print(f"Error fitting Markov model: {e}")
            raise
    
    def get_volatility_state(self, date=None):
        """
        Get the volatility state (high or low) for a specific date.
        
        Parameters:
        -----------
        date : str or datetime
            The date to check volatility state for. If None, returns all dates.
            
        Returns:
        --------
        str or pandas.Series
            'high' or 'low' if date is provided, otherwise a Series with states for all dates
        """
        if not hasattr(self, 'markov'):
            raise ValueError("Model not fitted. Call fit_markov() first.")
        
        states = pd.Series('low', index=self.markov.index)
        states[self.markov['High_var'] > 0.5] = 'high'
        
        if date is not None:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            if date not in states.index:
                closest_date = min(states.index, key=lambda x: abs(x - date))
                print(f"Warning: Exact date {date} not found. Using closest date {closest_date}.")
                date = closest_date
                
            return states.loc[date]
        return states  # Fixed: Removed erroneous else clause and simplified return
    
    def plot_volatility(self):
        """Plot price and volatility states using Plotly."""
        if not hasattr(self, 'markov'):
            raise ValueError("Model not fitted. Call fit_markov() first.")
            
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=('Asset Price', 'Volatility Regime Probabilities'))
        
        fig.add_trace(
            go.Scatter(
                x=self.asset_data.index, 
                y=self.asset_data, 
                name='Price', 
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        states = self.get_volatility_state()
        
        for i in range(len(states)):
            if i == 0 or states.iloc[i] != states.iloc[i-1]:
                start_date = states.index[i]
                end_idx = i
                while end_idx < len(states)-1 and states.iloc[end_idx] == states.iloc[end_idx+1]:
                    end_idx += 1
                
                end_date = states.index[end_idx] if end_idx == len(states)-1 else states.index[end_idx+1]
                
                color = 'rgba(255,0,0,0.2)' if states.iloc[i] == 'high' else 'rgba(0,255,0,0.2)'
                
                fig.add_vrect(
                    x0=start_date, x1=end_date,
                    fillcolor=color, opacity=0.5,
                    layer="below", line_width=0,
                    row=1, col=1
                )
        
        # Moved these traces outside the loop to plot them once
        fig.add_trace(
            go.Scatter(
                x=self.markov.index, 
                y=self.markov['High_var'], 
                name='High Volatility Probability', 
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.markov.index, 
                y=self.markov['Low_var'], 
                name='Low Volatility Probability', 
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.markov.index, 
                y=[0.5] * len(self.markov), 
                name='Threshold', 
                line=dict(color='black', width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis_title="Price",
            yaxis2_title="Probability",
        )
        
        fig.update_xaxes(
            tickformat="%Y-%m",
            tickangle=45,
            tickmode="auto"
        )
        
        fig.show()
        
    def save_model(self, filename=None, directory='/Users/valter.rebelo/MissionControl/models/'):
        """Save the model to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"markov_volatility_model_{timestamp}.pkl"
        
        os.makedirs(directory, exist_ok=True)
        full_path = os.path.join(directory, filename)
        
        metadata = {
            'saved_date': datetime.now().strftime("%Y-%m-%d"),
            'data_path': self.data_path,
            'train_test_split': self.train_test_split,
            'init_date': self.init_date,
            'embargo_pct': self.embargo_pct,
            'data_range': {
                'start': self.data.index[0].strftime("%Y-%m-%d"),
                'end': self.data.index[-1].strftime("%Y-%m-%d")
            },
            'train_range': {
                'start': self.train_data.index[0].strftime("%Y-%m-%d"),
                'end': self.train_data.index[-1].strftime("%Y-%m-%d")
            },
            'test_range': {
                'start': self.test_data.index[0].strftime("%Y-%m-%d"),
                'end': self.test_data.index[-1].strftime("%Y-%m-%d")
            },
            'n_regimes': 2,
            'total_records': len(self.data),
            'train_records': len(self.train_data),
            'test_records': len(self.test_data)
        }
        
        with open(full_path, 'wb') as f:
            pickle.dump({
                'markov': self.markov if hasattr(self, 'markov') else None,
                'msreg_results': self.msreg_results if hasattr(self, 'msreg_results') else None,
                'metadata': metadata
            }, f)
        
        import json
        metadata_path = full_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Model saved to {full_path}")
        print(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load_model(cls, model=None):
        """Load a saved model."""
        data_path = '/Users/valter.rebelo/MissionControl/data/micro/candleData/bitcoin_candles.csv'
        instance = cls(data_path)
        
        if model is None:
            raise ValueError("Model filename must be provided")
            
        model_path = f"/Users/valter.rebelo/MissionControl/models/markov_volatility_model_{model}.pkl"
        
        with open(model_path, 'rb') as f:
            saved_model = pickle.load(f)
        
        for key, value in saved_model.items():
            if key != 'metadata':
                setattr(instance, key, value)
        
        if 'metadata' in saved_model:
            instance.metadata = saved_model['metadata']
            print(f"Model metadata:")
            print(f"  Saved on: {instance.metadata.get('saved_date', 'unknown')}")
            print(f"  Data range: {instance.metadata.get('data_range', {}).get('start', 'unknown')} to {instance.metadata.get('data_range', {}).get('end', 'unknown')}")
            print(f"  Records: {instance.metadata.get('total_records', 'unknown')} total, {instance.metadata.get('train_records', 'unknown')} train, {instance.metadata.get('test_records', 'unknown')} test")
        
        print(f"Model loaded from {model_path}")
        return instance
    
    def get_metadata(self):
        """Return the metadata of the model."""
        if hasattr(self, 'metadata'):
            return self.metadata
        else:
            metadata = {
                'data_path': self.data_path,
                'train_test_split': self.train_test_split,
                'init_date': self.init_date,
                'embargo_pct': self.embargo_pct,
                'data_range': {
                    'start': self.data.index[0].strftime("%Y-%m-%d"),
                    'end': self.data.index[-1].strftime("%Y-%m-%d")
                },
                'train_range': {
                    'start': self.train_data.index[0].strftime("%Y-%m-%d"),
                    'end': self.train_data.index[-1].strftime("%Y-%m-%d")
                },
                'test_range': {
                    'start': self.test_data.index[0].strftime("%Y-%m-%d"),
                    'end': self.test_data.index[-1].strftime("%Y-%m-%d")
                },
                'n_regimes': 2,
                'total_records': len(self.data),
                'train_records': len(self.train_data),
                'test_records': len(self.test_data),
                'model_fitted': hasattr(self, 'markov')
            }
            return metadata