import numpy as np
import pandas as pd
import logging
import pickle
import os
import statsmodels.api as sm
from datetime import datetime


class MarkovVolModel:
    """
    Implementation of a two-state Markov-Switching Regression model
    for volatility regime detection that can be used by the soros_system.
    
    This acts as a wrapper and interface to the MarkovVolatility model.
    """
    
    def __init__(self, model_path=None, btc_data_path=None, train_test_split=0.80, init_date='2014-01-01', embargo_pct=0.01):
        """
        Initialize the MarkovVolModel.
        
        Args:
            model_path (str, optional): Path to a pre-trained MarkovVolatility model pickle file.
                If None, will attempt to load the default model from the models directory.
            btc_data_path (str, optional): Path to the Bitcoin data CSV file.
                Used for training if needed.
            train_test_split (float, optional): Ratio for train/test split. Default is 0.80.
            init_date (str, optional): Initial date for data filtering. Default is '2014-01-01'.
            embargo_pct (float, optional): Percentage of data to use as embargo between train and test. Default is 0.01.
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.volatility_cache = {}
        self.btc_data_path = btc_data_path
        self.train_test_split = train_test_split
        self.init_date = init_date
        self.embargo_pct = embargo_pct
        
        # Attributes for training and data
        self.data = None
        self.train_data = None
        self.test_data = None
        self.msreg_results = None
        self.markov = None
        
        # Load the model if provided
        if model_path:
            self.load_model(model_path)
        else:
            # Try to load the default model from the models directory
            try:
                default_model_path = self._find_default_model()
                if default_model_path:
                    self.load_model(default_model_path)
            except Exception as e:
                self.logger.warning(f"Could not load default model: {e}")
    
    def _find_default_model(self):
        """
        Find the most recent MarkovVolatility model in the models directory.
        
        Returns:
            str: Path to the most recent model, or None if no models found
        """
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'models')
        
        if not os.path.exists(models_dir):
            self.logger.warning(f"Models directory not found: {models_dir}")
            return None
        
        # Find all markov volatility models
        model_files = [f for f in os.listdir(models_dir) 
                      if f.startswith('markov_volatility_model_') and f.endswith('.pkl')]
        
        if not model_files:
            self.logger.warning(f"No markov volatility models found in {models_dir}")
            return None
        
        # Sort by date (assuming filename format includes timestamp)
        model_files.sort(reverse=True)
        latest_model = model_files[0]
        
        self.logger.info(f"Found latest model: {latest_model}")
        return os.path.join(models_dir, latest_model)
    
    def load_model(self, model_path):
        """
        Load a MarkovVolatility model from a pickle file.
        
        Args:
            model_path (str): Path to the model pickle file
        """
        try:
            self.logger.info(f"Loading MarkovVolatility model from {model_path}")
            
            with open(model_path, 'rb') as f:
                saved_model = pickle.load(f)
            
            # Check if the model is properly formatted
            if isinstance(saved_model, dict) and 'markov' in saved_model:
                self.model = saved_model
                self.markov = saved_model.get('markov')
                self.msreg_results = saved_model.get('msreg_results')
                self.logger.info(f"MarkovVolatility model loaded successfully")
                
                # Log metadata if available
                if 'metadata' in saved_model:
                    metadata = saved_model['metadata']
                    self.metadata = metadata
                    self.logger.info(f"Model date range: {metadata.get('data_range', {}).get('start', 'unknown')} "
                                    f"to {metadata.get('data_range', {}).get('end', 'unknown')}")
            else:
                self.logger.warning(f"Invalid model format")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = None
    
    def load_data(self, data_path=None):
        """
        Load and prepare the price data.
        
        Args:
            data_path (str, optional): Path to the data CSV file. If None, uses self.btc_data_path.
        """
        try:
            # Use provided data_path or fall back to btc_data_path
            if data_path is None:
                if self.btc_data_path is None:
                    raise ValueError("No data path provided. Set btc_data_path or provide data_path.")
                data_path = self.btc_data_path
                
            self.logger.info(f"Loading data from {data_path}")
            
            # Load data
            self.data = pd.read_csv(data_path)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.set_index('date', inplace=True)
            
            # Clean data
            self.data = self.data.dropna(subset=['close'])
            self.data = self.data[self.data.index >= self.init_date]
            
            # Split into train and test sets
            split_idx = int(len(self.data) * self.train_test_split)
            self.train_data = self.data.iloc[:split_idx]
            self.test_data = self.data.iloc[split_idx:]
            
            self.logger.info(f"Data loaded successfully: {len(self.data)} records")
            self.logger.info(f"Training set: {len(self.train_data)} records")
            self.logger.info(f"Test set: {len(self.test_data)} records")
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def train(self, data_path=None, no_regimes=2):
        """
        Train the Markov Switching Regression model for volatility detection.
        
        Args:
            data_path (str, optional): Path to the data file. If None, uses the initialized btc_data_path.
            no_regimes (int, optional): Number of regimes for the Markov model. Default is 2.
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Load data if not already loaded
            if self.data is None or (data_path is not None and data_path != self.btc_data_path):
                success = self.load_data(data_path)
                if not success:
                    return False
            
            self.logger.info("Training Markov Switching Regression model...")
            
            # Calculate log returns from BTC price
            log_returns_full = np.log(self.data['close']).diff().dropna()
            
            # Calculate split indices
            split_idx = int(len(self.data) * self.train_test_split)
            embargo_size = int(len(self.data) * self.embargo_pct)
            
            # Prepare data for training
            log_returns_full.index = self.data.index[1:]
            train_end_idx = split_idx - embargo_size
            log_returns_train = log_returns_full.iloc[:train_end_idx]
            
            # Fit Markov Switching Regression model
            markov_model = sm.tsa.MarkovRegression(
                log_returns_train.iloc[1:], 
                k_regimes=no_regimes, 
                switching_variance=True, 
                exog=log_returns_train.iloc[:-1]
            )
            
            self.logger.info("Fitting Markov model (this may take a while)...")
            self.msreg_results = markov_model.fit()
            
            # Apply fitted parameters to full dataset
            params = self.msreg_results.params
            
            full_model = sm.tsa.MarkovRegression(
                log_returns_full.iloc[1:],
                k_regimes=no_regimes,
                switching_variance=True,
                exog=log_returns_full.iloc[:-1]
            )
            
            full_results = full_model.smooth(params)
            
            # Extract probability states
            low_var = full_results.smoothed_marginal_probabilities[0]
            high_var = full_results.smoothed_marginal_probabilities[1]
            
            # Create Markov dataframe
            markov_data = pd.concat([low_var, high_var], axis=1)
            markov_data.columns = ['Low_var', 'High_var']
            
            # Mark dataset regions
            markov_data['dataset'] = 1  # Training
            markov_data.iloc[train_end_idx:split_idx, -1] = 0  # Embargo
            markov_data.iloc[split_idx:, -1] = 2  # Testing
            
            self.markov = markov_data
            
            # Update model dictionary
            self.model = {
                'markov': self.markov,
                'msreg_results': self.msreg_results,
                'metadata': self.get_metadata()
            }
            
            self.logger.info("Markov Switching Regression model fitted successfully.")
            self.logger.info(f"Training data: {train_end_idx} points")
            self.logger.info(f"Embargo data: {embargo_size} points")
            self.logger.info(f"Test data: {len(markov_data) - split_idx} points")
            
            return True
        except Exception as e:
            self.logger.error(f"Error training Markov model: {e}")
            return False
    
    def save_model(self, filename=None, directory=None):
        """
        Save the trained model to a file.
        
        Args:
            filename (str, optional): Filename for the model. If None, generates a timestamped name.
            directory (str, optional): Directory to save the model in. If None, uses the default models directory.
            
        Returns:
            str: Path to the saved model file or None if saving failed
        """
        if self.model is None and self.markov is None:
            self.logger.error("No trained model to save")
            return None
        
        try:
            # Create directory if needed
            if directory is None:
                directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), 'models')
            
            os.makedirs(directory, exist_ok=True)
            
            # Generate filename if needed
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"markov_volatility_model_{timestamp}.pkl"
            
            full_path = os.path.join(directory, filename)
            
            # Get metadata
            metadata = self.get_metadata()
            
            # Create or update model dictionary
            model_dict = {
                'markov': self.markov,
                'msreg_results': self.msreg_results,
                'metadata': metadata
            }
            
            # Save model
            with open(full_path, 'wb') as f:
                pickle.dump(model_dict, f)
            
            # Save metadata separately for easy access
            metadata_path = full_path.replace('.pkl', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            self.logger.info(f"Model saved to {full_path}")
            self.logger.info(f"Metadata saved to {metadata_path}")
            
            return full_path
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return None
    
    def get_metadata(self):
        """
        Return the metadata of the model.
        
        Returns:
            dict: Model metadata
        """
        if hasattr(self, 'metadata'):
            return self.metadata
        
        # Create metadata if not available
        metadata = {
            'saved_date': datetime.now().strftime("%Y-%m-%d"),
            'data_path': self.btc_data_path,
            'train_test_split': self.train_test_split,
            'init_date': self.init_date,
            'embargo_pct': self.embargo_pct,
        }
        
        # Add data range information if data is available
        if self.data is not None:
            metadata.update({
                'data_range': {
                    'start': self.data.index[0].strftime("%Y-%m-%d") if not self.data.empty else 'unknown',
                    'end': self.data.index[-1].strftime("%Y-%m-%d") if not self.data.empty else 'unknown'
                },
                'total_records': len(self.data),
            })
        
        # Add train/test information if available
        if self.train_data is not None and self.test_data is not None:
            metadata.update({
                'train_range': {
                    'start': self.train_data.index[0].strftime("%Y-%m-%d") if not self.train_data.empty else 'unknown',
                    'end': self.train_data.index[-1].strftime("%Y-%m-%d") if not self.train_data.empty else 'unknown'
                },
                'test_range': {
                    'start': self.test_data.index[0].strftime("%Y-%m-%d") if not self.test_data.empty else 'unknown',
                    'end': self.test_data.index[-1].strftime("%Y-%m-%d") if not self.test_data.empty else 'unknown'
                },
                'train_records': len(self.train_data),
                'test_records': len(self.test_data),
            })
        
        # Add model status
        metadata['model_fitted'] = self.markov is not None
        metadata['n_regimes'] = 2  # Default for this implementation
        
        return metadata
    
    def get_volatility_state(self, date):
        """
        Get the volatility state (high or low) for a specific date.
        
        Args:
            date (str or datetime): Date to get the volatility state for
            
        Returns:
            str: 'high' or 'low', or None if not available
        """
        # Check cache first
        if date in self.volatility_cache:
            return self.volatility_cache[date]
        
        if self.model is None and self.markov is None:
            self.logger.warning(f"No MarkovVolatility model loaded")
            return None
        
        try:
            # Ensure date is in the correct format
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            date_str = date.strftime('%Y-%m-%d')
            
            # Get the markov data from the model
            markov_data = self.markov if self.markov is not None else self.model.get('markov')
            if markov_data is None:
                self.logger.warning(f"No markov data in model")
                return None
            
            # Create states series (similar to MarkovVolatility.get_volatility_state)
            states = pd.Series('low', index=markov_data.index)
            states[markov_data['High_var'] > 0.5] = 'high'
            
            # Find the closest date if exact date not found
            if pd.to_datetime(date_str) not in states.index:
                # Find the closest date that's available
                dates_array = pd.to_datetime(states.index)
                
                # Guard against being out of range
                if pd.to_datetime(date_str) < dates_array.min():
                    self.logger.debug(f"Date {date_str} before model data range start {dates_array.min()}")
                    closest_date = dates_array.min()
                elif pd.to_datetime(date_str) > dates_array.max():
                    self.logger.debug(f"Date {date_str} after model data range end {dates_array.max()}")
                    closest_date = dates_array.max()
                else:
                    # Find closest date
                    closest_date = dates_array[np.argmin(np.abs(dates_array - pd.to_datetime(date_str)))]
                
                self.logger.debug(f"Exact date {date_str} not found. Using closest date {closest_date}")
                state = states.loc[closest_date]
            else:
                state = states.loc[pd.to_datetime(date_str)]
            
            # Cache result
            self.volatility_cache[date] = state
            return state
            
        except Exception as e:
            self.logger.error(f"Error getting volatility state for {date}: {e}")
            return None
    
    def predict_volatility(self, date):
        """
        Get the volatility state in numeric format (-1 for high, 1 for low).
        This is an alternative interface to get_volatility_state.
        
        Args:
            date (str or datetime): Date to predict volatility for
            
        Returns:
            int: 1 for low volatility, -1 for high volatility, or 0 if not available
        """
        state = self.get_volatility_state(date)
        if state is None:
            return 0
        return 1 if state == 'low' else -1
    
    def clear_cache(self):
        """Clear the volatility state cache."""
        self.volatility_cache = {}
    
    def predict_volatility_states(self, dates):
        """
        Get volatility states for a list of dates. This is useful for batch prediction.
        
        Args:
            dates (list): List of dates to predict volatility for
            
        Returns:
            dict: Dictionary mapping dates to volatility states (1 for low, -1 for high, 0 if not available)
        """
        results = {}
        for date in dates:
            results[date] = self.predict_volatility(date)
        return results
    
    def predict_volatility_for_new_data(self, price_data=None, log_return=None, current_date=None):
        """
        Predict volatility state for new data that was not in the original training set.
        Unlike get_volatility_state which retrieves pre-calculated states from the model,
        this method makes a true prediction for new data.
        
        Args:
            price_data (float, optional): Latest closing price. If provided, log return will be calculated 
                                         using the previous price from the model.
            log_return (float, optional): Pre-calculated log return value. If provided, price_data is ignored.
            current_date (str or datetime, optional): Date for the new data. If None, current date is used.
            
        Returns:
            str: Predicted volatility state ('high' or 'low')
            int: Numeric volatility state (1 for low, -1 for high)
            float: Probability of high volatility regime
        """
        if self.model is None and self.msreg_results is None:
            self.logger.error("No trained model available for prediction")
            return 'low', 1, 0.0  # Default to low volatility
        
        try:
            # Set current date if not provided
            if current_date is None:
                current_date = datetime.now()
            elif isinstance(current_date, str):
                current_date = pd.to_datetime(current_date)
            
            # Get log return if not provided
            if log_return is None and price_data is not None:
                # Get the most recent price from our data
                markov_data = self.markov if self.markov is not None else self.model.get('markov')
                if markov_data is None or self.data is None:
                    self.logger.error("Cannot calculate log return: no historical data available")
                    return 'low', 1, 0.0
                
                # Get latest date in our data
                latest_date = self.data.index[-1]
                previous_price = self.data.loc[latest_date, 'close']
                
                # Calculate log return
                log_return = np.log(price_data / previous_price)
                self.logger.info(f"Calculated log return from price {previous_price} -> {price_data}: {log_return}")
            
            if log_return is None:
                self.logger.error("Either price_data or log_return must be provided")
                return 'low', 1, 0.0
            
            # Get the Markov model results
            msreg_results = self.msreg_results if self.msreg_results is not None else self.model.get('msreg_results')
            if msreg_results is None:
                self.logger.error("No Markov model parameters available")
                return 'low', 1, 0.0
            
            # Create a pd.Series with the log return
            log_return_series = pd.Series([log_return], index=[current_date])
            
            # Create exog variable if the model used it
            # In our implementation we use the previous log return as exog
            # For new prediction, we can use the latest log return from our data
            markov_data = self.markov if self.markov is not None else self.model.get('markov')
            if markov_data is None:
                exog = None
            else:
                # Get the latest log return from our data
                prev_log_return = np.log(self.data['close']).diff().dropna().iloc[-1]
                exog = pd.Series([prev_log_return], index=[current_date])
            
            # Get the parameters from the fitted model
            params = msreg_results.params
            
            # Create a new Markov model for prediction
            k_regimes = 2  # Our model uses 2 regimes (high and low volatility)
            if exog is not None:
                pred_model = sm.tsa.MarkovRegression(
                    log_return_series, 
                    k_regimes=k_regimes, 
                    switching_variance=True,
                    exog=exog
                )
            else:
                pred_model = sm.tsa.MarkovRegression(
                    log_return_series, 
                    k_regimes=k_regimes, 
                    switching_variance=True
                )
            
            # Predict using the parameters from our fitted model
            pred_results = pred_model.smooth(params)
            
            # Extract regime probabilities
            # In our model convention: regime 0 = low volatility, regime 1 = high volatility
            high_vol_prob = pred_results.smoothed_marginal_probabilities[1][0]
            
            # Determine the state
            state = 'high' if high_vol_prob > 0.5 else 'low'
            numeric_state = -1 if state == 'high' else 1
            
            # Log prediction
            self.logger.info(f"Predicted volatility state for {current_date}: {state} (prob={high_vol_prob:.4f})")
            
            # Add to cache
            self.volatility_cache[current_date] = state
            
            return state, numeric_state, high_vol_prob
            
        except Exception as e:
            self.logger.error(f"Error predicting volatility for new data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 'low', 1, 0.0  # Default to low volatility
    
    
    def get_volatility_states_df(self, start_date=None, end_date=None):
        """
        Get a DataFrame of volatility states for a date range.
        
        Args:
            start_date (str or datetime, optional): Start date. If None, uses the earliest date in the model.
            end_date (str or datetime, optional): End date. If None, uses the latest date in the model.
            
        Returns:
            pd.DataFrame: DataFrame with volatility states
        """
        if self.model is None and self.markov is None:
            self.logger.warning("No model data available")
            return pd.DataFrame()
        
        try:
            markov_data = self.markov if self.markov is not None else self.model.get('markov')
            if markov_data is None:
                self.logger.warning("No markov data available")
                return pd.DataFrame()
            
            # Create a copy to avoid modifying the original
            df = markov_data.copy()
            
            # Add state column
            df['state'] = 'low'
            df.loc[df['High_var'] > 0.5, 'state'] = 'high'
            
            # Add numeric state column
            df['state_num'] = 1
            df.loc[df['state'] == 'high', 'state_num'] = -1
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Make sure the date column is named 'date'
            if 'date' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'date'})
            
            # Filter by date range if provided
            if start_date is not None:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date]
                
            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                df = df[df['date'] <= end_date]
            
            return df
        except Exception as e:
            self.logger.error(f"Error creating volatility states DataFrame: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def load_model_by_timestamp(timestamp):
        """
        Load a MarkovVolatility model by its timestamp identifier.
        
        Args:
            timestamp (str): Timestamp identifier in the model filename (e.g., '20250325_144631')
            
        Returns:
            MarkovVolModel: Loaded model instance
        """
        # Construct the model path
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'models')
        
        model_path = os.path.join(models_dir, f"markov_volatility_model_{timestamp}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create an instance and load the model
        instance = MarkovVolModel(model_path=model_path)
        return instance 