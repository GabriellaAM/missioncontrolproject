import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

class MarkovAnalyzer:
    """
    Class for analyzing market states and transitions using Markov models.
    """
    def __init__(self, markov_model=None):
        """
        Initialize the MarkovAnalyzer.
        
        Args:
            markov_model: Optional external Markov model to use for volatility analysis
        """
        self.logger = logging.getLogger(__name__)
        self.markov_model = markov_model
        self.volatility_cache = {}  # Cache for volatility states
    
    def calculate_transition_probabilities(self, data, trend_type, lookback_days=None):
        """
        Calculate transition probabilities between market states.
        
        Args:
            data (pd.DataFrame): DataFrame with trend classification data
            trend_type (str): Type of trend to analyze ('USD' or 'BTC')
            lookback_days (int or str, optional): Number of days to look back, or 'all' to use all data
            
        Returns:
            pd.DataFrame: Transition probability matrix
        """
        if data.empty:
            self.logger.warning("Empty data provided for transition probability calculation.")
            return self._create_default_transition_matrix()
        
        # Log input data details
        self.logger.info(f"Calculating transitions for {trend_type} data with shape {data.shape}")
        self.logger.debug(f"Columns in input data: {data.columns.tolist()}")
        
        # Create a copy of the data
        df = data.copy()
        
        # Ensure we have date as index
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('date', inplace=True)
        
        # Filter data by lookback period if specified and not 'all'
        if lookback_days is not None and lookback_days != 'all':
            try:
                # Convert to int in case it's a numeric string
                lookback_days_int = int(lookback_days)
                end_date = df.index.max()
                start_date = end_date - pd.Timedelta(days=lookback_days_int)
                df = df[df.index >= start_date]
                self.logger.debug(f"Filtered data by lookback period: {len(df)} rows from {start_date} to {end_date}")
            except (ValueError, TypeError):
                # If conversion fails and it's not 'all', log a warning
                if lookback_days != 'all':
                    self.logger.warning(f"Invalid lookback_days value: {lookback_days}. Using all available data.")
        
        # Set trend column name
        trend_col = f'overall_trend_{trend_type}'
        
        # Check for the trend column
        if trend_col not in df.columns:
            self.logger.warning(f"Column {trend_col} not found in transition data. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Check for non-null values
        non_null_count = df[trend_col].count()
        self.logger.info(f"Found {non_null_count} non-null values out of {len(df)} for {trend_col}")
        
        # Get trend values and drop NaN
        trends = df[trend_col].dropna()
        
        if len(trends) < 2:
            self.logger.warning(f"Not enough trend data points for transition probability calculation. Only {len(trends)} valid points found.")
            return self._create_default_transition_matrix()
        
        # Log the first few trend values for debugging
        self.logger.debug(f"First few trend values: {trends.head().tolist()}")
        
        # Get unique trend values
        unique_trends = sorted(trends.unique())
        self.logger.info(f"Unique trend values: {unique_trends}")
        n_states = len(unique_trends)
        
        # Create transition count matrix
        transition_counts = np.zeros((n_states, n_states))
        
        # Count transitions
        for i in range(len(trends) - 1):
            from_idx = np.where(unique_trends == trends.iloc[i])[0][0]
            to_idx = np.where(unique_trends == trends.iloc[i + 1])[0][0]
            transition_counts[from_idx, to_idx] += 1
        
        # Calculate probabilities
        transition_probs = np.zeros((n_states, n_states))
        for i in range(n_states):
            row_sum = np.sum(transition_counts[i, :])
            if row_sum > 0:
                transition_probs[i, :] = transition_counts[i, :] / row_sum
                
        # Log the transition counts
        self.logger.debug(f"Transition counts matrix:\n{transition_counts}")
        
        # Create DataFrame for the transition matrix
        transition_matrix = pd.DataFrame(
            transition_probs,
            index=[f"From {t}" for t in unique_trends],
            columns=[f"To {t}" for t in unique_trends]
        )
        
        self.logger.info(f"Successfully created transition matrix for {trend_type} with shape {transition_matrix.shape}")
        
        return transition_matrix
    
    def _create_default_transition_matrix(self):
        """
        Create a default transition matrix when there's not enough data.
        
        Returns:
            pd.DataFrame: Default transition matrix
        """
        # Create a simple 5x5 matrix for the 5 trend states (-2 to 2)
        default_trends = [-2, -1, 0, 1, 2]
        n_states = len(default_trends)
        
        # Create a matrix with equal transition probabilities
        transition_probs = np.ones((n_states, n_states)) / n_states
        
        # Create DataFrame for the transition matrix
        transition_matrix = pd.DataFrame(
            transition_probs,
            index=[f"From {self._get_trend_name(t)}" for t in default_trends],
            columns=[f"To {self._get_trend_name(t)}" for t in default_trends]
        )
        
        return transition_matrix
    
    def _get_trend_name(self, trend_value):
        """Get the name of a trend value."""
        trend_names = {
            -2: "Strong Bear",
            -1: "Weak Bear",
            0: "Neutral",
            1: "Weak Bull",
            2: "Strong Bull"
        }
        return trend_names.get(trend_value, f"Trend {trend_value}")
    
    def get_volatility_state(self, date):
        """
        Get the volatility state for a specific date using the Markov model.
        
        Args:
            date (datetime): Date to get the volatility state for
            
        Returns:
            int: Volatility state (0 for low, 1 for high), or None if not available
        """
        if self.markov_model is None:
            self.logger.warning("No Markov model available for volatility state prediction.")
            return None
        
        # Check cache first
        if date in self.volatility_cache:
            return self.volatility_cache[date]
        
        try:
            # Convert to pandas datetime if needed
            if not isinstance(date, pd.Timestamp):
                date = pd.Timestamp(date)
            
            # Get the volatility state from the model
            vol_state = self.markov_model.predict_volatility(date)
            
            # Cache the result
            self.volatility_cache[date] = vol_state
            
            return vol_state
        except Exception as e:
            self.logger.error(f"Error getting volatility state for {date}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the volatility state cache."""
        self.volatility_cache = {} 