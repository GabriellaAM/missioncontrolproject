import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from scipy.stats import skew

class MetricsCalculator:
    """
    Class for calculating various financial metrics and performance indicators.
    """
    def __init__(self):
        """Initialize the MetricsCalculator."""
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def calculate_sharpe_sortino(data, returns_column):
        """
        Calculate Sharpe and Sortino ratios for a returns series.
        
        Args:
            data (pd.DataFrame): DataFrame containing returns data
            returns_column (str): Column name for returns data
            
        Returns:
            tuple: (Sharpe ratio, Sortino ratio, Avg Return, Max Drawdown)
        """
        if returns_column not in data.columns or data.empty:
            return np.nan, np.nan, np.nan, np.nan
        
        # Extract returns and convert to numeric
        returns = pd.to_numeric(data[returns_column], errors='coerce')
        returns = returns.dropna()
        
        if len(returns) < 2:
            return np.nan, np.nan, np.nan, np.nan
        
        # Calculate average return and standard deviation
        avg_return = returns.mean()
        std_dev = returns.std()
        
        # Calculate downside deviation (standard deviation of negative returns only)
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() if len(downside_returns) > 0 else 0.0001
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe = avg_return / std_dev if std_dev != 0 else np.nan
        
        # Calculate Sortino ratio
        sortino = avg_return / downside_dev if downside_dev != 0 else np.nan
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = MetricsCalculator.max_drawdown(cumulative_returns)
        
        return sharpe, sortino, avg_return, max_drawdown
    
    @staticmethod
    def max_drawdown(series):
        """
        Calculate maximum drawdown for a price or cumulative returns series.
        
        Args:
            series (pd.Series): Series of prices or cumulative returns
            
        Returns:
            float: Maximum drawdown as a percentage (0 to 1)
        """
        if len(series) < 2:
            return 0.0
            
        # Calculate running maximum
        running_max = series.cummax()
        
        # Calculate drawdown
        drawdown = (series - running_max) / running_max
        
        # Get maximum drawdown
        max_dd = drawdown.min()
        
        return abs(max_dd) if not np.isnan(max_dd) else 0.0
    
    def calculate_trend_type_metrics(self, data, trend_type, lookback):
        """
        Calculate performance metrics for different trend types.
        
        Args:
            data (pd.DataFrame): DataFrame with price and trend classification data
            trend_type (str): Type of trend to analyze ('USD' or 'BTC')
            lookback (str or int): Lookback period ('all' or number of days)
            
        Returns:
            pd.DataFrame: DataFrame with trend metrics
        """
        if data.empty:
            self.logger.warning("Empty data provided for trend metrics calculation.")
            return pd.DataFrame()
        
        # Create a copy of the data
        df = data.copy()
        
        # Ensure we have date as index
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('date', inplace=True)
        
        # Filter data by lookback period if specified and not 'all'
        if lookback != 'all':
            try:
                # Convert to int in case it's a numeric string
                lookback_days = int(lookback)
                end_date = df.index.max()
                start_date = end_date - pd.Timedelta(days=lookback_days)
                df = df[df.index >= start_date]
            except (ValueError, TypeError):
                # If conversion fails, log a warning
                self.logger.warning(f"Invalid lookback value: {lookback}. Using all available data.")
        
        # Ensure we have trend column
        trend_col = f'overall_trend_{trend_type}'
        if trend_col not in df.columns:
            self.logger.warning(f"Column {trend_col} not found in data.")
            return pd.DataFrame()
        
        # Get price column based on trend type
        if trend_type == 'USD':
            price_col = 'close'
        else:
            # For BTC, try to find the appropriate price column using several methods
            price_col = self._find_btc_price_column(df)
            
            if not price_col:
                self.logger.warning(f"No BTC price column found for {trend_type} trend metrics.")
                return pd.DataFrame()
        
        # Check if price column exists
        if price_col not in df.columns:
            self.logger.warning(f"Price column '{price_col}' not found in data.")
            return pd.DataFrame()
        
        # Calculate daily returns
        df['daily_return'] = df[price_col].pct_change()
        
        # Initialize results DataFrame
        metrics = []
        
        # Calculate metrics for each trend classification (-2 to 2)
        for trend in range(-2, 3):
            trend_data = df[df[trend_col] == trend]
            
            if trend_data.empty:
                # Skip if no data for this trend classification
                continue
            
            # Calculate metrics
            sharpe, sortino, avg_return, max_dd = self.calculate_sharpe_sortino(trend_data, 'daily_return')
            
            # Calculate additional return statistics
            returns = trend_data['daily_return'].dropna()
            median_return = returns.median() if not returns.empty else np.nan
            return_skew = skew(returns.dropna()) if len(returns.dropna()) > 2 else np.nan
            
            # Calculate trend duration statistics
            durations = []
            current_duration = 0
            current_trend = None
            
            for i, row in df.iterrows():
                if row[trend_col] == trend:
                    if current_trend == trend:
                        current_duration += 1
                    else:
                        current_trend = trend
                        current_duration = 1
                elif current_trend == trend:
                    durations.append(current_duration)
                    current_trend = None
                    current_duration = 0
            
            # Add last duration if still in the trend
            if current_trend == trend and current_duration > 0:
                durations.append(current_duration)
            
            # Calculate duration statistics
            avg_duration = np.mean(durations) if durations else np.nan
            max_duration = np.max(durations) if durations else np.nan
            
            # Create metrics dictionary
            metrics_dict = {
                'Trend': trend,
                'Description': self._get_trend_description(trend),
                'Count': len(trend_data),
                'Frequency': len(trend_data) / len(df) if len(df) > 0 else 0,
                'Avg_Return': avg_return,
                'Median_Return': median_return,
                'Return_Skew': return_skew,
                'Sharpe': sharpe,
                'Sortino': sortino,
                'Max_Drawdown': max_dd,
                'Avg_Duration': avg_duration,
                'Max_Duration': max_duration
            }
            
            metrics.append(metrics_dict)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        return metrics_df
    
    def _find_btc_price_column(self, df):
        """
        Find the BTC price column in the DataFrame using various strategies.
        
        Args:
            df (pd.DataFrame): DataFrame to search for BTC price column
            
        Returns:
            str: BTC price column name or None if not found
        """
        # Strategy 1: Look for asset_id column to get the asset name
        if 'asset_id' in df.columns:
            asset_id = df['asset_id'].iloc[0]
            btc_col = f'{asset_id}_btc'
            if btc_col in df.columns:
                return btc_col
        
        # Strategy 2: Look for columns ending with _btc
        btc_cols = [col for col in df.columns if col.endswith('_btc') and not col.endswith('_btc_open') 
                    and not col.endswith('_btc_high') and not col.endswith('_btc_low')]
        if btc_cols:
            return btc_cols[0]
        
        # Strategy 3: Look for columns containing 'btc' but not as '_btc_open', etc.
        btc_cols = [col for col in df.columns if 'btc' in col.lower() 
                   and not any(suffix in col.lower() for suffix in ['_open', '_high', '_low'])]
        if btc_cols:
            return btc_cols[0]
        
        # Strategy 4: Try to extract asset name from column names and guess the BTC column
        asset_cols = [col for col in df.columns if col.endswith('_open') or col.endswith('_high') 
                     or col.endswith('_low') or col.endswith('_close')]
        if asset_cols:
            asset_prefix = asset_cols[0].split('_')[0]
            btc_col = f'{asset_prefix}_btc'
            if btc_col in df.columns:
                return btc_col
        
        # No BTC price column found
        return None
    
    @staticmethod
    def _get_trend_description(trend):
        """
        Get a text description for a trend classification.
        
        Args:
            trend (int): Trend classification (-2 to 2)
            
        Returns:
            str: Text description of the trend
        """
        trend_descriptions = {
            -2: "Strong Bear",
            -1: "Weak Bear",
            0: "Neutral",
            1: "Weak Bull",
            2: "Strong Bull"
        }
        return trend_descriptions.get(trend, "Unknown") 
    
    def calculate_trend_term_metrics(self, data, trend_type, term, lookback):
        """
        Calculate performance metrics for a specific trend term.
        
        Args:
            data (pd.DataFrame): DataFrame with price and trend classification data
            trend_type (str): Type of trend to analyze ('USD' or 'BTC')
            term (str): Term to analyze ('Short Term', 'Medium Term', 'Long Term', 'Overall')
            lookback (str or int): Lookback period ('all' or number of days)
            
        Returns:
            pd.DataFrame: DataFrame with trend metrics for the specified term
        """
        if data.empty:
            self.logger.warning("Empty data provided for trend metrics calculation.")
            return pd.DataFrame()
        
        # Create a copy of the data
        df = data.copy()
        
        # Ensure we have date as index
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('date', inplace=True)
        
        # Filter data by lookback period if specified and not 'all'
        if lookback != 'all':
            try:
                # Convert to int in case it's a numeric string
                lookback_days = int(lookback)
                end_date = df.index.max()
                start_date = end_date - pd.Timedelta(days=lookback_days)
                df = df[df.index >= start_date]
            except (ValueError, TypeError):
                # If conversion fails, log a warning
                self.logger.warning(f"Invalid lookback value: {lookback}. Using all available data.")

        # Map term to snake_case column name format
        term_mapping = {
            'Short Term': 'short_term',
            'Medium Term': 'medium_term',
            'Long Term': 'long_term',
            'Overall': 'overall'
        }
        snake_case_term = term_mapping.get(term, term.lower().replace(' ', '_'))
        
        # Determine trend column using snake_case format
        trend_col = f'{snake_case_term}_trend_{trend_type}'
            
        if trend_col not in df.columns:
            self.logger.warning(f"Column {trend_col} not found in data.")
            return pd.DataFrame()
        
        # Get price column based on trend type
        if trend_type == 'USD':
            price_col = 'close'
        else:
            # For BTC, try to find the appropriate price column using several methods
            price_col = self._find_btc_price_column(df)
            
            if not price_col:
                self.logger.warning(f"No BTC price column found for {trend_type} trend metrics.")
                return pd.DataFrame()
        
        # Check if price column exists
        if price_col not in df.columns:
            self.logger.warning(f"Price column '{price_col}' not found in data.")
            return pd.DataFrame()
        
        # Calculate daily returns
        df['daily_return'] = df[price_col].pct_change()
        
        # Initialize results DataFrame
        metrics = []
        
        # Calculate metrics for each trend classification (-2 to 2)
        for trend in range(-2, 3):
            trend_data = df[df[trend_col] == trend]
            
            if trend_data.empty:
                # Skip if no data for this trend classification
                continue
            
            # Calculate metrics
            sharpe, sortino, avg_return, max_dd = self.calculate_sharpe_sortino(trend_data, 'daily_return')
            
            # Calculate additional return statistics
            returns = trend_data['daily_return'].dropna()
            median_return = returns.median() if not returns.empty else np.nan
            return_skew = skew(returns.dropna()) if len(returns.dropna()) > 2 else np.nan
            
            # Calculate trend duration statistics
            durations = []
            current_duration = 0
            current_trend = None
            
            for i, row in df.iterrows():
                if row[trend_col] == trend:
                    if current_trend == trend:
                        current_duration += 1
                    else:
                        current_trend = trend
                        current_duration = 1
                elif current_trend == trend:
                    durations.append(current_duration)
                    current_trend = None
                    current_duration = 0
            
            # Add last duration if still in the trend
            if current_trend == trend and current_duration > 0:
                durations.append(current_duration)
            
            # Calculate duration statistics
            avg_duration = np.mean(durations) if durations else np.nan
            max_duration = np.max(durations) if durations else np.nan
            
            # Create metrics dictionary
            metrics_dict = {
                'Trend': trend,
                'Description': self._get_trend_description(trend),
                'Count': len(trend_data),
                'Frequency': len(trend_data) / len(df) if len(df) > 0 else 0,
                'Avg_Return': avg_return,
                'Median_Return': median_return,
                'Return_Skew': return_skew,
                'Sharpe': sharpe,
                'Sortino': sortino,
                'Max_Drawdown': max_dd,
                'Avg_Duration': avg_duration,
                'Max_Duration': max_duration
            }
            
            metrics.append(metrics_dict)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        return metrics_df
    
    def calculate_consolidated_trend_metrics(self, data, trend_type, lookback):
        """
        Calculate consolidated performance metrics for all trend terms.
        
        Args:
            data (pd.DataFrame): DataFrame with price and trend classification data
            trend_type (str): Type of trend to analyze ('USD' or 'BTC')
            lookback (str or int): Lookback period ('all' or number of days)
            
        Returns:
            pd.DataFrame: DataFrame with trend metrics for all terms, including a 'Term' column
        """
        if data.empty:
            self.logger.warning("Empty data provided for trend metrics calculation.")
            return pd.DataFrame()
        
        # Calculate metrics for each term and combine them
        all_metrics = []
        
        # Define term mapping to match the new column naming scheme
        term_mapping = {
            'Short Term': 'short_term',
            'Medium Term': 'medium_term',
            'Long Term': 'long_term',
            'Overall': 'overall'
        }
        
        for display_term, snake_case_term in term_mapping.items():
            # Check if the corresponding column exists
            col_name = f'{snake_case_term}_trend_{trend_type}'
            
            if col_name in data.columns:
                term_metrics = self.calculate_trend_term_metrics(data, trend_type, display_term, lookback)
                
                if not term_metrics.empty:
                    # Add term column
                    term_metrics['Term'] = display_term
                    all_metrics.append(term_metrics)
        
        if not all_metrics:
            self.logger.warning(f"No trend metrics calculated for {trend_type}.")
            return pd.DataFrame()
        
        # Concatenate all metrics
        consolidated_metrics = pd.concat(all_metrics, ignore_index=True)
        
        # Reorder columns to put Term near the beginning
        cols = consolidated_metrics.columns.tolist()
        # Remove 'Term' from its current position
        cols.remove('Term')
        # Insert 'Term' after 'Description'
        description_index = cols.index('Description')
        cols.insert(description_index + 1, 'Term')
        
        # Apply the new column order
        consolidated_metrics = consolidated_metrics[cols]
        
        return consolidated_metrics
        
    def calculate_all_metrics(self, data, asset_id, trend_type, lookback):
        """
        Calculate all metrics and transition matrices for an asset with a single call.
        
        Args:
            data (pd.DataFrame): DataFrame with price and trend data
            asset_id (str): ID of the asset
            trend_type (str): Type of trend to analyze ('USD' or 'BTC')
            lookback (str or int): Lookback period ('all' or number of days)
            
        Returns:
            tuple: (trend_metrics, transition_matrices, consolidated_metrics)
        """
        if data.empty:
            self.logger.warning(f"Empty data provided for all metrics calculation for {asset_id}.")
            return pd.DataFrame(), {}, pd.DataFrame()
            
        # Check if trend column exists
        trend_col = f'overall_trend_{trend_type}'
        if trend_col not in data.columns:
            self.logger.warning(f"Column {trend_col} not found in data for {asset_id}. Available columns: {data.columns.tolist()}")
            return pd.DataFrame(), {}, pd.DataFrame()
            
        # Calculate trend type metrics
        try:
            trend_metrics = self.calculate_trend_type_metrics(data, trend_type, lookback)
        except Exception as e:
            self.logger.error(f"Error calculating trend metrics for {asset_id} {trend_type}: {str(e)}")
            trend_metrics = pd.DataFrame()
            
        # Calculate consolidated metrics
        try:
            consolidated_metrics = self.calculate_consolidated_trend_metrics(data, trend_type, lookback)
        except Exception as e:
            self.logger.error(f"Error calculating consolidated metrics for {asset_id} {trend_type}: {str(e)}")
            consolidated_metrics = pd.DataFrame()
            
        # Calculate transition matrices for each term
        transition_matrices = {}
        
        # Define consistent term mapping using snake_case format
        term_mapping = {
            'short_term': 'short_term',
            'medium_term': 'medium_term', 
            'long_term': 'long_term',
            'overall': 'overall'
        }
        
        # Log available trend columns to help with debugging
        trend_cols = [col for col in data.columns if '_trend_' in col and trend_type in col]
        self.logger.info(f"Available trend columns for {asset_id} {trend_type}: {trend_cols}")
        
        # Log data shape and first few rows for debugging
        self.logger.debug(f"Data shape for {asset_id}: {data.shape}")
        if not data.empty:
            self.logger.debug(f"First row date for {asset_id}: {data['date'].iloc[0] if 'date' in data.columns else 'No date column'}")
        
        for snake_case_term in term_mapping.keys():
            # For each term, calculate transition probabilities
            term_col = f'{snake_case_term}_trend_{trend_type}'
            self.logger.info(f"Checking for column {term_col} for {asset_id}")
            
            if term_col in data.columns:
                try:
                    self.logger.info(f"Calculating transition matrix for {asset_id} {term_col}")
                    # Create a copy of the required data
                    transition_df = data[[term_col]].copy()
                    if 'date' in data.columns:
                        transition_df['date'] = data['date']
                        
                    # Count non-null values
                    non_null_count = transition_df[term_col].count()
                    self.logger.info(f"Non-null values in {term_col}: {non_null_count} out of {len(transition_df)}")
                    
                    # Print first few values for debugging
                    if non_null_count > 0:
                        first_values = transition_df[term_col].dropna().head(5).tolist()
                        self.logger.debug(f"First few {term_col} values: {first_values}")
                    else:
                        self.logger.warning(f"No non-null values found in {term_col} for {asset_id}")
                    
                    # Rename column for consistency with what the Markov analyzer expects
                    temp_col_name = f'overall_trend_{trend_type}'
                    transition_df[temp_col_name] = transition_df[term_col]
                    
                    # We would normally call markov_analyzer here, but since it's not
                    # directly accessible in the metrics calculator, we'll return the
                    # prepared dataframe for the caller to use
                    transition_matrices[snake_case_term] = transition_df
                    self.logger.info(f"Successfully prepared transition data for {asset_id} {term_col}")
                except Exception as e:
                    self.logger.error(f"Error preparing transition data for {asset_id} {term_col}: {str(e)}")
                    self.logger.exception(e)
            else:
                self.logger.warning(f"Column {term_col} not found in data for {asset_id}. Available columns: {data.columns.tolist()}")
        
        # Log summary of prepared transition matrices
        self.logger.info(f"Prepared {len(transition_matrices)} transition matrices for {asset_id} {trend_type}: {list(transition_matrices.keys())}")
        
        return trend_metrics, transition_matrices, consolidated_metrics

    def calculate_all_transition_metrics(self, data, asset_id, trend_type, lookback_days=30):
        """
        Calculate transition metrics for all trend terms (short, medium, long term and overall).
        
        Args:
            data (pd.DataFrame): Data with trend classifications
            asset_id (str): Asset ID for logging
            trend_type (str): Type of trend (USD or BTC)
            lookback_days (int, optional): Number of days to look back
            
        Returns:
            dict: Dict of DataFrames with transition data for each term
        """
        # Prepare trend column name
        trend_col = f'overall_trend_{trend_type}'
        
        if trend_col not in data.columns:
            self.logger.warning(f"Column {trend_col} not found in data, cannot calculate transition metrics")
            return {}
            
        # Assign trend terms mapping using snake_case format
        trend_terms = {
            'short_term': 'short_term',
            'medium_term': 'medium_term',
            'long_term': 'long_term',
            'overall': 'overall'
        }
        
        # Apply lookback filter if needed
        if lookback_days != 'all' and lookback_days > 0:
            if 'date' in data.columns:
                last_date = data['date'].max()
                start_date = last_date - pd.Timedelta(days=lookback_days)
                data = data[data['date'] >= start_date].copy()
            else:
                # If no date column, assume the index is a date
                if isinstance(data.index, pd.DatetimeIndex):
                    last_date = data.index.max()
                    start_date = last_date - pd.Timedelta(days=lookback_days)
                    data = data[data.index >= start_date].copy()
                else:
                    # If no date information, take the last n rows
                    data = data.iloc[-min(lookback_days, len(data)):].copy()
        
        # Initialize result dictionary
        transition_dfs = {}
        
        # Calculate transition metrics for each term
        for storage_term, data_term in trend_terms.items():
            # For each term, calculate transition probabilities
            term_col = f'{data_term}_trend_{trend_type}'
            if term_col in data.columns:
                try:
                    # Create a copy of the data for this term
                    transition_df = data[['date', term_col]].copy() if 'date' in data.columns else data[[term_col]].copy()
                    
                    # Rename column for consistency with what the Markov analyzer expects
                    temp_col_name = f'overall_trend_{trend_type}'
                    transition_df[temp_col_name] = transition_df[term_col]
                    
                    # Store in result dictionary
                    transition_dfs[storage_term] = transition_df
                except Exception as e:
                    self.logger.error(f"Error calculating transition metrics for {asset_id} {trend_type} {storage_term}: {str(e)}")
            else:
                self.logger.warning(f"Column {term_col} not found in data for {asset_id} {trend_type} {storage_term}")
        
        return transition_dfs 

    @staticmethod
    def get_cumulative_returns_column(df):
        """
        Get the appropriate cumulative returns column name.
        
        Args:
            df (pd.DataFrame): DataFrame to check for column names
            
        Returns:
            str: The appropriate cumulative returns column name
        """
        if 'strategy_cum_returns' in df.columns:
            return 'strategy_cum_returns'
        elif 'cum_returns' in df.columns:
            return 'cum_returns'
        elif 'strategy_cum_return' in df.columns:
            return 'strategy_cum_return'
        elif 'cum_return' in df.columns:
            return 'cum_return'
        return None 