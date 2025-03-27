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
        
        # Filter data by lookback period if specified
        if lookback != 'all' and isinstance(lookback, int):
            end_date = df.index.max()
            start_date = end_date - pd.Timedelta(days=lookback)
            df = df[df.index >= start_date]
        
        # Ensure we have trend column
        trend_col = f'Overall_Trend_{trend_type}'
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