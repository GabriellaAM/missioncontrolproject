import pandas as pd
import numpy as np
import logging

class RSICalculator:
    """
    Class for calculating Relative Strength Index (RSI) indicators.
    """
    def __init__(self):
        """Initialize the RSICalculator."""
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def calculate_rsi(prices, window=14):
        """
        Calculate RSI for a series of prices.
        
        Args:
            prices (pd.Series or numpy.ndarray): Series of prices
            window (int, optional): Window size for RSI calculation. Defaults to 14.
            
        Returns:
            float: RSI value (0-100)
        """
        # Convert numpy array to pandas Series if needed
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
            
        if len(prices) < window + 1:
            return np.nan
        
        # Convert to numeric and drop NaN values
        prices = pd.to_numeric(prices, errors='coerce').dropna()
        if len(prices) < window + 1:
            return np.nan
        
        # Calculate price changes
        price_diff = prices.diff(1)
        
        # Create positive and negative gain/loss Series
        gains = price_diff.where(price_diff > 0, 0)
        losses = -price_diff.where(price_diff < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=window, min_periods=1).mean()
        avg_loss = losses.rolling(window=window, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        
        # Handle zero division
        if avg_loss.iloc[-1] == 0:
            return 100
        
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        return rsi
    
    def calculate_smooth_rsi(self, data, price_column, rsi_length=28, roc_length=28):
        """
        Calculate smoothed RSI for OHLC data.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLC data
            price_column (str): Column name for price data
            rsi_length (int, optional): Window size for RSI calculation. Defaults to 28.
            roc_length (int, optional): Window size for RoC calculation. Defaults to 28.
            
        Returns:
            pd.DataFrame: DataFrame with RSI and RoC columns added
        """
        if data.empty:
            self.logger.warning("Empty data provided for RSI calculation.")
            return data
            
        if price_column not in data.columns:
            self.logger.warning(f"Column '{price_column}' not found for RSI calculation.")
            return data
        
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Ensure all price columns are numeric
        df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
        
        # Check if we have enough data
        if len(df) < rsi_length + 1:
            self.logger.warning(f"Not enough data points for RSI calculation. Needed: {rsi_length+1}, Got: {len(df)}")
            return df
        
        # Infer OHLC columns from price column
        high_col, open_col, low_col, close_col = self._infer_ohlc_columns(df, price_column)
        
        # Ensure all needed columns are present
        required_cols = [col for col in [high_col, open_col, low_col, close_col] if col is not None]
        if not all(col in df.columns for col in required_cols):
            self.logger.warning(f"Not all required OHLC columns found for {price_column}.")
            # Try to use price_column for all calculations
            high_col = open_col = low_col = close_col = price_column
        
        # Calculate RSI for each OHLC column (if available)
        rsi_values = {}
        
        for col_name, col in zip(['high', 'open', 'low', 'close'], [high_col, open_col, low_col, close_col]):
            if col in df.columns:
                try:
                    # Convert to Series first to ensure consistent handling
                    df[col] = pd.Series(df[col])
                    rsi_values[col_name] = df[col].rolling(rsi_length).apply(
                        lambda x: self.calculate_rsi(x), raw=False).bfill()
                except Exception as e:
                    self.logger.warning(f"Error calculating RSI for {col}: {e}")
                    rsi_values[col_name] = pd.Series(np.nan, index=df.index)
        
        # Calculate RoC for each OHLC column (if available)
        roc_values = {}
        
        for col_name, col in zip(['high', 'open', 'low', 'close'], [high_col, open_col, low_col, close_col]):
            if col in df.columns:
                try:
                    # Ensure column is a Series
                    df[col] = pd.Series(df[col])
                    roc_values[col_name] = (df[col] - df[col].shift(roc_length)) / df[col].shift(roc_length) * 100
                except Exception as e:
                    self.logger.warning(f"Error calculating RoC for {col}: {e}")
                    roc_values[col_name] = pd.Series(np.nan, index=df.index)
        
        # Smooth RSI and RoC as average of available values
        if rsi_values:
            # Combine available RSI values
            rsi_series = [series for series in rsi_values.values() if not series.isna().all()]
            if rsi_series:
                smooth_rsi = pd.concat(rsi_series, axis=1).mean(axis=1)
            else:
                smooth_rsi = pd.Series(np.nan, index=df.index)
        else:
            smooth_rsi = pd.Series(np.nan, index=df.index)
        
        if roc_values:
            # Combine available RoC values
            roc_series = [series for series in roc_values.values() if not series.isna().all()]
            if roc_series:
                smooth_roc = pd.concat(roc_series, axis=1).mean(axis=1)
            else:
                smooth_roc = pd.Series(np.nan, index=df.index)
        else:
            smooth_roc = pd.Series(np.nan, index=df.index)
        
        # Add RSI and RoC columns
        df[f'RSI_{price_column}'] = smooth_rsi
        df[f'RoC_{price_column}'] = smooth_roc
        
        # Binary signal: +1 for Bull (RSI > 50 AND RoC > 0), -1 for Bear
        df[f'RSI_Signal_{price_column}'] = np.where(
            (smooth_rsi > 50) & (smooth_roc > 0), 
            1, 
            np.where(
                (smooth_rsi <= 50) & (smooth_roc < 0),
                -1,
                0  # Neutral for mixed signals
            )
        )
        
        return df
    
    def _infer_ohlc_columns(self, df, price_column):
        """
        Infer OHLC column names from the price column name.
        
        Args:
            df (pd.DataFrame): DataFrame containing price data
            price_column (str): Name of the price column
            
        Returns:
            tuple: (high, open, low, close) column names
        """
        # If price_column is 'close', try to find other OHLC columns
        if price_column == 'close':
            high_col = 'high' if 'high' in df.columns else None
            open_col = 'open' if 'open' in df.columns else None
            low_col = 'low' if 'low' in df.columns else None
            close_col = price_column
        # If price_column ends with '_btc', try to find corresponding OHLC columns
        elif price_column.endswith('_btc'):
            asset_prefix = price_column[:-4]  # Remove '_btc' suffix
            high_col = f'{asset_prefix}_btc_high' if f'{asset_prefix}_btc_high' in df.columns else None
            open_col = f'{asset_prefix}_btc_open' if f'{asset_prefix}_btc_open' in df.columns else None
            low_col = f'{asset_prefix}_btc_low' if f'{asset_prefix}_btc_low' in df.columns else None
            close_col = price_column
        # For any other price column, use it for all OHLC values
        else:
            self.logger.info(f"Using {price_column} for all OHLC values in RSI calculation.")
            high_col = open_col = low_col = close_col = price_column
        
        return high_col, open_col, low_col, close_col 