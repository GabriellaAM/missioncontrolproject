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
        Calculate RSI for a series of prices using the same method as TradingView (RMA).
        
        Args:
            prices (pd.Series or numpy.ndarray): Series of prices
            window (int, optional): Window size for RSI calculation. Defaults to 14.
            
        Returns:
            pd.Series: RSI values for the entire price series (with NaN for the first window elements)
        """
        # Convert numpy array to pandas Series if needed
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
            
        # Create a result Series filled with NaN values
        rsi_series = pd.Series(np.nan, index=prices.index)
        
        if len(prices) < window + 1:
            print(f"WARNING: Insufficient data for RSI calculation. Need at least {window + 1} points, but only have {len(prices)}")
            return rsi_series
        
        # Convert to numeric
        prices = pd.to_numeric(prices, errors='coerce')
        
        # Return NaN Series if we don't have enough data after handling NaNs
        if prices.count() < window + 1:
            print(f"WARNING: Not enough valid data points for RSI calculation after converting to numeric. Needed: {window+1}, Got: {prices.count()}")
            return rsi_series
        
        # Calculate price changes
        price_diff = prices.diff(1)
        
        # Create gains and losses series
        gains = price_diff.where(price_diff > 0, 0.0)
        losses = -price_diff.where(price_diff < 0, 0.0)
        
        # Calculate RMA (Relative Moving Average) - this is what TradingView uses
        # RMA is equivalent to EMA with alpha=1/length
        # First value is a simple average for the first window periods
        avg_gain = np.nan_to_num(gains.iloc[:window].mean())
        avg_loss = np.nan_to_num(losses.iloc[:window].mean())
        
        # Calculate RMA for each point after the initial window
        rsi_values = []
        for i in range(window, len(prices)):
            avg_gain = (avg_gain * (window - 1) + gains.iloc[i]) / window
            avg_loss = (avg_loss * (window - 1) + losses.iloc[i]) / window
            
            if avg_loss == 0:
                rsi = 100.0
            elif avg_gain == 0:
                rsi = 0.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
                
            rsi_values.append(rsi)
        
        # Set RSI values for data after the initial window
        rsi_series.iloc[window:] = rsi_values
        
        return rsi_series
    
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
        self.logger.debug(f"Calculating RSI for {price_column} with length {rsi_length}")
        
        if data.empty:
            self.logger.warning("Empty data provided for RSI calculation.")
            raise ValueError("Empty data provided for RSI calculation.")
        
        if price_column not in data.columns:
            self.logger.warning(f"Column '{price_column}' not found for RSI calculation. Available columns: {data.columns.tolist()}")
            raise ValueError(f"Column '{price_column}' not found for RSI calculation.")
        
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Ensure all price columns are numeric
        df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
        
        # Infer OHLC columns from price column
        high_col, open_col, low_col, close_col = self._infer_ohlc_columns(df, price_column)
        
        # Ensure all needed columns are present
        required_cols = [col for col in [high_col, open_col, low_col, close_col] if col is not None]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            self.logger.warning(f"Missing OHLC columns for RSI calculation: {missing_cols}")
            raise ValueError(f"Missing required OHLC columns for RSI calculation: {missing_cols}")
        
        # Calculate RSI for each OHLC column (if available)
        rsi_values = {}
        
        for col_name, col in zip(['high', 'open', 'low', 'close'], [high_col, open_col, low_col, close_col]):
            if col in df.columns:
                try:
                    # Convert to Series first to ensure consistent handling
                    df[col] = pd.Series(df[col])
                    
                    # Calculate RSI directly (returns full series)
                    rsi_values[col_name] = self.calculate_rsi(df[col], window=rsi_length)
                except Exception as e:
                    self.logger.error(f"Error calculating RSI for {col}: {e}")
                    raise RuntimeError(f"Error calculating RSI for {col}: {e}")
        
        # Calculate RoC for each OHLC column (if available)
        roc_values = {}
        
        for col_name, col in zip(['high', 'open', 'low', 'close'], [high_col, open_col, low_col, close_col]):
            if col in df.columns:
                try:
                    # Ensure column is a Series
                    df[col] = pd.Series(df[col])
                    roc_values[col_name] = (df[col] - df[col].shift(roc_length)) / df[col].shift(roc_length) * 100
                except Exception as e:
                    self.logger.error(f"Error calculating RoC for {col}: {e}")
                    raise RuntimeError(f"Error calculating RoC for {col}: {e}")
        
        # Smooth RSI and RoC as average of available values
        if rsi_values:
            # Combine available RSI values
            rsi_series = [series for series in rsi_values.values() if not series.isna().all()]
            if rsi_series:
                smooth_rsi = pd.concat(rsi_series, axis=1).mean(axis=1)
            else:
                self.logger.warning("No valid RSI series found, all values are NaN")
                smooth_rsi = pd.Series(np.nan, index=df.index)
        else:
            self.logger.warning("No RSI values calculated")
            smooth_rsi = pd.Series(np.nan, index=df.index)
        
        if roc_values:
            # Combine available RoC values
            roc_series = [series for series in roc_values.values() if not series.isna().all()]
            if roc_series:
                smooth_roc = pd.concat(roc_series, axis=1).mean(axis=1)
            else:
                self.logger.warning("No valid RoC series found, all values are NaN")
                smooth_roc = pd.Series(np.nan, index=df.index)
        else:
            self.logger.warning("No RoC values calculated")
            smooth_roc = pd.Series(np.nan, index=df.index)
        
        # Add RSI and RoC columns
        df[f'RSI_{price_column}'] = smooth_rsi
        df[f'RoC_{price_column}'] = smooth_roc
        
        # Binary signal: +1 for Bull (RSI > 50 AND RoC > 0), -1 for Bear
        df[f'RSI_Signal_{price_column}'] = np.where(
            pd.isna(smooth_rsi) | pd.isna(smooth_roc),
            np.nan,  # If either RSI or RoC is NaN, signal should be NaN
            np.where(
                (smooth_rsi > 50) & (smooth_roc > 0), 
                1, 
                np.where(
                    (smooth_rsi <= 50) & (smooth_roc < 0),
                    -1,
                    0  # Neutral for mixed signals
                )
            )
        )
        
        self.logger.debug(f"RSI calculation complete for {price_column}")
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
        # For any other price column, set non-close columns to None
        else:
            self.logger.warning(f"Unable to infer OHLC columns for {price_column}. Will use only close price.")
            high_col = None
            open_col = None
            low_col = None
            close_col = price_column
        
        # Log the inferred columns
        self.logger.debug(f"Inferred OHLC columns for {price_column}: high={high_col}, open={open_col}, low={low_col}, close={close_col}")
        
        return high_col, open_col, low_col, close_col 