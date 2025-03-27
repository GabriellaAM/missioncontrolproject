import pandas as pd
import numpy as np
import logging

class MovingAverageCalculator:
    """
    Class for calculating various types of moving averages and related indicators.
    """
    def __init__(self):
        """
        Initialize the MovingAverageCalculator with predefined MA periods grouped by term.
        """
        self.logger = logging.getLogger(__name__)
        
        # Define moving average periods categorized by term
        self.ma_periods = {
            'Short Term': [3, 5, 7, 14],
            'Medium Term': [21, 30, 45, 63],
            'Long Term': [84, 100, 120, 150, 200, 252, 365]
        }
    
    def calculate_all_mas(self, data, price_column):
        """
        Calculate all moving averages for a given price column.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data
            price_column (str): Column name for price data
            
        Returns:
            tuple: (DataFrame with MAs added, dict of MA columns by term, dict of RoC columns by term)
        """
        if data.empty:
            self.logger.warning("Empty data provided for MA calculation.")
            return data, {}, {}
            
        if price_column not in data.columns:
            self.logger.warning(f"Column '{price_column}' not found in data.")
            return data, {}, {}
        
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Ensure price column is numeric and is a Series
        if isinstance(df[price_column], np.ndarray):
            df[price_column] = pd.Series(df[price_column], index=df.index)
        df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
        
        if df[price_column].isna().all():
            self.logger.warning(f"No valid numeric data in column '{price_column}'.")
            return df, {}, {}
            
        # Check if we have enough data points for at least some MAs
        if len(df) < 3:  # Need at least 3 points for the shortest MA
            self.logger.warning(f"Not enough data points ({len(df)}) for MA calculation in {price_column}.")
            return df, {}, {}
        
        # Dictionaries to store MA and RoC columns by term
        ma_columns = {term: [] for term in self.ma_periods.keys()}
        roc_columns = {term: [] for term in self.ma_periods.keys()}
        
        # Calculate MAs and RoCs for each term and period
        for term, periods in self.ma_periods.items():
            for period in periods:
                if len(df) >= period:
                    # Calculate MA
                    ma_col = f"MA{period}_{price_column}"
                    df[ma_col] = df[price_column].rolling(window=period).mean()
                    
                    # Only add to result if we have at least one valid MA value
                    if not df[ma_col].isna().all():
                        ma_columns[term].append(ma_col)
                        
                        # Calculate Rate of Change (RoC)
                        roc_col = f"RoC{period}_{price_column}"
                        df[roc_col] = (df[price_column] / df[price_column].shift(period) - 1) * 100
                        
                        # Only add to result if we have at least one valid RoC value
                        if not df[roc_col].isna().all():
                            roc_columns[term].append(roc_col)
                else:
                    self.logger.debug(f"Not enough data points ({len(df)}) to calculate MA{period} for {price_column}.")
        
        # Check if we have any valid MA/RoC columns
        has_valid_columns = any(len(cols) > 0 for cols in ma_columns.values())
        if not has_valid_columns:
            self.logger.warning(f"No valid MA/RoC columns could be calculated for {price_column}.")
            return df, {}, {}
            
        return df, ma_columns, roc_columns
    
    def calculate_ma_crossovers(self, data, price_column, fast_period=20, slow_period=50):
        """
        Calculate moving average crossovers for trend detection.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data
            price_column (str): Column name for price data
            fast_period (int): Period for fast moving average
            slow_period (int): Period for slow moving average
            
        Returns:
            pd.DataFrame: DataFrame with crossover signals added
        """
        if data.empty or price_column not in data.columns:
            self.logger.warning(f"Column '{price_column}' not found or data is empty.")
            return data
        
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Ensure price column is numeric and is a Series
        if isinstance(df[price_column], np.ndarray):
            df[price_column] = pd.Series(df[price_column], index=df.index)
        df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
        
        # Calculate fast and slow MAs
        fast_ma = f"MA{fast_period}_{price_column}"
        slow_ma = f"MA{slow_period}_{price_column}"
        
        if fast_ma not in df.columns:
            df[fast_ma] = df[price_column].rolling(window=fast_period).mean()
        
        if slow_ma not in df.columns:
            df[slow_ma] = df[price_column].rolling(window=slow_period).mean()
        
        # Calculate crossover signal
        df[f"CrossOver_{fast_period}_{slow_period}_{price_column}"] = np.where(
            df[fast_ma] > df[slow_ma], 1, # Fast MA above Slow MA = Bullish
            np.where(df[fast_ma] < df[slow_ma], -1, 0) # Fast MA below Slow MA = Bearish, Equal = Neutral
        )
        
        return df 