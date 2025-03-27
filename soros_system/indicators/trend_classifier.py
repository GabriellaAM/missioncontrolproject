import pandas as pd
import numpy as np
import logging

class TrendClassifier:
    """
    Class for classifying market trends based on various technical indicators.
    """
    def __init__(self, ma_calculator):
        """
        Initialize the TrendClassifier.
        
        Args:
            ma_calculator: An instance of MovingAverageCalculator
        """
        self.logger = logging.getLogger(__name__)
        self.ma_calculator = ma_calculator
        
    def calculate_indicators(self, data, price_column, asset_id, suffix=''):
        """
        Calculate technical indicators for trend classification.
        
        Args:
            data (pd.DataFrame): DataFrame containing price data
            price_column (str): Column name for price data
            asset_id (str): Asset ID for logging
            suffix (str, optional): Suffix to add to column names
            
        Returns:
            tuple: (DataFrame with indicators, MA columns by term, RoC columns by term)
        """
        if price_column not in data.columns or data.empty:
            self.logger.warning(f"Column '{price_column}' not found for {asset_id} or data is empty.")
            return data, {}, {}
        
        # Ensure price column is numeric and convert to Series if needed
        if isinstance(data[price_column], np.ndarray):
            data[price_column] = pd.Series(data[price_column], index=data.index)
        data[price_column] = pd.to_numeric(data[price_column], errors='coerce')
        
        # Calculate moving averages
        data, ma_columns, roc_columns = self.ma_calculator.calculate_all_mas(data, price_column)
        
        return data, ma_columns, roc_columns
    
    def classify_trend(self, data, asset_id, price_type='USD'):
        """
        Classify trend for an asset based on technical indicators.
        
        Args:
            data (pd.DataFrame): DataFrame with technical indicators
            asset_id (str): Asset ID for logging
            price_type (str, optional): Type of price to analyze ('USD' or 'BTC'). Defaults to 'USD'.
            
        Returns:
            pd.DataFrame: DataFrame with trend classifications added
        """
        if data.empty:
            self.logger.warning(f"Empty data provided for trend classification for {asset_id}.")
            return data
        
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Determine which price column to use
        price_col = 'close' if price_type == 'USD' else f'{asset_id}_btc'
        suffix = f'_{price_type}'
        
        self.logger.info(f"Processing {price_type} trend data for {asset_id}")
        
        # Calculate indicators if not already present
        if not any(col.startswith('MA') and col.endswith(price_col) for col in df.columns):
            df, ma_columns, roc_columns = self.calculate_indicators(df, price_col, asset_id, suffix)
        else:
            # Extract existing MA and RoC columns
            ma_columns = {}
            roc_columns = {}
            
            for term in ['Short Term', 'Medium Term', 'Long Term']:
                ma_columns[term] = [col for col in df.columns if col.startswith('MA') and col.endswith(price_col)]
                roc_columns[term] = [col for col in df.columns if col.startswith('RoC') and col.endswith(price_col)]
        
        # Create columns for trend classifications
        for term in ['Short Term', 'Medium Term', 'Long Term']:
            df[f'Trend_{term}{suffix}'] = np.nan
            
            ma_cols = ma_columns.get(term, [])
            ma_cols_filtered = [col for col in ma_cols if col in df.columns]
            roc_cols = roc_columns.get(term, [])
            roc_cols_filtered = [col for col in roc_cols if col in df.columns]
            
            if not ma_cols_filtered or not roc_cols_filtered:
                self.logger.warning(f"Not enough MA or RoC columns for {term}{suffix} trend classification")
                continue
            
            # For each row in the DataFrame
            for i in range(len(df)):
                row = df.iloc[i]
                
                # Get MA and RoC values for this row
                try:
                    ma_values = [row[col] for col in ma_cols_filtered if pd.notna(row[col])]
                    roc_values = [row[col] for col in roc_cols_filtered if pd.notna(row[col])]
                except KeyError as e:
                    self.logger.warning(f"Column access error in trend classification: {e}")
                    continue
                
                if not ma_values or not roc_values:
                    continue
                
                # Calculate MA trend: avg MA above price = bearish (-1), below = bullish (+1)
                price_col = ma_cols_filtered[0].split('_', 1)[1]  # Extract price column name from MA column
                
                # Make sure we're getting the value from the correct row
                if price_col not in row:
                    self.logger.warning(f"Price column {price_col} not found in row")
                    continue
                    
                price_val = row[price_col]
                if pd.isna(price_val):
                    continue
                    
                avg_ma = np.mean(ma_values)
                ma_trend = 1 if price_val > avg_ma else -1
                
                # Calculate RoC trend: positive avg RoC = bullish (+1), negative = bearish (-1)
                avg_roc = np.mean(roc_values)
                roc_trend = 1 if avg_roc > 0 else -1
                
                # Combined trend: bullish only if both MA and RoC trends agree
                combined_trend = ma_trend if ma_trend == roc_trend else 0
                
                # Set trend value for this row
                df.loc[df.index[i], f'Trend_{term}{suffix}'] = combined_trend
        
        # Create overall trend classification
        self._calculate_overall_trend(df, price_type)
        
        # Calculate trend coverage
        trend_columns = [col for col in df.columns if col.startswith('Trend_') and col.endswith(suffix)]
        trend_coverage = (df[trend_columns].count().sum() / (len(df) * len(trend_columns))) * 100
        self.logger.info(f"{asset_id} {price_type} trend coverage: {trend_coverage:.2f}%")
        
        return df
    
    def _calculate_overall_trend(self, df, price_type='USD'):
        """
        Calculate overall trend based on short, medium, and long term trends.
        
        Args:
            df (pd.DataFrame): DataFrame with trend classifications
            price_type (str, optional): Type of price to analyze ('USD' or 'BTC'). Defaults to 'USD'.
            
        Returns:
            pd.DataFrame: DataFrame with overall trend column added
        """
        suffix = f'_{price_type}'
        short_term_col = f'Trend_Short Term{suffix}'
        medium_term_col = f'Trend_Medium Term{suffix}'
        long_term_col = f'Trend_Long Term{suffix}'
        
        # Initialize overall trend column
        overall_col = f'Overall_Trend{suffix}'
        df[overall_col] = np.nan
        
        # Return if not all required columns are present
        if not all(col in df.columns for col in [short_term_col, medium_term_col, long_term_col]):
            return df
        
        # Calculate for each row, skipping NaN values
        for i in range(len(df)):
            # Get trend values for this row
            short = df.loc[df.index[i], short_term_col]
            medium = df.loc[df.index[i], medium_term_col]
            long = df.loc[df.index[i], long_term_col]
            
            # Skip if any required value is NaN
            if pd.isna(short) or pd.isna(medium) or pd.isna(long):
                continue
            
            # Calculate weighted average (more weight to medium term)
            # Short term: 30%, Medium term: 40%, Long term: 30%
            weighted_trend = (0.3 * short) + (0.4 * medium) + (0.3 * long)
            
            # Discretize: >0.3 is bullish, <-0.3 is bearish, else neutral
            if weighted_trend > 0.3:
                overall_trend = 1  # Bullish
            elif weighted_trend < -0.3:
                overall_trend = -1  # Bearish
            else:
                overall_trend = 0  # Neutral
            
            # Set value
            df.loc[df.index[i], overall_col] = overall_trend
        
        return df
    
    def compute_overall_classification(self, trends):
        """
        Compute overall trend classification from individual trend signals.
        
        Args:
            trends (pd.Series or pd.DataFrame): Series or DataFrame with trend values
            
        Returns:
            int: Overall classification (-2 to 2)
        """
        # Extract all trend values
        trend_values = []
        
        # Check if trends is a Series with integer index (as in create_classified_data)
        if isinstance(trends, pd.Series) and trends.index.dtype.kind in 'iu':
            # Just use the values directly
            trend_values = [v for v in trends.values if pd.notna(v)]
        else:
            # Original logic for DataFrame with column names
            for col in trends.index:
                if isinstance(col, str) and col.startswith('Trend_') and pd.notna(trends[col]):
                    trend_values.append(trends[col])
        
        if not trend_values:
            return np.nan  # Return NaN if no valid trend values
        
        # Calculate overall trend
        avg_trend = np.mean(trend_values)
        
        # Map average to discrete classification
        if avg_trend <= -1.5:
            return -2
        elif avg_trend <= -0.5:
            return -1
        elif avg_trend <= 0.5:
            return 0
        elif avg_trend <= 1.5:
            return 1
        else:
            return 2
    
    def create_classified_data(self, data, asset_id):
        """
        Create a dataset with trend classifications for both USD and BTC price data.
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            asset_id (str): Asset ID
            
        Returns:
            pd.DataFrame: DataFrame with trend classifications
        """
        if data.empty:
            self.logger.warning(f"Empty data for {asset_id}, cannot create classified data.")
            return pd.DataFrame()
        
        # Create copy for classified data
        classified_data = data.copy()
        
        # Process USD price data
        if 'close' in data.columns:
            usd_classified = self.classify_trend(data, asset_id, 'USD')
            
            # Get all newly created columns from USD classification
            usd_cols = [col for col in usd_classified.columns if col not in classified_data.columns]
            
            # Add them to the result DataFrame
            for col in usd_cols:
                classified_data[col] = usd_classified[col]
        else:
            self.logger.warning(f"No 'close' column found for {asset_id}, USD trend classification skipped")
        
        # Process BTC price data (if available)
        btc_price_col = f'{asset_id}_btc'
        if btc_price_col in data.columns:
            btc_classified = self.classify_trend(data, asset_id, 'BTC')
            
            # Get all newly created columns from BTC classification
            btc_cols = [col for col in btc_classified.columns if col not in classified_data.columns]
            
            # Add them to the result DataFrame
            for col in btc_cols:
                classified_data[col] = btc_classified[col]
        else:
            self.logger.debug(f"No '{btc_price_col}' column found for {asset_id}, BTC trend classification skipped")
        
        return classified_data 