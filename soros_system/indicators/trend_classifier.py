import pandas as pd
import numpy as np
import logging
from tqdm.auto import tqdm

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
        
    def _calculate_indicators(self, data, price_column, asset_id, suffix=''):
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
        
        # Create a copy of the data to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Ensure price column is numeric and convert to Series if needed
        if isinstance(df[price_column], np.ndarray):
            df[price_column] = pd.Series(df[price_column], index=df.index)
        df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
        
        # Calculate moving averages
        df, ma_columns, roc_columns = self.ma_calculator.calculate_all_mas(df, price_column)
        
        return df, ma_columns, roc_columns
    
    def classify_trend(self, mas, rocs, data, suffix):
        """
        Classify trend for an asset based on technical indicators.
        
        Args:
            mas (dict): Dictionary of MA columns by term
            rocs (dict): Dictionary of RoC columns by term
            data (pd.DataFrame): DataFrame with price data
            suffix (str): Suffix for trend columns ('_USD' or '_BTC')
            
        Returns:
            pd.DataFrame: DataFrame with trend classifications added
        """
        if data.empty:
            self.logger.warning(f"Empty data provided for trend classification.")
            return data
        
        # Make a copy to avoid modifying the original DataFrame
        df = data.copy()
        
        # Extract price column from suffix
        price_type = suffix.strip('_')
        price_col = 'close' if price_type == 'USD' else next((col for col in df.columns if col.endswith('_btc')), None)
        
        if not price_col:
            self.logger.warning(f"No price column found for {price_type}")
            return df
        
        # Create columns for trend classifications
        term_mapping = {
            'Short Term': 'short_term',
            'Medium Term': 'medium_term',
            'Long Term': 'long_term'
        }
        
        terms = ['Short Term', 'Medium Term', 'Long Term']
        for term in terms:
            # Use the new snake_case column naming format
            snake_case_term = term_mapping[term]
            df[f'{snake_case_term}_trend{suffix}'] = np.nan
            
            ma_cols = mas.get(term, [])
            ma_cols_filtered = [col for col in ma_cols if col in df.columns]
            roc_cols = rocs.get(term, [])
            roc_cols_filtered = [col for col in roc_cols if col in df.columns]
            
            if not ma_cols_filtered or not roc_cols_filtered:
                self.logger.warning(f"Not enough MA or RoC columns for {term}{suffix} trend classification")
                continue
            
            # For each row in the DataFrame - with a progress bar for long datasets
            total_rows = len(df)
            if total_rows > 5000:  # Only show progress for large datasets
                iterator = tqdm(range(total_rows), desc=f"Classifying {term}{suffix}", leave=False)
            else:
                iterator = range(total_rows)
                
            for i in iterator:
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
                
                # Make sure we're getting the value from the correct row
                if price_col not in row:
                    self.logger.warning(f"Price column {price_col} not found in row")
                    continue
                    
                price_val = row[price_col]
                if pd.isna(price_val):
                    continue
                    
                # Calculate MA trend proportions
                ma_bullish_count = sum(1 for ma in ma_values if price_val > ma)
                ma_bearish_count = sum(1 for ma in ma_values if price_val < ma)
                ma_total = len(ma_values)
                ma_bullish_prop = ma_bullish_count / ma_total if ma_total > 0 else 0
                ma_bearish_prop = ma_bearish_count / ma_total if ma_total > 0 else 0
                
                # Calculate RoC trend proportions
                roc_bullish_count = sum(1 for roc in roc_values if roc > 0)
                roc_bearish_count = sum(1 for roc in roc_values if roc < 0)
                roc_total = len(roc_values)
                roc_bullish_prop = roc_bullish_count / roc_total if roc_total > 0 else 0
                roc_bearish_prop = roc_bearish_count / roc_total if roc_total > 0 else 0
                
                # Determine MA trend direction
                if ma_bullish_prop > 0.6:  # Strong bullish MA
                    ma_trend = 1
                elif ma_bearish_prop > 0.6:  # Strong bearish MA
                    ma_trend = -1
                else:  # Mixed or weak MA signals
                    ma_trend = 0
                
                # Determine RoC trend direction
                if roc_bullish_prop > 0.6:  # Strong bullish RoC
                    roc_trend = 1
                elif roc_bearish_prop > 0.6:  # Strong bearish RoC
                    roc_trend = -1
                else:  # Mixed or weak RoC signals
                    roc_trend = 0
                
                # Calculate trend strength based on agreement and proportions
                if ma_trend == roc_trend:
                    if ma_trend == 1:  # Bullish
                        # Strong bull if both MA and RoC show strong bullish proportions
                        if ma_bullish_prop > 0.8 and roc_bullish_prop > 0.8:
                            combined_trend = 2  # Strong Bull
                        else:
                            combined_trend = 1  # Weak Bull
                    elif ma_trend == -1:  # Bearish
                        # Strong bear if both MA and RoC show strong bearish proportions
                        if ma_bearish_prop > 0.8 and roc_bearish_prop > 0.8:
                            combined_trend = -2  # Strong Bear
                        else:
                            combined_trend = -1  # Weak Bear
                    else:  # Neutral
                        combined_trend = 0
                else:
                    # If trends disagree, use neutral
                    combined_trend = 0
                
                # Set trend value for this row using the new naming convention
                df.loc[df.index[i], f'{snake_case_term}_trend{suffix}'] = combined_trend
        
        # Create overall trend classification
        self._calculate_overall_trend(df, price_type)
        
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
        
        # Use the new snake_case column names
        short_term_col = f'short_term_trend{suffix}'
        medium_term_col = f'medium_term_trend{suffix}'
        long_term_col = f'long_term_trend{suffix}'
        
        # Initialize overall trend column with the new naming format
        overall_col = f'overall_trend{suffix}'
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
            
            # Discretize based on thresholds
            # Five classifications: -2, -1, 0, 1, 2
            if weighted_trend > 1.5:
                overall_trend = 2  # Strong Bullish
            elif weighted_trend > 0.5:
                overall_trend = 1  # Weak Bullish
            elif weighted_trend > -0.5:
                overall_trend = 0  # Neutral
            elif weighted_trend > -1.5:
                overall_trend = -1  # Weak Bearish
            else:
                overall_trend = -2  # Strong Bearish
            
            # Set value
            df.loc[df.index[i], overall_col] = overall_trend
        
        return df
    
    def _compute_overall_classification(self, trends):
        """
        Compute overall trend classification from individual trend signals.
        
        Args:
            trends (pd.Series or pd.DataFrame): Series or DataFrame with trend values
            
        Returns:
            int: Overall classification (-2 to 2)
        """
        # Extract all trend values
        trend_values = []
        
        # Check if trends is a Series with integer index
        if isinstance(trends, pd.Series) and trends.index.dtype.kind in 'iu':
            # Just use the values directly
            trend_values = [v for v in trends.values if pd.notna(v)]
        else:
            # For DataFrame with column names
            for col in trends.index:
                if isinstance(col, str) and '_trend_' in col and pd.notna(trends[col]):
                    trend_values.append(trends[col])
        
        if not trend_values:
            return np.nan  # Return NaN if no valid trend values
        
        # Calculate overall trend
        avg_trend = np.mean(trend_values)
        
        # Map average to discrete classification
        if avg_trend <= -1.5:
            return -2  # Strong Bear
        elif avg_trend <= -0.5:
            return -1  # Weak Bear
        elif avg_trend <= 0.5:
            return 0   # Neutral
        elif avg_trend <= 1.5:
            return 1   # Weak Bull
        else:
            return 2   # Strong Bull
    
    def _create_classified_data(self, data, asset_id):
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
        self.logger.info(f"Creating classified data for {asset_id} with {len(data)} rows")
        
        # Process USD and BTC price data with progress tracking
        with tqdm(total=2, desc=f"Creating trend data for {asset_id}", leave=False) as pbar:
            # Process USD price data
            if 'close' in data.columns:
                # Calculate indicators for USD
                df_usd, ma_usd, roc_usd = self._calculate_indicators(data, 'close', asset_id)
                
                # Classify USD trends
                usd_classified = self.classify_trend(ma_usd, roc_usd, df_usd, '_USD')
                
                # Get all newly created columns from USD classification
                usd_cols = [col for col in usd_classified.columns if col not in classified_data.columns]
                
                # Add them to the result DataFrame
                for col in usd_cols:
                    classified_data[col] = usd_classified[col]
                
                pbar.update(1)
            else:
                self.logger.warning(f"No 'close' column found for {asset_id}, USD trend classification skipped")
                pbar.update(1)
            
            # Process BTC price data (if available)
            btc_price_col = f'{asset_id}_btc'
            if btc_price_col in data.columns:
                # Calculate indicators for BTC
                df_btc, ma_btc, roc_btc = self._calculate_indicators(data, btc_price_col, asset_id)
                
                # Classify BTC trends
                btc_classified = self.classify_trend(ma_btc, roc_btc, df_btc, '_BTC')
                
                # Get all newly created columns from BTC classification
                btc_cols = [col for col in btc_classified.columns if col not in classified_data.columns]
                
                # Add them to the result DataFrame
                for col in btc_cols:
                    classified_data[col] = btc_classified[col]
                
                pbar.update(1)
            else:
                self.logger.debug(f"No '{btc_price_col}' column found for {asset_id}, BTC trend classification skipped")
                pbar.update(1)
        
        # Log summary of classification
        trend_cols = [col for col in classified_data.columns if '_trend_' in col and (col.endswith('_USD') or col.endswith('_BTC'))]
        self.logger.debug(f"Created {len(trend_cols)} trend columns for {asset_id}")
        
        return classified_data
        
    # Alias for backwards compatibility
    create_classified_data = _create_classified_data 