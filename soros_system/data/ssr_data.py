import pandas as pd
import numpy as np
import os
import logging

class SSRDataHandler:
    """
    Handles loading and processing of Stock-to-Flow Ratio (SSR) data.
    """
    def __init__(self, ssr_data_path):
        """
        Initialize the SSRDataHandler.
        
        Args:
            ssr_data_path (str): Path to the SSR data file
        """
        self.ssr_data_path = ssr_data_path
        self.logger = logging.getLogger(__name__)
        self.ssr_data = self._load_ssr_data()
        
    def _load_ssr_data(self):
        """
        Load SSR data from CSV file.
        
        Returns:
            pd.DataFrame: DataFrame containing the SSR data, or None if loading fails
        """
        if not self.ssr_data_path or not os.path.exists(self.ssr_data_path):
            self.logger.warning(f"SSR data file not found: {self.ssr_data_path}")
            return None
        
        try:
            ssr_data = pd.read_csv(self.ssr_data_path)
            ssr_data['date'] = pd.to_datetime(ssr_data['date'])
            ssr_data.set_index('date', inplace=True)
            self.logger.info(f"Successfully loaded SSR data with {len(ssr_data)} rows.")
            return ssr_data
        except Exception as e:
            self.logger.error(f"Failed to load SSR data: {e}")
            return None
        
    def get_ssr_signal(self, date):
        """
        Get the SSR signal for a specific date.
        
        Args:
            date (datetime): Date to get the SSR signal for
            
        Returns:
            int: SSR signal (-1 for bearish, 0 for neutral, 1 for bullish), or 0 if data is not available
        """
        if self.ssr_data is None:
            return 0
        
        try:
            # Convert to pandas datetime if needed
            if not isinstance(date, pd.Timestamp):
                date = pd.Timestamp(date)
            
            # Find the closest date in the SSR data
            closest_date = self.ssr_data.index[self.ssr_data.index <= date]
            if len(closest_date) == 0:
                self.logger.warning(f"No SSR data available for or before {date}")
                return 0
            
            closest_date = closest_date[-1]
            
            # Get the SSR signal
            ssr_value = self.ssr_data.loc[closest_date, 'ssr']
            
            # Classify the signal based on SSR value
            if ssr_value >= 2.0:
                return 1  # Bullish
            elif ssr_value <= 0.5:
                return -1  # Bearish
            else:
                return 0  # Neutral
        except Exception as e:
            self.logger.error(f"Error getting SSR signal for {date}: {e}")
            return 0 