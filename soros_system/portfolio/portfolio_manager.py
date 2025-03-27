import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime

class PortfolioManager:
    """
    Class for managing different portfolios and their criteria.
    """
    def __init__(self, portfolios_dir='portfolios'):
        """
        Initialize the PortfolioManager.
        
        Args:
            portfolios_dir (str, optional): Directory for storing portfolio data
        """
        self.logger = logging.getLogger(__name__)
        self.portfolios_dir = portfolios_dir
        self.portfolios = {}
        self.basket_criteria = {}
        
        # Create portfolios directory if it doesn't exist
        if not os.path.exists(portfolios_dir):
            os.makedirs(portfolios_dir)
            self.logger.info(f"Created portfolios directory: {portfolios_dir}")
        
        # Load existing portfolios
        self._load_portfolios()
    
    def _load_portfolios(self):
        """Load existing portfolios from the portfolios directory."""
        try:
            for filename in os.listdir(self.portfolios_dir):
                if filename.endswith('.json'):
                    portfolio_path = os.path.join(self.portfolios_dir, filename)
                    portfolio_name = filename[:-5]  # Remove '.json'
                    
                    with open(portfolio_path, 'r') as f:
                        portfolio_data = json.load(f)
                    
                    self.portfolios[portfolio_name] = portfolio_data
                    self.logger.info(f"Loaded portfolio: {portfolio_name}")
        except Exception as e:
            self.logger.error(f"Error loading portfolios: {e}")
    
    def _save_portfolio(self, portfolio_name):
        """
        Save a portfolio to disk.
        
        Args:
            portfolio_name (str): Name of the portfolio to save
        """
        if portfolio_name not in self.portfolios:
            self.logger.warning(f"Portfolio '{portfolio_name}' does not exist.")
            return
        
        try:
            portfolio_path = os.path.join(self.portfolios_dir, f"{portfolio_name}.json")
            with open(portfolio_path, 'w') as f:
                json.dump(self.portfolios[portfolio_name], f, indent=4)
            self.logger.info(f"Saved portfolio: {portfolio_name}")
        except Exception as e:
            self.logger.error(f"Error saving portfolio '{portfolio_name}': {e}")
    
    def create_portfolio(self, portfolio_name, **criteria):
        """
        Create a new portfolio with specified criteria.
        
        Args:
            portfolio_name (str): Name of the portfolio
            **criteria: Criteria for the portfolio including:
                usd_conditions (list/dict): List of valid trend values or dict mapping trend terms to valid values
                btc_conditions (list/dict): List of valid trend values or dict mapping trend terms to valid values
                btc_trend_gating (int): Minimum BTC trend value to allow trading
                rsi_conditions_usd (bool): Whether to apply RSI filter for USD trends
                rsi_conditions_btc (bool): Whether to apply RSI filter for BTC trends
                use_btc_rsi_signal (bool): Whether to use BTC RSI as signal
                btc_only (bool): Whether to only include BTC in the portfolio
                follow_portfolio (str): Name of portfolio to follow
                use_ssr_signal (bool): Whether to use SSR signal
                use_ssr_gate (bool): Whether to use SSR as gate
                use_volatility_filter (bool): Whether to filter by volatility
                volatility_weight (float): Weight for volatility
                max_assets (int): Maximum number of assets to include
            
        Returns:
            bool: True if portfolio was created successfully, False otherwise
        """
        if portfolio_name in self.portfolios:
            self.logger.warning(f"Portfolio '{portfolio_name}' already exists.")
            return False
        
        # Process trend conditions to ensure support for both numeric values and text descriptions
        if 'usd_conditions' in criteria:
            criteria['usd_conditions'] = self._process_trend_conditions(criteria['usd_conditions'])
        
        if 'btc_conditions' in criteria:
            criteria['btc_conditions'] = self._process_trend_conditions(criteria['btc_conditions'])
        
        # Create portfolio with criteria and metadata
        portfolio = {
            'criteria': criteria,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'assets': []
        }
        
        self.portfolios[portfolio_name] = portfolio
        self._save_portfolio(portfolio_name)
        
        self.logger.info(f"Created portfolio: {portfolio_name}")
        return True
    
    def _process_trend_conditions(self, conditions):
        """
        Process trend conditions to handle both numeric values and text descriptions.
        
        Args:
            conditions: Trend conditions, can be list, dict, or single value
            
        Returns:
            dict or list: Processed trend conditions
        """
        # Mapping between text descriptions and numeric values
        trend_mapping = {
            'Strong Bear': -2,
            'Weak Bear': -1,
            'Neutral': 0,
            'Weak Bull': 1,
            'Strong Bull': 2
        }
        
        reverse_mapping = {v: k for k, v in trend_mapping.items()}
        
        # If conditions is a dictionary (mapping terms to values)
        if isinstance(conditions, dict):
            processed_conditions = {}
            for term, values in conditions.items():
                if isinstance(values, list):
                    # Convert any string values to numeric
                    processed_values = []
                    for value in values:
                        if isinstance(value, str) and value in trend_mapping:
                            processed_values.append(trend_mapping[value])
                        else:
                            processed_values.append(value)
                    processed_conditions[term] = processed_values
                elif isinstance(values, str) and values in trend_mapping:
                    processed_conditions[term] = [trend_mapping[values]]
                else:
                    processed_conditions[term] = [values] if not isinstance(values, list) else values
            return processed_conditions
        
        # If conditions is a list or single value
        processed_conditions = []
        if not isinstance(conditions, list):
            conditions = [conditions]
        
        for value in conditions:
            if isinstance(value, str) and value in trend_mapping:
                processed_conditions.append(trend_mapping[value])
            else:
                processed_conditions.append(value)
        
        return processed_conditions
    
    def create_roc_based_portfolio(self, portfolio_name, **criteria):
        """
        Create a new portfolio based on Rate of Change (RoC) criteria.
        
        Args:
            portfolio_name (str): Name of the portfolio
            **criteria: RoC-based criteria for the portfolio
            
        Returns:
            bool: True if portfolio was created successfully, False otherwise
        """
        # Add specific RoC flags to criteria
        criteria['is_roc_based'] = True
        
        # Add trend conditions if not present
        if 'trend_conditions' not in criteria:
            criteria['trend_conditions'] = {
                'short_term': criteria.get('short_term_condition', True),
                'medium_term': criteria.get('medium_term_condition', False),
                'long_term': criteria.get('long_term_condition', False)
            }
        
        return self.create_portfolio(portfolio_name, **criteria)
    
    def list_portfolios(self):
        """
        List all available portfolios.
        
        Returns:
            pd.DataFrame: DataFrame with portfolio details
        """
        if not self.portfolios:
            self.logger.info("No portfolios available.")
            return pd.DataFrame()
        
        portfolio_data = []
        
        for name, data in self.portfolios.items():
            # Extract basic metadata
            created_at = data.get('created_at', 'Unknown')
            updated_at = data.get('updated_at', 'Unknown')
            
            # Extract key criteria
            criteria = data.get('criteria', {})
            btc_trend_gating = criteria.get('btc_trend_gating')
            usd_conditions = criteria.get('usd_conditions')
            btc_conditions = criteria.get('btc_conditions')
            
            # Extract trend conditions
            trend_conditions = criteria.get('trend_conditions', {})
            short_term = trend_conditions.get('short_term', False)
            medium_term = trend_conditions.get('medium_term', False)
            long_term = trend_conditions.get('long_term', False)
            
            # Create summary text
            summary = []
            if btc_trend_gating:
                summary.append(f"BTC gating: {btc_trend_gating}")
            if usd_conditions:
                summary.append(f"USD cond: {usd_conditions}")
            if btc_conditions:
                summary.append(f"BTC cond: {btc_conditions}")
            if any([short_term, medium_term, long_term]):
                terms = []
                if short_term:
                    terms.append("Short")
                if medium_term:
                    terms.append("Medium")
                if long_term:
                    terms.append("Long")
                summary.append(f"Terms: {', '.join(terms)}")
            
            # Create portfolio entry
            entry = {
                'Name': name,
                'Created': created_at,
                'Updated': updated_at,
                'Summary': '; '.join(summary),
                'Assets': len(data.get('assets', []))
            }
            
            portfolio_data.append(entry)
        
        # Convert to DataFrame
        return pd.DataFrame(portfolio_data)
    
    def delete_portfolio(self, portfolio_name):
        """
        Delete a portfolio.
        
        Args:
            portfolio_name (str): Name of the portfolio to delete
            
        Returns:
            bool: True if portfolio was deleted successfully, False otherwise
        """
        if portfolio_name not in self.portfolios:
            self.logger.warning(f"Portfolio '{portfolio_name}' does not exist.")
            return False
        
        try:
            # Remove from memory
            del self.portfolios[portfolio_name]
            
            # Remove from disk
            portfolio_path = os.path.join(self.portfolios_dir, f"{portfolio_name}.json")
            if os.path.exists(portfolio_path):
                os.remove(portfolio_path)
            
            self.logger.info(f"Deleted portfolio: {portfolio_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting portfolio '{portfolio_name}': {e}")
            return False
    
    def get_portfolio_details(self, portfolio_name):
        """
        Get detailed information about a portfolio.
        
        Args:
            portfolio_name (str): Name of the portfolio
            
        Returns:
            dict: Portfolio details, or None if portfolio doesn't exist
        """
        if portfolio_name not in self.portfolios:
            self.logger.warning(f"Portfolio '{portfolio_name}' does not exist.")
            return None
        
        return self.portfolios[portfolio_name]
    
    def set_basket_criteria(self, criteria):
        """
        Set criteria for filtering the asset basket.
        
        Args:
            criteria (dict): Criteria for filtering assets
        """
        self.basket_criteria = criteria
        self.logger.info(f"Set basket criteria: {criteria}")
    
    def filter_basket_historical(self, df, dates=None):
        """
        Filter historical data based on basket criteria.
        
        Args:
            df (pd.DataFrame): DataFrame with asset data
            dates (list, optional): Specific dates to filter on
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # Filter by date if specified
        if dates:
            filtered_df = filtered_df[filtered_df['date'].isin(dates)]
        
        # Apply criteria filtering
        if self.basket_criteria:
            criteria = self.basket_criteria
            
            # Filter by minimum market cap
            if 'min_market_cap' in criteria and criteria['min_market_cap'] > 0:
                if 'market_cap' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['market_cap'] >= criteria['min_market_cap']]
            
            # Filter by minimum volume
            if 'min_volume' in criteria and criteria['min_volume'] > 0:
                if 'volume' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['volume'] >= criteria['min_volume']]
            
            # Filter by trend strength
            if 'min_trend_strength' in criteria:
                min_strength = criteria['min_trend_strength']
                trend_cols = [col for col in filtered_df.columns if col.startswith('Trend_')]
                
                if trend_cols:
                    # Calculate absolute trend strength
                    filtered_df['trend_strength'] = filtered_df[trend_cols].abs().mean(axis=1)
                    filtered_df = filtered_df[filtered_df['trend_strength'] >= min_strength]
        
        return filtered_df 