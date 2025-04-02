import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
from tqdm import tqdm

class PortfolioManager:
    """
    Class for managing different portfolios and their criteria.
    """
    def __init__(self, portfolios_dir='portfolios', ssr_handler=None, markov_vol_model=None):
        """
        Initialize the PortfolioManager.
        
        Args:
            portfolios_dir (str, optional): Directory for storing portfolio data
            ssr_handler: Handler for SSR data
            markov_vol_model: Model for volatility prediction
        """
        self.logger = logging.getLogger(__name__)
        self.portfolios_dir = portfolios_dir
        self.portfolios = {}
        self.basket_criteria = {}
        self.ssr_handler = ssr_handler
        self.markov_vol_model = markov_vol_model
        
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
            # Create a copy of the portfolio data without the classified_data for saving
            save_data = self.portfolios[portfolio_name].copy()
            if 'classified_data' in save_data:
                del save_data['classified_data']  # Don't save DataFrames to JSON
            
            portfolio_path = os.path.join(self.portfolios_dir, f"{portfolio_name}.json")
            with open(portfolio_path, 'w') as f:
                json.dump(save_data, f, indent=4)
            self.logger.info(f"Saved portfolio: {portfolio_name}")
        except Exception as e:
            self.logger.error(f"Error saving portfolio '{portfolio_name}': {e}")
    
    def create_portfolio(self, portfolio_name, classified_data=None, **criteria):
        """
        Create a new portfolio with specified criteria and classified data.
        
        Args:
            portfolio_name (str): Name of the portfolio
            classified_data (dict, optional): Dictionary mapping asset_ids to DataFrames with signals
            **criteria: Criteria for the portfolio including:
                usd_conditions (list/dict): List of valid trend values or dict mapping trend terms to valid values
                btc_conditions (list/dict): List of valid trend values or dict mapping trend terms to valid values
                btc_trend_gating (int/list): Minimum BTC trend value(s) to allow trading
                rsi_conditions_usd (bool/dict): Whether/how to apply RSI filter for USD trends
                rsi_conditions_btc (bool/dict): Whether/how to apply RSI filter for BTC trends
                use_btc_rsi_signal (bool): Whether to use BTC RSI as signal
                btc_only (bool): Whether to only include BTC in the portfolio
                follow_portfolio (str): Name of portfolio to follow
                use_ssr_signal (bool): Whether to use SSR signal
                use_ssr_gate (bool): Whether to use SSR as gate
                use_volatility_filter (bool): Whether to filter by volatility
                volatility_weight (float): Weight for volatility
                signal_threshold (float): Threshold for percentage of positive signals (default: 50)
                max_assets (int): Maximum number of assets to include
                
        Returns:
            bool: True if portfolio was created successfully, False otherwise
        """
        if portfolio_name in self.portfolios:
            self.logger.warning(f"Portfolio '{portfolio_name}' already exists. Overwriting.")
        
        # Process trend conditions to ensure support for both numeric values and text descriptions
        if 'usd_conditions' in criteria:
            criteria['usd_conditions'] = self._process_trend_conditions(criteria['usd_conditions'])
        
        if 'btc_conditions' in criteria:
            criteria['btc_conditions'] = self._process_trend_conditions(criteria['btc_conditions'])
        
        if 'btc_trend_gating' in criteria:
            criteria['btc_trend_gating'] = self._process_trend_conditions(criteria['btc_trend_gating'])
        
        # Ensure signal_threshold has a valid value
        if 'signal_threshold' not in criteria:
            criteria['signal_threshold'] = 50  # Default threshold
        
        # Create portfolio with criteria and metadata
        portfolio = {
            'criteria': criteria,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'assets': [],
            'classified_data': {}  # Store classified data in memory but not in JSON
        }
        
        # Store the classified data if provided
        if classified_data:
            portfolio['classified_data'] = classified_data
            
            # Compute signals for all assets in the classified data
            self._compute_portfolio_signals(portfolio)
            
            # Extract asset IDs from classified_data for the assets list
            portfolio['assets'] = list(classified_data.keys())
        
        self.portfolios[portfolio_name] = portfolio
        self._save_portfolio(portfolio_name)
        
        self.logger.info(f"Created portfolio: {portfolio_name} with {len(portfolio.get('assets', []))} assets")
        return True
    
    def _compute_portfolio_signals(self, portfolio):
        """
        Compute trading signals for all assets in a portfolio based on criteria.
        
        Args:
            portfolio (dict): Portfolio dictionary with criteria and classified_data
        """
        criteria = portfolio.get('criteria', {})
        classified_data = portfolio.get('classified_data', {})
        
        if not classified_data:
            self.logger.warning("No classified data available for signal computation")
            return
        
        # Extract criteria
        usd_conditions = criteria.get('usd_conditions')
        btc_conditions = criteria.get('btc_conditions')
        btc_trend_gating = criteria.get('btc_trend_gating')
        use_volatility_filter = criteria.get('use_volatility_filter', False)
        use_ssr_signal = criteria.get('use_ssr_signal', False)
        use_ssr_gate = criteria.get('use_ssr_gate', False)
        rsi_conditions_usd = criteria.get('rsi_conditions_usd', False)
        rsi_conditions_btc = criteria.get('rsi_conditions_btc', False)
        signal_threshold = criteria.get('signal_threshold', 50)
        slippage_pct = criteria.get('slippage_pct', 0)  # Slippage percentage, default 0
        
        # Get Bitcoin data if available (for gating)
        btc_data = classified_data.get('bitcoin', None)
        
        # Process each asset
        for asset_id, data_df in classified_data.items():
            if data_df.empty:
                self.logger.warning(f"Empty DataFrame for {asset_id}, skipping signal computation")
                continue
                
            self.logger.info(f"Computing signals for {asset_id}")
            signal_components = []  # Keep track of which signals are used
            
            # Ensure data_df has a date index for alignment
            if 'date' in data_df.columns and not isinstance(data_df.index, pd.DatetimeIndex):
                data_df = data_df.set_index('date')
            
            # 1. USD Trend Signal
            if usd_conditions is not None:
                trend_col = 'overall_trend_USD'
                if trend_col in data_df.columns:
                    allowed_values = usd_conditions
                    # Ensure allowed_values is a list or set for 'in' check
                    if not isinstance(allowed_values, (list, set)):
                        allowed_values = [allowed_values]
                    # Ensure comparison works with string or numeric types in the column
                    allowed_values_str = {str(v) for v in allowed_values}
                    # Signal is 1 if trend is allowed, -1 otherwise
                    data_df['usd_trend_signal'] = data_df[trend_col].apply(
                        lambda x: 1 if str(x) in allowed_values_str else -1
                    )
                    signal_components.append('usd_trend_signal')
                    self.logger.debug(f"Generated 'usd_trend_signal' from '{trend_col}' for {asset_id}")
                else:
                    self.logger.warning(f"'{trend_col}' column not found for {asset_id}. Defaulting signal to -1.")
                    data_df['usd_trend_signal'] = -1
            else:
                data_df['usd_trend_signal'] = -1  # Default to -1 if not used
            
            # 2. BTC Trend Signal (for altcoins)
            if btc_conditions is not None and asset_id != 'bitcoin':
                trend_col = 'overall_trend_BTC'
                if trend_col in data_df.columns:
                    allowed_values = btc_conditions
                    if not isinstance(allowed_values, (list, set)):
                        allowed_values = [allowed_values]
                    allowed_values_str = {str(v) for v in allowed_values}
                    # Signal is 1 if trend is allowed, -1 otherwise
                    data_df['btc_trend_signal'] = data_df[trend_col].apply(
                        lambda x: 1 if str(x) in allowed_values_str else -1
                    )
                    signal_components.append('btc_trend_signal')
                    self.logger.debug(f"Generated 'btc_trend_signal' from '{trend_col}' for {asset_id}")
                else:
                    self.logger.warning(f"'{trend_col}' column not found for {asset_id}. Defaulting signal to -1.")
                    data_df['btc_trend_signal'] = -1
            else:
                data_df['btc_trend_signal'] = -1  # Default to -1 if not used
            
            # 3. RSI Signal (USD)
            if rsi_conditions_usd:
                rsi_col = 'RSI_Signal_28_USD'
                if rsi_col in data_df.columns:
                    # Ensure RSI signal is numeric before comparison
                    data_df[rsi_col] = pd.to_numeric(data_df[rsi_col], errors='coerce')
                    # Signal is 1 if RSI > 0, -1 otherwise (or if NaN)
                    data_df['rsi_usd_signal'] = data_df[rsi_col].apply(
                        lambda x: 1 if pd.notna(x) and x > 0 else -1
                    )
                    signal_components.append('rsi_usd_signal')
                    self.logger.debug(f"Generated 'rsi_usd_signal' from '{rsi_col}' for {asset_id}")
                else:
                    self.logger.warning(f"'{rsi_col}' column not found for {asset_id}. Defaulting signal to -1.")
                    data_df['rsi_usd_signal'] = -1
            else:
                data_df['rsi_usd_signal'] = -1  # Default to -1 if not used
            
            # 4. RSI Signal (BTC) - for altcoins
            if rsi_conditions_btc and asset_id != 'bitcoin':
                rsi_col = 'RSI_Signal_28_BTC'
                if rsi_col in data_df.columns:
                    # Ensure RSI signal is numeric before comparison
                    data_df[rsi_col] = pd.to_numeric(data_df[rsi_col], errors='coerce')
                    # Signal is 1 if RSI > 0, -1 otherwise (or if NaN)
                    data_df['rsi_btc_signal'] = data_df[rsi_col].apply(
                        lambda x: 1 if pd.notna(x) and x > 0 else -1
                    )
                    signal_components.append('rsi_btc_signal')
                    self.logger.debug(f"Generated 'rsi_btc_signal' from '{rsi_col}' for {asset_id}")
                else:
                    self.logger.warning(f"'{rsi_col}' column not found for {asset_id}. Defaulting signal to -1.")
                    data_df['rsi_btc_signal'] = -1
            else:
                data_df['rsi_btc_signal'] = -1  # Default to -1 if not used
            
            # 5. Volatility Signal
            if use_volatility_filter and self.markov_vol_model:
                self.logger.debug(f"Calculating volatility signals for {asset_id}...")
                vol_signals = {}
                dates_to_analyze = data_df.index if isinstance(data_df.index, pd.DatetimeIndex) else pd.to_datetime(data_df['date'])
                
                for date in tqdm(dates_to_analyze, desc=f"Volatility Signal ({asset_id})", leave=False):
                    try:
                        signal_value = 1  # Default to low vol (1)
                        if hasattr(self.markov_vol_model, 'get_volatility_state'):
                            state = self.markov_vol_model.get_volatility_state(date)
                            signal_value = 1 if state == 'low' else -1  # 1 for low vol, -1 for high vol
                        elif hasattr(self.markov_vol_model, 'predict_volatility'):
                            state = self.markov_vol_model.predict_volatility(date)
                            signal_value = 1 if state == 1 else -1  # 1 for state 1 (low vol), -1 for high vol
                        else:
                            self.logger.warning("Markov analyzer provided but has no recognized volatility methods. Defaulting to low vol.")
                        vol_signals[date] = signal_value
                    except Exception as e:
                        self.logger.warning(f"Error getting volatility for {date}: {e}. Defaulting to low vol (1).")
                        vol_signals[date] = 1
                
                # Add volatility signals to data
                if isinstance(data_df.index, pd.DatetimeIndex):
                    data_df['volatility_signal'] = data_df.index.map(vol_signals).fillna(1)
                else:
                    # Convert date column to datetime if needed
                    if 'date' in data_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(data_df['date']):
                            data_df['date'] = pd.to_datetime(data_df['date'])
                        # Use date column to map volatility signals
                        data_df['volatility_signal'] = data_df['date'].map(vol_signals).fillna(1)
                    else:
                        self.logger.warning(f"No date column or index found for {asset_id}. Using default low vol (1).")
                        data_df['volatility_signal'] = 1
                
                signal_components.append('volatility_signal')
                self.logger.debug(f"Generated 'volatility_signal' for {asset_id}")
            else:
                data_df['volatility_signal'] = 1  # Default to 1 (low vol) if filter not used or no model
            
            # 6. SSR Signal
            if use_ssr_signal and self.ssr_handler:
                self.logger.debug(f"Calculating SSR signals for {asset_id}...")
                ssr_signals = {}
                dates_to_analyze = data_df.index if isinstance(data_df.index, pd.DatetimeIndex) else pd.to_datetime(data_df['date'])
                
                for date in tqdm(dates_to_analyze, desc=f"SSR Signal ({asset_id})", leave=False):
                    try:
                        ssr_signal = self.ssr_handler.get_ssr_signal(date)
                        ssr_signals[date] = ssr_signal
                    except Exception as e:
                        self.logger.warning(f"Error getting SSR signal for {date}: {e}. Defaulting to -1.")
                        ssr_signals[date] = -1
                
                # Add SSR signals to data
                if isinstance(data_df.index, pd.DatetimeIndex):
                    data_df['ssr_signal'] = data_df.index.map(ssr_signals).fillna(-1)
                else:
                    # Convert date column to datetime if needed
                    if 'date' in data_df.columns:
                        if not pd.api.types.is_datetime64_any_dtype(data_df['date']):
                            data_df['date'] = pd.to_datetime(data_df['date'])
                        # Use date column to map SSR signals
                        data_df['ssr_signal'] = data_df['date'].map(ssr_signals).fillna(-1)
                    else:
                        self.logger.warning(f"No date column or index found for {asset_id}. Using default SSR signal (-1).")
                        data_df['ssr_signal'] = -1
                
                signal_components.append('ssr_signal')
                self.logger.debug(f"Generated 'ssr_signal' for {asset_id}")
            else:
                # Check if SSR column exists in data
                ssr_col = 'SSR'
                if use_ssr_signal and ssr_col in data_df.columns:
                    # Use SSR from data if available
                    ssr_threshold = criteria.get('ssr_threshold', 0.5)
                    data_df[ssr_col] = pd.to_numeric(data_df[ssr_col], errors='coerce')
                    data_df['ssr_signal'] = data_df[ssr_col].apply(
                        lambda x: 1 if pd.notna(x) and x >= ssr_threshold else -1
                    )
                    signal_components.append('ssr_signal')
                    self.logger.debug(f"Generated 'ssr_signal' from '{ssr_col}' column for {asset_id}")
                else:
                    data_df['ssr_signal'] = -1  # Default to -1 if not used or no handler
            
            # 7. Combine Signals
            if not signal_components:
                self.logger.warning(f"No signal components configured for {asset_id}. Defaulting combined signal to 100.")
                data_df['combined_signal_sum'] = 100.0
                data_df['positive_signals_count'] = 1
                data_df['positive_signals_pct'] = 100.0
            else:
                self.logger.info(f"Combining signals for {asset_id}: {signal_components}")
                # Calculate sum of signals (now 1 or -1)
                data_df['combined_signal_sum'] = data_df[signal_components].sum(axis=1)
                # Calculate number of positive signals (== 1)
                data_df['positive_signals_count'] = (data_df[signal_components] == 1).sum(axis=1)
                # Calculate percentage of positive signals for threshold check
                data_df['positive_signals_pct'] = (data_df['positive_signals_count'] / len(signal_components)) * 100.0
            
            # 8. Apply BTC Trend Gating
            if btc_trend_gating is not None and btc_data is not None:
                self.logger.info(f"Applying BTC trend gating for {asset_id}")
                if isinstance(data_df.index, pd.DatetimeIndex) and isinstance(btc_data.index, pd.DatetimeIndex):
                    # Get the BTC trend gate series with matching index
                    gate_col = 'overall_trend_USD'  # Use USD trend for BTC
                    if gate_col in btc_data.columns:
                        # Prepare allowed gate values
                        allowed_gate_values = btc_trend_gating
                        if not isinstance(allowed_gate_values, (list, set)):
                            allowed_gate_values = [allowed_gate_values]
                        allowed_gate_values_str = {str(v) for v in allowed_gate_values}
                        
                        # Create gate condition series from BTC data
                        gate_condition = btc_data[gate_col].apply(
                            lambda x: str(x) in allowed_gate_values_str
                        )
                        
                        # Create a date-aligned Series for the gate_condition
                        gate_series = pd.Series(gate_condition)
                        
                        # Apply gate to asset data - reindex gate_condition to match data_df's index
                        aligned_gate = gate_series.reindex(data_df.index, method='ffill').fillna(False)
                        
                        # Where gate condition is False, force signals to negative
                        data_df['combined_signal_sum'] = data_df['combined_signal_sum'].where(
                            aligned_gate, -len(signal_components) - 1
                        )
                        data_df['positive_signals_pct'] = data_df['positive_signals_pct'].where(
                            aligned_gate, 0
                        )
                        
                        self.logger.debug(f"Applied BTC trend gate for {asset_id}")
                    else:
                        self.logger.warning(f"'{gate_col}' column not found in BTC data. Gate not applied for {asset_id}.")
                else:
                    self.logger.warning(f"Index mismatch between {asset_id} and BTC data. BTC gating not applied.")
            
            # 9. Calculate Final Decision
            def calculate_final_decision(row):
                # 1. Check if percentage of positive signals meets threshold
                if pd.notna(row['positive_signals_pct']) and row['positive_signals_pct'] >= signal_threshold:
                    # 2. If threshold met, check sign of combined signal sum
                    if pd.notna(row['combined_signal_sum']) and row['combined_signal_sum'] > 0:
                        return 1  # Buy/Hold
                    else:
                        return 0  # Sell/Cash (sum is zero or negative)
                else:
                    # If threshold not met, stay in cash / sell
                    return 0
            
            data_df['final_decision'] = data_df.apply(calculate_final_decision, axis=1)
            
            # 10. Shift Final Decision
            # Shift signals by 1 day to avoid lookahead bias
            data_df['final_decision_shifted'] = data_df['final_decision'].shift(1)
            
            # Store the updated DataFrame back in the portfolio
            portfolio['classified_data'][asset_id] = data_df
            
            self.logger.info(f"Signal computation completed for {asset_id}")
    
    def get_portfolio_signals(self, portfolio_name):
        """
        Get signal tables for all assets in a portfolio.
        
        Args:
            portfolio_name (str): Name of the portfolio
            
        Returns:
            dict: Dictionary mapping asset_ids to DataFrames with date, open, close, and final_decision_shifted
                 or None if portfolio doesn't exist or has no signals
        """
        if portfolio_name not in self.portfolios:
            self.logger.warning(f"Portfolio '{portfolio_name}' does not exist.")
            return None
        
        portfolio = self.portfolios[portfolio_name]
        classified_data = portfolio.get('classified_data', {})
        
        if not classified_data:
            self.logger.warning(f"No classified data found for portfolio '{portfolio_name}'.")
            return None
        
        # Extract the required columns for each asset
        signal_tables = {}
        for asset_id, data_df in classified_data.items():
            if data_df is None or data_df.empty:
                self.logger.warning(f"No data available for {asset_id} in portfolio '{portfolio_name}'.")
                continue
            
            # Make a copy to avoid modifying the original data
            signal_df = data_df.copy()
            
            # Ensure DataFrame has date as a column (not just index)
            if isinstance(signal_df.index, pd.DatetimeIndex) and 'date' not in signal_df.columns:
                signal_df = signal_df.reset_index()
            
            # Keep only required columns
            required_cols = ['date', 'open', 'close', 'final_decision_shifted']
            extra_cols = [col for col in signal_df.columns if col.endswith('_signal')]
            
            # Get all columns that exist in the DataFrame
            available_cols = [col for col in required_cols + extra_cols if col in signal_df.columns]
            
            # If open or close are missing, log warning
            if 'open' not in signal_df.columns:
                self.logger.warning(f"'open' column missing for {asset_id}. Using 'close' if available.")
                if 'close' in signal_df.columns:
                    signal_df['open'] = signal_df['close']
            
            if 'close' not in signal_df.columns:
                self.logger.warning(f"'close' column missing for {asset_id}. Using 'open' if available.")
                if 'open' in signal_df.columns:
                    signal_df['close'] = signal_df['open']
            
            # Create the signal table with available columns
            signal_tables[asset_id] = signal_df[available_cols]
            
            self.logger.debug(f"Created signal table for {asset_id} with columns: {available_cols}")
        
        return signal_tables
    
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
            
            # Count assets in classified_data
            assets_count = len(data.get('assets', []))
            classified_assets = len(data.get('classified_data', {}))
            
            # Create portfolio entry
            entry = {
                'Name': name,
                'Created': created_at,
                'Updated': updated_at,
                'Summary': '; '.join(summary),
                'Assets': assets_count
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
        
        # Return a copy without the classified_data to avoid performance issues
        portfolio = self.portfolios[portfolio_name].copy()
        if 'classified_data' in portfolio:
            # Replace classified_data with just the asset IDs
            portfolio['classified_data_assets'] = list(portfolio['classified_data'].keys())
            del portfolio['classified_data']
            
        return portfolio
    
    def set_basket_criteria(self, criteria):
        """
        Set criteria for filtering the asset basket.
        
        Args:
            criteria (dict): Criteria for filtering assets
        """
        self.basket_criteria = criteria
        self.logger.info(f"Set basket criteria: {criteria}")
    
    def get_portfolio(self, portfolio_name):
        """
        Get a portfolio with the specified name.
        
        Args:
            portfolio_name (str): Name of the portfolio to retrieve
            
        Returns:
            dict: Portfolio object with its criteria, or None if not found
        """
        try:
            # First check if it's in the manager's portfolio cache
            if hasattr(self, 'portfolios') and portfolio_name in self.portfolios:
                self.logger.info(f"Found portfolio '{portfolio_name}' in memory cache")
                portfolio = self.portfolios[portfolio_name]
                
                # Format the portfolio data structure
                if 'criteria' not in portfolio:
                    # The portfolio itself is likely the criteria
                    portfolio = {
                        'name': portfolio_name,
                        'criteria': portfolio
                    }
                
                return portfolio
                
            # Otherwise, try to load portfolio details from database
            portfolio_path = os.path.join(self.portfolios_dir, f"{portfolio_name}.json")
            if not os.path.exists(portfolio_path):
                self.logger.error(f"Portfolio '{portfolio_name}' not found in database")
                return None
                
            # Get portfolio details
            with open(portfolio_path, 'r') as f:
                portfolio_details = json.load(f)
            
            if not portfolio_details:
                self.logger.error(f"Failed to retrieve details for portfolio '{portfolio_name}'")
                return None
                
            # Create a formatted portfolio object with criteria
            criteria = portfolio_details.get('criteria', {})
            if not criteria and 'usd_conditions' in portfolio_details:
                # The portfolio itself might contain the criteria
                criteria = portfolio_details
                
            portfolio = {
                'name': portfolio_name,
                'criteria': criteria
            }
            
            # Cache the portfolio for future use
            if not hasattr(self, 'portfolios'):
                self.portfolios = {}
            self.portfolios[portfolio_name] = portfolio
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error retrieving portfolio '{portfolio_name}': {str(e)}")
            return None 