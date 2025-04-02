import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from tqdm.auto import tqdm
import argparse

# Import all submodules
try:
    # Try relative imports (for package usage)
    from .data.data_loader import DataLoader
    from .data.ssr_data import SSRDataHandler #ssr_oscillator 
    from .indicators.rsi import RSICalculator # calculates smoothed RSI indicator for an asset in USD and BTC quotes
    from .indicators.moving_averages import MovingAverageCalculator # calculates moving averages for an asset for USD and BTC quotes
    from .indicators.trend_classifier import TrendClassifier # classifies trends for an asset in USD and BTC quotes
    from .analysis.metrics import MetricsCalculator # calculates metrics for different trend types to facilitate trend analysis and signal generation
    from .analysis.markov import MarkovAnalyzer # calculates transition matrices for different trend types to facilitate trend analysis and signal generation
    from .portfolio.portfolio_manager import PortfolioManager # manages the creation of a strategy based on chosen signals and the available assets  
    from .portfolio.backtest import PortfolioBacktester # backtests the strategy 
    from .visualization.plotters import PortfolioVisualizer # vizualization tooling for the portfolio class
except ImportError:
    # Fall back to absolute imports (for direct script execution)
    import sys
    import os
    
    # Add the project root to sys.path to allow imports to work
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        # Try importing from the current directory
        from data.data_loader import DataLoader
        from data.ssr_data import SSRDataHandler
        from indicators.rsi import RSICalculator
        from indicators.moving_averages import MovingAverageCalculator
        from indicators.trend_classifier import TrendClassifier
        from analysis.metrics import MetricsCalculator
        from analysis.markov import MarkovAnalyzer
        from portfolio.portfolio_manager import PortfolioManager
        from portfolio.backtest import PortfolioBacktester
        from visualization.plotters import PortfolioVisualizer
    except ImportError:
        # If that fails, try importing from the soros_system package
        from soros_system.data.data_loader import DataLoader
        from soros_system.data.ssr_data import SSRDataHandler
        from soros_system.indicators.rsi import RSICalculator
        from soros_system.indicators.moving_averages import MovingAverageCalculator
        from soros_system.indicators.trend_classifier import TrendClassifier
        from soros_system.analysis.metrics import MetricsCalculator
        from soros_system.analysis.markov import MarkovAnalyzer
        from soros_system.portfolio.portfolio_manager import PortfolioManager
        from soros_system.portfolio.backtest import PortfolioBacktester
        from soros_system.visualization.plotters import PortfolioVisualizer

class TrendAnalyzer:
    """
    Main coordinator class that provides a unified API for the modularized components.
    """
    
    def __init__(self, asset_ids=None, data_path=None, btc_data_path=None, lookback_days=30, 
                 ssr_data_path=None, data_loader=None, logger=None, portfolio_manager=None, 
                 backtester=None, plotter=None, metrics_calculator=None, 
                 trend_classifier=None, markov_analyzer=None, markov_vol_model_path=None):
        """
        Initialize the TrendAnalyzer with necessary components.
        
        Args:
            asset_ids (list, optional): List of asset IDs to analyze.
            data_path (str, optional): Path to the data directory containing CSV files.
            btc_data_path (str, optional): Path to the Bitcoin data CSV file.
            lookback_days (int or str, optional): Number of days to look back for metrics or 'all' for all available data.
                Defaults to 30.
            ssr_data_path (str, optional): Path to the SSR data CSV file.
            data_loader (DataLoader, optional): Pre-initialized DataLoader instance.
            logger (logging.Logger, optional): Logger instance.
            portfolio_manager (PortfolioManager, optional): Pre-initialized PortfolioManager instance.
            backtester (PortfolioBacktester, optional): Pre-initialized PortfolioBacktester instance.
            plotter (Plotter, optional): Pre-initialized Plotter instance.
            metrics_calculator (MetricsCalculator, optional): Pre-initialized MetricsCalculator instance.
            trend_classifier (TrendClassifier, optional): Pre-initialized TrendClassifier instance.
            markov_analyzer (MarkovAnalyzer, optional): Pre-initialized MarkovAnalyzer instance for trend analysis.
            markov_vol_model_path (str, optional): Path to a pre-trained MarkovVolatility model.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing TrendAnalyzer...")
        
        # Initialize asset IDs
        self.asset_ids = asset_ids or []
        
        # Set configuration
        self.use_btc_adjusted = True  # Default to using BTC-adjusted prices
        self.skip_metrics_comp = False  # Set default for skipping metrics computation
        self.lookback_days = lookback_days  # Set lookback period for metrics calculations
        
        # Initialize data loading components
        if data_loader:
            self.data_loader = data_loader
        else:
            from soros_system.data.data_loader import DataLoader
            self.data_loader = DataLoader(data_path, btc_data_path, asset_ids=self.asset_ids)
        
        self.logger.info(f"Initialized DataLoader with {len(self.asset_ids)} assets")
        
        # Initialize SSR data handler if path is provided
        self.ssr_handler = None
        if ssr_data_path:
            from soros_system.data.ssr_data import SSRDataHandler
            self.ssr_handler = SSRDataHandler(ssr_data_path)
            self.logger.info(f"Initialized SSRDataHandler with data from {ssr_data_path}")
            
            # Attach SSR handler to data_loader for easy access
            self.data_loader.ssr_handler = self.ssr_handler
        
        # Initialize portfolio management components
        if portfolio_manager:
            self.portfolio_manager = portfolio_manager
        else:
            from soros_system.portfolio.portfolio_manager import PortfolioManager
            # Pass SSR handler and markov_vol_model to PortfolioManager
            self.portfolio_manager = PortfolioManager(ssr_handler=self.ssr_handler)
        
        # Initialize Markov model for trend analysis
        if markov_analyzer is not None and hasattr(markov_analyzer, 'calculate_transition_probabilities'):
            # Only use if it's a valid MarkovAnalyzer instance
            self.markov_analyzer = markov_analyzer
            self.logger.info("Using provided MarkovAnalyzer for trend analysis")
        else:
            # Create a new MarkovAnalyzer instance
            from soros_system.analysis.markov import MarkovAnalyzer
            self.markov_analyzer = MarkovAnalyzer()
            self.logger.info("Created new MarkovAnalyzer for trend analysis")
        
        # Initialize Markov Volatility model for regime detection
        from soros_system.analysis.markov_vol_model import MarkovVolModel
        
        # Check if the provided markov_analyzer is actually a MarkovVolModel
        # This handles backward compatibility if user passed MarkovVolModel to markov_analyzer
        if markov_analyzer is not None and hasattr(markov_analyzer, 'get_volatility_state') and not hasattr(markov_analyzer, 'calculate_transition_probabilities'):
            self.logger.warning("Detected MarkovVolModel passed as markov_analyzer - using it for volatility analysis")
            self.markov_vol_model = markov_analyzer
        else:
            # Create a new MarkovVolModel with the provided path
            self.markov_vol_model = MarkovVolModel(model_path=markov_vol_model_path)
            
        # Pass markov_vol_model to portfolio_manager if not already set
        if self.portfolio_manager.markov_vol_model is None:
            self.portfolio_manager.markov_vol_model = self.markov_vol_model
        
        self.logger.info("Initialized MarkovVolModel for volatility analysis")
        
        # Initialize backtesting components
        if backtester:
            self.backtester = backtester
        else:
            from soros_system.portfolio.backtest import PortfolioBacktester
            # Note: No need to pass markov_analyzer anymore as signal calculation is in PortfolioManager
            self.backtester = PortfolioBacktester(
                self.portfolio_manager, 
                self.data_loader
            )
        
        # Initialize visualization components
        if plotter:
            self.plotter = plotter
        else:
            from soros_system.visualization.plotters import PortfolioVisualizer
            self.plotter = PortfolioVisualizer()
        
        # Initialize analysis components
        if metrics_calculator:
            self.metrics_calculator = metrics_calculator
        else:
            from soros_system.analysis.metrics import MetricsCalculator
            self.metrics_calculator = MetricsCalculator()
        
        # Initialize technical indicators
        from soros_system.indicators.moving_averages import MovingAverageCalculator
        self.ma_calculator = MovingAverageCalculator()
        
        from soros_system.indicators.rsi import RSICalculator
        self.rsi_calculator = RSICalculator()
        
        if trend_classifier:
            self.trend_classifier = trend_classifier
        else:
            from soros_system.indicators.trend_classifier import TrendClassifier
            self.trend_classifier = TrendClassifier(self.ma_calculator)
        
        # Initialize cache for processed data
        self.classified_data = {}
        self.raw_data = {}
        self.transition_matrices = {}
        self.trend_metrics = {}
        self.consolidated_metrics = {}
        
        self.logger.info("TrendAnalyzer initialization complete")
    
    def _setup_logging(self, verbose):
        """
        Set up logging configuration.
        
        Args:
            verbose (bool): Whether to output verbose logs
        """
        log_level = logging.INFO if verbose else logging.WARNING
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('trend_analyzer.log')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _load_data(self, asset_id):
        """
        Load data for a specific asset.
        
        Args:
            asset_id (str): ID of the asset to load
            
        Returns:
            pd.DataFrame: DataFrame with price data
        """
        self.logger.debug(f"Loading data for {asset_id}")
        
        try:
            # Use the data loader to get the asset data
            data = self.data_loader.load_asset_data(asset_id)
            
            if data.empty:
                self.logger.warning(f"No data found for {asset_id}")
            else:
                self.logger.debug(f"Loaded {len(data)} rows for {asset_id}")
                
                # Add asset_id column if not present
                if 'asset_id' not in data.columns:
                    data['asset_id'] = asset_id
                
                # Store raw data in cache
                self.raw_data[asset_id] = data.copy()
            
            return data
        except Exception as e:
            self.logger.error(f"Error loading data for {asset_id}: {str(e)}")
            return pd.DataFrame()
    
    def analyze_asset(self, asset_id):
        """
        Analyze a single asset.
        
        Args:
            asset_id (str): Asset ID to analyze
            
        Returns:
            pd.DataFrame: DataFrame with classified data
        """
        self.logger.debug(f"Starting analysis for {asset_id}")
        
        # See if we have the data cached
        if asset_id in self.classified_data:
            self.logger.debug(f"Using cached data for {asset_id}")
            return self.classified_data[asset_id]
        
        # Load data
        data = self._load_data(asset_id)
        
        if data.empty:
            self.logger.warning(f"No data available for {asset_id}")
            return pd.DataFrame()
        
        # Check if we have price columns
        usd_cols_exist = 'close' in data.columns
        btc_cols_exist = f'{asset_id}_btc' in data.columns
        
        self.logger.debug(f"USD columns exist: {usd_cols_exist}, BTC columns exist: {btc_cols_exist}")
        
        if not usd_cols_exist and not btc_cols_exist:
            self.logger.error(f"Neither USD nor BTC price columns found for {asset_id}, cannot perform trend analysis")
            return pd.DataFrame()
        
        # Classify trends with progress tracking
        self.logger.debug(f"Classifying trends for {asset_id}")
        with tqdm(total=1, desc=f"Classifying {asset_id}", leave=False) as pbar:
            classified_data = self.trend_classifier.create_classified_data(data, asset_id)
            pbar.update(1)
        
        # Calculate RSI indicators for USD price data
        if usd_cols_exist:
            self.logger.debug(f"Calculating RSI indicators for {asset_id} (USD)")
            try:
                # Calculate RSI with standard length 28
                classified_data = self.rsi_calculator.calculate_smooth_rsi(classified_data, 'close', rsi_length=28, roc_length=28)
                
                # Rename standard RSI columns for consistency and drop intermediate columns
                if 'RSI_close' in classified_data.columns:
                    # Only keep the signal column
                    classified_data['RSI_Signal_28_USD'] = classified_data['RSI_Signal_close']
                    # Drop intermediate columns
                    classified_data.drop(['RSI_close', 'RoC_close', 'RSI_Signal_close'], axis=1, inplace=True, errors='ignore')
                else:
                    self.logger.warning(f"Expected RSI columns not found after calculation for {asset_id} (USD)")
            except ValueError as e:
                self.logger.error(f"Error calculating USD RSI for {asset_id} - missing data: {str(e)}")
            except RuntimeError as e:
                self.logger.error(f"Error calculating USD RSI for {asset_id} - runtime error: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error calculating USD RSI for {asset_id}: {str(e)}")
        
        # Calculate RSI indicators for BTC price data (if available)
        btc_price_col = f'{asset_id}_btc'
        if btc_cols_exist:
            self.logger.debug(f"Calculating RSI indicators for {asset_id} (BTC)")
            try:
                # Calculate RSI with standard length 28
                classified_data = self.rsi_calculator.calculate_smooth_rsi(classified_data, btc_price_col, rsi_length=28, roc_length=28)
                
                # Rename standard RSI columns for consistency and drop intermediate columns
                if f'RSI_{btc_price_col}' in classified_data.columns:
                    # Only keep the signal column
                    classified_data['RSI_Signal_28_BTC'] = classified_data[f'RSI_Signal_{btc_price_col}']
                    # Drop intermediate columns
                    classified_data.drop([f'RSI_{btc_price_col}', f'RoC_{btc_price_col}', f'RSI_Signal_{btc_price_col}'], axis=1, inplace=True, errors='ignore')
                else:
                    self.logger.warning(f"Expected RSI columns not found after calculation for {asset_id} (BTC)")
            except ValueError as e:
                self.logger.error(f"Error calculating BTC RSI for {asset_id} - missing data: {str(e)}")
            except RuntimeError as e:
                self.logger.error(f"Error calculating BTC RSI for {asset_id} - runtime error: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error calculating BTC RSI for {asset_id}: {str(e)}")
        
        # Verify trend classification
        has_usd_trends = any(col.startswith('short_term_trend_') and col.endswith('_USD') for col in classified_data.columns)
        has_btc_trends = any(col.startswith('short_term_trend_') and col.endswith('_BTC') for col in classified_data.columns)
        
        self.logger.debug(f"{asset_id} USD trend columns found: {has_usd_trends}")
        self.logger.debug(f"{asset_id} BTC trend columns found: {has_btc_trends}")
        
        # Calculate returns for metrics
        if 'close' in classified_data.columns:
            classified_data['returns_usd'] = classified_data['close'].pct_change()
        
        if btc_price_col in classified_data.columns:
            classified_data['returns_btc'] = classified_data[btc_price_col].pct_change()
        
        # Log trend classification stats
        usd_trend_status = 'overall_trend_USD' in classified_data.columns
        btc_trend_status = 'overall_trend_BTC' in classified_data.columns
        
        self.logger.info(f"{asset_id} trend classification - USD: {usd_trend_status}, BTC: {btc_trend_status}")
        
        # Log data sample for debugging
        if not classified_data.empty:
            trend_cols = [col for col in classified_data.columns if '_trend_' in col and (col.endswith('_USD') or col.endswith('_BTC'))]
            if trend_cols:
                last_row = classified_data.iloc[-1]
                trend_values = {col: last_row[col] for col in trend_cols if col in last_row}
                self.logger.debug(f"{asset_id} latest trend values: {trend_values}")
        
        # Cache results
        self.classified_data[asset_id] = classified_data
        
        return classified_data
    
    def _compute_lookback_metrics(self):
        """
        Compute trend metrics and transition matrices for all assets.
        
        Returns:
            tuple: (trend_metrics, transition_matrices)
        """
        trend_metrics = {}
        transition_matrices = {}
        consolidated_metrics = {}
        
        with tqdm(total=len(self.asset_ids), desc="Computing metrics") as pbar:
            for asset_id in self.asset_ids:
                if asset_id not in self.classified_data:
                    self.analyze_asset(asset_id)
                
                data = self.classified_data.get(asset_id)
                if data is None or data.empty:
                    self.logger.warning(f"No classified data for {asset_id}, skipping metrics computation")
                    pbar.update(1)
                    continue
                
                # Define consistent term mapping using snake_case format
                term_mappings = {
                    'short_term': 'short_term',
                    'medium_term': 'medium_term',
                    'long_term': 'long_term',
                    'overall': 'overall'
                }
                
                # Calculate metrics for USD trend
                if 'overall_trend_USD' in data.columns:
                    self.logger.info(f"Computing USD trend metrics for {asset_id}")
                    
                    # Use the new all-in-one metrics calculation
                    usd_trend_metrics, usd_transition_dfs, usd_consolidated_metrics = self.metrics_calculator.calculate_all_metrics(
                        data, asset_id, 'USD', self.lookback_days)
                    
                    # Process transition matrices
                    for term, transition_df in usd_transition_dfs.items():
                        try:
                            self.logger.debug(f"Processing transition matrix for {asset_id} USD {term}")
                            # Calculate transitions using the MarkovAnalyzer
                            transition_matrix = self.markov_analyzer.calculate_transition_probabilities(
                                transition_df, 'USD', self.lookback_days)
                            
                            if not transition_matrix.empty:
                                # Convert term to snake_case if needed
                                snake_case_term = term_mappings.get(term.lower(), term.lower())
                                transition_matrices[f'{asset_id}_usd_{snake_case_term}'] = transition_matrix
                                self.logger.debug(f"Created transition matrix: {asset_id}_usd_{snake_case_term} with shape {transition_matrix.shape}")
                            else:
                                self.logger.warning(f"Empty transition matrix for {asset_id} USD {term}")
                        except Exception as e:
                            self.logger.error(f"Error calculating transition matrix for {asset_id} USD {term}: {str(e)}")
                            self.logger.exception(e)
                    
                    # Store metrics
                    if not usd_trend_metrics.empty:
                        trend_metrics[f'{asset_id}_usd'] = usd_trend_metrics
                    
                    if not usd_consolidated_metrics.empty:
                        consolidated_metrics[f'{asset_id}_usd'] = usd_consolidated_metrics
                else:
                    self.logger.warning(f"Column overall_trend_USD not found in data for {asset_id}. Skipping USD trend metrics calculation.")
                    self.logger.debug(f"Available columns: {data.columns.tolist()}")
                    # Create empty DataFrames to avoid errors downstream
                    trend_metrics[f'{asset_id}_usd'] = pd.DataFrame()
                    consolidated_metrics[f'{asset_id}_usd'] = pd.DataFrame()
                
                # Calculate metrics for BTC trend
                if 'overall_trend_BTC' in data.columns:
                    self.logger.info(f"Computing BTC trend metrics for {asset_id}")
                    
                    # Use the new all-in-one metrics calculation
                    btc_trend_metrics, btc_transition_dfs, btc_consolidated_metrics = self.metrics_calculator.calculate_all_metrics(
                        data, asset_id, 'BTC', self.lookback_days)
                    
                    # Process transition matrices
                    for term, transition_df in btc_transition_dfs.items():
                        try:
                            self.logger.debug(f"Processing transition matrix for {asset_id} BTC {term}")
                            # Calculate transitions using the MarkovAnalyzer
                            transition_matrix = self.markov_analyzer.calculate_transition_probabilities(
                                transition_df, 'BTC', self.lookback_days)
                            
                            if not transition_matrix.empty:
                                # Convert term to snake_case if needed
                                snake_case_term = term_mappings.get(term.lower(), term.lower())
                                transition_matrices[f'{asset_id}_btc_{snake_case_term}'] = transition_matrix
                                self.logger.debug(f"Created transition matrix: {asset_id}_btc_{snake_case_term} with shape {transition_matrix.shape}")
                            else:
                                self.logger.warning(f"Empty transition matrix for {asset_id} BTC {term}")
                        except Exception as e:
                            self.logger.error(f"Error calculating transition matrix for {asset_id} BTC {term}: {str(e)}")
                            self.logger.exception(e)
                    
                    # Store metrics
                    if not btc_trend_metrics.empty:
                        trend_metrics[f'{asset_id}_btc'] = btc_trend_metrics
                    
                    if not btc_consolidated_metrics.empty:
                        consolidated_metrics[f'{asset_id}_btc'] = btc_consolidated_metrics
                else:
                    self.logger.warning(f"Column overall_trend_BTC not found in data for {asset_id}. Skipping BTC trend metrics calculation.")
                    self.logger.debug(f"Available columns: {data.columns.tolist()}")
                    # Create empty DataFrames to avoid errors downstream
                    trend_metrics[f'{asset_id}_btc'] = pd.DataFrame()
                    consolidated_metrics[f'{asset_id}_btc'] = pd.DataFrame()
                
                pbar.update(1)
        
        # Log summary of metrics computed
        usd_metrics_count = sum(1 for k in trend_metrics.keys() if k.endswith('_usd') and not trend_metrics[k].empty)
        btc_metrics_count = sum(1 for k in trend_metrics.keys() if k.endswith('_btc') and not trend_metrics[k].empty)
        
        usd_matrices_count = sum(1 for k in transition_matrices.keys() if '_usd_' in k)
        btc_matrices_count = sum(1 for k in transition_matrices.keys() if '_btc_' in k)
        
        self.logger.info(f"Computed trend metrics - USD: {usd_metrics_count}, BTC: {btc_metrics_count}")
        self.logger.info(f"Computed transition matrices - USD: {usd_matrices_count}, BTC: {btc_matrices_count}")
        
        # Cache the results
        self.trend_metrics = trend_metrics
        self.transition_matrices = transition_matrices
        self.consolidated_metrics = consolidated_metrics
        
        return trend_metrics, transition_matrices
    
    def analyze_multiple_assets(self):
        """
        Analyze all assets in the TrendAnalyzer's asset_ids list.
        
        Returns:
            list: List of asset IDs successfully analyzed
        """
        # Log start of analysis
        self.logger.info(f"Starting analysis of {len(self.asset_ids)} assets")
        
        # Initialize results
        results = []
        
        # Process each asset with progress bar
        for asset_id in tqdm(self.asset_ids, desc="Analyzing assets"):
            try:
                # Load and analyze the asset
                classified_data = self.analyze_asset(asset_id)
                
                if not classified_data.empty:
                    results.append(asset_id)
            except Exception as e:
                self.logger.error(f"Error analyzing {asset_id}: {str(e)}")
        
        # Log completion
        self.logger.info(f"Completed analysis of {len(results)}/{len(self.asset_ids)} assets")
        
        # Update metrics
        if not self.skip_metrics_comp and len(results) > 0:
            self._compute_lookback_metrics()
        
        return results
    
    
    def get_asset_raw_data(self, asset_id):
        """
        Get raw price data for an asset.
        
        Args:
            asset_id (str): ID of the asset
            
        Returns:
            pd.DataFrame: DataFrame with raw price data
        """
        if asset_id not in self.raw_data:
            data = self.data_loader.load_asset_data(asset_id)
            if not data.empty:
                self.raw_data[asset_id] = data
        
        return self.raw_data.get(asset_id, pd.DataFrame())
    
    def get_asset_processed_data(self, asset_id):
        """
        Get processed data with indicators and trend classifications for an asset.
        
        Args:
            asset_id (str): ID of the asset
            
        Returns:
            pd.DataFrame: DataFrame with processed data
        """
        if asset_id not in self.classified_data:
            self.analyze_asset(asset_id)
        
        return self.classified_data.get(asset_id, pd.DataFrame())
    
    def get_asset_information(self, asset_id, lookback_days=None):
        """
        Get comprehensive information about an asset including raw data, processed data, and trend metrics.
        
        Args:
            asset_id (str): ID of the asset
            lookback_days (int or str, optional): Number of days to look back for metrics calculation
                                                 or 'all' for all available data.
                                                 If None, uses the current class lookback_days setting.
            
        Returns:
            dict: Dictionary with asset information
        """
        if asset_id not in self.asset_ids:
            self.logger.warning(f"Asset {asset_id} not in the list of analyzed assets.")
            return {}
        
        # Ensure asset has been analyzed
        if asset_id not in self.classified_data:
            self.logger.info(f"Asset {asset_id} has not been analyzed yet. Analyzing now...")
            self.analyze_asset(asset_id)
        
        # Get the data for this asset
        data = self.classified_data.get(asset_id)
        if data is None or data.empty:
            self.logger.warning(f"No classified data for {asset_id}")
            return {'asset_id': asset_id}
        
        # Initialize info dictionary with core asset data
        info = {
            'asset_id': asset_id,
            'raw_data': self.get_asset_raw_data(asset_id),
            'processed_data': self.get_asset_processed_data(asset_id),
        }
        
        # Define consistent term mapping using snake_case format
        term_mappings = {
            'short_term': 'short_term',
            'medium_term': 'medium_term',
            'long_term': 'long_term',
            'overall': 'overall'
        }
        
        # If lookback_days is provided, calculate metrics with that value 
        if lookback_days is not None:
            self.logger.info(f"Computing metrics for {asset_id} with lookback period: {lookback_days}")
            
            # Calculate metrics for USD trend
            if 'overall_trend_USD' in data.columns:
                # Use the all-in-one metrics calculation with the provided lookback_days
                usd_trend_metrics, usd_transition_dfs, usd_consolidated_metrics = self.metrics_calculator.calculate_all_metrics(
                    data, asset_id, 'USD', lookback_days)
                
                # Add metrics to info dictionary
                if not usd_trend_metrics.empty:
                    info['trend_metrics_usd'] = usd_trend_metrics
                
                if not usd_consolidated_metrics.empty:
                    info['consolidated_trend_metrics_usd'] = usd_consolidated_metrics
                
                # Process transition matrices
                for term, transition_df in usd_transition_dfs.items():
                    try:
                        # Calculate transitions using the MarkovAnalyzer with the provided lookback_days
                        transition_matrix = self.markov_analyzer.calculate_transition_probabilities(
                            transition_df, 'USD', lookback_days)
                        
                        if not transition_matrix.empty:
                            # Convert term to snake_case if needed
                            snake_case_term = term_mappings.get(term.lower(), term.lower())
                            # Use consistent naming with underscores
                            matrix_key = f"transition_matrix_usd_{snake_case_term}"
                            info[matrix_key] = transition_matrix
                    except Exception as e:
                        self.logger.error(f"Error calculating transition matrix for {asset_id} USD {term}: {str(e)}")
            
            # Calculate metrics for BTC trend
            if 'overall_trend_BTC' in data.columns:
                # Use the all-in-one metrics calculation with the provided lookback_days
                btc_trend_metrics, btc_transition_dfs, btc_consolidated_metrics = self.metrics_calculator.calculate_all_metrics(
                    data, asset_id, 'BTC', lookback_days)
                
                # Add metrics to info dictionary
                if not btc_trend_metrics.empty:
                    info['trend_metrics_btc'] = btc_trend_metrics
                
                if not btc_consolidated_metrics.empty:
                    info['consolidated_trend_metrics_btc'] = btc_consolidated_metrics
                
                # Process transition matrices
                for term, transition_df in btc_transition_dfs.items():
                    try:
                        # Calculate transitions using the MarkovAnalyzer with the provided lookback_days
                        transition_matrix = self.markov_analyzer.calculate_transition_probabilities(
                            transition_df, 'BTC', lookback_days)
                        
                        if not transition_matrix.empty:
                            # Convert term to snake_case if needed
                            snake_case_term = term_mappings.get(term.lower(), term.lower())
                            # Use consistent naming with underscores
                            matrix_key = f"transition_matrix_btc_{snake_case_term}"
                            info[matrix_key] = transition_matrix
                    except Exception as e:
                        self.logger.error(f"Error calculating transition matrix for {asset_id} BTC {term}: {str(e)}")
        else:
            # Use the already computed metrics with the class lookback_days value
            
            # Ensure metrics have been computed
            if not self.trend_metrics:
                self.logger.info(f"Metrics have not been computed yet. Computing now...")
                self._compute_lookback_metrics()
            
            # Log available transition matrices for this asset
            transition_keys = [k for k in self.transition_matrices.keys() if asset_id in k]
            self.logger.info(f"Available transition matrix keys for {asset_id}: {transition_keys}")
            
            # Add consolidated metrics (new format) if not empty
            usd_consolidated = self.consolidated_metrics.get(f"{asset_id}_usd", pd.DataFrame())
            if not usd_consolidated.empty:
                info['consolidated_trend_metrics_usd'] = usd_consolidated
                
            btc_consolidated = self.consolidated_metrics.get(f"{asset_id}_btc", pd.DataFrame())
            if not btc_consolidated.empty:
                info['consolidated_trend_metrics_btc'] = btc_consolidated
            
            # Add overall metrics if not empty
            usd_metrics = self.trend_metrics.get(f"{asset_id}_usd", pd.DataFrame())
            if not usd_metrics.empty:
                info['trend_metrics_usd'] = usd_metrics
                
            btc_metrics = self.trend_metrics.get(f"{asset_id}_btc", pd.DataFrame())
            if not btc_metrics.empty:
                info['trend_metrics_btc'] = btc_metrics
            
            # Define term mapping with consistent underscore format
            term_mapping = {
                'short_term': ['ShortTerm', 'shortterm', 'short_term'],
                'medium_term': ['MediumTerm', 'mediumterm', 'medium_term'],
                'long_term': ['LongTerm', 'longterm', 'long_term'],
                'overall': ['Overall', 'overall']
            }
            
            # Add transition matrices with consistent naming
            for display_term, possible_keys in term_mapping.items():
                # Try to find transition matrices for both USD and BTC
                for quote in ['usd', 'btc']:
                    # First check if any transition matrix exists with any of the possible key formats
                    transition_matrix = None
                    
                    # Try all possible format variations
                    for key_format in possible_keys:
                        matrix_key = f"{asset_id}_{quote}_{key_format}"
                        if matrix_key in self.transition_matrices and not self.transition_matrices[matrix_key].empty:
                            transition_matrix = self.transition_matrices[matrix_key]
                            break
                    
                    if transition_matrix is not None and not transition_matrix.empty:
                        # Use consistent naming with underscores
                        matrix_key = f"transition_matrix_{quote}_{display_term}"
                        info[matrix_key] = transition_matrix
        
        # Log summary of returned information
        matrix_keys = [k for k in info.keys() if k.startswith('transition_matrix')]
        self.logger.info(f"Transition matrices included for {asset_id}: {matrix_keys}")
        
        return info
    
    def create_portfolio(self, portfolio_name, **criteria):
        """
        Create a new portfolio with specified criteria.
        
        Args:
            portfolio_name (str): Name of the portfolio
            **criteria: Criteria for the portfolio including:
                usd_conditions (dict/list): Dict mapping trend terms to valid values or list of valid overall trend values
                    Example: {'Short Term': [1, 2], 'Overall': [1, 2]} or [1, 2]
                btc_conditions (dict/list): Dict mapping trend terms to valid values or list of valid overall trend values (for altcoins)
                btc_trend_gating (dict/list): Dict mapping trend terms to valid values for gating or list of valid overall trend values
                gate_mode (str): How to apply btc_trend_gating when multiple terms - 'all', 'any', or 'majority' (default: 'all')
                
                # RSI parameters
                rsi_conditions_usd (bool/dict): Whether/how to apply RSI filter for USD trends
                rsi_conditions_btc (bool/dict): Whether/how to apply RSI filter for BTC trends (for altcoins)
                
                # Signal calculation parameters
                signal_threshold (float): Threshold for signals to trigger trades (0-100, default 50)
                
                # Volatility filter parameters
                use_volatility_filter (bool): Whether to apply volatility filtering
                
                # SSR parameters
                use_ssr_signal (bool): Whether to use SSR signal
                use_ssr_gate (bool): Whether to use SSR as gate
                
                # Trading parameters
                btc_only (bool): Whether to only include BTC in the portfolio
                
                # Transaction costs
                alt_cost (float): Transaction cost for altcoins (default: 0.005 or 0.5%)
                btc_cost (float): Transaction cost for Bitcoin (default: 0.001 or 0.1%)
            
        Returns:
            bool: True if portfolio was created successfully, False otherwise
        """
        # Ensure needed assets have been analyzed
        assets_to_analyze = set(self.asset_ids)
        if 'btc_trend_gating' in criteria and 'bitcoin' not in assets_to_analyze:
            assets_to_analyze.add('bitcoin')
            self.logger.info("Adding Bitcoin to analyzed assets for BTC gating")
        
        # Analyze all needed assets if not already analyzed
        for asset_id in assets_to_analyze:
            if asset_id not in self.classified_data:
                self.analyze_asset(asset_id)
        
        # Collect all classified data
        classified_data = {}
        for asset_id in assets_to_analyze:
            if asset_id in self.classified_data and not self.classified_data[asset_id].empty:
                classified_data[asset_id] = self.classified_data[asset_id].copy()
            else:
                self.logger.warning(f"No classified data for {asset_id}, skipping")
        
        # Pass ssr_handler and markov_vol_model to PortfolioManager if not already set
        if self.portfolio_manager.ssr_handler is None and hasattr(self, 'ssr_handler'):
            self.portfolio_manager.ssr_handler = self.ssr_handler
        
        if self.portfolio_manager.markov_vol_model is None and hasattr(self, 'markov_vol_model'):
            self.portfolio_manager.markov_vol_model = self.markov_vol_model
        
        # Create the portfolio with classified data
        result = self.portfolio_manager.create_portfolio(
            portfolio_name, 
            classified_data=classified_data,
            **criteria
        )
        
        return result
    
    def list_portfolios(self):
        """
        List all available portfolios.
        
        Returns:
            pd.DataFrame: DataFrame with portfolio details
        """
        return self.portfolio_manager.list_portfolios()
    
    def delete_portfolio(self, portfolio_name):
        """
        Delete a portfolio.
        
        Args:
            portfolio_name (str): Name of the portfolio to delete
            
        Returns:
            bool: True if portfolio was deleted successfully, False otherwise
        """
        return self.portfolio_manager.delete_portfolio(portfolio_name)
    
    def get_portfolio_details(self, portfolio_name):
        """
        Get detailed information about a portfolio.
        
        Args:
            portfolio_name (str): Name of the portfolio
            
        Returns:
            dict: Portfolio details, or None if portfolio doesn't exist
        """
        return self.portfolio_manager.get_portfolio_details(portfolio_name)
    
    def backtest_portfolio(self, portfolio_name, start_date=None, end_date=None, initial_capital=10000, 
                          signal_threshold=50, btc_cost=0.001, alt_cost=0.005, slippage_pct=0.0):
        """
        Backtest a portfolio strategy.
        
        Args:
            portfolio_name (str): Name of the portfolio to backtest
            start_date (str or datetime, optional): Start date for backtesting
            end_date (str or datetime, optional): End date for backtesting
            initial_capital (float, optional): Initial capital for backtesting
            signal_threshold (float, optional): Threshold for signals to trigger trades (0-100)
            btc_cost (float, optional): Fee rate for Bitcoin transactions (default: 0.001)
            alt_cost (float, optional): Fee rate for altcoin transactions (default: 0.005)
            slippage_pct (float, optional): Slippage percentage for trading (default: 0.0)
            
        Returns:
            dict: Dictionary with backtest results
        """
        # Convert dates to datetime if needed
        if start_date is None:
            start_date = pd.Timestamp.now() - pd.Timedelta(days=365)  # Default to 1 year ago
        if end_date is None:
            end_date = pd.Timestamp.now()
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Ensure portfolio exists
        portfolio = self.portfolio_manager.get_portfolio(portfolio_name)
        if not portfolio:
            self.logger.error(f"Portfolio '{portfolio_name}' not found.")
            return None
        
        # Get portfolio criteria
        portfolio_criteria = portfolio.get('criteria', {})
        # Override signal_threshold if specified in portfolio criteria
        if 'signal_threshold' in portfolio_criteria:
            signal_threshold = portfolio_criteria.get('signal_threshold')
            self.logger.info(f"Using signal threshold from portfolio criteria: {signal_threshold}")
        
        # Run backtest using the portfolio manager's signals
        backtest_results = self.backtester.backtest_portfolio(
            portfolio_name=portfolio_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            btc_cost=btc_cost,
            alt_cost=alt_cost,
            slippage_pct=slippage_pct
        )
        
        # Check and process results
        if backtest_results is not None:
            self.logger.info(f"Successfully completed backtest for portfolio {portfolio_name}")
            return backtest_results
        else:
            self.logger.error(f"Backtest failed for portfolio {portfolio_name}")
            return None
    
    def get_portfolio_signals_df(self, backtest_results):
        """
        Get a DataFrame of signals for each asset in a portfolio.
        
        Args:
            backtest_results (dict): Results from portfolio backtesting
            
        Returns:
            pd.DataFrame: DataFrame with signals for each asset
        """
        if not backtest_results or 'signals' not in backtest_results:
            self.logger.warning("No signal data available in backtest results.")
            return pd.DataFrame()
        
        signals = backtest_results['signals']
        
        # Convert signals to DataFrame
        signals_df = pd.DataFrame(signals)
        
        # Pivot to have dates as index and assets as columns
        if 'date' in signals_df.columns and 'asset' in signals_df.columns:
            signals_df = signals_df.pivot(
                index='date', 
                columns='asset', 
                values=['usd_trend_signal', 'btc_trend_signal', 'rsi_usd_signal', 
                       'rsi_btc_signal', 'volatility_signal', 'ssr_signal', 
                       'ssr_gate', 'btc_gate', 'btc_rsi_gate', 'btc_rsi_signal',
                       'followed_portfolio_signal', 'combined_signal', 'final_decision']
            )
            
            # Flatten multi-level columns
            signals_df.columns = [f"{asset}_{signal}" for signal, asset in signals_df.columns]
        
        return signals_df
    
    def get_portfolio_asset_metrics(self, backtest_results):
        """
        Get performance metrics for each asset in a portfolio.
        
        Args:
            backtest_results (dict): Results from portfolio backtesting
            
        Returns:
            dict: Dictionary with performance metrics for each asset
        """
        if not backtest_results or 'asset_performance' not in backtest_results:
            self.logger.warning("No asset performance data available in backtest results.")
            return {}
        
        return backtest_results['asset_performance']
    
    def get_portfolio_daily_results(self, backtest_results):
        """
        Get daily results for a portfolio.
        
        Args:
            backtest_results (dict): Results from portfolio backtesting
            
        Returns:
            pd.DataFrame: DataFrame with daily results
        """
        if not backtest_results or 'dates' not in backtest_results:
            self.logger.warning("No daily results available in backtest results.")
            return pd.DataFrame()
        
        # Create DataFrame with daily results
        results_df = pd.DataFrame({
            'date': backtest_results['dates'],
            'portfolio_value': backtest_results['portfolio_values'],
            'daily_return': backtest_results['returns']
        })
        
        # Add holdings for each date
        for i, holdings in enumerate(backtest_results.get('holdings', [])):
            for asset, amount in holdings.items():
                if asset != 'cash':
                    col_name = f"{asset}_position"
                    if col_name not in results_df.columns:
                        results_df[col_name] = 0
                    
                    results_df.loc[i, col_name] = 1 if amount > 0 else 0
        
        return results_df
    
    def get_recommended_assets_table(self, backtest_results):
        """
        Get a table showing which assets were recommended on each date.
        
        Args:
            backtest_results (dict): Results from portfolio backtesting
            
        Returns:
            pd.DataFrame: DataFrame with recommended assets
        """
        if not backtest_results or 'dates' not in backtest_results:
            self.logger.warning("No data available for recommended assets table.")
            return pd.DataFrame()
        
        # Extract dates and holdings
        dates = backtest_results['dates']
        holdings = backtest_results.get('holdings', [])
        
        if not dates or not holdings or len(dates) != len(holdings):
            self.logger.warning("Inconsistent data for recommended assets table.")
            return pd.DataFrame()
        
        # Get all unique assets across all holdings
        all_assets = set()
        for holding in holdings:
            for asset in holding.keys():
                if asset != 'cash':
                    all_assets.add(asset)
        
        # Create DataFrame with recommendations
        data = {'date': dates}
        for asset in all_assets:
            data[asset] = [1 if asset in h and h[asset] > 0 else 0 for h in holdings]
        
        return pd.DataFrame(data)
    
    def predict_volatility_for_new_data(self, btc_price=None, log_return=None, prediction_date=None):
        """
        Predict volatility state for new BTC data point (latest close).
        
        This method is designed for daily updates and real-time signal generation,
        making actual predictions for new data points not in the original training set.
        
        Args:
            btc_price (float, optional): Latest BTC closing price. If provided, log return will be calculated
                                        using the previous price from the model. Either btc_price or log_return is required.
            log_return (float, optional): Pre-calculated log return value. If provided, btc_price is ignored.
            prediction_date (str or datetime, optional): Date for the prediction. If None, current date is used.
            
        Returns:
            dict: Dictionary containing predicted volatility information:
                 - 'state': String volatility state ('high' or 'low')
                 - 'signal': Numeric volatility signal (1 for low, -1 for high)
                 - 'probability': Probability of high volatility regime (between 0 and 1)
                 - 'date': The prediction date
        """
        # Check if MarkovVolModel is available
        if not hasattr(self, 'markov_vol_model') or self.markov_vol_model is None:
            self.logger.error("No MarkovVolModel available for prediction")
            return {
                'state': 'low',  # Default to low volatility
                'signal': 1,
                'probability': 0.0,
                'date': prediction_date or pd.Timestamp.now().strftime('%Y-%m-%d')
            }
        
        # Ensure we have either btc_price or log_return
        if btc_price is None and log_return is None:
            self.logger.error("Either btc_price or log_return must be provided")
            return {
                'state': 'low',  # Default to low volatility
                'signal': 1,
                'probability': 0.0,
                'date': prediction_date or pd.Timestamp.now().strftime('%Y-%m-%d')
            }
            
        # Make prediction using the MarkovVolModel
        self.logger.info(f"Predicting volatility for new data: price={btc_price}, log_return={log_return}, date={prediction_date}")
        state, signal, probability = self.markov_vol_model.predict_volatility_for_new_data(
            price_data=btc_price,
            log_return=log_return,
            current_date=prediction_date
        )
        
        prediction_date = prediction_date or pd.Timestamp.now()
        if isinstance(prediction_date, str):
            prediction_date = pd.to_datetime(prediction_date)
            
        # Return prediction as a dictionary
        prediction = {
            'state': state,
            'signal': signal,
            'probability': probability,
            'date': prediction_date.strftime('%Y-%m-%d')
        }
        
        self.logger.info(f"Volatility prediction: {prediction}")
        return prediction
    
    def update_volatility_model_with_new_data(self, new_data, retrain=False):
        """
        Update the volatility model with new BTC price data.
        
        Args:
            new_data (pd.DataFrame or str): New data as DataFrame or path to CSV file.
                If DataFrame, must have 'date' and 'close' columns.
            retrain (bool): Whether to retrain the model with the combined data.
                If False, will only append data for lookup but not retrain the model.
                
        Returns:
            bool: True if update was successful
        """
        # Check if MarkovVolModel is available
        if not hasattr(self, 'markov_vol_model') or self.markov_vol_model is None:
            self.logger.error("No MarkovVolModel available to update")
            return False
            
        self.logger.info(f"Updating volatility model with new data (retrain={retrain})")
        success = self.markov_vol_model.update_with_new_data(new_data, retrain=retrain)
        
        if success:
            self.logger.info("Successfully updated volatility model")
        else:
            self.logger.error("Failed to update volatility model")
            
        return success
    
    def plot_portfolio_performance(self, backtest_results, show_plot=True):
        """
        Plot portfolio performance.
        
        Args:
            backtest_results (dict): Results from portfolio backtesting
            show_plot (bool, optional): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        return self.plotter.plot_portfolio_with_volatility(backtest_results, show_plot=show_plot)
    
    def plot_asset_performance(self, backtest_results, asset_filter=None, show_plot=True):
        """
        Plot performance of individual assets in a portfolio.
        
        Args:
            backtest_results (dict): Results from portfolio backtesting
            asset_filter (list, optional): List of asset IDs to include
            show_plot (bool, optional): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        return self.plotter.plot_individual_asset_performance(
            backtest_results, asset_filter=asset_filter, show_plot=show_plot
        )
    
    def plot_trend_distribution(self, asset_id, trend_type='USD', show_plot=True):
        """
        Plot distribution of trend classifications for an asset.
        
        Args:
            asset_id (str): ID of the asset to plot
            trend_type (str, optional): Type of trend to analyze ('USD' or 'BTC')
            show_plot (bool, optional): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if asset_id not in self.classified_data:
            self.analyze_asset(asset_id)
        
        data = self.classified_data.get(asset_id)
        if data is None or data.empty:
            self.logger.warning(f"No data available for {asset_id}")
            return None
        
        return self.plotter.plot_trend_distribution(data, trend_type=trend_type, show_plot=show_plot)
    
    def clear_cache(self):
        """
        Clear all cached data.
        """
        self.raw_data = {}
        self.classified_data = {}
        self.transition_matrices = {}
        self.trend_metrics = {}
        self.consolidated_metrics = {}
        self.data_loader.clear_cache()
        self.markov_analyzer.clear_cache()
        
        self.logger.info("Cleared all cached data")
    
    def get_ticker_from_id(self, asset_id):
        """
        Get ticker symbol for a given asset ID.
        
        Args:
            asset_id (str): Asset ID
            
        Returns:
            str: Ticker symbol
        """
        return self.data_loader.get_ticker_from_id(asset_id)
    
    def run_cli(self):
        """
        Run the command-line interface.
        """
        parser = argparse.ArgumentParser(description='Soros System CLI')
        subparsers = parser.add_subparsers(dest='command', help='Command to run')
        
        # Add subparsers
        self._add_backtest_parser(subparsers)
        self._add_analyze_parser(subparsers)
        self._add_portfolio_parser(subparsers)
        self._add_sync_parser(subparsers)
        
        # Parse arguments
        args = parser.parse_args()
        
        # Process commands
        if args.command == 'backtest':
            start_date = args.start_date if hasattr(args, 'start_date') else None
            end_date = args.end_date if hasattr(args, 'end_date') else None
            initial_capital = args.initial_capital if hasattr(args, 'initial_capital') else 10000
            
            print(f"Starting backtest for portfolio {args.portfolio} from {start_date} to {end_date}")
            print(f"Initial capital: ${initial_capital}")
            
            # Create the portfolio if it doesn't exist yet
            if args.portfolio == 'btc_trend_follower':
                # Check if portfolio already exists
                portfolios = self.list_portfolios()
                portfolio_exists = False
                
                # Check if the portfolio exists in the list
                if isinstance(portfolios, pd.DataFrame) and 'name' in portfolios.columns:
                    portfolio_exists = args.portfolio in portfolios['name'].values
                
                if not portfolio_exists:
                    print(f"Creating portfolio {args.portfolio}...")
                    
                    # Create a customizable portfolio with explicit parameters
                    self.create_portfolio(
                        'btc_trend_follower',
                        # Trend conditions
                        usd_conditions={'Short Term': [1, 2], 'Overall': [1, 2]},
                        
                        # Custom signal weights
                        signal_weights={
                            'usd_trend': 0.7,  # Emphasize trend more
                            'btc_trend': 0.0,
                            'rsi': 0.2,  # Reduce RSI influence
                            'volatility': 0.1
                        },
                        
                        # RSI configuration
                        rsi_conditions_usd={
                            'oversold': 30,
                            'overbought': 70,
                            'oversold_signal': 100,  # Full bullish signal when oversold
                            'overbought_signal': -50,  # Partial bearish signal when overbought
                            'neutral_signal_function': 'linear'  # Linear interpolation between extremes
                        },
                        
                        # Volatility filter
                        use_volatility_filter=True,
                        volatility_params={
                            'max_volatility': 80,  # Lower max volatility threshold
                            'min_volatility': 20,
                            'max_signal': 100,
                            'min_signal': 0
                        },
                        
                        # Trading parameters
                        trade_mode='all_in',  # All-in/all-out trading
                        position_sizing=1.0,
                        scale_in=False,
                        scale_out=False,
                        min_trade_size=0.01,  # Minimum 1% position size
                        
                        # Decision criteria
                        decision_criteria='threshold',
                        
                        # Transaction costs
                        alt_cost=0.005,
                        btc_cost=0.001,
                        
                        # Portfolio type
                        btc_only=True
                    )
                    print(f"Portfolio {args.portfolio} created successfully.")
                else:
                    print(f"Portfolio {args.portfolio} already exists.")
            
            # Parse additional backtest parameters
            start_date = args.start_date if hasattr(args, 'start_date') else "2022-01-01"
            end_date = args.end_date if hasattr(args, 'end_date') else "2023-12-31"
            initial_capital = float(args.initial_capital) if hasattr(args, 'initial_capital') else 10000.0
            signal_threshold = int(args.signal_threshold) if hasattr(args, 'signal_threshold') else 50
            
            print(f"Running backtest for {args.portfolio} from {start_date} to {end_date}")
            print(f"Initial capital: ${initial_capital}, Signal threshold: {signal_threshold}")
            
            # Run backtest
            result = self.backtest_portfolio(
                args.portfolio, 
                start_date, 
                end_date, 
                initial_capital,
                signal_threshold
            )
            
            # Print results
            if result is not None:
                print("\nBacktest completed successfully!")
                print(f"Results dataframe shape: {result['results_df'].shape}")
                print("\nFirst few rows:")
                print(result['results_df'].head())
                print("\nLast few rows:")
                print(result['results_df'].tail())
                
                if 'portfolio_value' in result['results_df'].columns:
                    final_value = result['results_df']['portfolio_value'].iloc[-1]
                    total_return = ((final_value / initial_capital) - 1) * 100
                    
                    print(f"\nInitial investment: ${initial_capital}")
                    print(f"Final value: ${final_value:.2f}")
                    print(f"Total return: {total_return:.2f}%")
                    
                    # Print trading metrics
                    if 'trade_metrics' in result:
                        tm = result['trade_metrics']
                        print(f"\nTrading Metrics:")
                        print(f"Total Trades: {tm.get('total_trades', 0)}")
                        print(f"Win Rate: {tm.get('win_rate', 0):.2f}%")
                        print(f"Average Win: {tm.get('avg_win', 0):.2f}%")
                        print(f"Average Loss: {tm.get('avg_loss', 0):.2f}%")
                else:
                    print("\nERROR: No portfolio_value column in results")
                    print(f"Available columns: {result['results_df'].columns.tolist()}")
            else:
                print("\nERROR: Backtest failed to produce results")
        elif args.command == 'analyze':
            # Handle analyze command
            pass
        elif args.command == 'portfolio':
            # Implementation of portfolio command
            pass
        elif args.command == 'sync':
            # Implementation of sync command
            pass

if __name__ == '__main__':
    import sys
    import os
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
        ]
    )
    
    # Define default data paths
    base_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_path = os.path.join(base_data_path, 'micro', 'candleData')
    btc_data_path = os.path.join(base_data_path, 'micro', 'candleData', 'bitcoin_candles.csv')
    
    print(f"Using data path: {data_path}")
    print(f"Using BTC data path: {btc_data_path}")
    
    # Create instance
    system = TrendAnalyzer(
        data_path=data_path,
        btc_data_path=btc_data_path,
        asset_ids=['bitcoin']  # Init with Bitcoin already in asset_ids
    )
    
    print("Command line arguments:", sys.argv)
    
    # Check if we have arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'backtest' and len(sys.argv) >= 3:
            portfolio_name = sys.argv[2]
            
            # Create the portfolio if it doesn't exist yet
            if portfolio_name == 'btc_trend_follower':
                # Check if portfolio already exists
                portfolios = system.list_portfolios()
                portfolio_exists = False
                
                # Check if the portfolio exists in the list
                if isinstance(portfolios, pd.DataFrame) and 'name' in portfolios.columns:
                    portfolio_exists = portfolio_name in portfolios['name'].values
                
                if not portfolio_exists:
                    print(f"Creating portfolio {portfolio_name}...")
                    
                    # Create a customizable portfolio with explicit parameters
                    system.create_portfolio(
                        'btc_trend_follower',
                        # Trend conditions
                        usd_conditions={'Short Term': [1, 2], 'Overall': [1, 2]},
                        
                        # Custom signal weights
                        signal_weights={
                            'usd_trend': 0.7,  # Emphasize trend more
                            'btc_trend': 0.0,
                            'rsi': 0.2,  # Reduce RSI influence
                            'volatility': 0.1
                        },
                        
                        # RSI configuration
                        rsi_conditions_usd={
                            'oversold': 30,
                            'overbought': 70,
                            'oversold_signal': 100,  # Full bullish signal when oversold
                            'overbought_signal': -50,  # Partial bearish signal when overbought
                            'neutral_signal_function': 'linear'  # Linear interpolation between extremes
                        },
                        
                        # Volatility filter
                        use_volatility_filter=True,
                        volatility_params={
                            'max_volatility': 80,  # Lower max volatility threshold
                            'min_volatility': 20,
                            'max_signal': 100,
                            'min_signal': 0
                        },
                        
                        # Trading parameters
                        trade_mode='all_in',  # All-in/all-out trading
                        position_sizing=1.0,
                        scale_in=False,
                        scale_out=False,
                        min_trade_size=0.01,  # Minimum 1% position size
                        
                        # Decision criteria
                        decision_criteria='threshold',
                        
                        # Transaction costs
                        alt_cost=0.005,
                        btc_cost=0.001,
                        
                        # Portfolio type
                        btc_only=True
                    )
                    print(f"Portfolio {portfolio_name} created successfully.")
                else:
                    print(f"Portfolio {portfolio_name} already exists.")
            
            # Parse additional backtest parameters
            start_date = sys.argv[3] if len(sys.argv) > 3 else "2022-01-01"
            end_date = sys.argv[4] if len(sys.argv) > 4 else "2023-12-31"
            initial_capital = float(sys.argv[5]) if len(sys.argv) > 5 else 10000.0
            signal_threshold = int(sys.argv[6]) if len(sys.argv) > 6 else 50
            
            print(f"Running backtest for {portfolio_name} from {start_date} to {end_date}")
            print(f"Initial capital: ${initial_capital}, Signal threshold: {signal_threshold}")
            
            # Run backtest
            result = system.backtest_portfolio(
                portfolio_name, 
                start_date, 
                end_date, 
                initial_capital,
                signal_threshold
            )
            
            # Print results
            if result is not None:
                print("\nBacktest completed successfully!")
                print(f"Results dataframe shape: {result['results_df'].shape}")
                print("\nFirst few rows:")
                print(result['results_df'].head())
                print("\nLast few rows:")
                print(result['results_df'].tail())
                
                if 'portfolio_value' in result['results_df'].columns:
                    final_value = result['results_df']['portfolio_value'].iloc[-1]
                    total_return = ((final_value / initial_capital) - 1) * 100
                    
                    print(f"\nInitial investment: ${initial_capital}")
                    print(f"Final value: ${final_value:.2f}")
                    print(f"Total return: {total_return:.2f}%")
                    
                    # Print trading metrics
                    if 'trade_metrics' in result:
                        tm = result['trade_metrics']
                        print(f"\nTrading Metrics:")
                        print(f"Total Trades: {tm.get('total_trades', 0)}")
                        print(f"Win Rate: {tm.get('win_rate', 0):.2f}%")
                        print(f"Average Win: {tm.get('avg_win', 0):.2f}%")
                        print(f"Average Loss: {tm.get('avg_loss', 0):.2f}%")
                else:
                    print("\nERROR: No portfolio_value column in results")
                    print(f"Available columns: {result['results_df'].columns.tolist()}")
            else:
                print("\nERROR: Backtest failed to produce results")
        elif command == 'analyze':
            # Handle analyze command
            pass
        else:
            print(f"Unknown command: {command}")
            print("Available commands: backtest, analyze")
    else:
        print("Usage: python -m soros_system.main <command> [args]")
        print("Available commands: backtest, analyze")
        print("\nBacktest example:")
        print("  python -m soros_system.main backtest btc_trend_follower 2022-01-01 2023-12-31 10000 50")
        print("  Arguments: <portfolio_name> <start_date> <end_date> <initial_capital> <signal_threshold>")
        print("\nPortfolio customization example:")
        print("  system.create_portfolio(")
        print("      'custom_portfolio',")
        print("      usd_conditions={'Short Term': [1, 2]},")
        print("      signal_weights={'usd_trend': 0.7, 'rsi': 0.2, 'volatility': 0.1},")
        print("      rsi_conditions_usd={'oversold': 30, 'overbought': 70},")
        print("      trade_mode='proportional',")
        print("      btc_only=True")
        print("  )") 