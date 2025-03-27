import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os

# Import all submodules
try:
    # Try relative imports (for package usage)
    from .data.data_loader import DataLoader
    from .data.ssr_data import SSRDataHandler
    from .indicators.rsi import RSICalculator
    from .indicators.moving_averages import MovingAverageCalculator
    from .indicators.trend_classifier import TrendClassifier
    from .analysis.metrics import MetricsCalculator
    from .analysis.markov import MarkovAnalyzer
    from .portfolio.portfolio_manager import PortfolioManager
    from .portfolio.backtest import PortfolioBacktester
    from .visualization.plotters import PortfolioVisualizer
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
    
    def __init__(self, asset_ids, data_path, btc_data_path, use_btc_adjusted=True, verbose=True, 
                 lookback_days=90, ssr_data_path=None, markov_model_name=None):
        """
        Initialize the TrendAnalyzer.
        
        Args:
            asset_ids (list): List of asset IDs to analyze
            data_path (str): Path to the directory containing asset data files
            btc_data_path (str): Path to the Bitcoin data file
            use_btc_adjusted (bool, optional): Whether to use BTC-adjusted prices. Defaults to True.
            verbose (bool, optional): Whether to output verbose logs. Defaults to True.
            lookback_days (int, optional): Lookback period for analysis and trend metrics. Defaults to 90.
            ssr_data_path (str, optional): Path to the SSR data file. Defaults to None.
            markov_model_name (str, optional): Name of the Markov model to use. Defaults to None.
        """
        # Initialize logger
        self._setup_logging(verbose)
        
        self.logger.info("Initializing TrendAnalyzer...")
        
        # Store configuration
        self.asset_ids = asset_ids
        self.use_btc_adjusted = use_btc_adjusted
        self.lookback_days = lookback_days
        
        # Initialize components
        self.data_loader = DataLoader(data_path, btc_data_path, ssr_data_path)
        self.logger.info(f"Initialized DataLoader with {len(asset_ids)} assets")
        
        if ssr_data_path:
            self.ssr_handler = SSRDataHandler(ssr_data_path)
            self.logger.info(f"Initialized SSRDataHandler with data from {ssr_data_path}")
        else:
            self.ssr_handler = None
        
        self.rsi_calculator = RSICalculator()
        self.ma_calculator = MovingAverageCalculator()
        self.trend_classifier = TrendClassifier(self.ma_calculator)
        self.metrics_calculator = MetricsCalculator()
        
        # Initialize Markov components
        markov_model = None
        if markov_model_name:
            try:
                # Import MarkovVolatility only if needed to avoid circular imports
                from markov_volatility import MarkovVolatility
                markov_model = MarkovVolatility.load_model(markov_model_name)
                self.logger.info(f"Loaded Markov volatility model: {markov_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load Markov volatility model: {e}")
        
        self.markov_analyzer = MarkovAnalyzer(markov_model)
        
        # Initialize portfolio components
        self.portfolio_manager = PortfolioManager()
        self.backtester = PortfolioBacktester(self.portfolio_manager, self.data_loader)
        self.visualizer = PortfolioVisualizer()
        
        # Initialize cache for processed data
        self.classified_data = {}
        self.raw_data = {}
        self.transition_matrices = {}
        self.trend_metrics = {}
        
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
    
    def analyze_asset(self, asset_id):
        """
        Analyze a single asset and generate trend classifications.
        
        Args:
            asset_id (str): ID of the asset to analyze
            
        Returns:
            pd.DataFrame: DataFrame with trend classifications
        """
        self.logger.info(f"Analyzing asset: {asset_id}")
        
        # Load data
        data = self.data_loader.load_asset_data(asset_id)
        if data.empty:
            self.logger.warning(f"No data available for {asset_id}")
            return pd.DataFrame()
        
        # Store raw data
        self.raw_data[asset_id] = data.copy()
        
        # Make sure asset_id is in the data for reference
        if 'asset_id' not in data.columns:
            data['asset_id'] = asset_id
        
        # Calculate RSI for USD prices
        if 'close' in data.columns:
            self.logger.debug(f"Calculating USD RSI for {asset_id}")
            data = self.rsi_calculator.calculate_smooth_rsi(data, 'close')
        else:
            self.logger.warning(f"No 'close' column found for {asset_id}, USD trend analysis may be incomplete")
        
        # Calculate RSI for BTC prices (if applicable)
        btc_price_col = f'{asset_id}_btc'
        if self.use_btc_adjusted and btc_price_col in data.columns:
            self.logger.debug(f"Calculating BTC RSI for {asset_id}")
            data = self.rsi_calculator.calculate_smooth_rsi(data, btc_price_col)
        elif self.use_btc_adjusted:
            self.logger.warning(f"No '{btc_price_col}' column found for {asset_id}, BTC trend analysis will be skipped")
        
        # Verify we have necessary data for trend classification
        usd_cols_exist = 'close' in data.columns
        btc_cols_exist = btc_price_col in data.columns if self.use_btc_adjusted else False
        
        if not usd_cols_exist and not btc_cols_exist:
            self.logger.error(f"Neither USD nor BTC price columns found for {asset_id}, cannot perform trend analysis")
            return pd.DataFrame()
        
        # Classify trends
        self.logger.debug(f"Classifying trends for {asset_id}")
        classified_data = self.trend_classifier.create_classified_data(data, asset_id)
        
        # Verify trend classification
        has_usd_trends = any(col.startswith('Trend_') and col.endswith('_USD') for col in classified_data.columns)
        has_btc_trends = any(col.startswith('Trend_') and col.endswith('_BTC') for col in classified_data.columns)
        
        self.logger.debug(f"{asset_id} USD trend columns found: {has_usd_trends}")
        self.logger.debug(f"{asset_id} BTC trend columns found: {has_btc_trends}")
        
        # Calculate returns for metrics
        if 'close' in classified_data.columns:
            classified_data['returns_usd'] = classified_data['close'].pct_change()
        
        if btc_price_col in classified_data.columns:
            classified_data['returns_btc'] = classified_data[btc_price_col].pct_change()
        
        # Log trend classification stats
        usd_trend_status = 'Overall_Trend_USD' in classified_data.columns
        btc_trend_status = 'Overall_Trend_BTC' in classified_data.columns
        
        self.logger.info(f"{asset_id} trend classification - USD: {usd_trend_status}, BTC: {btc_trend_status}")
        
        # Log data sample for debugging
        if not classified_data.empty:
            trend_cols = [col for col in classified_data.columns if col.startswith('Trend_') or col.startswith('Overall_Trend_')]
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
        
        for asset_id in self.asset_ids:
            if asset_id not in self.classified_data:
                self.analyze_asset(asset_id)
            
            data = self.classified_data.get(asset_id)
            if data is None or data.empty:
                continue
            
            # Calculate metrics for USD trend
            if 'Overall_Trend_USD' in data.columns:
                usd_metrics = self.metrics_calculator.calculate_trend_type_metrics(
                    data, 'USD', self.lookback_days
                )
                if not usd_metrics.empty:
                    trend_metrics[f"{asset_id}_USD"] = usd_metrics
                
                # Calculate transition matrix
                usd_matrix = self.markov_analyzer.calculate_transition_probabilities(
                    data, 'USD', self.lookback_days
                )
                if not usd_matrix.empty:
                    transition_matrices[f"{asset_id}_USD"] = usd_matrix
            
            # Calculate metrics for BTC trend if available
            if 'Overall_Trend_BTC' in data.columns:
                btc_metrics = self.metrics_calculator.calculate_trend_type_metrics(
                    data, 'BTC', self.lookback_days
                )
                if not btc_metrics.empty:
                    trend_metrics[f"{asset_id}_BTC"] = btc_metrics
                
                # Calculate transition matrix
                btc_matrix = self.markov_analyzer.calculate_transition_probabilities(
                    data, 'BTC', self.lookback_days
                )
                if not btc_matrix.empty:
                    transition_matrices[f"{asset_id}_BTC"] = btc_matrix
        
        # Update cache
        self.trend_metrics = trend_metrics
        self.transition_matrices = transition_matrices
        
        return trend_metrics, transition_matrices
    
    def analyze_multiple_assets(self):
        """
        Analyze all assets in the asset_ids list.
        
        Returns:
            dict: Dictionary of DataFrames with trend classifications for each asset
        """
        self.logger.info(f"Analyzing {len(self.asset_ids)} assets...")
        
        results = {}
        for asset_id in self.asset_ids:
            results[asset_id] = self.analyze_asset(asset_id)
        
        # Compute metrics and transition matrices
        self._compute_lookback_metrics()
        
        return results
    
    def get_latest_trends(self):
        """
        Get the latest trend classifications for all assets.
        
        Returns:
            pd.DataFrame: DataFrame with latest trend classifications for all assets
        """
        if not self.classified_data:
            self.analyze_multiple_assets()
        
        latest_trends = []
        
        for asset_id, data in self.classified_data.items():
            if data.empty:
                continue
            
            # Get latest date
            latest_date = data['date'].max() if 'date' in data.columns else data.index.max()
            
            # Get latest row
            latest_row = data[data['date'] == latest_date].iloc[0] if 'date' in data.columns else data.loc[latest_date]
            
            # Get ticker from asset ID
            ticker = self.data_loader.get_ticker_from_id(asset_id)
            if ticker is None:
                # If no ticker mapping exists, use uppercase asset_id as fallback
                ticker = asset_id.upper()
            
            # Extract trend data
            trend_data = {
                'ticker': ticker,  # Use ticker instead of asset_id
                'date': latest_date,
                'close': latest_row['close'] if 'close' in latest_row else None,
            }
            
            # Add only trend columns, skip BTC price columns
            for col in data.columns:
                if col.startswith('Trend_') or col.startswith('Overall_Trend_'):
                    trend_data[col] = latest_row[col]
            
            latest_trends.append(trend_data)
        
        return pd.DataFrame(latest_trends)
    
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
    
    def get_asset_trend_metrics(self, asset_id, price_type='USD'):
        """
        Get trend metrics for an asset.
        
        Args:
            asset_id (str): ID of the asset
            price_type (str, optional): Type of price to analyze ('USD' or 'BTC'). Defaults to 'USD'.
            
        Returns:
            dict: Dictionary with trend metrics for different trend types
        """
        if not self.trend_metrics:
            self._compute_lookback_metrics()
        
        key = f"{asset_id}_{price_type}"
        return self.trend_metrics.get(key, pd.DataFrame())
    
    def get_asset_information(self, asset_id):
        """
        Get comprehensive information about an asset including raw data, processed data, and trend metrics.
        
        Args:
            asset_id (str): ID of the asset
            
        Returns:
            dict: Dictionary with asset information
        """
        if asset_id not in self.asset_ids:
            self.logger.warning(f"Asset {asset_id} not in the list of analyzed assets.")
            return {}
        
        # Ensure asset has been analyzed
        if asset_id not in self.classified_data:
            self.analyze_asset(asset_id)
        
        # Ensure metrics have been computed
        if not self.trend_metrics:
            self._compute_lookback_metrics()
        
        # Gather all information
        info = {
            'asset_id': asset_id,
            'raw_data': self.get_asset_raw_data(asset_id),
            'processed_data': self.get_asset_processed_data(asset_id),
            'trend_metrics_usd': self.get_asset_trend_metrics(asset_id, 'USD'),
            'trend_metrics_btc': self.get_asset_trend_metrics(asset_id, 'BTC'),
            'transition_matrix_usd': self.transition_matrices.get(f"{asset_id}_USD", pd.DataFrame()),
            'transition_matrix_btc': self.transition_matrices.get(f"{asset_id}_BTC", pd.DataFrame())
        }
        
        return info
    
    def create_portfolio(self, portfolio_name, **criteria):
        """
        Create a new portfolio with specified criteria.
        
        Args:
            portfolio_name (str): Name of the portfolio
            **criteria: Criteria for the portfolio
            
        Returns:
            bool: True if portfolio was created successfully, False otherwise
        """
        return self.portfolio_manager.create_portfolio(portfolio_name, **criteria)
    
    def create_roc_based_portfolio(self, portfolio_name, **criteria):
        """
        Create a new portfolio based on Rate of Change (RoC) criteria.
        
        Args:
            portfolio_name (str): Name of the portfolio
            **criteria: RoC-based criteria for the portfolio
            
        Returns:
            bool: True if portfolio was created successfully, False otherwise
        """
        return self.portfolio_manager.create_roc_based_portfolio(portfolio_name, **criteria)
    
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
    
    def backtest_portfolio(self, portfolio_name, start_date, end_date, initial_capital=10000, 
                          alt_cost=0.005, btc_cost=0.001, signal_threshold=75, show_plot=False):
        """
        Backtest a portfolio over a specific time period.
        
        Args:
            portfolio_name (str): Name of the portfolio to backtest
            start_date (str or datetime): Start date for backtesting
            end_date (str or datetime): End date for backtesting
            initial_capital (float, optional): Initial capital for backtesting
            alt_cost (float, optional): Transaction cost for altcoins
            btc_cost (float, optional): Transaction cost for Bitcoin
            signal_threshold (int, optional): Threshold for signal strength
            show_plot (bool, optional): Whether to display the backtest plot
            
        Returns:
            dict: Backtest results
        """
        # Ensure we have analyzed the assets
        if not self.classified_data:
            self.analyze_multiple_assets()
        
        results = self.backtester.backtest_portfolio(
            portfolio_name, start_date, end_date, initial_capital, 
            alt_cost, btc_cost, signal_threshold
        )
        
        # Show plot if requested
        if show_plot and results:
            self.plot_portfolio_performance(results, show_plot=True)
        
        return results
    
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
    
    def plot_portfolio_performance(self, backtest_results, show_plot=True):
        """
        Plot portfolio performance.
        
        Args:
            backtest_results (dict): Results from portfolio backtesting
            show_plot (bool, optional): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        return self.visualizer.plot_portfolio_with_volatility(backtest_results, show_plot=show_plot)
    
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
        return self.visualizer.plot_individual_asset_performance(
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
        
        return self.visualizer.plot_trend_distribution(data, trend_type=trend_type, show_plot=show_plot)
    
    def plot_transition_matrix(self, asset_id, trend_type='USD', show_plot=True):
        """
        Plot transition matrix for an asset.
        
        Args:
            asset_id (str): ID of the asset to plot
            trend_type (str, optional): Type of trend to analyze ('USD' or 'BTC')
            show_plot (bool, optional): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        key = f"{asset_id}_{trend_type}"
        
        if not self.transition_matrices or key not in self.transition_matrices:
            self._compute_lookback_metrics()
        
        matrix = self.transition_matrices.get(key)
        if matrix is None or matrix.empty:
            self.logger.warning(f"No transition matrix available for {key}")
            return None
        
        return self.visualizer.plot_transitions_heatmap(matrix, show_plot=show_plot)
    
    def compare_trend_performance(self, asset_id, trend_term='Overall', price_type='USD'):
        """
        Compare performance of different trend classifications for an asset.
        
        Args:
            asset_id (str): ID of the asset
            trend_term (str, optional): Term of trend to analyze ('Short Term', 'Medium Term', 'Long Term', 'Overall'). 
                                      Defaults to 'Overall'.
            price_type (str, optional): Type of price to analyze ('USD' or 'BTC'). Defaults to 'USD'.
            
        Returns:
            pd.DataFrame: DataFrame comparing performance metrics across trend classifications
        """
        # Ensure asset has been analyzed
        if asset_id not in self.classified_data:
            self.analyze_asset(asset_id)
        
        # Ensure metrics have been computed
        if not self.trend_metrics:
            self._compute_lookback_metrics()
        
        key = f"{asset_id}_{price_type}"
        trend_metrics = self.trend_metrics.get(key)
        
        if trend_metrics is None or trend_metrics.empty:
            self.logger.warning(f"No trend metrics available for {key}")
            return pd.DataFrame()
        
        # Format and return the results
        return trend_metrics[
            ['Trend', 'Description', 'Count', 'Frequency', 'Avg_Return', 'Median_Return', 
             'Return_Skew', 'Sharpe', 'Sortino', 'Max_Drawdown', 'Avg_Duration']
        ].sort_values('Sharpe', ascending=False)
    
    def clear_cache(self):
        """Clear all cached data."""
        self.classified_data = {}
        self.raw_data = {}
        self.transition_matrices = {}
        self.trend_metrics = {}
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