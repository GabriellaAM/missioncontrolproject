"""
Portfolio analyzer for Soros System.

This module provides a unified interface for interacting with the
Soros System components for daily operations, acting as a facade over
the existing signal, analysis, and portfolio modules.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Import core data structures
from .asset_data import AssetData
from .signal_data import SignalData

# Import existing components
from ..signals.signal_registry import get_signal, get_all_signals, registry, get_signal_names
from ..portfolio.backtester import Backtester
from ..portfolio.portfolio_manager import PortfolioManager
# Import the ensure_signals_loaded module to make sure signals are registered
from ..signals.ensure_signals_loaded import signals as registered_signals


class PortfolioAnalyzer:
    """
    Main interface for the Soros System.
    
    This class serves as a facade over the existing components,
    providing a unified interface for daily operations.
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        data_path: Optional[str] = None,
        btc_data_path: Optional[str] = None,
        ssr_data_path: Optional[str] = None,
        market_data_path: Optional[str] = None,
        markov_analyzer = None,
        signal_threshold: float = 0.0,
        lookback_days: Union[int, str] = 365,
        parallelize: bool = True,
        asset_ids: Optional[List[str]] = None,
        verbose: bool = False,
        num_workers: Optional[int] = None,
        use_meta_labeling: bool = False,
    ):
        """Initialize the portfolio analyzer.
        
        Args:
            data_dir: Base directory for data storage (default path)
            data_path: Path to candle data directory (micro/candleData)
            btc_data_path: Path to Bitcoin candle data file
            ssr_data_path: Path to SSR data file
            market_data_path: Path to market data directory
            markov_analyzer: Pre-loaded Markov volatility model
            signal_threshold: Threshold for final decision (default: 0.0)
            lookback_days: Number of days to look back or 'all' (default: 365)
            parallelize: Whether to use parallel processing (default: True)
            asset_ids: List of asset IDs to initially load
            verbose: Whether to show detailed log messages
            num_workers: Number of worker processes to use for parallel processing.
                         If None, defaults to number of CPU cores minus 1 (minimum 1).
            use_meta_labeling: Whether to use meta-labeling features (placeholder for future implementation)
        """
        # Set up logging first
        self.setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        
        self.data_dir = data_dir
        
        # Store specific data paths
        self.data_path = data_path
        self.btc_data_path = btc_data_path
        self.ssr_data_path = ssr_data_path
        self.market_data_path = market_data_path
        self.markov_analyzer = markov_analyzer
        
        # Store the asset_ids parameter
        self.asset_ids = asset_ids or []
        
        # Configure SSR paths for signals
        if ssr_data_path:
            os.environ['SSR_DATA_PATH'] = ssr_data_path
        
        self.signal_threshold = signal_threshold
        self.lookback_days = lookback_days
        self.parallelize = parallelize
        self.verbose = verbose
        self.use_meta_labeling = use_meta_labeling
        
        # Set number of workers for parallel processing
        if num_workers is None:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_workers = max(1, num_workers)
        self.logger.info(f"Using {self.num_workers} workers for parallel processing")
        
        # Storage for asset data
        self.assets = {}  # {asset_id: AssetData}
        
        # Initialize components
        self.portfolio_manager = PortfolioManager()
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            self.logger.info(f"Created data directory: {data_dir}")
            
        # Initialize cache for loaded data
        self.data_cache = {}
        
        # Signal weights configuration
        self.signal_weights = {}  # {signal_name: weight}
        self.signal_decay_periods = {}  # {signal_name: decay_period}
        
        # Load initial assets if provided
        if asset_ids:
            self.load_data(assets=asset_ids)
            
    def setup_logging(self, verbose=False):
        """Set up logging with appropriate handlers and filters.
        
        Args:
            verbose: Whether to show DEBUG level messages
        """
        # Create a class to filter out repeated messages
        class DuplicateFilter(logging.Filter):
            def __init__(self, name=''):
                super().__init__(name)
                self.seen = set()
                
            def filter(self, record):
                # Check if we've seen this message before
                key = (record.module, record.levelno, record.msg)
                if key in self.seen:
                    return 0  # Don't log
                self.seen.add(key)
                return 1  # Log it
        
        # Get the root logger
        root_logger = logging.getLogger()
        
        # Set the level
        if verbose:
            root_logger.setLevel(logging.DEBUG)
        else:
            root_logger.setLevel(logging.INFO)
            
        # Create a console handler if one doesn't exist already
        console_handler = None
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and \
               getattr(handler, 'stream', None) is sys.stdout:
                console_handler = handler
                break
                
        if console_handler is None:
            console_handler = logging.StreamHandler(sys.stdout)
            # Create a formatter and set it on the handler
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(formatter)
            
            # Add the handler to the root logger
            root_logger.addHandler(console_handler)
            
        # Add the duplicate filter to the handler
        duplicate_filter = DuplicateFilter()
        console_handler.addFilter(duplicate_filter)
        
        # Suppress verbose warnings from specific modules
        logging.getLogger('pandas').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
    def load_data(
        self,
        assets: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        force_reload: bool = False
    ) -> Dict[str, AssetData]:
        """Load data for specified assets.
        
        Args:
            assets: List of asset IDs to load. If None, loads from assetsRoster.
            start_date: Start date for filtering
            end_date: End date for filtering
            force_reload: Whether to force reloading data from disk
            
        Returns:
            dict: Dictionary mapping asset IDs to AssetData objects
        """
        self.logger.info("Loading data for assets...")
        
        # If no assets specified, use entire roster
        if assets is None:
            try:
                from scripts.assetsRoster import ROSTER
                assets = ROSTER
                self.logger.info(f"Using ROSTER with {len(assets)} assets")
            except ImportError:
                self.logger.warning("Could not import ROSTER. Please specify assets explicitly.")
                return {}
                
        # Convert dates to datetime objects if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Initialize progress tracking
        loaded_assets = {}
        total_assets = len(assets)
        
        # Set up parallel loading if enabled
        if self.parallelize and total_assets > 1:
            self.logger.info(f"Using parallel loading for {total_assets} assets")
            
            # Define loading function
            def load_asset_data(asset_id):
                try:
                    asset_data = self._load_asset_data(
                        asset_id, start_date, end_date, force_reload
                    )
                    return asset_id, asset_data
                except Exception as e:
                    self.logger.error(f"Error loading data for {asset_id}: {e}")
                    return asset_id, None
            
            # Execute in parallel
            with ThreadPoolExecutor(max_workers=min(8, total_assets)) as executor:
                future_to_asset = {
                    executor.submit(load_asset_data, asset): asset 
                    for asset in assets
                }
                
                # Collect results as they complete
                for future in tqdm(as_completed(future_to_asset), total=total_assets, desc="Loading assets"):
                    asset_id, asset_data = future.result()
                    if asset_data is not None:
                        self.assets[asset_id] = asset_data
                        loaded_assets[asset_id] = asset_data
        else:
            # Sequential loading
            for asset_id in tqdm(assets, desc="Loading assets"):
                try:
                    asset_data = self._load_asset_data(
                        asset_id, start_date, end_date, force_reload
                    )
                    self.assets[asset_id] = asset_data
                    loaded_assets[asset_id] = asset_data
                except Exception as e:
                    self.logger.error(f"Error loading data for {asset_id}: {e}")
        
        self.logger.info(f"Loaded data for {len(loaded_assets)} out of {total_assets} assets")
        
        return loaded_assets
        
    def _load_asset_data(
        self,
        asset_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_reload: bool = False
    ) -> AssetData:
        """Load data for a specific asset.
        
        Args:
            asset_id: ID of the asset
            start_date: Start date for filtering
            end_date: End date for filtering
            force_reload: Whether to force reloading data from disk
            
        Returns:
            AssetData: Asset data object
        """
        # Check if already loaded
        if not force_reload and asset_id in self.assets:
            asset_data = self.assets[asset_id]
            self.logger.debug(f"Using cached data for {asset_id}")
            return asset_data
            
        # Create cache key
        cache_key = f"{asset_id}"
        if start_date:
            cache_key += f"_{start_date.strftime('%Y%m%d')}"
        if end_date:
            cache_key += f"_{end_date.strftime('%Y%m%d')}"
            
        # Check cache
        if not force_reload and cache_key in self.data_cache:
            self.logger.debug(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key]
        
        # Initialize DataLoader to properly handle BTC price calculations
        from ..data.data_loader import DataLoader
        
        data_loader = DataLoader(
            data_path=self.data_path,
            btc_data_path=self.btc_data_path,
            ssr_data_path=self.ssr_data_path,
            market_data_path=self.market_data_path,
            asset_ids=self.asset_ids
        )
        
        # Use DataLoader to load data with BTC columns
        try:
            price_data = data_loader.load_asset_data(asset_id)
            
            # Set the date as index if it's not already
            if 'date' in price_data.columns:
                price_data['date'] = pd.to_datetime(price_data['date'])
                price_data.set_index('date', inplace=True)
            
            # Apply date filtering
            if start_date is not None:
                price_data = price_data[price_data.index >= start_date]
            if end_date is not None:
                price_data = price_data[price_data.index <= end_date]
                
            self.logger.info(
                f"Loaded data for {asset_id} with {len(price_data)} rows"
            )
            
            # Check if BTC columns were created for non-bitcoin assets
            if asset_id != 'bitcoin':
                btc_col = f"{asset_id}_btc"
                if btc_col in price_data.columns:
                    self.logger.debug(f"BTC column {btc_col} found for {asset_id}")
                else:
                    self.logger.warning(f"BTC column {btc_col} not found for {asset_id}")
                
        except Exception as e:
            self.logger.error(f"Error loading data for {asset_id}: {e}")
            price_data = pd.DataFrame()
            
        # Create AssetData object
        asset_data = AssetData(asset_id=asset_id, price_data=price_data)
        
        # Cache for future use
        self.data_cache[cache_key] = asset_data
        
        return asset_data
        
    def get_asset_data(self, asset_id: str) -> Optional[AssetData]:
        """Get asset data for a specific asset.
        
        Args:
            asset_id: ID of the asset
            
        Returns:
            AssetData if found, None otherwise
        """
        return self.assets.get(asset_id)
        
    def get_asset_ids(self) -> List[str]:
        """Get IDs of all loaded assets.
        
        Returns:
            list: List of asset IDs
        """
        return list(self.assets.keys())
        
    def register_signals(
        self, 
        asset_id: str, 
        signal_names: Optional[List[str]] = None,
        calculate_values: bool = True,
        signal_categories: Optional[List[str]] = None
    ) -> Dict[str, SignalData]:
        """Register signals for an asset.
        
        Args:
            asset_id: ID of the asset
            signal_names: List of signal names to register. If None, registers all available signals.
            calculate_values: Whether to calculate signal values immediately
            signal_categories: Filter signals by category ('trend', 'rsi', 'ssr', 'markov', etc.)
                              This can dramatically reduce calculation time.
            
        Returns:
            dict: Dictionary mapping signal names to SignalData objects
        """
        # Get asset data
        asset_data = self.get_asset_data(asset_id)
        if not asset_data:
            self.logger.warning(f"Asset {asset_id} not found")
            return {}
        
        # If no signal names provided, use filtered or all available signals
        if signal_names is None:
            if signal_categories:
                # Filter signals by category
                all_signals = get_signal_names()
                signal_names = []
                
                # Apply category filters
                for category in signal_categories:
                    if category.lower() == 'trend':
                        # Select just one trend signal from each timeframe for efficiency
                        signal_names.extend(['ShortTermNeutralUSD', 'MediumTermNeutralUSD', 
                                           'LongTermNeutralUSD', 'OverallNeutralUSD'])
                    elif category.lower() == 'rsi':
                        signal_names.extend([s for s in all_signals if s.startswith('RSI_')])
                    elif category.lower() == 'ssr':
                        signal_names.extend([s for s in all_signals if s.startswith('SSR_')])
                    elif category.lower() == 'markov':
                        signal_names.extend([s for s in all_signals if s.startswith('Markov')])
                    # Add other categories as needed
                
                self.logger.info(f"Using {len(signal_names)} filtered signals for {asset_id}")
            else:
                signal_names = get_signal_names()
                self.logger.debug(f"Using all {len(signal_names)} available signals")
        
        # Register signals with asset data
        registered_signals = {}
        
        for signal_name in signal_names:
            try:
                # Get signal instance
                signal = get_signal(signal_name)
                
                if signal is None:
                    # Signal not found in registry
                    self.logger.debug(f"Signal {signal_name} not found in registry")
                    
                    # Create an empty SignalData anyway for consistency
                    signal_data = SignalData(signal_name=signal_name, asset_id=asset_id)
                    asset_data.add_signal(signal_name, signal_data)
                    registered_signals[signal_name] = signal_data
                    continue
                
                # Initialize params with asset_id
                params = {'asset_id': asset_id}
                
                # Register signal with asset data
                signal_data = asset_data.register_signal(signal, params)
                
                if signal_data:
                    registered_signals[signal_name] = signal_data
                    self.logger.debug(f"Registered signal {signal_name} for {asset_id}")
                    
                    # Calculate values if requested
                    if calculate_values:
                        # Pass the price data to calculate_values to avoid warnings
                        values = signal_data.calculate_values(asset_data.price_data)
                        self.logger.debug(
                            f"Calculated values for {signal_name} on {asset_id}: "
                            f"{len(values)} data points"
                        )
                
            except Exception as e:
                self.logger.error(f"Error registering signal {signal_name} for {asset_id}: {e}")
                continue
        
        return registered_signals
    
    def register_signals_parallel(
        self,
        asset_ids: List[str],
        signal_names: Optional[List[str]] = None,
        calculate_values: bool = True,
        signal_categories: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, SignalData]]:
        """Register signals for multiple assets in parallel.
        
        Args:
            asset_ids: List of asset IDs to register signals for
            signal_names: List of signal names to register. If None, register all available signals.
            calculate_values: Whether to calculate signal values
            signal_categories: Filter signals by category ('trend', 'rsi', 'ssr', 'markov', etc.)
                              This can dramatically reduce calculation time.
            
        Returns:
            dict: Dictionary mapping asset IDs to dictionaries mapping signal names to SignalData objects
        """
        self.logger.info(f"Registering signals in parallel for {len(asset_ids)} assets...")
        
        # Clear trend cache before starting
        try:
            from ..signals.trend_signals import TrendSignalBase
            TrendSignalBase.clear_trend_cache()
            self.logger.info("Cleared trend signal cache")
        except Exception as e:
            self.logger.warning(f"Could not clear trend signal cache: {e}")
        
        results = {}
        
        # Function to register signals for a single asset and handle exceptions
        def _register_for_asset(asset_id):
            try:
                self.logger.info(f"Starting signal registration for {asset_id}")
                start_time = datetime.now()
                result = self.register_signals(
                    asset_id, 
                    signal_names, 
                    calculate_values,
                    signal_categories
                )
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                self.logger.info(f"Completed signal registration for {asset_id} in {duration:.2f} seconds")
                return asset_id, result
            except Exception as e:
                self.logger.error(f"Error registering signals for {asset_id}: {e}")
                return asset_id, {}
        
        # Process assets in parallel or sequentially based on the parallelize flag
        if not self.parallelize:
            for asset_id in tqdm(asset_ids, desc="Registering signals"):
                asset_signals = _register_for_asset(asset_id)[1]
                results[asset_id] = asset_signals
        else:
            # Use thread pool for I/O-bound operations
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_asset = {executor.submit(_register_for_asset, asset_id): asset_id for asset_id in asset_ids}
                
                for future in tqdm(as_completed(future_to_asset), total=len(asset_ids), desc="Registering signals"):
                    asset_id, asset_signals = future.result()
                    results[asset_id] = asset_signals
        
        # Log summary
        total_signals = sum(len(signals) for signals in results.values())
        self.logger.info(f"Registered a total of {total_signals} signals across {len(results)} assets")
        
        return results
        
    def calculate_signal_values(
        self,
        asset_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_recalculate: bool = False
    ) -> pd.DataFrame:
        """Calculate signal values for a specific asset.
        
        Args:
            asset_id: ID of the asset
            start_date: Start date for filtering
            end_date: End date for filtering
            force_recalculate: Whether to force recalculation of signal values
            
        Returns:
            DataFrame with signal values where columns are signal names
        """
        asset_data = self.get_asset_data(asset_id)
        if asset_data is None:
            self.logger.warning(f"Asset data not found for {asset_id}")
            return pd.DataFrame()
            
        # Get price data
        price_data = asset_data.price_data
        if price_data.empty:
            self.logger.warning(f"Price data is empty for {asset_id}")
            return pd.DataFrame()
            
        # Apply date filtering to price data
        if start_date is not None or end_date is not None:
            filtered_price_data = price_data.copy()
            if start_date is not None:
                filtered_price_data = filtered_price_data[filtered_price_data.index >= start_date]
            if end_date is not None:
                filtered_price_data = filtered_price_data[filtered_price_data.index <= end_date]
        else:
            filtered_price_data = price_data
            
        # Get all registered signals
        signal_names = asset_data.get_signal_names()
        
        if not signal_names:
            self.logger.warning(f"No signals registered for {asset_id}")
            return pd.DataFrame()
            
        # If force recalculate, recalculate all signal values
        if force_recalculate:
            self.logger.info(f"Recalculating all signal values for {asset_id}")
            
            for signal_name in signal_names:
                signal_instance = get_signal(signal_name)
                if signal_instance is None:
                    self.logger.warning(f"Signal {signal_name} not found in registry")
                    continue
                    
                signal_data = asset_data.get_signal(signal_name)
                
                try:
                    # Calculate signal values using the full dataset to ensure proper calculation
                    values = signal_instance.calculate(filtered_price_data, asset_id)
                    
                    # Store values and make sure they're properly persisted
                    signal_data.set_values(values)
                    
                    # Log how many values we have
                    self.logger.info(f"Calculated {len(values)} values for {signal_name} on {asset_id}")
                except Exception as e:
                    self.logger.error(f"Error calculating values for {signal_name} on {asset_id}: {e}")
                    continue
                    
        # Get signal values
        signal_values = asset_data.get_signal_values(start_date, end_date)
        
        # Log what we're returning
        self.logger.info(f"Returning {len(signal_values)} signal value rows for {asset_id} with {len(signal_values.columns)} signals")
        
        return signal_values
        
    def get_current_recommendations(
        self,
        asset_ids: Optional[List[str]] = None,
        current_date: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get current recommendations for specified assets using binary signal logic.
        
        Args:
            asset_ids: List of asset IDs to get recommendations for. If None, uses all loaded assets.
            current_date: Current date for recommendations. If None, uses the latest date in the data.
            
        Returns:
            dict: Dictionary mapping asset IDs to recommendation details
        """
        # Use loaded assets if not specified
        if asset_ids is None:
            asset_ids = list(self.assets.keys())
            
        # If no assets loaded, return empty dict
        if not asset_ids:
            self.logger.warning("No assets loaded for recommendations")
            return {}
            
        # Initialize recommendations
        recommendations = {}
        
        for asset_id in asset_ids:
            # Get asset data
            asset_data = self.get_asset_data(asset_id)
            if not asset_data:
                self.logger.warning(f"Asset {asset_id} not found")
                continue
                
            # Get price data
            price_data = asset_data.get_price_data()
            if price_data.empty:
                self.logger.warning(f"No price data for {asset_id}")
                continue
                
            # Determine current date if not specified
            if current_date is None:
                current_date = price_data.index[-1]
            
            # Get signal values for the current date
            signal_values = self.calculate_signal_values(
                asset_id, 
                start_date=current_date, 
                end_date=current_date
            )
            
            if signal_values.empty:
                self.logger.warning(f"No signal values for {asset_id} on {current_date}")
                continue
                
            # Extract the latest row (should be just one row for the current date)
            latest_signals = signal_values.iloc[-1].to_dict()
            
            # Count active signals and calculate combined weight
            active_signals = {name: value for name, value in latest_signals.items() if value == 1}
            combined_weight = sum(active_signals.values())
            
            # Make simple binary decision based on threshold
            decision = 1 if combined_weight >= self.signal_threshold else 0
                
            # Get latest price
            latest_price_row = price_data.loc[price_data.index <= current_date].iloc[-1]
            latest_price = latest_price_row.get('close', None)
            
            # Store recommendation
            recommendations[asset_id] = {
                'date': current_date,
                'decision': decision,
                'latest_price': latest_price,
                'active_signals': active_signals,
                'combined_weight': combined_weight
            }
            
        return recommendations
        
    def run_backtest(
        self,
        backtest_name: str,
        assets: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        initial_capital: float = 10000.0,
        trade_cost: float = 0.001,
        slippage_pct: float = 0.0
    ) -> Dict[str, Any]:
        """Run a simplified backtest on specified assets using binary signal logic.
        
        Args:
            backtest_name: Name of the backtest
            assets: List of asset IDs to backtest. If None, uses all loaded assets.
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for the portfolio
            trade_cost: Cost per trade as a fraction
            slippage_pct: Slippage as a percentage
            
        Returns:
            dict: Backtest results
        """
        self.logger.info(f"Starting backtest: {backtest_name}")
        
        # Use loaded assets if not specified
        if assets is None:
            assets = list(self.assets.keys())
            
        # If no assets loaded, return empty dict
        if not assets:
            self.logger.warning("No assets loaded for backtest")
            return {'success': False, 'error': 'No assets loaded'}
            
        # Convert dates to datetime objects if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Create price data for all assets
        price_data = {}
        for asset_id in assets:
            # Get asset data
            asset_data = self.get_asset_data(asset_id)
            if not asset_data:
                self.logger.warning(f"Asset {asset_id} not found, skipping in backtest")
                continue
                
            # Get price data
            price_df = asset_data.get_price_data()
            if price_df.empty:
                self.logger.warning(f"No price data for {asset_id}, skipping in backtest")
                continue
                
            # Filter by date if specified
            if start_date is not None:
                price_df = price_df[price_df.index >= start_date]
            if end_date is not None:
                price_df = price_df[price_df.index <= end_date]
                
            # Skip if no data after filtering
            if price_df.empty:
                self.logger.warning(f"No price data for {asset_id} in date range, skipping")
                continue
                
            # Store for backtest
            price_data[asset_id] = price_df
            
        if not price_data:
            self.logger.error("No valid price data for backtest")
            return {'success': False, 'error': 'No valid price data'}
            
        # Determine overall date range from data
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index)
        date_range = pd.DatetimeIndex(sorted(all_dates))
        
        # Initialize dictionary to store decisions for each asset
        decisions = {}
        
        # Generate decisions for each asset using binary signal logic
        for asset_id in assets:
            # Skip asset if it's not in price_data
            if asset_id not in price_data:
                continue
                
            # Calculate signal values for the entire period
            signal_values = self.calculate_signal_values(
                asset_id=asset_id,
                start_date=date_range[0] if len(date_range) > 0 else None,
                end_date=date_range[-1] if len(date_range) > 0 else None,
                force_recalculate=False
            )
            
            if signal_values.empty:
                self.logger.warning(f"No signal values for {asset_id}, skipping in backtest")
                continue
                
            # Calculate decisions based on signals - sum signals and compare with threshold
            signal_sum = signal_values.sum(axis=1)
            binary_decisions = (signal_sum >= self.signal_threshold).astype(int)
            
            # Reindex to match the full date range
            asset_decisions = pd.Series(0, index=date_range)
            asset_decisions.loc[binary_decisions.index] = binary_decisions
            
            decisions[asset_id] = asset_decisions
            
        # Use the existing Backtester from backtester.py
        backtester = Backtester(
            initial_capital=initial_capital,
            trade_cost=trade_cost,
            slippage_pct=slippage_pct
        )
        
        # Run backtest
        results = backtester.run_backtest(
            price_data=price_data,
            decisions=decisions,
            start_date=date_range[0] if len(date_range) > 0 else None,
            end_date=date_range[-1] if len(date_range) > 0 else None
        )
            
        # Add metadata to results
        results['backtest_name'] = backtest_name
        results['assets'] = list(price_data.keys())
        results['start_date'] = date_range[0] if len(date_range) > 0 else None
        results['end_date'] = date_range[-1] if len(date_range) > 0 else None
        results['success'] = True
        
        return results
        
    def get_available_signals(self) -> List[str]:
        """Get names of all available signals.
        
        Returns:
            list: List of signal names
        """
        return get_all_signals()
        
    def train_meta_labels(
        self,
        asset_id: str,
        feature_columns: List[str],
        min_return_threshold: float = 0.0,
        signals: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, Tuple[float, int]]:
        """Train meta-labeling models for signals on a specific asset.
        
        This is a placeholder for future implementation. Currently returns empty results.
        
        Args:
            asset_id: ID of the asset
            feature_columns: List of column names to use as features
            min_return_threshold: Minimum return to consider a trade successful
            signals: List of signal names to train meta-labels for. If None, train for all signals.
            verbose: Whether to print detailed information
            
        Returns:
            dict: Dictionary mapping signal names to tuples of (accuracy, sample_count)
        """
        self.logger.info(f"Meta-labeling feature is a placeholder for future implementation")
        self.logger.info(f"Would train meta-labels for {asset_id} with {len(feature_columns)} features")
        
        asset_data = self.get_asset_data(asset_id)
        if asset_data is None:
            self.logger.warning(f"Asset data not found for {asset_id}")
            return {}
        
        # Get signals to train
        all_signals = asset_data.get_signals()
        if signals is None:
            # Use all signals
            signals_to_train = all_signals
        else:
            # Use specified signals
            signals_to_train = {name: all_signals[name] for name in signals if name in all_signals}
        
        if not signals_to_train:
            self.logger.warning(f"No signals to train meta-labels for on {asset_id}")
            return {}
        
        self.logger.info(f"Would train meta-labels for {len(signals_to_train)} signals on {asset_id}")
        
        # Return placeholder results (empty dict for now)
        return {}
        
    def clear_cache(self):
        """Clear all caches."""
        self.data_cache = {}
        for asset_id, asset_data in self.assets.items():
            asset_data.clear_cache()
        
        self.logger.info("Cleared all caches")
        
    def run_analysis(self) -> Dict[str, float]:
        """Run the simplified analysis pipeline on all assets.
        
        This method:
        1. Registers signals for each asset
        2. Calculates signal values
        3. Runs a single backtest for all assets
        4. Updates portfolio weights based on backtest results
        
        Returns:
            dict: Dictionary mapping asset IDs to portfolio weights
        """
        self.logger.info("Running simplified portfolio analysis...")
        
        # Step 1: Register and calculate signals for each asset
        asset_ids = list(self.assets.keys())
        
        if not asset_ids:
            self.logger.warning("No assets loaded. Use load_data() first.")
            return {}
            
        self.logger.info(f"Processing signals for {len(asset_ids)} assets...")
        
        # SPECIFIC SIGNAL SELECTION - to diagnose performance issues
        # Instead of categories, let's explicitly select specific signals by name
        signal_names = [
            # Essential trend signals - just the neutral ones
            'ShortTermNeutralUSD', 'MediumTermNeutralUSD', 
            'LongTermNeutralUSD', 'OverallNeutralUSD',
            
            # Essential RSI signals - just the oversold/overbought
            'RSI_Oversold_USD', 'RSI_Overbought_USD',
            
            # SSR signals 
            'SSR_RiskOn', 'SSR_RiskOff'
            
            # Explicitly EXCLUDE Donchian signals
            # 'DonchianEnsembleUSD', 'DonchianEnsembleBTC'
        ]
        
        self.logger.info(f"Using a minimal set of {len(signal_names)} specific signals")
        
        # Process signals in parallel for all assets
        self.register_signals_parallel(
            asset_ids=asset_ids,
            signal_names=signal_names,  # Use explicit list instead of categories
            calculate_values=True
        )
        
        # Step 2: Run a single backtest for all assets
        self.logger.info("Running backtest...")
        
        backtest_results = self.run_backtest(
            backtest_name="combined_backtest",
            assets=asset_ids
        )
        
        if not backtest_results.get('success', False):
            self.logger.warning("Backtest failed")
            return {}
            
        # Step 3: Update portfolio weights
        self.logger.info("Updating portfolio weights...")
        
        try:
            portfolio_weights = self.portfolio_manager.optimize_weights(
                backtest_results=backtest_results
            )
            
            # Store portfolio weights
            self.portfolio_weights = portfolio_weights
            
            return portfolio_weights
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio weights: {str(e)}")
            return {}
            
    def _create_asset_data(self, asset_id: str) -> AssetData:
        """Create an empty asset data object for testing.
        
        Args:
            asset_id: ID of the asset
            
        Returns:
            AssetData: Empty asset data object
        """
        from .asset_data import AssetData
        return AssetData(asset_id=asset_id) 