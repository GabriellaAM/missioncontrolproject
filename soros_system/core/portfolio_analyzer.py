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
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable, Type, cast
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
from ..analysis.metrics import MetricsCalculator
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
        
        # Handle case where a single asset ID string is passed
        if isinstance(assets, str):
            assets = [assets]
            self.logger.info(f"Converting single asset ID string '{assets[0]}' to a list")
        # Convert assets to list if it's a set or other iterable (but not a string)
        elif not isinstance(assets, list):
            assets = list(assets)
            self.logger.info(f"Converted assets to list with {len(assets)} items")
                
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
            with ThreadPoolExecutor(max_workers=min(self.num_workers, total_assets)) as executor:
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
        
        # First check which signals are already registered to avoid duplicate work
        existing_signals = asset_data.get_signal_names()
        new_signals = [s for s in signal_names if s not in existing_signals]
        reuse_signals = [s for s in signal_names if s in existing_signals]
        
        if reuse_signals:
            self.logger.info(f"Reusing {len(reuse_signals)} already registered signals for {asset_id}")
            # Add existing signals to the result
            for signal_name in reuse_signals:
                signal_data = asset_data.get_signal(signal_name)
                registered_signals[signal_name] = signal_data
        
        # Process only new signals
        if new_signals:
            self.logger.info(f"Registering {len(new_signals)} new signals for {asset_id}")
            for signal_name in new_signals:
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
        signal_categories: Optional[List[str]] = None,
        skip_trend_signals: bool = False,
        signal_groups: Optional[List[str]] = None,
        specific_signals: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, SignalData]]:
        """Register signals for multiple assets in parallel.
        
        Args:
            asset_ids: List of asset IDs to register signals for
            signal_names: List of signal names to register. If None, register all available signals.
            calculate_values: Whether to calculate signal values
            signal_categories: Filter signals by category ('trend', 'rsi', 'ssr', 'markov', etc.)
                              This can dramatically reduce calculation time.
            skip_trend_signals: If True, skip all trend signals regardless of other parameters
            signal_groups: A list of signal groups to register (e.g., ['rsi', 'markov']). 
                           Overrides signal_categories if provided.
            specific_signals: A list of specific signal names to register (e.g., ['BullLowVarianceSignalUSD', 
                              'RSI_Oversold_USD']). Overrides all other signal selection parameters if provided.
            
        Returns:
            dict: Dictionary mapping asset IDs to dictionaries mapping signal names to SignalData objects
        """
        self.logger.info(f"Registering signals in parallel for {len(asset_ids)} assets...")
        
        # Convert asset_ids to list if it's a set or other iterable
        if not isinstance(asset_ids, list):
            asset_ids = list(asset_ids)
            self.logger.info(f"Converted asset_ids to list with {len(asset_ids)} items")
        
        # If specific_signals is provided, use that directly and ignore other signal selection parameters
        if specific_signals and len(specific_signals) > 0:
            signal_names = specific_signals
            self.logger.info(f"Using {len(signal_names)} specific signals provided by user")
            skip_trend_signals = False  # Don't skip trend signals if user explicitly specified them
        else:
            # Clear trend cache before starting (unless skipping trend signals)
            if not skip_trend_signals and (signal_groups is None or 'trend' in signal_groups or 'regime' in signal_groups):
                try:
                    from ..signals.trend_signals import TrendSignalBase
                    TrendSignalBase.clear_trend_cache()
                    self.logger.info("Cleared trend signal cache")
                except Exception as e:
                    self.logger.warning(f"Could not clear trend signal cache: {e}")
            else:
                self.logger.info("Trend signals will be skipped")
                
            # If signal_groups is provided, convert to signal_categories format
            if signal_groups:
                # Handle the 'regime' group specially - it includes all regime detection signals
                if 'regime' in signal_groups:
                    all_signals = get_signal_names()
                    regime_signals = [s for s in all_signals if (
                        s.startswith('Bull') or s.startswith('Bear') or
                        'Regime' in s or 'Variance' in s
                    )]
                    
                    # If signal_names is None, initialize it; otherwise append to existing list
                    if signal_names is None:
                        signal_names = regime_signals
                    else:
                        signal_names = list(set(signal_names + regime_signals))
                    
                    # Remove 'regime' from signal_groups to prevent double processing
                    signal_groups = [g for g in signal_groups if g != 'regime']
                
                signal_categories = signal_groups
                self.logger.info(f"Using signal groups: {signal_groups}")
            
            # If skipping trend signals and signal_names is None, filter out trend signals
            if skip_trend_signals and signal_names is None and not signal_categories:
                # Get all signal names
                all_signals = get_signal_names()
                # Filter out trend signals (those likely to be trend-related)
                signal_names = [s for s in all_signals if not s.startswith('ShortTerm') 
                               and not s.startswith('MediumTerm')
                               and not s.startswith('LongTerm')
                               and not s.startswith('Overall')
                               and 'Trend' not in s]
                self.logger.info(f"Skipping trend signals, using {len(signal_names)} non-trend signals")
        
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
            # Limit max_workers to something reasonable (min of num_workers or number of assets)
            max_workers = min(self.num_workers, len(asset_ids))
            self.logger.info(f"Using {max_workers} workers for parallel signal registration")
            
            # Process in groups to better manage memory
            # Use groups of 5 assets, process each group completely before moving to the next
            group_size = 5
            for i in range(0, len(asset_ids), group_size):
                group = asset_ids[i:i+group_size]
                group_num = i // group_size + 1
                total_groups = (len(asset_ids) + group_size - 1) // group_size
                
                self.logger.info(f"Processing asset group {group_num}/{total_groups} with {len(group)} assets")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_asset = {executor.submit(_register_for_asset, asset_id): asset_id for asset_id in group}
                    
                    for future in tqdm(as_completed(future_to_asset), total=len(group), 
                                       desc=f"Registering signals group {group_num}/{total_groups}"):
                        asset_id, asset_signals = future.result()
                        results[asset_id] = asset_signals
                
                # Optional: suggest garbage collection between groups
                if len(asset_ids) > group_size * 2:  # Only do this for larger datasets
                    import gc
                    gc.collect()
        
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
        
    def get_available_signals(self, group_by_category: bool = False, return_groups: bool = False) -> Union[List[str], Dict[str, List[str]], List[str]]:
        """Get names of all available signals.
        
        Args:
            group_by_category: If True, returns a dictionary where keys are signal categories and values are lists of signals
            return_groups: If True, returns a list of available signal group names instead of signal names
            
        Returns:
            If group_by_category is True: Dictionary mapping categories to lists of signal names
            If return_groups is True: List of available signal group names
            Otherwise: List of all signal names
        """
        all_signals = get_all_signals()
        
        if return_groups:
            # Return available signal groups
            return ['rsi', 'trend', 'ssr', 'markov', 'donchian', 'regime', 'all']
            
        if not group_by_category:
            return all_signals
            
        # Group signals by category
        signal_groups = {
            'trend': [s for s in all_signals if (s.startswith('ShortTerm') or 
                                              s.startswith('MediumTerm') or 
                                              s.startswith('LongTerm') or 
                                              s.startswith('Overall') or
                                              'Trend' in s)],
            'rsi': [s for s in all_signals if s.startswith('RSI_')],
            'ssr': [s for s in all_signals if s.startswith('SSR_')],
            'markov': [s for s in all_signals if s.startswith('Markov')],
            'donchian': [s for s in all_signals if 'Donchian' in s],
            'regime': [s for s in all_signals if (s.startswith('Bull') or 
                                               s.startswith('Bear') or 
                                               'Regime' in s or 
                                               'Variance' in s)],
            'other': []
        }
        
        # Add signals that don't match any category to 'other'
        for signal_name in all_signals:
            if not any(signal_name in category_signals for category_signals in signal_groups.values()):
                signal_groups['other'].append(signal_name)
                
        return signal_groups
        
    def list_signals_by_prefix(self, prefix: str = '', display_count: bool = True) -> List[str]:
        """List all available signals that start with the given prefix.
        
        This is a convenience method for exploring available signals in 
        interactive sessions like Jupyter notebooks.
        
        Args:
            prefix: Optional prefix to filter signals (e.g., 'RSI_', 'Bull')
            display_count: Whether to print the count of matching signals
            
        Returns:
            List of signal names matching the prefix
        """
        all_signals = get_all_signals()
        matching_signals = [s for s in all_signals if s.startswith(prefix)]
        
        if display_count:
            self.logger.info(f"Found {len(matching_signals)} signals matching prefix '{prefix}'")
            
        return sorted(matching_signals)
    
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

    def run_signal_backtest(
        self,
        backtest_name: str,
        assets: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        initial_capital: float = 10000.0,
        usd_signals: Optional[List[str]] = None,
        btc_signals: Optional[List[str]] = None,
        include_metrics: bool = True,
        plot_results: bool = True
    ) -> Dict[str, Any]:
        """Run a backtest for individual assets based on specified signals.
        
        This method:
        1. Runs individual backtests for each asset separately
        2. Stores results within each asset's data in the analyzer
        3. For altcoins (non-BTC), requires positive signals against both USD and BTC
        4. Uses specific transaction costs for BTC (0.1%) and altcoins (0.5%)
        5. Provides detailed signal activation tracking per day
        
        Args:
            backtest_name: Name of the backtest for identification
            assets: List of asset IDs to backtest. If None, uses all loaded assets.
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for the individual asset backtests
            usd_signals: List of signal names to use for USD evaluation
            btc_signals: List of signal names to use for BTC evaluation (for altcoins)
            include_metrics: Whether to calculate additional performance metrics
            plot_results: Whether to plot the results
            
        Returns:
            dict: Backtest results with comprehensive metrics for each asset
        """
        self.logger.info(f"Starting signal-focused backtest: {backtest_name}")
        
        # Validate signal parameters
        if usd_signals is None and btc_signals is None:
            raise ValueError("At least one of usd_signals or btc_signals must be provided")
            
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
        
        # Log the signals being used
        if usd_signals:
            self.logger.info(f"Using USD signals: {', '.join(usd_signals)}")
        if btc_signals:
            self.logger.info(f"Using BTC signals: {', '.join(btc_signals)}")
            
        # Store overall results and asset-specific results
        overall_results = {
            'backtest_name': backtest_name,
            'start_date': date_range[0] if len(date_range) > 0 else None,
            'end_date': date_range[-1] if len(date_range) > 0 else None,
            'assets': list(price_data.keys()),
            'signals_used': {
                'usd': usd_signals if usd_signals else [],
                'btc': btc_signals if btc_signals else []
            },
            'asset_results': {}
        }
        
        # Create individual asset backtests
        for asset_id in price_data.keys():
            self.logger.info(f"Running backtest for {asset_id}")
            
            # Get asset data
            asset_data = self.get_asset_data(asset_id)
            asset_price_data = price_data[asset_id]
            
            # Calculate signal values for this asset
            signal_values = self.calculate_signal_values(
                asset_id=asset_id,
                start_date=date_range[0] if len(date_range) > 0 else None,
                end_date=date_range[-1] if len(date_range) > 0 else None,
                force_recalculate=False
            )
            
            if signal_values.empty:
                self.logger.warning(f"No signal values for {asset_id}, skipping in backtest")
                continue
                
            # Filter signals based on provided lists
            asset_usd_signals = pd.DataFrame()
            asset_btc_signals = pd.DataFrame()
            
            # Process USD signals if provided
            if usd_signals:
                for signal_name in usd_signals:
                    if signal_name in signal_values.columns:
                        asset_usd_signals[signal_name] = signal_values[signal_name]
                    else:
                        self.logger.warning(f"Signal {signal_name} not found for {asset_id}")
            
            # Process BTC signals if provided (for altcoins)
            if btc_signals and asset_id != 'bitcoin':
                for signal_name in btc_signals:
                    if signal_name in signal_values.columns:
                        asset_btc_signals[signal_name] = signal_values[signal_name]
                    else:
                        self.logger.warning(f"Signal {signal_name} not found for {asset_id}")
            
            # Determine if we have valid signals
            has_usd_signals = not asset_usd_signals.empty
            has_btc_signals = not asset_btc_signals.empty
            
            # Create DataFrame to track daily signal counts and trading decisions
            signal_tracker = pd.DataFrame(index=date_range)
            
            # For Bitcoin or USD-only signals, only check USD signals
            if asset_id == 'bitcoin' or not has_btc_signals:
                if has_usd_signals:
                    # Add columns for each USD signal
                    for col in asset_usd_signals.columns:
                        common_dates = sorted(set(date_range) & set(asset_usd_signals.index))
                        signal_tracker.loc[common_dates, col] = asset_usd_signals.loc[common_dates, col]
                    
                    # Fill NaN values with 0
                    signal_tracker.fillna(0, inplace=True)
                    
                    # Calculate number of positive signals each day
                    signal_tracker['usd_positive_count'] = (signal_tracker[asset_usd_signals.columns] > 0).sum(axis=1)
                    signal_tracker['usd_total_count'] = len(asset_usd_signals.columns)
                    signal_tracker['usd_positive_ratio'] = signal_tracker['usd_positive_count'] / signal_tracker['usd_total_count']
                    
                    # Generate trading decisions - buy if majority of signals are positive
                    signal_tracker['decision'] = (signal_tracker['usd_positive_ratio'] > 0.5).astype(int)
                    
                    self.logger.info(f"{asset_id}: Using {len(asset_usd_signals.columns)} USD signals for decisions")
                else:
                    self.logger.warning(f"No valid USD signals for {asset_id}, skipping backtest")
                    continue
            else:
                # For altcoins, check both USD and BTC signals
                if has_usd_signals and has_btc_signals:
                    # Add columns for each USD signal
                    for col in asset_usd_signals.columns:
                        common_dates = sorted(set(date_range) & set(asset_usd_signals.index))
                        signal_tracker.loc[common_dates, f"USD_{col}"] = asset_usd_signals.loc[common_dates, col]
                    
                    # Add columns for each BTC signal
                    for col in asset_btc_signals.columns:
                        common_dates = sorted(set(date_range) & set(asset_btc_signals.index))
                        signal_tracker.loc[common_dates, f"BTC_{col}"] = asset_btc_signals.loc[common_dates, col]
                    
                    # Fill NaN values with 0
                    signal_tracker.fillna(0, inplace=True)
                    
                    # Calculate number of positive signals each day for USD and BTC
                    usd_columns = [col for col in signal_tracker.columns if col.startswith('USD_')]
                    btc_columns = [col for col in signal_tracker.columns if col.startswith('BTC_')]
                    
                    signal_tracker['usd_positive_count'] = (signal_tracker[usd_columns] > 0).sum(axis=1)
                    signal_tracker['usd_total_count'] = len(usd_columns)
                    signal_tracker['usd_positive_ratio'] = signal_tracker['usd_positive_count'] / signal_tracker['usd_total_count']
                    
                    signal_tracker['btc_positive_count'] = (signal_tracker[btc_columns] > 0).sum(axis=1)
                    signal_tracker['btc_total_count'] = len(btc_columns)
                    signal_tracker['btc_positive_ratio'] = signal_tracker['btc_positive_count'] / signal_tracker['btc_total_count']
                    
                    # Generate trading decisions - buy if majority of signals are positive in BOTH USD and BTC
                    signal_tracker['decision'] = ((signal_tracker['usd_positive_ratio'] > 0.5) & 
                                                 (signal_tracker['btc_positive_ratio'] > 0.5)).astype(int)
                    
                    self.logger.info(f"{asset_id}: Using {len(usd_columns)} USD signals and " +
                                     f"{len(btc_columns)} BTC signals for decisions")
                else:
                    self.logger.warning(f"Missing signals for {asset_id} - need both USD and BTC signals for altcoins")
                    continue
            
            # Create trading decisions series
            asset_decisions = signal_tracker['decision']
            
            # Run an individual backtest for this asset
            # Create a single-asset price data dictionary
            single_asset_price = {asset_id: asset_price_data}
            single_asset_decisions = {asset_id: asset_decisions}
            
            # Determine transaction cost based on asset type
            asset_costs = {
                'bitcoin': 0.001,  # 0.1% for Bitcoin
                'default': 0.005   # 0.5% for altcoins
            }
            
            # Run backtest using the Backtester
            backtester = Backtester(
                initial_capital=initial_capital,
                trade_cost=0.001,  # Default value, will be overridden
                slippage_pct=0.001  # Default slippage
            )
            
            asset_result = backtester.run_backtest(
                price_data=single_asset_price,
                decisions=single_asset_decisions,
                start_date=date_range[0] if len(date_range) > 0 else None,
                end_date=date_range[-1] if len(date_range) > 0 else None,
                asset_specific_costs=asset_costs
            )
            
            # Calculate buy and hold performance
            start_price = asset_price_data['close'].iloc[0]
            end_price = asset_price_data['close'].iloc[-1]
            buy_hold_return = (end_price / start_price) - 1
            
            # Add buy and hold metrics
            asset_result['buy_hold_return'] = buy_hold_return
            asset_result['outperformance'] = asset_result['total_return'] - buy_hold_return
            
            # Add signal tracking data
            asset_result['signal_tracker'] = signal_tracker
            
            # Store trading data in the asset data object
            if asset_data:
                asset_data.backtest_results = asset_result
                asset_data.signal_tracker = signal_tracker
            
            # Store in overall results
            overall_results['asset_results'][asset_id] = asset_result
        
        # Calculate overall portfolio metrics
        if len(overall_results['asset_results']) > 0:
            # Aggregate metrics across assets
            total_returns = [result['total_return'] for result in overall_results['asset_results'].values()]
            sharpe_ratios = [result['sharpe_ratio'] for result in overall_results['asset_results'].values() 
                             if not np.isnan(result['sharpe_ratio'])]
            drawdowns = [result['max_drawdown'] for result in overall_results['asset_results'].values()]
            
            overall_results['avg_total_return'] = np.mean(total_returns)
            overall_results['avg_sharpe_ratio'] = np.mean(sharpe_ratios) if sharpe_ratios else np.nan
            overall_results['avg_max_drawdown'] = np.mean(drawdowns)
            overall_results['success'] = True
        else:
            overall_results['success'] = False
            overall_results['error'] = 'No valid asset results'
        
        # Plot results if requested
        if plot_results and len(overall_results['asset_results']) > 0:
            try:
                import matplotlib.pyplot as plt
                import matplotlib.gridspec as gridspec
                from matplotlib.ticker import FuncFormatter
                import matplotlib.dates as mdates
                from matplotlib.lines import Line2D
                
                # Create separate plots for each asset
                for asset_id, asset_result in overall_results['asset_results'].items():
                    # Get price data and signal tracker
                    price_series = price_data[asset_id]['close']
                    signal_df = asset_result.get('signal_tracker', pd.DataFrame())
                    
                    # Calculate buy & hold performance
                    cumulative_price = price_series / price_series.iloc[0] * initial_capital
                    
                    # Calculate drawdowns
                    strategy_drawdown = (asset_result['portfolio_value'] / asset_result['portfolio_value'].cummax() - 1)
                    buyhold_drawdown = (cumulative_price / cumulative_price.cummax() - 1)
                    
                    # Calculate daily returns
                    strategy_returns = asset_result['portfolio_value'].pct_change().fillna(0)
                    buyhold_returns = price_series.pct_change().fillna(0)
                    
                    # Set up main figure
                    plt.figure(figsize=(16, 14))
                    grid = gridspec.GridSpec(4, 2, height_ratios=[3, 1.5, 1.5, 2])
                    
                    # 1. Strategy value vs buy & hold with asset price on dual axis
                    ax1 = plt.subplot(grid[0, :])
                    
                    # Plot strategy and buy & hold values
                    ax1.plot(asset_result['portfolio_value'].index, asset_result['portfolio_value'], 
                            label='Strategy', color='green', linewidth=2)
                    ax1.plot(cumulative_price.index, cumulative_price, 
                            label='Buy & Hold', color='blue', linewidth=2)
                    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    
                    # Create secondary axis for raw asset price
                    ax1_2 = ax1.twinx()
                    ax1_2.plot(price_series.index, price_series, color='gray', alpha=0.3, linestyle='--')
                    ax1_2.set_ylabel(f'{asset_id.capitalize()} Price', color='gray', fontsize=10)
                    
                    # Format x-axis dates
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                    
                    # Add legend for first plot
                    ax1.legend(loc='upper left')
                    
                    ax1.set_title(f'{asset_id.capitalize()} - Strategy vs Buy & Hold Performance', fontsize=14)
                    
                    # 2. Drawdown comparison chart
                    ax2 = plt.subplot(grid[1, :], sharex=ax1)
                    ax2.fill_between(strategy_drawdown.index, 0, strategy_drawdown, color='red', alpha=0.5, label='Strategy Drawdown')
                    ax2.plot(buyhold_drawdown.index, buyhold_drawdown, color='blue', alpha=0.7, label='Buy & Hold Drawdown')
                    ax2.set_ylabel('Drawdown', fontsize=12)
                    ax2.set_ylim(min(strategy_drawdown.min(), buyhold_drawdown.min()) * 1.1, 0.05)
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='lower right')
                    ax2.set_title('Drawdown Comparison', fontsize=12)
                    
                    # 3. Daily returns comparison
                    ax3 = plt.subplot(grid[2, 0], sharex=ax1)
                    
                    # Get a sample of dates (we can't show all daily returns as bars)
                    if len(strategy_returns) > 60:
                        # Sample roughly 60 points spread evenly
                        sample_size = 60
                        step = len(strategy_returns) // sample_size
                        sample_indices = range(0, len(strategy_returns), step)
                        sampled_dates = strategy_returns.index[sample_indices]
                        
                        # Sample strategy and buy & hold returns
                        sampled_strategy = strategy_returns.loc[sampled_dates]
                        sampled_buyhold = buyhold_returns.loc[sampled_dates]
                        
                        # Use these for the bar chart
                        x = range(len(sampled_dates))
                        width = 0.35
                        
                        # Plot bars
                        ax3.bar([i - width/2 for i in x], sampled_strategy, width, label='Strategy', color='green', alpha=0.7)
                        ax3.bar([i + width/2 for i in x], sampled_buyhold, width, label='Buy & Hold', color='blue', alpha=0.7)
                        
                        # Format x-axis with representative dates
                        ax3.set_xticks(x[::len(x)//6])  # Show 6 date labels
                        ax3.set_xticklabels([d.strftime('%Y-%m') for d in sampled_dates[::len(x)//6]], rotation=45)
                    else:
                        # Just plot all dates if we have few enough
                        x = range(len(strategy_returns))
                        width = 0.35
                        
                        # Plot bars
                        ax3.bar([i - width/2 for i in x], strategy_returns, width, label='Strategy', color='green', alpha=0.7)
                        ax3.bar([i + width/2 for i in x], buyhold_returns, width, label='Buy & Hold', color='blue', alpha=0.7)
                        
                        # Format x-axis
                        ax3.set_xticks(x[::len(x)//6])
                        ax3.set_xticklabels([d.strftime('%Y-%m') for d in strategy_returns.index[::len(x)//6]], rotation=45)
                    
                    ax3.set_ylabel('Daily Return', fontsize=12)
                    ax3.grid(True, alpha=0.3, axis='y')
                    ax3.legend()
                    ax3.set_title('Daily Returns Comparison (Sample)', fontsize=12)
                    
                    # 4. Signal lollipop chart with decision
                    ax4 = plt.subplot(grid[2, 1])
                    
                    if not signal_df.empty and asset_id == 'bitcoin':
                        # For Bitcoin, just show USD signals and decision
                        # Get signal columns (excluding metadata columns)
                        signal_cols = [col for col in signal_df.columns 
                                    if col not in ['usd_positive_count', 'usd_total_count', 
                                                'usd_positive_ratio', 'decision']]
                        
                        # Calculate average signal values
                        signal_averages = {}
                        for col in signal_cols:
                            signal_averages[col] = signal_df[col].mean()
                        
                        # Sort by average value
                        sorted_signals = dict(sorted(signal_averages.items(), key=lambda x: x[1], reverse=True))
                        
                        # Plot lollipop chart
                        y_pos = range(len(sorted_signals))
                        signal_names = list(sorted_signals.keys())
                        signal_values = list(sorted_signals.values())
                        
                        # Plot lines
                        for i, (name, value) in enumerate(zip(signal_names, signal_values)):
                            ax4.plot([0, value], [i, i], color='blue', alpha=0.5)
                        
                        # Plot markers
                        ax4.scatter(signal_values, y_pos, s=100, color='blue', zorder=2)
                        
                        # Add decision line
                        decision_value = signal_df['decision'].mean()
                        ax4.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
                        ax4.text(0.51, len(sorted_signals) - 0.5, 'Buy', color='green', 
                                va='center', ha='left', fontsize=10, fontweight='bold')
                        ax4.text(0.49, len(sorted_signals) - 0.5, 'Sell', color='red', 
                                va='center', ha='right', fontsize=10, fontweight='bold')
                        
                        ax4.set_yticks(y_pos)
                        ax4.set_yticklabels(signal_names)
                        ax4.set_xlim(0, 1)
                        ax4.set_xlabel('Average Signal Value', fontsize=12)
                        ax4.grid(True, alpha=0.3, axis='x')
                        ax4.set_title('USD Signal Analysis', fontsize=12)
                    elif not signal_df.empty:
                        # For other assets, show both USD and BTC signals
                        # Get USD signal columns
                        usd_cols = [col for col in signal_df.columns if col.startswith('USD_')]
                        btc_cols = [col for col in signal_df.columns if col.startswith('BTC_')]
                        
                        # Calculate average USD signal values
                        usd_averages = {}
                        for col in usd_cols:
                            usd_averages[col.replace('USD_', '')] = signal_df[col].mean()
                        
                        # Calculate average BTC signal values
                        btc_averages = {}
                        for col in btc_cols:
                            btc_averages[col.replace('BTC_', '')] = signal_df[col].mean()
                        
                        # Sort by average value
                        sorted_usd = dict(sorted(usd_averages.items(), key=lambda x: x[1], reverse=True))
                        sorted_btc = dict(sorted(btc_averages.items(), key=lambda x: x[1], reverse=True))
                        
                        # Get combined signal names (union of USD and BTC)
                        combined_signals = sorted(set(list(sorted_usd.keys()) + list(sorted_btc.keys())))
                        
                        # Plot lollipop chart
                        y_pos = range(len(combined_signals))
                        
                        # Plot USD signals
                        for i, signal in enumerate(combined_signals):
                            if signal in sorted_usd:
                                value = sorted_usd[signal]
                                ax4.plot([0, value], [i, i], color='blue', alpha=0.5)
                                ax4.scatter(value, i, s=100, color='blue', zorder=2, label='USD' if i == 0 else '')
                        
                        # Plot BTC signals
                        for i, signal in enumerate(combined_signals):
                            if signal in sorted_btc:
                                value = sorted_btc[signal]
                                ax4.plot([0, value], [i, i], color='orange', alpha=0.5)
                                ax4.scatter(value, i, s=100, color='orange', zorder=2, label='BTC' if i == 0 else '')
                        
                        # Add decision thresholds
                        ax4.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
                        ax4.text(0.51, len(combined_signals) - 0.5, 'Buy', color='green', 
                                va='center', ha='left', fontsize=10, fontweight='bold')
                        ax4.text(0.49, len(combined_signals) - 0.5, 'Sell', color='red', 
                                va='center', ha='right', fontsize=10, fontweight='bold')
                        
                        ax4.set_yticks(y_pos)
                        ax4.set_yticklabels(combined_signals)
                        ax4.set_xlim(0, 1)
                        ax4.set_xlabel('Average Signal Value', fontsize=12)
                        ax4.grid(True, alpha=0.3, axis='x')
                        ax4.legend(loc='upper right')
                        ax4.set_title('Signal Analysis (USD & BTC)', fontsize=12)
                    
                    # 5. Enhanced metrics table
                    ax5 = plt.subplot(grid[3, :])
                    ax5.axis('off')
                    
                    # Calculate and format all metrics
                    # Calculate annualized metrics
                    days = (price_series.index[-1] - price_series.index[0]).days
                    years = days / 365
                    
                    strategy_annr = ((1 + asset_result['total_return']) ** (1 / years)) - 1 if years > 0 else 0
                    buyhold_annr = ((1 + asset_result['buy_hold_return']) ** (1 / years)) - 1 if years > 0 else 0
                    
                    # Calculate Sortino ratio if possible
                    strategy_returns_series = asset_result['portfolio_value'].pct_change().dropna()
                    neg_returns = strategy_returns_series[strategy_returns_series < 0]
                    sortino = (strategy_returns_series.mean() / neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 0 else np.nan
                    
                    # Get win rate and calculate average trade metrics
                    win_rate = asset_result.get('win_rate', np.nan)
                    
                    # Count trades
                    trades_count = len(asset_result['trades']) if 'trades' in asset_result else 0
                    
                    # Calculate volatility (annualized)
                    strategy_vol = strategy_returns_series.std() * np.sqrt(252)
                    buyhold_vol = buyhold_returns.dropna().std() * np.sqrt(252)
                    
                    # Calculate max drawdown duration
                    strategy_dd_duration = self._calculate_max_dd_duration(strategy_drawdown)
                    buyhold_dd_duration = self._calculate_max_dd_duration(buyhold_drawdown)
                    
                    # Prepare metrics table data
                    metrics_data = [
                        ['Metric', 'Strategy', 'Buy & Hold', 'Difference/Ratio'],
                        ['Total Return', f"{asset_result['total_return']:.2%}", f"{asset_result['buy_hold_return']:.2%}", 
                         f"{asset_result['total_return'] - asset_result['buy_hold_return']:.2%}"],
                        ['Annualized Return', f"{strategy_annr:.2%}", f"{buyhold_annr:.2%}", 
                         f"{strategy_annr - buyhold_annr:.2%}"],
                        ['Sharpe Ratio', f"{asset_result['sharpe_ratio']:.2f}", "N/A", "N/A"],
                        ['Sortino Ratio', f"{sortino:.2f}" if not np.isnan(sortino) else "N/A", "N/A", "N/A"],
                        ['Volatility (Ann.)', f"{strategy_vol:.2%}", f"{buyhold_vol:.2%}", 
                         f"{strategy_vol/buyhold_vol:.2f}x" if buyhold_vol > 0 else "N/A"],
                        ['Max Drawdown', f"{asset_result['max_drawdown']:.2%}", f"{MetricsCalculator.max_drawdown(cumulative_price):.2%}", 
                         f"{asset_result['max_drawdown']/MetricsCalculator.max_drawdown(cumulative_price):.2f}x" if MetricsCalculator.max_drawdown(cumulative_price) > 0 else "N/A"],
                        ['Max DD Duration', f"{strategy_dd_duration} days", f"{buyhold_dd_duration} days", 
                         f"{strategy_dd_duration/buyhold_dd_duration:.2f}x" if buyhold_dd_duration > 0 else "N/A"],
                        ['Win Rate', f"{win_rate:.2%}" if not np.isnan(win_rate) else "N/A", "N/A", "N/A"],
                        ['# of Trades', f"{trades_count}", "1", f"{trades_count}x"],
                        ['Transaction Cost', f"{0.1:.1%}" if asset_id == 'bitcoin' else f"{0.5:.1%}", "0.0%", "N/A"]
                    ]
                    
                    # Create the table
                    table = ax5.table(
                        cellText=metrics_data,
                        loc='center',
                        cellLoc='center'
                    )
                    
                    # Style the table
                    table.auto_set_font_size(False)
                    table.set_fontsize(11)
                    table.scale(1, 1.8)
                    
                    # Make header row and first column bold
                    for (row, col), cell in table.get_celld().items():
                        if row == 0 or col == 0:
                            cell.set_text_props(fontproperties=plt.matplotlib.font_manager.FontProperties(weight='bold'))
                            cell.set_facecolor('#e6e6e6')
                    
                    # Color-code the difference/ratio column
                    for row in range(1, len(metrics_data)):
                        cell = table.get_celld().get((row, 3))
                        if cell and "x" in cell.get_text().get_text():
                            # For ratios (like volatility or drawdown), lower is better
                            value_str = cell.get_text().get_text().split('x')[0]
                            try:
                                value = float(value_str)
                                if value < 1:
                                    cell.set_facecolor('#d4f7d4')  # Light green
                                else:
                                    cell.set_facecolor('#f7d4d4')  # Light red
                            except ValueError:
                                pass
                        elif cell and "%" in cell.get_text().get_text():
                            # For percentages (like returns), higher is better
                            value_str = cell.get_text().get_text().rstrip('%')
                            try:
                                value = float(value_str)
                                if value > 0:
                                    cell.set_facecolor('#d4f7d4')  # Light green
                                else:
                                    cell.set_facecolor('#f7d4d4')  # Light red
                            except ValueError:
                                pass
                    
                    plt.suptitle(f'{asset_id.capitalize()} Performance Analysis', fontsize=16, y=0.99)
                    plt.tight_layout(rect=[0, 0, 1, 0.98])
                    plt.subplots_adjust(hspace=0.3)
                    
                plt.show()
            except Exception as e:
                self.logger.error(f"Error plotting results: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
    
    def _calculate_max_dd_duration(self, drawdown_series):
        """Calculate the maximum drawdown duration in days."""
        if drawdown_series.empty:
            return 0
            
        max_duration = 0
        current_duration = 0
        in_drawdown = False
        
        for dd in drawdown_series:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    in_drawdown = False
                    current_duration = 0
        
        # Check if still in drawdown at the end of the series
        if in_drawdown:
            max_duration = max(max_duration, current_duration)
            
        return max_duration

    def backtest_asset(self,
                      asset_id: str,
                      start_date: Union[str, datetime],
                      end_date: Union[str, datetime],
                      initial_capital: float = 10000.0,
                      trade_cost: float = 0.001,
                      slippage_pct: float = 0.0,
                      usd_signals: Optional[List[str]] = None,
                      btc_signals: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Backtest a trading strategy for a single asset using specified signals.
        
        This is a wrapper around the PortfolioBacktester.backtest_asset method that provides
        a simpler interface and handles creating the backtester if needed.
        
        Args:
            asset_id: Asset ID to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital for backtesting
            trade_cost: Fee rate for transactions
            slippage_pct: Slippage percentage for trading
            usd_signals: List of USD signal names to use
            btc_signals: List of BTC signal names to use
            
        Returns:
            dict: Dictionary with backtest results, including dataframe, evaluation metrics, and summary
        """
        # Ensure the asset is loaded
        if asset_id not in self.assets:
            self.logger.error(f"Asset {asset_id} not loaded. Cannot backtest.")
            return None
        
        # If no signals are provided, use default criteria
        if (usd_signals is None or len(usd_signals) == 0) and (btc_signals is None or len(btc_signals) == 0):
            # For Bitcoin, default to standard regime signals
            if asset_id == 'bitcoin':
                usd_signals = ["BullLowVarianceSignalUSD", "RSI_Oversold_USD"]
            # For altcoins, include both USD and BTC signals
            else:
                usd_signals = ["BullLowVarianceSignalUSD", "RSI_Oversold_USD"]
                btc_signals = ["BullLowVarianceSignalBTC", "RSI_Oversold_BTC"]
        
        # Enforce Bitcoin signal rules
        if asset_id == 'bitcoin' and btc_signals:
            self.logger.warning("BTC signals are not applicable for Bitcoin itself. Using only USD signals.")
            btc_signals = None
        
        # Ensure signals are registered
        asset = self.assets.get(asset_id)
        if asset:
            # Check if USD signals are registered
            if usd_signals:
                unregistered_signals = []
                for signal_name in usd_signals:
                    if not asset.has_signal(signal_name):
                        unregistered_signals.append(signal_name)
                
                # Register any missing signals
                if unregistered_signals:
                    self.logger.info(f"Registering USD signals for {asset_id}: {unregistered_signals}")
                    self.register_signals(asset_id=asset_id, signal_names=unregistered_signals, calculate_values=True)
            
            # Check if BTC signals are registered (for altcoins only)
            if btc_signals and asset_id != 'bitcoin':
                unregistered_signals = []
                for signal_name in btc_signals:
                    if not asset.has_signal(signal_name):
                        unregistered_signals.append(signal_name)
                
                # Register any missing signals
                if unregistered_signals:
                    self.logger.info(f"Registering BTC signals for {asset_id}: {unregistered_signals}")
                    self.register_signals(asset_id=asset_id, signal_names=unregistered_signals, calculate_values=True)
        
        # Create backtester if we don't have one
        from ..portfolio.backtest import PortfolioBacktester
        backtester = PortfolioBacktester(None, self)
        
        # Run the backtest
        self.logger.info(f"Running backtest for {asset_id} from {start_date} to {end_date}")
        result = backtester.backtest_asset(
            asset_id=asset_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            trade_cost=trade_cost,
            slippage_pct=slippage_pct,
            usd_signals=usd_signals,
            btc_signals=btc_signals
        )
        
        # If we got a result, store it in the asset data object for future reference
        if result and asset_id in self.assets:
            asset = self.assets[asset_id]
            # Create a unique name for the backtest based on the signals used
            backtest_name = f"USD:{':'.join(usd_signals or [])}"
            if btc_signals:
                backtest_name += f" BTC:{':'.join(btc_signals)}"
            self.logger.info(f"Stored backtest results in asset {asset_id} with name '{backtest_name}'")
            asset.backtest_results = result
        
        return result
    
    def backtest_assets(self,
                       asset_ids: List[str],
                       start_date: Union[str, datetime],
                       end_date: Union[str, datetime],
                       initial_capital: float = 10000.0,
                       btc_cost: float = 0.001,
                       alt_cost: float = 0.005,
                       slippage_pct: float = 0.0,
                       usd_signals: Optional[List[str]] = None,
                       btc_signals: Optional[List[str]] = None,
                       use_btc_filter: bool = False) -> Dict[str, Any]:
        """Backtest multiple assets at once, automatically handling differences between BTC and altcoins.
        
        This method will:
        1. Process each asset in the list
        2. Apply only USD signals to Bitcoin
        3. Apply both USD and BTC signals to altcoins
        4. When use_btc_filter=True, only allow altcoin buys when Bitcoin signals are bullish
        5. Return consolidated results for all assets
        
        Args:
            asset_ids: List of asset IDs to backtest
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital for each asset backtest
            btc_cost: Trading cost for Bitcoin
            alt_cost: Trading cost for altcoins
            slippage_pct: Slippage percentage for trading
            usd_signals: List of USD signal names to use (applied to all assets)
            btc_signals: List of BTC signal names to use (applied only to altcoins)
            use_btc_filter: If True, only allow altcoin buys when Bitcoin signals are bullish
            
        Returns:
            dict: Dictionary with:
                - asset_results: Dictionary mapping asset IDs to their backtest results
                - summary: DataFrame with comparison of results across assets (using tickers)
        """
        # Validate and normalize asset_ids
        if isinstance(asset_ids, str):
            # If a single string was passed, convert to a list
            self.logger.warning(f"Converting single asset ID string '{asset_ids}' to a list")
            asset_ids = [asset_ids]
        elif not isinstance(asset_ids, (list, tuple, set)):
            # If not a string or a list/tuple/set, try to convert to list
            try:
                asset_ids = list(asset_ids)
            except Exception as e:
                self.logger.error(f"Could not convert asset_ids to list: {str(e)}")
                return {'asset_results': {}, 'summary': pd.DataFrame()}
        
        # Filter out invalid asset IDs
        valid_asset_ids = []
        for asset_id in asset_ids:
            if not isinstance(asset_id, str):
                self.logger.warning(f"Skipping non-string asset ID: {asset_id}")
                continue
            if len(asset_id) <= 1:
                self.logger.warning(f"Skipping too short asset ID: '{asset_id}'")
                continue
            valid_asset_ids.append(asset_id)
        
        if len(valid_asset_ids) < len(asset_ids):
            self.logger.warning(f"Filtered out {len(asset_ids) - len(valid_asset_ids)} invalid asset IDs")
        
        if not valid_asset_ids:
            self.logger.error("No valid asset IDs provided")
            return {'asset_results': {}, 'summary': pd.DataFrame()}
        
        self.logger.info(f"Starting backtest for {len(valid_asset_ids)} assets from {start_date} to {end_date}")
        
        # If no signals are provided, use default criteria
        if (usd_signals is None or len(usd_signals) == 0) and (btc_signals is None or len(btc_signals) == 0):
            usd_signals = ["BullLowVarianceSignalUSD", "RSI_Oversold_USD"]
            btc_signals = ["BullLowVarianceSignalBTC", "RSI_Oversold_BTC"]
            self.logger.info(f"Using default signals: USD={usd_signals}, BTC={btc_signals}")
        
        # Initialize results dictionary
        results = {}
        
        # If we're using Bitcoin as a filter for altcoin trades, we need to run Bitcoin first
        bitcoin_decisions = None
        if use_btc_filter and 'bitcoin' in valid_asset_ids:
            self.logger.info("Running Bitcoin backtest first to use as a filter for altcoins...")
            
            # Make sure bitcoin is loaded
            if 'bitcoin' not in self.assets:
                self.logger.info("Loading data for bitcoin...")
                self.load_data(assets='bitcoin')
                
                # Exit if we couldn't load bitcoin
                if 'bitcoin' not in self.assets:
                    self.logger.error("Failed to load data for bitcoin, cannot use as filter.")
                    use_btc_filter = False
            
            if 'bitcoin' in self.assets:
                # Run the Bitcoin backtest
                bitcoin_result = self.backtest_asset(
                    asset_id='bitcoin',
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    trade_cost=btc_cost,
                    slippage_pct=slippage_pct,
                    usd_signals=usd_signals,
                    btc_signals=None  # Bitcoin doesn't use BTC signals
                )
                
                # Store the Bitcoin result
                results['bitcoin'] = bitcoin_result
                
                # Extract the trading decisions if the backtest was successful
                if bitcoin_result and 'results_df' in bitcoin_result:
                    # Get the trading decisions (1 for buy, 0 for sell)
                    bitcoin_decisions = bitcoin_result['results_df'].get('final_decision_shifted', pd.Series())
                    
                    # Log information about the Bitcoin decisions
                    btc_buy_days = bitcoin_decisions[bitcoin_decisions == 1].count()
                    total_days = len(bitcoin_decisions)
                    self.logger.info(f"Bitcoin signaled buy on {btc_buy_days} out of {total_days} days ({btc_buy_days/total_days:.2%})")
                else:
                    self.logger.error("Bitcoin backtest failed, cannot use as filter.")
                    use_btc_filter = False
                    
        # Process each asset (skipping Bitcoin if we already processed it)
        for asset_id in valid_asset_ids:
            # Skip bitcoin if we already ran it above
            if asset_id == 'bitcoin' and 'bitcoin' in results:
                continue
                
            self.logger.info(f"Backtesting {asset_id}...")
            
            # Make sure the asset is loaded
            if asset_id not in self.assets:
                self.logger.info(f"Loading data for {asset_id}...")
                self.load_data(assets=asset_id)
                
                # Skip if we couldn't load the asset
                if asset_id not in self.assets:
                    self.logger.error(f"Failed to load data for {asset_id}, skipping.")
                    continue
            
            # Set trade cost based on asset type
            trade_cost = btc_cost if asset_id == 'bitcoin' else alt_cost
            
            # For Bitcoin, only use USD signals
            if asset_id == 'bitcoin':
                asset_result = self.backtest_asset(
                    asset_id=asset_id,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    trade_cost=trade_cost,
                    slippage_pct=slippage_pct,
                    usd_signals=usd_signals,
                    btc_signals=None  # Bitcoin doesn't use BTC signals
                )
            # For altcoins, use both USD and BTC signals, and apply Bitcoin filter if enabled
            else:
                # If we're using Bitcoin as a filter and we have valid Bitcoin decisions
                if use_btc_filter and bitcoin_decisions is not None and len(bitcoin_decisions) > 0:
                    self.logger.info(f"Using Bitcoin as a filter for {asset_id}")
                    
                    # Custom backtest that uses Bitcoin decisions as a filter
                    asset_result = self._backtest_asset_with_btc_filter(
                        asset_id=asset_id,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        trade_cost=trade_cost,
                        slippage_pct=slippage_pct,
                        usd_signals=usd_signals,
                        btc_signals=btc_signals,
                        bitcoin_decisions=bitcoin_decisions
                    )
                else:
                    # Standard backtest without Bitcoin filter
                    asset_result = self.backtest_asset(
                        asset_id=asset_id,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        trade_cost=trade_cost,
                        slippage_pct=slippage_pct,
                        usd_signals=usd_signals,
                        btc_signals=btc_signals
                    )
            
            results[asset_id] = asset_result
            
            # Print summary of results
            if asset_result:
                metrics = asset_result['metrics_comparison']
                self.logger.info(f"  Total Return: Strategy={metrics.loc['Total Return', 'Strategy']:.2%}, Buy & Hold={metrics.loc['Total Return', 'Buy & Hold']:.2%}")
                self.logger.info(f"  Sharpe Ratio: Strategy={metrics.loc['Sharpe Ratio', 'Strategy']:.2f}")
                if 'Win Rate' in metrics.index:
                    self.logger.info(f"  Win Rate: Strategy={metrics.loc['Win Rate', 'Strategy']:.2%}")
                self.logger.info(f"  Trade Count: {len(asset_result['trades'])}")
        
        # Create a mapping from asset_id to ticker
        ticker_mapping = {}
        try:
            from ..data.data_loader import DataLoader
            data_loader = DataLoader(
                data_path=self.data_path,
                btc_data_path=self.btc_data_path,
                ssr_data_path=self.ssr_data_path,
                market_data_path=self.market_data_path,
                asset_ids=self.asset_ids
            )
            
            for asset_id in results.keys():
                ticker = data_loader.get_ticker_from_id(asset_id)
                ticker_mapping[asset_id] = ticker.upper() if ticker else asset_id.upper()
        except Exception as e:
            self.logger.warning(f"Could not create ticker mapping: {e}")
            # Fallback to using asset_ids as tickers
            for asset_id in results.keys():
                ticker_mapping[asset_id] = asset_id.upper()
        
        # Create summary dataframe
        summary_data = []
        
        for asset_id, result in results.items():
            if result is None:
                continue
            
            # Get ticker for the asset
            ticker = ticker_mapping.get(asset_id, asset_id.upper())
            
            # Skip stablecoins (USDT, USDC, TUSD)
            if ticker in ['USDT', 'USDC', 'TUSD']:
                self.logger.info(f"Skipping stablecoin {ticker} in summary table")
                continue
                
            strategy_metrics = result['strategy_metrics']
            buy_hold_metrics = result['buy_hold_metrics']
            
            # Calculate total transaction costs from trades
            total_cost = 0.0
            if 'trades' in result:
                for trade in result['trades']:
                    entry_cost = trade.get('entry_cost', 0.0)
                    exit_cost = trade.get('exit_cost', 0.0)
                    total_cost += entry_cost + exit_cost
            
            # Calculate peak performance (maximum portfolio value)
            peak_return = 0.0
            if 'results_df' in result and 'portfolio_value' in result['results_df'].columns:
                max_portfolio_value = result['results_df']['portfolio_value'].max()
                peak_return = (max_portfolio_value / initial_capital) - 1
            
            # Calculate peak return for buy & hold
            bh_peak_return = 0.0
            if 'results_df' in result and 'close' in result['results_df'].columns:
                # Most reliable source: close prices directly from results_df
                price_series = result['results_df']['close']
                max_price = price_series.max()
                initial_price = price_series.iloc[0]
                if initial_price > 0:
                    bh_peak_return = (max_price / initial_price) - 1
            elif asset_id in self.assets:
                # Fallback: get the price data from the original asset data
                asset_data = self.get_asset_data(asset_id)
                if asset_data is not None:
                    # Filter to match the backtest period
                    price_data = asset_data.get_price_data()
                    if start_date is not None:
                        if isinstance(start_date, str):
                            start_date = pd.to_datetime(start_date)
                        price_data = price_data[price_data.index >= start_date]
                    if end_date is not None:
                        if isinstance(end_date, str):
                            end_date = pd.to_datetime(end_date)
                        price_data = price_data[price_data.index <= end_date]
                    
                    if not price_data.empty and 'close' in price_data.columns:
                        price_series = price_data['close']
                        max_price = price_series.max()
                        initial_price = price_series.iloc[0]
                        if initial_price > 0:
                            bh_peak_return = (max_price / initial_price) - 1
            else:
                self.logger.warning(f"Could not extract close price data for {asset_id} peak return calculation")
            
            # Calculate retention ratio (how much of peak return is retained)
            retention_ratio = 0.0
            if peak_return > 0:
                retention_ratio = strategy_metrics['total_return'] / peak_return
            
            summary_data.append({
                'Asset': ticker,  # Use ticker instead of asset_id
                'Total Return (%)': round(strategy_metrics['total_return'] * 100, 2),
                'Peak Return (%)': round(peak_return * 100, 2),
                'Buy & Hold Return (%)': round(buy_hold_metrics['total_return'] * 100, 2),
                'Buy & Hold Peak (%)': round(bh_peak_return * 100, 2),
                'Sharpe Ratio': round(strategy_metrics['sharpe_ratio'], 2),
                'Buy & Hold Sharpe': round(buy_hold_metrics['sharpe_ratio'], 2),
                'Sortino Ratio': round(strategy_metrics['sortino_ratio'], 2),
                'Buy & Hold Sortino': round(buy_hold_metrics['sortino_ratio'], 2),
                'Max Drawdown (%)': round(strategy_metrics['max_drawdown'] * 100, 2),
                'Buy & Hold Drawdown (%)': round(buy_hold_metrics['max_drawdown'] * 100, 2),
                'Annualized Volatility (%)': round(strategy_metrics['annualized_volatility'] * 100, 2),
                'Buy & Hold Volatility (%)': round(buy_hold_metrics['annualized_volatility'] * 100, 2),
                'Trade Count': len(result['trades']) if 'trades' in result else 0,
                'Total Cost': round(total_cost, 2)
            })
        
        # Create summary dataframe
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            summary_df = summary_df.set_index('Asset')
            
        self.logger.info(f"Completed backtesting for {len(results)} assets")
        
        return {
            'asset_results': results,
            'summary': summary_df
        }
        
    def _backtest_asset_with_btc_filter(self,
                                      asset_id: str,
                                      start_date: Union[str, datetime],
                                      end_date: Union[str, datetime],
                                      initial_capital: float = 10000.0,
                                      trade_cost: float = 0.005,
                                      slippage_pct: float = 0.0,
                                      usd_signals: Optional[List[str]] = None,
                                      btc_signals: Optional[List[str]] = None,
                                      bitcoin_decisions: pd.Series = None) -> Optional[Dict[str, Any]]:
        """Run a backtest for an altcoin with Bitcoin decisions as a filter.
        
        This method first calculates the altcoin's own trading signals, then applies
        the Bitcoin decisions as a filter - the altcoin can only enter a position
        when Bitcoin signals are bullish.
        
        Args:
            asset_id: Asset ID to backtest (must be an altcoin)
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_capital: Initial capital for backtesting
            trade_cost: Fee rate for transactions
            slippage_pct: Slippage percentage for trading
            usd_signals: List of USD signal names to use
            btc_signals: List of BTC signal names to use
            bitcoin_decisions: Series of Bitcoin trading decisions (1 for buy, 0 for sell)
            
        Returns:
            dict: Dictionary with backtest results
        """
        # Ensure this is not being called for Bitcoin
        if asset_id == 'bitcoin':
            self.logger.error("_backtest_asset_with_btc_filter should not be called for Bitcoin")
            return None
            
        # Ensure we have Bitcoin decisions
        if bitcoin_decisions is None or len(bitcoin_decisions) == 0:
            self.logger.error("No Bitcoin decisions provided for filtering")
            return None
        
        # First, make sure all required signals are registered
        asset = self.assets.get(asset_id)
        if asset:
            # Check if USD signals are registered
            if usd_signals:
                unregistered_signals = []
                for signal_name in usd_signals:
                    if not asset.has_signal(signal_name):
                        unregistered_signals.append(signal_name)
                
                # Register any missing signals
                if unregistered_signals:
                    self.logger.info(f"Registering USD signals for {asset_id}: {unregistered_signals}")
                    self.register_signals(asset_id=asset_id, signal_names=unregistered_signals, calculate_values=True)
            
            # Check if BTC signals are registered
            if btc_signals:
                unregistered_signals = []
                for signal_name in btc_signals:
                    if not asset.has_signal(signal_name):
                        unregistered_signals.append(signal_name)
                
                # Register any missing signals
                if unregistered_signals:
                    self.logger.info(f"Registering BTC signals for {asset_id}: {unregistered_signals}")
                    self.register_signals(asset_id=asset_id, signal_names=unregistered_signals, calculate_values=True)
        
        # First, run a standard backtest to get the asset's own signals
        from ..portfolio.backtest import PortfolioBacktester
        backtester = PortfolioBacktester(None, self)
        
        # Run the standard backtest to get the signals
        standard_result = backtester.backtest_asset(
            asset_id=asset_id,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            trade_cost=trade_cost,
            slippage_pct=slippage_pct,
            usd_signals=usd_signals,
            btc_signals=btc_signals
        )
        
        if not standard_result or 'results_df' not in standard_result:
            self.logger.error(f"Standard backtest failed for {asset_id}")
            return None
            
        # Get the backtest DataFrame
        backtest_df = standard_result['results_df'].copy()
        
        # Now apply the Bitcoin filter to the final decision
        # Reindex the Bitcoin decisions to match the altcoin's date index
        btc_decisions_aligned = bitcoin_decisions.reindex(backtest_df.index, fill_value=0)
        
        # Store the original decisions
        backtest_df['original_decision'] = backtest_df['final_decision_shifted'] 
        
        # Apply the Bitcoin filter - can only enter when Bitcoin is bullish
        # Format: original decision AND Bitcoin decision
        # Convert to boolean first to avoid "unsupported operand type(s) for &: 'float' and 'float'" error
        backtest_df['final_decision_shifted'] = (
            backtest_df['final_decision_shifted'].astype(bool) & 
            btc_decisions_aligned.astype(bool)
        ).astype(int)
        
        self.logger.info(f"Applied Bitcoin filter to {asset_id}: "
                        f"Original buy signals: {backtest_df['original_decision'].sum()}, "
                        f"After BTC filter: {backtest_df['final_decision_shifted'].sum()}")
        
        # Now run the trading simulation with the filtered decisions
        # Clear the existing trading related columns to start from scratch
        for col in ['position', 'cash', 'holdings_qty', 'holdings_value', 'portfolio_value', 
                    'trade_executed', 'daily_return', 'strategy_cumulative_return']:
            if col in backtest_df.columns:
                backtest_df[col] = np.nan
        
        # Initialize trading simulation
        backtest_df['position'] = 0
        backtest_df['cash'] = initial_capital
        backtest_df['holdings_qty'] = 0.0
        backtest_df['holdings_value'] = 0.0
        backtest_df['portfolio_value'] = initial_capital
        backtest_df['trade_executed'] = False
        backtest_df['daily_return'] = 0.0
        backtest_df['strategy_cumulative_return'] = 1.0
        
        # Trading simulation with filtered decisions
        current_cash = initial_capital
        current_holdings_qty = 0.0
        total_cost_rate = trade_cost + slippage_pct
        trades = []
        
        # Track position entry details
        entry_price = 0.0
        entry_date = None
        trade_id = 0
        
        # Exclude the first row with NaN values
        for date in backtest_df.index[1:]:
            row = backtest_df.loc[date]
            signal = row['final_decision_shifted']
            open_price = row['open']
            close_price = row['close']
            
            # Skip if signal is NaN or prices are invalid
            if pd.isna(signal) or open_price <= 0:
                backtest_df.loc[date, 'cash'] = current_cash
                backtest_df.loc[date, 'holdings_qty'] = current_holdings_qty
                backtest_df.loc[date, 'holdings_value'] = current_holdings_qty * close_price
                backtest_df.loc[date, 'portfolio_value'] = current_cash + backtest_df.loc[date, 'holdings_value']
                continue
            
            # Flag for trade execution
            trade_executed = False
            
            # Buy signal (1) and not holding
            if signal == 1 and current_holdings_qty == 0:
                trade_id += 1
                
                # Calculate transaction cost
                transaction_cost = current_cash * total_cost_rate
                cash_for_purchase = current_cash - transaction_cost
                
                # Buy with all available cash
                quantity = cash_for_purchase / open_price
                
                # Update holdings
                current_cash = 0
                current_holdings_qty = quantity
                entry_price = open_price
                entry_date = date
                
                # Record trade for later analysis
                entry_trade = {
                    'trade_id': trade_id,
                    'asset': asset_id,
                    'entry_date': date,
                    'exit_date': None,
                    'holding_days': 0,
                    'entry_price': open_price,
                    'exit_price': None,
                    'price_return': 0.0,
                    'entry_value': cash_for_purchase,
                    'entry_cost': transaction_cost,
                    'exit_value': None,
                    'exit_cost': None,
                    'trade_return': 0.0,
                    'is_open': True
                }
                trades.append(entry_trade)
                trade_executed = True
                
                # Mark the position
                backtest_df.loc[date, 'position'] = 1
                
            # Sell signal (0) and currently holding
            elif signal == 0 and current_holdings_qty > 0:
                # Calculate gross value and transaction cost
                gross_value = current_holdings_qty * open_price
                transaction_cost = gross_value * total_cost_rate
                net_value = gross_value - transaction_cost
                
                # Update holdings
                current_cash = net_value
                current_holdings_qty = 0
                
                # Update the exit information for the last entry trade
                holding_days = (date - entry_date).days if entry_date else 0
                price_return = (open_price / entry_price - 1) if entry_price > 0 else 0
                trade_return = (net_value / trades[-1]['entry_value'] - 1) if trades[-1]['entry_value'] > 0 else 0
                
                trades[-1].update({
                    'exit_date': date,
                    'holding_days': holding_days,
                    'exit_price': open_price,
                    'price_return': price_return,
                    'exit_value': gross_value,
                    'exit_cost': transaction_cost,
                    'trade_return': trade_return,
                    'is_open': False
                })
                
                trade_executed = True
                
                # Mark the position
                backtest_df.loc[date, 'position'] = 0
                
            else:
                # No change in position
                backtest_df.loc[date, 'position'] = 1 if current_holdings_qty > 0 else 0
            
            # Update the backtest DataFrame
            backtest_df.loc[date, 'cash'] = current_cash
            backtest_df.loc[date, 'holdings_qty'] = current_holdings_qty
            backtest_df.loc[date, 'holdings_value'] = current_holdings_qty * close_price
            backtest_df.loc[date, 'portfolio_value'] = current_cash + backtest_df.loc[date, 'holdings_value']
            backtest_df.loc[date, 'trade_executed'] = trade_executed
        
        # Calculate strategy returns
        backtest_df['daily_return'] = backtest_df['portfolio_value'].pct_change()
        backtest_df['strategy_cumulative_return'] = (1 + backtest_df['daily_return']).cumprod()
        
        # If the position is still open at the end, close it for metrics calculation
        if current_holdings_qty > 0 and len(trades) > 0 and trades[-1]['is_open']:
            last_date = backtest_df.index[-1]
            last_close = backtest_df.loc[last_date, 'close']
            
            gross_value = current_holdings_qty * last_close
            transaction_cost = gross_value * total_cost_rate
            net_value = gross_value - transaction_cost
            
            holding_days = (last_date - entry_date).days if entry_date else 0
            price_return = (last_close / entry_price - 1) if entry_price > 0 else 0
            trade_return = (net_value / trades[-1]['entry_value'] - 1) if trades[-1]['entry_value'] > 0 else 0
            
            trades[-1].update({
                'exit_date': last_date,
                'holding_days': holding_days,
                'exit_price': last_close,
                'price_return': price_return,
                'exit_value': gross_value,
                'exit_cost': transaction_cost,
                'trade_return': trade_return,
                'is_open': False  # Mark as closed for final analysis
            })
        
        # Calculate performance metrics for the filtered strategy
        strategy_metrics = backtester._calculate_backtest_metrics(
            backtest_df, 
            trades, 
            'daily_return', 
            'strategy_cumulative_return', 
            initial_capital
        )
        
        # Reuse the buy & hold metrics from the standard backtest
        buy_hold_metrics = standard_result['buy_hold_metrics']
        
        # Create metrics comparison table
        metrics_comparison = pd.DataFrame({
            'Strategy': [
                strategy_metrics['total_return'],
                strategy_metrics['max_drawdown'],
                strategy_metrics['sharpe_ratio'],
                strategy_metrics['sortino_ratio'],
                strategy_metrics['calmar_ratio'],
                strategy_metrics['win_rate'],
                strategy_metrics['annualized_volatility'],
                strategy_metrics['annualized_return'],
                strategy_metrics.get('accuracy', np.nan),
                strategy_metrics.get('precision', np.nan),
                strategy_metrics.get('recall', np.nan),
                strategy_metrics.get('f1_score', np.nan)
            ],
            'Buy & Hold': [
                buy_hold_metrics['total_return'],
                buy_hold_metrics['max_drawdown'],
                buy_hold_metrics['sharpe_ratio'],
                buy_hold_metrics['sortino_ratio'],
                buy_hold_metrics['calmar_ratio'],
                buy_hold_metrics['win_rate'],
                buy_hold_metrics['annualized_volatility'],
                buy_hold_metrics['annualized_return'],
                buy_hold_metrics.get('accuracy', np.nan),
                buy_hold_metrics.get('precision', np.nan),
                buy_hold_metrics.get('recall', np.nan),
                buy_hold_metrics.get('f1_score', np.nan)
            ]
        }, index=[
            'Total Return',
            'Max Drawdown',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Win Rate',
            'Annualized Volatility',
            'Annualized Return',
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score'
        ])
        
        # Calculate total transaction costs
        total_cost = 0.0
        for trade in trades:
            entry_cost = trade.get('entry_cost', 0.0)
            exit_cost = trade.get('exit_cost', 0.0)
            total_cost += entry_cost + exit_cost
            
        # Return the filtered backtest results
        return {
            'asset_id': asset_id,
            'results_df': backtest_df,
            'trades': trades,
            'metrics_comparison': metrics_comparison,
            'strategy_metrics': strategy_metrics,
            'buy_hold_metrics': buy_hold_metrics,
            'parameters': {
                'initial_capital': initial_capital,
                'trade_cost': trade_cost,
                'slippage_pct': slippage_pct,
                'usd_signals': usd_signals,
                'btc_signals': btc_signals,
                'start_date': start_date,
                'end_date': end_date,
                'bitcoin_filtered': True
            }
        }
    
    def get_registered_signals(self, asset_id: str) -> List[str]:
        """Get registered signals for a specific asset.
        
        Args:
            asset_id: ID of the asset
            
        Returns:
            list: List of registered signal names
        """
        asset_data = self.get_asset_data(asset_id)
        if asset_data:
            return asset_data.get_signal_names()
        else:
            self.logger.warning(f"Asset {asset_id} not found")
            return [] 