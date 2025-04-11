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
import json

# Import core data structures
from .asset_data import AssetData
from .signal_data import SignalData
from .signal_storage import SignalEvaluationStorage

# Import existing components
from ..signals.signal_registry import get_signal, get_all_signals, registry, get_signal_names
from ..signals.signal_event_tracker import SignalEventTracker
from ..analysis.forward_returns.signal_evaluator import SignalEvaluator
from ..portfolio.signal_combiner import SignalCombiner
from ..portfolio.event_backtest import EventDrivenBacktester
from ..portfolio.portfolio_manager import PortfolioManager


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
        use_meta_labeling: bool = False,
        lookback_days: Union[int, str] = 365,
        parallelize: bool = True,
        asset_ids: Optional[List[str]] = None,
        verbose: bool = False,
        num_workers: Optional[int] = None,
        use_stored_evaluations: bool = True,
        signal_storage_dir: str = None,
        signal_storage_file: str = 'signal_evaluations.json'
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
            use_meta_labeling: Whether to use meta-labeling (default: False)
            lookback_days: Number of days to look back or 'all' (default: 365)
            parallelize: Whether to use parallel processing (default: True)
            asset_ids: List of asset IDs to initially load
            verbose: Whether to show detailed log messages
            num_workers: Number of worker processes to use for parallel processing.
                         If None, defaults to number of CPU cores minus 1 (minimum 1).
            use_stored_evaluations: Whether to use stored signal evaluations (default: True)
            signal_storage_dir: Directory for signal evaluation storage (default: data_dir/signal_evaluations)
            signal_storage_file: Filename for signal evaluation storage
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
        self.use_meta_labeling = use_meta_labeling
        self.lookback_days = lookback_days
        self.parallelize = parallelize
        self.verbose = verbose
        self.use_stored_evaluations = use_stored_evaluations
        
        # Set number of workers for parallel processing
        if num_workers is None:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_workers = max(1, num_workers)
        self.logger.info(f"Using {self.num_workers} workers for parallel processing")
        
        # Storage for asset data
        self.assets = {}  # {asset_id: AssetData}
        
        # Initialize components
        self.signal_evaluator = SignalEvaluator(lookback_days=lookback_days if isinstance(lookback_days, int) else 365)
        self.signal_combiner = SignalCombiner(
            signal_evaluator=self.signal_evaluator,
            threshold=signal_threshold,
            auto_weights=True
        )
        self.signal_event_tracker = SignalEventTracker(threshold=signal_threshold)
        self.portfolio_manager = PortfolioManager()
        
        # Initialize signal evaluation storage
        if signal_storage_dir is None:
            signal_storage_dir = os.path.join(data_dir, 'signal_evaluations')
        
        self.signal_storage = SignalEvaluationStorage(
            storage_dir=signal_storage_dir,
            filename=signal_storage_file,
        )
        
        # Store whether to use stored evaluations (but don't load automatically)
        # The use_stored_evaluations flag only indicates preference, not automatic loading
        self.use_stored_evaluations = use_stored_evaluations
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            self.logger.info(f"Created data directory: {data_dir}")
            
        # Initialize cache for loaded data
        self.data_cache = {}
        
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
        calculate_values: bool = True
    ) -> Dict[str, SignalData]:
        """Register signals for a specific asset.
        
        Args:
            asset_id: ID of the asset
            signal_names: List of signal names to register. If None, register all available signals.
            calculate_values: Whether to calculate signal values
            
        Returns:
            dict: Dictionary mapping signal names to SignalData objects
        """
        self.logger.info(f"Registering signals for {asset_id}...")
        
        # Get asset data
        asset_data = self.get_asset_data(asset_id)
        if asset_data is None:
            self.logger.error(f"Asset data not found for {asset_id}")
            return {}
            
        # Get price data
        price_data = asset_data.price_data
        if price_data.empty:
            self.logger.error(f"Price data is empty for {asset_id}")
            return {}
        
        # Reset index to have date as a column if it's the index
        if isinstance(price_data.index, pd.DatetimeIndex):
            price_data = price_data.reset_index()
            
        # Make sure asset_id is available in the data for signal calculations
        if 'asset_id' not in price_data.columns:
            price_data['asset_id'] = asset_id
            
        # Get signal names to register
        if signal_names is None:
            # Use all available signals
            signal_names = get_signal_names()
            self.logger.info(f"Using all {len(signal_names)} available signals")
        else:
            self.logger.info(f"Using {len(signal_names)} specified signals")
            
        if not signal_names:
            self.logger.warning("No signals to register")
            return {}
            
        # Initialize result dictionary for registered signals
        registered_signals = {}
        
        # Register signals
        for signal_name in signal_names:
            try:
                # Get the signal class
                signal = get_signal(signal_name)
                if signal is None:
                    self.logger.warning(f"Signal {signal_name} not found in registry")
                    continue
                
                # Set up signal parameters based on type
                signal_params = {'asset_id': asset_id}
                
                # Add SSR data path for SSR signals
                if signal_name.startswith('SSR_') and self.ssr_data_path:
                    signal_params['ssr_data_path'] = self.ssr_data_path
                    self.logger.info(f"Using SSR data path for {signal_name}: {self.ssr_data_path}")
                    # Also set the environment variable for better compatibility
                    os.environ['SSR_DATA_PATH'] = self.ssr_data_path
                
                # Add these parameters to the signal
                if hasattr(signal, 'params'):
                    signal.params.update(signal_params)
                
                # Create SignalData object
                signal_data = SignalData(signal_name=signal_name, asset_id=asset_id)
                
                # Calculate signal values if requested
                if calculate_values:
                    try:
                        # Calculate signal values
                        values = signal.calculate(price_data, asset_id)
                        
                        # Special handling for SSR signals with numeric index
                        # (optimization: detect if the signal is an SSR signal and has consistent values)
                        if signal_name.startswith('SSR_') and isinstance(values, pd.Series):
                            unique_values = values.unique()
                            if len(unique_values) == 1:
                                self.logger.info(f"Special handling for SSR signal {signal_name} with numeric index")
                                self.logger.info(f"{signal_name} has consistent values of {unique_values[0]}")
                                
                                # No need to set anything, values are already correct
                        
                        signal_data.set_values(values)
                        self.logger.info(f"Set values for {signal_name} with {values.sum() if isinstance(values, pd.Series) else 'unknown'} active signals")
                    except Exception as e:
                        self.logger.error(f"Error calculating values for {signal_name} on {asset_id}: {e}")
                        continue
                        
                # Register signal with asset
                asset_data.add_signal(signal_name, signal_data)
                registered_signals[signal_name] = signal_data
                
            except Exception as e:
                self.logger.error(f"Error registering signal {signal_name} for {asset_id}: {e}")
                continue
            
        self.logger.info(f"Registered {len(registered_signals)} signals for {asset_id}")
        
        return registered_signals
    
    def register_signals_parallel(
        self,
        asset_ids: List[str],
        signal_names: Optional[List[str]] = None,
        calculate_values: bool = True
    ) -> Dict[str, Dict[str, SignalData]]:
        """Register signals for multiple assets in parallel.
        
        Args:
            asset_ids: List of asset IDs to register signals for
            signal_names: List of signal names to register. If None, register all available signals.
            calculate_values: Whether to calculate signal values
            
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
                return asset_id, self.register_signals(asset_id, signal_names, calculate_values)
            except Exception as e:
                self.logger.error(f"Error registering signals for {asset_id}: {e}")
                return asset_id, {}
        
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
        
    def evaluate_signals(
        self,
        asset_ids: Optional[List[str]] = None,
        signal_names: Optional[List[str]] = None,
        force_recalculate: bool = False,
        use_stored: bool = False,
        save_to_storage: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate signals for effectiveness and calculate weights.
        
        Args:
            asset_ids: List of asset IDs to evaluate signals for (defaults to all loaded assets)
            signal_names: List of signal names to evaluate (defaults to all registered signals)
            force_recalculate: Whether to force recalculation of signal evaluations
                              even if they are already stored
            use_stored: Whether to use stored evaluations if available (overrides self.use_stored_evaluations)
                       If False, signals will always be freshly evaluated
            save_to_storage: Whether to save evaluations to storage after calculation
            
        Returns:
            dict: Dictionary mapping asset IDs to dictionaries of signal evaluations
        """
        self.logger.info("Evaluating signals for effectiveness...")
        
        # Determine whether to use stored evaluations
        use_stored_evaluations = use_stored or (self.use_stored_evaluations and not force_recalculate)
        
        # Default to all loaded assets if not specified
        if asset_ids is None:
            asset_ids = list(self.assets.keys())
        
        if not asset_ids:
            self.logger.warning("No assets to evaluate signals for")
            return {}
        
        # Initialize results
        evaluations = {}
        
        # Process each asset
        for asset_id in asset_ids:
            if asset_id not in self.assets:
                self.logger.warning(f"Asset {asset_id} not loaded, skipping evaluation")
                continue
            
            asset_data = self.assets[asset_id]
            asset_signals = asset_data.get_signals()
            
            # Filter signals if requested
            if signal_names:
                asset_signals = {
                    name: signal for name, signal in asset_signals.items() 
                    if name in signal_names
                }
            
            if not asset_signals:
                self.logger.warning(f"No signals to evaluate for {asset_id}")
                continue
            
            # Check if we can use stored evaluations
            asset_evaluations = {}
            all_found = True
            
            if use_stored_evaluations:
                for signal_name in asset_signals.keys():
                    if self.signal_storage.has_signal_evaluation(asset_id, signal_name):
                        # Use stored evaluation
                        stored_eval = self.signal_storage.get_signal_evaluation(asset_id, signal_name)
                        asset_evaluations[signal_name] = stored_eval
                        self.logger.debug(f"Using stored evaluation for {signal_name} on {asset_id}")
                    else:
                        # Need to calculate this one
                        all_found = False
                
                if all_found:
                    self.logger.info(f"Using stored evaluations for all signals on {asset_id}")
                    evaluations[asset_id] = asset_evaluations
                    continue
            
            # If we need to calculate any or force_recalculate is True
            self.logger.info(f"Evaluating {len(asset_signals)} signals for {asset_id}...")
            
            # Get price data
            if not asset_data.has_data():
                self.logger.warning(f"No data available for {asset_id}, skipping evaluation")
                continue
            
            price_data = asset_data.get_data()
            
            # Extract signal names and get signal instances from registry
            signal_names_list = list(asset_signals.keys())
            signals = []
            
            # Import get_signal here to avoid circular imports
            from ..signals.signal_registry import get_signal
            
            for signal_name in signal_names_list:
                try:
                    # Get signal instance from registry
                    signal_instance = get_signal(signal_name)
                    if signal_instance is None:
                        self.logger.warning(f"Signal {signal_name} not found in registry")
                        continue
                    
                    # Set asset_id in signal parameters
                    if hasattr(signal_instance, 'params'):
                        signal_instance.params['asset_id'] = asset_id
                    
                    signals.append(signal_instance)
                    self.logger.info(f"Evaluating signal {signal_name} for {asset_id}")
                except Exception as e:
                    self.logger.error(f"Error evaluating signal {signal_name} for {asset_id}: {e}")
            
            if not signals:
                self.logger.warning(f"No valid signal instances to evaluate for {asset_id}")
                continue
            
            # Evaluate all signals
            signal_results = self.signal_evaluator.evaluate_multiple_signals(
                signals=signals,
                data=price_data,
                asset_id=asset_id
            )
            
            # Update the SignalData objects with evaluation results
            for signal_instance, evaluation in zip(signals, signal_results.values()):
                signal_name = signal_instance.name
                
                # Skip if signal name not in asset_signals
                if signal_name not in asset_signals:
                    continue
                
                # Get signal data object
                signal_data = asset_signals[signal_name]
                
                # Update signal data with evaluation results
                signal_data.is_effective = evaluation.get('overall_effectiveness', False)
                signal_data.weight = evaluation.get('weight', 0.0)
                signal_data.optimal_decay = evaluation.get('optimal_decay', 7)
                
                self.logger.info(
                    f"Updated {signal_name} for {asset_id}: "
                    f"effective={signal_data.is_effective}, "
                    f"weight={signal_data.weight:.4f}, "
                    f"optimal_period={signal_data.optimal_decay}"
                )
            
            # Store results
            evaluations[asset_id] = signal_results
            
            # Store in signal storage for future use if requested
            if save_to_storage:
                for signal_name, evaluation in signal_results.items():
                    # Don't store invalid evaluations
                    if evaluation.get('valid', False):
                        self.signal_storage.add_signal_evaluation(
                            asset_id=asset_id,
                            signal_name=signal_name,
                            evaluation=evaluation
                        )
        
        # Save to disk if requested
        if save_to_storage:
            self.signal_storage.save()
            self.logger.info("Saved signal evaluations to storage")
            
        self.logger.info(f"Evaluated signals for {len(evaluations)} assets")
        return evaluations
    
    def get_effective_signals(
        self, 
        asset_id: str,
        use_stored: bool = False,
        save_to_storage: bool = False
    ) -> Dict[str, SignalData]:
        """Get effective signals for an asset based on evaluation results.
        
        Args:
            asset_id: ID of the asset
            use_stored: Whether to use stored evaluations if available (overrides self.use_stored_evaluations)
                       If False, signals will always be freshly evaluated
            save_to_storage: Whether to save evaluations to storage after calculation
            
        Returns:
            dict: Dictionary mapping signal names to SignalData objects for effective signals
        """
        if asset_id not in self.assets:
            self.logger.warning(f"Asset {asset_id} not loaded")
            return {}
        
        asset_data = self.assets[asset_id]
        signals = asset_data.get_signals()
        
        effective_signals = {}
        
        # Determine whether to use stored evaluations
        use_stored_evaluations = use_stored or self.use_stored_evaluations
        
        # First check if we have stored evaluations
        if use_stored_evaluations and self.signal_storage.has_asset(asset_id):
            # Use stored evaluations to determine effective signals
            stored_effective = self.signal_storage.get_asset_effective_signals(asset_id)
            
            for signal_name, evaluation in stored_effective.items():
                if (signal_name in signals 
                    and evaluation.get('optimal_decay', 0) > 0):
                    signal = signals[signal_name]
                    # Add evaluation data to the signal
                    signal.weight = evaluation.get('weight', 0.0)
                    signal.optimal_decay = evaluation.get('optimal_decay', 0)
                    effective_signals[signal_name] = signal
            
            self.logger.debug(
                f"Found {len(effective_signals)} effective signals for {asset_id} "
                f"from stored evaluations"
            )
            
            if effective_signals:
                return effective_signals
        
        # If we don't have stored evaluations or found none, evaluate now
        self.logger.debug(
            f"No effective signals found in storage for {asset_id}, evaluating now..."
        )
        
        # Evaluate all signals
        evaluations = self.evaluate_signals(asset_ids=[asset_id], use_stored=use_stored, save_to_storage=save_to_storage)
        
        if not evaluations or asset_id not in evaluations:
            self.logger.warning(f"No evaluations found for {asset_id}")
            return {}
        
        asset_evaluations = evaluations[asset_id]
        
        # Filter for effective signals
        for signal_name, evaluation in asset_evaluations.items():
            if (signal_name in signals 
                and evaluation.get('overall_effectiveness', False)
                and evaluation.get('optimal_decay', 0) > 0):
                signal = signals[signal_name]
                # Add evaluation data to the signal
                signal.weight = evaluation.get('weight', 0.0)
                signal.optimal_decay = evaluation.get('optimal_decay', 0)
                effective_signals[signal_name] = signal
        
        self.logger.debug(
            f"Found {len(effective_signals)} effective signals for {asset_id} "
            f"from fresh evaluation"
        )
        
        return effective_signals
        
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
                    values = signal_instance.calculate(filtered_price_data, asset_id)
                    signal_data.set_values(values)
                except Exception as e:
                    self.logger.error(f"Error calculating values for {signal_name} on {asset_id}: {e}")
                    continue
                    
        # Get signal values
        return asset_data.get_signal_values(start_date, end_date)
        
    def track_signal_events(
        self,
        asset_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_effective_signals_only: bool = True,
        reset_tracker: bool = True
    ) -> Dict[str, List[datetime]]:
        """Track signal activation events for a specific asset.
        
        Args:
            asset_id: ID of the asset
            start_date: Start date for tracking
            end_date: End date for tracking
            use_effective_signals_only: Whether to only track effective signals
            reset_tracker: Whether to reset the signal event tracker before tracking
            
        Returns:
            dict: Dictionary mapping signal names to lists of activation dates
        """
        asset_data = self.get_asset_data(asset_id)
        if asset_data is None:
            self.logger.warning(f"Asset data not found for {asset_id}")
            return {}
            
        # Reset tracker if requested
        if reset_tracker:
            self.signal_event_tracker.clear_active_events(asset_id)
            
        # Get price data
        price_data = asset_data.price_data
        if price_data.empty:
            self.logger.warning(f"Price data is empty for {asset_id}")
            return {}
            
        # Apply date filtering
        if start_date is not None or end_date is not None:
            filtered_price_data = price_data.copy()
            if start_date is not None:
                filtered_price_data = filtered_price_data[filtered_price_data.index >= start_date]
            if end_date is not None:
                filtered_price_data = filtered_price_data[filtered_price_data.index <= end_date]
                
            dates = filtered_price_data.index
        else:
            dates = price_data.index
            
        # Get signals to track
        if use_effective_signals_only:
            signals_dict = asset_data.get_effective_signals()
            self.logger.info(f"Tracking {len(signals_dict)} effective signals for {asset_id}")
        else:
            signals_dict = {name: asset_data.get_signal(name) for name in asset_data.get_signal_names()}
            self.logger.info(f"Tracking all {len(signals_dict)} signals for {asset_id}")
            
        if not signals_dict:
            self.logger.warning(f"No signals to track for {asset_id}")
            return {}
            
        # Track signals
        activation_dates = {}
        
        for signal_name, signal_data in signals_dict.items():
            # Skip signals without values
            if signal_data.values is None or len(signal_data.values) == 0:
                self.logger.warning(f"No values for signal {signal_name} on {asset_id}")
                continue
                
            # Extract activation dates
            dates_list = signal_data.get_activations_in_range(start_date, end_date)
            activation_dates[signal_name] = dates_list
            
            # Register each activation event
            for date in dates_list:
                # Ensure date is a proper datetime object
                if not isinstance(date, datetime):
                    date = pd.to_datetime(date)
                
                # Check meta-labeling approval if enabled
                meta_approved = True
                if self.use_meta_labeling:
                    # Get meta-labeling approval
                    if signal_data.use_meta_labeling:
                        meta_approved = signal_data.predict_meta_label(date, price_data)
                        
                        if not meta_approved:
                            self.logger.debug(
                                f"Meta-labeling rejected {signal_name} activation on "
                                f"{date.strftime('%Y-%m-%d')} for {asset_id}"
                            )
                
                # Register event with tracker
                self.signal_event_tracker.register_signal_event(
                    signal_name=signal_name,
                    asset_id=asset_id,
                    activation_date=date,
                    decay=signal_data.optimal_decay,
                    weight=signal_data.weight,
                    meta_approved=meta_approved
                )
                
        self.logger.info(
            f"Tracked events for {len(activation_dates)} signals on {asset_id}, "
            f"found {sum(len(dates) for dates in activation_dates.values())} activations"
        )
                
        return activation_dates
        
    def get_active_signals(
        self,
        asset_id: str,
        current_date: datetime,
    ) -> Dict[str, Any]:
        """Get active signal events for a specific asset on a given date.
        
        Args:
            asset_id: ID of the asset
            current_date: Current date to check against
            
        Returns:
            dict: Dictionary with active signal information
        """
        # Get active signals from tracker
        active_signals = self.signal_event_tracker.get_active_signals(asset_id, current_date)
        
        if not active_signals:
            return {
                'active_signals': {},
                'count': 0,
                'combined_weight': 0.0,
                'decision': 0
            }
            
        # Calculate combined weight
        combined_weight = self.signal_event_tracker.calculate_combined_weight(asset_id, current_date)
        
        # Get final decision
        decision = self.signal_event_tracker.get_final_decision(asset_id, current_date)
        
        return {
            'active_signals': active_signals,
            'count': len(active_signals),
            'combined_weight': combined_weight,
            'decision': decision
        }
        
    def get_current_recommendations(
        self,
        asset_ids: Optional[List[str]] = None,
        current_date: Optional[datetime] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get current recommendations for specified assets.
        
        Args:
            asset_ids: List of asset IDs to get recommendations for. If None, use all loaded assets.
            current_date: Date to use for recommendations. If None, use the latest date in price data.
            
        Returns:
            dict: Dictionary mapping asset IDs to recommendation information
        """
        self.logger.info("Getting current recommendations...")
        
        # If no asset IDs specified, use all loaded assets
        if asset_ids is None:
            asset_ids = self.get_asset_ids()
            
        if not asset_ids:
            self.logger.warning("No assets to check")
            return {}
            
        # Initialize results
        recommendations = {}
        
        # Process each asset
        for asset_id in asset_ids:
            asset_data = self.get_asset_data(asset_id)
            if asset_data is None:
                self.logger.warning(f"Asset data not found for {asset_id}")
                continue
                
            # Get price data
            price_data = asset_data.price_data
            if price_data.empty:
                self.logger.warning(f"Price data is empty for {asset_id}")
                continue
                
            # Determine current date if not specified
            if current_date is None:
                # Use the latest date in price data
                if isinstance(price_data.index, pd.DatetimeIndex):
                    asset_current_date = price_data.index[-1]
                elif 'date' in price_data.columns:
                    asset_current_date = price_data['date'].max()
                else:
                    self.logger.warning(f"Cannot determine current date for {asset_id}")
                    continue
            else:
                asset_current_date = current_date
                
            # Get active signals
            active_info = self.get_active_signals(asset_id, asset_current_date)
            
            # Get current price
            try:
                current_row = price_data.loc[asset_current_date]
                current_price = current_row['close']
            except (KeyError, TypeError):
                self.logger.warning(f"Price data not available for {asset_id} on {asset_current_date}")
                current_price = None
                
            # Create recommendation
            recommendation = {
                'asset_id': asset_id,
                'date': asset_current_date,
                'price': current_price,
                'active_signals_count': active_info['count'],
                'combined_weight': active_info['combined_weight'],
                'decision': active_info['decision'],
                'active_signals': active_info['active_signals']
            }
            
            recommendations[asset_id] = recommendation
            
        self.logger.info(f"Generated recommendations for {len(recommendations)} assets")
        
        return recommendations
        
    def run_backtest(
        self,
        backtest_name: str,
        assets: Optional[List[str]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        initial_capital: float = 10000.0,
        trade_cost: float = 0.001,
        slippage_pct: float = 0.0,
        mode: str = 'event_driven'
    ) -> Dict[str, Any]:
        """Run a backtest for specified assets.
        
        Args:
            backtest_name: Name of the backtest
            assets: List of asset IDs to include in the backtest. If None, use all loaded assets.
            start_date: Start date for the backtest
            end_date: End date for the backtest
            initial_capital: Initial capital for the backtest
            trade_cost: Trading cost as a percentage
            slippage_pct: Slippage percentage
            mode: Backtest mode ('event_driven' or 'binary')
            
        Returns:
            dict: Dictionary with backtest results
        """
        self.logger.info(f"Running {mode} backtest '{backtest_name}'...")
        
        # Convert dates to datetime objects if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # If no assets specified, use all loaded assets
        if assets is None:
            assets = self.get_asset_ids()
            
        if not assets:
            self.logger.warning("No assets to backtest")
            return {}
            
        # Prepare backtest parameters
        backtest_params = {
            'name': backtest_name,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'mode': mode,
            'trade_cost': trade_cost,
            'slippage_pct': slippage_pct,
            'signal_threshold': self.signal_threshold
        }
        
        # Track signal events for each asset
        for asset_id in tqdm(assets, desc="Tracking signal events"):
            # Skip if we don't have any effective signals
            effective_signals = self.get_effective_signals(asset_id)
            if not effective_signals:
                self.logger.warning(f"No effective signals for {asset_id}, skipping")
                continue
                
            # Track signal events
            self.track_signal_events(
                asset_id,
                start_date=start_date,
                end_date=end_date,
                use_effective_signals_only=True,
                reset_tracker=True
            )
            
        # Run backtest
        if mode == 'event_driven':
            # Create EventDrivenBacktester
            backtester = EventDrivenBacktester(analyzer=self)
            
            # Run backtest
            results = backtester.backtest_assets(
                asset_ids=assets,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                trade_cost=trade_cost,
                slippage_pct=slippage_pct,
                signal_threshold=self.signal_threshold
            )
        else:
            # Default to binary mode using PortfolioManager
            # This is a placeholder and should be replaced with actual implementation
            self.logger.warning("Binary backtest mode not fully implemented yet, using event_driven")
            
            # Create EventDrivenBacktester
            backtester = EventDrivenBacktester(analyzer=self)
            
            # Run backtest
            results = backtester.backtest_assets(
                asset_ids=assets,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                trade_cost=trade_cost,
                slippage_pct=slippage_pct,
                signal_threshold=self.signal_threshold
            )
            
        # Store backtest results
        for asset_id in assets:
            asset_data = self.get_asset_data(asset_id)
            if asset_data is None:
                continue
                
            # Get asset results
            asset_results = results.get('asset_results', {}).get(asset_id)
            if asset_results is not None:
                asset_data.add_backtest_result(backtest_name, asset_results)
            
        self.logger.info(f"Completed backtest '{backtest_name}'")
        
        return results
        
    def train_meta_labels(
        self,
        asset_id: str,
        feature_columns: List[str],
        min_return_threshold: float = 0.0,
        signals: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, Tuple[float, int]]:
        """Train meta-labeling models for signals on a specific asset.
        
        Args:
            asset_id: ID of the asset
            feature_columns: List of column names to use as features
            min_return_threshold: Minimum return to consider a trade successful
            signals: List of signal names to train meta-labels for. If None, train for all effective signals.
            verbose: Whether to print detailed information
            
        Returns:
            dict: Dictionary mapping signal names to tuples of (accuracy, sample_count)
        """
        self.logger.info(f"Training meta-labels for {asset_id}...")
        
        asset_data = self.get_asset_data(asset_id)
        if asset_data is None:
            self.logger.warning(f"Asset data not found for {asset_id}")
            return {}
            
        # Get price data
        price_data = asset_data.price_data
        if price_data.empty:
            self.logger.warning(f"Price data is empty for {asset_id}")
            return {}
            
        # Verify feature columns exist in price data
        missing_columns = [col for col in feature_columns if col not in price_data.columns]
        if missing_columns:
            self.logger.warning(
                f"Missing feature columns for {asset_id}: {missing_columns}"
            )
            return {}
            
        # Get signals to train
        if signals is None:
            # Use all effective signals
            signals_dict = asset_data.get_effective_signals()
            self.logger.info(f"Training meta-labels for {len(signals_dict)} effective signals on {asset_id}")
        else:
            # Use specified signals
            signals_dict = {}
            for name in signals:
                signal_data = asset_data.get_signal(name)
                if signal_data is not None and signal_data.is_effective:
                    signals_dict[name] = signal_data
            self.logger.info(f"Training meta-labels for {len(signals_dict)} specified signals on {asset_id}")
            
        if not signals_dict:
            self.logger.warning(f"No effective signals to train meta-labels for on {asset_id}")
            return {}
            
        # Train meta-labels
        results = {}
        
        for signal_name, signal_data in signals_dict.items():
            try:
                accuracy, sample_count = signal_data.train_meta_model(
                    price_data=price_data,
                    feature_columns=feature_columns,
                    min_return_threshold=min_return_threshold,
                    verbose=verbose
                )
                
                results[signal_name] = (accuracy, sample_count)
                
                if sample_count > 0:
                    self.logger.info(
                        f"Trained meta-model for {signal_name} on {asset_id}: "
                        f"accuracy={accuracy:.4f}, samples={sample_count}"
                    )
                else:
                    self.logger.warning(
                        f"Insufficient data to train meta-model for {signal_name} on {asset_id}"
                    )
                    
            except Exception as e:
                self.logger.error(
                    f"Error training meta-model for {signal_name} on {asset_id}: {e}"
                )
                continue
                
        self.logger.info(f"Trained meta-models for {len(results)} signals on {asset_id}")
        
        return results
        
    def save_state(self, path: str) -> bool:
        """Save the current state to disk.
        
        Args:
            path: Path to save the state
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Saving state to {path}...")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Extract state to save
            state = {
                'asset_ids': self.get_asset_ids(),
                'signal_threshold': self.signal_threshold,
                'use_meta_labeling': self.use_meta_labeling,
                'lookback_days': self.lookback_days,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save minimal state data (pointer to assets rather than actual data)
            with open(path, 'w') as f:
                json.dump(state, f, indent=4)
                
            self.logger.info(f"Saved state to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return False
            
    def load_state(self, path: str) -> bool:
        """Load state from disk.
        
        Args:
            path: Path to load the state from
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Loading state from {path}...")
        
        try:
            # Load state
            with open(path, 'r') as f:
                state = json.load(f)
                
            # Extract state data
            asset_ids = state.get('asset_ids', [])
            signal_threshold = state.get('signal_threshold', 0.0)
            use_meta_labeling = state.get('use_meta_labeling', False)
            lookback_days = state.get('lookback_days', 365)
            
            # Update state
            self.signal_threshold = signal_threshold
            self.use_meta_labeling = use_meta_labeling
            self.lookback_days = lookback_days
            
            # Update components
            self.signal_evaluator = SignalEvaluator(lookback_days=lookback_days)
            self.signal_combiner = SignalCombiner(
                signal_evaluator=self.signal_evaluator,
                threshold=signal_threshold,
                auto_weights=True
            )
            self.signal_event_tracker = SignalEventTracker(threshold=signal_threshold)
            
            # Load assets if not already loaded
            if asset_ids:
                self.load_data(assets=asset_ids)
                
            self.logger.info(
                f"Loaded state from {path} (saved at {state.get('timestamp', 'unknown')})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False
        
    def clear_cache(self):
        """Clear all caches."""
        self.data_cache = {}
        for asset_id, asset_data in self.assets.items():
            asset_data.clear_cache()
        self.signal_evaluator.clear_cache()
        self.signal_event_tracker.clear_active_events()
        self.signal_event_tracker.clear_history()
        
        self.logger.info("Cleared all caches")
        
    def get_available_signals(self) -> List[str]:
        """Get names of all available signals.
        
        Returns:
            list: List of signal names
        """
        return get_all_signals()
        
    def run_analysis(self) -> Dict[str, float]:
        """Run the full analysis pipeline on all assets.
        
        This method:
        1. Evaluates all registered signals
        2. Tracks signal events
        3. Runs backtests
        4. Updates portfolio weights
        
        Returns:
            dict: Dictionary mapping asset IDs to portfolio weights
        """
        self.logger.info("Running full portfolio analysis...")
        
        # Step 1: Evaluate signals
        asset_ids = list(self.assets.keys())
        
        if not asset_ids:
            self.logger.warning("No assets loaded. Use load_data() first.")
            return {}
            
        self.logger.info(f"Evaluating {len(asset_ids)} assets...")
        
        # Evaluate signals for each asset
        for asset_id in tqdm(asset_ids, desc="Evaluating assets"):
            asset_data = self.assets[asset_id]
            
            if asset_data.price_data.empty:
                self.logger.warning(f"Price data is empty for {asset_id}")
                continue
                
            # Evaluate all registered signals
            self.evaluate_signals(asset_id)
            
        # Step 2: Track signal events
        self.logger.info("Tracking signal events...")
        
        for asset_id in tqdm(asset_ids, desc="Tracking signal events"):
            asset_data = self.assets[asset_id]
            
            # Skip if no signals registered
            if not hasattr(asset_data, 'signals') or not asset_data.signals:
                self.logger.warning(f"No signals registered for {asset_id}")
                continue
                
            # Get signal data
            signal_data_dict = {}
            for signal_name, signal_data in asset_data.signals.items():
                if signal_data.values is not None and not signal_data.values.empty:
                    signal_data_dict[signal_name] = signal_data.values
                    
            if not signal_data_dict:
                self.logger.warning(f"No effective signals for {asset_id}, skipping")
                continue
                
            # Track signal events
            self.signal_event_tracker.track_events(
                asset_id=asset_id,
                signal_data=signal_data_dict,
                price_data=asset_data.price_data
            )
            
        # Step 3: Run backtests
        self.logger.info("Running backtests...")
        
        # Initialize backtester
        backtest_results = {}
        
        for asset_id in tqdm(asset_ids, desc="Backtesting assets"):
            asset_data = self.assets[asset_id]
            
            # Skip if price data is empty
            if asset_data.price_data.empty:
                self.logger.warning(f"Price data is empty for {asset_id}, skipping asset")
                continue
                
            # Skip if no signal events
            if not hasattr(asset_data, 'signal_events'):
                self.logger.warning(f"No signal events for {asset_id}, skipping asset")
                continue
                
            # Get signal events
            signal_events = self.signal_event_tracker.get_events(asset_id)
            
            if not signal_events or signal_events.empty:
                self.logger.warning(f"No signal events found for {asset_id}, skipping asset")
                continue
                
            # Run backtest
            try:
                backtest = EventDrivenBacktester(
                    signal_events=signal_events,
                    price_data=asset_data.price_data,
                    threshold=self.signal_threshold
                )
                
                result = backtest.run()
                backtest_results[asset_id] = result
                
                # Store result in asset data
                asset_data.backtest_result = result
                
                self.logger.info(
                    f"Backtest for {asset_id}: "
                    f"Return: {result.get('total_return', 0):.2%}, "
                    f"Sharpe: {result.get('sharpe_ratio', 0):.2f}"
                )
            except Exception as e:
                self.logger.error(f"Error in backtest for {asset_id}: {str(e)}")
                
        if not backtest_results:
            self.logger.warning("No assets could be processed successfully in backtest")
            
        # Step 4: Update portfolio weights
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
            
    def get_portfolio_recommendations(self) -> Dict[str, float]:
        """Get portfolio recommendations based on analysis.
        
        Returns:
            dict: Dictionary mapping asset IDs to portfolio weights
        """
        if hasattr(self, 'portfolio_weights'):
            return self.portfolio_weights
        else:
            # Run analysis if not done yet
            return self.run_analysis()
    
    def plot_signals(self, asset_id: str, **kwargs):
        """Plot signals for a specific asset.
        
        Args:
            asset_id: ID of the asset
            **kwargs: Additional arguments for plotting
        """
        asset_data = self.get_asset_data(asset_id)
        
        if asset_data is None:
            self.logger.warning(f"Asset {asset_id} not found")
            return
        
        if not hasattr(asset_data, 'signals') or not asset_data.signals:
            self.logger.warning(f"No signals registered for {asset_id}")
            return
            
        # TODO: Implement plotting
        self.logger.info(f"Plotting signals for {asset_id}")
        # (Placeholder for actual plotting implementation)

    # Add new methods for signal persistence
    
    def load_signal_evaluations(self) -> bool:
        """Explicitly load signal evaluations from storage.
        
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        loaded = self.signal_storage.load()
        if loaded:
            self.logger.info("Successfully loaded signal evaluations from storage")
        else:
            self.logger.warning("Failed to load signal evaluations from storage")
        return loaded
    
    def save_signal_evaluations(self, backup: bool = True) -> bool:
        """Save current signal evaluations to storage.
        
        Args:
            backup: Whether to create a backup of existing evaluations
            
        Returns:
            bool: True if successful, False otherwise
        """
        if backup:
            backup_path = self.signal_storage.create_backup()
            if backup_path:
                self.logger.info(f"Created backup of signal evaluations at {backup_path}")
        
        saved = self.signal_storage.save()
        if saved:
            self.logger.info("Saved signal evaluations to storage")
        else:
            self.logger.error("Failed to save signal evaluations")
        
        return saved
    
    def force_reevaluate_signals(
        self, 
        asset_ids: Optional[List[str]] = None,
        backup: bool = True,
        save_to_storage: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Force re-evaluation of all signals for specified assets.
        
        This will clear any stored evaluations for the assets and recalculate.
        
        Args:
            asset_ids: List of asset IDs to re-evaluate (defaults to all loaded assets)
            backup: Whether to create a backup before re-evaluation
            save_to_storage: Whether to save evaluations to storage after calculation
            
        Returns:
            dict: Dictionary mapping asset IDs to dictionaries of signal evaluations
        """
        self.logger.info("Forcing re-evaluation of signals...")
        
        # Default to all loaded assets if not specified
        if asset_ids is None:
            asset_ids = list(self.assets.keys())
        
        if not asset_ids:
            self.logger.warning("No assets to re-evaluate signals for")
            return {}
        
        # Create a backup if requested
        if backup and save_to_storage and self.use_stored_evaluations:
            backup_path = self.signal_storage.create_backup("before_reevaluation")
            self.logger.info(f"Created backup of signal evaluations at {backup_path}")
        
        # Remove existing evaluations for the assets
        for asset_id in asset_ids:
            if self.signal_storage.has_asset(asset_id):
                self.signal_storage.remove_asset(asset_id)
                self.logger.info(f"Removed stored evaluations for {asset_id}")
        
        # Evaluate signals with force_recalculate=True
        evaluations = self.evaluate_signals(
            asset_ids=asset_ids,
            force_recalculate=True,
            save_to_storage=save_to_storage
        )
        
        return evaluations
    
    def get_signal_storage_info(self) -> Dict[str, Any]:
        """Get information about the signal storage.
        
        Returns:
            dict: Information about the signal storage
        """
        return self.signal_storage.get_storage_info() 