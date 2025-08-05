# SOROS System Changelog

## [1.1.9] - 2025-04-01

### Fixed
- Fixed portfolio backtesting date range filtering issue
- Enhanced column name handling to support both "Short Term" and "ShortTerm" formats in trend conditions
- Lowered default signal threshold from 75 to 50 to increase trading activity
- Added robust trend column detection in _get_btc_trend method
- Added detailed logging for signal calculation and trade decisions
- Fixed bug in BTC trend calculation where trend columns weren't being found

## [1.1.8] - 2025-04-01

### Fixed
- Resolved "All arrays must be of the same length" error in backtest_portfolio when creating results DataFrame
- Fixed inconsistent array length issue by properly initializing and appending to the dates and btc_prices arrays
- Added array length validation before creating the results DataFrame
- Enhanced debug logging to track array lengths during backtest execution

## [1.1.7] - 2025-04-01

### Fixed
- Resolved "Cannot set a DataFrame with multiple columns to the single column bitcoin" error
- Fixed file path construction in DataLoader to use os.path.join for better platform compatibility
- Added proper None value checks throughout the backtest code to prevent "NoneType has no attribute empty" errors
- Fixed asset data variable reuse in portfolio backtester that was causing data corruption
- Enhanced logging in DataLoader to provide better diagnostics for data loading issues

## [1.1.6] - 2025-04-01

### Fixed
- Resolved IndexError in portfolio backtest when calculating returns
- Enhanced asset data loading to properly handle missing data and empty DataFrames
- Added validation of loaded data before running backtest to prevent runtime errors
- Improved error handling in _get_asset_data to ensure valid data is available
- Fixed trade tracking and properly initialized cash and holdings for backtests
- Added robust date range filtering for assets to ensure consistency with BTC data

## [1.1.5] - 2025-03-31

### Fixed
- Fixed term-specific transition matrices showing default values (0.2 probability for all transitions)
- Updated transition matrix calculation to properly use the specific trend terms (Short Term, Medium Term, Long Term)
- Added temporary column renaming to ensure proper transition probability calculation for all trend terms

## [1.1.4] - 2025-03-31

### Fixed
- Added robust handling for assets without complete trend classification
- Improved error handling in trend metrics calculation to proceed even with missing trend data
- Fixed issues with missing "Overall_Trend_USD" and "Overall_Trend_BTC" columns by creating them when possible
- Enhanced analyze_asset method to provide detailed diagnostics about which trend terms are missing
- Added fallback mechanism to create empty placeholder DataFrames for missing trend metrics

## [1.1.3] - 2025-03-31

### Fixed
- Fixed transition matrices to be specific to trend terms (ShortTerm, MediumTerm, LongTerm, Overall)
- Corrected the parameter order in PortfolioBacktester initialization to fix portfolio backtesting
- Removed redundant get_asset_trend_metrics method, using direct trend_metrics access instead
- Enhanced plot_transition_matrix to handle different term naming formats and display meaningful titles
- Fixed the mapping between trend term column names and transition matrix storage keys

## [1.1.2] - 2025-03-27

### Added
- Enhanced `get_asset_information` method to include term-specific trend metrics (Short Term, Medium Term, Long Term)
- Added new method `calculate_trend_term_metrics` to MetricsCalculator for analyzing specific trend terms
- Added consolidated trend metrics with a 'Term' column for easier analysis and visualization

### Fixed
- Enhanced lookback_days parameter to support the 'all' option for using all available data
- Fixed ticker display in get_latest_trends() to show proper cryptocurrency symbols (BTC, ETH, etc.)
- Modified Markov analyzer to correctly handle string values for lookback_days
- Fixed RSI calculation to properly handle numpy arrays and pandas Series conversion

## [1.1.1] - 2025-03-27

### Fixed
- Fixed imports by correcting the module name from 'plotter' to 'plotters'
- Replaced directory-based `__init__.py` with proper files for all modules
- Enhanced import mechanism to support multiple import scenarios (direct or as a package)
- Improved path handling for both relative and absolute imports

## [1.1.0] - 2025-03-27

### Fixed
- Fixed the USD trend classification for altcoins, ensuring overall trends are properly calculated
- Improved robustness of BTC price column detection in metrics calculation
- Enhanced error handling in trend classification to handle different column patterns
- Fixed moving average calculation to better handle limited data
- Improved RSI calculation for altcoins with proper column detection
- Added better handling of NaN values in trend calculations

### Added
- Comprehensive logging for trend classification to aid in debugging
- Better asset ID tracking in processed data
- Compatibility with direct script execution through improved imports

### Improved
- Enhanced trend classification stability with better column validation
- Improved moving average calculation to prevent errors with insufficient data
- More robust filtering of MA/RoC columns to prevent errors
- Updated how overall trends are calculated with better NaN handling 