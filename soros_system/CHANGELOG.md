# SOROS System Changelog

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