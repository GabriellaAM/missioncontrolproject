import pandas as pd
import numpy as np
import ta
import plotly.express as px  # Added for interactive heatmap
from typing import Dict, List, Tuple, Set, Any, Optional, Union
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import glob
import os
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def generate_features(
    data: pd.DataFrame, 
    symbol: str = None,
    include_technical: bool = True,
    include_onchain: bool = True,
    include_macro: bool = True,
    onchain_data_path: str = None,
    macro_data_path: str = None,
    macro_features: List[str] = None,
    verbose: bool = True,
    show_plots: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive feature generation pipeline for technical, on-chain, and macro indicators.
    Args:
        data: Input DataFrame with OHLCV data
        symbol: Asset symbol/id (bitcoin, ethereum, etc.)
        include_technical: Whether to generate technical indicators
        include_onchain: Whether to include on-chain metrics
        include_macro: Whether to include macro features
        onchain_data_path: Path to on-chain data files
        macro_data_path: Path to macro data files
        macro_features: List of specific macro features to load
        verbose: Whether to print analysis progress
        show_plots: Whether to display visualizations
        
    Returns:
        Dictionary containing processed data and analysis results
    """
    if verbose:
        logger.info("Starting feature generation pipeline...")
    
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain all required columns: {required_columns}")

    # Ensure date indices are datetime
    df.index = pd.to_datetime(df.index)

    # 1. Technical Indicators
    if include_technical:
        if verbose:
            logger.info("Generating technical indicators...")
        df = generate_technical_indicators(df)
    
    # 2. On-chain Metrics
    onchain_columns = []

    if include_onchain and symbol:
        if verbose:
            logger.info("Loading and processing on-chain metrics...")
        onchain_df = load_onchain_data(symbol, onchain_data_path, verbose=verbose)
        if not onchain_df.empty:
            # Ensure datetime indices
            onchain_df.index = pd.to_datetime(onchain_df.index)
            # Drop fully NaN columns explicitly
            onchain_df = onchain_df.dropna(axis=1, how='all')
            # Merge explicitly
            df = df.merge(onchain_df, left_index=True, right_index=True, how='left')
            onchain_columns = onchain_df.columns.tolist()
            # Generate derived features from on-chain metrics
            df = derive_onchain_features(df, onchain_columns)
    
    # 3. Macro Features
    macro_columns = []
    if include_macro:
        if verbose:
            logger.info("Loading and processing macro features...")
        macro_df = load_macro_data(macro_data_path, macro_features, verbose=verbose)
        if not macro_df.empty:
            # Ensure datetime indices
            macro_df.index = pd.to_datetime(macro_df.index)
            # Drop fully NaN columns explicitly
            macro_df = macro_df.dropna(axis=1, how='all')
            # Merge explicitly
            df = df.merge(macro_df, left_index=True, right_index=True, how='left')
            macro_columns = macro_df.columns.tolist()
            # Generate derived features from macro metrics
            df = derive_macro_features(df, macro_columns)
    
    # 4. Add log returns if not present
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['Close']).diff()
    
    # 5. Clean data
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(how='all')
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    # 6. Calculate correlation matrix for analysis
    # Instead of using hardcoded feature patterns, use all numeric columns
    # except for the original OHLCV columns which are used to derive features
    
    # Define columns to exclude from analysis
    exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Get all numeric columns for analysis
    numeric_columns = df_clean.select_dtypes(include=['number']).columns.tolist()
    features_to_analyze = [col for col in numeric_columns if col not in exclude_columns]
    
    # Ensure log_return is included
    if 'log_return' in df_clean.columns and 'log_return' not in features_to_analyze:
        features_to_analyze = ['log_return'] + features_to_analyze
    
    # Calculate correlation matrix
    correlation_matrix = df_clean[features_to_analyze].corr()
    
    if verbose:
        logger.info(f"Generated {len(features_to_analyze)} features for analysis")
        logger.info(f"Final dataframe shape: {df_clean.shape}")
        
        # Create visualizations if show_plots is True
        if show_plots:
            visualize_features(df_clean, features_to_analyze, correlation_matrix)
    
    # Check date ranges explicitly
   #print("Main data date range:", df.index.min(), "to", df.index.max())
    #print("On-chain data date range:", onchain_df.index.min(), "to", onchain_df.index.max())
    #print("Macro data date range:", macro_df.index.min(), "to", macro_df.index.max())
    
    # Check intersection of dates explicitly
    #print("Intersection with on-chain data:", len(df.index.intersection(onchain_df.index)))
    #print("Intersection with macro data:", len(df.index.intersection(macro_df.index)))
    
    # Check if on-chain data is empty or fully NaN
    #print("On-chain data empty:", onchain_df.empty)
    #print("On-chain data fully NaN columns:", onchain_df.columns[onchain_df.isna().all()].tolist())

    # Check if macro data is empty or fully NaN
    #print("Macro data empty:", macro_df.empty)
    #print("Macro data fully NaN columns:", macro_df.columns[macro_df.isna().all()].tolist())
    
    return {
        'processed_data': df_clean,
        'correlation_matrix': correlation_matrix,
        'features': features_to_analyze,
        'onchain_columns': onchain_columns,
        'macro_columns': macro_columns
    }

def generate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical indicators for price data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators added
    """
    # Ensure we're working with a copy
    df = df.copy()
    
    # Momentum Indicators
    # RSI with multiple windows
    for window in [7, 14]:
        df[f'rsi_{window}'] = ta.momentum.RSIIndicator(close=df['Close'], window=window).rsi()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Trend Indicators

    

    # Moving Averages
    for window in [7, 30, 90, 365]:
        df[f'sma_{window}'] = ta.trend.SMAIndicator(close=df['Close'], window=window).sma_indicator()
        df[f'ema_{window}'] = ta.trend.EMAIndicator(close=df['Close'], window=window).ema_indicator()

    # Moving Average Crosses
    # Generate MA cross signals using already calculated SMAs
    ma_pairs = [(7, 30), (30, 90), (7, 90)]
    for fast, slow in ma_pairs:
        fast_col = f'sma_{fast}'
        slow_col = f'sma_{slow}'
        
        # Create cross signals (1 for golden cross, -1 for death cross, 0 for no cross)
        df[f'ma_cross_{fast}_{slow}'] = np.where(
            df[fast_col] > df[slow_col], 1, 
            np.where(df[fast_col] < df[slow_col], -1, 0)
        )
        
        # Normalized distance between MAs using z-score
        distance = (df[fast_col] - df[slow_col]) / df[slow_col] * 100
        distance_mean = distance.rolling(window=90).mean()
        distance_std = distance.rolling(window=90).std()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        df[f'ma_distance_norm_{fast}_{slow}'] = (
            (distance - distance_mean) / (distance_std + epsilon)
        )
    
    # ADX
    df['adx'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
    
    # Volatility Indicators
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['Close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    df['distance_from_upper_band'] = df['bb_upper'] / df['Close'] - 1
    df['distance_from_lower_band'] = df['Close'] / df['bb_lower'] - 1

    df['distance_from_upper_band_zscore'] = (
        df['distance_from_upper_band'] - 
        df['distance_from_upper_band'].rolling(window=20).mean()
    ) / df['distance_from_upper_band'].rolling(window=20).std()

    df['distance_from_lower_band_zscore'] = (
        df['distance_from_lower_band'] - 
        df['distance_from_lower_band'].rolling(window=20).mean()
    ) / df['distance_from_lower_band'].rolling(window=20).std()

    df['bzs'] = df['distance_from_upper_band_zscore'] - df['distance_from_lower_band_zscore']
    df['bzsd'] = df['bzs'].pct_change()

    # Average True Range
    df['atr'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    
    # Volume Indicators
    # Volume Moving Average
    df['volume_ma_30'] = df['Volume'].rolling(window=30).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_30']
    
    # On-Balance Volume
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
    
    # Custom Price Features
    df['log_return'] = np.log(df['Close']).diff()
    df['realized_vol_7'] = df['log_return'].rolling(window=7).std() * np.sqrt(365)
    df['realized_vol_14'] = df['log_return'].rolling(window=14).std() * np.sqrt(365)
    df['realized_vol_30'] = df['log_return'].rolling(window=30).std() * np.sqrt(365)
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    
    # Price to MA ratios
    df['price_sma7_ratio'] = df['Close'] / df['sma_7']
    df['price_sma30_ratio'] = df['Close'] / df['sma_30']
    df['price_sma365_ratio'] = df['Close'] / df['sma_365']

    df['7day_low'] = df['Low'].rolling(window=7).min()
    df['14day_low'] = df['Low'].rolling(window=14).min()
    df['30day_low'] = df['Low'].rolling(window=30).min()

    df['7day_high'] = df['High'].rolling(window=7).max()
    df['14day_high'] = df['High'].rolling(window=14).max()
    df['30day_high'] = df['High'].rolling(window=30).max()


    df['support'] = (df['7day_low'] + df['14day_low'] + df['30day_low']) / 3
    df['resistance'] = (df['7day_high'] + df['14day_high'] + df['30day_high']) / 3

    df['support_relative'] =  df['support'] / df['Close']
    df['resistance_relative'] =  df['Close'] / df['resistance']



    # Calculate price equilibrium and normalize it using z-score
    df['price_equilibrium'] = (df['support_relative'] + df['resistance_relative'])
    # Use rolling window for z-score normalization to prevent lookahead bias
    window_size = 7  # Using 30-day rolling window
    df['price_equilibrium_zscore'] = (
        df['price_equilibrium'] - df['price_equilibrium'].rolling(window=window_size).mean()
    ) / df['price_equilibrium'].rolling(window=window_size).std()

    # Cyclical encoding of time features
    df['day_of_week_sin'] = np.sin(df.index.dayofweek * (2 * np.pi / 7))
    df['day_of_week_cos'] = np.cos(df.index.dayofweek * (2 * np.pi / 7))
    df['month_sin'] = np.sin((df.index.month-1) * (2 * np.pi / 12))
    df['month_cos'] = np.cos((df.index.month-1) * (2 * np.pi / 12))
    
    # Regime change indicators
    df['regime_rsi'] = np.where(df['rsi_14'] > 70, 1, np.where(df['rsi_14'] < 30, -1, 0))
    df['regime_bb'] = np.where(df['Close'] > df['bb_upper'], 1, 
                              np.where(df['Close'] < df['bb_lower'], -1, 0))

    # Price momentum regimes
    df['price_regime'] = np.where(
        df['Close'] > df['sma_30'] * 1.1, 2,  # Strong bull
        np.where(
            df['Close'] > df['sma_30'], 1,    # Mild bull
            np.where(
                df['Close'] < df['sma_30'] * 0.9, -2,  # Strong bear
                np.where(
                    df['Close'] < df['sma_30'], -1,    # Mild bear
                    0  # Neutral
                )
            )
        )
    )
    
    # Add interaction terms between important features
    # Example (would need to be added where both features are available):
    df['rsi_vol_interaction'] = df['rsi_14'] * df['realized_vol_14']
    df['price_volume_correlation'] = df['Close'].rolling(window=30).corr(df['Volume'])
    
    # Additional oscillators
    df['cci_20'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci()
    df['williams_r_14'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14).williams_r()

    # Ichimoku Cloud components
    ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
    
    # Volatility ratios
    df['vol_ratio_7_30'] = df['realized_vol_7'] / df['realized_vol_30']
    df['high_low_vol'] = (df['High'] - df['Low']).rolling(window=14).std() / df['Close']
    
    return df

def load_onchain_data(symbol: str, data_path: str = None, verbose: bool = False) -> pd.DataFrame:
    """
    Load on-chain metrics for a specific cryptocurrency with improved handling of sparse data.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'bitcoin', 'ethereum')
        data_path: Path to on-chain data files
        verbose: Whether to print progress
        
    Returns:
        DataFrame with on-chain metrics
    """
    if verbose:
        logger.info(f"Loading on-chain data for {symbol}...")
    
    if not data_path:
        data_path = "/Users/valter.rebelo/MissionControl/data/onchainData"
    
    ticker_map = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "solana": "SOL"
    }
    
    if symbol not in ticker_map:
        if verbose:
            logger.warning(f"No on-chain data available for: {symbol}")
        return pd.DataFrame()
        
    ticker = ticker_map[symbol]
    
    # Find all CSV files for the ticker
    files = glob.glob(f"{data_path}/{ticker}_*.csv")
    
    if not files:
        if verbose:
            logger.warning(f"No on-chain data files found for {symbol}")
        return pd.DataFrame()
    
    # Process each file
    all_dfs = []
    for file in files:
        try:
            # Extract metric name from filename
            metric_name = os.path.basename(file).replace(f"{ticker}_", "").replace(".csv", "")
            
            # Read the CSV file
            df = pd.read_csv(file)
            if 'date' not in df.columns:
                if verbose:
                    logger.warning(f"No date column in {file}, skipping")
                continue
                
            # Set date as index
            df.set_index(pd.to_datetime(df['date']), inplace=True)
            df.drop(columns=['date'], inplace=True)
            
            # Process each column separately
            for col in df.columns:
                # Skip empty columns
                if df[col].isna().all():
                    if verbose:
                        logger.warning(f"Column {col} in {file} is all NaN, skipping")
                    continue
                
                # Check if the column has sufficient non-NaN values
                non_nan_count = df[col].count()
                if non_nan_count < 30:
                    if verbose:
                        logger.warning(f"Column {col} in {file} has only {non_nan_count} non-NaN values, skipping")
                    continue
                
                # Check for sufficient data coverage (at least 1 year of data)
                data_span = df.index.max() - df.index.min()
                if data_span.days < 365:
                    if verbose:
                        logger.warning(f"Column {col} in {file} has insufficient data span ({data_span.days} days), skipping")
                    continue
                
                # Create a new dataframe for this column
                col_df = pd.DataFrame()
                
                # Use the column name directly without the file name prefix
                # This fixes the issue with column naming in the CSV files
                col_df[col] = df[col]
                
                # Only append if there are columns left
                if not col_df.empty:
                    all_dfs.append(col_df)
                
        except Exception as e:
            if verbose:
                logger.warning(f"Error loading {file}: {str(e)}")
    
    if not all_dfs:
        if verbose:
            logger.warning(f"No valid on-chain data loaded for {symbol}")
        return pd.DataFrame()
    
    # Combine all metrics
    combined_df = pd.concat(all_dfs, axis=1)
    
    if verbose:
        logger.info(f"Loaded {combined_df.shape[1]} on-chain metrics")
    
    return combined_df

def derive_onchain_features(df: pd.DataFrame, onchain_columns: List[str]) -> pd.DataFrame:
    """
    Generate derived features from on-chain metrics with robust handling of edge cases.
    
    Args:
        df: DataFrame with on-chain metrics
        onchain_columns: List of on-chain column names
        
    Returns:
        DataFrame with derived features added
    """
    # Ensure we're working with a copy
    df = df.copy()
    
    # Create a copy of onchain_columns to avoid modifying the original list
    processed_columns = []
    
    for col in onchain_columns:
        if col not in df.columns:
            continue
        
        # Check the type of the column
        if isinstance(df[col], pd.DataFrame):
            logging.warning(f"Column {col} is a DataFrame, not a Series. Skipping this column.")
            continue
            
        # Now we know it's a Series, proceed with processing
        processed_columns.append(col)
        
        # Check if column contains string values and convert to numeric if possible
        if pd.api.types.is_object_dtype(df[col]):
            try:
                # Try to convert strings to numeric values
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logging.warning(f"Failed to convert column {col} to numeric: {str(e)}")
                continue
        
        # Skip columns with too many NaN values
        valid_count = df[col].notna().sum()
        if valid_count <= 30:
            logging.warning(f"Column {col} has only {valid_count} non-NaN values, insufficient for rolling calculations")
            continue
            
        # Continue with the rest of your feature derivation code...
        # Add your feature derivation logic here
        
        # Example: Create rolling means and standard deviations
        df[f'{col}_ma7'] = df[col].rolling(window=7).mean()
        df[f'{col}_ma30'] = df[col].rolling(window=30).mean()
        df[f'{col}_std30'] = df[col].rolling(window=30).std()
        
        # Create momentum indicators
        df[f'{col}_mom7'] = df[col].diff(7)
        df[f'{col}_mom30'] = df[col].diff(30)
        
        # Create z-scores
        df[f'{col}_zscore'] = (df[col] - df[f'{col}_ma30']) / df[f'{col}_std30']

        # Log transformations for highly skewed features
        df[f'{col}_log'] = np.log1p(df[col] - df[col].min() + 1e-8)  # Safe log

        # Power transformations
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col])) * np.sign(df[col])

    logging.info(f"Processed {len(processed_columns)} on-chain metrics out of {len(onchain_columns)} total")
    return df

def load_macro_data(data_path: str = None, macro_features: List[str] = None, verbose: bool = False) -> pd.DataFrame:
    """
    Load macroeconomic data.
    
    Args:
        data_path: Path to macro data files
        macro_features: List of specific macro features to load (e.g., ['vix'])
        verbose: Whether to print progress
        
    Returns:
        DataFrame with macro metrics
    """
    if verbose:
        logger.info("Loading macro data...")
    
    if not data_path:
        if verbose:
            logger.warning("No macro data path provided")
        return pd.DataFrame()
    
    if not macro_features:
        if verbose:
            logger.warning("No macro features specified")
        return pd.DataFrame()
    
    dfs = []
    for feature in macro_features:
        file_path = os.path.join(data_path, f"{feature.lower()}.csv")
        if not os.path.exists(file_path):
            if verbose:
                logger.warning(f"Macro data file not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Check if the dataframe has sufficient non-NaN values (at least 90 for rolling calculations)
            for col in df.columns:
                non_nan_count = df[col].count()
                if non_nan_count < 90:
                    if verbose:
                        logger.warning(f"Column {col} in {feature} has only {non_nan_count} non-NaN values, dropping")
                    df.drop(columns=[col], inplace=True)
            
            # Only append if there are columns left
            if not df.empty and len(df.columns) > 0:
                dfs.append(df)
                
        except Exception as e:
            if verbose:
                logger.warning(f"Error loading {file_path}: {str(e)}")
    
    if not dfs:
        if verbose:
            logger.warning("No macro data loaded")
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, axis=1)
    
    if verbose:
        logger.info(f"Loaded {combined_df.shape[1]} macro metrics")
    
    return combined_df

def derive_macro_features(df: pd.DataFrame, macro_columns: List[str]) -> pd.DataFrame:
    """
    Generate derived features from macro metrics with robust handling of edge cases.
    
    Args:
        df: DataFrame with macro metrics
        macro_columns: List of macro column names
        
    Returns:
        DataFrame with derived features added
    """
    # Ensure we're working with a copy
    df = df.copy()
    
    # Skip if no macro columns
    if not macro_columns:
        return df
    
    for col in macro_columns:
        if col not in df.columns:
            continue
            
        # Check if column contains string values and convert to numeric if possible
        if df[col].dtype == 'object':
            try:
                # Try to convert strings to numeric values
                col_numeric = pd.to_numeric(df[col], errors='coerce')
                # If conversion was successful (not all NaNs), replace the column
                if not col_numeric.isna().all():
                    df[col] = col_numeric
                    logging.info(f"Successfully converted column {col} from string to numeric")
                else:
                    logging.warning(f"Column {col} could not be converted to numeric, skipping derived features")
                    continue
            except Exception as e:
                logging.warning(f"Error converting column {col} to numeric: {str(e)}, skipping derived features")
                continue
        
        # Check if column has enough non-NaN values (at least 90 for longer rolling calculations)
        non_nan_count = df[col].count()
        if non_nan_count < 90:
            logging.warning(f"Column {col} has only {non_nan_count} non-NaN values, insufficient for rolling calculations")
            continue
            
        # Fill NaN values for calculation purposes - more robust approach
        # First interpolate linearly where possible
        col_filled = df[col].interpolate(method='linear', limit_direction='both')
        # Then use forward/backward fill for any remaining NaNs at edges
        col_filled = col_filled.fillna(method='ffill').fillna(method='bfill')
        
        # Check if filling worked
        if col_filled.isna().any():
            logging.warning(f"Column {col} still contains NaN values after filling, skipping derived features")
            continue
            
        # Check for constant values
        if col_filled.nunique() <= 1:
            logging.warning(f"Column {col} has constant values, skipping derived features")
            continue
        
        # Returns
        df[f'{col}_ret'] = col_filled.pct_change()
        
        # Volatility
        df[f'{col}_vol'] = df[f'{col}_ret'].rolling(window=30).std()
        
        # Moving averages
        df[f'{col}_ma30'] = col_filled.rolling(window=30).mean()
        df[f'{col}_ma90'] = col_filled.rolling(window=90).mean()
        df[f'{col}_ma180'] = col_filled.rolling(window=180).mean()

        # Standard deviation for z-score with epsilon to avoid division by zero
        std_30 = col_filled.rolling(window=30).std()
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        safe_std = std_30.replace(0, epsilon)
        
        # Only calculate z-score if std is mostly non-zero
        if (std_30 <= epsilon).mean() < 0.5:  # If less than 50% of std values are effectively zero
            df[f'{col}_zscore_30'] = (col_filled - df[f'{col}_ma30']) / safe_std
            df[f'{col}_zscore_90'] = (col_filled - df[f'{col}_ma90']) / safe_std
            df[f'{col}_zscore_180'] = (col_filled - df[f'{col}_ma180']) / safe_std

            
            logging.warning(f"Column {col} has too many zero standard deviation values, skipping z-score calculation")
        
        # Momentum is less sensitive to std issues
        df[f'{col}_momentum'] = col_filled.pct_change(periods=30)
        
        # Acceleration
        df[f'{col}_acceleration'] = df[f'{col}_momentum'].diff()

        # Log transformations for highly skewed features
        df[f'{col}_log'] = np.log1p(df[col] - df[col].min() + 1e-8)  # Safe log

        # Power transformations
        df[f'{col}_squared'] = df[col] ** 2
        df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col])) * np.sign(df[col])
    
    return df

def visualize_features(df: pd.DataFrame, features: List[str], correlation_matrix: pd.DataFrame) -> None:
    """
    Create visualizations for feature analysis.
    
    Args:
        df: DataFrame with features
        features: List of feature names
        correlation_matrix: Correlation matrix of features
    """

    import plotly.graph_objects as go
    logger.info("Generating visualizations...")
    
    # Create correlation heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2),
        hoverongaps=False
    ))

    fig.update_layout(
        title='Feature Correlation Matrix',
        width=1000,
        height=1000,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig.show()

def explore_features(data, target_col='log_return', show_plots=False):
    """
    Explore features and their relationships with the target variable
    
    Args:
        data: DataFrame containing features and target
        target_col: Name of the target column
        show_plots: Whether to display visualizations
        
    Returns:
        Dictionary with exploration results
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Calculate correlations with target
    correlations = df.corrwith(df[target_col]).sort_values(ascending=False)
    
    # Get top and bottom correlated features
    top_correlated = correlations.drop(target_col).abs().sort_values(ascending=False).head(20)
    
    # Calculate feature importance using a simple model
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle NaN values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Train a simple model for feature importance
    model = LassoCV(cv=5, random_state=42)
    model.fit(X_scaled, y)
    
    # Get feature importance
    importance = pd.Series(np.abs(model.coef_), index=X.columns)
    top_importance = importance.sort_values(ascending=False).head(20)
    
    if show_plots:
        # Plot correlation with target
        plt.figure(figsize=(12, 8))
        top_correlated.plot(kind='bar')
        plt.title(f'Top Features by Correlation with {target_col}')
        plt.tight_layout()
        plt.show()
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_importance.plot(kind='bar')
        plt.title('Top Features by Importance')
        plt.tight_layout()
        plt.show()
    
    return {
        'processed_data': df,
        'correlations': correlations,
        'top_correlated': top_correlated,
        'importance': importance,
        'top_importance': top_importance
    }


def consensus_feature_selection(
    data, 
    target_col='log_return',
    correlation_threshold=0.05,
    min_consensus=2,
    verbose=True
):
    """
    Perform consensus feature selection using multiple regularization methods
    
    Args:
        data: DataFrame containing features and target
        target_col: Name of the target column
        correlation_threshold: Minimum absolute correlation with target to keep feature
        min_consensus: Minimum number of methods that must select a feature
        verbose: Whether to print progress
        
    Returns:
        Dictionary with consensus features and analysis results
    """
    if verbose:
        print("\n" + "="*80)
        print("STARTING CONSENSUS FEATURE SELECTION")
        print("="*80)
        print(f"Input data shape: {data.shape} ({data.shape[1]} features)")
        print(f"Target column: {target_col}")
        print(f"Correlation threshold: {correlation_threshold}")
        print(f"Minimum consensus required: {min_consensus} methods")
        print("-"*80)
    
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Step 1: Filter by correlation with target
    if verbose:
        print("\nSTEP 1: Correlation Filtering")
        print("-"*50)
    
    correlations = df.corrwith(df[target_col]).drop(target_col)
    
    if verbose:
        print(f"Calculating correlations with {target_col}...")
        print(f"Range of correlations: {correlations.min():.4f} to {correlations.max():.4f}")
        print(f"Mean absolute correlation: {correlations.abs().mean():.4f}")
        
        # Count positive and negative correlations above threshold
        pos_corr = correlations[correlations >= correlation_threshold]
        neg_corr = correlations[correlations <= -correlation_threshold]
        print(f"Features with positive correlation >= {correlation_threshold}: {len(pos_corr)}")
        print(f"Features with negative correlation <= -{correlation_threshold}: {len(neg_corr)}")
        
        # Show top positive and negative correlations
        if len(pos_corr) > 0:
            print("\nTop 5 positively correlated features:")
            for feat, corr in pos_corr.sort_values(ascending=False).head(5).items():
                print(f"  {feat}: +{corr:.4f}")
                
        if len(neg_corr) > 0:
            print("\nTop 5 negatively correlated features:")
            for feat, corr in neg_corr.sort_values().head(5).items():
                print(f"  {feat}: {corr:.4f}")
    
    corr_filtered_features = correlations[correlations.abs() >= correlation_threshold].index.tolist()
    
    if verbose:
        print(f"\nFeatures after correlation filtering (|correlation| >= {correlation_threshold}): {len(corr_filtered_features)}")
        print(f"Removed {len(correlations) - len(corr_filtered_features)} features with low correlation")
    
    # Create filtered dataset
    X_filtered = df[corr_filtered_features]
    y = df[target_col]
    
    # Handle NaN values
    X_filtered = X_filtered.fillna(X_filtered.mean())
    y = y.fillna(y.mean())
    
    # Step 2: Apply multiple regularization methods
    if verbose:
        print("\nSTEP 2: Multiple Regularization Methods")
        print("-"*50)
    
    # Standardize features for model-based selection
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=X_filtered.columns, index=X_filtered.index)
    
    if verbose:
        print(f"Standardized {X_scaled.shape[1]} features for model-based selection")
    
    # Method 1: L1 Regularization (Lasso)
    if verbose:
        print("\nMethod 1: L1 Regularization (Lasso with Cross-Validation)")
    
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)
    lasso_coef = pd.Series(lasso.coef_, index=X_filtered.columns)
    lasso_features = X_filtered.columns[np.abs(lasso.coef_) > 0].tolist()
    
    if verbose:
        print(f"  Regularization parameter (alpha) selected: {lasso.alpha_:.6f}")
        print(f"  Selected {len(lasso_features)} features with non-zero coefficients")
        print(f"  Dropped {len(X_filtered.columns) - len(lasso_features)} features")
        
        # Show coefficient distribution by sign
        pos_coef = lasso_coef[lasso_coef > 0]
        neg_coef = lasso_coef[lasso_coef < 0]
        zero_coef = lasso_coef[lasso_coef == 0]
        print(f"  Coefficient distribution:")
        print(f"    Positive coefficients: {len(pos_coef)}")
        print(f"    Negative coefficients: {len(neg_coef)}")
        print(f"    Zero coefficients (dropped): {len(zero_coef)}")
        
        # Show top positive and negative coefficients
        if len(pos_coef) > 0:
            print("\n  Top 5 features with positive coefficients:")
            for feat, coef in pos_coef.sort_values(ascending=False).head(5).items():
                print(f"    {feat}: +{coef:.6f} (correlation: {correlations[feat]:.4f})")
                
        if len(neg_coef) > 0:
            print("\n  Top 5 features with negative coefficients:")
            for feat, coef in neg_coef.sort_values().head(5).items():
                print(f"    {feat}: {coef:.6f} (correlation: {correlations[feat]:.4f})")
    
    # Method 2: L2 Regularization (Ridge)
    if verbose:
        print("\nMethod 2: L2 Regularization (Ridge with Cross-Validation)")
    
    ridge = RidgeCV(cv=5)
    ridge.fit(X_scaled, y)
    ridge_coef = pd.Series(ridge.coef_, index=X_filtered.columns)
    ridge_threshold = ridge_coef.abs().mean()
    ridge_features = ridge_coef[ridge_coef.abs() >= ridge_threshold].index.tolist()
    
    if verbose:
        print(f"  Regularization parameter (alpha) selected: {ridge.alpha_:.6f}")
        print(f"  Ridge coefficient threshold (mean of absolute coefficients): {ridge_threshold:.6f}")
        print(f"  Selected {len(ridge_features)} features with coefficient magnitude >= threshold")
        print(f"  Dropped {len(X_filtered.columns) - len(ridge_features)} features")
        
        # Show coefficient distribution by sign
        pos_coef = ridge_coef[ridge_coef > 0]
        neg_coef = ridge_coef[ridge_coef < 0]
        pos_selected = ridge_coef[(ridge_coef > 0) & (ridge_coef.abs() >= ridge_threshold)]
        neg_selected = ridge_coef[(ridge_coef < 0) & (ridge_coef.abs() >= ridge_threshold)]
        print(f"  Coefficient distribution:")
        print(f"    Positive coefficients: {len(pos_coef)} (selected: {len(pos_selected)})")
        print(f"    Negative coefficients: {len(neg_coef)} (selected: {len(neg_selected)})")
        
        # Show top positive and negative coefficients that were selected
        if len(pos_selected) > 0:
            print("\n  Top 5 selected features with positive coefficients:")
            for feat, coef in pos_selected.sort_values(ascending=False).head(5).items():
                print(f"    {feat}: +{coef:.6f} (correlation: {correlations[feat]:.4f})")
                
        if len(neg_selected) > 0:
            print("\n  Top 5 selected features with negative coefficients:")
            for feat, coef in neg_selected.sort_values().head(5).items():
                print(f"    {feat}: {coef:.6f} (correlation: {correlations[feat]:.4f})")
    
    # Method 3: ElasticNet (Combined L1 and L2)
    if verbose:
        print("\nMethod 3: ElasticNet Regularization (Combined L1 and L2 with Cross-Validation)")
    
    elastic = ElasticNetCV(cv=5, random_state=42)
    elastic.fit(X_scaled, y)
    elastic_coef = pd.Series(elastic.coef_, index=X_filtered.columns)
    elastic_features = X_filtered.columns[np.abs(elastic.coef_) > 0].tolist()
    
    if verbose:
        print(f"  Regularization parameter (alpha) selected: {elastic.alpha_:.6f}")
        print(f"  L1 ratio selected: {elastic.l1_ratio_:.6f}")
        print(f"  Selected {len(elastic_features)} features with non-zero coefficients")
        print(f"  Dropped {len(X_filtered.columns) - len(elastic_features)} features")
        
        # Show coefficient distribution by sign
        pos_coef = elastic_coef[elastic_coef > 0]
        neg_coef = elastic_coef[elastic_coef < 0]
        zero_coef = elastic_coef[elastic_coef == 0]
        print(f"  Coefficient distribution:")
        print(f"    Positive coefficients: {len(pos_coef)}")
        print(f"    Negative coefficients: {len(neg_coef)}")
        print(f"    Zero coefficients (dropped): {len(zero_coef)}")
        
        # Show top positive and negative coefficients
        if len(pos_coef) > 0:
            print("\n  Top 5 features with positive coefficients:")
            for feat, coef in pos_coef.sort_values(ascending=False).head(5).items():
                print(f"    {feat}: +{coef:.6f} (correlation: {correlations[feat]:.4f})")
                
        if len(neg_coef) > 0:
            print("\n  Top 5 features with negative coefficients:")
            for feat, coef in neg_coef.sort_values().head(5).items():
                print(f"    {feat}: {coef:.6f} (correlation: {correlations[feat]:.4f})")
    
    # Collect results from all methods
    method_results = {
        'lasso': {
            'final_features': lasso_features,
            'details': lasso_coef
        },
        'ridge': {
            'final_features': ridge_features,
            'details': ridge_coef
        },
        'elasticnet': {
            'final_features': elastic_features,
            'details': elastic_coef
        }
    }
    
    if verbose:
        print("\nSummary of feature selection methods:")
        for method, result in method_results.items():
            method_name = {
                'lasso': 'L1 Regularization (Lasso)',
                'ridge': 'L2 Regularization (Ridge)',
                'elasticnet': 'ElasticNet Regularization'
            }.get(method, method)
            
            # Count positive and negative coefficients
            coef = result['details']
            pos_selected = len(coef[coef > 0])
            neg_selected = len(coef[coef < 0])
            
            print(f"  {method_name}: {len(result['final_features'])} features (positive: {pos_selected}, negative: {neg_selected})")
    
    # Step 3: Find consensus features
    if verbose:
        print("\nSTEP 3: Finding Consensus Features")
        print("-"*50)
    
    all_features = set()
    for method, result in method_results.items():
        all_features.update(result['final_features'])
    
    if verbose:
        print(f"Total unique features selected by at least one method: {len(all_features)}")
    
    # Count how many methods selected each feature
    feature_counts = {feature: 0 for feature in all_features}
    for method, result in method_results.items():
        for feature in result['final_features']:
            feature_counts[feature] += 1
    
    # Get consensus features
    consensus_features = [feature for feature, count in feature_counts.items() 
                         if count >= min_consensus]
    
    if verbose:
        print(f"\nConsensus features (selected by at least {min_consensus} methods): {len(consensus_features)}")
        
        # Show feature selection counts
        count_summary = pd.Series(feature_counts).value_counts().sort_index(ascending=False)
        print("\nFeature selection count distribution:")
        for count, num_features in count_summary.items():
            print(f"  Selected by {count} methods: {num_features} features")
        
        # Analyze consensus features by correlation sign
        if consensus_features:
            consensus_corr = correlations[consensus_features]
            pos_consensus = sum(consensus_corr > 0)
            neg_consensus = sum(consensus_corr < 0)
            print(f"\nConsensus features by correlation sign:")
            print(f"  Positive correlation: {pos_consensus} features")
            print(f"  Negative correlation: {neg_consensus} features")
            
            # Show top consensus features
            print("\nTop consensus features:")
            # Sort by absolute correlation with target for display
            feature_corr = correlations[consensus_features].abs().sort_values(ascending=False)
            for i, (feat, corr) in enumerate(feature_corr.head(20).items()):
                # Get original correlation (with sign)
                orig_corr = correlations[feat]
                corr_sign = "+" if orig_corr > 0 else ""
                
                # Get coefficients from each method
                lasso_c = lasso_coef[feat]
                ridge_c = ridge_coef[feat]
                elastic_c = elastic_coef[feat]
                
                # Format coefficient strings with signs
                lasso_str = f"{lasso_c:.4f}" if lasso_c != 0 else "dropped"
                ridge_str = f"{ridge_c:.4f}" if abs(ridge_c) >= ridge_threshold else "dropped"
                elastic_str = f"{elastic_c:.4f}" if elastic_c != 0 else "dropped"
                
                methods_list = []
                for m, r in method_results.items():
                    if feat in r['final_features']:
                        method_name = {
                            'lasso': 'Lasso',
                            'ridge': 'Ridge',
                            'elasticnet': 'ElasticNet'
                        }.get(m, m)
                        methods_list.append(method_name)
                methods_str = ", ".join(methods_list)
                
                print(f"  {i+1}. {feat}")
                print(f"     Correlation: {corr_sign}{orig_corr:.4f}")
                print(f"     Coefficients: Lasso={lasso_str}, Ridge={ridge_str}, ElasticNet={elastic_str}")
                print(f"     Selected by: {methods_str}")
            
            if len(consensus_features) > 20:
                print(f"  ... and {len(consensus_features) - 20} more consensus features")
    
    # Create correlation heatmap for consensus features
    correlation_heatmap = None
    if consensus_features and len(consensus_features) > 1:
        if verbose:
            print("\nCreating correlation heatmap for consensus features...")
        
        corr_matrix = df[consensus_features + [target_col]].corr()
        
        # Create interactive heatmap with plotly
        correlation_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix of Consensus Features"
        )
    
    if verbose:
        print("\n" + "="*80)
        print("CONSENSUS FEATURE SELECTION COMPLETE")
        print("="*80)
        print(f"Final consensus features: {len(consensus_features)}")
        print("="*80)

    return {
        'consensus_features': consensus_features,
        'feature_counts': feature_counts,
        'method_results': method_results,
        'correlation_heatmap': correlation_heatmap,
        'corr_filtered_features': corr_filtered_features
    }

def select_features(
    data, 
    correlation_threshold=0.05, 
    min_consensus=2,
    verbose=True
):
    """
    Complete feature selection pipeline
    
    Args:
        data: DataFrame containing all features
        correlation_threshold: Minimum absolute correlation with log_return to keep feature
        min_consensus: Minimum number of methods that must select a feature
        
    Returns:
        Dictionary containing consensus features and analysis results
    """
    # First run feature exploration
    exploration_results = explore_features(data)
    processed_data = exploration_results['processed_data']

    # Then run consensus feature selection
    consensus_results = consensus_feature_selection(
        processed_data, 
        correlation_threshold=correlation_threshold,
        verbose=verbose,
        min_consensus=min_consensus
    )

    # Display results
    print("\nConsensus Feature Selection Results:")
    print("===================================")
    print(f"\nTotal consensus features: {len(consensus_results['consensus_features'])}")

    if consensus_results['consensus_features']:
        print("\nConsensus Features:")
        print(consensus_results['consensus_features'])
        
        # Show correlation heatmap
        if consensus_results['correlation_heatmap']:
            consensus_results['correlation_heatmap'].show()
    else:
        print("\nNo consensus features found. Try lowering the min_consensus value or adjusting thresholds.")

    # Create a feature selection summary table
    methods = ['lasso', 'ridge', 'elasticnet']
    feature_selection = {}
    print("\nComplete list of consensus features:")
    print("====================================")
    for feature in consensus_results['feature_counts'].keys():
        feature_selection[feature] = {
            'correlation': processed_data.corrwith(processed_data['log_return']).abs()[feature],
            'consensus_count': consensus_results['feature_counts'][feature]
        }
        
        # Add which methods selected this feature
        for method in methods:
            feature_selection[feature][method] = feature in set(consensus_results['method_results'][method]['final_features'])

    # Convert to DataFrame for better display
    selection_df = pd.DataFrame.from_dict(feature_selection, orient='index')
    selection_df = selection_df.sort_values(['consensus_count', 'correlation'], ascending=False)

    print("\nFeature Selection Summary:")
    print(selection_df)

    return consensus_results


    