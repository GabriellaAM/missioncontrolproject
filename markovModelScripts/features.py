import numpy as np
import pandas as pd
import pandas_ta as ta
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler
from arch import arch_model
import logging
import pickle

def calculate_atr(high, low, close, window=14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

def apply_initial_features(df, asset):
    df = df.copy()

    kf = KalmanFilter(
        transition_matrices=[1], observation_matrices=[1], initial_state_mean=df['Close'].iloc[0],
        initial_state_covariance=1, observation_covariance=1, transition_covariance=0.01
    )
    state_means, _ = kf.filter(df['Close'].values)

    df['kalman_close'] = state_means.flatten()
    smoothed_state_means, _ = kf.smooth(df['kalman_close'].values)
    df['kalman_smooth'] = smoothed_state_means.flatten()
    df['close_kalman_diff'] = np.abs(df['Close'] - df['kalman_smooth'])
    df['close_slope'] = df['Close'].diff(periods=7)
    df['kalman_smooth_slope'] = df['kalman_smooth'].diff(periods=7)
    df['kalman_close_slope'] = df['kalman_close'].diff(periods=7)
    df['RSI14'] = ta.rsi(df['Close'], length=14)

    if asset != "bitcoin":
        btc_dom_data = pd.read_csv('/Users/valter.rebelo/MissionControl/data/micro/marketData/btc_dominance.csv',
                                   index_col=0, parse_dates=True)
        
        df = df.merge(btc_dom_data, left_index=True, right_index=True, how='left')

        df.rename(columns={btc_dom_data.columns[0]: 'btc_dom'}, inplace=True)

    return df

def process_features(asset, df, scaler=None, is_training=False):

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger('feature_processing')

    processed_df = df.copy().dropna()

    logger.info(f"Processing features for {'training' if is_training else 'test/embargo'} set")

    processed_df['volume_return'] = processed_df['Volume'].pct_change()

    low_lim = processed_df['volume_return'].quantile(0.02)

    up_lim = processed_df['volume_return'].quantile(0.97)

    processed_df['volume_return'].clip(lower=low_lim, upper=up_lim, inplace=True)

    processed_df['log'] = np.log(processed_df['Close'].replace(0, np.nan))

    processed_df['log_return'] = processed_df['log'].diff()

    processed_df['log_return_30'] = processed_df['log'].diff(periods=30)
    
    processed_df['log_return_7'] = processed_df['log'].diff(periods=7)

    if 'btc_dom' in processed_df.columns:
        processed_df['btc_dom_return_7'] = processed_df['btc_dom'].pct_change(periods=7)

    for feature, log_name in [('move', 'move_log'), ('treasury5YInflationExpectation', 'treasury5YInflationExpectation_log'),
                              ('creditSpreads', 'creditSpreads_log')]:
        
        if feature in processed_df.columns:
            processed_df[log_name] = np.log(processed_df[feature].replace(0, np.nan))
            processed_df[f'{feature}_log_return'] = processed_df[log_name].diff()

    processed_df['lnrange'] = np.log((processed_df['High'] / processed_df['Low']).replace([np.inf, -np.inf], np.nan))
    processed_df['ln_atr'] = np.log(calculate_atr(processed_df['High'], processed_df['Low'], processed_df['Close'], window=14).replace(0, np.nan))
    clean_returns = processed_df['log_return'].replace([np.inf, -np.inf], np.nan).dropna()
    clean_returns *= 100 

    try:
        garch_model = arch_model(clean_returns, vol='GARCH', p=1, q=1, mean='Zero', dist='studentst')

        garch_fit = garch_model.fit(disp='off')

        processed_df['garch_volatility'] = pd.Series(garch_fit.conditional_volatility, index=clean_returns.index)

    except Exception as e:

        logger.error(f"GARCH fitting failed: {str(e)}")
        processed_df['garch_volatility'] = processed_df['ln_atr']

    
    if asset != "bitcoin":

        features_to_scale = ['log', 'log_return', 'lnrange', 'RSI14', 'log_return_30', 'log_return_7', 'ln_atr',
                            'kalman_close', 'kalman_smooth', 'kalman_smooth_slope', 'close_kalman_diff',
                            'volume_return', 'garch_volatility', 'marketCap', 'btc_dom', 'btc_dom_return_7']
        
    else:

        features_to_scale = ['log', 'log_return', 'lnrange', 'RSI14', 'log_return_30', 'log_return_7', 'ln_atr',
                            'kalman_close', 'kalman_smooth', 'kalman_smooth_slope', 'close_kalman_diff',
                            'volume_return', 'garch_volatility', 'marketCap']
        
    macro_features = ['creditSpreads_log_return', 'creditSpreads', 'creditSpreads_log',
                      'treasury5YInflationExpectation_log_return', 'treasury5YInflationExpectation_log',
                      'treasury5YInflationExpectation', 'move_log_return', 'move_log', 'move']
    
    features_to_scale.extend([f for f in macro_features if f in processed_df.columns])

    for feature in features_to_scale:
        if feature in processed_df.columns:
            processed_df[feature] = processed_df[feature].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    if is_training:

        scaler = StandardScaler()
        processed_df[features_to_scale] = scaler.fit_transform(processed_df[features_to_scale])

    else:

        if scaler is None:
            raise ValueError("Scaler must be provided when processing test data")
        processed_df[features_to_scale] = scaler.transform(processed_df[features_to_scale])

    return processed_df, scaler

def save_scaler(scaler, filepath):

    """Save the fitted scaler to a file using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filepath}")

def load_scaler(filepath):

    """Load a fitted scaler from a file using pickle."""
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {filepath}")
    return scaler