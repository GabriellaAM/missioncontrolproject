# %% 

import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from experiments.utils.feature_loader import FeatureLoader
import warnings
warnings.filterwarnings('ignore')

# %%

loader = FeatureLoader(start_date='2020-01-01', end_date='2025-10-12')
features_df = loader.build_feature_set(
    crypto_assets=['bitcoin'],
    fred_indicators={'creditSpreads': 'credit_spread', 
                     'treasury5YInflationExpectation': 'inflation_expectation_5y'},
    #yahoo_tickers={'vix': 'vix'},
    #calculated_features={'rty_ym_ratio': 'rty_ym_ratio'}
)

# %%

features_df.head()

# %%
def calculate_information_coefficient(df, feature_column, return_column='bitcoin_close', periods=[5, 10, 20, 30]):
    """
    Calculate the Information Coefficient (IC) between a feature and future non-overlapping returns
    for different time periods, with returns adjusted based on realized volatility over each period.
    
    Parameters:
    - df: DataFrame containing the feature and return data
    - feature_column: str, name of the column containing the feature to evaluate
    - return_column: str, name of the column to calculate returns from (default: 'bitcoin_close')
    - periods: list, time periods for calculating future returns (default: [5, 10, 20, 30])
    
    Returns:
    - ic_results: dict, Information Coefficients for each period
    """
    ic_results = {}
    
    # Calculate future returns for each period
    for period in periods:
        # Calculate non-overlapping future returns
        df[f'future_return_{period}d'] = df[return_column].pct_change(period).shift(-period)
        # Calculate realized volatility over the same future period (using squared returns as proxy)
        df[f'realized_vol_{period}d'] = (df[return_column].pct_change().rolling(window=period).std() * np.sqrt(252)).shift(-period)
        # Adjust returns by realized volatility (vol-adjusted returns)
        df[f'vol_adj_return_{period}d'] = df[f'future_return_{period}d'] / df[f'realized_vol_{period}d']
        # Calculate correlation between feature and vol-adjusted future returns
        ic = df[[feature_column, f'vol_adj_return_{period}d']].corr().iloc[0, 1]
        ic_results[period] = ic
    
    return ic_results

def plot_information_coefficient(ic_results, feature_name):
    """
    Plot the Information Coefficient for different time periods.
    
    Parameters:
    - ic_results: dict, Information Coefficients for each period
    - feature_name: str, name of the feature being evaluated
    """
    periods = list(ic_results.keys())
    ic_values = list(ic_results.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(periods, ic_values, color='skyblue')
    plt.xlabel('Return Period (Days)')
    plt.ylabel('Information Coefficient')
    plt.title(f'Information Coefficient for {feature_name} vs Future Returns')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(ic_values):
        plt.text(periods[i], v, f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top')
    plt.show()

# %%

# EMA signals
features_df['ema_9'] = features_df['bitcoin_close'].ewm(span=9, adjust=False).mean()
features_df['ema_21'] = features_df['bitcoin_close'].ewm(span=21, adjust=False).mean()
features_df['ema_50'] = features_df['bitcoin_close'].ewm(span=50, adjust=False).mean()
features_df['ema_fast_cross'] = np.where(features_df['ema_9'] >= features_df['ema_21'], 1, 0)
features_df['ema_fast_cross'] = features_df['ema_fast_cross'].fillna(0)
features_df['ema_slow_cross'] = np.where(features_df['ema_21'] >= features_df['ema_50'], 1, 0)
features_df['ema_slow_cross'] = features_df['ema_slow_cross'].fillna(0)

# Calculate and plot IC for EMA fast cross
ic_ema_fast = calculate_information_coefficient(features_df, 'ema_fast_cross')
plot_information_coefficient(ic_ema_fast, 'EMA Fast Cross (9/21)')

# Calculate and plot IC for EMA slow cross
ic_ema_slow = calculate_information_coefficient(features_df, 'ema_slow_cross')
plot_information_coefficient(ic_ema_slow, 'EMA Slow Cross (21/50)')

# %%

# Consecutive up days
features_df['consecutive_up_days'] = (features_df['bitcoin_close'].pct_change() > 0.005).astype(int).groupby((features_df['bitcoin_close'].pct_change() <= 0.005).astype(int).cumsum()).cumcount() + 1
features_df['consecutive_up_days'] = features_df['consecutive_up_days'].fillna(0)

# Calculate and plot IC for consecutive up days
ic_up_days = calculate_information_coefficient(features_df, 'consecutive_up_days')
plot_information_coefficient(ic_up_days, 'Consecutive Up Days')

# %%

# Rolling OHLC Z-Score
features_df['ohlc'] = (features_df['bitcoin_open'] + features_df['bitcoin_high'] + features_df['bitcoin_low'] + features_df['bitcoin_close']) / 4
features_df['ohlc_z_score_20'] = (features_df['ohlc'] - features_df['ohlc'].rolling(window=20).mean()) / features_df['ohlc'].rolling(window=20).std()
features_df['ohlc_z_score_10'] = (features_df['ohlc'] - features_df['ohlc'].rolling(window=10).mean()) / features_df['ohlc'].rolling(window=10).std()
features_df['ohlc_z_score_5'] = (features_df['ohlc'] - features_df['ohlc'].rolling(window=5).mean()) / features_df['ohlc'].rolling(window=5).std()
features_df['ohlc_z_score_30'] = (features_df['ohlc'] - features_df['ohlc'].rolling(window=30).mean()) / features_df['ohlc'].rolling(window=30).std()

# Calculate and plot IC for OHLC Z-Scores
ic_z_score_5 = calculate_information_coefficient(features_df, 'ohlc_z_score_5')
plot_information_coefficient(ic_z_score_5, 'OHLC Z-Score (5-day)')

ic_z_score_10 = calculate_information_coefficient(features_df, 'ohlc_z_score_10')
plot_information_coefficient(ic_z_score_10, 'OHLC Z-Score (10-day)')

ic_z_score_20 = calculate_information_coefficient(features_df, 'ohlc_z_score_20')
plot_information_coefficient(ic_z_score_20, 'OHLC Z-Score (20-day)')

ic_z_score_30 = calculate_information_coefficient(features_df, 'ohlc_z_score_30')
plot_information_coefficient(ic_z_score_30, 'OHLC Z-Score (30-day)')

# %%

