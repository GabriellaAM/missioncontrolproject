# %% 

import os
import warnings
warnings.filterwarnings('ignore')
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

asset = 'bitcoin'
prefix = f"{asset}_"
loader = FeatureLoader(start_date='2018-01-01', end_date='2025-12-01')
features_df = loader.build_feature_set(
    crypto_assets=['bitcoin', 'dogecoin'],
    #fred_indicators={'creditSpreads': 'credit_spread', 
    #                'treasury5YInflationExpectation': 'inflation_expectation_5y'},
    #yahoo_tickers={'vix': 'vix', 'move': 'move'},
    #calculated_features={'rty_ym_ratio': 'rty_ym_ratio'}
)

# %%

features_df.head(20)



# %%
def calculate_information_coefficient(df, feature_column, return_column=f'{prefix}close', periods=[5, 10, 15, 20, 25, 30]):
    """
    Calculate the Information Coefficient (IC) between a feature and future non-overlapping returns
    for different time periods, with returns adjusted based on realized volatility over each period.
    
    Parameters:
    - df: DataFrame containing the feature and return data
    - feature_column: str, name of the column containing the feature to evaluate
    - return_column: str, name of the column to calculate returns from (default: f'{prefix}close')
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

features_df['volume_ma_ratio'] = features_df[f'{prefix}total_volume'] / features_df[f'{prefix}total_volume'].rolling(window=30).mean()
features_df['volume_price_correlation'] = features_df[f'{prefix}total_volume'].rolling(window=30).corr(features_df[f'{prefix}close'])

ic_volume_ma_ratio = calculate_information_coefficient(features_df, 'volume_ma_ratio')
plot_information_coefficient(ic_volume_ma_ratio, f'{asset} Volume MA Ratio')
ic_volume_price_correlation = calculate_information_coefficient(features_df, 'volume_price_correlation')
plot_information_coefficient(ic_volume_price_correlation, 'Volume Price Correlation')


# %%

# EMA signals
features_df['ema_9'] = features_df[f'{prefix}close'].ewm(span=9, adjust=False).mean()
features_df['ema_21'] = features_df[f'{prefix}close'].ewm(span=21, adjust=False).mean()
features_df['ema_50'] = features_df[f'{prefix}close'].ewm(span=50, adjust=False).mean()
features_df['ema_fast_cross'] = np.where(features_df['ema_9'] >= features_df['ema_21'], 1, -1)
features_df['ema_diff'] = features_df['ema_9']-features_df['ema_21']
features_df['ema_fast_cross'] = features_df['ema_fast_cross'].fillna(0)
features_df['ema_slow_cross'] = np.where(features_df['ema_21'] >= features_df['ema_50'], 1, -1)
features_df['ema_slow_cross'] = features_df['ema_slow_cross'].fillna(0)

# Calculate and plot IC for EMA fast cross
ic_ema_fast = calculate_information_coefficient(features_df, 'ema_fast_cross')
plot_information_coefficient(ic_ema_fast, 'EMA Fast Cross (9/21)')

# Calculate and plot IC for EMA slow cross
ic_ema_slow = calculate_information_coefficient(features_df, 'ema_diff')
plot_information_coefficient(ic_ema_slow, 'EMA Diff (9/21)')


# %%
features_df['btc_log_return_7d'] = np.log(features_df[f'{prefix}close'] / features_df[f'{prefix}close'].shift(7))
features_df['btc_log_return_7d'] = features_df['btc_log_return_7d'].fillna(0)
features_df['btc_log_return_30d'] = np.log(features_df[f'{prefix}close'] / features_df[f'{prefix}close'].shift(30))
features_df['btc_log_return_30d'] = features_df['btc_log_return_30d'].fillna(0)
features_df['btc_log_return'] = np.log(features_df[f'{prefix}close'] / features_df[f'{prefix}close'].shift(1))

ic_btc_log_return_30d = calculate_information_coefficient(features_df, 'btc_log_return_30d')
plot_information_coefficient(ic_btc_log_return_30d, 'BTC Log Return (30-day)')

ic_btc_log_return = calculate_information_coefficient(features_df, 'btc_log_return')
plot_information_coefficient(ic_btc_log_return, 'BTC Log Return (1-day)')


# Calculate the 7-day rolling linear correlation between bitcoin_total_volume and btc_log_return
rolling_corr = features_df[f'{prefix}total_volume'].rolling(window=7).corr(features_df['btc_log_return'])
# Compute the exponential moving average (EMA) with a span of 7 days on the rolling correlation
ema_corr = rolling_corr.ewm(span=7, adjust=False).mean()
# Store the result as a new column in features_df
features_df['ema_corr_volume_log_return_7d'] = ema_corr

ic_ema_corr_volume_log_return_7d = calculate_information_coefficient(features_df, 'ema_corr_volume_log_return_7d')
plot_information_coefficient(ic_ema_corr_volume_log_return_7d, 'EMA Correlation Volume Log Return (7-day)')


# %%

features_df['std_30d'] = features_df['btc_log_return'].rolling(window=30).std()
features_df['std_60d'] = features_df['btc_log_return'].rolling(window=60).std()

ic_std_30d = calculate_information_coefficient(features_df, 'std_30d')
plot_information_coefficient(ic_std_30d, 'STD (30-day)')

ic_std_60d = calculate_information_coefficient(features_df, 'std_60d')
plot_information_coefficient(ic_std_60d, 'STD (60-day)')


# %% 

# Calculate ADX (Average Directional Index)
# First, calculate True Range (TR)
features_df['high_low'] = features_df[f'{prefix}high'] - features_df[f'{prefix}low']
features_df['high_close'] = abs(features_df[f'{prefix}high'] - features_df[f'{prefix}close'].shift(1))
features_df['low_close'] = abs(features_df[f'{prefix}low'] - features_df[f'{prefix}close'].shift(1))
features_df['tr'] = features_df[['high_low', 'high_close', 'low_close']].max(axis=1)

# Calculate Directional Movement (DM)
features_df['dm_plus'] = np.where(
    (features_df[f'{prefix}high'] - features_df[f'{prefix}high'].shift(1)) > (features_df[f'{prefix}low'].shift(1) - features_df[f'{prefix}low']),
    np.maximum(features_df[f'{prefix}high'] - features_df[f'{prefix}high'].shift(1), 0),
    0
)
features_df['dm_minus'] = np.where(
    (features_df[f'{prefix}low'].shift(1) - features_df[f'{prefix}low']) > (features_df[f'{prefix}high'] - features_df[f'{prefix}high'].shift(1)),
    np.maximum(features_df[f'{prefix}low'].shift(1) - features_df[f'{prefix}low'], 0),
    0
)

# Smooth the TR and DM values (typically using 14 periods)
features_df['tr_14'] = features_df['tr'].rolling(window=14).mean()
features_df['dm_plus_14'] = features_df['dm_plus'].rolling(window=14).mean()
features_df['dm_minus_14'] = features_df['dm_minus'].rolling(window=14).mean()

# Calculate Directional Index (DI)
features_df['di_plus'] = (features_df['dm_plus_14'] / features_df['tr_14']) * 100
features_df['di_minus'] = (features_df['dm_minus_14'] / features_df['tr_14']) * 100

# Calculate DX (Directional Movement Index)
features_df['di_diff'] = abs(features_df['di_plus'] - features_df['di_minus'])
features_df['di_sum'] = features_df['di_plus'] + features_df['di_minus']
features_df['dx'] = (features_df['di_diff'] / features_df['di_sum']) * 100

# Calculate ADX by smoothing DX (typically using 14 periods)
features_df['adx'] = features_df['dx'].rolling(window=14).mean()

# Fill NaN values
features_df['adx'] = features_df['adx'].fillna(0)

# Calculate and plot IC for ADX
ic_adx = calculate_information_coefficient(features_df, 'adx')
plot_information_coefficient(ic_adx, 'ADX (14-day)')

# %% 
# Calculate RSI (Relative Strength Index)
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

features_df['rsi_14'] = calculate_rsi(features_df[f'{prefix}close'], 14)
features_df['rsi_14'] = features_df['rsi_14'].fillna(0)

# Calculate and plot IC for RSI
ic_rsi = calculate_information_coefficient(features_df, 'rsi_14')
plot_information_coefficient(ic_rsi, 'RSI (14-day)')

# Calculate MACD (Moving Average Convergence Divergence)
def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
    exp1 = data.ewm(span=fastperiod, adjust=False).mean()
    exp2 = data.ewm(span=slowperiod, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    return macd, signal

features_df['macd'], features_df['macd_signal'] = calculate_macd(features_df[f'{prefix}close'])
features_df['macd_hist'] = features_df['macd'] - features_df['macd_signal']
features_df[['macd', 'macd_signal', 'macd_hist']] = features_df[['macd', 'macd_signal', 'macd_hist']].fillna(0)
features_df['macds'] = features_df['macd'] - features_df['macd_signal']


# Calculate and plot IC for MACD components
ic_macd = calculate_information_coefficient(features_df, 'macd')
plot_information_coefficient(ic_macd, 'MACD')

ic_macd_signal = calculate_information_coefficient(features_df, 'macd_signal')
plot_information_coefficient(ic_macd_signal, 'MACD Signal')

ic_macd_hist = calculate_information_coefficient(features_df, 'macd_hist')
plot_information_coefficient(ic_macd_hist, 'MACD Histogram')

ic_macds = calculate_information_coefficient(features_df, 'macds')
plot_information_coefficient(ic_macds, 'MACDS')



# %%

# Consecutive up days
features_df['consecutive_up_days'] = (features_df[f'{prefix}close'].pct_change() > 0.005).astype(int).groupby((features_df[f'{prefix}close'].pct_change() <= 0.005).astype(int).cumsum()).cumcount() + 1
features_df['consecutive_up_days'] = features_df['consecutive_up_days'].fillna(0)

features_df['consecutive_down_days'] = (features_df[f'{prefix}close'].pct_change() < 0.005).astype(int).groupby((features_df[f'{prefix}close'].pct_change() >= 0.005).astype(int).cumsum()).cumcount() + 1
features_df['consecutive_down_days'] = features_df['consecutive_down_days'].fillna(0)

# Calculate and plot IC for consecutive up days
ic_up_days = calculate_information_coefficient(features_df, 'consecutive_up_days')
plot_information_coefficient(ic_up_days, 'Consecutive Up Days (0.5%)')

ic_down_days = calculate_information_coefficient(features_df, 'consecutive_down_days')
plot_information_coefficient(ic_down_days, 'Consecutive Down Days (0.5%)')


features_df['consecutive_up_days'].head(100).plot()
features_df['consecutive_down_days'].head(100).plot()

# %%

# Rolling OHLC Z-Score
features_df['ohlc'] = (features_df[f'{prefix}open'] + features_df[f'{prefix}high'] + features_df[f'{prefix}low'] + features_df[f'{prefix}close']) / 4
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

features_df['credit_spread_7d'] = features_df['credit_spread'].diff(7)
features_df['credit_spread_14d'] = features_df['credit_spread'].diff(14)
features_df['credit_spread_21d'] = features_df['credit_spread'].diff(21)

ic_credit_spread_7d = calculate_information_coefficient(features_df, 'credit_spread_7d')
plot_information_coefficient(ic_credit_spread_7d, 'Credit Spread (7-day)')

ic_credit_spread_14d = calculate_information_coefficient(features_df, 'credit_spread_14d')
plot_information_coefficient(ic_credit_spread_14d, 'Credit Spread (14-day)')

ic_credit_spread_21d = calculate_information_coefficient(features_df, 'credit_spread_21d')
plot_information_coefficient(ic_credit_spread_21d, 'Credit Spread (21-day)')



# %%

features_df['inflation_expectation_5y_7d'] = features_df['inflation_expectation_5y'].diff(7)
features_df['inflation_expectation_5y_14d'] = features_df['inflation_expectation_5y'].diff(14)
features_df['inflation_expectation_5y_21d'] = features_df['inflation_expectation_5y'].diff(21)
features_df['inflation_expectation_5y_60d'] = features_df['inflation_expectation_5y'].diff(60)


ic_inflation_expectation_5y_7d = calculate_information_coefficient(features_df, 'inflation_expectation_5y_7d')
plot_information_coefficient(ic_inflation_expectation_5y_7d, 'Inflation Expectation (5-year) (7-day)')

ic_inflation_expectation_5y_14d = calculate_information_coefficient(features_df, 'inflation_expectation_5y_14d')
plot_information_coefficient(ic_inflation_expectation_5y_14d, 'Inflation Expectation (5-year) (14-day)')

ic_inflation_expectation_5y_21d = calculate_information_coefficient(features_df, 'inflation_expectation_5y_21d')
plot_information_coefficient(ic_inflation_expectation_5y_21d, 'Inflation Expectation (5-year) (21-day)')

ic_inflation_expectation_5y_60d = calculate_information_coefficient(features_df, 'inflation_expectation_5y_60d')
plot_information_coefficient(ic_inflation_expectation_5y_60d, 'Inflation Expectation (5-year) (60-day)')

# %% 

# %%

features_df['move_close_7d'] = features_df['move_close'].diff(7)
features_df['move_close_14d'] = features_df['move_close'].diff(14)
features_df['move_close_21d'] = features_df['move_close'].diff(21)
features_df['move_close_60d'] = features_df['move_close'].diff(60)

ic_move_close_14d = calculate_information_coefficient(features_df, 'move_close_14d')
plot_information_coefficient(ic_move_close_14d, 'Move Close (14-day)')



# %%

features_df['vix_close_7d'] = features_df['vix_close'].diff(7)
features_df['vix_close_14d'] = features_df['vix_close'].diff(14)
features_df['vix_close_21d'] = features_df['vix_close'].diff(21)
features_df['vix_close_60d'] = features_df['vix_close'].diff(60)


ic_vix_close_60d = calculate_information_coefficient(features_df, 'vix_close_60d')
plot_information_coefficient(ic_vix_close_60d, 'VIX Close (60-day)')

# %%

features_df[ 'rty_ym_ratio_7d'] = features_df['rty_ym_ratio'].diff(7)
features_df[ 'rty_ym_ratio_14d'] = features_df['rty_ym_ratio'].diff(14)
features_df[ 'rty_ym_ratio_90d'] = features_df['rty_ym_ratio'].diff(90)

ic_rty_ym_ratio_7d = calculate_information_coefficient(features_df, 'rty_ym_ratio_7d')
plot_information_coefficient(ic_rty_ym_ratio_7d, 'RTY YM Ratio (7-day)')

ic_rty_ym_ratio_14d = calculate_information_coefficient(features_df, 'rty_ym_ratio_14d')
plot_information_coefficient(ic_rty_ym_ratio_14d, 'RTY YM Ratio (14-day)')

ic_rty_ym_ratio_90d = calculate_information_coefficient(features_df, 'rty_ym_ratio_90d')
plot_information_coefficient(ic_rty_ym_ratio_90d, 'RTY YM Ratio (90-day)')

# %%

# Correlation matrix plot of features
import seaborn as sns
import matplotlib.pyplot as plt

# Select features for correlation analysis
features_for_correlation = [
    'btc_log_return_30d', 'ema_diff',
    'macd', 'macds', 'rsi_14', 'btc_log_return_7d',
    'consecutive_up_days', 'consecutive_down_days',
    'credit_spread_14d', 'ohlc_z_score_30', 'ema_corr_volume_log_return_7d'
   
]
# %%
# Calculate correlation matrix
correlation_matrix = features_df[features_for_correlation].corr()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()

# %%

# Train-test split and preprocessing for model training
from sklearn.model_selection import train_test_split
import numpy as np

# Define the features to use for modeling (from features_for_correlation)
selected_features = features_for_correlation

# Define the target variable as Bitcoin's 5-day forward log return
features_df['target'] = np.log(features_df[f'{prefix}close'] / features_df[f'{prefix}close'].shift(5)).shift(-5)
features_df['ret_ser'] = np.log(features_df[f'{prefix}close'] / features_df[f'{prefix}close'].shift(1))
features_df['target'] = features_df['target'].fillna(0)

# Drop rows with NaN values in features or target
model_data = features_df[selected_features + ['target']].dropna()

# Split into features (X) and target (y)
X = model_data[selected_features]
y = model_data['target']

# Perform train-test split (80-20 split, adjust as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# Rolling window clipping and standardization
def rolling_clip_and_standardize(data, window=30, clip_std=3):
    """
    Apply rolling clipping and standardization to the data.
    Clips data at clip_std standard deviations and standardizes using rolling mean and std.
    """
    data_clipped = data.copy()
    data_standardized = data.copy()
    
    for col in data.columns:
        # Rolling mean and std for clipping
        rolling_mean = data[col].rolling(window=window, min_periods=1).mean()
        rolling_std = data[col].rolling(window=window, min_periods=1).std()
        
        # Clip data at clip_std standard deviations
        upper_bound = rolling_mean + clip_std * rolling_std
        lower_bound = rolling_mean - clip_std * rolling_std
        data_clipped[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Standardize using rolling statistics
        data_standardized[col] = (data_clipped[col] - rolling_mean) / rolling_std
    
    return data_standardized.fillna(0)

# Apply rolling clipping and standardization to training data
X_train_processed = rolling_clip_and_standardize(X_train)

# Apply rolling clipping and standardization to test data
# Note: Using the same rolling window approach, but in practice, you might want to use train stats
X_test_processed = rolling_clip_and_standardize(X_test)

# Output variables ready for modeling
print("Train and test sets are ready for modeling.")
print(f"Training set shape: {X_train_processed.shape}")
print(f"Test set shape: {X_test_processed.shape}")

# %% 

from jumpmodels.sparse_jump import SparseJumpModel
#from jumpmodels.plot import plot_regimes_and_cumret, savefig_plt

max_feats=4.
jump_penalty=50

# init sjm instance
sjm = SparseJumpModel(n_components=2, max_feats=max_feats, jump_penalty=jump_penalty, )
# fit
sjm.fit(X_train_processed, ret_ser=features_df['ret_ser'], sort_by="cumret")
print("SJM Feature Weights:", "-"*50, sjm.feat_weights, sep="\n")


# %% 

# Disable LaTeX rendering to avoid errors if latex is not installed
#plt.rcParams['text.usetex'] = False

# Create a figure with two subplots
#fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the first 100 labels from the SparseJumpModel on the first axis
labels_df = sjm.labels_.to_frame(name='Labels')
labels_df.plot()




# %% 

labels_test_online_sjm = sjm.predict_online(X_test_processed)


# %% 

import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Combine in-sample and out-of-sample labels
in_sample_labels = sjm.labels_.to_frame(name='Labels')
out_sample_labels = pd.DataFrame(labels_test_online_sjm, columns=['Labels'])
combined_labels = pd.concat([in_sample_labels, out_sample_labels])

# Create a plotly figure
fig = go.Figure()

# Plot Bitcoin close prices as a line chart using the pallet's light grey color
bitcoin_prices = features_df[f'{prefix}close']  # Using close prices directly
fig.add_trace(go.Scatter(
    x=bitcoin_prices.index,
    y=bitcoin_prices,
    mode='lines',
    name=asset,
    line=dict(color='#000C1A', width=1)
))

# Define colors for different regimes using the provided pallet:
# For Risk-On (regime 0) use a light green, for Risk-Off (regime 1) use a dark green.
regime_color_map = {0: "#39FAA2", 1: "red"}

# Overlay price points with dots colored by regime label
regime_indices = combined_labels.index
regimes = combined_labels['Labels'].values
for regime in np.unique(regimes):
    regime_mask = regimes == regime
    regime_name = "Risk-On" if regime == 0 else "Risk-Off"
    hover_text = [
        f"Date: {idx.strftime('%d-%m-%Y')}<br>Regime: {regime_name}<br>Close Price: {bitcoin_prices.loc[idx]:.2f}"
        for idx in regime_indices[regime_mask]
    ]
    fig.add_trace(go.Scatter(
        x=regime_indices[regime_mask],
        y=bitcoin_prices.loc[regime_indices[regime_mask]],
        mode='markers',
        name=regime_name,
        marker=dict(size=4.5, color=regime_color_map.get(regime, "#9CFFD1")),
        showlegend=True,
        text=hover_text,
        hoverinfo='text'
    ))

# Add a vertical line to separate in-sample and out-of-sample periods using the deep navy from the pallet
split_date = in_sample_labels.index[-1]
fig.add_shape(
    type="line",
    x0=split_date,
    y0=0,
    x1=split_date,
    y1=1,
    xref="x",
    yref="paper",
    line=dict(color="#000C1A", dash="dash"),
    name="In/Out Sample Split"
)

# Update layout with title and labels, setting the background color to #D9D9D9
fig.update_layout(
    title=f"Classificação de Regimes por SJM para {asset}",
    xaxis_title="",
    yaxis_title=f"Preço de {asset}",
    width=1000,
    height=600,
    showlegend=True,
    paper_bgcolor="#D9D9D9",
    plot_bgcolor="#D9D9D9",
    xaxis=dict(
        tickformat="%b %Y",
        tickangle=45
    )
)

# Show the plot
fig.show()



# %% 

# Calculate statistical characteristics for each regime, split by train and test sets, in terms of returns
print("Statistical Characteristics of Regimes (Train and Test Sets) - Returns:")

# Since bitcoin_prices is a pandas Series with 2127 rows and regimes is a numpy array with 2067 elements,
# assume that regimes correspond to the last 2067 entries of bitcoin_prices.
price_series = features_df[f'{prefix}close'].rename("Price")
regime_series = pd.Series(regimes, index=price_series.index[-len(regimes):]).rename("Regime")

# Align data by taking the subset of price_series that corresponds to the regime_series index
data_df = pd.concat([price_series[-len(regimes):], regime_series], axis=1)

# Drop any rows with missing data to ensure alignment (should be none, but for safety)
data_df = data_df.dropna()

# Calculate daily returns (in percentage) using log returns
data_df['Return'] = np.log(data_df['Price'] / data_df['Price'].shift(1)) * 100
data_df = data_df.dropna()  # Drop rows where return calculation resulted in NaN

# Split into train and test sets based on the split_date
train_df = data_df[data_df.index <= split_date]
test_df = data_df[data_df.index > split_date]

def calculate_additional_metrics(df):
    # Calculate Profit Factor: Sum of positive returns divided by absolute sum of negative returns.
    pos_sum = df.loc[df['Return'] > 0, 'Return'].sum()
    neg_sum = abs(df.loc[df['Return'] < 0, 'Return'].sum())
    profit_factor = pos_sum / neg_sum if neg_sum != 0 else np.nan
    # Calculate Sharpe Ratio: mean/standard deviation. (Assume risk-free rate=0)
    sharpe_ratio = df['Return'].mean() / df['Return'].std() if df['Return'].std() != 0 else np.nan
    # Calculate Maximum Drawdown: using cumulative returns computed from log returns.
    # Convert log returns back to simple returns via exponential transformation.
    cum_returns = np.exp(df['Return'] / 100).cumprod()
    drawdown = cum_returns / cum_returns.cummax() - 1
    max_drawdown = drawdown.min() * 100  # expressed in percentage
    return profit_factor, sharpe_ratio, max_drawdown

def compute_mean_regime_duration(df, regime):
    """
    Compute the mean duration (in days) of consecutive periods for the specified regime in the given DataFrame.
    """
    durations = []
    # Ensure the DataFrame is sorted by date (index)
    df_sorted = df.sort_index()
    # Group by contiguous segments where the regime label remains the same
    groups = df_sorted.groupby((df_sorted['Regime'] != df_sorted['Regime'].shift()).cumsum())
    for _, group in groups:
        if group['Regime'].iloc[0] == regime:
            # Duration in days = difference between last and first date + 1
            duration = (group.index[-1] - group.index[0]).days + 1
            durations.append(duration)
    return np.mean(durations) if durations else np.nan

# Calculate stats for each regime in train and test sets based on returns
for regime in [0, 1]:
    regime_name = "Bull" if regime == 0 else "Bear"
    
    # Train set stats
    train_regime_df = train_df[train_df['Regime'] == regime]
    if not train_regime_df.empty:
        profit_factor, sharpe_ratio, max_drawdown = calculate_additional_metrics(train_regime_df)
        mean_regime_length = compute_mean_regime_duration(train_df, regime)
        train_stats = {
            'Count': len(train_regime_df),
            'Mean Return (%)': train_regime_df['Return'].mean(),
            'Standard Deviation (%)': train_regime_df['Return'].std(),
            'Minimum Return (%)': train_regime_df['Return'].min(),
            'Maximum Return (%)': train_regime_df['Return'].max(),
            'Kurtosis': train_regime_df['Return'].kurt(),
            'Skewness': train_regime_df['Return'].skew(),
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Mean Regime Length (days)': mean_regime_length
        }
        print(f"\n{regime_name} Regime (Regime {regime}) - Train Set:")
        for stat_name, stat_value in train_stats.items():
            if stat_name == 'Count':
                print(f"  {stat_name}: {stat_value}")
            else:
                print(f"  {stat_name}: {stat_value:.2f}")
    else:
        print(f"\n{regime_name} Regime (Regime {regime}) - Train Set: No data available")
    
    # Test set stats
    test_regime_df = test_df[test_df['Regime'] == regime]
    if not test_regime_df.empty:
        profit_factor, sharpe_ratio, max_drawdown = calculate_additional_metrics(test_regime_df)
        mean_regime_length = compute_mean_regime_duration(test_df, regime)
        test_stats = {
            'Count': len(test_regime_df),
            'Mean Return (%)': test_regime_df['Return'].mean(),
            'Standard Deviation (%)': test_regime_df['Return'].std(),
            'Minimum Return (%)': test_regime_df['Return'].min(),
            'Maximum Return (%)': test_regime_df['Return'].max(),
            'Kurtosis': test_regime_df['Return'].kurt(),
            'Skewness': test_regime_df['Return'].skew(),
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Mean Regime Length (days)': mean_regime_length
        }
        print(f"\n{regime_name} Regime (Regime {regime}) - Test Set:")
        for stat_name, stat_value in test_stats.items():
            if stat_name == 'Count':
                print(f"  {stat_name}: {stat_value}")
            else:
                print(f"  {stat_name}: {stat_value:.2f}")
    else:
        print(f"\n{regime_name} Regime (Regime {regime}) - Test Set: No data available")

# Perform t-tests to check if the mean returns of the two regimes are significantly different

from scipy.stats import ttest_ind

# T-test for Train Set Returns
train_returns_regime_0 = train_df[train_df['Regime'] == 0]['Return']
train_returns_regime_1 = train_df[train_df['Regime'] == 1]['Return']
if not train_returns_regime_0.empty and not train_returns_regime_1.empty:
    t_stat, p_value = ttest_ind(train_returns_regime_0, train_returns_regime_1, equal_var=False, alternative='greater')
    print("\nT-test for Train Set Returns between Bull (Regime 0) and Bear (Regime 1):")
    print(f"  t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
else:
    print("\nT-test for Train Set Returns: Insufficient data for both regimes")

# T-test for Test Set Returnsalter
test_returns_regime_0 = test_df[test_df['Regime'] == 0]['Return']
test_returns_regime_1 = test_df[test_df['Regime'] == 1]['Return']
if not test_returns_regime_0.empty and not test_returns_regime_1.empty:
    t_stat, p_value = ttest_ind(test_returns_regime_0, test_returns_regime_1, equal_var=False, alternative='greater')
    print("\nT-test for Test Set Returns between Bull (Regime 0) and Bear (Regime 1):")
    print(f"  t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
else:
    print("\nT-test for Test Set Returns: Insufficient data for both regimes")

# Plot the distribution of returns for both Train and Test sets
# Assuming matplotlib is available; if not, please import it as: import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)

for ax, (df, title) in zip(axes, [(train_df, 'Train Set Returns'), (test_df, 'Test Set Returns')]):
    for regime, color in zip([0, 1], ['blue', 'red']):
        regime_name = "Bull" if regime == 0 else "Bear"
        returns = df[df['Regime'] == regime]['Return']
        if not returns.empty:
            ax.hist(returns, bins=30, alpha=0.5, label=f"{regime_name}", color=color, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
plt.show()


# %% 




# %%

# %%
from expectation.seqtest.sequential_e_testing import SequentialTesting

# %%

# Initialize a test for H0: μ = 0 vs H1: μ > 0
test = SequentialTesting(
    test_type="mean",
    null_value=0,
    alternative="greater"
)

# First batch of data
result1 = test.update([0.5, 1.2, 0.8])
print(f"After 3 observations:")
print(f"E-value: {result1.e_value:.2f}")
print(f"Reject null: {result1.reject_null}")

# More data arrives
result2 = test.update([1.5, 1.1])
print(f"\nAfter 5 observations:")
print(f"E-value: {result2.e_value:.2f}")
print(f"Cumulative e-value: {result2.e_process.cumulative_value:.2f}")
print(f"Reject null: {result2.reject_null}")


# %%
