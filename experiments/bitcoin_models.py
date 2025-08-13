# %%

########################################################
# Imports
########################################################

from config import setup_mlflow, create_experiment
import mlflow
import mlflow.sklearn
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List
from utils.evaluation_metrics import calculate_profit_factor, calculate_sharpe_ratio, calculate_max_drawdown, calculate_sortino_ratio, calculate_calmar_ratio
from utils.validation import in_sample_permutation_test
from utils.feature_loader import FeatureLoader
from utils.feature_engineering import calculate_log_returns, shift_features_for_prediction
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# %%

########################################################
# Setup
########################################################


# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup MLflow
setup_mlflow()
#create_experiment("bitcoin-models")
#mlflow.set_experiment("bitcoin-models")

# %%

########################################################
# Load Features
########################################################

# You can change this to any asset(s) supported by your FeatureLoader
assets = ['chainlink']  # e.g., ['ethereum'], ['aapl'], ['gold'], etc.

# Helper: get the asset prefix for column names (assume first asset for single-asset analysis)
asset = assets[0]
prefix = f"{asset}_"  # e.g., 'bitcoin_', 'ethereum_', etc.

# Initialize feature loader with global date range
loader = FeatureLoader(start_date='2020-01-01', end_date='2025-08-10')

# Build feature set with selected asset(s) + macro features
features_df = loader.build_feature_set(
    crypto_assets=assets,
    #fred_indicators={'creditSpreads': 'credit_spread'},
    #yahoo_tickers={'vix': 'vix'},
    #calculated_features={'rty_ym_ratio': 'rty_ym_ratio'}
)

print(f"Loaded {len(features_df)} rows of combined data")
print("\nColumns:", features_df.columns.tolist())
print("\nSample data:")
print(features_df.head(20))


# %%

########################################################
# Feature Exploration
########################################################

# Helper to get column names for the selected asset
def col(name):
    return f"{prefix}{name}"

features_df['ema_9'] = features_df[col('close')].ewm(span=9, adjust=False).mean()
features_df['ema_21'] = features_df[col('close')].ewm(span=21, adjust=False).mean()
features_df['ema_50'] = features_df[col('close')].ewm(span=50, adjust=False).mean()

# --- ADD OHLCV-BASED FEATURES ---

# 1. Price Position Within Daily Range (0-1, where close is in the day's range)
features_df['close_position'] = (features_df[col('close')] - features_df[col('low')]) / (features_df[col('high')] - features_df[col('low')])

# 2. Daily Range Volatility
features_df['daily_range'] = (features_df[col('high')] - features_df[col('low')]) / features_df[col('close')]
features_df['range_ema'] = features_df['daily_range'].ewm(span=21, adjust=False).mean()

# 3. Bullish/Bearish Candlestick Strength
features_df['body_size'] = abs(features_df[col('close')] - features_df[col('open')]) / features_df[col('open')]
features_df['is_bullish'] = (features_df[col('close')] > features_df[col('open')]).astype(int)
features_df['upper_wick'] = (features_df[col('high')] - features_df[[col('open'), col('close')]].max(axis=1)) / features_df[col('close')]
features_df['lower_wick'] = (features_df[[col('open'), col('close')]].min(axis=1) - features_df[col('low')]) / features_df[col('close')]

# 4. Volume Confirmation
features_df['volume_ema'] = features_df[col('total_volume')].ewm(span=21, adjust=False).mean()
features_df['volume_ratio'] = features_df[col('total_volume')] / features_df['volume_ema']

# 5. Price Momentum using High/Low
features_df['high_momentum'] = features_df[col('high')].pct_change(9)
features_df['low_momentum'] = features_df[col('low')].pct_change(9)

# --- KEEP ORIGINAL EXHAUSTIVE REGIME DEFINITIONS (NO NEUTRALS) ---

features_df['uptrend'] = np.where(
    (features_df['ema_9'] >= features_df['ema_21']) & 
    (features_df['ema_21'] >= features_df['ema_50']),
    1, 0
)

features_df['slowdown'] = np.where(
    (features_df['ema_9'] < features_df['ema_21']) & 
    (features_df['ema_21'] >= features_df['ema_50']),
    1, 0
)

features_df['recovery'] = np.where(
    (features_df['ema_9'] >= features_df['ema_21']) & 
    (features_df['ema_21'] < features_df['ema_50']),
    1, 0
)

features_df['downtrend'] = np.where(
    (features_df['ema_9'] < features_df['ema_21']) & 
    (features_df['ema_21'] < features_df['ema_50']),
    1, 0
)

# Helper function to get current regime
def get_current_regime(row):
    if row['uptrend'] == 1:
        return 'uptrend'
    elif row['downtrend'] == 1:
        return 'downtrend'
    elif row['slowdown'] == 1:
        return 'slowdown'
    elif row['recovery'] == 1:
        return 'recovery'
    else:
        return 'neutral'  # This should never happen with exhaustive definitions

# Method 1: Distance-based confidence (ENHANCED with OHLCV)
def distance_based_confidence(row):
    ema9 = row['ema_9']
    ema21 = row['ema_21']
    ema50 = row['ema_50']
    
    # Original EMA distances
    dist_9_21 = abs((ema9 - ema21) / ema21) * 100
    dist_21_50 = abs((ema21 - ema50) / ema50) * 100
    dist_9_50 = abs((ema9 - ema50) / ema50) * 100
    
    # Base confidence from EMA distances
    avg_distance = (dist_9_21 + dist_21_50 + dist_9_50) / 3
    base_confidence = 100 * (1 - np.exp(-avg_distance / 5))
    
    # OHLCV quality factors (reduce confidence if they don't align)
    current_regime = get_current_regime(row)
    quality_multiplier = 1.0
    
    if current_regime == 'uptrend':
        # Reduce confidence if closing weak or no higher lows
        if row['close_position'] < 0.6:
            quality_multiplier *= 0.8
        if row['low_momentum'] <= 0:
            quality_multiplier *= 0.8
            
    elif current_regime == 'downtrend':
        # Reduce confidence if closing strong or no lower highs
        if row['close_position'] > 0.4:
            quality_multiplier *= 0.8
        if row['high_momentum'] >= 0:
            quality_multiplier *= 0.8
            
    elif current_regime == 'slowdown':
        # Reduce confidence if closing strong or making higher highs
        if row['close_position'] > 0.5:
            quality_multiplier *= 0.9
        if row['high_momentum'] >= 0:
            quality_multiplier *= 0.9
            
    elif current_regime == 'recovery':
        # Reduce confidence if closing weak or no volume
        if row['close_position'] < 0.7:
            quality_multiplier *= 0.85
        if row['volume_ratio'] < 1.2:
            quality_multiplier *= 0.85
    
    confidence = base_confidence * quality_multiplier
    return min(confidence, 100)

# Method 2: Slope consistency confidence (ENHANCED)
def slope_consistency_confidence(df, idx, window=20):
    if idx < window:
        return 50
        
    # Original EMA slopes
    ema9_slope = (df.iloc[idx]['ema_9'] - df.iloc[idx-window]['ema_9']) / df.iloc[idx-window]['ema_9']
    ema21_slope = (df.iloc[idx]['ema_21'] - df.iloc[idx-window]['ema_21']) / df.iloc[idx-window]['ema_21']
    ema50_slope = (df.iloc[idx]['ema_50'] - df.iloc[idx-window]['ema_50']) / df.iloc[idx-window]['ema_50']
    
    # Add high/low slopes for trend strength
    high_slope = (df.iloc[idx][col('high')] - df.iloc[idx-window][col('high')]) / df.iloc[idx-window][col('high')]
    low_slope = (df.iloc[idx][col('low')] - df.iloc[idx-window][col('low')]) / df.iloc[idx-window][col('low')]
    
    current_regime = get_current_regime(df.iloc[idx])
    
    if current_regime == 'uptrend':
        # Base score from EMAs
        base_score = (
            (ema9_slope > 0) * 30 + 
            (ema21_slope > 0) * 30 + 
            (ema50_slope > 0) * 30 +
            (ema9_slope > ema21_slope > ema50_slope) * 10
        )
        # Bonus for OHLCV alignment
        ohlcv_bonus = (
            (low_slope > 0) * 10 +      # Higher lows
            (high_slope > 0) * 10       # Higher highs
        )
        slope_score = base_score + ohlcv_bonus
        
    elif current_regime == 'downtrend':
        base_score = (
            (ema9_slope < 0) * 30 + 
            (ema21_slope < 0) * 30 + 
            (ema50_slope < 0) * 30 +
            (ema9_slope < ema21_slope < ema50_slope) * 10
        )
        ohlcv_bonus = (
            (high_slope < 0) * 10 +     # Lower highs
            (low_slope < 0) * 10        # Lower lows
        )
        slope_score = base_score + ohlcv_bonus
        
    elif current_regime == 'slowdown':
        base_score = (
            (ema9_slope < ema21_slope) * 50 +
            (ema21_slope > 0) * 25 +
            (ema50_slope > 0) * 25
        )
        ohlcv_penalty = (high_slope >= 0) * -10  # Penalty if still making highs
        slope_score = base_score + ohlcv_penalty
        
    else:  # recovery
        base_score = (
            (ema9_slope > ema21_slope) * 50 +
            (ema21_slope < 0) * 25 +
            (ema50_slope < 0) * 25
        )
        ohlcv_bonus = (low_slope > 0) * 10  # Bonus for higher lows
        slope_score = base_score + ohlcv_bonus
        
    return max(0, min(slope_score, 100))

# Method 3: Fuzzy membership confidence (ENHANCED)
def fuzzy_membership_confidence(row):
    ema9 = row['ema_9']
    ema21 = row['ema_21']
    ema50 = row['ema_50']
    
    # Calculate membership scores for each regime
    memberships = {}
    
    # Get current regime
    current_regime = get_current_regime(row)
    
    # Base membership from EMA configuration
    if current_regime == 'uptrend':
        spread = (ema9 - ema50) / ema50
        base_membership = min(100, spread * 1000)
        # Enhance with OHLCV
        close_factor = max(0.5, row['close_position'])  # 0.5 to 1.0
        volume_factor = min(1.5, max(0.5, row['volume_ratio'])) / 1.5  # 0.33 to 1.0
        memberships['uptrend'] = base_membership * close_factor * volume_factor
        
    elif current_regime == 'downtrend':
        spread = (ema50 - ema9) / ema50
        base_membership = min(100, spread * 1000)
        # Enhance with OHLCV
        close_factor = max(0.5, 1 - row['close_position'])  # 0.5 to 1.0
        volume_factor = min(1.5, max(0.5, row['volume_ratio'])) / 1.5
        memberships['downtrend'] = base_membership * close_factor * volume_factor
        
    elif current_regime == 'slowdown':
        cross_depth = (ema21 - ema9) / ema21
        trend_strength = (ema21 - ema50) / ema50
        base_membership = min(100, (cross_depth + trend_strength) * 500)
        # Enhance with volatility
        range_factor = min(2, max(0.5, row['daily_range'] / row['range_ema']))
        memberships['slowdown'] = base_membership * (0.5 + 0.5 * range_factor)
        
    elif current_regime == 'recovery':
        cross_strength = (ema9 - ema21) / ema21
        downtrend_depth = (ema50 - ema21) / ema50
        base_membership = min(100, (cross_strength + downtrend_depth) * 500)
        # Enhance with volume
        volume_boost = min(2, max(0.5, row['volume_ratio']))
        memberships['recovery'] = base_membership * (0.5 + 0.5 * volume_boost)
    
    return memberships.get(current_regime, 50)

# Apply all confidence calculations
print("\nCalculating confidence scores...")

# Add regime column
features_df['regime'] = features_df.apply(get_current_regime, axis=1)

# Distance-based confidence
features_df['confidence_distance'] = features_df.apply(distance_based_confidence, axis=1)

# Slope consistency confidence
features_df['confidence_slope'] = [slope_consistency_confidence(features_df, i) for i in range(len(features_df))]

# Fuzzy membership confidence
features_df['confidence_fuzzy'] = features_df.apply(fuzzy_membership_confidence, axis=1)

# Ensemble confidence (weighted average)
features_df['confidence_ensemble'] = (
    features_df['confidence_distance'] * 0.4 +
    features_df['confidence_slope'] * 0.3 +
    features_df['confidence_fuzzy'] * 0.3
)

# Calculate regime probabilities (ML-style output)
def calculate_regime_probabilities(row):
    ema9 = row['ema_9']
    ema21 = row['ema_21']
    ema50 = row['ema_50']
    
    # Calculate raw scores for each regime WITH OHLCV modifiers
    scores = {}
    
    # Uptrend score
    if ema9 >= ema21 and ema21 >= ema50:
        base_score = (ema9/ema21 - 1) + (ema21/ema50 - 1)
        # Boost score if OHLCV aligns
        ohlcv_modifier = 1.0
        if row['close_position'] > 0.6:
            ohlcv_modifier += 0.2
        if row['low_momentum'] > 0:
            ohlcv_modifier += 0.2
        scores['uptrend'] = base_score * ohlcv_modifier
    else:
        scores['uptrend'] = -abs(ema9/ema21 - 1) - abs(ema21/ema50 - 1)
    
    # Downtrend score
    if ema9 <= ema21 and ema21 <= ema50:
        base_score = (ema21/ema9 - 1) + (ema50/ema21 - 1)
        ohlcv_modifier = 1.0
        if row['close_position'] < 0.4:
            ohlcv_modifier += 0.2
        if row['high_momentum'] < 0:
            ohlcv_modifier += 0.2
        scores['downtrend'] = base_score * ohlcv_modifier
    else:
        scores['downtrend'] = -abs(ema21/ema9 - 1) - abs(ema50/ema21 - 1)
    
    # Slowdown score
    if ema9 < ema21 and ema21 >= ema50:
        base_score = (ema21/ema9 - 1) + (ema21/ema50 - 1)
        ohlcv_modifier = 1.0
        if row['close_position'] < 0.5:
            ohlcv_modifier += 0.1
        if row['high_momentum'] < 0:
            ohlcv_modifier += 0.1
        scores['slowdown'] = base_score * ohlcv_modifier
    else:
        scores['slowdown'] = -abs(ema21/ema9 - 1) - abs(ema21/ema50 - 1)
    
    # Recovery score
    if ema9 >= ema21 and ema21 < ema50:
        base_score = (ema9/ema21 - 1) + (ema50/ema21 - 1)
        ohlcv_modifier = 1.0
        if row['close_position'] > 0.7:
            ohlcv_modifier += 0.15
        if row['volume_ratio'] > 1.2:
            ohlcv_modifier += 0.15
        scores['recovery'] = base_score * ohlcv_modifier
    else:
        scores['recovery'] = -abs(ema9/ema21 - 1) - abs(ema50/ema21 - 1)
    
    # Apply softmax-like transformation
    exp_scores = {k: np.exp(v * 10) for k, v in scores.items()}
    total = sum(exp_scores.values())
    
    probabilities = {k: v/total for k, v in exp_scores.items()}
    
    return probabilities

# Add probability columns
probs = features_df.apply(calculate_regime_probabilities, axis=1)
features_df['prob_uptrend'] = [p['uptrend'] for p in probs]
features_df['prob_downtrend'] = [p['downtrend'] for p in probs]
features_df['prob_slowdown'] = [p['slowdown'] for p in probs]
features_df['prob_recovery'] = [p['recovery'] for p in probs]

import pandas as pd
from scipy.stats import skew, kurtosis

# Calculate returns if not already present
if 'return' not in features_df.columns:
    features_df['return'] = features_df[col('close')].pct_change()

print("\nStatistical characteristics of returns for each regime:")
returns_stats = features_df.groupby('regime')['return'].agg(
    ['count', 'mean', 'std', 'min', 'max']
)
returns_stats['median'] = features_df.groupby('regime')['return'].median()
returns_stats['skew'] = features_df.groupby('regime')['return'].apply(lambda x: skew(x.dropna()))
returns_stats['kurtosis'] = features_df.groupby('regime')['return'].apply(lambda x: kurtosis(x.dropna()))

# Profit factor using 90% quantile (5% to 95%)
def profit_factor_90(x):
    x = x.dropna()
    lower = x.quantile(0.05)
    upper = x.quantile(0.95)
    filtered = x[(x >= lower) & (x <= upper)]
    gross_profit = filtered[filtered > 0].sum()
    gross_loss = -filtered[filtered < 0].sum()
    if gross_loss == 0:
        return float('inf')
    return gross_profit / gross_loss

returns_stats['profit_factor_90pct'] = features_df.groupby('regime')['return'].apply(profit_factor_90)
print(returns_stats)

print("\nRegime counts:")
print(features_df['regime'].value_counts())


# %%

import plotly.graph_objs as go

# Ensure we have a timestamp column for x-axis
if 'timestamp' in features_df.columns:
    x_vals = features_df['timestamp']
else:
    x_vals = features_df.index

# --- Main Price Trace ---
price_trace = go.Scatter(
    x=x_vals,
    y=features_df[col('close')],
    mode='lines',
    name=f"{asset.capitalize()} Close",
    line=dict(color='black', width=2),
    yaxis='y1',
    hovertemplate=f"<b>{asset.capitalize()} Close</b><br>Price: %{{y}}<br>Time: %{{x}}<extra></extra>"
)

# --- Regime Probability Area Traces (stacked, on secondary y-axis) ---
# Use more vivid, saturated colors for each regime
prob_cols = [
    ('prob_uptrend', 'Uptrend', 'rgba(0, 180, 0, 0.35)', 'solid'),         # Bright green
    ('prob_downtrend', 'Downtrend', 'rgba(220, 0, 0, 0.35)', 'solid'),     # Bright red
    ('prob_slowdown', 'Slowdown', 'rgba(255, 170, 0, 0.32)', 'solid'),     # Bright orange
    ('prob_recovery', 'Recovery', 'rgba(0, 70, 255, 0.32)', 'solid')       # Bright blue
]

area_traces = []
for i, (colname, name, color, dash) in enumerate(prob_cols):
    # Custom hover text: show regime and probability, formatted nicely, and remove extra box
    hover_text = (
        f"<b>Regime:</b> {name}<br>"
        f"<b>Probability:</b> %{{y:.2%}}<br>"
        f"<b>Time:</b> %{{x}}<extra></extra>"
    )
    area_traces.append(
        go.Scatter(
            x=x_vals,
            y=features_df[colname],
            mode='lines',
            name=name,
            line=dict(width=2, color=color, dash=dash),
            fill='tozeroy' if i == 0 else 'tonexty',
            stackgroup='one',
            yaxis='y2',
            hovertemplate=hover_text,
            showlegend=True
        )
    )

# --- Compose Figure ---
fig = go.Figure()

# Add regime probability areas first (so they're in the background)
for trace in area_traces:
    fig.add_trace(trace)

# Add price trace on top
fig.add_trace(price_trace)

# --- Layout ---
fig.update_layout(
    title=f'{asset.capitalize()} Close Price with Trend Regime Probabilities',
    xaxis=dict(title='Timestamp'),
    yaxis=dict(
        title=f'{asset.capitalize()} Close Price',
        side='right',
        showgrid=True,
        zeroline=False
    ),
    yaxis2=dict(
        title='Regime Probability',
        overlaying='y',
        side='left',
        range=[0, 1],
        showgrid=False,
        zeroline=False
    ),
    legend=dict(x=0.01, y=0.99),
    height=600,
    width=1100,
    template='plotly_white',
    margin=dict(t=60, b=40),
    plot_bgcolor='rgba(245,245,245,1)'
)

fig.show()

# %%

features_df.tail(20)



# %%

########################################################
# Feature Engineering Functions
########################################################

calculate_log_returns(features_df, periods=[1, 5])




# %%

# %%

########################################################
# Model Training and In-Sample Evaluation 
########################################################




# %%

########################################################
# In-Sample Permutation Test
########################################################




# %%

########################################################
# Out-of-Sample Walk-Forward Validation
########################################################


# %%

########################################################
# Out-of-Sample Permuted Walk-Forward Test
######################################################## 





# %%
# Main Pipeline Execution

def run_training_pipeline(features_df: pd.DataFrame,
                         model_class,
                         model_params: Dict = {},
                         feature_params: Dict = {},
                         validation_params: Dict = {}) -> Dict:
    """
    Run the complete training and testing pipeline.
    """
    
    with mlflow.start_run(run_name="bitcoin-training-pipeline"):
        
        # Log initial parameters
        mlflow.log_params({
            'model_class': str(model_class),
            'feature_params': str(feature_params),
            'validation_params': str(validation_params)
        })
        
        # 1. Feature Engineering
        print("Step 1: Feature Engineering...")
        
        # Get asset name (default to bitcoin)
        asset = feature_params.get('asset', 'bitcoin')
        
        # Calculate log returns for all assets
        features_df = calculate_log_returns(
            features_df,
            assets=feature_params.get('assets', None),  # Auto-detect if not provided
            periods=feature_params.get('return_periods', [1, 5, 20])
        )
        
        # Create target variable
        features_df, target_col = create_target_variable(
            features_df,
            asset=asset,
            prediction_horizon=feature_params.get('prediction_horizon', 1)
        )
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        # Define feature columns (exclude timestamp and target)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['timestamp', target_col]]
        
        # Shift features to prevent look-ahead bias
        features_df = shift_features_for_prediction(
            features_df,
            feature_cols,
            target_col,
            shift_periods=1
        )
        
        # Remove NaN values again after shifting
        features_df = features_df.dropna()
        
        mlflow.log_metric("n_features", len(feature_cols))
        mlflow.log_metric("n_samples", len(features_df))
        
        # 2. Train-Test Split
        print("Step 2: Train-Test Split...")
        
        train_size = validation_params.get('train_size', 0.7)
        split_index = int(len(features_df) * train_size)
        
        train_df = features_df.iloc[:split_index]
        test_df = features_df.iloc[split_index:]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        # 3. Model Training
        print("Step 3: Model Training...")
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # 4. Calculate Metrics
        print("Step 4: Calculating Metrics...")
        
        # Training metrics
        train_sharpe = calculate_sharpe_ratio(y_train.values)
        train_pf = calculate_profit_factor(y_train.values, np.sign(train_pred))
        
        mlflow.log_metric("train_sharpe", train_sharpe)
        mlflow.log_metric("train_profit_factor", train_pf)
        
        # 5. Permutation Testing
        print("Step 5: Permutation Testing...")
        
        perm_results = permutation_test(
            y_train.values,
            train_pred,
            n_permutations=validation_params.get('n_permutations', 1000),
            metric='sharpe'
        )
        
        mlflow.log_metric("permutation_p_value", perm_results['p_value'])
        mlflow.log_metric("permutation_passes", int(perm_results['passes_test']))
        
        print(f"Permutation test p-value: {perm_results['p_value']:.4f}")
        print(f"Strategy {'PASSES' if perm_results['passes_test'] else 'FAILS'} permutation test")
        
        # 6. Walk-Forward Validation
        print("Step 6: Walk-Forward Validation...")
        
        wf_params = {
            'initial_train_size': validation_params.get('initial_train_size', 252),
            'test_size': validation_params.get('test_size', 63),
            'retrain_frequency': validation_params.get('retrain_frequency', 20)
        }
        
        wf_results = walk_forward_validation(
            X_train, y_train, model, **wf_params
        )
        
        mlflow.log_metric("wf_sharpe", wf_results['overall_sharpe'])
        mlflow.log_metric("wf_profit_factor", wf_results['overall_profit_factor'])
        
        # 7. Permuted Walk-Forward Test
        print("Step 7: Permuted Walk-Forward Testing...")
        
        pwf_results = permuted_walk_forward_test(
            X_train, y_train, model, wf_params,
            n_permutations=validation_params.get('pwf_permutations', 100),
            p_value_threshold=validation_params.get('pwf_threshold', 0.10)
        )
        
        mlflow.log_metric("pwf_p_value", pwf_results['p_value'])
        mlflow.log_metric("pwf_passes", int(pwf_results['passes_test']))
        
        print(f"Permuted WF test p-value: {pwf_results['p_value']:.4f}")
        print(f"Strategy {'PASSES' if pwf_results['passes_test'] else 'FAILS'} permuted WF test")
        
        # 8. Final Test Set Evaluation
        print("Step 8: Test Set Evaluation...")
        
        test_sharpe = calculate_sharpe_ratio(y_test.values)
        test_pf = calculate_profit_factor(y_test.values, np.sign(test_pred))
        
        mlflow.log_metric("test_sharpe", test_sharpe)
        mlflow.log_metric("test_profit_factor", test_pf)
        
        # Performance degradation check
        performance_ratio = test_sharpe / train_sharpe if train_sharpe != 0 else 0
        mlflow.log_metric("performance_ratio", performance_ratio)
        
        print(f"Test Sharpe: {test_sharpe:.4f}")
        print(f"Test Profit Factor: {test_pf:.4f}")
        print(f"Performance Ratio (test/train): {performance_ratio:.2f}")
        
        # 9. Save Model for Serving
        print("Step 9: Saving Model for Serving...")
        
        # Log the model with MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="bitcoin_trading_model",
            signature=mlflow.models.infer_signature(X_train, train_pred)
        )
        
        # Save feature columns for serving
        mlflow.log_dict({"feature_columns": feature_cols}, "feature_columns.json")
        
        # Save preprocessing parameters
        preprocessing_params = {
            'return_periods': feature_params.get('return_periods', [1, 5, 20]),
            'prediction_horizon': feature_params.get('prediction_horizon', 1),
            'target_type': feature_params.get('target_type', 'returns'),
            'shift_periods': 1
        }
        mlflow.log_dict(preprocessing_params, "preprocessing_params.json")
        
        # Create summary report
        summary = {
            'train_metrics': {
                'sharpe': train_sharpe,
                'profit_factor': train_pf,
                'samples': len(X_train)
            },
            'test_metrics': {
                'sharpe': test_sharpe,
                'profit_factor': test_pf,
                'samples': len(X_test)
            },
            'validation': {
                'permutation_p_value': perm_results['p_value'],
                'permutation_passes': perm_results['passes_test'],
                'wf_sharpe': wf_results['overall_sharpe'],
                'pwf_p_value': pwf_results['p_value'],
                'pwf_passes': pwf_results['passes_test']
            },
            'model_info': {
                'class': str(model_class),
                'params': model_params,
                'n_features': len(feature_cols)
            }
        }
        
        mlflow.log_dict(summary, "model_summary.json")
        
        print("\n" + "="*50)
        print("✅ Training Pipeline Complete!")
        print("="*50)
        print(f"Model logged to MLflow with name: bitcoin_trading_model")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        return summary


# %%
# Example Usage with a Simple Model

if __name__ == "__main__":
    
    # Example with a simple linear model
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    
    # Define parameters
    feature_params = {
        'asset': 'bitcoin',  # Primary asset for prediction
        'assets': None,  # Auto-detect all assets in the data
        'return_periods': [1, 5, 20],
        'prediction_horizon': 1
    }
    
    validation_params = {
        'train_size': 0.7,
        'n_permutations': 1000,
        'initial_train_size': 252,  # 1 year
        'test_size': 63,  # 3 months
        'retrain_frequency': 20,  # Retrain every month
        'pwf_permutations': 100,
        'pwf_threshold': 0.10
    }
    
    # Run with Linear Regression
    print("Running pipeline with Linear Regression...")
    lr_summary = run_training_pipeline(
        features_df,
        LinearRegression,
        model_params={},
        feature_params=feature_params,
        validation_params=validation_params
    )
    
    # You can also run with other models
    # print("\nRunning pipeline with Random Forest...")
    # rf_summary = run_training_pipeline(
    #     features_df,
    #     RandomForestRegressor,
    #     model_params={'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
    #     feature_params=feature_params,
    #     validation_params=validation_params
    # )


# %%
# Model Serving Preparation

def prepare_model_for_serving(run_id: str) -> Dict:
    """
    Prepare a trained model for serving by loading it and its preprocessing params.
    """
    client = mlflow.tracking.MlflowClient()
    
    # Load model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Load preprocessing parameters
    preprocessing_params = client.download_artifacts(run_id, "preprocessing_params.json")
    feature_columns = client.download_artifacts(run_id, "feature_columns.json")
    
    return {
        'model': model,
        'preprocessing_params': preprocessing_params,
        'feature_columns': feature_columns,
        'run_id': run_id
    }


def serve_prediction(model_dict: Dict, new_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on new data using a served model.
    """
    model = model_dict['model']
    feature_cols = model_dict['feature_columns']
    
    # Apply preprocessing
    # In production, you'd apply the same preprocessing as training
    processed_data = new_data[feature_cols]
    
    # Make predictions
    predictions = model.predict(processed_data)
    
    return predictions