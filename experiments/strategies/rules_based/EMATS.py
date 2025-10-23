# %% 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from utils.feature_loader import FeatureLoader

loader = FeatureLoader(start_date='2024-01-01', end_date='2025-10-19')
df = loader.build_feature_set(crypto_assets=['bitcoin'])
print(df.head())

# %%

# Params

ema_fast = 9
ema_slow = 21
vol_mult = 1.5

# %%
df.set_index('timestamp', inplace=True)
df['ema_fast'] = df['bitcoin_close'].ewm(span=ema_fast, adjust=False).mean()
df['ema_slow'] = df['bitcoin_close'].ewm(span=ema_slow, adjust=False).mean()
df['ema_signal'] = np.where(df['ema_fast'] >= df['ema_slow'], 1, -1)
df['ema_signal'] = df['ema_signal'].fillna(0)
print(df.head())

# %%

tr1 = df['bitcoin_high'] - df['bitcoin_low']
tr2 = (df['bitcoin_high'] - df['bitcoin_close'].shift()).abs()
tr3 = (df['bitcoin_low'] - df['bitcoin_close'].shift()).abs()
df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14, min_periods=1).mean()
print(df)  

# %%

df['long_atr'] = np.where(df['ema_signal'] == 1, df['bitcoin_close'] - df['atr'] * vol_mult, 0)
df['short_atr'] = np.where(df['ema_signal'] == -1, df['bitcoin_close'] + df['atr'] * vol_mult, 0)

# %%
df['long_trail_stop'] = np.where(df['ema_signal'] == 1, df['long_atr'].groupby(df['ema_signal'].ne(df['ema_signal'].shift()).cumsum()).transform('cummax'), 0)
df['short_trail_stop'] = np.where(df['ema_signal'] == -1, df['short_atr'].groupby(df['ema_signal'].ne(df['ema_signal'].shift()).cumsum()).transform('cummin'), 0)

# %%

df['long_stop'] = 0
df['short_stop'] = 0

# Initialize state variables for tracking stop conditions
long_stop_active = False
short_stop_active = False

# Iterate through the dataframe to apply the stop logic
for i in range(1, len(df)):
    # Check if ema_signal has changed, reset stops if it has
    if df['ema_signal'].iloc[i] != df['ema_signal'].iloc[i-1]:
        long_stop_active = False
        short_stop_active = False
    
    # Long stop logic
    if df['ema_signal'].iloc[i] == 1:
        if long_stop_active:
            df.loc[df.index[i], 'long_stop'] = 1
            # Check if close goes above long_trail_stop to deactivate stop
            if df['bitcoin_close'].iloc[i] > df['long_trail_stop'].iloc[i]:
                long_stop_active = False
        else:
            # Check if close drops below long_trail_stop to activate stop
            if df['bitcoin_close'].iloc[i] < df['long_trail_stop'].iloc[i]:
                long_stop_active = True
                df.loc[df.index[i], 'long_stop'] = 1
    
    # Short stop logic
    if df['ema_signal'].iloc[i] == -1:
        if short_stop_active:
            df.loc[df.index[i], 'short_stop'] = 1
            # Check if close goes below short_trail_stop to deactivate stop
            if df['bitcoin_close'].iloc[i] < df['short_trail_stop'].iloc[i]:
                short_stop_active = False
        else:
            # Check if close goes above short_trail_stop to activate stop
            if df['bitcoin_close'].iloc[i] > df['short_trail_stop'].iloc[i]:
                short_stop_active = True
                df.loc[df.index[i], 'short_stop'] = 1


# %%

df['signal'] = np.where((df['ema_signal'] == 1) & (df['long_stop'] != 1), 1, 
                       np.where((df['ema_signal'] == -1) & (df['short_stop'] != 1), -1, 0))

df['log_ret'] = np.log(df['bitcoin_close'] / df['bitcoin_close'].shift(1))
df['cum_log_ret'] = df['log_ret'].cumsum()

df['strategy_ret'] = df['signal'].shift(1) * df['log_ret']
df['cum_strategy_ret'] = df['strategy_ret'].cumsum()

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplot with 2 rows
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    subplot_titles=('Cumulative Returns', 'Signals'),
                    vertical_spacing=0.1)

# Plot cumulative strategy return
fig.add_trace(
    go.Scatter(x=df.index, y=df['cum_strategy_ret'], name='Strategy Return',
               line=dict(color='blue')),
    row=1, col=1
)

# Plot cumulative log return
fig.add_trace(
    go.Scatter(x=df.index, y=df['cum_log_ret'], name='Asset Return',
               line=dict(color='orange')),
    row=1, col=1
)

# Plot signals as markers
# Long signals (+1) as green up triangles
long_signals = df[df['signal'] == 1]
fig.add_trace(
    go.Scatter(x=long_signals.index, y=[0]*len(long_signals), name='Long Signal',
               mode='markers', marker=dict(symbol='triangle-up', color='green', size=8)),
    row=2, col=1
)

# Neutral signals (0) as yellow squares
neutral_signals = df[df['signal'] == 0]
fig.add_trace(
    go.Scatter(x=neutral_signals.index, y=[0]*len(neutral_signals), name='Neutral Signal',
               mode='markers', marker=dict(symbol='square', color='yellow', size=8)),
    row=2, col=1
)

# Short signals (-1) as red down triangles
short_signals = df[df['signal'] == -1]
fig.add_trace(
    go.Scatter(x=short_signals.index, y=[0]*len(short_signals), name='Short Signal',
               mode='markers', marker=dict(symbol='triangle-down', color='red', size=8)),
    row=2, col=1
)

# Update layout
fig.update_layout(
    height=600,
    width=1000,
    title_text="EMATS Strategy Performance and Signals",
    showlegend=True
)

# Update y-axis titles
fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
fig.update_yaxes(title_text="Signal", row=2, col=1, range=[-1.5, 1.5])

fig.show()

# %%
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots()

# Plot long_stop on the first y-axis
ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color='tab:blue')
ax1.plot(df.index, df['bitcoin_close'], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis for long_trail_stop
ax2 = ax1.twinx()
ax2.set_ylabel('Signal', color='tab:orange')
ax2.plot(df.index, df['signal'], color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Title and layout
plt.title('EMATS Strategy')
fig.tight_layout()
plt.show()

# %%
df
# %%
