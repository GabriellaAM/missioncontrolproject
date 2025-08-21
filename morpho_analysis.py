import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# MORPHO OHLCV data from Gate.io (last 100 hours)
morpho_data = [
    {"time_period_start": "2025-08-21T13:00:00.0000000Z", "price_open": 2.3357, "price_high": 2.3542, "price_low": 2.3252, "price_close": 2.34175, "volume_traded": 69616.49999999997},
    {"time_period_start": "2025-08-21T12:00:00.0000000Z", "price_open": 2.3654, "price_high": 2.3841, "price_low": 2.3234, "price_close": 2.3358, "volume_traded": 402284.65999999986},
    {"time_period_start": "2025-08-21T11:00:00.0000000Z", "price_open": 2.3727, "price_high": 2.3852, "price_low": 2.3465, "price_close": 2.3665, "volume_traded": 280227.8499999999},
    {"time_period_start": "2025-08-21T10:00:00.0000000Z", "price_open": 2.3484, "price_high": 2.3779, "price_low": 2.329, "price_close": 2.37095, "volume_traded": 276676.08999999985},
    {"time_period_start": "2025-08-21T09:00:00.0000000Z", "price_open": 2.3757, "price_high": 2.3773, "price_low": 2.3147, "price_close": 2.34895, "volume_traded": 293986.0600000002},
    {"time_period_start": "2025-08-21T08:00:00.0000000Z", "price_open": 2.3675, "price_high": 2.3952, "price_low": 2.33375, "price_close": 2.3421, "volume_traded": 288770.52},
    {"time_period_start": "2025-08-21T07:00:00.0000000Z", "price_open": 2.352, "price_high": 2.38345, "price_low": 2.3195, "price_close": 2.37285, "volume_traded": 151794.4199999999},
    {"time_period_start": "2025-08-21T06:00:00.0000000Z", "price_open": 2.3385, "price_high": 2.3548, "price_low": 2.3175, "price_close": 2.3548, "volume_traded": 156524.60999999984},
    {"time_period_start": "2025-08-21T05:00:00.0000000Z", "price_open": 2.31465, "price_high": 2.3554, "price_low": 2.306, "price_close": 2.3323, "volume_traded": 164840.02999999985},
    {"time_period_start": "2025-08-21T04:00:00.0000000Z", "price_open": 2.30385, "price_high": 2.3412, "price_low": 2.29335, "price_close": 2.3216, "volume_traded": 305415.92000000016},
    {"time_period_start": "2025-08-21T03:00:00.0000000Z", "price_open": 2.2621, "price_high": 2.3074, "price_low": 2.2613, "price_close": 2.2965, "volume_traded": 324283.7699999999},
    {"time_period_start": "2025-08-21T02:00:00.0000000Z", "price_open": 2.2943, "price_high": 2.3037, "price_low": 2.2493, "price_close": 2.2621, "volume_traded": 123373.81999999977},
    {"time_period_start": "2025-08-21T01:00:00.0000000Z", "price_open": 2.262, "price_high": 2.3024, "price_low": 2.251, "price_close": 2.27555, "volume_traded": 132048.80999999985},
    {"time_period_start": "2025-08-21T00:00:00.0000000Z", "price_open": 2.2446, "price_high": 2.283, "price_low": 2.2345, "price_close": 2.26195, "volume_traded": 189316.61000000004},
    {"time_period_start": "2025-08-20T23:00:00.0000000Z", "price_open": 2.25705, "price_high": 2.2643, "price_low": 2.2334, "price_close": 2.2524, "volume_traded": 134532.81999999975},
    {"time_period_start": "2025-08-20T22:00:00.0000000Z", "price_open": 2.23825, "price_high": 2.2632, "price_low": 2.2271, "price_close": 2.2547, "volume_traded": 170449.32999999996},
    {"time_period_start": "2025-08-20T21:00:00.0000000Z", "price_open": 2.23205, "price_high": 2.24345, "price_low": 2.21605, "price_close": 2.23915, "volume_traded": 103963.08999999994},
    {"time_period_start": "2025-08-20T20:00:00.0000000Z", "price_open": 2.19155, "price_high": 2.2383, "price_low": 2.1851, "price_close": 2.2249, "volume_traded": 142206.5099999999},
    {"time_period_start": "2025-08-20T19:00:00.0000000Z", "price_open": 2.175, "price_high": 2.2091, "price_low": 2.1718, "price_close": 2.1924, "volume_traded": 118790.71999999977},
    {"time_period_start": "2025-08-20T18:00:00.0000000Z", "price_open": 2.1788, "price_high": 2.2141, "price_low": 2.15215, "price_close": 2.1853, "volume_traded": 120237.86999999997},
    {"time_period_start": "2025-08-20T17:00:00.0000000Z", "price_open": 2.18395, "price_high": 2.217, "price_low": 2.16915, "price_close": 2.18095, "volume_traded": 111290.53999999992},
    {"time_period_start": "2025-08-20T16:00:00.0000000Z", "price_open": 2.14215, "price_high": 2.1851, "price_low": 2.1246, "price_close": 2.18385, "volume_traded": 261133.63},
    {"time_period_start": "2025-08-20T15:00:00.0000000Z", "price_open": 2.14125, "price_high": 2.1667, "price_low": 2.1184, "price_close": 2.13855, "volume_traded": 141398.0899999999},
    {"time_period_start": "2025-08-20T14:00:00.0000000Z", "price_open": 2.0715, "price_high": 2.149, "price_low": 2.07015, "price_close": 2.1414, "volume_traded": 155365.23999999985},
    {"time_period_start": "2025-08-20T13:00:00.0000000Z", "price_open": 2.0964, "price_high": 2.1227, "price_low": 2.06075, "price_close": 2.0715, "volume_traded": 120675.05999999998},
    {"time_period_start": "2025-08-20T12:00:00.0000000Z", "price_open": 2.1059, "price_high": 2.1305, "price_low": 2.09135, "price_close": 2.09595, "volume_traded": 250061.05999999997},
    {"time_period_start": "2025-08-20T11:00:00.0000000Z", "price_open": 2.1133, "price_high": 2.1266, "price_low": 2.0753, "price_close": 2.1059, "volume_traded": 199705.13},
    {"time_period_start": "2025-08-20T10:00:00.0000000Z", "price_open": 2.09675, "price_high": 2.1229, "price_low": 2.0938, "price_close": 2.10645, "volume_traded": 104970.43999999984},
    {"time_period_start": "2025-08-20T09:00:00.0000000Z", "price_open": 2.10585, "price_high": 2.1222, "price_low": 2.0981, "price_close": 2.09845, "volume_traded": 107102.38999999988},
    {"time_period_start": "2025-08-20T08:00:00.0000000Z", "price_open": 2.0873, "price_high": 2.1207, "price_low": 2.0791, "price_close": 2.10585, "volume_traded": 137586.28999999995},
    {"time_period_start": "2025-08-20T07:00:00.0000000Z", "price_open": 2.10005, "price_high": 2.111, "price_low": 2.07915, "price_close": 2.0885, "volume_traded": 155194.83},
    {"time_period_start": "2025-08-20T06:00:00.0000000Z", "price_open": 2.0784, "price_high": 2.1024, "price_low": 2.06075, "price_close": 2.10075, "volume_traded": 109048.36999999994},
    {"time_period_start": "2025-08-20T05:00:00.0000000Z", "price_open": 2.0625, "price_high": 2.07835, "price_low": 2.0504, "price_close": 2.0782, "volume_traded": 92756.48000000001},
    {"time_period_start": "2025-08-20T04:00:00.0000000Z", "price_open": 2.0457, "price_high": 2.0713, "price_low": 2.0456, "price_close": 2.0624, "volume_traded": 177765.9899999999},
    {"time_period_start": "2025-08-20T03:00:00.0000000Z", "price_open": 2.034, "price_high": 2.0608, "price_low": 2.0315, "price_close": 2.05595, "volume_traded": 116759.34999999983},
    {"time_period_start": "2025-08-20T02:00:00.0000000Z", "price_open": 2.0284, "price_high": 2.04775, "price_low": 2.0187, "price_close": 2.03445, "volume_traded": 101785.00999999986},
    {"time_period_start": "2025-08-20T01:00:00.0000000Z", "price_open": 2.0136, "price_high": 2.032, "price_low": 1.9968, "price_close": 2.02845, "volume_traded": 133997.41999999987},
    {"time_period_start": "2025-08-20T00:00:00.0000000Z", "price_open": 1.9999, "price_high": 2.03045, "price_low": 1.9889, "price_close": 2.0195, "volume_traded": 206307.8000000001},
    {"time_period_start": "2025-08-19T23:00:00.0000000Z", "price_open": 2.02225, "price_high": 2.0435, "price_low": 1.9846, "price_close": 1.9999, "volume_traded": 124928.3999999999},
    {"time_period_start": "2025-08-19T22:00:00.0000000Z", "price_open": 2.04645, "price_high": 2.0557, "price_low": 2.02, "price_close": 2.02185, "volume_traded": 106326.60999999991},
    {"time_period_start": "2025-08-19T21:00:00.0000000Z", "price_open": 2.0526, "price_high": 2.0623, "price_low": 2.0398, "price_close": 2.0468, "volume_traded": 80913.11999999988},
    {"time_period_start": "2025-08-19T20:00:00.0000000Z", "price_open": 2.03265, "price_high": 2.05525, "price_low": 2.0231, "price_close": 2.04275, "volume_traded": 130051.00000000003},
    {"time_period_start": "2025-08-19T19:00:00.0000000Z", "price_open": 2.0244, "price_high": 2.0655, "price_low": 2.0185, "price_close": 2.0342, "volume_traded": 142025.49999999988},
    {"time_period_start": "2025-08-19T18:00:00.0000000Z", "price_open": 2.0631, "price_high": 2.0699, "price_low": 2.02185, "price_close": 2.035, "volume_traded": 239851.26000000024},
    {"time_period_start": "2025-08-19T17:00:00.0000000Z", "price_open": 2.0422, "price_high": 2.07165, "price_low": 2.0299, "price_close": 2.063, "volume_traded": 115901.34999999983},
    {"time_period_start": "2025-08-19T16:00:00.0000000Z", "price_open": 2.0528, "price_high": 2.0778, "price_low": 2.029, "price_close": 2.04245, "volume_traded": 147054.69999999992},
    {"time_period_start": "2025-08-19T15:00:00.0000000Z", "price_open": 2.04705, "price_high": 2.0772, "price_low": 2.0335, "price_close": 2.0617, "volume_traded": 151340.41999999978},
    {"time_period_start": "2025-08-19T14:00:00.0000000Z", "price_open": 2.0912, "price_high": 2.106, "price_low": 2.03395, "price_close": 2.0419, "volume_traded": 146438.52999999982},
    {"time_period_start": "2025-08-19T13:00:00.0000000Z", "price_open": 2.0553, "price_high": 2.11375, "price_low": 2.0471, "price_close": 2.0912, "volume_traded": 142374.31999999998},
    {"time_period_start": "2025-08-19T12:00:00.0000000Z", "price_open": 2.0508, "price_high": 2.0648, "price_low": 2.0445, "price_close": 2.0569, "volume_traded": 214604.88999999996},
    {"time_period_start": "2025-08-19T11:00:00.0000000Z", "price_open": 2.0358, "price_high": 2.05845, "price_low": 2.0269, "price_close": 2.05695, "volume_traded": 151283.5599999999},
    {"time_period_start": "2025-08-19T10:00:00.0000000Z", "price_open": 2.0312, "price_high": 2.043, "price_low": 2.0177, "price_close": 2.0365, "volume_traded": 124359.45999999996}
]

# Convert to DataFrame and process
df = pd.DataFrame(morpho_data)
df['timestamp'] = pd.to_datetime(df['time_period_start'])
df = df.sort_values('timestamp')

# Calculate technical indicators to proxy order book imbalance
def calculate_order_book_imbalance_proxy(df):
    """
    Calculate a proxy for Order Book Imbalance using price action and volume
    OBI Proxy = Volume-Weighted Price Momentum + Volatility Asymmetry
    """
    
    # Price change and direction
    df['price_change'] = df['price_close'] - df['price_open']
    df['price_change_pct'] = (df['price_close'] - df['price_open']) / df['price_open'] * 100
    
    # Volume-weighted price momentum (proxy for buy/sell pressure)
    df['vwap'] = (df['price_high'] + df['price_low'] + df['price_close']) / 3
    df['volume_ma'] = df['volume_traded'].rolling(window=12).mean()
    df['volume_ratio'] = df['volume_traded'] / df['volume_ma']
    
    # Price position within the candle (bullish vs bearish structure)
    df['candle_position'] = (df['price_close'] - df['price_low']) / (df['price_high'] - df['price_low'])
    df['candle_position'] = df['candle_position'].fillna(0.5)  # neutral if high=low
    
    # Upper vs Lower shadow (buying vs selling pressure proxy)
    df['upper_shadow'] = df['price_high'] - np.maximum(df['price_open'], df['price_close'])
    df['lower_shadow'] = np.minimum(df['price_open'], df['price_close']) - df['price_low']
    df['shadow_ratio'] = (df['upper_shadow'] - df['lower_shadow']) / (df['price_high'] - df['price_low'])
    df['shadow_ratio'] = df['shadow_ratio'].fillna(0)
    
    # Volume-momentum combination
    df['momentum'] = df['price_change_pct'].rolling(window=3).mean()
    df['volume_momentum'] = df['volume_ratio'] * df['momentum']
    
    # Final OBI Proxy: combination of multiple factors
    # Positive = more buying pressure, Negative = more selling pressure
    df['obi_proxy'] = (
        0.4 * df['volume_momentum'] +           # Volume-weighted momentum
        0.3 * (df['candle_position'] - 0.5) * 2 +  # Candle structure (-1 to 1)
        0.2 * (-df['shadow_ratio']) +           # Shadow analysis (inverted)
        0.1 * df['price_change_pct']            # Raw price change
    )
    
    # Smooth the OBI proxy
    df['obi_smooth'] = df['obi_proxy'].rolling(window=6).mean()
    
    return df

# Calculate OBI proxy
df = calculate_order_book_imbalance_proxy(df)

# Create the visualization
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

# Color scheme
colors = {
    'price': '#00D2FF',
    'obi_pos': '#00FF88',
    'obi_neg': '#FF4444',
    'volume': '#FFB347',
    'grid': '#333333'
}

# Top plot: Price and Volume
ax1_vol = ax1.twinx()

# Price line
price_line = ax1.plot(df['timestamp'], df['price_close'], 
                     color=colors['price'], linewidth=2.5, 
                     label='MORPHO Price', alpha=0.9)

# Volume bars
volume_bars = ax1_vol.bar(df['timestamp'], df['volume_traded'], 
                         alpha=0.3, color=colors['volume'], 
                         label='Volume', width=0.03)

# Style top plot
ax1.set_title('MORPHO/USDT - Price Action & Order Book Imbalance Analysis\n(Gate.io - Last 100 Hours)', 
             fontsize=18, fontweight='bold', color='white', pad=20)
ax1.set_ylabel('Price (USDT)', fontsize=14, color=colors['price'], fontweight='bold')
ax1_vol.set_ylabel('Volume (MORPHO)', fontsize=14, color=colors['volume'], fontweight='bold')

ax1.tick_params(colors='white', labelsize=11)
ax1_vol.tick_params(colors='white', labelsize=11)
ax1.grid(True, alpha=0.3, color=colors['grid'])

# Format price axis
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.3f}'))

# Bottom plot: Order Book Imbalance
obi_pos = df['obi_smooth'] >= 0
obi_colors = [colors['obi_pos'] if pos else colors['obi_neg'] for pos in obi_pos]

# OBI bars
obi_bars = ax2.bar(df['timestamp'], df['obi_smooth'], 
                  color=obi_colors, alpha=0.8, width=0.03)

# Zero line
ax2.axhline(y=0, color='white', linestyle='-', alpha=0.5, linewidth=1)

# OBI trend line
ax2.plot(df['timestamp'], df['obi_smooth'], 
         color='white', linewidth=1.5, alpha=0.7)

# Style bottom plot
ax2.set_title('Order Book Imbalance Proxy', fontsize=14, fontweight='bold', color='white')
ax2.set_xlabel('Time (UTC)', fontsize=14, fontweight='bold')
ax2.set_ylabel('OBI Score', fontsize=14, fontweight='bold')

ax2.tick_params(colors='white', labelsize=11)
ax2.grid(True, alpha=0.3, color=colors['grid'])

# Add OBI interpretation text
obi_text = """OBI Interpretation:
Positive (Green): Buying pressure dominates
Negative (Red): Selling pressure dominates
Magnitude: Strength of imbalance"""

ax2.text(0.02, 0.98, obi_text, transform=ax2.transAxes, 
         fontsize=10, color='white', alpha=0.8,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Add statistics box
current_price = df['price_close'].iloc[-1]
price_change = df['price_change_pct'].iloc[-1]
current_obi = df['obi_smooth'].iloc[-1]
max_obi = df['obi_smooth'].max()
min_obi = df['obi_smooth'].min()

stats_text = f"""Current Stats:
Price: ${current_price:.4f} ({price_change:+.2f}%)
OBI Score: {current_obi:.3f}
OBI Range: [{min_obi:.3f}, {max_obi:.3f}]
Period: {len(df)} hours"""

ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         fontsize=12, color='white', fontweight='bold',
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

# Rotate x-axis labels
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Tight layout
plt.tight_layout()

# Add subtle background gradient
fig.patch.set_facecolor('#0a0a0a')
ax1.set_facecolor('#0f0f0f')
ax2.set_facecolor('#0f0f0f')

# Save the plot
plt.savefig('/Users/valter.rebelo/MissionControl/morpho_obi_analysis.png', 
            dpi=300, bbox_inches='tight', facecolor='#0a0a0a')

print("✅ MORPHO Order Book Imbalance Analysis Complete!")
print(f"📊 Plot saved as: /Users/valter.rebelo/MissionControl/morpho_obi_analysis.png")
print(f"📈 Current Price: ${current_price:.4f} ({price_change:+.2f}%)")
print(f"⚖️  Current OBI Score: {current_obi:.3f}")

if current_obi > 0.5:
    print("🟢 Strong buying pressure detected")
elif current_obi > 0:
    print("🟡 Mild buying pressure")
elif current_obi > -0.5:
    print("🟡 Mild selling pressure")
else:
    print("🔴 Strong selling pressure detected")

plt.show()