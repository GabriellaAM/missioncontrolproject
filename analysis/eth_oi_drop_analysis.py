"""
Ethereum Open Interest Drop Analysis
Analyzes forward returns after significant OI drops (30-40%+)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
print("Loading Ethereum OI and price data...")
df = pd.read_csv('/Users/valter.rebelo/Downloads/eth-futures-open-interest-perpetual.csv')

# The CSV has duplicate columns - let's parse it correctly
# Columns: timestamp (OI), OI value, timestamp (price), price
df.columns = ['oi_timestamp', 'open_interest', 'price_timestamp', 'price']

# Keep only rows where OI data exists
df_clean = df[df['open_interest'].notna()].copy()

# Convert timestamps to datetime
df_clean['date'] = pd.to_datetime(df_clean['oi_timestamp'])
df_clean = df_clean.sort_values('date').reset_index(drop=True)

# Keep only relevant columns
df_clean = df_clean[['date', 'open_interest', 'price']]

print(f"Data range: {df_clean['date'].min()} to {df_clean['date'].max()}")
print(f"Total days: {len(df_clean)}")

# Calculate OI changes
df_clean['oi_pct_change_1d'] = df_clean['open_interest'].pct_change() * 100

# Calculate rolling max OI (30-day lookback) and drawdown from that max
lookback_window = 30
df_clean['oi_rolling_max_30d'] = df_clean['open_interest'].rolling(window=lookback_window).max()
df_clean['oi_drawdown_pct'] = ((df_clean['open_interest'] - df_clean['oi_rolling_max_30d']) /
                                 df_clean['oi_rolling_max_30d'] * 100)

# Identify significant OI drops (≥30% from recent high)
threshold = -30
drops = df_clean[df_clean['oi_drawdown_pct'] <= threshold].copy()

# Filter to get distinct events (avoid counting consecutive days of same drop)
drop_events = []
last_drop_date = None

for idx, row in drops.iterrows():
    current_date = row['date']

    if last_drop_date is None or (current_date - last_drop_date).days > 7:
        drop_events.append({
            'date': current_date,
            'oi_value': row['open_interest'],
            'price': row['price'],
            'drop_pct': row['oi_drawdown_pct'],
            'idx': idx
        })
        last_drop_date = current_date

drop_df = pd.DataFrame(drop_events)
print(f"\nFound {len(drop_df)} significant OI drop events (≥30% drop from 30-day high)")

# Calculate forward returns for each event
horizons = [10, 30, 90, 120]
results = []

for _, event in drop_df.iterrows():
    event_idx = event['idx']
    event_date = event['date']
    event_price = event['price']

    result = {
        'event_date': event_date,
        'event_price': event_price,
        'oi_drop_pct': event['drop_pct'],
        'oi_value': event['oi_value']
    }

    for horizon in horizons:
        future_idx = event_idx + horizon

        if future_idx < len(df_clean):
            future_price = df_clean.loc[future_idx, 'price']
            future_date = df_clean.loc[future_idx, 'date']
            forward_return = ((future_price - event_price) / event_price) * 100

            result[f'return_{horizon}d'] = forward_return
            result[f'price_{horizon}d'] = future_price
            result[f'date_{horizon}d'] = future_date
        else:
            result[f'return_{horizon}d'] = np.nan
            result[f'price_{horizon}d'] = np.nan
            result[f'date_{horizon}d'] = np.nan

    results.append(result)

results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('/Users/valter.rebelo/MissionControl/analysis/eth_oi_drop_results.csv', index=False)
print(f"\nResults saved to: analysis/eth_oi_drop_results.csv")

# Print summary statistics
print("\n" + "="*80)
print("ETHEREUM OPEN INTEREST DROP ANALYSIS - FORWARD RETURNS REPORT")
print("="*80)

print(f"\n1. SAMPLE OVERVIEW")
print(f"   {'─'*70}")
print(f"   Total OI drop events analyzed: {len(results_df)}")
print(f"   Period: {results_df['event_date'].min().strftime('%Y-%m-%d')} to " +
      f"{results_df['event_date'].max().strftime('%Y-%m-%d')}")
print(f"   Average OI drop magnitude: {results_df['oi_drop_pct'].mean():.2f}%")
print(f"   Median OI drop magnitude: {results_df['oi_drop_pct'].median():.2f}%")
print(f"   Largest OI drop: {results_df['oi_drop_pct'].min():.2f}%")

print(f"\n2. FORWARD RETURNS SUMMARY")
print(f"   {'─'*70}")

for horizon in horizons:
    col = f'return_{horizon}d'
    returns = results_df[col].dropna()

    if len(returns) == 0:
        continue

    avg_return = returns.mean()
    median_return = returns.median()
    win_rate = (returns > 0).sum() / len(returns) * 100
    positive_count = (returns > 0).sum()
    negative_count = (returns < 0).sum()

    print(f"\n   {horizon} Day Forward Returns:")
    print(f"      • Average Return: {avg_return:+.2f}%")
    print(f"      • Median Return: {median_return:+.2f}%")
    print(f"      • Win Rate: {win_rate:.1f}% ({positive_count} positive, {negative_count} negative)")
    print(f"      • Best Case: {returns.max():+.2f}%")
    print(f"      • Worst Case: {returns.min():+.2f}%")
    print(f"      • Standard Deviation: {returns.std():.2f}%")

print(f"\n3. KEY FINDINGS - Was Being Bearish Profitable?")
print(f"   {'─'*70}")

for horizon in horizons:
    col = f'return_{horizon}d'
    returns = results_df[col].dropna()

    if len(returns) == 0:
        continue

    avg = returns.mean()
    win_rate = (returns > 0).sum() / len(returns) * 100

    if avg < 0 and win_rate < 50:
        verdict = "✓ YES - Bearish was correct (negative avg return, win rate <50%)"
    elif avg > 0 and win_rate > 50:
        verdict = "✗ NO - Actually BULLISH (positive avg return, win rate >50%)"
    else:
        verdict = "~ MIXED signals (average and win rate disagree)"

    print(f"\n   {horizon} Day Horizon: {verdict}")
    print(f"      Average Return: {avg:+.2f}%")
    print(f"      Win Rate: {win_rate:.1f}%")

print(f"\n4. STATISTICAL SIGNIFICANCE")
print(f"   {'─'*70}")

from scipy import stats

for horizon in horizons:
    col = f'return_{horizon}d'
    returns = results_df[col].dropna()

    if len(returns) < 3:
        continue

    t_stat, p_value = stats.ttest_1samp(returns, 0)
    significance = "Statistically significant" if p_value < 0.05 else "Not statistically significant"
    direction = "positive" if returns.mean() > 0 else "negative"

    print(f"   {horizon} days: {significance} {direction} bias (p={p_value:.4f}, t={t_stat:.2f})")

print(f"\n5. INDIVIDUAL EVENTS")
print(f"   {'─'*70}")
print(f"\n   All OI Drop Events:")

for idx, row in results_df.iterrows():
    print(f"\n   {idx+1}. Event: {row['event_date'].strftime('%Y-%m-%d')}")
    print(f"      • OI Drop: {row['oi_drop_pct']:.2f}%")
    print(f"      • ETH Price at Event: ${row['event_price']:,.2f}")
    for horizon in horizons:
        ret = row[f'return_{horizon}d']
        if not pd.isna(ret):
            emoji = "📈" if ret > 0 else "📉"
            print(f"      • Return at {horizon}d: {ret:+.2f}% {emoji}")

print(f"\n{'='*80}\n")

# Create visualizations
print("\nCreating visualizations...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Ethereum Open Interest Drops: Forward Returns Analysis',
             fontsize=16, fontweight='bold')

# 1. OI over time with drop events
ax1 = axes[0, 0]
ax1.plot(df_clean['date'], df_clean['open_interest'], label='Open Interest', linewidth=1, alpha=0.7)
ax1.scatter(results_df['event_date'], results_df['oi_value'],
           color='red', s=100, marker='v', label='OI Drop Events (≥30%)', zorder=5)
ax1.set_title('Ethereum Open Interest with Drop Events', fontweight='bold')
ax1.set_ylabel('Open Interest')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Price over time with drop events
ax2 = axes[0, 1]
ax2.plot(df_clean['date'], df_clean['price'], label='ETH Price', linewidth=1, alpha=0.7)
ax2.scatter(results_df['event_date'], results_df['event_price'],
           color='red', s=100, marker='v', label='OI Drop Events', zorder=5)
ax2.set_title('Ethereum Price at OI Drop Events', fontweight='bold')
ax2.set_ylabel('Price (USD)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Distribution of forward returns
ax3 = axes[1, 0]
return_cols = [f'return_{h}d' for h in horizons]
box_data = [results_df[col].dropna() for col in return_cols]

bp = ax3.boxplot(box_data, labels=[f'{h}d' for h in horizons], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')

ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero Return')
ax3.set_title('Distribution of Forward Returns After OI Drops', fontweight='bold')
ax3.set_xlabel('Time Horizon')
ax3.set_ylabel('Return (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Average returns by horizon
ax4 = axes[1, 1]
avg_returns = [results_df[col].mean() for col in return_cols]
colors = ['green' if r > 0 else 'red' for r in avg_returns]
bars = ax4.bar([f'{h}d' for h in horizons], avg_returns, color=colors, alpha=0.7)

for bar, val in zip(bars, avg_returns):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_title('Average Forward Returns After OI Drops', fontweight='bold')
ax4.set_xlabel('Time Horizon')
ax4.set_ylabel('Average Return (%)')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Win rate
ax5 = axes[2, 0]
win_rates = [(results_df[col] > 0).sum() / len(results_df[col].dropna()) * 100
             for col in return_cols]
bars = ax5.bar([f'{h}d' for h in horizons], win_rates, color='steelblue', alpha=0.7)

for bar, val in zip(bars, win_rates):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')

ax5.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% (Random)')
ax5.set_title('Win Rate (% of Positive Returns)', fontweight='bold')
ax5.set_xlabel('Time Horizon')
ax5.set_ylabel('Win Rate (%)')
ax5.set_ylim(0, 100)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Individual event returns over time - with different line styles and markers
ax6 = axes[2, 1]
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']
colors_line = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, horizon in enumerate(horizons):
    col = f'return_{horizon}d'
    ax6.plot(results_df['event_date'], results_df[col],
             marker=markers[idx],
             linestyle=line_styles[idx],
             color=colors_line[idx],
             label=f'{horizon}d',
             alpha=0.8,
             linewidth=2,
             markersize=6,
             markerfacecolor=colors_line[idx],
             markeredgewidth=1.5,
             markeredgecolor='white')

ax6.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax6.set_title('Forward Returns by Event Date', fontweight='bold')
ax6.set_xlabel('Event Date')
ax6.set_ylabel('Forward Return (%)')
ax6.legend(loc='best', framealpha=0.9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/valter.rebelo/MissionControl/analysis/eth_oi_drop_analysis.png',
            dpi=300, bbox_inches='tight')
print("Chart saved to: analysis/eth_oi_drop_analysis.png")

print("\n✅ Analysis complete!")
