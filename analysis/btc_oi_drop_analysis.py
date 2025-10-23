"""
Bitcoin Open Interest Drop Analysis
Analyzes forward returns after significant OI drops (30-40%+)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
print("Loading Bitcoin OI and price data...")
df = pd.read_csv('/Users/valter.rebelo/Downloads/btc-futures-open-interest-perpetual.csv')

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

# Adjust OI for price to get contract count instead of USD value
df_clean['oi_contracts'] = df_clean['open_interest'] / df_clean['price']

print(f"Data range: {df_clean['date'].min()} to {df_clean['date'].max()}")
print(f"Total days: {len(df_clean)}")

# Calculate OI changes (in contracts)
df_clean['oi_pct_change_1d'] = df_clean['oi_contracts'].pct_change() * 100

# Calculate rolling max OI (30-day lookback) and drawdown from that max (in contracts)
lookback_window = 30
df_clean['oi_rolling_max_30d'] = df_clean['oi_contracts'].rolling(window=lookback_window).max()
df_clean['oi_drawdown_pct'] = ((df_clean['oi_contracts'] - df_clean['oi_rolling_max_30d']) /
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
            'oi_contracts': row['oi_contracts'],
            'price': row['price'],
            'drop_pct': row['oi_drawdown_pct'],
            'idx': idx
        })
        last_drop_date = current_date

drop_df = pd.DataFrame(drop_events)
print(f"\nFound {len(drop_df)} significant OI drop events (≥30% drop from 30-day high in contracts)")

# Calculate forward returns for each event, ensuring no overlap
horizons = [10, 30, 90, 120]
results = []
used_dates = set()  # Track dates used in forward return periods to avoid overlap

for _, event in drop_df.iterrows():
    event_idx = event['idx']
    event_date = event['date']
    event_price = event['price']
    overlap = False

    # Check if this event's forward periods overlap with already used dates
    for horizon in horizons:
        future_idx = event_idx + horizon
        if future_idx < len(df_clean):
            future_date = df_clean.loc[future_idx, 'date']
            if future_date in used_dates:
                overlap = True
                break

    if overlap:
        continue  # Skip this event if there's any overlap

    result = {
        'event_date': event_date,
        'event_price': event_price,
        'oi_drop_pct': event['drop_pct'],
        'oi_value': event['oi_value'],
        'oi_contracts': event['oi_contracts']
    }

    # Mark forward dates as used
    for horizon in horizons:
        future_idx = event_idx + horizon
        if future_idx < len(df_clean):
            future_date = df_clean.loc[future_idx, 'date']
            used_dates.add(future_date)

    # Calculate forward returns
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
print(f"\nAfter filtering for non-overlapping events, {len(results_df)} events remain for analysis")

# Save results
results_df.to_csv('/Users/valter.rebelo/MissionControl/analysis/btc_oi_drop_results.csv', index=False)
print(f"\nResults saved to: analysis/btc_oi_drop_results.csv")

# Print summary statistics
print("\n" + "="*80)
print("BITCOIN OPEN INTEREST DROP ANALYSIS - FORWARD RETURNS REPORT")
print("="*80)

print(f"\n1. SAMPLE OVERVIEW")
print(f"   {'─'*70}")
print(f"   Total OI drop events analyzed: {len(results_df)}")
print(f"   Date range: {results_df['event_date'].min().strftime('%Y-%m-%d')} to " +
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

    print(f"\n   {horizon}-Day Forward Returns:")
    print(f"      • Average return: {avg_return:+.2f}%")
    print(f"      • Median return: {median_return:+.2f}%")
    print(f"      • Win rate: {win_rate:.1f}% ({positive_count} positive, {negative_count} negative)")
    print(f"      • Best case: {returns.max():+.2f}%")
    print(f"      • Worst case: {returns.min():+.2f}%")
    print(f"      • Std deviation: {returns.std():.2f}%")

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
        verdict = "✓ YES - Bearish was correct (negative avg return, <50% win rate)"
    elif avg > 0 and win_rate > 50:
        verdict = "✗ NO - Actually BULLISH (positive avg return, >50% win rate)"
    else:
        verdict = "~ MIXED signals (avg and win rate disagree)"

    print(f"\n   {horizon}-day horizon: {verdict}")
    print(f"      Average return: {avg:+.2f}%")
    print(f"      Win rate: {win_rate:.1f}%")

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

    print(f"   {horizon}-day: {significance} {direction} bias (p={p_value:.4f}, t={t_stat:.2f})")

print(f"\n5. INDIVIDUAL EVENTS")
print(f"   {'─'*70}")
print(f"\n   All OI Drop Events:")

for idx, row in results_df.iterrows():
    print(f"\n   {idx+1}. Event: {row['event_date'].strftime('%Y-%m-%d')}")
    print(f"      • OI drop: {row['oi_drop_pct']:.2f}%")
    print(f"      • BTC price at event: ${row['event_price']:,.2f}")
    for horizon in horizons:
        ret = row[f'return_{horizon}d']
        if not pd.isna(ret):
            emoji = "📈" if ret > 0 else "📉"
            print(f"      • {horizon}d return: {ret:+.2f}% {emoji}")

print(f"\n{'='*80}\n")

# Create visualizations
print("\nCreating visualizations...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('Quedas de Open Interest do Bitcoin: Análise de Retornos Futuros',
             fontsize=16, fontweight='bold')

# 1. OI over time with drop events (showing contracts instead of USD)
ax1 = axes[0, 0]
ax1.plot(df_clean['date'], df_clean['oi_contracts'], label='Open Interest (Contracts)', linewidth=1, alpha=0.7)
ax1.scatter(results_df['event_date'], results_df['oi_contracts'],
           color='red', s=100, marker='v', label='Eventos de Queda de OI (≥30%)', zorder=5)
ax1.set_title('Open Interest do Bitcoin com Eventos de Queda', fontweight='bold')
ax1.set_ylabel('Open Interest (Contracts)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Price over time with drop events
ax2 = axes[0, 1]
ax2.plot(df_clean['date'], df_clean['price'], label='Preço BTC', linewidth=1, alpha=0.7)
ax2.scatter(results_df['event_date'], results_df['event_price'],
           color='red', s=100, marker='v', label='Eventos de Queda de OI', zorder=5)
ax2.set_title('Preço do Bitcoin nos Eventos de Queda de OI', fontweight='bold')
ax2.set_ylabel('Preço (USD)')
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

ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Retorno Zero')
ax3.set_title('Distribuição dos Retornos Futuros Após Quedas de OI', fontweight='bold')
ax3.set_xlabel('Horizonte Temporal')
ax3.set_ylabel('Retorno (%)')
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
ax4.set_title('Retornos Médios Futuros Após Quedas de OI', fontweight='bold')
ax4.set_xlabel('Horizonte Temporal')
ax4.set_ylabel('Retorno Médio (%)')
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

ax5.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% (Aleatório)')
ax5.set_title('Taxa de Acerto (% de Retornos Positivos)', fontweight='bold')
ax5.set_xlabel('Horizonte Temporal')
ax5.set_ylabel('Taxa de Acerto (%)')
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
ax6.set_title('Retornos Futuros por Data do Evento', fontweight='bold')
ax6.set_xlabel('Data do Evento')
ax6.set_ylabel('Retorno Futuro (%)')
ax6.legend(loc='best', framealpha=0.9)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/valter.rebelo/MissionControl/analysis/btc_oi_drop_analysis.png',
            dpi=300, bbox_inches='tight')
print("Chart saved to: analysis/btc_oi_drop_analysis.png")

print("\n✅ Analysis complete!")
