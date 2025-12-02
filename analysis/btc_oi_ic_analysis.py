"""
Bitcoin Open Interest Information Coefficient Analysis
Investigates causality between OI flushes and forward returns by:
1. Detrending returns using rolling linear regression (365-day window)
2. Vol-adjusting residual returns using EWMA (30-day span)
3. Calculating Information Coefficients for 10d, 30d, 90d, 120d horizons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BITCOIN OI INFORMATION COEFFICIENT ANALYSIS")
print("Trend-Adjusted & Vol-Adjusted Causality Investigation")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND OI FLUSH CALCULATION
# ============================================================================
print("\n[1/6] Loading Bitcoin OI and price data...")
df = pd.read_csv('/Users/valter.rebelo/Downloads/btc-futures-open-interest-perpetual.csv')

# Parse columns (same structure as ETH script)
df.columns = ['oi_timestamp', 'open_interest', 'price_timestamp', 'price']

# Keep only rows where OI data exists
df_clean = df[df['open_interest'].notna()].copy()

# Convert timestamps to datetime
df_clean['date'] = pd.to_datetime(df_clean['oi_timestamp'])
df_clean = df_clean.sort_values('date').reset_index(drop=True)

# Keep only relevant columns
df_clean = df_clean[['date', 'open_interest', 'price']].copy()

print(f"   Data range: {df_clean['date'].min()} to {df_clean['date'].max()}")
print(f"   Total days: {len(df_clean)}")

# Calculate OI drawdown from 30-day rolling max (OI flush signal)
lookback_window = 30
df_clean['oi_rolling_max_30d'] = df_clean['open_interest'].rolling(window=lookback_window).max()
df_clean['oi_drawdown_pct'] = ((df_clean['open_interest'] - df_clean['oi_rolling_max_30d']) /
                                 df_clean['oi_rolling_max_30d'] * 100)

print(f"   OI drawdown signal calculated (30-day rolling max)")
print(f"   Mean OI drawdown: {df_clean['oi_drawdown_pct'].mean():.2f}%")
print(f"   Largest flush: {df_clean['oi_drawdown_pct'].min():.2f}%")

# ============================================================================
# 2. ROLLING LINEAR REGRESSION FOR DETRENDING
# ============================================================================
print("\n[2/6] Detrending forward returns using 365-day rolling linear regression...")

# Define horizons
horizons = [10, 30, 90, 120]
regression_window = 365

# Create time index for regression
df_clean['time_index'] = np.arange(len(df_clean))

# Calculate forward returns and detrended returns for each horizon
for horizon in horizons:
    print(f"   Processing {horizon}-day horizon...")

    # Calculate raw forward returns
    df_clean[f'raw_return_{horizon}d'] = df_clean['price'].pct_change(horizon).shift(-horizon) * 100

    # Initialize columns for expected and residual returns
    df_clean[f'expected_return_{horizon}d'] = np.nan
    df_clean[f'residual_return_{horizon}d'] = np.nan

    # Rolling linear regression to estimate expected returns
    for i in range(regression_window, len(df_clean) - horizon):
        # Get the rolling window of data
        window_indices = df_clean['time_index'].iloc[i-regression_window:i].values.reshape(-1, 1)
        window_returns = df_clean[f'raw_return_{horizon}d'].iloc[i-regression_window:i].values

        # Skip if insufficient non-null data
        if pd.isna(window_returns).sum() > regression_window * 0.3:  # Allow max 30% missing
            continue

        # Fit linear regression
        valid_mask = ~pd.isna(window_returns)
        if valid_mask.sum() < 30:  # Need at least 30 points
            continue

        X = window_indices[valid_mask]
        y = window_returns[valid_mask]

        try:
            model = LinearRegression()
            model.fit(X, y)

            # Predict expected return for current point
            current_time = df_clean['time_index'].iloc[i]
            expected = model.predict([[current_time]])[0]

            # Store expected return
            df_clean.loc[i, f'expected_return_{horizon}d'] = expected

            # Calculate residual
            actual = df_clean.loc[i, f'raw_return_{horizon}d']
            if not pd.isna(actual):
                df_clean.loc[i, f'residual_return_{horizon}d'] = actual - expected
        except:
            continue

print("   Detrending complete for all horizons")

# ============================================================================
# 3. VOLATILITY ADJUSTMENT USING EWMA
# ============================================================================
print("\n[3/6] Calculating EWMA volatility and adjusting residual returns...")

# Calculate daily returns for volatility estimation
df_clean['daily_return'] = df_clean['price'].pct_change() * 100

# Calculate EWMA volatility (30-day span, annualized)
ewma_span = 30
df_clean['ewma_vol'] = df_clean['daily_return'].ewm(span=ewma_span, adjust=False).std() * np.sqrt(252)

print(f"   EWMA volatility calculated (span={ewma_span} days)")
print(f"   Mean annualized vol: {df_clean['ewma_vol'].mean():.2f}%")
print(f"   Vol range: {df_clean['ewma_vol'].min():.2f}% to {df_clean['ewma_vol'].max():.2f}%")

# Vol-adjust residual returns for each horizon
for horizon in horizons:
    df_clean[f'vol_adj_residual_{horizon}d'] = (
        df_clean[f'residual_return_{horizon}d'] / df_clean['ewma_vol']
    )

print("   Volatility adjustment complete for all horizons")

# ============================================================================
# 4. CALCULATE INFORMATION COEFFICIENTS
# ============================================================================
print("\n[4/6] Calculating Information Coefficients...")

ic_results = {}
ic_stats = {}

for horizon in horizons:
    # Get valid data (both OI drawdown and vol-adjusted residual must be non-null)
    valid_mask = (
        df_clean['oi_drawdown_pct'].notna() &
        df_clean[f'vol_adj_residual_{horizon}d'].notna()
    )

    oi_signal = df_clean.loc[valid_mask, 'oi_drawdown_pct']
    returns_signal = df_clean.loc[valid_mask, f'vol_adj_residual_{horizon}d']

    if len(oi_signal) < 30:  # Need sufficient data
        print(f"   {horizon}d: Insufficient data (n={len(oi_signal)})")
        continue

    # Calculate Information Coefficient (Pearson correlation)
    ic = oi_signal.corr(returns_signal)

    # Calculate Spearman rank correlation (robust to outliers)
    spearman_ic, spearman_p = stats.spearmanr(oi_signal, returns_signal)

    # T-test for significance
    n = len(oi_signal)
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2) if abs(ic) < 1 else np.inf
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

    # Store results
    ic_results[horizon] = ic
    ic_stats[horizon] = {
        'ic': ic,
        'spearman_ic': spearman_ic,
        'spearman_p': spearman_p,
        't_stat': t_stat,
        'p_value': p_value,
        'n_obs': n,
        'significant': p_value < 0.05
    }

    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"   {horizon}d: IC = {ic:.4f} {sig_marker} (p={p_value:.4f}, n={n})")

# Also calculate IC for raw returns (without detrending) for comparison
ic_raw = {}
for horizon in horizons:
    valid_mask = (
        df_clean['oi_drawdown_pct'].notna() &
        df_clean[f'raw_return_{horizon}d'].notna() &
        df_clean['ewma_vol'].notna()
    )

    if valid_mask.sum() < 30:
        continue

    oi_signal = df_clean.loc[valid_mask, 'oi_drawdown_pct']
    raw_returns = df_clean.loc[valid_mask, f'raw_return_{horizon}d']
    vol = df_clean.loc[valid_mask, 'ewma_vol']

    # Vol-adjust but don't detrend
    vol_adj_raw = raw_returns / vol
    ic_raw[horizon] = oi_signal.corr(vol_adj_raw)

print("\n   Comparison (IC with vs without detrending):")
for horizon in horizons:
    if horizon in ic_raw and horizon in ic_results:
        print(f"   {horizon}d: Raw={ic_raw[horizon]:.4f} vs Detrended={ic_results[horizon]:.4f}")

# ============================================================================
# 5. CREATE VISUALIZATIONS
# ============================================================================
print("\n[5/6] Creating visualizations...")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. IC Bar Chart (detrended vs raw)
ax1 = fig.add_subplot(gs[0, 0])
x_pos = np.arange(len(horizons))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, [ic_raw.get(h, 0) for h in horizons],
                width, label='Raw Returns', alpha=0.7, color='lightblue')
bars2 = ax1.bar(x_pos + width/2, [ic_results.get(h, 0) for h in horizons],
                width, label='Detrended Returns', alpha=0.7, color='navy')

ax1.set_xlabel('Forward Horizon (days)')
ax1.set_ylabel('Information Coefficient')
ax1.set_title('IC: OI Drawdown vs Vol-Adjusted Returns\n(Detrended vs Raw)', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{h}d' for h in horizons])
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height != 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8)

# 2. Statistical Significance
ax2 = fig.add_subplot(gs[0, 1])
p_values = [ic_stats.get(h, {}).get('p_value', 1) for h in horizons]
colors = ['green' if p < 0.05 else 'orange' if p < 0.10 else 'red' for p in p_values]
bars = ax2.bar([f'{h}d' for h in horizons], p_values, color=colors, alpha=0.7)

ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2,
           label='p=0.05 threshold', alpha=0.7)
ax2.axhline(y=0.01, color='darkred', linestyle='--', linewidth=2,
           label='p=0.01 threshold', alpha=0.7)
ax2.set_ylabel('p-value')
ax2.set_xlabel('Forward Horizon')
ax2.set_title('Statistical Significance of IC\n(Lower is more significant)', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, max(p_values) * 1.1)  # Set y-limit for better visibility

# 3. Sample Size
ax3 = fig.add_subplot(gs[0, 2])
sample_sizes = [ic_stats.get(h, {}).get('n_obs', 0) for h in horizons]
ax3.bar([f'{h}d' for h in horizons], sample_sizes, color='steelblue', alpha=0.7)
ax3.set_ylabel('Number of Observations')
ax3.set_xlabel('Forward Horizon')
ax3.set_title('Sample Size by Horizon', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(sample_sizes):
    ax3.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# 4-7. Scatter plots: OI Drawdown vs Vol-Adjusted Residual Returns
scatter_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
scatter_axes.append(fig.add_subplot(gs[2, 0]))

for idx, horizon in enumerate(horizons):
    ax = scatter_axes[idx]

    valid_mask = (
        df_clean['oi_drawdown_pct'].notna() &
        df_clean[f'vol_adj_residual_{horizon}d'].notna()
    )

    x = df_clean.loc[valid_mask, 'oi_drawdown_pct']
    y = df_clean.loc[valid_mask, f'vol_adj_residual_{horizon}d']

    # Scatter plot with alpha for density
    ax.scatter(x, y, alpha=0.4, s=20, color='navy')

    # Add regression line
    if len(x) > 10:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Fit line')

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(x=-30, color='red', linestyle='--', linewidth=1, alpha=0.5, label='30% OI flush')

    ic_val = ic_results.get(horizon, 0)
    sig = "***" if ic_stats.get(horizon, {}).get('p_value', 1) < 0.001 else \
          "**" if ic_stats.get(horizon, {}).get('p_value', 1) < 0.01 else \
          "*" if ic_stats.get(horizon, {}).get('p_value', 1) < 0.05 else ""

    ax.set_xlabel('OI Drawdown (%)')
    ax.set_ylabel(f'Vol-Adj Residual Return (σ)')
    ax.set_title(f'{horizon}-Day Forward: IC = {ic_val:.4f}{sig}', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

# 8. Time series: EWMA Volatility
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(df_clean['date'], df_clean['ewma_vol'], linewidth=1, alpha=0.8, color='purple')
ax8.set_ylabel('Annualized Volatility (%)')
ax8.set_xlabel('Date')
ax8.set_title('EWMA Volatility (30-day span)', fontweight='bold')
ax8.grid(True, alpha=0.3)

# 9. Time series: OI Drawdown
ax9 = fig.add_subplot(gs[2, 2])
ax9.fill_between(df_clean['date'], 0, df_clean['oi_drawdown_pct'],
                 where=(df_clean['oi_drawdown_pct'] < 0),
                 alpha=0.3, color='red', label='OI Drawdown')
ax9.plot(df_clean['date'], df_clean['oi_drawdown_pct'],
        linewidth=0.5, alpha=0.7, color='darkred')
ax9.axhline(y=-30, color='red', linestyle='--', linewidth=2,
           label='30% Flush Threshold', alpha=0.7)
ax9.set_ylabel('OI Drawdown from 30d Max (%)')
ax9.set_xlabel('Date')
ax9.set_title('OI Flush Events Over Time', fontweight='bold')
ax9.legend(fontsize=8)
ax9.grid(True, alpha=0.3)

plt.suptitle('Bitcoin OI Information Coefficient Analysis\nTrend-Adjusted & Vol-Adjusted Causality',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/Users/valter.rebelo/MissionControl/analysis/btc_oi_ic_analysis.png',
            dpi=300, bbox_inches='tight')
print("   Visualization saved: analysis/btc_oi_ic_analysis.png")

# ============================================================================
# 6. SUMMARY REPORT
# ============================================================================
print("\n[6/6] Generating summary report...")

print("\n" + "="*80)
print("SUMMARY REPORT: BTC OI INFORMATION COEFFICIENT ANALYSIS")
print("="*80)

print("\n1. METHODOLOGY")
print("   " + "─"*70)
print("   • OI Flush Signal: Drawdown from 30-day rolling max OI")
print("   • Detrending: 365-day rolling linear regression")
print("   • Vol Adjustment: EWMA volatility (30-day span)")
print("   • Forward Horizons: 10d, 30d, 90d, 120d")

print("\n2. INFORMATION COEFFICIENTS (Detrended & Vol-Adjusted)")
print("   " + "─"*70)
for horizon in horizons:
    if horizon in ic_stats:
        stats_dict = ic_stats[horizon]
        sig_stars = "***" if stats_dict['p_value'] < 0.001 else \
                   "**" if stats_dict['p_value'] < 0.01 else \
                   "*" if stats_dict['p_value'] < 0.05 else ""

        print(f"\n   {horizon}-Day Forward:")
        print(f"      • Pearson IC: {stats_dict['ic']:+.4f} {sig_stars}")
        print(f"      • Spearman IC: {stats_dict['spearman_ic']:+.4f}")
        print(f"      • T-statistic: {stats_dict['t_stat']:.2f}")
        print(f"      • P-value: {stats_dict['p_value']:.6f}")
        print(f"      • Sample size: {stats_dict['n_obs']}")
        print(f"      • Significant: {'YES' if stats_dict['significant'] else 'NO'} (α=0.05)")

print("\n3. COMPARISON: RAW vs DETRENDED RETURNS")
print("   " + "─"*70)
print("   Horizon | Raw IC  | Detrended IC | Improvement")
print("   " + "─"*70)
for horizon in horizons:
    if horizon in ic_raw and horizon in ic_results:
        raw = ic_raw[horizon]
        detrended = ic_results[horizon]
        improvement = detrended - raw
        print(f"   {horizon:3d}d    | {raw:+.4f} | {detrended:+.4f}     | {improvement:+.4f}")

print("\n4. KEY FINDINGS")
print("   " + "─"*70)

# Determine overall signal direction
avg_ic = np.mean([ic_results.get(h, 0) for h in horizons if h in ic_results])
significant_count = sum(1 for h in horizons if ic_stats.get(h, {}).get('significant', False))

print(f"   • Average IC across all horizons: {avg_ic:+.4f}")
print(f"   • Significant horizons (p<0.05): {significant_count}/{len(horizons)}")

if avg_ic < -0.05:
    print("   • Signal: NEGATIVE IC → OI flushes predict LOWER forward returns")
    print("     (More negative drawdown → worse forward performance)")
elif avg_ic > 0.05:
    print("   • Signal: POSITIVE IC → OI flushes predict HIGHER forward returns")
    print("     (More negative drawdown → better forward performance)")
else:
    print("   • Signal: NEUTRAL → Weak/no predictive relationship")

# Compare to raw
raw_avg = np.mean([ic_raw.get(h, 0) for h in horizons if h in ic_raw])
if abs(avg_ic) > abs(raw_avg):
    print(f"   • Detrending STRENGTHENED the signal (raw avg: {raw_avg:+.4f})")
else:
    print(f"   • Detrending WEAKENED the signal (raw avg: {raw_avg:+.4f})")

print("\n5. STATISTICAL ROBUSTNESS")
print("   " + "─"*70)
min_obs = min(ic_stats.get(h, {}).get('n_obs', 0) for h in horizons if h in ic_stats)
max_obs = max(ic_stats.get(h, {}).get('n_obs', 0) for h in horizons if h in ic_stats)
print(f"   • Sample size range: {min_obs} to {max_obs} observations")

# Check consistency across horizons
ic_values = [ic_results.get(h, 0) for h in horizons if h in ic_results]
ic_std = np.std(ic_values)
print(f"   • IC standard deviation: {ic_std:.4f}")
print(f"   • Consistency: {'HIGH' if ic_std < 0.05 else 'MODERATE' if ic_std < 0.10 else 'LOW'}")

print("\n" + "="*80)

# Save results to CSV
results_summary = pd.DataFrame({
    'horizon': horizons,
    'ic_detrended': [ic_results.get(h, np.nan) for h in horizons],
    'ic_raw': [ic_raw.get(h, np.nan) for h in horizons],
    'spearman_ic': [ic_stats.get(h, {}).get('spearman_ic', np.nan) for h in horizons],
    'p_value': [ic_stats.get(h, {}).get('p_value', np.nan) for h in horizons],
    'n_observations': [ic_stats.get(h, {}).get('n_obs', 0) for h in horizons],
    'significant': [ic_stats.get(h, {}).get('significant', False) for h in horizons]
})

results_summary.to_csv('/Users/valter.rebelo/MissionControl/analysis/btc_oi_ic_results.csv', index=False)
print("\n✅ Results saved to: analysis/btc_oi_ic_results.csv")
print("✅ Analysis complete!")
