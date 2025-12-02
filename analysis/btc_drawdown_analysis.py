# %%
import os
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from experiments.utils.feature_loader import FeatureLoader

# %%
# Configuration
asset = 'bitcoin'
start_date = '2023-01-01'
end_date = '2025-12-01'

# Load data using FeatureLoader
print(f"Carregando dados de {asset} de {start_date} a {end_date}...")
loader = FeatureLoader(start_date=start_date, end_date=end_date)
data = loader.build_feature_set(crypto_assets=[asset])

# Set timestamp as index
data.set_index('timestamp', inplace=True)

# Rename columns to remove asset prefix
data.rename(columns={
    f'{asset}_close': 'close',
}, inplace=True)

data.dropna(inplace=True)

print(f"Carregados {len(data)} pontos de dados de {data.index.min()} a {data.index.max()}")

# %%
# Calculate drawdown
# Drawdown is the decline from peak to trough (as a percentage)
data['cummax'] = data['close'].cummax()
data['drawdown'] = (data['close'] - data['cummax']) / data['cummax'] * 100

print(f"\nEstatísticas de Drawdown:")
print(f"Drawdown Máximo: {data['drawdown'].min():.2f}%")
print(f"Drawdown Médio: {data['drawdown'].mean():.2f}%")
print(f"Drawdown Mediano: {data['drawdown'].median():.2f}%")
print(f"Drawdown Atual: {data['drawdown'].iloc[-1]:.2f}%")

# %%
# Create visualizations
import matplotlib.dates as mdates

def plot_drawdown_analysis(data, start_date, end_date, language='pt'):
    """
    Plot Bitcoin price, drawdown time series, and drawdown distribution with language toggle.
    
    Parameters:
    - data: DataFrame with price and drawdown data
    - start_date: Start date for the plot title
    - end_date: End date for the plot title
    - language: str, 'pt' for Portuguese or 'en' for English
    """
    # Define text based on language
    if language == 'pt':
        title_price = f'Preço do Bitcoin ({start_date} a {end_date})'
        ylabel_price = 'Preço (USD)'
        price_text = f'Atual: ${data["close"].iloc[-1]:,.0f}\nMáx: ${data["close"].max():,.0f}\nMín: ${data["close"].min():,.0f}'
        title_drawdown = 'Drawdown Acumulado'
        ylabel_drawdown = 'Drawdown (%)'
        drawdown_text = f'DD Atual: {data["drawdown"].iloc[-1]:.2f}%\nDD Máximo: {data["drawdown"].min():.2f}%'
        title_distribution = 'Distribuição de Drawdown (Buckets de 1%)'
        xlabel_distribution = 'Drawdown (%)'
        ylabel_distribution = 'Frequência (%)'
        mean_label = f'Média: {data["drawdown"].mean():.2f}%'
        median_label = f'Mediana: {data["drawdown"].median():.2f}%'
        max_label = f'DD Máx: {data["drawdown"].min():.2f}%'
        save_message = "\nGráfico salvo como btc_drawdown_analysis.png"
    else:
        title_price = f'Bitcoin Price ({start_date} to {end_date})'
        ylabel_price = 'Price (USD)'
        price_text = f'Current: ${data["close"].iloc[-1]:,.0f}\nMax: ${data["close"].max():,.0f}\nMin: ${data["close"].min():,.0f}'
        title_drawdown = 'Cumulative Drawdown'
        ylabel_drawdown = 'Drawdown (%)'
        drawdown_text = f'Current DD: {data["drawdown"].iloc[-1]:.2f}%\nMax DD: {data["drawdown"].min():.2f}%'
        title_distribution = 'Drawdown Distribution (1% Buckets)'
        xlabel_distribution = 'Drawdown (%)'
        ylabel_distribution = 'Frequency (%)'
        mean_label = f'Mean: {data["drawdown"].mean():.2f}%'
        median_label = f'Median: {data["drawdown"].median():.2f}%'
        max_label = f'Max DD: {data["drawdown"].min():.2f}%'
        save_message = "\nChart saved as btc_drawdown_analysis.png"

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Plot 1: BTC Price Time Series
    axes[0].plot(data.index, data['close'], color='steelblue', linewidth=1.5)
    axes[0].fill_between(data.index, data['close'], alpha=0.3, color='steelblue')
    axes[0].set_title(title_price, fontsize=16, fontweight='bold', pad=20)
    axes[0].set_ylabel(ylabel_price, fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0].xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add price annotations
    axes[0].text(0.01, 0.97, price_text,
                 transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Plot 2: Rolling Drawdown Time Series
    axes[1].fill_between(data.index, data['drawdown'], 0, color='red', alpha=0.3)
    axes[1].plot(data.index, data['drawdown'], color='darkred', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_title(title_drawdown, fontsize=16, fontweight='bold', pad=20)
    axes[1].set_ylabel(ylabel_drawdown, fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add drawdown annotations
    axes[1].text(0.01, 0.03, drawdown_text,
                 transform=axes[1].transAxes, fontsize=11, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Plot 3: Drawdown Distribution Histogram (1% buckets)
    # Create bins in 1% increments
    min_dd = np.floor(data['drawdown'].min())
    max_dd = np.ceil(data['drawdown'].max())
    bins = np.arange(min_dd, max_dd + 1, 1)  # 1% buckets

    # Calculate histogram with percentages
    counts, bin_edges = np.histogram(data['drawdown'], bins=bins)
    percentages = (counts / len(data)) * 100

    axes[2].bar(bin_edges[:-1], percentages, width=0.9, color='darkred', alpha=0.7, edgecolor='black', linewidth=0.5, align='edge')
    axes[2].axvline(x=data['drawdown'].mean(), color='blue', linestyle='--', linewidth=2.5, label=mean_label)
    axes[2].axvline(x=data['drawdown'].median(), color='green', linestyle='--', linewidth=2.5, label=median_label)
    axes[2].axvline(x=data['drawdown'].min(), color='red', linestyle='--', linewidth=2.5, label=max_label)
    axes[2].set_title(title_distribution, fontsize=16, fontweight='bold', pad=20)
    axes[2].set_xlabel(xlabel_distribution, fontsize=13)
    axes[2].set_ylabel(ylabel_distribution, fontsize=13)
    axes[2].legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='gray')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(hspace=0.3)
    plt.savefig('/Users/valter.rebelo/MissionControl/analysis/btc_drawdown_analysis.png', dpi=300, bbox_inches='tight')
    print(save_message)
    plt.show()

# Call the plotting function with language toggle (default to Portuguese)
plot_drawdown_analysis(data, start_date, end_date, language='pt')

# %%
# Additional Statistics: Drawdown Buckets
print("\n" + "="*60)
print("DISTRIBUIÇÃO DE DRAWDOWN POR BUCKET")
print("="*60)

# Count days in each 1% bucket
bucket_counts = pd.cut(data['drawdown'], bins=bins).value_counts().sort_index()

print(f"\n{'Faixa de Drawdown':<20} {'Dias':<10} {'Percentual':<15}")
print("-" * 45)
total_days = len(data)
for bucket, count in bucket_counts.items():
    pct = (count / total_days) * 100
    print(f"{bucket!s:<20} {count:<10} {pct:>6.2f}%")

# %%
# Drawdown Duration Analysis
print("\n" + "="*60)
print("ANÁLISE DE DURAÇÃO DOS DRAWDOWNS")
print("="*60)

# Identify drawdown periods (when drawdown < 0)
in_drawdown = data['drawdown'] < -0.1  # Consider drawdowns below -0.1% as significant
drawdown_periods = []
start_idx = None

for idx in range(len(in_drawdown)):
    if in_drawdown.iloc[idx] and start_idx is None:
        start_idx = idx
    elif not in_drawdown.iloc[idx] and start_idx is not None:
        drawdown_periods.append((start_idx, idx - 1))
        start_idx = None

# Handle case where drawdown continues to the end
if start_idx is not None:
    drawdown_periods.append((start_idx, len(in_drawdown) - 1))

if drawdown_periods:
    print(f"\nNúmero de períodos de drawdown significativos: {len(drawdown_periods)}")
    print(f"\n{'Data Início':<12} {'Data Fim':<12} {'Duração (dias)':<20} {'DD Máx (%)':<15} {'Recuperação'}")
    print("-" * 75)

    for start_idx, end_idx in drawdown_periods:
        start_date = data.index[start_idx].strftime('%Y-%m-%d')
        end_date = data.index[end_idx].strftime('%Y-%m-%d')
        duration = end_idx - start_idx + 1
        max_dd_in_period = data['drawdown'].iloc[start_idx:end_idx+1].min()

        # Check if recovered (drawdown back to 0)
        if end_idx < len(data) - 1:
            recovery = "Sim" if data['drawdown'].iloc[end_idx+1] > -0.1 else "Não"
        else:
            recovery = "Em curso"

        print(f"{start_date:<12} {end_date:<12} {duration:<20} {max_dd_in_period:<15.2f} {recovery}")

print("\n" + "="*60)

# %%
