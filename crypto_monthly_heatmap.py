import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_crypto_data():
    """Load BTC and ETH price data from parquet files"""
    btc_df = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/crypto_data/coingecko/bitcoin/data.parquet')
    eth_df = pd.read_parquet('/Users/valter.rebelo/MissionControl/data_parquet/crypto_data/coingecko/ethereum/data.parquet')
    
    # Filter data from 2017 onwards
    btc_df = btc_df[btc_df['timestamp'] >= '2017-01-01'].copy()
    eth_df = eth_df[eth_df['timestamp'] >= '2017-01-01'].copy()
    
    # Set timestamp as index
    btc_df.set_index('timestamp', inplace=True)
    eth_df.set_index('timestamp', inplace=True)
    
    return btc_df, eth_df

def calculate_monthly_returns(df, asset_name):
    """Calculate monthly returns for the given asset"""
    # Resample to monthly data using last price of each month
    monthly_prices = df['close'].resample('M').last()
    
    # Calculate monthly returns
    monthly_returns = monthly_prices.pct_change().dropna() * 100
    
    # Create DataFrame with year, month, and returns
    results = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values,
        'asset': asset_name
    })
    
    return results

def create_heatmap_data(monthly_returns_df):
    """Create pivot table for heatmap visualization"""
    # Create pivot table
    heatmap_data = monthly_returns_df.pivot(index='year', columns='month', values='return')
    
    # Ensure all months are present
    month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    heatmap_data.columns = [month_names[i-1] for i in heatmap_data.columns]
    
    return heatmap_data

def normalize_by_year(heatmap_data):
    """Normalize colors by year (each row normalized independently)"""
    normalized_data = heatmap_data.copy()
    
    for year in heatmap_data.index:
        year_data = heatmap_data.loc[year]
        year_data_clean = year_data.dropna()
        
        if len(year_data_clean) > 0:
            # Normalize to -1 to 1 scale for each year
            min_val = year_data_clean.min()
            max_val = year_data_clean.max()
            
            if max_val != min_val:
                year_range = max(abs(min_val), abs(max_val))
                normalized_data.loc[year] = year_data / year_range
            else:
                normalized_data.loc[year] = 0
                
    return normalized_data

def create_crypto_heatmap(asset_name, heatmap_data, normalized_data, save_path):
    """Create heatmap visualization with MORPHO-style aesthetics"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color scheme similar to MORPHO analysis
    colors = {
        'positive': '#00FF88',
        'negative': '#FF4444',
        'neutral': '#333333'
    }
    
    # Create custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap_colors = ['#FF4444', '#330000', '#000000', '#003300', '#00FF88']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('crypto', cmap_colors, N=n_bins)
    
    # Create the heatmap using normalized data for colors
    mask = heatmap_data.isna()
    
    # Plot heatmap
    sns.heatmap(normalized_data, 
                annot=heatmap_data,  # Show actual values
                fmt='.1f',
                cmap=cmap,
                center=0,
                vmin=-1, vmax=1,
                mask=mask,
                cbar_kws={'label': 'Retornos Mensais Normalizados', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='#222222',
                ax=ax)
    
    # Styling
    ax.set_title(f'{asset_name} Mapa de Calor - Retornos Mensais (2017-2025)\nEscala de Cores Normalizada por Ano', 
                fontsize=18, fontweight='bold', color='white', pad=20)
    ax.set_xlabel('Mês', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Ano', fontsize=14, fontweight='bold', color='white')
    
    # Invert y-axis so years are ascending from bottom (origin) to top
    ax.invert_yaxis()
    
    # Style ticks
    ax.tick_params(colors='white', labelsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Set background
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0f0f0f')
    
    # Add statistics - positioned outside the heatmap area
    total_months = heatmap_data.count().sum()
    positive_months = (heatmap_data > 0).sum().sum()
    negative_months = (heatmap_data < 0).sum().sum()
    
    avg_return = heatmap_data.mean().mean()
    best_month = heatmap_data.max().max()
    worst_month = heatmap_data.min().min()
    
    stats_text = f"""Estatísticas ({asset_name}):
Total de Meses: {total_months}
Positivos: {positive_months} ({positive_months/total_months*100:.1f}%)
Negativos: {negative_months} ({negative_months/total_months*100:.1f}%)

Retorno Médio Mensal: {avg_return:.1f}%
Melhor Mês: {best_month:.1f}%
Pior Mês: {worst_month:.1f}%"""
    
    # Position statistics box outside the plot area (bottom left)
    ax.text(-0.15, -0.25, stats_text, transform=ax.transAxes,
            fontsize=10, color='white', fontweight='bold',
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    # Add interpretation text - positioned outside the heatmap area
    interpretation_text = """Interpretação das Cores:
🟢 Verde: Retornos positivos (normalizado por ano)
🔴 Vermelho: Retornos negativos (normalizado por ano)
📊 Números: Percentuais reais de retorno mensal
⚡ Cada linha normalizada independentemente"""
    
    # Position interpretation box outside the plot area (bottom right)
    ax.text(1.15, -0.25, interpretation_text, transform=ax.transAxes,
            fontsize=10, color='white', alpha=0.9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Adjust layout to accommodate external text boxes
    plt.subplots_adjust(left=0.15, bottom=0.25, right=0.85, top=0.9)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a0a0a')
    
    return fig

def main():
    print("🚀 Carregando dados de preços BTC e ETH...")
    
    # Load data
    btc_df, eth_df = load_crypto_data()
    
    print(f"📊 Dados BTC: {len(btc_df)} dias de {btc_df.index.min().date()} até {btc_df.index.max().date()}")
    print(f"📊 Dados ETH: {len(eth_df)} dias de {eth_df.index.min().date()} até {eth_df.index.max().date()}")
    
    # Calculate monthly returns
    print("\n📈 Calculando retornos mensais...")
    btc_monthly = calculate_monthly_returns(btc_df, 'BTC')
    eth_monthly = calculate_monthly_returns(eth_df, 'ETH')
    
    # Create heatmap data
    btc_heatmap = create_heatmap_data(btc_monthly)
    eth_heatmap = create_heatmap_data(eth_monthly)
    
    # Normalize by year
    btc_normalized = normalize_by_year(btc_heatmap)
    eth_normalized = normalize_by_year(eth_heatmap)
    
    print("\n🎨 Criando mapa de calor BTC...")
    btc_fig = create_crypto_heatmap('Bitcoin (BTC)', btc_heatmap, btc_normalized, 
                                   '/Users/valter.rebelo/MissionControl/btc_monthly_heatmap.png')
    
    print("🎨 Criando mapa de calor ETH...")
    eth_fig = create_crypto_heatmap('Ethereum (ETH)', eth_heatmap, eth_normalized, 
                                   '/Users/valter.rebelo/MissionControl/eth_monthly_heatmap.png')
    
    print("\n✅ Análise Concluída!")
    print("📊 Mapa de calor BTC salvo como: btc_monthly_heatmap.png")
    print("📊 Mapa de calor ETH salvo como: eth_monthly_heatmap.png")
    
    # Display summary statistics
    print(f"\n📈 Resumo BTC (2017-2025):")
    print(f"   Retorno Médio Mensal: {btc_monthly['return'].mean():.1f}%")
    print(f"   Melhor Mês: {btc_monthly['return'].max():.1f}%")
    print(f"   Pior Mês: {btc_monthly['return'].min():.1f}%")
    print(f"   Meses Positivos: {(btc_monthly['return'] > 0).sum()}/{len(btc_monthly)} ({(btc_monthly['return'] > 0).mean()*100:.1f}%)")
    
    print(f"\n📈 Resumo ETH (2017-2025):")
    print(f"   Retorno Médio Mensal: {eth_monthly['return'].mean():.1f}%")
    print(f"   Melhor Mês: {eth_monthly['return'].max():.1f}%")
    print(f"   Pior Mês: {eth_monthly['return'].min():.1f}%")
    print(f"   Meses Positivos: {(eth_monthly['return'] > 0).sum()}/{len(eth_monthly)} ({(eth_monthly['return'] > 0).mean()*100:.1f}%)")
    
    plt.show()

if __name__ == "__main__":
    main()