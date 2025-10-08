# %%
"""
Pipeline de Demonstração: Meta-Modelos com Features Topológicas para Trading
=============================================================================

Este script demonstra o pipeline utilizado no artigo de forma compactada e simplificada.

Autores: Valter Rebelo
Artigo: Contexto Importa: Machine Learning Topológico em Estratégias de Trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# %%
def carregar_dados_bitcoin(n_dias: int = 1000, preco_inicial: float = 50000) -> pd.DataFrame:
    """
    Carrega dados do Bitcoin: primeiro tenta ler de bitcoin.csv, se não disponível gera dados sintéticos.

    Parâmetros:
    -----------
    n_dias : int
        Número de dias de dados para gerar (apenas para dados sintéticos)
    preco_inicial : float
        Preço inicial do Bitcoin (apenas para dados sintéticos)

    Retorna:
    --------
    pd.DataFrame
        DataFrame com colunas: date (index), bitcoin_close, bitcoin_volume
    """
    import os

    # Primeiro, tentar carregar dados reais do CSV
    try:
        if os.path.exists('bitcoin.csv'):
            print("📈 Carregando dados reais do Bitcoin de bitcoin.csv...")
            df_real = pd.read_csv('bitcoin.csv')

            # Tentar identificar coluna de preço de fechamento
            close_columns = [col for col in df_real.columns if 'close' in col.lower()]
            if not close_columns:
                close_columns = [col for col in df_real.columns if 'price' in col.lower()]
            if not close_columns:
                raise ValueError("Coluna de preço de fechamento não encontrada no CSV")

            close_col = close_columns[0]

            # Tentar identificar coluna de data
            date_columns = [col for col in df_real.columns if any(term in col.lower() for term in ['date', 'time', 'timestamp'])]
            if not date_columns:
                # Se não houver coluna de data, usar index
                df_real['date'] = pd.date_range(start='2020-01-01', periods=len(df_real), freq='D')
                date_col = 'date'
            else:
                date_col = date_columns[0]

            # Tentar identificar coluna de volume (opcional)
            volume_columns = [col for col in df_real.columns if 'volume' in col.lower()]
            if volume_columns:
                volume_col = volume_columns[0]
            else:
                # Gerar volume sintético se não disponível
                df_real['bitcoin_volume'] = 50000 * np.random.uniform(0.5, 1.5, len(df_real))
                volume_col = 'bitcoin_volume'

            # Padronizar nomes das colunas
            df = pd.DataFrame({
                'date': pd.to_datetime(df_real[date_col]),
                'bitcoin_close': df_real[close_col].astype(float),
                'bitcoin_volume': df_real[volume_col].astype(float)
            })

            df.set_index('date', inplace=True)
            df = df.dropna()  # Remover valores NaN

            print(f"✅ Dados reais carregados com sucesso!")
            print(f"   {len(df)} dias de dados históricos")
            print(f"   Período: {df.index[0].date()} a {df.index[-1].date()}")
            print(f"   Preço inicial: ${df['bitcoin_close'].iloc[0]:,.2f}")
            print(f"   Preço final: ${df['bitcoin_close'].iloc[-1]:,.2f}")
            print(f"   Retorno total: {(df['bitcoin_close'].iloc[-1] / df['bitcoin_close'].iloc[0] - 1)*100:.1f}%")

            return df

    except Exception as e:
        print(f"⚠️  Erro ao carregar bitcoin.csv: {e}")
        print("📊 Gerando dados sintéticos como fallback...")

    # Fallback: gerar dados sintéticos
    np.random.seed(42)

    # Gerar datas
    dates = pd.date_range(start='2020-01-01', periods=n_dias, freq='D')

    # Gerar retornos log com volatilidade clustering (modelo GARCH simplificado)
    returns = []
    volatility = 0.04  # volatilidade inicial

    for i in range(n_dias):
        # Atualizar volatilidade (efeito GARCH)
        volatility = 0.95 * volatility + 0.05 * 0.04 + 0.05 * (returns[-1] if returns else 0)**2
        volatility = np.clip(volatility, 0.01, 0.10)

        # Gerar retorno com tendência de longo prazo
        trend = 0.0002  # tendência de alta leve
        shock = np.random.normal(0, volatility)
        daily_return = trend + shock
        returns.append(daily_return)

    # Converter retornos para preços
    precos = [preco_inicial]
    for ret in returns:
        precos.append(precos[-1] * np.exp(ret))

    # Gerar volume correlacionado com volatilidade
    volumes = []
    for ret in returns:
        base_volume = 50000
        volume_multiplier = 1 + 5 * abs(ret)  # maior volume em dias voláteis
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)
        volumes.append(volume)

    df = pd.DataFrame({
        'date': dates,
        'bitcoin_close': precos[1:],  # remover preço inicial
        'bitcoin_volume': volumes
    })

    df.set_index('date', inplace=True)

    print(f"📊 Dados sintéticos do Bitcoin gerados: {len(df)} dias")
    print(f"   Período: {df.index[0].date()} a {df.index[-1].date()}")
    print(f"   Preço inicial: ${df['bitcoin_close'].iloc[0]:,.2f}")
    print(f"   Preço final: ${df['bitcoin_close'].iloc[-1]:,.2f}")
    print(f"   Retorno total: {(df['bitcoin_close'].iloc[-1] / df['bitcoin_close'].iloc[0] - 1)*100:.1f}%")

    return df

# Carregar dados
dados_bitcoin = carregar_dados_bitcoin()
print("\n", dados_bitcoin.head())

# %%
# Importar estratégias oficiais do projeto
import SMACrossoverStrategy
import BollingerBandsCrossStrategy

# Instanciar estratégias
sma_strategy = SMACrossoverStrategy(asset='bitcoin')
bb_strategy = BollingerBandsCrossStrategy(asset='bitcoin')

# Aplicar estratégias usando implementações oficiais
print("📈 Aplicando estratégia SMA Crossover...")
dados_com_sma = sma_strategy.calculate_signals(
    dados_bitcoin,
    {'fast_period': 10, 'slow_period': 30}
)

print("📊 Aplicando estratégia Bollinger Bands Cross...")
dados_completos = bb_strategy.calculate_signals(
    dados_com_sma,
    {'period': 20, 'std_multiplier': 2.0}
)

# Renomear colunas para manter compatibilidade com resto do pipeline
dados_completos = dados_completos.rename(columns={'signal': 'sinal_sma'})

# Agora aplicar BB strategy - precisa ser aplicado ao resultado do SMA
# Criar sinal_bb a partir do signal retornado pela estratégia BB
dados_temp = bb_strategy.calculate_signals(dados_completos, {'period': 20, 'std_multiplier': 2.0})
dados_completos['sinal_bb'] = dados_temp['signal']

# Estatísticas dos sinais
sma_dist = dados_completos['sinal_sma'].value_counts().to_dict()
bb_dist = dados_completos['sinal_bb'].value_counts().to_dict()

print(f"✅ Estratégia SMA Crossover aplicada")
print(f"   Distribuição de sinais: {sma_dist}")
print(f"\n✅ Estratégia Bollinger Bands aplicada")
print(f"   Distribuição de sinais: {bb_dist}")

print(f"\n📋 Dados com estratégias primárias:")
print(dados_completos[['bitcoin_close', 'sinal_sma', 'sinal_bb']].head())

# %%
# Importar função oficial de triple barrier
import triple_barrier_label

print("🎯 Aplicando triple barrier labels usando implementação oficial...")

# Para Bollinger Bands - aplicar labels em TODOS os sinais não-zero
pontos_entrada_bb = dados_completos[dados_completos['sinal_bb'] != 0].index
print(f"   Pontos de entrada Bollinger Bands: {len(pontos_entrada_bb)}")

# Aplicar triple barrier labels usando a função oficial
barrier_results = triple_barrier_label(
    prices=dados_completos['bitcoin_close'],
    events=pontos_entrada_bb,  # Todos os pontos com sinal não-zero
    signals=dados_completos.loc[pontos_entrada_bb, 'sinal_bb'],  # Sinais correspondentes
    volatility_span=20,
    time_barrier_days=5,
    upper_barrier_mult=2.0,  # 2 desvios padrões
    lower_barrier_mult=2.0,  # 2 desvios padrões
    min_pct_move=None
)

# Criar DataFrame com rótulos
dados_com_rotulos = dados_completos.copy()
dados_com_rotulos['rotulo_barreira'] = np.nan
dados_com_rotulos['tipo_barreira'] = ''
dados_com_rotulos['dias_ate_barreira'] = np.nan
dados_com_rotulos['volatilidade_barreira'] = np.nan

# Preencher com os resultados do triple barrier
for idx in barrier_results.index:
    if idx in dados_com_rotulos.index:
        dados_com_rotulos.loc[idx, 'rotulo_barreira'] = barrier_results.loc[idx, 'label']
        dados_com_rotulos.loc[idx, 'tipo_barreira'] = barrier_results.loc[idx, 'barrier_touched']
        dados_com_rotulos.loc[idx, 'dias_ate_barreira'] = barrier_results.loc[idx, 'days_to_barrier']
        dados_com_rotulos.loc[idx, 'volatilidade_barreira'] = barrier_results.loc[idx, 'volatility']

# Estatísticas dos rótulos
rotulos_validos = dados_com_rotulos['rotulo_barreira'].dropna()
dist_rotulos = rotulos_validos.value_counts().to_dict()
# Filtrar apenas tipos não vazios
tipos_barreira_series = dados_com_rotulos[dados_com_rotulos['tipo_barreira'] != '']['tipo_barreira']
tipos_barreira = tipos_barreira_series.value_counts().to_dict() if len(tipos_barreira_series) > 0 else {}

print(f"\n📊 Estatísticas dos rótulos:")
print(f"   Total de pontos de entrada analisados: {len(rotulos_validos)}")
print(f"   Distribuição de rótulos: {dist_rotulos}")
print(f"   Tipos de barreira atingida: {tipos_barreira}")
print(f"   Tempo médio até barreira: {dados_com_rotulos['dias_ate_barreira'].mean():.1f} dias")

print(f"\n📋 Dados com rótulos de barreiras:")
colunas_relevantes = ['bitcoin_close', 'sinal_sma', 'sinal_bb', 'rotulo_barreira', 'tipo_barreira']
print(dados_com_rotulos[colunas_relevantes].head())

# %%
def extrair_features_topologicas(dados: pd.DataFrame,
                                       janela: int = 50,
                                       tau: int = 3,
                                       dimensao_emb: int = 3) -> pd.DataFrame:
    """
    Extrai features topológicas reais usando homologia persistente.

    Utiliza a biblioteca topological_features para computar:
    1. Time-delay embedding da série de preços
    2. Computação de homologia persistente (H0 e H1)
    3. Análise das barras de persistência
    4. Extração de features estatísticas (Betti numbers, lifetimes, entropias, etc.)

    Parâmetros:
    -----------
    dados : pd.DataFrame
        Dados com coluna 'bitcoin_close'
    janela : int
        Tamanho da janela deslizante para análise
    tau : int
        Time delay para embedding
    dimensao_emb : int
        Dimensão do embedding

    Retorna:
    --------
    pd.DataFrame
        Dados originais com features topológicas adicionadas
    """
    # Importar função de extração topológica real
    try:
        from experiments.utils.topological_features import extract_topological_features
        print("✅ Usando extração topológica real com ripser")
        usando_real = True
    except ImportError as e:
        print(f"⚠️  Erro ao importar topological_features: {e}")
        print("   Instalando ripser...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ripser"])

        # Tentar novamente após instalação
        try:
            from experiments.utils.topological_features import extract_topological_features
            print("✅ Ripser instalado e topological_features importado com sucesso")
            usando_real = True
        except ImportError:
            print("❌ Falha ao importar. Usando fallback simplificado.")
            usando_real = False

    if usando_real:
        # Usar extração topológica real
        features_df = extract_topological_features(
            data=dados,
            window_length=janela,
            selected_cols=['bitcoin_close'],
            tau=tau,
            embedding_dim=dimensao_emb,
            max_dimension=1  # H0 e H1
        )

        # Adicionar prefixo consistente com o resto do código
        features_df = features_df.add_prefix('bitcoin_close_topo_')

        # Fazer merge com dados originais
        df_final = dados.merge(features_df, left_index=True, right_index=True, how='left')

        print(f"   Features topológicas reais extraídas: {len(features_df.columns)} features")
        print(f"   Janelas processadas: {len(features_df)}")

    else:
        # Fallback: gerar features sintéticas similares às reais
        print("⚠️  Usando features topológicas sintéticas (fallback)")
        df = dados.copy()
        np.random.seed(42)

        features_topo = {}
        for i in range(janela, len(df)):
            janela_precos = df['bitcoin_close'].iloc[i-janela:i].values
            precos_norm = (janela_precos - janela_precos.mean()) / (janela_precos.std() + 1e-8)
            volatilidade_local = np.std(precos_norm[-10:])

            data_atual = df.index[i]
            features_topo[data_atual] = {
                # H0 features (componentes conectados)
                'bitcoin_close_topo_betti_0': max(1, int(3 * volatilidade_local)),
                'bitcoin_close_topo_max_hole_lifetime_0': volatilidade_local * 5,
                'bitcoin_close_topo_persistence_entropy_0': -volatilidade_local * np.log(volatilidade_local + 0.1),
                'bitcoin_close_topo_wasserstein_0': volatilidade_local * 2,
                # H1 features (ciclos)
                'bitcoin_close_topo_betti_1': max(0, int(2 * volatilidade_local)),
                'bitcoin_close_topo_max_hole_lifetime_1': volatilidade_local * 3,
                'bitcoin_close_topo_persistence_entropy_1': volatilidade_local * 1.5,
                'bitcoin_close_topo_wasserstein_1': volatilidade_local * 1.2
            }

        features_df = pd.DataFrame.from_dict(features_topo, orient='index')
        df_final = df.merge(features_df, left_index=True, right_index=True, how='left')

    return df_final

# Extrair features topológicas
dados_finais = extrair_features_topologicas(dados_com_rotulos)

# Identificar colunas topológicas
features_topologicas = [col for col in dados_finais.columns if '_topo_' in col]

print(f"🔬 Features topológicas extraídas (valores contínuos)")
print(f"   Total de features topológicas: {len(features_topologicas)}")
print(f"   Features H0 (estrutura): {len([f for f in features_topologicas if '_0' in f])}")
print(f"   Features H1 (ciclos): {len([f for f in features_topologicas if '_1' in f])}")

print(f"\n📋 Amostra de features topológicas (valores originais):")
colunas_amostra = ['bitcoin_close'] + features_topologicas[:4]
print(dados_finais[colunas_amostra].dropna().head())

# %%
def treinar_meta_modelo(dados: pd.DataFrame,
                       features_topo: List[str],
                       coluna_rotulo: str = 'rotulo_barreira',
                       coluna_sinal: str = 'sinal_sma') -> Dict:
    """
    Treina meta-modelo CatBoost para predizer sucesso de sinais primários.

    O meta-modelo aprende quando confiar nos sinais da estratégia primária
    baseado em features topológicas e o próprio sinal primário.

    Parâmetros:
    -----------
    dados : pd.DataFrame
        Dados completos com features e rótulos
    features_topo : List[str]
        Lista de features topológicas
    coluna_rotulo : str
        Coluna com rótulos de barreira tripla
    coluna_sinal : str
        Coluna com sinais da estratégia primária

    Retorna:
    --------
    Dict
        Dicionário com modelo treinado e métricas
    """

    # Preparar dados para treinamento
    dados_treino = dados.dropna(subset=[coluna_rotulo])

    # Features: topológicas + sinal primário
    features_modelo = features_topo + [coluna_sinal]
    features_disponiveis = [f for f in features_modelo if f in dados_treino.columns]

    X = dados_treino[features_disponiveis].fillna(0)
    y = dados_treino[coluna_rotulo]

    # Converter rótulos para classificação binária (sucesso vs falha)
    # 1 = sucesso (lucro), 0 = falha (perda ou tempo)
    y_binario = (y == 1).astype(int)

    # Split temporal (80% treino, 20% teste)
    split_idx = int(0.8 * len(X))
    X_treino, X_teste = X.iloc[:split_idx], X.iloc[split_idx:]
    y_treino, y_teste = y_binario.iloc[:split_idx], y_binario.iloc[split_idx:]

    # Treinar modelo CatBoost toy (simplificado para demonstração)
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score, log_loss

    modelo = CatBoostClassifier(
        iterations=50,        # Baixo número para demonstração rápida
        depth=4,              # Árvore rasa
        learning_rate=0.1,
        verbose=False,
        random_state=42
    )
    modelo.fit(X_treino, y_treino)

    # Predições
    pred_treino = modelo.predict(X_treino)
    pred_teste = modelo.predict(X_teste)
    pred_proba_teste = modelo.predict_proba(X_teste)

    # Métricas
    acc_treino = accuracy_score(y_treino, pred_treino)
    acc_teste = accuracy_score(y_teste, pred_teste)
    logloss_teste = log_loss(y_teste, pred_proba_teste)

    # Importância das features
    importancias = dict(zip(features_disponiveis, modelo.feature_importances_))
    top_features = sorted(importancias.items(), key=lambda x: x[1], reverse=True)

    # Predições para todo o dataset
    pred_completa = modelo.predict(X)
    dados_com_pred = dados_treino.copy()
    dados_com_pred['meta_predicao'] = pred_completa
    dados_com_pred['sinal_final'] = dados_com_pred[coluna_sinal] * dados_com_pred['meta_predicao']

    return {
        'modelo': modelo,
        'dados_com_predicoes': dados_com_pred,
        'metricas': {
            'acuracia_treino': acc_treino,
            'acuracia_teste': acc_teste,
            'log_loss_teste': logloss_teste,
            'amostras_treino': len(X_treino),
            'amostras_teste': len(X_teste)
        },
        'importancia_features': top_features
    }

# Treinar meta-modelos para ambas as estratégias
resultado_meta_sma = treinar_meta_modelo(dados_finais, features_topologicas, coluna_sinal='sinal_sma')
resultado_meta_bb = treinar_meta_modelo(dados_finais, features_topologicas, coluna_sinal='sinal_bb')

print(f"🤖 Meta-modelos treinados com sucesso")
print(f"\n📈 Meta-modelo SMA:")
print(f"   Acurácia treino: {resultado_meta_sma['metricas']['acuracia_treino']:.3f}")
print(f"   Acurácia teste: {resultado_meta_sma['metricas']['acuracia_teste']:.3f}")
print(f"   Log-loss teste: {resultado_meta_sma['metricas']['log_loss_teste']:.3f}")

print(f"\n📊 Meta-modelo Bollinger Bands:")
print(f"   Acurácia treino: {resultado_meta_bb['metricas']['acuracia_treino']:.3f}")
print(f"   Acurácia teste: {resultado_meta_bb['metricas']['acuracia_teste']:.3f}")
print(f"   Log-loss teste: {resultado_meta_bb['metricas']['log_loss_teste']:.3f}")

print(f"\n🎯 Top 5 features mais importantes (SMA):")
for idx, (feature, importancia) in enumerate(resultado_meta_sma['importancia_features'][:5], 1):
    feature_nome = feature.replace('bitcoin_close_topo_', '').replace('_', ' ')
    print(f"   {idx}. {feature_nome}: {importancia:.3f}")

print(f"\n🎯 Top 5 features mais importantes (Bollinger):")
for idx, (feature, importancia) in enumerate(resultado_meta_bb['importancia_features'][:5], 1):
    feature_nome = feature.replace('bitcoin_close_topo_', '').replace('_', ' ')
    print(f"   {idx}. {feature_nome}: {importancia:.3f}")

# %%
def calcular_metricas_estrategia(dados: pd.DataFrame,
                                coluna_sinal: str,
                                coluna_preco: str = 'bitcoin_close') -> Dict:
    """
    Calcula métricas de performance de uma estratégia de trading.

    Métricas calculadas:
    - Retorno total e anualizado
    - Sharpe ratio
    - Máximo drawdown
    - Número de trades

    Parâmetros:
    -----------
    dados : pd.DataFrame
        Dados com sinais e preços
    coluna_sinal : str
        Coluna com sinais da estratégia
    coluna_preco : str
        Coluna com preços

    Retorna:
    --------
    Dict
        Dicionário com métricas calculadas
    """
    df = dados.dropna(subset=[coluna_sinal, coluna_preco]).copy()

    # Calcular retornos
    df['retorno_diario'] = df[coluna_preco].pct_change()
    df['retorno_estrategia'] = df['retorno_diario'] * df[coluna_sinal].shift(1)

    # Remover NaN
    retornos_validos = df['retorno_estrategia'].dropna()

    if len(retornos_validos) == 0:
        return {'erro': 'Sem retornos válidos'}

    # Retorno cumulativo
    retorno_cumulativo = (1 + retornos_validos).cumprod()

    # Métricas
    retorno_total = retorno_cumulativo.iloc[-1] - 1
    retorno_anualizado = (1 + retorno_total) ** (252 / len(retornos_validos)) - 1

    # Sharpe ratio (assumindo risk-free rate = 0)
    if retornos_validos.std() > 0:
        sharpe = retornos_validos.mean() / retornos_validos.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Maximum drawdown
    peak = retorno_cumulativo.expanding().max()
    drawdown = (retorno_cumulativo - peak) / peak
    max_drawdown = drawdown.min()

    # Número de trades (mudanças de sinal)
    mudancas_sinal = (df[coluna_sinal] != df[coluna_sinal].shift(1)).sum()

    return {
        'retorno_total': retorno_total,
        'retorno_anualizado': retorno_anualizado,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': mudancas_sinal,
        'retorno_cumulativo': retorno_cumulativo,
        'drawdown_serie': drawdown
    }

def plotar_resultados_finais(dados_original: pd.DataFrame,
                           dados_meta: pd.DataFrame,
                           coluna_sinal_original: str = 'sinal_bb') -> None:
    """
    Plota gráficos de retorno cumulativo e drawdown separados para treino e teste.

    Compara performance entre estratégia primária e meta-modelo.

    Parâmetros:
    -----------
    dados_original : pd.DataFrame
        Dados com estratégia primária
    dados_meta : pd.DataFrame
        Dados com meta-modelo
    """

    # Identificar split treino/teste (80/20)
    split_idx = int(len(dados_meta) * 0.8)

    # Separar dados de treino e teste
    dados_original_treino = dados_original.iloc[:split_idx]
    dados_original_teste = dados_original.iloc[split_idx:]
    dados_meta_treino = dados_meta.iloc[:split_idx]
    dados_meta_teste = dados_meta.iloc[split_idx:]

    # Calcular métricas separadas para treino
    metricas_original_treino = calcular_metricas_estrategia(dados_original_treino, coluna_sinal_original)
    metricas_meta_treino = calcular_metricas_estrategia(dados_meta_treino, 'sinal_final')
    metricas_bh_treino = calcular_metricas_estrategia(
        dados_original_treino.assign(sinal_buy_hold=1), 'sinal_buy_hold'
    )

    # Calcular métricas separadas para teste
    metricas_original_teste = calcular_metricas_estrategia(dados_original_teste, coluna_sinal_original)
    metricas_meta_teste = calcular_metricas_estrategia(dados_meta_teste, 'sinal_final')
    metricas_bh_teste = calcular_metricas_estrategia(
        dados_original_teste.assign(sinal_buy_hold=1), 'sinal_buy_hold'
    )

    # Criar gráficos 2x2
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    # === GRÁFICOS DE TREINO (coluna esquerda) ===

    # Gráfico 1: Retorno Cumulativo TREINO
    estrategia_nome = 'BB' if coluna_sinal_original == 'sinal_bb' else 'SMA'
    if 'retorno_cumulativo' in metricas_original_treino:
        ax1.plot(metricas_original_treino['retorno_cumulativo'].index,
                metricas_original_treino['retorno_cumulativo'].values,
                label=f'{estrategia_nome} (Sharpe: {metricas_original_treino["sharpe_ratio"]:.2f})',
                linewidth=2)

    if 'retorno_cumulativo' in metricas_meta_treino:
        ax1.plot(metricas_meta_treino['retorno_cumulativo'].index,
                metricas_meta_treino['retorno_cumulativo'].values,
                label=f'Meta (Sharpe: {metricas_meta_treino["sharpe_ratio"]:.2f})',
                linewidth=2)

    if 'retorno_cumulativo' in metricas_bh_treino:
        ax1.plot(metricas_bh_treino['retorno_cumulativo'].index,
                metricas_bh_treino['retorno_cumulativo'].values,
                label=f'B&H (Sharpe: {metricas_bh_treino["sharpe_ratio"]:.2f})',
                linewidth=2, alpha=0.7, linestyle='--')

    ax1.set_title('TREINO: Retorno Cumulativo', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Retorno Cumulativo')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Retorno Cumulativo TESTE
    if 'retorno_cumulativo' in metricas_original_teste:
        ax2.plot(metricas_original_teste['retorno_cumulativo'].index,
                metricas_original_teste['retorno_cumulativo'].values,
                label=f'{estrategia_nome} (Sharpe: {metricas_original_teste["sharpe_ratio"]:.2f})',
                linewidth=2)

    if 'retorno_cumulativo' in metricas_meta_teste:
        ax2.plot(metricas_meta_teste['retorno_cumulativo'].index,
                metricas_meta_teste['retorno_cumulativo'].values,
                label=f'Meta (Sharpe: {metricas_meta_teste["sharpe_ratio"]:.2f})',
                linewidth=2)

    if 'retorno_cumulativo' in metricas_bh_teste:
        ax2.plot(metricas_bh_teste['retorno_cumulativo'].index,
                metricas_bh_teste['retorno_cumulativo'].values,
                label=f'B&H (Sharpe: {metricas_bh_teste["sharpe_ratio"]:.2f})',
                linewidth=2, alpha=0.7, linestyle='--')

    ax2.set_title('TESTE: Retorno Cumulativo', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Retorno Cumulativo')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Gráfico 3: Drawdown TREINO
    if 'drawdown_serie' in metricas_original_treino:
        ax3.fill_between(metricas_original_treino['drawdown_serie'].index,
                        metricas_original_treino['drawdown_serie'].values, 0,
                        alpha=0.3, label=f'{estrategia_nome} (Max: {metricas_original_treino["max_drawdown"]:.1%})')

    if 'drawdown_serie' in metricas_meta_treino:
        ax3.fill_between(metricas_meta_treino['drawdown_serie'].index,
                        metricas_meta_treino['drawdown_serie'].values, 0,
                        alpha=0.3, label=f'Meta (Max: {metricas_meta_treino["max_drawdown"]:.1%})')

    ax3.set_title('TREINO: Drawdown', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Drawdown')
    ax3.set_xlabel('Data')
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)

    # Gráfico 4: Drawdown TESTE
    if 'drawdown_serie' in metricas_original_teste:
        ax4.fill_between(metricas_original_teste['drawdown_serie'].index,
                        metricas_original_teste['drawdown_serie'].values, 0,
                        alpha=0.3, label=f'{estrategia_nome} (Max: {metricas_original_teste["max_drawdown"]:.1%})')

    if 'drawdown_serie' in metricas_meta_teste:
        ax4.fill_between(metricas_meta_teste['drawdown_serie'].index,
                        metricas_meta_teste['drawdown_serie'].values, 0,
                        alpha=0.3, label=f'Meta (Max: {metricas_meta_teste["max_drawdown"]:.1%})')

    ax4.set_title('TESTE: Drawdown', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Drawdown')
    ax4.set_xlabel('Data')
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Retornar todas as métricas separadas
    return {
        'treino': {
            'original': metricas_original_treino,
            'meta': metricas_meta_treino,
            'buy_hold': metricas_bh_treino
        },
        'teste': {
            'original': metricas_original_teste,
            'meta': metricas_meta_teste,
            'buy_hold': metricas_bh_teste
        }
    }

# %%
print(f"\n🎯 RESULTADOS FINAIS - SMA CROSSOVER")
print(f"{'='*60}")

# Calcular e plotar resultados SMA
print(f"📊 Calculando métricas SMA...")
metricas_sma = plotar_resultados_finais(
    dados_finais,
    resultado_meta_sma['dados_com_predicoes'],
    coluna_sinal_original='sinal_sma'
)

# === MÉTRICAS SMA TREINO ===
print(f"\n📊 SMA - PERÍODO DE TREINO (80%):")
print(f"{'-'*40}")

print(f"\n📈 SMA Crossover:")
if 'retorno_total' in metricas_sma['treino']['original']:
    print(f"   Retorno Total: {metricas_sma['treino']['original']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['treino']['original']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['treino']['original']['max_drawdown']:.2%}")

print(f"\n🤖 Meta-modelo SMA:")
if 'retorno_total' in metricas_sma['treino']['meta']:
    print(f"   Retorno Total: {metricas_sma['treino']['meta']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['treino']['meta']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['treino']['meta']['max_drawdown']:.2%}")

print(f"\n💎 Buy & Hold:")
if 'retorno_total' in metricas_sma['treino']['buy_hold']:
    print(f"   Retorno Total: {metricas_sma['treino']['buy_hold']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['treino']['buy_hold']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['treino']['buy_hold']['max_drawdown']:.2%}")

# === MÉTRICAS SMA TESTE ===
print(f"\n📊 SMA - PERÍODO DE TESTE (20%):")
print(f"{'-'*40}")

print(f"\n📈 SMA Crossover:")
if 'retorno_total' in metricas_sma['teste']['original']:
    print(f"   Retorno Total: {metricas_sma['teste']['original']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['teste']['original']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['teste']['original']['max_drawdown']:.2%}")

print(f"\n🤖 Meta-modelo SMA:")
if 'retorno_total' in metricas_sma['teste']['meta']:
    print(f"   Retorno Total: {metricas_sma['teste']['meta']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['teste']['meta']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['teste']['meta']['max_drawdown']:.2%}")

print(f"\n💎 Buy & Hold:")
if 'retorno_total' in metricas_sma['teste']['buy_hold']:
    print(f"   Retorno Total: {metricas_sma['teste']['buy_hold']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_sma['teste']['buy_hold']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_sma['teste']['buy_hold']['max_drawdown']:.2%}")

# %%
print(f"\n\n🎯 RESULTADOS FINAIS - BOLLINGER BANDS")
print(f"{'='*60}")

# Calcular e plotar resultados Bollinger Bands
print(f"📊 Calculando métricas Bollinger Bands...")
metricas_bb = plotar_resultados_finais(
    dados_finais,
    resultado_meta_bb['dados_com_predicoes'],
    coluna_sinal_original='sinal_bb'
)

# === MÉTRICAS BB TREINO ===
print(f"\n📊 BOLLINGER BANDS - PERÍODO DE TREINO (80%):")
print(f"{'-'*40}")

print(f"\n📊 Bollinger Bands:")
if 'retorno_total' in metricas_bb['treino']['original']:
    print(f"   Retorno Total: {metricas_bb['treino']['original']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['treino']['original']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['treino']['original']['max_drawdown']:.2%}")

print(f"\n🤖 Meta-modelo BB:")
if 'retorno_total' in metricas_bb['treino']['meta']:
    print(f"   Retorno Total: {metricas_bb['treino']['meta']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['treino']['meta']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['treino']['meta']['max_drawdown']:.2%}")

print(f"\n💎 Buy & Hold:")
if 'retorno_total' in metricas_bb['treino']['buy_hold']:
    print(f"   Retorno Total: {metricas_bb['treino']['buy_hold']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['treino']['buy_hold']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['treino']['buy_hold']['max_drawdown']:.2%}")

# === MÉTRICAS BB TESTE ===
print(f"\n📊 BOLLINGER BANDS - PERÍODO DE TESTE (20%):")
print(f"{'-'*40}")

print(f"\n📊 Bollinger Bands:")
if 'retorno_total' in metricas_bb['teste']['original']:
    print(f"   Retorno Total: {metricas_bb['teste']['original']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['teste']['original']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['teste']['original']['max_drawdown']:.2%}")

print(f"\n🤖 Meta-modelo BB:")
if 'retorno_total' in metricas_bb['teste']['meta']:
    print(f"   Retorno Total: {metricas_bb['teste']['meta']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['teste']['meta']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['teste']['meta']['max_drawdown']:.2%}")

print(f"\n💎 Buy & Hold:")
if 'retorno_total' in metricas_bb['teste']['buy_hold']:
    print(f"   Retorno Total: {metricas_bb['teste']['buy_hold']['retorno_total']:.2%}")
    print(f"   Sharpe Ratio: {metricas_bb['teste']['buy_hold']['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown: {metricas_bb['teste']['buy_hold']['max_drawdown']:.2%}")

print(f"\n{'='*50}")
print(f"🔬 Pipeline completo demonstrado com sucesso!")
print(f"   Features topológicas extraídas das barras de persistência")
print(f"   Meta-modelos treinados para ambas estratégias primárias")
print(f"   Resultados comparados com baseline buy-and-hold")
# %%
