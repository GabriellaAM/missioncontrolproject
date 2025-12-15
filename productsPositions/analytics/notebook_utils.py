"""
Utilitários para uso em notebooks Jupyter
Facilita a visualização e análise dos dados de produtos e posições
"""
import pandas as pd
import sys
from pathlib import Path

# Adicionar o diretório pai ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.queries import *
from storage.sqlite_repo import SQLiteRepo

def display_produtos():
    """
    Exibe todos os produtos disponíveis em formato de tabela
    
    Returns:
        pd.DataFrame: DataFrame com todos os produtos
    """
    repo = SQLiteRepo()
    produtos = repo.listar_produtos()
    
    if not produtos:
        print("Nenhum produto encontrado")
        return pd.DataFrame()
    
    df = pd.DataFrame(produtos)
    return df

def display_posicoes_abertas(produto_id=None, formatar=True):
    """
    Exibe posições abertas em formato de tabela com atributos do produto
    
    Args:
        produto_id: ID do produto (opcional, None para todos)
        formatar: Se True, formata valores monetários e atributos
    
    Returns:
        pd.DataFrame: DataFrame com posições abertas e atributos
    """
    df = posicoes_abertas(produto_id)
    
    if df.empty:
        print("Nenhuma posição aberta encontrada")
        return df
    
    if formatar:
        # Função auxiliar para formatar valor + % de distância até o alvo/stop
        def _formatar_valor_com_percent(row, coluna_alvo):
            # Aqui usamos os valores numéricos originais (antes de formatar como string)
            if 'preco_atual' not in row or pd.isna(row['preco_atual']):
                return ""
            valor = row[coluna_alvo]
            preco_atual = row['preco_atual']
            if pd.isna(valor) or pd.isna(preco_atual) or preco_atual == 0:
                return ""
            try:
                pct = ((valor / preco_atual) - 1) * 100
            except ZeroDivisionError:
                return f"${valor:,.2f}"
            # Exemplo desejado: $0.24 (-15.20%)
            sinal_pct = f"{pct:.2f}%"
            return f"${valor:,.2f} ({sinal_pct})"

        # Formatar valores monetários principais
        if 'preco_entrada' in df.columns:
            df['preco_entrada'] = df['preco_entrada'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else ""
            )

        # Formatar atributos do produto com porcentagem relativa ao preço atual
        if 'alvo1' in df.columns and 'preco_atual' in df.columns:
            df['alvo1'] = df.apply(lambda row: _formatar_valor_com_percent(row, 'alvo1'), axis=1)
        if 'alvo2' in df.columns and 'preco_atual' in df.columns:
            df['alvo2'] = df.apply(lambda row: _formatar_valor_com_percent(row, 'alvo2'), axis=1)
        if 'stop_atual' in df.columns and 'preco_atual' in df.columns:
            df['stop_atual'] = df.apply(lambda row: _formatar_valor_com_percent(row, 'stop_atual'), axis=1)

        # Agora, depois de usar os valores numéricos para as porcentagens, formatar preco_atual como string
        if 'preco_atual' in df.columns:
            df['preco_atual'] = df['preco_atual'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
            )

        # RR é calculado dinamicamente, formatar se existir
        if 'rr' in df.columns:
            df['rr'] = df['rr'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else ""
            )
        # PnL em porcentagem, se existir (apenas Crypto Signals)
        if 'pnl' in df.columns:
            df['pnl'] = df['pnl'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else ""
            )
        if 'perfil' in df.columns:
            df['perfil'] = df['perfil'].apply(lambda x: x if pd.notna(x) else "")
        # Para posições abertas, motivo não é relevante, então ignoramos

    # Ordenar pela data de entrada (da mais antiga para a mais nova), se existir
    if 'data_entrada' in df.columns:
        try:
            df['_data_entrada_sort'] = pd.to_datetime(df['data_entrada'])
        except Exception:
            df['_data_entrada_sort'] = df['data_entrada']
        df = df.sort_values('_data_entrada_sort').drop(columns=['_data_entrada_sort'])

    # Remover colunas que não fazem sentido na visualização de posições abertas
    colunas_para_remover = ['produto_id', 'coingecko_id', 'data_saida', 'preco_saida', 'status', 'motivo']
    df = df.drop(columns=[c for c in colunas_para_remover if c in df.columns])

    # Para o produto Crypto Signals, usar exatamente as colunas e ordem solicitadas
    if produto_id == 4970919917:
        colunas_signals = [
            'data_entrada',
            'ativo',
            'side',
            'perfil',
            'preco_entrada',
            'preco_atual',
            'pnl',
            'alvo1',
            'alvo2',
            'stop_atual',
            'rr',
        ]
        colunas_existentes = [c for c in colunas_signals if c in df.columns]
        df = df[colunas_existentes]
        return df

    # Para outros produtos, manter ordenação mais genérica
    colunas_ordenadas = []
    colunas_principais = ['id', 'ativo', 'side', 'data_entrada', 'preco_entrada', 'preco_atual']
    colunas_atributos = ['perfil', 'pnl', 'rr', 'alvo1', 'alvo2', 'stop_atual']
    colunas_outras = [col for col in df.columns if col not in colunas_atributos + colunas_principais]
    
    for col in colunas_principais + colunas_atributos + colunas_outras:
        if col in df.columns:
            colunas_ordenadas.append(col)
    
    df = df[colunas_ordenadas]
    return df

def display_carteira(produto_id, formatar=True):
    """
    Exibe carteira de um produto em formato de tabela
    
    Args:
        produto_id: ID do produto
        formatar: Se True, formata valores monetários
    
    Returns:
        pd.DataFrame: DataFrame com informações da carteira
    """
    df = carteira_do_produto(produto_id)
    
    if df.empty:
        print(f"Nenhuma carteira encontrada para o produto {produto_id}")
        return df
    
    if formatar:
        colunas_monetarias = ['valor_disponivel', 'valor_investido', 'pnl_nao_realizado', 'valor_total']
        for col in colunas_monetarias:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
    
    return df

def display_alocacoes(produto_id, formatar=True):
    """
    Exibe alocações de um produto com informações das posições
    
    Args:
        produto_id: ID do produto
        formatar: Se True, formata valores monetários e percentuais
    
    Returns:
        pd.DataFrame: DataFrame com alocações
    """
    df = resumo_alocacoes(produto_id)
    
    if df.empty:
        print(f"Nenhuma alocação encontrada para o produto {produto_id}")
        return df
    
    if formatar:
        if 'valor_usd' in df.columns:
            df['valor_usd'] = df['valor_usd'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
        if 'percentual' in df.columns:
            df['percentual'] = df['percentual'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "0.00%")
    
    return df

def display_resumo_completo(produto_id, formatar=True):
    """
    Exibe resumo completo de um produto (produto + carteira)
    
    Args:
        produto_id: ID do produto
        formatar: Se True, formata valores monetários
    
    Returns:
        pd.DataFrame: DataFrame com resumo completo
    """
    df = resumo_completo_produto(produto_id)
    
    if df.empty:
        print(f"Produto {produto_id} não encontrado")
        return df
    
    if formatar:
        colunas_monetarias = ['capital_inicial', 'valor_disponivel', 'valor_investido', 
                              'pnl_nao_realizado', 'valor_total']
        for col in colunas_monetarias:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "$0.00")
    
    return df

def get_valores_ativo(ativo, data_inicio=None, data_fim=None):
    """
    Obtém valores diários de um ativo (pronto para gráficos)
    
    Args:
        ativo: Nome do ativo (ex: "BTC")
        data_inicio: Data inicial (YYYY-MM-DD) - opcional
        data_fim: Data final (YYYY-MM-DD) - opcional
    
    Returns:
        pd.DataFrame: DataFrame com colunas 'data' e 'preco'
    """
    df = valores_do_ativo(ativo, data_inicio, data_fim)
    
    if not df.empty and 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'])
        df = df.sort_values('data')
    
    return df

def display_posicoes_fechadas(produto_id=None, formatar=True):
    """
    Exibe posições fechadas em formato de tabela com atributos do produto
    
    Args:
        produto_id: ID do produto (opcional, None para todos)
        formatar: Se True, formata valores monetários e atributos
    
    Returns:
        pd.DataFrame: DataFrame com posições fechadas e atributos
    """
    df = posicoes_fechadas(produto_id)
    
    if df.empty:
        print("Nenhuma posição fechada encontrada")
        return df
    
    if formatar:
        # Formatar valores monetários
        if 'preco_entrada' in df.columns:
            df['preco_entrada'] = df['preco_entrada'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        if 'preco_saida' in df.columns:
            df['preco_saida'] = df['preco_saida'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        if 'preco_atual' in df.columns:
            df['preco_atual'] = df['preco_atual'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        
        # Formatar atributos do produto
        if 'alvo1' in df.columns:
            df['alvo1'] = df['alvo1'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        if 'alvo2' in df.columns:
            df['alvo2'] = df['alvo2'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        if 'rr' in df.columns:
            df['rr'] = df['rr'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        if 'pnl' in df.columns:
            df['pnl'] = df['pnl'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "")
        if 'perfil' in df.columns:
            df['perfil'] = df['perfil'].apply(lambda x: x if pd.notna(x) else "")
        if 'motivo' in df.columns:
            df['motivo'] = df['motivo'].apply(lambda x: x if pd.notna(x) else "")

    # Ordenar pela data de entrada (mais antiga -> mais nova), se existir
    if 'data_entrada' in df.columns:
        try:
            df['_data_entrada_sort'] = pd.to_datetime(df['data_entrada'])
        except Exception:
            df['_data_entrada_sort'] = df['data_entrada']
        df = df.sort_values('_data_entrada_sort').drop(columns=['_data_entrada_sort'])

    # Para Crypto Signals, usar exatamente as colunas e ordem solicitadas
    if produto_id == 4970919917:
        colunas_signals = [
            'data_entrada',
            'data_saida',
            'ativo',
            'side',
            'perfil',
            'preco_entrada',
            'preco_saida',
            'pnl',
            'motivo',
        ]
        colunas_existentes = [c for c in colunas_signals if c in df.columns]
        df = df[colunas_existentes]
        return df
    
    # Reordenar colunas para melhor visualização (caso genérico)
    colunas_ordenadas = []
    colunas_atributos = ['perfil', 'motivo', 'pnl', 'rr', 'alvo1', 'alvo2']
    colunas_principais = ['id', 'ativo', 'side', 'data_entrada', 'preco_entrada', 'data_saida', 'preco_saida', 'status']
    colunas_outras = [col for col in df.columns if col not in colunas_atributos + colunas_principais]
    
    for col in colunas_principais + colunas_atributos + colunas_outras:
        if col in df.columns:
            colunas_ordenadas.append(col)
    
    df = df[colunas_ordenadas]
    return df


def display_manutencoes_signals(produto_id=4970919917, formatar=True):
    """
    Exibe manutenções (stops adicionais) das posições abertas do Crypto Signals.

    Cada linha representa um stop de manutenção (a partir do segundo stop) de uma posição aberta.
    """
    from analytics.queries import manutencoes_signals

    df = manutencoes_signals(produto_id)

    if df.empty:
        print("Nenhuma manutenção encontrada para o produto Crypto Signals")
        return df

    # Ordenar pela data de manutenção (mais antiga -> mais nova), se existir
    if 'data_manutencao' in df.columns:
        try:
            df['_data_manutencao_sort'] = pd.to_datetime(df['data_manutencao'])
        except Exception:
            df['_data_manutencao_sort'] = df['data_manutencao']
        df = df.sort_values('_data_manutencao_sort').drop(columns=['_data_manutencao_sort'])

    if formatar:
        # Data de manutenção em formato dia/mês/ano
        if 'data_manutencao' in df.columns:
            df['data_manutencao'] = pd.to_datetime(df['data_manutencao']).dt.strftime('%d/%m/%Y')

        # Função auxiliar para formatar valor + % de distância até o alvo/stop
        def _formatar_valor_com_percent(row, coluna_alvo):
            if 'preco_atual' not in row or pd.isna(row['preco_atual']):
                return ""
            valor = row[coluna_alvo]
            preco_atual = row['preco_atual']
            if pd.isna(valor) or pd.isna(preco_atual) or preco_atual == 0:
                return ""
            try:
                pct = ((valor / preco_atual) - 1) * 100
            except ZeroDivisionError:
                return f"${valor:,.2f}"
            sinal_pct = f"{pct:.2f}%"
            return f"${valor:,.2f} ({sinal_pct})"

        # Formatar preços base
        if 'preco_entrada' in df.columns:
            df['preco_entrada'] = df['preco_entrada'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")

        # Formatar atributos com % relativa ao preço atual
        if 'alvo1' in df.columns and 'preco_atual' in df.columns:
            df['alvo1'] = df.apply(lambda row: _formatar_valor_com_percent(row, 'alvo1'), axis=1)
        if 'alvo2' in df.columns and 'preco_atual' in df.columns:
            df['alvo2'] = df.apply(lambda row: _formatar_valor_com_percent(row, 'alvo2'), axis=1)
        if 'stop_valor' in df.columns and 'preco_atual' in df.columns:
            df['stop_valor'] = df.apply(lambda row: _formatar_valor_com_percent(row, 'stop_valor'), axis=1)

        # Depois de usar os valores numéricos para as porcentagens, formatar preco_atual como string
        if 'preco_atual' in df.columns:
            df['preco_atual'] = df['preco_atual'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")

        # PnL em porcentagem
        if 'pnl' in df.columns:
            df['pnl'] = df['pnl'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "")

        # RR em número simples
        if 'rr' in df.columns:
            df['rr'] = df['rr'].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "")

    # Ordenar colunas na ordem desejada (sem exibir RR na tabela)
    colunas_desejadas = [
        'data_manutencao',
        'ativo',
        'side',
        'perfil',
        'preco_entrada',
        'preco_atual',
        'pnl',
        'alvo1',
        'alvo2',
        'stop_valor',
    ]
    colunas_existentes = [c for c in colunas_desejadas if c in df.columns]
    df = df[colunas_existentes]

    return df


def display_historico_posicoes(produto_id=None, formatar=True):
    """
    Exibe histórico consolidado de posições (abertas + fechadas)
    ordenado pela data de entrada.
    """
    from analytics.queries import historico_posicoes

    df = historico_posicoes(produto_id)

    if df.empty:
        print("Nenhuma posição encontrada para o histórico")
        return df

    if formatar:
        # Formatar valores monetários básicos
        if 'preco_entrada' in df.columns:
            df['preco_entrada'] = df['preco_entrada'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
            )
        if 'preco_saida' in df.columns:
            df['preco_saida'] = df['preco_saida'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
            )
        if 'preco_atual' in df.columns:
            df['preco_atual'] = df['preco_atual'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
            )

        # Atributos do produto (Signals)
        for col in ['alvo1', 'alvo2', 'stop_atual']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
                )

        if 'pnl' in df.columns:
            df['pnl'] = df['pnl'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "—"
            )
        if 'rr' in df.columns:
            df['rr'] = df['rr'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else "—"
            )
        if 'perfil' in df.columns:
            df['perfil'] = df['perfil'].apply(lambda x: x if pd.notna(x) else "—")
        if 'motivo' in df.columns:
            df['motivo'] = df['motivo'].apply(lambda x: x if pd.notna(x) else "—")

    # Para Crypto Signals, usar exatamente as colunas e ordem solicitadas
    if produto_id == 4970919917:
        colunas_signals = [
            'status',
            'data_entrada',
            'data_saida',
            'ativo',
            'side',
            'perfil',
            'preco_entrada',
            'preco_saida',
            'preco_atual',
            'pnl',
            'alvo1',
            'alvo2',
            'stop_atual',
            'rr',
            'motivo',
        ]
        colunas_existentes = [c for c in colunas_signals if c in df.columns]
        df = df[colunas_existentes]
        # Substituir quaisquer NaN remanescentes por "—"
        df = df.fillna("—")
        return df

    # Caso genérico: manter ordenação padrão (priorizar principais + atributos)
    colunas_principais = [
        'id', 'ativo', 'side', 'data_entrada', 'preco_entrada',
        'data_saida', 'preco_saida', 'status', 'preco_atual'
    ]
    colunas_atributos = ['perfil', 'motivo', 'pnl', 'rr', 'alvo1', 'alvo2', 'stop_atual']
    colunas_outras = [c for c in df.columns if c not in colunas_principais + colunas_atributos]

    colunas_ordenadas = []
    for c in colunas_principais + colunas_atributos + colunas_outras:
        if c in df.columns:
            colunas_ordenadas.append(c)

    df = df[colunas_ordenadas]
    df = df.fillna("—")
    return df

def get_valores_posicao(posicao_id):
    """
    Obtém valores diários de uma posição (pronto para gráficos)
    
    Args:
        posicao_id: ID da posição
    
    Returns:
        pd.DataFrame: DataFrame com colunas 'data' e 'preco'
    """
    df = valores_da_posicao(posicao_id)
    
    if not df.empty and 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'])
        df = df.sort_values('data')
    
    return df

