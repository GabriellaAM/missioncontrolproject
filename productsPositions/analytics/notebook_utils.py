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
    Exibe posições abertas em formato de tabela
    
    Args:
        produto_id: ID do produto (opcional, None para todos)
        formatar: Se True, formata valores monetários
    
    Returns:
        pd.DataFrame: DataFrame com posições abertas
    """
    df = posicoes_abertas(produto_id)
    
    if df.empty:
        print("Nenhuma posição aberta encontrada")
        return df
    
    if formatar and 'preco_entrada' in df.columns:
        df['preco_entrada'] = df['preco_entrada'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    
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

