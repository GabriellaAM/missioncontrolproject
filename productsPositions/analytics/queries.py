import pandas as pd
from pathlib import Path
from storage.parquet_repo import ParquetRepo

# Queries usando Parquet (substituiu SQLite)

def posicoes_abertas(produto_id=None):
    """Retorna posições abertas"""
    repo = ParquetRepo()
    return repo.carregar_posicoes_abertas(produto_id)

def posicoes_fechadas(produto_id=None):
    """Retorna posições fechadas"""
    repo = ParquetRepo()
    return repo.carregar_posicoes_fechadas(produto_id)

def valores_do_ativo(ativo, data_inicio=None, data_fim=None):
    """Retorna valores diários de um ativo"""
    repo = ParquetRepo()
    valores = repo.obter_valores_diarios_ativo(ativo, data_inicio, data_fim)
    
    if not valores:
        return pd.DataFrame(columns=['data', 'preco'])
    
    return pd.DataFrame(valores, columns=['data', 'preco'])

def valores_da_posicao(posicao_id):
    """Retorna valores diários de uma posição (via JOIN com ativo)"""
    repo = ParquetRepo()
    
    # Carregar posição para obter o ativo
    df_posicoes = repo._carregar_df(repo.posicoes_path)
    posicao = df_posicoes[df_posicoes['id'] == posicao_id]
    
    if posicao.empty:
        return pd.DataFrame()
    
    ativo = posicao.iloc[0]['ativo']
    
    # Retornar valores do ativo
    return valores_do_ativo(ativo)

def alocacoes_do_produto(produto_id):
    """Retorna alocações ativas de um produto"""
    repo = ParquetRepo()
    return repo.carregar_alocacoes_ativas(produto_id)

def alocacoes_da_posicao(posicao_id):
    """Retorna alocações de uma posição específica"""
    repo = ParquetRepo()
    df_alocacoes = repo._carregar_df(repo.alocacoes_path)
    resultado = df_alocacoes[
        (df_alocacoes['posicao_id'] == posicao_id) & 
        (df_alocacoes['status'] == 'active')
    ]
    return resultado

def resumo_alocacoes(produto_id):
    """Resumo de alocações com informações de posições"""
    repo = ParquetRepo()
    
    # Carregar alocações
    df_alocacoes = repo.carregar_alocacoes_ativas(produto_id)
    
    if df_alocacoes.empty:
        return pd.DataFrame()
    
    # Carregar posições
    df_posicoes = repo._carregar_df(repo.posicoes_path)
    
    # Fazer JOIN
    resultado = df_alocacoes.merge(
        df_posicoes[['id', 'ativo', 'side']],
        left_on='posicao_id',
        right_on='id',
        suffixes=('', '_posicao')
    )
    
    resultado = resultado.sort_values('percentual', ascending=False)
    return resultado

def carteira_do_produto(produto_id):
    """Retorna carteira de um produto"""
    repo = ParquetRepo()
    carteira = repo.carregar_carteira(produto_id)
    
    if carteira:
        return pd.DataFrame([{
            'produto_id': carteira.produto_id,
            'valor_disponivel': carteira.valor_disponivel,
            'valor_investido': carteira.valor_investido,
            'pnl_nao_realizado': carteira.pnl_nao_realizado,
            'valor_total': carteira.valor_total,
            'data_atualizacao': carteira.data_atualizacao
        }])
    return pd.DataFrame()

def resumo_completo_produto(produto_id):
    """Resumo completo: produto + carteira"""
    repo = ParquetRepo()
    
    # Carregar produto
    produto = repo.carregar_produto(produto_id)
    if not produto:
        return pd.DataFrame()
    
    # Carregar carteira
    carteira = repo.carregar_carteira(produto_id)
    
    resultado = {
        'produto_id': produto_id,
        'produto_nome': produto['nome'],
        'data_inicio': produto['data_inicio'],
        'tipo': produto['tipo'],
        'capital_inicial': produto.get('capital_inicial', 0.0),
        'valor_disponivel': carteira.valor_disponivel if carteira else None,
        'valor_investido': carteira.valor_investido if carteira else None,
        'pnl_nao_realizado': carteira.pnl_nao_realizado if carteira else None,
        'valor_total': carteira.valor_total if carteira else None,
        'data_atualizacao': carteira.data_atualizacao if carteira else None
    }
    
    return pd.DataFrame([resultado])
