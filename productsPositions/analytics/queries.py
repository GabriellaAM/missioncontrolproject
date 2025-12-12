import pandas as pd
from pathlib import Path
from storage.sqlite_repo import SQLiteRepo
from services.valor_diario_service import ValorDiarioService

# Queries usando SQLite

def posicoes_abertas(produto_id=None):
    """Retorna posições abertas com preço atual"""
    repo = SQLiteRepo()
    df = repo.carregar_posicoes_abertas(produto_id)
    
    # Adicionar preço atual para posições abertas
    if not df.empty:
        precos_atuais = []
        for _, row in df.iterrows():
            coingecko_id = row.get('coingecko_id')
            if pd.notna(coingecko_id) and coingecko_id:
                preco_atual = ValorDiarioService.obter_preco_atual(coingecko_id)
                precos_atuais.append(preco_atual)
            else:
                precos_atuais.append(None)
        df['preco_atual'] = precos_atuais
    
    return df

def posicoes_fechadas(produto_id=None):
    """Retorna posições fechadas (preço atual = preço_saida para histórico)"""
    repo = SQLiteRepo()
    df = repo.carregar_posicoes_fechadas(produto_id)
    
    # Para posições fechadas, preço_atual = preço_saida (já está no histórico)
    if not df.empty:
        df['preco_atual'] = df['preco_saida']
    
    return df

def valores_do_ativo(ativo, data_inicio=None, data_fim=None):
    """Retorna valores diários de um ativo"""
    repo = SQLiteRepo()
    valores = repo.obter_valores_diarios_ativo(ativo, data_inicio, data_fim)
    
    if not valores:
        return pd.DataFrame(columns=['data', 'preco'])
    
    return pd.DataFrame(valores, columns=['data', 'preco'])

def valores_da_posicao(posicao_id):
    """Retorna valores diários de uma posição (via JOIN com ativo)"""
    repo = SQLiteRepo()
    
    # Carregar posição para obter o ativo
    posicao = repo.carregar_posicao(posicao_id)
    
    if not posicao:
        return pd.DataFrame()
    
    ativo = posicao['ativo']
    
    # Retornar valores do ativo
    return valores_do_ativo(ativo)

def alocacoes_do_produto(produto_id):
    """Retorna alocações ativas de um produto"""
    repo = SQLiteRepo()
    return repo.carregar_alocacoes_ativas(produto_id)

def alocacoes_da_posicao(posicao_id):
    """Retorna alocações de uma posição específica"""
    repo = SQLiteRepo()
    import sqlite3
    conn = sqlite3.connect(repo.db_path)
    try:
        df = pd.read_sql_query("""
            SELECT * FROM alocacoes 
            WHERE posicao_id = ? AND status = 'active'
        """, conn, params=(posicao_id,))
        return df
    finally:
        conn.close()

def resumo_alocacoes(produto_id):
    """Resumo de alocações com informações de posições"""
    repo = SQLiteRepo()
    
    # Carregar alocações com JOIN direto no SQL
    with repo._get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT a.*, p.ativo, p.side
            FROM alocacoes a
            INNER JOIN posicoes p ON a.posicao_id = p.id
            WHERE a.produto_id = ? AND a.status = 'active'
            ORDER BY a.percentual DESC
        """, conn, params=(produto_id,))
    
    return df

def carteira_do_produto(produto_id):
    """Retorna carteira de um produto"""
    repo = SQLiteRepo()
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
    repo = SQLiteRepo()
    
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
