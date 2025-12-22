"""
Utilitários para detectar atributos necessários por produto
"""
from typing import Dict, List, Set
from storage.sqlite_repo import SQLiteRepo


def obter_atributos_necessarios_produto(produto_id: int) -> Dict[str, bool]:
    """
    Detecta dinamicamente quais atributos são necessários para um produto específico.
    
    Args:
        produto_id: ID do produto
        
    Returns:
        Dict com flags indicando quais atributos são necessários:
        {
            'quantidade': bool,
            'preco_entrada_total': bool,
            'perfil': bool,
            'alvo1': bool,
            'alvo2': bool,
            'motivo': bool,
        }
    """
    repo = SQLiteRepo()
    produto = repo.carregar_produto(produto_id)
    
    if not produto:
        return {}
    
    tipo_produto = str(produto.get('tipo', '')).lower()
    nome_produto = str(produto.get('nome', '')).lower()
    
    atributos = {
        'quantidade': False,
        'preco_entrada_total': False,
        'perfil': False,
        'alvo1': False,
        'alvo2': False,
        'motivo': False,
    }
    
    # Caso especial: Crypto Signals (ID 4970919917)
    if produto_id == 4970919917:
        atributos['perfil'] = True
        atributos['alvo1'] = True
        atributos['alvo2'] = True
        atributos['motivo'] = True  # Para posições fechadas
        return atributos
    
    # Para produtos Spot e Perpétuos, verificar se usam quantidade
    if 'spot' in tipo_produto or 'perpétuo' in tipo_produto or 'perpetuo' in tipo_produto:
        atributos['quantidade'] = True
        atributos['preco_entrada_total'] = True
    
    # Verificar se há posições existentes com outros atributos
    # (para detectar produtos que podem ter atributos opcionais)
    try:
        import sqlite3
        conn = sqlite3.connect(repo.db_path)
        try:
            # Verificar quais atributos são realmente usados nas posições existentes
            query = """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN quantidade IS NOT NULL THEN 1 ELSE 0 END) as com_quantidade,
                    SUM(CASE WHEN preco_entrada_total IS NOT NULL THEN 1 ELSE 0 END) as com_preco_total,
                    SUM(CASE WHEN perfil IS NOT NULL THEN 1 ELSE 0 END) as com_perfil,
                    SUM(CASE WHEN alvo1 IS NOT NULL THEN 1 ELSE 0 END) as com_alvo1,
                    SUM(CASE WHEN alvo2 IS NOT NULL THEN 1 ELSE 0 END) as com_alvo2
                FROM posicao_atributos_produto
                WHERE produto_id = ?
            """
            cursor = conn.execute(query, (produto_id,))
            row = cursor.fetchone()
            
            if row and row[0] > 0:  # Se há posições existentes
                total = row[0]
                # Se mais de 50% das posições têm um atributo, consideramos necessário
                if row[1] and row[1] / total > 0.5:
                    atributos['quantidade'] = True
                if row[2] and row[2] / total > 0.5:
                    atributos['preco_entrada_total'] = True
                if row[3] and row[3] / total > 0.5:
                    atributos['perfil'] = True
                if row[4] and row[4] / total > 0.5:
                    atributos['alvo1'] = True
                if row[5] and row[5] / total > 0.5:
                    atributos['alvo2'] = True
        finally:
            conn.close()
    except Exception:
        # Se houver erro, usar apenas a lógica baseada em tipo
        pass
    
    return atributos


def obter_campos_editaveis_produto(produto_id: int) -> List[Dict[str, any]]:
    """
    Retorna lista de campos editáveis específicos para um produto.
    
    Args:
        produto_id: ID do produto
        
    Returns:
        Lista de dicionários com informações dos campos:
        [
            {
                'nome': 'quantidade',
                'label': 'Quantidade',
                'tipo': float,
                'obrigatorio': True,
                'descricao': 'Quantidade do ativo'
            },
            ...
        ]
    """
    atributos = obter_atributos_necessarios_produto(produto_id)
    repo = SQLiteRepo()
    produto = repo.carregar_produto(produto_id)
    
    if not produto:
        return []
    
    tipo_produto = str(produto.get('tipo', '')).lower()
    
    campos = []
    
    # Campos básicos sempre disponíveis
    campos.append({
        'nome': 'ativo',
        'label': 'Ativo',
        'tipo': str,
        'obrigatorio': True,
        'descricao': 'Nome do ativo (ex: BTC, ETH)'
    })
    
    # Side apenas para não-Spot
    if 'spot' not in tipo_produto:
        campos.append({
            'nome': 'side',
            'label': 'Side',
            'tipo': str,
            'obrigatorio': True,
            'descricao': 'Direção da posição (long/short)',
            'opcoes': ['long', 'short']
        })
    
    campos.append({
        'nome': 'coingecko_id',
        'label': 'CoinGecko ID',
        'tipo': str,
        'obrigatorio': False,
        'descricao': 'ID do CoinGecko (ex: bitcoin, ethereum)'
    })
    
    campos.append({
        'nome': 'data_entrada',
        'label': 'Data de Entrada',
        'tipo': 'date',
        'obrigatorio': True,
        'descricao': 'Data de entrada (YYYY-MM-DD)'
    })
    
    campos.append({
        'nome': 'preco_entrada',
        'label': 'Preço de Entrada',
        'tipo': float,
        'obrigatorio': True,
        'descricao': 'Preço de entrada em USD'
    })
    
    # Campos específicos de atributos
    if atributos.get('quantidade'):
        campos.append({
            'nome': 'quantidade',
            'label': 'Quantidade',
            'tipo': float,
            'obrigatorio': True,
            'descricao': 'Quantidade do ativo',
            'atributo': True
        })
    
    if atributos.get('perfil'):
        campos.append({
            'nome': 'perfil',
            'label': 'Perfil',
            'tipo': str,
            'obrigatorio': False,
            'descricao': 'Perfil de risco',
            'atributo': True
        })
    
    if atributos.get('alvo1'):
        campos.append({
            'nome': 'alvo1',
            'label': 'Alvo 1',
            'tipo': float,
            'obrigatorio': False,
            'descricao': 'Primeiro alvo de preço',
            'atributo': True
        })
    
    if atributos.get('alvo2'):
        campos.append({
            'nome': 'alvo2',
            'label': 'Alvo 2',
            'tipo': float,
            'obrigatorio': False,
            'descricao': 'Segundo alvo de preço',
            'atributo': True
        })
    
    # Campos de saída (sempre disponíveis para edição)
    campos.append({
        'nome': 'data_saida',
        'label': 'Data de Saída',
        'tipo': 'date',
        'obrigatorio': False,
        'descricao': 'Data de saída (YYYY-MM-DD)'
    })
    
    campos.append({
        'nome': 'preco_saida',
        'label': 'Preço de Saída',
        'tipo': float,
        'obrigatorio': False,
        'descricao': 'Preço de saída em USD'
    })
    
    campos.append({
        'nome': 'status',
        'label': 'Status',
        'tipo': str,
        'obrigatorio': True,
        'descricao': 'Status da posição (open/closed)',
        'opcoes': ['open', 'closed']
    })
    
    if atributos.get('motivo'):
        campos.append({
            'nome': 'motivo',
            'label': 'Motivo',
            'tipo': str,
            'obrigatorio': False,
            'descricao': 'Motivo do encerramento',
            'atributo': True
        })
    
    return campos

