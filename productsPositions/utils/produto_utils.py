"""
Utilitários para detectar atributos necessários por produto
"""
from typing import Dict, List
from storage.sqlite_repo import SQLiteRepo


def obter_atributos_necessarios_produto(produto_id: int) -> Dict[str, bool]:
    """
    Obtém quais atributos são necessários para um produto a partir da tabela de config.

    Args:
        produto_id: ID do produto

    Returns:
        Dict com flags indicando quais atributos são necessários:
        {
            'quantidade': bool,
            'perfil': bool,
            ...
        }
    """
    repo = SQLiteRepo()

    # Carregar configuração de atributos do produto
    configs = repo.carregar_atributos_config(produto_id)

    # Converter para dicionário de flags
    atributos = {}
    for config in configs:
        atributos[config['atributo_nome']] = True

    return atributos


def obter_campos_editaveis_produto(produto_id: int) -> List[Dict[str, any]]:
    """
    Retorna lista de campos editáveis específicos para um produto.
    Lê configuração de atributos da tabela produto_atributos_config.

    Args:
        produto_id: ID do produto

    Returns:
        Lista de dicionários com informações dos campos
    """
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
            'descricao': 'Direcao da posicao (long/short)',
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
        'label': 'Preco de Entrada',
        'tipo': float,
        'obrigatorio': True,
        'descricao': 'Preco de entrada em USD'
    })

    # Campos específicos de atributos (dinâmicos da config)
    configs = repo.carregar_atributos_config(produto_id)

    # Mapear tipos de string para tipos Python
    tipo_map = {
        'text': str,
        'float': float,
        'int': int,
        'date': 'date'
    }

    for config in configs:
        nome = config['atributo_nome']
        # Pular campos de saída (serão adicionados depois)
        if nome in ['motivo']:
            continue

        campos.append({
            'nome': nome,
            'label': config['atributo_label'] or nome.replace('_', ' ').title(),
            'tipo': tipo_map.get(config['atributo_tipo'], str),
            'obrigatorio': bool(config['obrigatorio']),
            'descricao': f"Atributo: {config['atributo_label'] or nome}",
            'atributo': True
        })

    # Campos de saída (sempre disponíveis para edição)
    campos.append({
        'nome': 'data_saida',
        'label': 'Data de Saida',
        'tipo': 'date',
        'obrigatorio': False,
        'descricao': 'Data de saida (YYYY-MM-DD)'
    })

    campos.append({
        'nome': 'preco_saida',
        'label': 'Preco de Saida',
        'tipo': float,
        'obrigatorio': False,
        'descricao': 'Preco de saida em USD'
    })

    campos.append({
        'nome': 'status',
        'label': 'Status',
        'tipo': str,
        'obrigatorio': True,
        'descricao': 'Status da posicao (open/closed)',
        'opcoes': ['open', 'closed']
    })

    # Adicionar motivo se configurado (campo de saída)
    for config in configs:
        if config['atributo_nome'] == 'motivo':
            campos.append({
                'nome': 'motivo',
                'label': config['atributo_label'] or 'Motivo',
                'tipo': str,
                'obrigatorio': bool(config['obrigatorio']),
                'descricao': 'Motivo do encerramento',
                'atributo': True
            })
            break

    return campos

