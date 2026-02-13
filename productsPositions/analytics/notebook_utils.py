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
from storage.sqlite_repo import get_repo

def display_produtos():
    """
    Exibe todos os produtos disponíveis em formato de tabela
    
    Returns:
        pd.DataFrame: DataFrame com todos os produtos
    """
    repo = get_repo()
    produtos = repo.listar_produtos()
    
    if not produtos:
        print("Nenhum produto encontrado")
        return pd.DataFrame()
    
    df = pd.DataFrame(produtos)
    return df

def display_posicoes_abertas(produto_id=None, formatar=True, filtrar_colunas=True):
    """
    Exibe posições abertas em formato de tabela com atributos do produto

    Args:
        produto_id: ID do produto (opcional, None para todos)
        formatar: Se True, formata valores monetários e atributos
        filtrar_colunas: Se True, aplica filtros de colunas hard-coded por produto.
                        Se False, mantém todas as colunas (para uso com visualizações do banco).

    Returns:
        pd.DataFrame: DataFrame com posições abertas e atributos
    """
    df = posicoes_abertas(produto_id)
    
    if df.empty:
        print("Nenhuma posição aberta encontrada")
        return df
    
    if formatar:
        # Verificar se o produto tem "meme" no nome para usar 5 casas decimais em preco_entrada e preco_saida
        is_meme = False
        if produto_id is not None:
            try:
                repo = get_repo()
                prod_info = repo.carregar_produto(produto_id)
                if prod_info and 'nome' in prod_info:
                    nome_produto = str(prod_info['nome']).lower()
                    is_meme = 'meme' in nome_produto
            except Exception:
                pass
        
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

        # Formatar preco_entrada com 5 casas se for meme, senão 2
        if 'preco_entrada' in df.columns:
            decimais_entrada = 5 if is_meme else 2
            df['preco_entrada'] = df['preco_entrada'].apply(
                lambda x: f"${x:,.{decimais_entrada}f}" if pd.notna(x) else ""
            )

        # Para produtos Spot, formatar quantidade e preco_entrada_total, se existirem
        # (essa flag é preenchida mais abaixo quando identificamos o tipo do produto)

        # Formatar atributos do produto com porcentagem relativa ao preço atual
        if 'alvo1' in df.columns and 'preco_atual' in df.columns:
            df['alvo1'] = df.apply(lambda row: _formatar_valor_com_percent(row, 'alvo1'), axis=1)
        if 'alvo2' in df.columns and 'preco_atual' in df.columns:
            df['alvo2'] = df.apply(lambda row: _formatar_valor_com_percent(row, 'alvo2'), axis=1)
        # Formatar stop_atual (para todos os produtos)
        if 'stop_atual' in df.columns:
            if 'preco_atual' in df.columns:
                # Para Crypto Signals, usar formatação com porcentagem
                if produto_id == 4970919917:
                    df['stop_atual'] = df.apply(lambda row: _formatar_valor_com_percent(row, 'stop_atual'), axis=1)
                else:
                    # Para outros produtos, apenas valor monetário ou "—"
                    def _formatar_stop_atual(x):
                        if pd.isna(x) or x is None:
                            return "—"
                        if isinstance(x, (int, float)):
                            if x == -1:
                                return "STOP ATINGIDO"
                            return f"${x:,.2f}"
                        return str(x)
                    df['stop_atual'] = df['stop_atual'].apply(_formatar_stop_atual)
            else:
                # Se não houver preco_atual, formatar apenas como monetário
                def _formatar_stop_atual(x):
                    if pd.isna(x) or x is None:
                        return "—"
                    if isinstance(x, (int, float)):
                        if x == -1:
                            return "STOP ATINGIDO"
                        return f"${x:,.2f}"
                    return str(x)
                df['stop_atual'] = df['stop_atual'].apply(_formatar_stop_atual)

        # Formatar alocação (percentual) se existir
        if 'alocacao' in df.columns:
            df['alocacao'] = df['alocacao'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "—"
            )

        # Agora, depois de usar os valores numéricos para as porcentagens, formatar preco_atual como string
        # IMPORTANTE: Salvar valor numérico do preco_atual antes de formatar (para usar em preco_saida depois)
        preco_atual_numerico = None
        if 'preco_atual' in df.columns:
            preco_atual_numerico = df['preco_atual'].copy()  # Salvar cópia numérica
            df['preco_atual'] = df['preco_atual'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
            )

        # RR é calculado dinamicamente, formatar se existir
        if 'rr' in df.columns:
            df['rr'] = df['rr'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) and x is not None else ""
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
    # (apenas se filtrar_colunas=True)
    if filtrar_colunas:
        colunas_para_remover = ['produto_id', 'coingecko_id', 'data_saida', 'preco_saida', 'status', 'motivo']
        df = df.drop(columns=[c for c in colunas_para_remover if c in df.columns])

    # Detectar tipo de produto (Spot ou Perpétuos, exceto 4970919917)
    tipo_spot = False
    tipo_perpetuos = False
    if produto_id is not None and produto_id != 4970919917:
        try:
            repo = get_repo()
            prod_info = repo.carregar_produto(produto_id)
            if prod_info and 'tipo' in prod_info and isinstance(prod_info['tipo'], str):
                tipo_str = prod_info['tipo'].lower()
                if 'spot' in tipo_str:
                    tipo_spot = True
                elif 'perpétuo' in tipo_str or 'perpetuo' in tipo_str:
                    tipo_perpetuos = True
        except Exception:
            pass

    # Formatar campos específicos de Spot se identificado
    if tipo_spot:
        if 'quantidade' in df.columns:
            df['quantidade'] = df['quantidade'].apply(
                lambda x: f"{x:,.2f}" if pd.notna(x) else ""
            )
        if 'preco_entrada_total' in df.columns:
            df['preco_entrada_total'] = df['preco_entrada_total'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else ""
            )
        if 'preco_atual_total' in df.columns:
            df['preco_atual_total'] = df['preco_atual_total'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else ""
            )

    # Formatar campos específicos de Perpétuos se identificado
    if tipo_perpetuos:
        if 'quantidade' in df.columns:
            df['quantidade'] = df['quantidade'].apply(
                lambda x: f"{x:,.2f}" if pd.notna(x) else ""
            )
        if 'preco_entrada_total' in df.columns:
            df['preco_entrada_total'] = df['preco_entrada_total'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else ""
            )
        # Para Perpétuos, preco_saida = preco_atual (para abertas)
        # Usar valor numérico salvo anteriormente, não a versão formatada
        if preco_atual_numerico is not None:
            df['preco_saida'] = preco_atual_numerico
        elif 'preco_atual' in df.columns:
            # Fallback: tentar extrair valor numérico da string formatada
            def _extrair_numero(x):
                if pd.isna(x) or x is None:
                    return None
                if isinstance(x, (int, float)):
                    return x
                if isinstance(x, str):
                    try:
                        cleaned = x.replace('$', '').replace(',', '').strip()
                        return float(cleaned)
                    except:
                        return None
                return None
            df['preco_saida'] = df['preco_atual'].apply(_extrair_numero)
        if 'preco_saida' in df.columns:
            decimais_saida = 5 if is_meme else 2
            def _formatar_preco_saida_abertas(x):
                if pd.isna(x) or x is None:
                    return "—"
                # Se já for string formatada, tentar extrair o número
                if isinstance(x, str):
                    try:
                        cleaned = x.replace('$', '').replace(',', '').strip()
                        x = float(cleaned)
                    except:
                        return str(x)  # Retornar como está se não conseguir converter
                if isinstance(x, (int, float)):
                    # Para valores muito pequenos, usar mais casas decimais para evitar mostrar apenas zeros
                    if abs(x) > 0 and abs(x) < 0.01:
                        # Usar até 8 casas decimais para valores muito pequenos (sem vírgula para evitar problemas)
                        return f"${x:.8f}".rstrip('0').rstrip('.')
                    else:
                        return f"${x:,.{decimais_saida}f}"
                return "—"
            df['preco_saida'] = df['preco_saida'].apply(_formatar_preco_saida_abertas)
        if 'preco_saida_total' in df.columns:
            df['preco_saida_total'] = df['preco_saida_total'].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else ""
            )

    # Formatar PnL em porcentagem (após todas as formatações específicas)
    # Se for USDT, exibir "—" ao invés de calcular PnL
    if 'pnl' in df.columns and 'ativo' in df.columns:
        def _formatar_pnl_com_usdt(row):
            ativo = str(row.get('ativo', '')).strip().upper()
            pnl = row.get('pnl')
            # Se for USDT, sempre exibir "—" (verificar antes de qualquer outra coisa)
            if ativo == 'USDT':
                return "—"
            # Caso contrário, formatar normalmente
            if pd.notna(pnl) and pnl is not None and isinstance(pnl, (int, float)):
                return f"{pnl:.2f}%"
            return "" if pd.isna(pnl) or pnl is None else str(pnl)
        
        df['pnl'] = df.apply(_formatar_pnl_com_usdt, axis=1)
    elif 'pnl' in df.columns:
        df['pnl'] = df['pnl'].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None and isinstance(x, (int, float)) else ("" if pd.isna(x) or x is None else str(x))
        )

    # Para produtos que NÃO são Crypto Signals, não exibir perfil/alvo1/alvo2 em posições abertas
    # (apenas se filtrar_colunas=True)
    if filtrar_colunas and produto_id is not None and produto_id != 4970919917:
        colunas_para_ocultar = ['perfil', 'alvo1', 'alvo2']
        df = df.drop(columns=[c for c in colunas_para_ocultar if c in df.columns])

    # Se for o produto Alphacoins (ID 3476245316), EXC (ID 2150859854), HB (ID 2000449260) ou LC (ID 2394004756), mostrar apenas colunas específicas
    # (apenas se filtrar_colunas=True)
    if filtrar_colunas and (produto_id == 3476245316 or produto_id == 2150859854 or produto_id == 2000449260 or produto_id == 2394004756):
        colunas_alphacoins_exc = [
            'data_entrada',
            'ativo',
            'preco_entrada',
            'preco_atual',
            'pnl',
            'alocacao',
            'stop_atual',
        ]
        colunas_existentes = [c for c in colunas_alphacoins_exc if c in df.columns]
        df = df[colunas_existentes]
        return df

    # Para o produto Crypto Signals, usar exatamente as colunas e ordem solicitadas (ordem original)
    # (apenas se filtrar_colunas=True)
    if filtrar_colunas and produto_id == 4970919917:
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
            'stop_atual',  # Ordem original: antes de rr
            'rr',
        ]
        colunas_existentes = [c for c in colunas_signals if c in df.columns]
        df = df[colunas_existentes]
        return df

    # Para outros produtos, manter ordenação mais genérica
    # (apenas se filtrar_colunas=True)
    if not filtrar_colunas:
        # Se não estamos filtrando colunas, apenas retornar o DataFrame formatado
        return df

    colunas_ordenadas = []
    colunas_principais = ['id', 'ativo', 'side', 'data_entrada', 'preco_entrada', 'preco_atual']
    colunas_atributos = ['perfil', 'pnl', 'rr', 'alvo1', 'alvo2', 'stop_atual']

    # Se for Spot, incluir quantidade e preco_entrada_total numa ordem mais natural
    if tipo_spot:
        colunas_principais = [
            'data_entrada',
            'ativo',
            'quantidade',
            'preco_entrada',
            'preco_entrada_total',
            'preco_atual',
            'preco_atual_total',
        ]
    # Se for Perpétuos, usar ordem específica
    elif tipo_perpetuos:
        # Salvar stop_atual antes de remover colunas
        stop_atual_col = df['stop_atual'].copy() if 'stop_atual' in df.columns else None

        colunas_perpetuos = [
            'data_entrada',
            'ativo',
            'side',
            'quantidade',
            'preco_entrada',
            'preco_entrada_total',
            'preco_saida',
            'preco_saida_total',
            'pnl',
        ]
        colunas_existentes = [c for c in colunas_perpetuos if c in df.columns]
        df = df[colunas_existentes]

        # Remover colunas indesejadas
        colunas_para_remover = ['id', 'produto_id', 'coingecko_id', 'preco_atual', 'status']
        df = df.drop(columns=[c for c in colunas_para_remover if c in df.columns])

        # Adicionar stop_atual como última coluna
        if stop_atual_col is not None:
            df['stop_atual'] = stop_atual_col
            # Formatar stop_atual se ainda não foi formatado
            if df['stop_atual'].dtype != 'object':  # Se ainda não foi formatado
                def _formatar_stop_atual(x):
                    if pd.isna(x) or x is None:
                        return "—"
                    if isinstance(x, (int, float)):
                        if x == -1:
                            return "STOP ATINGIDO"
                        return f"${x:,.2f}"
                    return str(x)
                df['stop_atual'] = df['stop_atual'].apply(_formatar_stop_atual)

        return df

    colunas_outras = [col for col in df.columns if col not in colunas_atributos + colunas_principais + ['stop_atual']]

    # Construir ordem: principais + outros + atributos (exceto stop_atual) + stop_atual por último
    for col in colunas_principais + colunas_outras:
        if col in df.columns and col != 'stop_atual':
            colunas_ordenadas.append(col)

    # Adicionar atributos exceto stop_atual
    for col in colunas_atributos:
        if col in df.columns and col != 'stop_atual':
            colunas_ordenadas.append(col)

    # stop_atual sempre por último
    if 'stop_atual' in df.columns:
        colunas_ordenadas.append('stop_atual')

    df = df[colunas_ordenadas]

    # Se for produto Spot, remover colunas side e id da visualização
    if tipo_spot:
        colunas_para_remover_spot = []
        if 'side' in df.columns:
            colunas_para_remover_spot.append('side')
        if 'id' in df.columns:
            colunas_para_remover_spot.append('id')
        if colunas_para_remover_spot:
            df = df.drop(columns=colunas_para_remover_spot)
        # Reordenar para garantir stop_atual por último
        if 'stop_atual' in df.columns:
            outras_cols = [c for c in df.columns if c != 'stop_atual']
            df = df[outras_cols + ['stop_atual']]

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

def display_posicoes_fechadas(produto_id=None, formatar=True, filtrar_colunas=True):
    """
    Exibe posições fechadas em formato de tabela com atributos do produto

    Args:
        produto_id: ID do produto (opcional, None para todos)
        formatar: Se True, formata valores monetários e atributos
        filtrar_colunas: Se True, aplica filtros de colunas hard-coded por produto.
                        Se False, mantém todas as colunas (para uso com visualizações do banco).

    Returns:
        pd.DataFrame: DataFrame com posições fechadas e atributos
    """
    df = posicoes_fechadas(produto_id)

    if df.empty:
        print("Nenhuma posição fechada encontrada")
        return df

    # Calcular dias em carteira (data_saida - data_entrada)
    if 'data_entrada' in df.columns and 'data_saida' in df.columns:
        def _calcular_dias_carteira(row):
            try:
                data_entrada = pd.to_datetime(row['data_entrada'])
                data_saida = pd.to_datetime(row['data_saida'])
                if pd.notna(data_entrada) and pd.notna(data_saida):
                    return (data_saida - data_entrada).days
            except Exception:
                pass
            return None
        df['dias_carteira'] = df.apply(_calcular_dias_carteira, axis=1)

    # Detectar tipo de produto (Spot ou Perpétuos, exceto 4970919917)
    tipo_spot = False
    tipo_perpetuos = False
    if produto_id is not None and produto_id != 4970919917:
        try:
            repo = get_repo()
            prod_info = repo.carregar_produto(produto_id)
            if prod_info and 'tipo' in prod_info and isinstance(prod_info['tipo'], str):
                tipo_str = prod_info['tipo'].lower()
                if 'spot' in tipo_str:
                    tipo_spot = True
                elif 'perpétuo' in tipo_str or 'perpetuo' in tipo_str:
                    tipo_perpetuos = True
        except Exception:
            pass
    
    if formatar:
        # Verificar se o produto tem "meme" no nome para usar 5 casas decimais em preco_entrada e preco_saida
        is_meme = False
        if produto_id is not None:
            try:
                repo = get_repo()
                prod_info = repo.carregar_produto(produto_id)
                if prod_info and 'nome' in prod_info:
                    nome_produto = str(prod_info['nome']).lower()
                    is_meme = 'meme' in nome_produto
            except Exception:
                pass
        
        # Formatar preco_entrada com 5 casas se for meme, senão 2
        if 'preco_entrada' in df.columns:
            decimais_entrada = 5 if is_meme else 2
            df['preco_entrada'] = df['preco_entrada'].apply(lambda x: f"${x:,.{decimais_entrada}f}" if pd.notna(x) else "")
        
        # Formatar preco_saida com 5 casas se for meme, senão 2. Stablecoins sem preço → "—"
        _STABLECOINS = {'USDT', 'USDC', 'TUSD', 'BUSD', 'DAI', 'UST', 'GUSD', 'USDP', 'FDUSD'}
        if 'preco_saida' in df.columns:
            decimais_saida = 5 if is_meme else 2
            def _formatar_preco_saida_fechadas(row):
                x = row['preco_saida'] if isinstance(row, pd.Series) else row
                ativo = (row['ativo'] if isinstance(row, pd.Series) and 'ativo' in row.index else None) or ''
                ativo_upper = str(ativo).strip().upper()
                if pd.isna(x) or x is None or (isinstance(x, (int, float)) and x == 0):
                    return "—" if ativo_upper in _STABLECOINS else ""
                if isinstance(x, (int, float)):
                    # Para valores muito pequenos, usar mais casas decimais para evitar mostrar apenas zeros
                    if abs(x) > 0 and abs(x) < 0.01:
                        return f"${x:.8f}".rstrip('0').rstrip('.')
                    else:
                        return f"${x:,.{decimais_saida}f}"
                return ""
            if 'ativo' in df.columns:
                df['preco_saida'] = df.apply(_formatar_preco_saida_fechadas, axis=1)
            else:
                df['preco_saida'] = df['preco_saida'].apply(
                    lambda x: "—" if (pd.isna(x) or x is None or x == 0) else (
                        f"${x:.8f}".rstrip('0').rstrip('.') if isinstance(x, (int, float)) and abs(x) > 0 and abs(x) < 0.01
                        else (f"${x:,.{decimais_saida}f}" if isinstance(x, (int, float)) else "")
                    )
                )
        
        # Formatar preco_atual (sempre 2 casas)
        if 'preco_atual' in df.columns:
            df['preco_atual'] = df['preco_atual'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        
        # Formatar campos específicos de Spot
        if tipo_spot:
            if 'quantidade' in df.columns:
                df['quantidade'] = df['quantidade'].apply(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
            if 'preco_entrada_total' in df.columns:
                df['preco_entrada_total'] = df['preco_entrada_total'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            if 'preco_saida_total' in df.columns:
                df['preco_saida_total'] = df['preco_saida_total'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        
        # Formatar campos específicos de Perpétuos
        if tipo_perpetuos:
            if 'quantidade' in df.columns:
                df['quantidade'] = df['quantidade'].apply(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
            if 'preco_entrada_total' in df.columns:
                df['preco_entrada_total'] = df['preco_entrada_total'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            if 'preco_saida_total' in df.columns:
                df['preco_saida_total'] = df['preco_saida_total'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        
        # Formatar atributos do produto (apenas para não-Spot e não-Perpétuos)
        if not tipo_spot and not tipo_perpetuos:
            if 'alvo1' in df.columns:
                df['alvo1'] = df['alvo1'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            if 'alvo2' in df.columns:
                df['alvo2'] = df['alvo2'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            if 'rr' in df.columns:
                df['rr'] = df['rr'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            if 'perfil' in df.columns:
                df['perfil'] = df['perfil'].apply(lambda x: x if pd.notna(x) else "")
            if 'motivo' in df.columns:
                df['motivo'] = df['motivo'].apply(lambda x: x if pd.notna(x) else "")
        
        # Formatar PnL: se for USDT, exibir "—"
        if 'pnl' in df.columns and 'ativo' in df.columns:
            def _formatar_pnl_fechadas(row):
                ativo = str(row.get('ativo', '')).strip().upper()
                pnl = row.get('pnl')
                if ativo == 'USDT':
                    return "—"
                if pd.notna(pnl) and pnl is not None:
                    return f"{pnl:.2f}%"
                return "—"
            df['pnl'] = df.apply(_formatar_pnl_fechadas, axis=1)
        elif 'pnl' in df.columns:
            df['pnl'] = df['pnl'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "—")
        
        # Formatar stop_atual (para todos os produtos)
        if 'stop_atual' in df.columns:
            def _formatar_stop_atual_fechadas(x):
                if pd.isna(x) or x is None:
                    return "—"
                if isinstance(x, (int, float)):
                    if x == -1:
                        return "STOP ATINGIDO"
                    return f"${x:,.2f}"
                return str(x)
            df['stop_atual'] = df['stop_atual'].apply(_formatar_stop_atual_fechadas)

    # Ordenar pela data de entrada (mais antiga -> mais nova), se existir
    if 'data_entrada' in df.columns:
        try:
            df['_data_entrada_sort'] = pd.to_datetime(df['data_entrada'])
        except Exception:
            df['_data_entrada_sort'] = df['data_entrada']
        df = df.sort_values('_data_entrada_sort').drop(columns=['_data_entrada_sort'])

    # Se não estamos filtrando colunas, retornar DataFrame formatado
    if not filtrar_colunas:
        return df

    # Se for o produto Alphacoins (ID 3476245316), EXC (ID 2150859854), HB (ID 2000449260) ou LC (ID 2394004756), mostrar apenas colunas específicas
    if produto_id == 3476245316 or produto_id == 2150859854 or produto_id == 2000449260 or produto_id == 2394004756:
        colunas_alphacoins_exc = [
            'dias_carteira',
            'data_entrada',
            'data_saida',
            'ativo',
            'preco_entrada',
            'preco_saida',
            'pnl',
            'alocacao',
            'stop_atual',
        ]
        colunas_existentes = [c for c in colunas_alphacoins_exc if c in df.columns]
        df = df[colunas_existentes]
        # Formatar alocação se existir
        if 'alocacao' in df.columns:
            df['alocacao'] = df['alocacao'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "—"
            )
        # Substituir quaisquer NaN remanescentes por "—"
        df = df.fillna("—")
        return df

    # Para Crypto Signals, usar exatamente as colunas e ordem solicitadas (ordem original)
    if produto_id == 4970919917:
        colunas_signals = [
            'dias_carteira',
            'data_entrada',
            'data_saida',
            'ativo',
            'side',
            'perfil',
            'preco_entrada',
            'preco_saida',
            'pnl',
            'motivo',
            # stop_atual não aparece em posições fechadas do Crypto Signals (ordem original)
        ]
        colunas_existentes = [c for c in colunas_signals if c in df.columns]
        df = df[colunas_existentes]
        return df

    # Para produtos Spot, usar ordem específica e remover colunas indesejadas
    if tipo_spot:
        # Salvar stop_atual antes de remover colunas
        stop_atual_col = df['stop_atual'].copy() if 'stop_atual' in df.columns else None

        colunas_spot = [
            'dias_carteira',
            'data_entrada',
            'data_saida',
            'ativo',
            'quantidade',
            'preco_entrada',
            'preco_entrada_total',
            'preco_saida',
            'preco_saida_total',
            'pnl',
        ]
        colunas_existentes = [c for c in colunas_spot if c in df.columns]
        df = df[colunas_existentes]

        # Remover colunas indesejadas explicitamente
        colunas_para_remover = ['id', 'produto_id', 'coingecko_id', 'status', 'side', 'perfil', 'motivo', 'rr', 'alvo1', 'alvo2']
        df = df.drop(columns=[c for c in colunas_para_remover if c in df.columns])

        # Adicionar stop_atual como última coluna
        if stop_atual_col is not None:
            df['stop_atual'] = stop_atual_col
            # Formatar se ainda não foi formatado
            if df['stop_atual'].dtype != 'object':
                def _formatar_stop_atual(x):
                    if pd.isna(x) or x is None:
                        return "—"
                    if isinstance(x, (int, float)):
                        if x == -1:
                            return "STOP ATINGIDO"
                        return f"${x:,.2f}"
                    return str(x)
                df['stop_atual'] = df['stop_atual'].apply(_formatar_stop_atual)

        return df

    # Para produtos Perpétuos, usar ordem específica
    if tipo_perpetuos:
        # Salvar stop_atual antes de remover colunas
        stop_atual_col = df['stop_atual'].copy() if 'stop_atual' in df.columns else None

        colunas_perpetuos = [
            'dias_carteira',
            'data_entrada',
            'data_saida',
            'ativo',
            'side',
            'quantidade',
            'preco_entrada',
            'preco_entrada_total',
            'preco_saida',
            'preco_saida_total',
            'pnl',
        ]
        colunas_existentes = [c for c in colunas_perpetuos if c in df.columns]
        df = df[colunas_existentes]

        # Remover colunas indesejadas
        colunas_para_remover = ['id', 'produto_id', 'coingecko_id', 'status', 'perfil', 'motivo', 'rr', 'alvo1', 'alvo2', 'preco_atual']
        df = df.drop(columns=[c for c in colunas_para_remover if c in df.columns])

        # Adicionar stop_atual como última coluna
        if stop_atual_col is not None:
            df['stop_atual'] = stop_atual_col
            # Formatar se ainda não foi formatado
            if df['stop_atual'].dtype != 'object':
                def _formatar_stop_atual(x):
                    if pd.isna(x) or x is None:
                        return "—"
                    if isinstance(x, (int, float)):
                        if x == -1:
                            return "STOP ATINGIDO"
                        return f"${x:,.2f}"
                    return str(x)
                df['stop_atual'] = df['stop_atual'].apply(_formatar_stop_atual)

        return df

    # Reordenar colunas para melhor visualização (caso genérico - não Spot, não Perpétuos, não Signals)
    colunas_ordenadas = []
    colunas_atributos = ['perfil', 'motivo', 'pnl', 'rr', 'alvo1', 'alvo2']
    colunas_principais = ['id', 'ativo', 'side', 'dias_carteira', 'data_entrada', 'preco_entrada', 'data_saida', 'preco_saida', 'status']
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
        # Verificar se o produto tem "meme" no nome para usar 5 casas decimais em preco_entrada e preco_saida
        is_meme = False
        if produto_id is not None:
            try:
                repo = get_repo()
                prod_info = repo.carregar_produto(produto_id)
                if prod_info and 'nome' in prod_info:
                    nome_produto = str(prod_info['nome']).lower()
                    is_meme = 'meme' in nome_produto
            except Exception:
                pass
        
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
                return f"${valor:,.{decimais}f}"
            sinal_pct = f"{pct:.2f}%"
            return f"${valor:,.{decimais}f} ({sinal_pct})"

        # Formatar preco_entrada com 5 casas se for meme, senão 2
        if 'preco_entrada' in df.columns:
            decimais_entrada = 5 if is_meme else 2
            df['preco_entrada'] = df['preco_entrada'].apply(lambda x: f"${x:,.{decimais_entrada}f}" if pd.notna(x) else "")

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
        # Formatar PnL: se for USDT, exibir "—"
        if 'pnl' in df.columns and 'ativo' in df.columns:
            def _formatar_pnl_fechadas(row):
                ativo = str(row.get('ativo', '')).strip().upper()
                pnl = row.get('pnl')
                if ativo == 'USDT':
                    return "—"
                return f"{pnl:.2f}%" if pd.notna(pnl) and pnl is not None else ""
            df['pnl'] = df.apply(_formatar_pnl_fechadas, axis=1)
        elif 'pnl' in df.columns:
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


def display_historico_posicoes(produto_id=None, formatar=True, filtrar_colunas=True):
    """
    Exibe histórico consolidado de posições (abertas + fechadas)
    ordenado pela data de entrada.

    Args:
        produto_id: ID do produto (opcional, None para todos)
        formatar: Se True, formata valores monetários e atributos
        filtrar_colunas: Se True, aplica filtros de colunas hard-coded por produto.
                        Se False, mantém todas as colunas (para uso com visualizações do banco).
    """
    from analytics.queries import historico_posicoes

    df = historico_posicoes(produto_id)

    if df.empty:
        print("Nenhuma posição encontrada para o histórico")
        return df

    # Calcular dias em carteira (data_saida - data_entrada)
    # Para posições abertas, exibe "—"
    if 'data_entrada' in df.columns:
        def _calcular_dias_carteira_historico(row):
            # Verificar se a posição é aberta (status == 'open' ou data_saida é nula)
            status = row.get('status', '')
            data_saida = row.get('data_saida')
            if status == 'open' or pd.isna(data_saida) or data_saida is None:
                return "—"
            try:
                data_entrada = pd.to_datetime(row['data_entrada'])
                data_saida = pd.to_datetime(data_saida)
                if pd.notna(data_entrada) and pd.notna(data_saida):
                    return (data_saida - data_entrada).days
            except Exception:
                pass
            return "—"
        df['dias_carteira'] = df.apply(_calcular_dias_carteira_historico, axis=1)

    # Detectar tipo de produto (Spot ou Perpétuos, exceto 4970919917)
    tipo_spot = False
    tipo_perpetuos = False
    if produto_id is not None and produto_id != 4970919917:
        try:
            repo = get_repo()
            prod_info = repo.carregar_produto(produto_id)
            if prod_info and 'tipo' in prod_info and isinstance(prod_info['tipo'], str):
                tipo_str = prod_info['tipo'].lower()
                if 'spot' in tipo_str:
                    tipo_spot = True
                elif 'perpétuo' in tipo_str or 'perpetuo' in tipo_str:
                    tipo_perpetuos = True
        except Exception:
            pass

    # Para produtos Spot ou Perpétuos, criar colunas unificadas de preço final
    # (preco_saida para fechadas, preco_atual para abertas)
    # Usamos os nomes preco_saida e preco_saida_total, mas preenchemos com valores corretos
    if tipo_spot or tipo_perpetuos:
        # Criar preco_saida unificado: preco_saida se fechada, preco_atual se aberta
        # Garantir que estamos usando valores numéricos
        def _to_float(val):
            """Converte valor para float, lidando com strings formatadas"""
            if pd.isna(val) or val is None:
                return None
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                try:
                    # Tentar extrair número de string formatada como "$1,234.56"
                    cleaned = val.replace('$', '').replace(',', '').strip()
                    return float(cleaned)
                except:
                    return None
            return None
        
        def _get_preco_saida_unificado(row):
            preco_saida = row.get('preco_saida')
            preco_atual = row.get('preco_atual')
            
            if pd.notna(preco_saida):
                return _to_float(preco_saida)
            elif pd.notna(preco_atual):
                return _to_float(preco_atual)
            return None
        
        def _get_preco_saida_total_unificado(row):
            preco_saida_total = row.get('preco_saida_total')
            preco_atual_total = row.get('preco_atual_total')
            
            if pd.notna(preco_saida_total):
                return _to_float(preco_saida_total)
            elif pd.notna(preco_atual_total):
                return _to_float(preco_atual_total)
            return None
        
        # Criar/sobrescrever preco_saida e preco_saida_total com valores unificados (numéricos)
        df['preco_saida'] = df.apply(_get_preco_saida_unificado, axis=1)
        df['preco_saida_total'] = df.apply(_get_preco_saida_total_unificado, axis=1)

    if formatar:
        # Verificar se o produto tem "meme" no nome para usar 5 casas decimais em preco_entrada e preco_saida
        is_meme = False
        if produto_id is not None:
            try:
                repo = get_repo()
                prod_info = repo.carregar_produto(produto_id)
                if prod_info and 'nome' in prod_info:
                    nome_produto = str(prod_info['nome']).lower()
                    is_meme = 'meme' in nome_produto
            except Exception:
                pass
        
        # Função auxiliar para formatar valores monetários
        def _formatar_monetario(x):
            if pd.isna(x) or x is None:
                return "—"
            if isinstance(x, (int, float)):
                return f"${x:,.2f}"
            # Se já for string formatada, retornar como está
            return str(x)
        
        # Formatar preco_entrada com 5 casas se for meme, senão 2
        if 'preco_entrada' in df.columns:
            decimais_entrada = 5 if is_meme else 2
            df['preco_entrada'] = df['preco_entrada'].apply(
                lambda x: f"${x:,.{decimais_entrada}f}" if pd.notna(x) and isinstance(x, (int, float)) else "—"
            )
        if 'preco_atual' in df.columns and not tipo_spot:
            # Só formatar preco_atual se não for Spot (Spot usa preco_saida unificado)
            df['preco_atual'] = df['preco_atual'].apply(_formatar_monetario)

        # Formatar campos específicos de Spot
        if tipo_spot:
            if 'quantidade' in df.columns:
                df['quantidade'] = df['quantidade'].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "—"
                )
            if 'preco_entrada_total' in df.columns:
                df['preco_entrada_total'] = df['preco_entrada_total'].apply(_formatar_monetario)
            if 'preco_saida' in df.columns:
                decimais_saida = 5 if is_meme else 2
                def _formatar_preco_saida(x):
                    if pd.isna(x) or x is None:
                        return "—"
                    # Se já for string formatada, tentar extrair o número
                    if isinstance(x, str):
                        try:
                            cleaned = x.replace('$', '').replace(',', '').strip()
                            x = float(cleaned)
                        except:
                            return str(x)  # Retornar como está se não conseguir converter
                    if isinstance(x, (int, float)):
                        # Para valores muito pequenos, usar mais casas decimais para evitar mostrar apenas zeros
                        if abs(x) > 0 and abs(x) < 0.01:
                            # Usar até 8 casas decimais para valores muito pequenos (sem vírgula para evitar problemas)
                            return f"${x:.8f}".rstrip('0').rstrip('.')
                        else:
                            return f"${x:,.{decimais_saida}f}"
                    return "—"
                df['preco_saida'] = df['preco_saida'].apply(_formatar_preco_saida)
            if 'preco_saida_total' in df.columns:
                df['preco_saida_total'] = df['preco_saida_total'].apply(_formatar_monetario)

        # Formatar campos específicos de Perpétuos
        if tipo_perpetuos:
            if 'quantidade' in df.columns:
                df['quantidade'] = df['quantidade'].apply(
                    lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "—"
                )
            if 'preco_entrada_total' in df.columns:
                df['preco_entrada_total'] = df['preco_entrada_total'].apply(_formatar_monetario)
            if 'preco_saida' in df.columns:
                decimais_saida = 5 if is_meme else 2
                def _formatar_preco_saida(x):
                    if pd.isna(x) or x is None:
                        return "—"
                    # Se já for string formatada, tentar extrair o número
                    if isinstance(x, str):
                        try:
                            cleaned = x.replace('$', '').replace(',', '').strip()
                            x = float(cleaned)
                        except:
                            return str(x)  # Retornar como está se não conseguir converter
                    if isinstance(x, (int, float)):
                        # Para valores muito pequenos, usar mais casas decimais para evitar mostrar apenas zeros
                        if abs(x) > 0 and abs(x) < 0.01:
                            # Usar até 8 casas decimais para valores muito pequenos (sem vírgula para evitar problemas)
                            return f"${x:.8f}".rstrip('0').rstrip('.')
                        else:
                            return f"${x:,.{decimais_saida}f}"
                    return "—"
                df['preco_saida'] = df['preco_saida'].apply(_formatar_preco_saida)
            if 'preco_saida_total' in df.columns:
                df['preco_saida_total'] = df['preco_saida_total'].apply(_formatar_monetario)
            # Formatar PnL: se for USDT, exibir "—"
            if 'pnl' in df.columns and 'ativo' in df.columns:
                def _formatar_pnl_historico(row):
                    ativo = str(row.get('ativo', '')).strip().upper()
                    pnl = row.get('pnl')
                    if ativo == 'USDT':
                        return "—"
                    return f"{pnl:.2f}%" if pd.notna(pnl) and pnl is not None else "—"
                df['pnl'] = df.apply(_formatar_pnl_historico, axis=1)
            elif 'pnl' in df.columns:
                df['pnl'] = df['pnl'].apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "—"
                )

        # Atributos do produto (Signals)
        for col in ['alvo1', 'alvo2']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"${x:,.2f}" if pd.notna(x) else "—"
                )
        
        # Formatar stop_atual (para todos os produtos)
        if 'stop_atual' in df.columns:
            def _formatar_stop_atual_historico(x):
                if pd.isna(x) or x is None:
                    return "—"
                if isinstance(x, (int, float)):
                    if x == -1:
                        return "STOP ATINGIDO"
                    return f"${x:,.2f}"
                return str(x)
            df['stop_atual'] = df['stop_atual'].apply(_formatar_stop_atual_historico)

        # Formatar PnL: se for USDT, exibir "—"
        if 'pnl' in df.columns and 'ativo' in df.columns:
            def _formatar_pnl_historico_final(row):
                ativo = str(row.get('ativo', '')).strip().upper()
                pnl = row.get('pnl')
                if ativo == 'USDT':
                    return "—"
                if isinstance(pnl, str):
                    return pnl
                return f"{pnl:.2f}%" if pd.notna(pnl) and pnl is not None else "—"
            df['pnl'] = df.apply(_formatar_pnl_historico_final, axis=1)
        elif 'pnl' in df.columns:
            df['pnl'] = df['pnl'].apply(
                lambda x: (
                    str(x) if isinstance(x, str) 
                    else f"{x:.2f}%" if pd.notna(x) and x is not None 
                    else "—"
                )
            )
        if 'rr' in df.columns:
            df['rr'] = df['rr'].apply(
                lambda x: (
                    str(x) if isinstance(x, str) 
                    else f"{x:.2f}" if pd.notna(x) and x is not None 
                    else "—"
                )
            )
        if 'perfil' in df.columns:
            df['perfil'] = df['perfil'].apply(lambda x: x if pd.notna(x) else "—")
        if 'motivo' in df.columns:
            df['motivo'] = df['motivo'].apply(lambda x: x if pd.notna(x) else "—")

    # Se não estamos filtrando colunas, retornar DataFrame formatado
    if not filtrar_colunas:
        return df

    # Para o produto Alphacoins (ID 3476245316), EXC (ID 2150859854), HB (ID 2000449260) ou LC (ID 2394004756), mostrar apenas colunas específicas
    if produto_id == 3476245316 or produto_id == 2150859854 or produto_id == 2000449260 or produto_id == 2394004756:
        colunas_alphacoins_exc = [
            'status',
            'dias_carteira',
            'data_entrada',
            'data_saida',
            'ativo',
            'preco_entrada',
            'preco_saida',
            'pnl',
            'alocacao',
            'stop_atual',
        ]
        colunas_existentes = [c for c in colunas_alphacoins_exc if c in df.columns]
        df = df[colunas_existentes]
        # Formatar alocação se existir
        if 'alocacao' in df.columns:
            df['alocacao'] = df['alocacao'].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) and x is not None else "—"
            )
        # Substituir quaisquer NaN remanescentes por "—"
        df = df.fillna("—")
        return df

    # Para Crypto Signals, usar exatamente as colunas e ordem solicitadas (ordem original)
    if produto_id == 4970919917:
        colunas_signals = [
            'status',
            'dias_carteira',
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
            'stop_atual',  # Ordem original: antes de rr e motivo
            'rr',
            'motivo',
        ]
        colunas_existentes = [c for c in colunas_signals if c in df.columns]
        df = df[colunas_existentes]
        # Substituir quaisquer NaN remanescentes por "—"
        df = df.fillna("—")
        return df

    # Para produtos Spot, usar ordem específica
    if tipo_spot:
        # Salvar stop_atual antes de remover colunas
        stop_atual_col = df['stop_atual'].copy() if 'stop_atual' in df.columns else None

        colunas_spot = [
            'status',
            'dias_carteira',
            'data_entrada',
            'data_saida',
            'ativo',
            'quantidade',
            'preco_entrada',
            'preco_entrada_total',
            'preco_saida',
            'preco_saida_total',
            'pnl',
        ]
        colunas_existentes = [c for c in colunas_spot if c in df.columns]
        df = df[colunas_existentes]
        
        # Remover colunas indesejadas explicitamente (mantendo preco_saida e preco_saida_total que foram unificados)
        colunas_para_remover = ['id', 'produto_id', 'coingecko_id', 'side', 'perfil', 'motivo', 'rr', 'alvo1', 'alvo2', 
                                'preco_atual', 'preco_atual_total']
        df = df.drop(columns=[c for c in colunas_para_remover if c in df.columns])
        
        # Adicionar stop_atual como última coluna
        if stop_atual_col is not None:
            df['stop_atual'] = stop_atual_col
            # Formatar se ainda não foi formatado
            if df['stop_atual'].dtype != 'object':
                def _formatar_stop_atual(x):
                    if pd.isna(x) or x is None:
                        return "—"
                    if isinstance(x, (int, float)):
                        if x == -1:
                            return "STOP ATINGIDO"
                        return f"${x:,.2f}"
                    return str(x)
                df['stop_atual'] = df['stop_atual'].apply(_formatar_stop_atual)

        # Substituir quaisquer NaN remanescentes por "—"
        df = df.fillna("—")
        return df

    # Para produtos Perpétuos, usar ordem específica
    if tipo_perpetuos:
        colunas_perpetuos = [
            'status',
            'dias_carteira',
            'data_entrada',
            'data_saida',
            'ativo',
            'side',
            'quantidade',
            'preco_entrada',
            'preco_entrada_total',
            'preco_saida',
            'preco_saida_total',
            'pnl',
        ]
        colunas_existentes = [c for c in colunas_perpetuos if c in df.columns]
        df = df[colunas_existentes]
        
        # Salvar stop_atual antes de remover colunas
        stop_atual_col = df['stop_atual'].copy() if 'stop_atual' in df.columns else None
        
        # Remover colunas indesejadas
        colunas_para_remover = ['id', 'produto_id', 'coingecko_id', 'perfil', 'motivo', 'rr', 'alvo1', 'alvo2', 'preco_atual']
        df = df.drop(columns=[c for c in colunas_para_remover if c in df.columns])
        
        # Adicionar stop_atual como última coluna
        if stop_atual_col is not None:
            df['stop_atual'] = stop_atual_col
            # Formatar se ainda não foi formatado
            if df['stop_atual'].dtype != 'object':
                def _formatar_stop_atual(x):
                    if pd.isna(x) or x is None:
                        return "—"
                    if isinstance(x, (int, float)):
                        if x == -1:
                            return "STOP ATINGIDO"
                        return f"${x:,.2f}"
                    return str(x)
                df['stop_atual'] = df['stop_atual'].apply(_formatar_stop_atual)

        # Substituir quaisquer NaN remanescentes por "—"
        df = df.fillna("—")
        return df

    # Caso genérico: manter ordenação padrão (priorizar principais + atributos)
    colunas_principais = [
        'id', 'ativo', 'side', 'status', 'dias_carteira', 'data_entrada', 'preco_entrada',
        'data_saida', 'preco_saida', 'preco_atual'
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

