import pandas as pd
from pathlib import Path
from storage.sqlite_repo import SQLiteRepo
from services.valor_diario_service import ValorDiarioService

# Queries usando SQLite

def calcular_rr(preco_atual, alvo2, stop_atual):
    """
    Calcula o Risk/Reward ratio dinamicamente usando alvo2 e stop atual.
    
    Fórmula:
      RR = |(alvo2 / preco_atual - 1)| / |(stop_atual / preco_atual - 1)|
    
    Args:
        preco_atual: Preço atual (ou de saída) do ativo
        alvo2: Segundo alvo de preço (Target 2)
        stop_atual: Último valor de stop da posição (ou stop de referência)
    
    Returns:
        float ou None: RR calculado (sempre positivo) ou None se não for possível calcular
    """
    if pd.isna(preco_atual) or pd.isna(alvo2) or pd.isna(stop_atual):
        return None
    
    if preco_atual == 0:
        return None
    
    # % até o alvo2 (ganho potencial)
    pct_alvo2 = abs((alvo2 / preco_atual) - 1)
    # % até o stop (perda potencial)
    pct_stop = abs((stop_atual / preco_atual) - 1)
    
    if pct_stop == 0:
        return None
    
    rr = pct_alvo2 / pct_stop
    return round(rr, 2)

def posicoes_abertas(produto_id=None):
    """Retorna posições abertas com preço atual e atributos do produto"""
    repo = SQLiteRepo()

    # Detectar tipo de produto (Spot ou Perpétuos, exceto 4970919917)
    tipo_spot = False
    tipo_perpetuos = False
    if produto_id and produto_id != 4970919917:
        try:
            prod_info = repo.carregar_produto(produto_id)
            if prod_info and 'tipo' in prod_info and isinstance(prod_info['tipo'], str):
                tipo_str = prod_info['tipo'].lower()
                if 'spot' in tipo_str:
                    tipo_spot = True
                elif 'perpétuo' in tipo_str or 'perpetuo' in tipo_str:
                    tipo_perpetuos = True
        except Exception:
            pass
    
    import sqlite3
    conn = sqlite3.connect(repo.db_path)
    try:
        if produto_id:
            if produto_id == 4970919917:
                # Produto Crypto Signals: incluir atributos específicos (motivo, perfil, alvos)
                query = """
                    SELECT p.*, 
                           a.motivo, a.perfil, a.alvo1, a.alvo2
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    WHERE p.status = 'open' AND p.produto_id = ?
                """
                df = pd.read_sql_query(query, conn, params=(produto_id,))
            else:
                # Outros produtos (inclui Spot e Perpétuos): incluir atributos genéricos
                query = """
                    SELECT 
                        p.*,
                        a.quantidade,
                        a.preco_entrada_total,
                        a.perfil,
                        a.motivo,
                        a.alvo1,
                        a.alvo2
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    WHERE p.status = 'open' AND p.produto_id = ?
                """
                df = pd.read_sql_query(query, conn, params=(produto_id,))
        else:
            # Sem filtro de produto: incluir atributos quando existirem (pode misturar produtos)
            query = """
                SELECT p.*, 
                       a.motivo, a.perfil, a.alvo1, a.alvo2
                FROM posicoes p
                LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                WHERE p.status = 'open'
            """
            df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    
    # Adicionar preço atual, stop atual, RR e PnL dinamicamente para posições abertas
    if not df.empty:
        # Primeiro, calcular preço atual para todas as posições
        precos_atuais = []
        for _, row in df.iterrows():
            coingecko_id = row.get('coingecko_id')
            if pd.notna(coingecko_id) and coingecko_id:
                preco_atual = ValorDiarioService.obter_preco_atual(coingecko_id)
                # Para mog-coin, multiplicar por 1M (1.000.000) pois o preço no CoinGecko é por token
                if coingecko_id == 'mog-coin' and preco_atual is not None:
                    preco_atual = preco_atual * 1_000_000
                precos_atuais.append(preco_atual)
            else:
                preco_atual = None
                precos_atuais.append(None)

        df['preco_atual'] = precos_atuais

        # Para produtos Spot, calcular preco_atual_total (quantidade * preco_atual)
        if tipo_spot:
            precos_atuais_totais = []
            for _, row in df.iterrows():
                quantidade = row.get('quantidade')
                preco_atual = row.get('preco_atual')
                if pd.notna(quantidade) and pd.notna(preco_atual) and quantidade != 0:
                    preco_atual_total = quantidade * preco_atual
                    precos_atuais_totais.append(preco_atual_total)
                else:
                    precos_atuais_totais.append(None)
            df['preco_atual_total'] = precos_atuais_totais

        # Calcular stop_atual para TODOS os produtos (último stop de cada posição)
        stops_atuais = []
        posicao_ids = df['id'].tolist()
        stops_map = {}
        if posicao_ids:
            import sqlite3
            conn_stops = sqlite3.connect(repo.db_path)
            try:
                for pos_id in posicao_ids:
                    df_stops = pd.read_sql_query(
                        "SELECT valor FROM stops WHERE posicao_id = ? ORDER BY data DESC LIMIT 1",
                        conn_stops,
                        params=(pos_id,)
                    )
                    if not df_stops.empty:
                        stops_map[pos_id] = df_stops.iloc[0]['valor']
            finally:
                conn_stops.close()
        
        for _, row in df.iterrows():
            posicao_id = row.get('id')
            stop_atual = stops_map.get(posicao_id) if posicao_id else None
            stops_atuais.append(stop_atual)
        
        df['stop_atual'] = stops_atuais

        # Se for o produto Crypto Signals, calcular RR e PnL
        if produto_id == 4970919917 or (produto_id is None and 'alvo2' in df.columns):
            rrs = []
            pnls = []
            
            for _, row in df.iterrows():
                posicao_id = row.get('id')
                stop_atual = stops_map.get(posicao_id) if posicao_id else None
                
                preco_atual = row.get('preco_atual')
                alvo2 = row.get('alvo2') if 'alvo2' in df.columns else None
                # RR: |(alvo2/preco_atual - 1)| / |(stop_atual/preco_atual - 1)|
                rr = calcular_rr(preco_atual, alvo2, stop_atual) if preco_atual is not None else None
                rrs.append(rr)

                # PnL para Crypto Signals: calcular como long e inverter sinal para short
                preco_entrada = row.get('preco_entrada')
                side = str(row.get('side', 'long')).lower()
                ativo = str(row.get('ativo', '')).strip().upper()
                eh_signals = (produto_id == 4970919917) or (produto_id is None and row.get('produto_id') == 4970919917)
                # Se for USDT, PnL deve ser None (será exibido como "—")
                if ativo == 'USDT':
                    pnl = None
                elif eh_signals and preco_atual is not None and pd.notna(preco_entrada) and preco_entrada not in (0,):
                    try:
                        # Calcular como long (rendimento normal)
                        pnl = ((preco_atual / preco_entrada) - 1) * 100.0
                        # Para short, inverter o sinal
                        if side == 'short':
                            pnl = -pnl
                    except ZeroDivisionError:
                        pnl = None
                else:
                    pnl = None
                pnls.append(pnl)
            
            df['rr'] = rrs
            df['pnl'] = pnls

        # Para produtos Spot, calcular PnL usando notional (quantidade * preço)
        if tipo_spot:
            pnls_spot = []
            for _, row in df.iterrows():
                ativo = str(row.get('ativo', '')).strip().upper()
                # Se for USDT, PnL deve ser None (será exibido como "—")
                if ativo == 'USDT':
                    pnls_spot.append(None)
                    continue
                
                preco_entrada = row.get('preco_entrada')
                preco_atual = row.get('preco_atual')
                quantidade = row.get('quantidade')

                if pd.isna(preco_entrada) or pd.isna(preco_atual):
                    pnls_spot.append(None)
                    continue

                # a = quantidade * preco_entrada
                # b = quantidade * preco_atual
                if quantidade is None or pd.isna(quantidade) or quantidade == 0:
                    a = preco_entrada
                    b = preco_atual
                else:
                    a = quantidade * preco_entrada
                    b = quantidade * preco_atual

                if a == 0:
                    pnls_spot.append(None)
                    continue

                try:
                    pnl = ((b / a) - 1.0) * 100.0
                except ZeroDivisionError:
                    pnl = None
                pnls_spot.append(pnl)

            df['pnl'] = pnls_spot

        # Para produtos Perpétuos (exceto 4970919917), calcular preco_saida, preco_saida_total e PnL
        if tipo_perpetuos:
            # Para posições abertas, preco_saida = preco_atual (puxado do coingecko)
            df['preco_saida'] = df['preco_atual']
            
            # Para posições abertas, preco_saida_total = quantidade * preco_saida (calculado dinamicamente)
            precos_saida_totais = []
            for _, row in df.iterrows():
                quantidade = row.get('quantidade')
                preco_saida = row.get('preco_saida')
                if pd.notna(quantidade) and pd.notna(preco_saida) and quantidade != 0:
                    preco_saida_total = quantidade * preco_saida
                    precos_saida_totais.append(preco_saida_total)
                else:
                    precos_saida_totais.append(None)
            df['preco_saida_total'] = precos_saida_totais

            # Calcular PnL para Perpétuos: calcular como long e inverter sinal para short
            pnls_perpetuos = []
            for _, row in df.iterrows():
                ativo = str(row.get('ativo', '')).strip().upper()
                # Se for USDT, PnL deve ser None (será exibido como "—")
                if ativo == 'USDT':
                    pnls_perpetuos.append(None)
                    continue
                
                quantidade = row.get('quantidade')
                preco_entrada = row.get('preco_entrada')
                preco_saida_total = row.get('preco_saida_total')
                side = str(row.get('side', 'long')).lower()

                if pd.isna(quantidade) or pd.isna(preco_entrada) or pd.isna(preco_saida_total):
                    pnls_perpetuos.append(None)
                    continue

                # a = quantidade * preco_entrada
                if quantidade is None or quantidade == 0:
                    pnls_perpetuos.append(None)
                    continue

                a = quantidade * preco_entrada
                b = preco_saida_total

                if a == 0 or b == 0:
                    pnls_perpetuos.append(None)
                    continue

                try:
                    # Calcular como long (rendimento normal)
                    pnl = ((b / a) - 1.0) * 100.0
                    # Para short, inverter o sinal
                    if side == 'short':
                        pnl = -pnl
                except ZeroDivisionError:
                    pnl = None
                pnls_perpetuos.append(pnl)

            df['pnl'] = pnls_perpetuos
        
        # Para o produto Alphacoins (ID 3476245316), EXC (ID 2150859854), HB (ID 2000449260) e LC (ID 2394004756), calcular PnL e adicionar alocação atual
        if produto_id == 3476245316 or produto_id == 2150859854 or produto_id == 2000449260 or produto_id == 2394004756:
            # Calcular PnL simples: (preco_atual / preco_entrada - 1) * 100
            if 'pnl' not in df.columns:
                pnls_alphacoins = []
                for _, row in df.iterrows():
                    ativo = str(row.get('ativo', '')).strip().upper()
                    # Se for USDT, PnL deve ser None (será exibido como "—")
                    if ativo == 'USDT':
                        pnls_alphacoins.append(None)
                        continue
                    
                    preco_entrada = row.get('preco_entrada')
                    preco_atual = row.get('preco_atual')
                    if pd.notna(preco_entrada) and pd.notna(preco_atual) and preco_entrada != 0:
                        try:
                            pnl = ((preco_atual / preco_entrada) - 1.0) * 100.0
                        except ZeroDivisionError:
                            pnl = None
                    else:
                        pnl = None
                    pnls_alphacoins.append(pnl)
                df['pnl'] = pnls_alphacoins
            
            # Adicionar alocação atual (percentual mais recente por ATIVO, não por posição)
            # Isso é necessário porque posições podem ter sido recriadas, mas a alocação é por ativo
            alocacoes_atuais = []
            if not df.empty:
                import sqlite3
                conn_aloc = sqlite3.connect(repo.db_path)
                try:
                    for _, row in df.iterrows():
                        ativo = str(row.get('ativo', '')).strip().upper()
                        # Buscar a última alocação do ativo (via JOIN com posições), independente do status
                        df_aloc = pd.read_sql_query("""
                            SELECT a.percentual 
                            FROM alocacoes a
                            JOIN posicoes p ON a.posicao_id = p.id
                            WHERE p.ativo = ? AND p.produto_id = ?
                            ORDER BY a.data DESC 
                            LIMIT 1
                        """, conn_aloc, params=(ativo, produto_id))
                        if not df_aloc.empty:
                            alocacoes_atuais.append(df_aloc.iloc[0]['percentual'])
                        else:
                            alocacoes_atuais.append(None)
                finally:
                    conn_aloc.close()
            else:
                alocacoes_atuais = []
            
            df['alocacao'] = alocacoes_atuais
    
    return df

def posicoes_fechadas(produto_id=None):
    """Retorna posições fechadas com atributos do produto"""
    repo = SQLiteRepo()
    
    # Detectar tipo de produto (Spot ou Perpétuos, exceto 4970919917)
    tipo_spot = False
    tipo_perpetuos = False
    if produto_id and produto_id != 4970919917:
        try:
            prod_info = repo.carregar_produto(produto_id)
            if prod_info and 'tipo' in prod_info and isinstance(prod_info['tipo'], str):
                tipo_str = prod_info['tipo'].lower()
                if 'spot' in tipo_str:
                    tipo_spot = True
                elif 'perpétuo' in tipo_str or 'perpetuo' in tipo_str:
                    tipo_perpetuos = True
        except Exception:
            pass
    
    import sqlite3
    conn = sqlite3.connect(repo.db_path)
    try:
        if produto_id:
            if produto_id == 4970919917:
                # Produto Crypto Signals: incluir atributos específicos
                query = """
                    SELECT p.*, 
                           a.motivo, a.perfil, a.alvo1, a.alvo2
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    WHERE p.status = 'closed' AND p.produto_id = ?
                """
                df = pd.read_sql_query(query, conn, params=(produto_id,))
            else:
                # Outros produtos (inclui Spot e Perpétuos): incluir quantidade e preco_entrada_total
                # preco_saida_total é calculado dinamicamente
                query = """
                    SELECT 
                        p.*,
                        a.quantidade,
                        a.preco_entrada_total,
                        a.perfil,
                        a.motivo,
                        a.alvo1,
                        a.alvo2
                    FROM posicoes p
                    LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                    WHERE p.status = 'closed' AND p.produto_id = ?
                """
                df = pd.read_sql_query(query, conn, params=(produto_id,))
        else:
            # Sem filtro de produto: incluir atributos quando existirem
            query = """
                SELECT p.*, 
                       a.motivo, a.perfil, a.alvo1, a.alvo2
                FROM posicoes p
                LEFT JOIN posicao_atributos_produto a ON p.id = a.posicao_id
                WHERE p.status = 'closed'
            """
            df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
    
    # Para posições fechadas, preço_atual = preço_saida
    # Calcular stop_atual para TODAS as posições fechadas
    # Calcular RR e PnL para Crypto Signals (produto_id 4970919917)
    # Calcular preco_saida_total e PnL para produtos Spot
    if not df.empty:
        df['preco_atual'] = df['preco_saida']
        
        # Calcular stop_atual (último stop de cada posição fechada)
        stops_atuais = []
        posicao_ids = df['id'].tolist()
        stops_map = {}
        if posicao_ids:
            import sqlite3
            conn_stops = sqlite3.connect(repo.db_path)
            try:
                for pos_id in posicao_ids:
                    df_stops = pd.read_sql_query(
                        "SELECT valor FROM stops WHERE posicao_id = ? ORDER BY data DESC LIMIT 1",
                        conn_stops,
                        params=(pos_id,)
                    )
                    if not df_stops.empty:
                        stops_map[pos_id] = df_stops.iloc[0]['valor']
            finally:
                conn_stops.close()
        
        for _, row in df.iterrows():
            posicao_id = row.get('id')
            stop_atual = stops_map.get(posicao_id) if posicao_id else None
            stops_atuais.append(stop_atual)
        
        df['stop_atual'] = stops_atuais
        
        # Para produtos Spot, calcular preco_saida_total (quantidade * preco_saida)
        if tipo_spot:
            precos_saida_totais = []
            for _, row in df.iterrows():
                quantidade = row.get('quantidade')
                preco_saida = row.get('preco_saida')
                if pd.notna(quantidade) and pd.notna(preco_saida) and quantidade != 0:
                    preco_saida_total = quantidade * preco_saida
                    precos_saida_totais.append(preco_saida_total)
                else:
                    precos_saida_totais.append(None)
            df['preco_saida_total'] = precos_saida_totais
        
        # Para produtos Perpétuos, calcular preco_saida_total (quantidade * preco_saida)
        if tipo_perpetuos:
            precos_saida_totais = []
            for _, row in df.iterrows():
                quantidade = row.get('quantidade')
                preco_saida = row.get('preco_saida')
                if pd.notna(quantidade) and pd.notna(preco_saida) and quantidade != 0:
                    preco_saida_total = quantidade * preco_saida
                    precos_saida_totais.append(preco_saida_total)
                else:
                    precos_saida_totais.append(None)
            df['preco_saida_total'] = precos_saida_totais
        
        rrs = []
        pnls = []
        for _, row in df.iterrows():
            eh_signals = (produto_id == 4970919917) or (produto_id is None and row.get('produto_id') == 4970919917)
            preco_entrada = row.get('preco_entrada')
            preco_saida = row.get('preco_saida')
            preco_atual = row.get('preco_atual')
            alvo2 = row.get('alvo2') if 'alvo2' in df.columns else None
            stop_ref = row.get('alvo1') if 'alvo1' in df.columns else None

            # RR apenas para Crypto Signals e se tivermos dados suficientes
            if eh_signals and preco_atual is not None and not pd.isna(alvo2) and not pd.isna(stop_ref):
                rr = calcular_rr(preco_atual, alvo2, stop_ref)
            else:
                rr = None
            rrs.append(rr)

            # PnL: para Crypto Signals: ((preco_saida / preco_entrada) - 1) * 100
            # Para Spot: ((quantidade * preco_saida) / (quantidade * preco_entrada) - 1) * 100
            # Para Perpétuos: ((preco_saida_total / (quantidade * preco_entrada)) - 1) * 100
            # Para Alphacoins: ((preco_saida / preco_entrada) - 1) * 100
            ativo = str(row.get('ativo', '')).strip().upper()
            # Se for USDT, PnL deve ser None (será exibido como "—")
            if ativo == 'USDT':
                pnl = None
            elif produto_id == 3476245316 or produto_id == 2150859854 or produto_id == 2000449260 or produto_id == 2394004756:
                # Alphacoins/EXC: calcular PnL simples (preco_saida / preco_entrada - 1) * 100
                if pd.notna(preco_entrada) and pd.notna(preco_saida) and preco_entrada != 0:
                    try:
                        pnl = ((preco_saida / preco_entrada) - 1.0) * 100.0
                    except ZeroDivisionError:
                        pnl = None
                else:
                    pnl = None
            elif tipo_spot:
                quantidade = row.get('quantidade')
                if pd.notna(preco_entrada) and pd.notna(preco_saida) and pd.notna(quantidade) and quantidade != 0:
                    a = quantidade * preco_entrada
                    b = quantidade * preco_saida
                    if a != 0:
                        try:
                            pnl = ((b / a) - 1.0) * 100.0
                        except ZeroDivisionError:
                            pnl = None
                    else:
                        pnl = None
                else:
                    pnl = None
            elif tipo_perpetuos:
                quantidade = row.get('quantidade')
                preco_saida = row.get('preco_saida')
                side = str(row.get('side', 'long')).lower()
                if pd.notna(preco_entrada) and pd.notna(preco_saida) and pd.notna(quantidade) and quantidade != 0:
                    a = quantidade * preco_entrada
                    b = quantidade * preco_saida  # preco_saida_total = quantidade * preco_saida
                    if a != 0 and b != 0:
                        try:
                            # Calcular como long (rendimento normal)
                            pnl = ((b / a) - 1.0) * 100.0
                            # Para short, inverter o sinal
                            if side == 'short':
                                pnl = -pnl
                        except ZeroDivisionError:
                            pnl = None
                    else:
                        pnl = None
                else:
                    pnl = None
            elif eh_signals and preco_saida is not None and not pd.isna(preco_entrada) and preco_entrada not in (0,):
                side = str(row.get('side', 'long')).lower()
                try:
                    # Calcular como long (rendimento normal)
                    pnl = ((preco_saida / preco_entrada) - 1) * 100.0
                    # Para short, inverter o sinal
                    if side == 'short':
                        pnl = -pnl
                except ZeroDivisionError:
                    pnl = None
            else:
                pnl = None
            pnls.append(pnl)

        df['rr'] = rrs
        df['pnl'] = pnls
    
    # Para o produto Alphacoins (ID 3476245316), EXC (ID 2150859854), HB (ID 2000449260) e LC (ID 2394004756), adicionar alocação atual
    if produto_id == 3476245316 or produto_id == 2150859854 or produto_id == 2000449260 or produto_id == 2394004756:
        # Adicionar alocação atual (percentual mais recente por ATIVO, não por posição)
        # Isso é necessário porque posições podem ter sido recriadas, mas a alocação é por ativo
        alocacoes_atuais = []
        if not df.empty:
            import sqlite3
            conn_aloc = sqlite3.connect(repo.db_path)
            try:
                for _, row in df.iterrows():
                    ativo = str(row.get('ativo', '')).strip().upper()
                    # Buscar a última alocação do ativo (via JOIN com posições), independente do status
                    df_aloc = pd.read_sql_query("""
                        SELECT a.percentual 
                        FROM alocacoes a
                        JOIN posicoes p ON a.posicao_id = p.id
                        WHERE p.ativo = ? AND p.produto_id = ?
                        ORDER BY a.data DESC 
                        LIMIT 1
                    """, conn_aloc, params=(ativo, produto_id))
                    if not df_aloc.empty:
                        alocacoes_atuais.append(df_aloc.iloc[0]['percentual'])
                    else:
                        alocacoes_atuais.append(None)
            finally:
                conn_aloc.close()
        else:
            alocacoes_atuais = []
        
        df['alocacao'] = alocacoes_atuais
    
    return df


def manutencoes_signals(produto_id=4970919917):
    """
    Retorna as manutenções (stops adicionais) das posições abertas do produto Crypto Signals.

    Cada linha representa um stop de manutenção (a partir do segundo stop) de uma posição aberta.

    Colunas retornadas:
        - posicao_id
        - data_manutencao (data do stop)
        - ativo
        - side
        - perfil
        - preco_entrada
        - preco_atual (preço atual do ativo)
        - pnl (%)
        - alvo1
        - alvo2
        - stop_valor (valor do stop desta manutenção)
        - rr (RR calculado com base no stop da linha)
    """
    # Garantir que é o produto Crypto Signals
    if produto_id != 4970919917:
        return pd.DataFrame(columns=[
            'posicao_id', 'data_manutencao', 'ativo', 'side', 'perfil',
            'preco_entrada', 'preco_atual', 'pnl', 'alvo1', 'alvo2',
            'stop_valor', 'rr'
        ])

    repo = SQLiteRepo()
    import sqlite3
    conn = sqlite3.connect(repo.db_path)

    try:
        # Selecionar todas as posições abertas do produto e seus stops (exceto o primeiro stop)
        query = """
            SELECT 
                p.id          AS posicao_id,
                s.data        AS data_manutencao,
                p.ativo,
                p.side,
                a.perfil,
                p.preco_entrada,
                p.coingecko_id,
                a.alvo1,
                a.alvo2,
                s.valor       AS stop_valor
            FROM posicoes p
            JOIN stops s 
                ON s.posicao_id = p.id
            LEFT JOIN posicao_atributos_produto a 
                ON a.posicao_id = p.id
            WHERE 
                p.produto_id = ?
                AND p.status = 'open'
                -- Apenas manutenções (ignora o primeiro stop da posição)
                AND s.data > (
                    SELECT MIN(s2.data) 
                    FROM stops s2 
                    WHERE s2.posicao_id = p.id
                )
            ORDER BY p.id, s.data
        """
        df = pd.read_sql_query(query, conn, params=(produto_id,))
    finally:
        conn.close()

    if df.empty:
        return df

    # Calcular preço atual por ativo (coingecko_id) de forma eficiente
    precos_atuais_map = {}
    coingeckos_unicos = df['coingecko_id'].dropna().unique()
    for cid in coingeckos_unicos:
        try:
            preco_atual = ValorDiarioService.obter_preco_atual(cid)
            # Para mog-coin, multiplicar por 1M (1.000.000) pois o preço no CoinGecko é por token
            if cid == 'mog-coin' and preco_atual is not None:
                preco_atual = preco_atual * 1_000_000
            precos_atuais_map[cid] = preco_atual
        except Exception:
            precos_atuais_map[cid] = None

    df['preco_atual'] = df['coingecko_id'].map(precos_atuais_map)

    # Calcular PnL: calcular como long e inverter sinal para short
    def _calc_pnl(row):
        ativo = str(row.get('ativo', '')).strip().upper()
        # Se for USDT, PnL deve ser None (será exibido como "—")
        if ativo == 'USDT':
            return None
        pe = row.get('preco_entrada')
        pa = row.get('preco_atual')
        side = str(row.get('side', 'long')).lower()
        if pd.isna(pe) or pe in (0, None) or pd.isna(pa):
            return None
        try:
            # Calcular como long (rendimento normal)
            pnl = (pa / pe - 1.0) * 100.0
            # Para short, inverter o sinal
            if side == 'short':
                pnl = -pnl
            return pnl
        except ZeroDivisionError:
            return None

    df['pnl'] = df.apply(_calc_pnl, axis=1)

    # Calcular RR para cada manutenção usando alvo2 e stop_valor
    def _calc_rr_row(row):
        pa = row.get('preco_atual')
        a2 = row.get('alvo2')
        sv = row.get('stop_valor')
        return calcular_rr(pa, a2, sv) if pa is not None else None

    df['rr'] = df.apply(_calc_rr_row, axis=1)

    # Selecionar apenas as colunas desejadas e remover coingecko_id
    colunas = [
        'posicao_id', 'data_manutencao', 'ativo', 'side', 'perfil',
        'preco_entrada', 'preco_atual', 'pnl', 'alvo1', 'alvo2',
        'stop_valor', 'rr'
    ]
    colunas_existentes = [c for c in colunas if c in df.columns]
    return df[colunas_existentes]


def historico_posicoes(produto_id=None):
    """
    Retorna um histórico consolidado de posições (abertas + fechadas)
    para um produto, ordenado pela data de entrada.
    """
    # Posições abertas e fechadas já trazem preço atual, RR, PnL etc.
    df_abertas = posicoes_abertas(produto_id)
    df_fechadas = posicoes_fechadas(produto_id)

    frames = []
    if df_abertas is not None and not df_abertas.empty:
        frames.append(df_abertas.copy())
    if df_fechadas is not None and not df_fechadas.empty:
        frames.append(df_fechadas.copy())

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Garantir ordenação pela data de entrada
    if 'data_entrada' in df.columns:
        try:
            df['_data_entrada_sort'] = pd.to_datetime(df['data_entrada'])
        except Exception:
            df['_data_entrada_sort'] = df['data_entrada']
        df = df.sort_values('_data_entrada_sort').drop(columns=['_data_entrada_sort'])

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
