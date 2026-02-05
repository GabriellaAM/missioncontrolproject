import pandas as pd
import sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from storage.sqlite_repo import SQLiteRepo
from services.valor_diario_service import ValorDiarioService
from services.atr_stop_service import atualizar_stops_posicoes_abertas
from services.bitget_service import sync_positions_with_exchange, get_bitget_credentials

# Queries usando SQLite

# ============================================================
# HELPER FUNCTIONS FOR BATCH LOADING (Performance Optimization)
# ============================================================

def _batch_load_stops(repo, posicao_ids: list) -> dict:
    """
    Batch load latest stop for each position in a single query.
    Returns dict mapping posicao_id -> stop_valor
    """
    if not posicao_ids:
        return {}

    placeholders = ','.join(['?' for _ in posicao_ids])
    query = f"""
        SELECT posicao_id, valor
        FROM stops s1
        WHERE posicao_id IN ({placeholders})
        AND data = (
            SELECT MAX(s2.data) FROM stops s2 WHERE s2.posicao_id = s1.posicao_id
        )
    """

    conn = sqlite3.connect(repo.db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query, posicao_ids)
        return {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        conn.close()


def _batch_load_prices(coingecko_ids: list) -> dict:
    """
    Batch load current prices for unique coingecko_ids.
    Returns dict mapping coingecko_id -> price
    Uses single API call to CoinGecko for all IDs (much faster than individual calls).
    """
    if not coingecko_ids:
        return {}

    unique_ids = list(set(cid for cid in coingecko_ids if cid and pd.notna(cid)))
    if not unique_ids:
        return {}

    # Single API call for all prices
    prices = ValorDiarioService.obter_precos_batch(unique_ids)

    # Apply mog-coin multiplier
    if 'mog-coin' in prices and prices['mog-coin'] is not None:
        prices['mog-coin'] = prices['mog-coin'] * 1_000_000

    return prices


def _batch_load_allocations(repo, produto_id: int, ativos: list) -> dict:
    """
    Batch load latest allocation for each ativo in a single query.
    Returns dict mapping ativo -> percentual

    OPTIMIZED: Uses window function instead of correlated subquery (100x+ faster).
    """
    if not ativos:
        return {}

    # Normalize ativos to uppercase
    ativos_upper = [str(a).strip().upper() for a in ativos if a]
    if not ativos_upper:
        return {}

    placeholders = ','.join(['?' for _ in ativos_upper])

    # Use CTE with ROW_NUMBER to get latest allocation per ativo efficiently
    # This avoids the O(n²) correlated subquery
    query = f"""
        WITH ranked_allocations AS (
            SELECT
                UPPER(TRIM(p.ativo)) as ativo,
                a.percentual,
                ROW_NUMBER() OVER (
                    PARTITION BY UPPER(TRIM(p.ativo))
                    ORDER BY a.data DESC
                ) as rn
            FROM alocacoes a
            JOIN posicoes p ON a.posicao_id = p.id
            WHERE p.produto_id = ?
        )
        SELECT ativo, percentual
        FROM ranked_allocations
        WHERE rn = 1 AND ativo IN ({placeholders})
    """

    conn = sqlite3.connect(repo.db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query, [produto_id] + ativos_upper)
        return {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        conn.close()


# Cache for product type detection (avoid repeated DB lookups)
_product_type_cache = {}

def _get_product_type(repo, produto_id):
    """Get product type with caching to avoid repeated DB lookups."""
    if produto_id in _product_type_cache:
        return _product_type_cache[produto_id]

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

    result = (tipo_spot, tipo_perpetuos)
    _product_type_cache[produto_id] = result
    return result

# ============================================================

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

    # Detectar tipo de produto (com cache)
    tipo_spot, tipo_perpetuos = _get_product_type(repo, produto_id)

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

    # ATR stop updates: only run if there are positions with atr_multiplier configured
    # Optimized to ~0.05s per position (was 0.15s before numpy optimization)
    if not df.empty and 'atr_multiplier' in df.columns:
        has_atr = df['atr_multiplier'].notna().any()
        if has_atr:
            atualizar_stops_posicoes_abertas(repo, produto_id, verbose=False)

    # OPTIMIZED: Run Bitget sync and CoinGecko price fetch in PARALLEL
    # This saves ~1s by overlapping network latencies
    price_map = {}
    bitget_ran = False

    if not df.empty and produto_id:
        coingecko_ids = df['coingecko_id'].tolist()
        produto_info = repo.carregar_produto(produto_id)
        has_bitget = produto_info and get_bitget_credentials(produto_info['nome'])

        if has_bitget:
            # Run both in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_prices = executor.submit(_batch_load_prices, coingecko_ids)
                future_bitget = executor.submit(sync_positions_with_exchange, repo, produto_id, False)

                price_map = future_prices.result()
                future_bitget.result()
                bitget_ran = True
        else:
            # Just fetch prices (no Bitget)
            price_map = _batch_load_prices(coingecko_ids)

    # Reload positions after Bitget sync to get updated quantities
    if bitget_ran:
        conn = sqlite3.connect(repo.db_path)
        try:
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
        finally:
            conn.close()

    # Adicionar preço atual, stop atual, RR e PnL dinamicamente para posições abertas
    if not df.empty:
        # Apply pre-fetched prices (already loaded in parallel above)
        if not price_map:
            price_map = _batch_load_prices(df['coingecko_id'].tolist())
        df['preco_atual'] = df['coingecko_id'].map(price_map)

        # Sobrescrever preco_atual com dados da Bitget quando disponíveis
        # Para perpétuos: preco_atual = entry + (pnl / qty) para LONG, entry - (pnl / qty) para SHORT
        if 'pnl_exchange' in df.columns and 'preco_entrada_exchange' in df.columns:
            for idx, row in df.iterrows():
                pnl = row.get('pnl_exchange')
                entry = row.get('preco_entrada_exchange')
                qty = row.get('quantidade')
                if pd.notna(pnl) and pd.notna(entry) and pd.notna(qty) and qty != 0:
                    side = str(row.get('side', 'long')).lower()
                    if side == 'short':
                        df.at[idx, 'preco_atual'] = entry - (pnl / qty)
                    else:
                        df.at[idx, 'preco_atual'] = entry + (pnl / qty)

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

        # OPTIMIZED: Batch load stops in single query
        posicao_ids = df['id'].tolist()
        stops_map = _batch_load_stops(repo, posicao_ids)
        df['stop_atual'] = df['id'].map(stops_map)

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
                preco_atual = row.get('preco_atual')
                preco_saida_total = row.get('preco_saida_total')
                side = str(row.get('side', 'long')).lower()

                # Verificar se temos preco_entrada válido
                if pd.isna(preco_entrada) or preco_entrada == 0:
                    pnls_perpetuos.append(None)
                    continue

                # Se quantidade existe, usar fórmula com notional
                if pd.notna(quantidade) and quantidade != 0 and pd.notna(preco_saida_total):
                    a = quantidade * preco_entrada
                    b = preco_saida_total

                    if a == 0 or b == 0:
                        pnls_perpetuos.append(None)
                        continue

                    try:
                        pnl = ((b / a) - 1.0) * 100.0
                        if side == 'short':
                            pnl = -pnl
                    except ZeroDivisionError:
                        pnl = None
                    pnls_perpetuos.append(pnl)
                # Se não tem quantidade, calcular PnL simples com preço
                elif pd.notna(preco_atual):
                    try:
                        pnl = ((preco_atual / preco_entrada) - 1.0) * 100.0
                        if side == 'short':
                            pnl = -pnl
                    except ZeroDivisionError:
                        pnl = None
                    pnls_perpetuos.append(pnl)
                else:
                    pnls_perpetuos.append(None)

            df['pnl'] = pnls_perpetuos
        
        # Para o produto Alphacoins (ID 3476245316), EXC (ID 2150859854), HB (ID 2000449260) e LC (ID 2394004756), calcular PnL e adicionar alocação atual
        if produto_id in (3476245316, 2150859854, 2000449260, 2394004756):
            # OPTIMIZED: Vectorized PnL calculation
            if 'pnl' not in df.columns:
                df['_ativo_upper'] = df['ativo'].astype(str).str.strip().str.upper()
                df['pnl'] = None
                mask = (df['_ativo_upper'] != 'USDT') & df['preco_entrada'].notna() & df['preco_atual'].notna() & (df['preco_entrada'] != 0)
                df.loc[mask, 'pnl'] = ((df.loc[mask, 'preco_atual'] / df.loc[mask, 'preco_entrada']) - 1.0) * 100.0
                df.drop(columns=['_ativo_upper'], inplace=True)

            # OPTIMIZED: Batch load allocations in single query
            ativos = df['ativo'].tolist()
            aloc_map = _batch_load_allocations(repo, produto_id, ativos)
            df['_ativo_key'] = df['ativo'].astype(str).str.strip().str.upper()
            df['alocacao'] = df['_ativo_key'].map(aloc_map)
            df.drop(columns=['_ativo_key'], inplace=True)
    
    return df

def posicoes_fechadas(produto_id=None):
    """Retorna posições fechadas com atributos do produto"""
    repo = SQLiteRepo()

    # Detectar tipo de produto (com cache)
    tipo_spot, tipo_perpetuos = _get_product_type(repo, produto_id)

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

        # OPTIMIZED: Batch load stops in single query
        posicao_ids = df['id'].tolist()
        stops_map = _batch_load_stops(repo, posicao_ids)
        df['stop_atual'] = df['id'].map(stops_map)
        
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
                # Se tem quantidade, usar fórmula com notional
                if pd.notna(preco_entrada) and pd.notna(preco_saida) and pd.notna(quantidade) and quantidade != 0:
                    a = quantidade * preco_entrada
                    b = quantidade * preco_saida
                    if a != 0 and b != 0:
                        try:
                            pnl = ((b / a) - 1.0) * 100.0
                            if side == 'short':
                                pnl = -pnl
                        except ZeroDivisionError:
                            pnl = None
                    else:
                        pnl = None
                # Se não tem quantidade, calcular PnL simples com preço
                elif pd.notna(preco_entrada) and pd.notna(preco_saida) and preco_entrada != 0:
                    try:
                        pnl = ((preco_saida / preco_entrada) - 1.0) * 100.0
                        if side == 'short':
                            pnl = -pnl
                    except ZeroDivisionError:
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
    if produto_id in (3476245316, 2150859854, 2000449260, 2394004756):
        # OPTIMIZED: Batch load allocations in single query
        if not df.empty:
            ativos = df['ativo'].tolist()
            aloc_map = _batch_load_allocations(repo, produto_id, ativos)
            df['_ativo_key'] = df['ativo'].astype(str).str.strip().str.upper()
            df['alocacao'] = df['_ativo_key'].map(aloc_map)
            df.drop(columns=['_ativo_key'], inplace=True)

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

    # OPTIMIZED: Batch load prices in single API call
    precos_atuais_map = _batch_load_prices(df['coingecko_id'].tolist())
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
