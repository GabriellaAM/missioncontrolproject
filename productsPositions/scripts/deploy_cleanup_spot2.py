"""
Deploy migration: Limpa e corrige posições do Soros Spot 2.

Dados oficiais: CSV (FIFO buy/sell) + API Bitget (paginação completa).
- 17 fechadas do CSV (pareamento FIFO)
- 8 fechadas da API (abertas no fim do CSV, vendidas depois)
- 1 fechada da API (ACH, pós-CSV)
- 1 fechada da API (SOL 14/01, buy do CSV + sell da API)
- 2 abertas da API (VIRTUAL + HYPE)
Total: 27 fechadas + 2 abertas = 29 posições

Idempotente: verifica contagem + coingecko_id.
"""
import os, sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))
sys.path.insert(0, _here)

from dotenv import load_dotenv
load_dotenv(os.path.join(_here, '..', '..', '.env'))

db_url = os.getenv('SUPABASE_DB_URL', '').strip()
if not db_url or db_url.startswith('sqlite://'):
    print("[CLEANUP-SPOT2] Pulando: sem SUPABASE_DB_URL configurado")
    sys.exit(0)

# ========================================================================
# Dados oficiais: CSV (FIFO) + Bitget API
# ========================================================================

CG = {
    'ENA': 'ethena', 'LQTY': 'liquity', 'MORPHO': 'morpho', 'ETH': 'ethereum',
    'FARTCOIN': 'fartcoin', 'HYPE': 'hyperliquid', 'RAY': 'raydium', 'LINK': 'chainlink',
    'PENDLE': 'pendle', 'OMNI1': 'omni-network', 'MYRIA': 'myria', 'SOL': 'solana',
    'ACH': 'alchemy-pay', 'VIRTUAL': 'virtual-protocol', 'BTC': 'bitcoin',
    'AAVE': 'aave', 'DOGE': 'dogecoin',
}

# 17 fechadas do CSV (pareamento FIFO — Spot 2 com dados de Pasta2.csv)
CLOSED_CSV = [
    {'ativo': 'ENA',      'entrada': '2025-06-02', 'saida': '2025-06-06', 'px_ent': 0.2976,     'px_saida': 0.2976,    'qty': 243.57},
    {'ativo': 'LQTY',     'entrada': '2025-06-13', 'saida': '2025-06-23', 'px_ent': 0.9807,     'px_saida': 1.4248,    'qty': 61.18},
    {'ativo': 'MORPHO',   'entrada': '2025-07-09', 'saida': '2025-07-17', 'px_ent': 1.4614,     'px_saida': 2.0529,    'qty': 36.88},
    {'ativo': 'ETH',      'entrada': '2025-06-06', 'saida': '2025-07-21', 'px_ent': 2497.46,    'px_saida': 3733.99,   'qty': 0.1199},
    {'ativo': 'FARTCOIN', 'entrada': '2025-06-03', 'saida': '2025-07-23', 'px_ent': 1.1039,     'px_saida': 1.5303,    'qty': 66.58},
    {'ativo': 'HYPE',     'entrada': '2025-06-02', 'saida': '2025-07-31', 'px_ent': 35.37,      'px_saida': 42.02,     'qty': 2.12},
    {'ativo': 'ENA',      'entrada': '2025-07-23', 'saida': '2025-07-31', 'px_ent': 0.4596,     'px_saida': 0.5904,    'qty': 329.07},
    {'ativo': 'ENA',      'entrada': '2025-08-07', 'saida': '2025-08-11', 'px_ent': 0.6158,     'px_saida': 0.7879,    'qty': 139.98},
    {'ativo': 'RAY',      'entrada': '2025-07-21', 'saida': '2025-08-14', 'px_ent': 3.18,       'px_saida': 3.867,     'qty': 21.98},
    {'ativo': 'LINK',     'entrada': '2025-07-27', 'saida': '2025-08-18', 'px_ent': 19.116,     'px_saida': 24.524,    'qty': 7.8389},
    {'ativo': 'PENDLE',   'entrada': '2025-08-07', 'saida': '2025-08-19', 'px_ent': 4.387,      'px_saida': 5.217,     'qty': 20.778},
    {'ativo': 'MORPHO',   'entrada': '2025-07-24', 'saida': '2025-08-25', 'px_ent': 1.8784,     'px_saida': 2.3911,    'qty': 38.82},
    {'ativo': 'OMNI1',    'entrada': '2025-08-11', 'saida': '2025-08-25', 'px_ent': 4.489,      'px_saida': 3.33,      'qty': 15.59},
    {'ativo': 'HYPE',     'entrada': '2025-08-14', 'saida': '2025-09-13', 'px_ent': 45.88,      'px_saida': 55.73,     'qty': 1.51},
    {'ativo': 'ETH',      'entrada': '2025-08-07', 'saida': '2025-10-17', 'px_ent': 3807.86,    'px_saida': 3796.37,   'qty': 0.0476},
    {'ativo': 'MORPHO',   'entrada': '2025-09-01', 'saida': '2025-11-12', 'px_ent': 1.9021,     'px_saida': 2.0033,    'qty': 39.38},
    {'ativo': 'MYRIA',    'entrada': '2025-06-16', 'saida': '2025-11-14', 'px_ent': 0.001435,   'px_saida': 0.000224,  'qty': 33532.46},
]

# 8 fechadas da API (abertas no fim do CSV, vendidas depois — Spot 2 API)
CLOSED_API = [
    {'ativo': 'VIRTUAL',  'entrada': '2025-06-05', 'saida': '2026-02-03', 'px_ent': 1.7261,     'px_saida': 0.6431,    'qty': 29.50},
    {'ativo': 'BTC',      'entrada': '2025-06-06', 'saida': '2026-02-03', 'px_ent': 104540.60,  'px_saida': 74910.03,  'qty': 0.008121},
    {'ativo': 'AAVE',     'entrada': '2025-06-13', 'saida': '2026-02-03', 'px_ent': 283.41,     'px_saida': 126.62,    'qty': 0.2114},
    {'ativo': 'LQTY',     'entrada': '2025-07-09', 'saida': '2026-02-03', 'px_ent': 1.231,      'px_saida': 0.319,     'qty': 27.84},
    {'ativo': 'FARTCOIN', 'entrada': '2025-08-12', 'saida': '2026-02-03', 'px_ent': 0.8758,     'px_saida': 0.2189,    'qty': 85.54},
    {'ativo': 'ENA',      'entrada': '2025-08-14', 'saida': '2026-02-03', 'px_ent': 0.7205,     'px_saida': 0.1365,    'qty': 97.07},
    {'ativo': 'DOGE',     'entrada': '2025-08-22', 'saida': '2026-02-03', 'px_ent': 0.23483,    'px_saida': 0.10568,   'qty': 638.1211},
    {'ativo': 'SOL',      'entrada': '2025-09-10', 'saida': '2026-02-03', 'px_ent': 224.38,     'px_saida': 99.18,     'qty': 1.0417},
]

# 2 fechadas da API (pós-CSV)
CLOSED_API_POST = [
    {'ativo': 'SOL',      'entrada': '2026-01-14', 'saida': '2026-01-22', 'px_ent': 147.15,     'px_saida': 128.32,    'qty': 0.6788},
    {'ativo': 'ACH',      'entrada': '2026-01-22', 'saida': '2026-02-03', 'px_ent': 0.01229,    'px_saida': 0.00788,   'qty': 6093.0},
]

ALL_CLOSED = CLOSED_CSV + CLOSED_API + CLOSED_API_POST
for p in ALL_CLOSED:
    p['cg'] = CG[p['ativo']]
    p['exch'] = p['ativo'] + 'USDT'
    if p['ativo'] == 'OMNI1':
        p['exch'] = 'OMNI1USDT'

# 2 abertas (API Bitget)
OPEN_POSITIONS = [
    {'ativo': 'VIRTUAL', 'entrada': '2026-03-03', 'px_ent': 0.7359, 'qty': 147.24, 'exch': 'VIRTUALUSDT', 'cg': 'virtual-protocol'},
    {'ativo': 'HYPE',    'entrada': '2025-09-22', 'px_ent': 47.85,  'qty': 1.80,   'exch': 'HYPEUSDT',    'cg': 'hyperliquid'},
]

EXPECTED_TOTAL = len(ALL_CLOSED) + len(OPEN_POSITIONS)  # 27 + 2 = 29

# ========================================================================

try:
    from storage.sqlite_repo import connect_pg
    conn = connect_pg(db_url)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM produtos WHERE nome = 'Soros Spot 2'")
    row = cursor.fetchone()
    if not row:
        print("[CLEANUP-SPOT2] Produto 'Soros Spot 2' não encontrado.")
        conn.close()
        sys.exit(0)
    prod_id = row[0] if isinstance(row, (list, tuple)) else row['id']
    print(f"[CLEANUP-SPOT2] Produto Soros Spot 2: id={prod_id}")

    cursor.execute("SELECT COUNT(*) FROM posicoes WHERE produto_id = %s", (prod_id,))
    r = cursor.fetchone()
    current_count = r[0] if isinstance(r, (list, tuple)) else list(r.values())[0]

    if current_count == EXPECTED_TOTAL:
        cursor.execute("""
            SELECT COUNT(*) FROM posicoes
            WHERE produto_id = %s AND coingecko_id IS NOT NULL AND coingecko_id != ''
        """, (prod_id,))
        cg_r = cursor.fetchone()
        cg_count = cg_r[0] if isinstance(cg_r, (list, tuple)) else list(cg_r.values())[0]
        if cg_count == EXPECTED_TOTAL:
            print(f"[CLEANUP-SPOT2] Já tem {current_count} posições com coingecko_id. Já corrigido.")
            conn.close()
            sys.exit(0)

    print(f"[CLEANUP-SPOT2] Posições atuais: {current_count}, esperado: {EXPECTED_TOTAL}")

    # STEP 1: Clean up turma data
    cursor.execute("SELECT id FROM turmas WHERE produto_id = %s", (prod_id,))
    turma_rows = cursor.fetchall()
    for tr in turma_rows:
        tid = tr[0] if isinstance(tr, (list, tuple)) else tr['id']
        cursor.execute("DELETE FROM carteira_turma WHERE turma_id = %s", (tid,))
        cursor.execute("DELETE FROM trades_turma WHERE turma_id = %s", (tid,))
    cursor.execute("DELETE FROM turmas WHERE produto_id = %s", (prod_id,))
    print(f"[CLEANUP-SPOT2] Removeu {len(turma_rows)} turma(s)")

    # STEP 2: Delete ALL existing positions
    cursor.execute("""
        DELETE FROM posicao_atributos_produto
        WHERE posicao_id IN (SELECT id FROM posicoes WHERE produto_id = %s)
    """, (prod_id,))
    cursor.execute("DELETE FROM posicoes WHERE produto_id = %s", (prod_id,))
    print(f"[CLEANUP-SPOT2] Removeu {current_count} posições antigas")

    # STEP 3: Insert CLOSED positions + quantidade
    inserted = 0
    for pos in ALL_CLOSED:
        cursor.execute("""
            INSERT INTO posicoes (produto_id, ativo, coingecko_id, side, data_entrada, preco_entrada,
                                  data_saida, preco_saida, status, exchange_symbol)
            VALUES (%s, %s, %s, 'long', %s, %s, %s, %s, 'closed', %s)
            RETURNING id
        """, (prod_id, pos['ativo'], pos['cg'], pos['entrada'], pos['px_ent'],
              pos['saida'], pos['px_saida'], pos['exch']))
        new_row = cursor.fetchone()
        new_id = new_row[0] if isinstance(new_row, (list, tuple)) else new_row['id']
        cursor.execute("""
            INSERT INTO posicao_atributos_produto (posicao_id, produto_id, quantidade)
            VALUES (%s, %s, %s)
        """, (new_id, prod_id, pos['qty']))
        inserted += 1
    print(f"[CLEANUP-SPOT2] Inseriu {inserted} posições fechadas")

    # STEP 4: Insert OPEN positions + quantidade
    for p in OPEN_POSITIONS:
        cursor.execute("""
            INSERT INTO posicoes (produto_id, ativo, coingecko_id, side, data_entrada, preco_entrada,
                                  status, exchange_symbol)
            VALUES (%s, %s, %s, 'long', %s, %s, 'open', %s)
            RETURNING id
        """, (prod_id, p['ativo'], p['cg'], p['entrada'], p['px_ent'], p['exch']))
        new_row = cursor.fetchone()
        new_id = new_row[0] if isinstance(new_row, (list, tuple)) else new_row['id']
        cursor.execute("""
            INSERT INTO posicao_atributos_produto (posicao_id, produto_id, quantidade)
            VALUES (%s, %s, %s)
        """, (new_id, prod_id, p['qty']))
        print(f"[CLEANUP-SPOT2] Inseriu aberta: {p['ativo']} ({p['entrada']})")

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM posicoes WHERE produto_id = %s", (prod_id,))
    r = cursor.fetchone()
    final_count = r[0] if isinstance(r, (list, tuple)) else list(r.values())[0]
    print(f"[CLEANUP-SPOT2] Concluído. Posições finais: {final_count} (esperado: {EXPECTED_TOTAL})")

    conn.close()

except Exception as e:
    print(f"[CLEANUP-SPOT2] Erro: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
