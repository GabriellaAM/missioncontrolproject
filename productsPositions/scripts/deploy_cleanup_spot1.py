"""
Deploy migration: Limpa e corrige posições do Soros Spot 1.

Dados oficiais: CSV (FIFO buy/sell) + API Bitget (paginação completa).
- 17 fechadas do CSV (pareamento FIFO)
- 8 fechadas da API (abertas no fim do CSV, vendidas depois)
- 1 fechada da API (ACH, pós-CSV)
- 1 fechada da API (SOL 14/01, buy do CSV + sell da API)
- 2 abertas da API (VIRTUAL + HYPE)
Total: 27 fechadas + 2 abertas = 29 posições

4 posições do CSV se sobrepõem com a API (HYPE, ETH, MORPHO, MYRIA) →
usamos dados do CSV (mais precisos).

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
    print("[CLEANUP-SPOT1] Pulando: sem SUPABASE_DB_URL configurado")
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

# 17 fechadas do CSV (pareamento FIFO compra/venda)
CLOSED_CSV = [
    {'ativo': 'ENA',      'entrada': '2025-06-02', 'saida': '2025-06-06', 'px_ent': 0.31,       'px_saida': 0.2974,    'qty': 244.76},
    {'ativo': 'LQTY',     'entrada': '2025-06-13', 'saida': '2025-06-23', 'px_ent': 0.979,      'px_saida': 1.417,     'qty': 61.21},
    {'ativo': 'MORPHO',   'entrada': '2025-07-09', 'saida': '2025-07-17', 'px_ent': 1.4679,     'px_saida': 2.0477,    'qty': 36.74},
    {'ativo': 'ETH',      'entrada': '2025-06-06', 'saida': '2025-07-21', 'px_ent': 2496.82,    'px_saida': 3730.42,   'qty': 0.1199},
    {'ativo': 'FARTCOIN', 'entrada': '2025-06-03', 'saida': '2025-07-23', 'px_ent': 1.1058,     'px_saida': 1.53,      'qty': 67.75},
    {'ativo': 'HYPE',     'entrada': '2025-06-02', 'saida': '2025-07-31', 'px_ent': 35.49,      'px_saida': 41.75,     'qty': 2.1},
    {'ativo': 'ENA',      'entrada': '2025-07-23', 'saida': '2025-07-31', 'px_ent': 0.4596,     'px_saida': 0.5875,    'qty': 330.38},
    {'ativo': 'ENA',      'entrada': '2025-08-07', 'saida': '2025-08-11', 'px_ent': 0.6161,     'px_saida': 0.7897,    'qty': 147.95},
    {'ativo': 'RAY',      'entrada': '2025-07-21', 'saida': '2025-08-14', 'px_ent': 3.18,       'px_saida': 3.847,     'qty': 21.98},
    {'ativo': 'LINK',     'entrada': '2025-07-27', 'saida': '2025-08-18', 'px_ent': 19.087,     'px_saida': 24.554,    'qty': 7.8508},
    {'ativo': 'PENDLE',   'entrada': '2025-08-07', 'saida': '2025-08-19', 'px_ent': 4.388,      'px_saida': 5.213,     'qty': 20.771},
    {'ativo': 'MORPHO',   'entrada': '2025-07-24', 'saida': '2025-08-25', 'px_ent': 1.8819,     'px_saida': 2.3905,    'qty': 38.74},
    {'ativo': 'OMNI1',    'entrada': '2025-08-11', 'saida': '2025-08-25', 'px_ent': 4.471,      'px_saida': 3.322,     'qty': 15.63},
    {'ativo': 'HYPE',     'entrada': '2025-08-14', 'saida': '2025-09-13', 'px_ent': 45.57,      'px_saida': 55.73,     'qty': 1.52},
    {'ativo': 'ETH',      'entrada': '2025-08-07', 'saida': '2025-10-17', 'px_ent': 3809.58,    'px_saida': 3779.70,   'qty': 0.0476},
    {'ativo': 'MORPHO',   'entrada': '2025-09-01', 'saida': '2025-11-12', 'px_ent': 1.8992,     'px_saida': 2.0085,    'qty': 39.44},
    {'ativo': 'MYRIA',    'entrada': '2025-06-16', 'saida': '2025-11-14', 'px_ent': 0.001437,   'px_saida': 0.000224,  'qty': 17983.61},
]

# 8 fechadas da API (abertas no fim do CSV, vendidas depois — confirmadas pela API)
CLOSED_API = [
    {'ativo': 'VIRTUAL',  'entrada': '2025-06-05', 'saida': '2026-02-03', 'px_ent': 1.7265,     'px_saida': 0.6397,    'qty': 29.48},
    {'ativo': 'BTC',      'entrada': '2025-06-06', 'saida': '2026-02-03', 'px_ent': 104475.13,  'px_saida': 74822.56,  'qty': 0.008126},
    {'ativo': 'AAVE',     'entrada': '2025-06-13', 'saida': '2026-02-03', 'px_ent': 283.40,     'px_saida': 126.51,    'qty': 0.2114},
    {'ativo': 'LQTY',     'entrada': '2025-07-09', 'saida': '2026-02-03', 'px_ent': 1.229,      'px_saida': 0.318,     'qty': 26.54},
    {'ativo': 'FARTCOIN', 'entrada': '2025-08-12', 'saida': '2026-02-03', 'px_ent': 0.8807,     'px_saida': 0.2168,    'qty': 85.06},
    {'ativo': 'ENA',      'entrada': '2025-08-14', 'saida': '2026-02-03', 'px_ent': 0.7139,     'px_saida': 0.1363,    'qty': 97.4},
    {'ativo': 'DOGE',     'entrada': '2025-08-22', 'saida': '2026-02-03', 'px_ent': 0.2357,     'px_saida': 0.10557,   'qty': 635.7657},
    {'ativo': 'SOL',      'entrada': '2025-09-10', 'saida': '2026-02-03', 'px_ent': 224.35,     'px_saida': 98.92,     'qty': 1.046},
]

# 2 fechadas da API (pós-CSV)
CLOSED_API_POST = [
    {'ativo': 'SOL',      'entrada': '2026-01-14', 'saida': '2026-01-22', 'px_ent': 146.88,     'px_saida': 128.62,    'qty': 0.7052},
    {'ativo': 'ACH',      'entrada': '2026-01-22', 'saida': '2026-02-03', 'px_ent': 0.01221,    'px_saida': 0.00791,   'qty': 6135.0},
]

ALL_CLOSED = CLOSED_CSV + CLOSED_API + CLOSED_API_POST
for p in ALL_CLOSED:
    p['cg'] = CG[p['ativo']]
    p['exch'] = p['ativo'] + 'USDT'
    if p['ativo'] == 'OMNI1':
        p['exch'] = 'OMNI1USDT'

# 2 abertas (API Bitget)
OPEN_POSITIONS = [
    {'ativo': 'VIRTUAL', 'entrada': '2026-03-03', 'px_ent': 0.7395, 'qty': 146.32, 'exch': 'VIRTUALUSDT', 'cg': 'virtual-protocol'},
    {'ativo': 'HYPE',    'entrada': '2025-09-22', 'px_ent': 47.86,  'qty': 1.75,   'exch': 'HYPEUSDT',    'cg': 'hyperliquid'},
]

EXPECTED_TOTAL = len(ALL_CLOSED) + len(OPEN_POSITIONS)  # 27 + 2 = 29

# ========================================================================

try:
    from storage.sqlite_repo import connect_pg
    conn = connect_pg(db_url)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM produtos WHERE nome = 'Soros Spot 1'")
    row = cursor.fetchone()
    if not row:
        print("[CLEANUP-SPOT1] Produto 'Soros Spot 1' não encontrado.")
        conn.close()
        sys.exit(0)
    prod_id = row[0] if isinstance(row, (list, tuple)) else row['id']
    print(f"[CLEANUP-SPOT1] Produto Soros Spot 1: id={prod_id}")

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
            print(f"[CLEANUP-SPOT1] Já tem {current_count} posições com coingecko_id. Já corrigido.")
            conn.close()
            sys.exit(0)

    print(f"[CLEANUP-SPOT1] Posições atuais: {current_count}, esperado: {EXPECTED_TOTAL}")

    # STEP 1: Clean up turma data
    cursor.execute("SELECT id FROM turmas WHERE produto_id = %s", (prod_id,))
    turma_rows = cursor.fetchall()
    for tr in turma_rows:
        tid = tr[0] if isinstance(tr, (list, tuple)) else tr['id']
        cursor.execute("DELETE FROM carteira_turma WHERE turma_id = %s", (tid,))
        cursor.execute("DELETE FROM trades_turma WHERE turma_id = %s", (tid,))
    cursor.execute("DELETE FROM turmas WHERE produto_id = %s", (prod_id,))
    print(f"[CLEANUP-SPOT1] Removeu {len(turma_rows)} turma(s)")

    # STEP 2: Delete ALL existing positions
    cursor.execute("""
        DELETE FROM posicao_atributos_produto
        WHERE posicao_id IN (SELECT id FROM posicoes WHERE produto_id = %s)
    """, (prod_id,))
    cursor.execute("DELETE FROM posicoes WHERE produto_id = %s", (prod_id,))
    print(f"[CLEANUP-SPOT1] Removeu {current_count} posições antigas")

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
    print(f"[CLEANUP-SPOT1] Inseriu {inserted} posições fechadas")

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
        print(f"[CLEANUP-SPOT1] Inseriu aberta: {p['ativo']} ({p['entrada']})")

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM posicoes WHERE produto_id = %s", (prod_id,))
    r = cursor.fetchone()
    final_count = r[0] if isinstance(r, (list, tuple)) else list(r.values())[0]
    print(f"[CLEANUP-SPOT1] Concluído. Posições finais: {final_count} (esperado: {EXPECTED_TOTAL})")

    conn.close()

except Exception as e:
    print(f"[CLEANUP-SPOT1] Erro: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
