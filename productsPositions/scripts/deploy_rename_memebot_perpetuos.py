"""
Deploy migration: Renomeia o produto "Memebot Perpétuos 1" para "Memebot Perpétuos".

Idempotente: verifica se o nome já foi alterado antes de executar.

Uso:
  python -m productsPositions.scripts.deploy_rename_memebot_perpetuos [--dry-run]
"""
import os
import sys
import re

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..'))
sys.path.insert(0, _here)

from dotenv import load_dotenv
load_dotenv(os.path.join(_here, '..', '..', '.env'))

db_url = os.getenv('SUPABASE_DB_URL', '').strip()
if not db_url or db_url.startswith('sqlite://'):
    print("[RENAME-MEMEBOT] Pulando: sem SUPABASE_DB_URL configurado")
    sys.exit(0)

DRY_RUN = '--dry-run' in sys.argv
TAG = "[RENAME-MEMEBOT]"

try:
    from storage.sqlite_repo import connect_pg
    conn = connect_pg(db_url)
    cursor = conn.cursor()

    NOME_ANTIGO = re.compile(r'^Memebot Perp[eé]tuos\s*1$')
    NOME_NOVO = 'Memebot Perpétuos'

    cursor.execute("SELECT id, nome FROM produtos WHERE nome LIKE 'Memebot Perp%%'")
    rows = cursor.fetchall()

    if not rows:
        print(f"{TAG} Nenhum produto 'Memebot Perp...' encontrado.")
        conn.close()
        sys.exit(0)

    print(f"{TAG} Produtos encontrados:")
    for r in rows:
        pid = r[0] if isinstance(r, (list, tuple)) else r['id']
        pnome = r[1] if isinstance(r, (list, tuple)) else r['nome']
        print(f"  id={pid}, nome='{pnome}'")

    prod_row = None
    for r in rows:
        pnome = (r[1] if isinstance(r, (list, tuple)) else r['nome']).strip()
        if NOME_ANTIGO.match(pnome):
            prod_row = r
            break

    if not prod_row:
        for r in rows:
            pnome = (r[1] if isinstance(r, (list, tuple)) else r['nome']).strip()
            if pnome == NOME_NOVO:
                print(f"{TAG} Produto já renomeado para '{pnome}'. Nada a fazer.")
                conn.close()
                sys.exit(0)

    if not prod_row:
        print(f"{TAG} Produto 'Memebot Perpétuos 1' não encontrado entre os resultados.")
        conn.close()
        sys.exit(0)

    prod_id = prod_row[0] if isinstance(prod_row, (list, tuple)) else prod_row['id']
    prod_nome_atual = (prod_row[1] if isinstance(prod_row, (list, tuple)) else prod_row['nome']).strip()

    if prod_nome_atual != NOME_NOVO:
        print(f"{TAG} Renomeando produto: '{prod_nome_atual}' -> '{NOME_NOVO}'")
        if not DRY_RUN:
            cursor.execute("UPDATE produtos SET nome = %s WHERE id = %s", (NOME_NOVO, prod_id))
    else:
        print(f"{TAG} Produto já com nome correto: '{NOME_NOVO}'")

    if DRY_RUN:
        print(f"\n{TAG} DRY-RUN: nenhuma alteração salva. Remova --dry-run para aplicar.")
        conn.rollback()
    else:
        conn.commit()
        print(f"\n{TAG} Alteração salva com sucesso!")

    conn.close()

except Exception as e:
    print(f"{TAG} Erro: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
