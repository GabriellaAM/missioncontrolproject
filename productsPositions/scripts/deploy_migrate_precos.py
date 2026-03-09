"""
Migração de preços históricos (CSV compactado → Supabase/PostgreSQL).

Executado automaticamente antes do start do app no Render.
Importa o arquivo precos_diarios_seed.csv.gz para a tabela precos_diarios,
garantindo que a nuvem tenha a mesma base de preços que o ambiente local.

Características:
  - ON CONFLICT DO UPDATE: sobrescreve preços existentes com os do seed
  - Preserva preços que existem APENAS no Supabase (datas mais recentes)
  - Batch inserts (execute_batch) para performance
  - Fingerprint do CSV → pula migração se nada mudou
  - Sempre exit 0 — erros são logados mas não bloqueiam o deploy
"""

import os
import sys
import gzip
import csv
import hashlib
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent

SEED_PATH = _ROOT / "data" / "precos_diarios_seed.csv.gz"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    db_url = os.getenv("SUPABASE_DB_URL", "").strip()
    if not db_url:
        print("[PRECOS-MIGRATION] SUPABASE_DB_URL não configurado. Pulando.")
        return

    if not SEED_PATH.exists():
        print(f"[PRECOS-MIGRATION] Seed não encontrado: {SEED_PATH.name}. Pulando.")
        return

    import psycopg2
    import psycopg2.extras

    print()
    print("=" * 60)
    print("  MIGRAÇÃO DE PREÇOS HISTÓRICOS → Supabase")
    print("=" * 60)
    t0 = time.time()

    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"[PRECOS-MIGRATION] Erro de conexão: {e}")
        return

    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS migration_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        conn.commit()

        current_fp = _file_hash(SEED_PATH)
        cur.execute("SELECT value FROM migration_state WHERE key = 'precos_seed_fingerprint'")
        row = cur.fetchone()
        stored_fp = row[0] if row else None

        if current_fp == stored_fp:
            elapsed = time.time() - t0
            print(f"\n  Seed inalterado desde a última migração — nada a fazer. ({elapsed:.1f}s)")
            print("=" * 60)
            conn.close()
            return

        print(f"\n  Fingerprint anterior: {stored_fp or '(nenhum)'}")
        print(f"  Fingerprint atual:    {current_fp[:16]}...")
        print(f"  Arquivo: {SEED_PATH.name} ({SEED_PATH.stat().st_size / (1024*1024):.1f} MB)")

        rows = []
        with gzip.open(str(SEED_PATH), 'rt', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append((
                    r['coingecko_id'],
                    r['data'],
                    float(r['preco']),
                    r.get('fonte', 'seed'),
                ))

        print(f"  Registros no seed: {len(rows)}")

        BATCH = 5000
        total_inserted = 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            psycopg2.extras.execute_batch(
                cur,
                """INSERT INTO precos_diarios (coingecko_id, data, preco, fonte)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (coingecko_id, data)
                   DO UPDATE SET preco = EXCLUDED.preco, fonte = EXCLUDED.fonte""",
                batch,
                page_size=1000,
            )
            conn.commit()
            total_inserted += len(batch)
            if (i // BATCH) % 10 == 0:
                print(f"    Progresso: {total_inserted}/{len(rows)}...", flush=True)

        cur.execute("""
            INSERT INTO migration_state (key, value, updated_at)
            VALUES ('precos_seed_fingerprint', %s, NOW())
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
        """, (current_fp,))
        conn.commit()

        elapsed = time.time() - t0
        print(f"\n  CONCLUÍDO: {total_inserted} registros importados — {elapsed:.1f}s")
        print("=" * 60)
        print()

    except Exception as e:
        print(f"[PRECOS-MIGRATION] Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
