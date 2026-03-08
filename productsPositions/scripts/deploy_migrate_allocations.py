"""
Migração de alocações (CSVs → Supabase/PostgreSQL).

Executado automaticamente antes do start do app no Render.
Lê os CSVs de alocação commitados no repositório e sincroniza as tabelas
`posicoes` e `alocacoes` no Supabase para os 4 produtos de alocação livre.

Características:
  - Conexão direta ao PostgreSQL via SUPABASE_DB_URL (sem fallback SQLite)
  - Fingerprint SHA-256 dos CSVs → pula migração se nada mudou
  - Idempotente: DELETE + INSERT dentro de transação por produto
  - Batch inserts (execute_batch) para performance
  - Sempre exit 0 — erros são logados mas não bloqueiam o deploy
"""

import os
import sys
import hashlib
import random
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT.parent))

ALLOCATIONS_DIR = _ROOT / "data" / "allocations"

PRODUTO_CONFIG = {
    'EXC': {'id': 2150859854, 'csv': 'EXC.csv', 'nome': 'Exponential Coins (Principal)'},
    'HB':  {'id': 2000449260, 'csv': 'HB.csv',  'nome': 'Exponential Coins (High Beta)'},
    'LC':  {'id': 2394004756, 'csv': 'LC.csv',  'nome': 'Exponential Coins (Low Caps)'},
    'AC':  {'id': 3476245316, 'csv': 'AC.csv',  'nome': 'Alphacoins'},
}


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _find_csv(csv_name: str):
    """Procura CSV em data/allocations/ e depois na raiz de productsPositions/."""
    for base in [ALLOCATIONS_DIR, _ROOT]:
        p = base / csv_name
        if p.exists():
            return p
    return None


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _combined_fingerprint() -> str:
    """Hash combinado de todos os CSVs encontrados (detecta qualquer mudança)."""
    parts = []
    for key in sorted(PRODUTO_CONFIG):
        csv_path = _find_csv(PRODUTO_CONFIG[key]['csv'])
        if csv_path:
            parts.append(f"{key}:{_file_hash(csv_path)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _gerar_id() -> int:
    return random.randint(1_000_000_000, 9_999_999_999)


def _parse_pct(x) -> float:
    if isinstance(x, str) and "%" in x:
        return float(x.strip("%").replace(",", ".").strip()) / 100.0
    try:
        v = float(x)
        return v / 100.0 if v > 1.0 else v
    except (ValueError, TypeError):
        return 0.0


def _contiguous_periods(dates_sorted: list):
    """Agrupa datas YYYY-MM-DD consecutivas em períodos (inicio, fim)."""
    if not dates_sorted:
        return []
    periods = []
    start = end = dates_sorted[0]
    for d in dates_sorted[1:]:
        a = datetime.strptime(end[:10], "%Y-%m-%d")
        b = datetime.strptime(d[:10], "%Y-%m-%d")
        if (b - a).days <= 1:
            end = d
        else:
            periods.append((start, end))
            start = end = d
    periods.append((start, end))
    return periods


# ---------------------------------------------------------------------------
# CSV → DataFrame
# ---------------------------------------------------------------------------

def _load_csv(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(
        csv_path, index_col="Data", parse_dates=["Data"], date_format="%Y-%m-%d"
    )
    df.columns = [c.strip().upper() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].apply(_parse_pct)
    return df.fillna(0.0)


# ---------------------------------------------------------------------------
# Controle de fingerprint (tabela migration_state)
# ---------------------------------------------------------------------------

def _ensure_migration_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS migration_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
    conn.commit()


def _get_stored_fingerprint(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT value FROM migration_state WHERE key = 'allocations_fingerprint'"
        )
        row = cur.fetchone()
    return row[0] if row else None


def _store_fingerprint(conn, fingerprint: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO migration_state (key, value, updated_at)
            VALUES ('allocations_fingerprint', %s, NOW())
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
            """,
            (fingerprint,),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Garantir integridade referencial
# ---------------------------------------------------------------------------

def _ensure_ativos(conn, ativos_set: set):
    """Insere na tabela `ativos` qualquer ticker referenciado que ainda não exista."""
    with conn.cursor() as cur:
        cur.execute("SELECT nome FROM ativos")
        existing = {row[0] for row in cur.fetchall()}

    novos = ativos_set - existing
    if not novos:
        return

    with conn.cursor() as cur:
        for ativo in sorted(novos):
            cur.execute(
                "INSERT INTO ativos (nome) VALUES (%s) ON CONFLICT DO NOTHING",
                (ativo,),
            )
    conn.commit()
    print(f"    [ativos] {len(novos)} novo(s): {', '.join(sorted(novos))}")


# ---------------------------------------------------------------------------
# Sincronização de um produto
# ---------------------------------------------------------------------------

def _sync_product(conn, produto_id: int, csv_path: Path):
    """
    Sincroniza um produto de alocação livre:
      1. Carrega CSV
      2. Garante que todos os ativos existam
      3. Remove alocações + stops + posições antigas
      4. Recria posições (uma por período contíguo de cada ativo)
      5. Recria alocações (uma por dia×ativo com percentual > 0)

    Retorna (n_posicoes, n_alocacoes).
    """
    import psycopg2.extras

    df = _load_csv(csv_path)
    if df.empty:
        print(f"    CSV vazio: {csv_path.name}")
        return 0, 0

    # Extrair tuplas (data_str, ativo, pct) com pct > 0
    rows = []
    for data in df.index:
        data_str = data.strftime("%Y-%m-%d")
        for ativo in df.columns:
            pct = float(df.loc[data, ativo])
            if pct > 0:
                rows.append((data_str, ativo, pct))

    if not rows:
        print(f"    Nenhuma alocação > 0 no CSV.")
        return 0, 0

    # Garantir integridade referencial
    ativos_usados = {r[1] for r in rows}
    _ensure_ativos(conn, ativos_usados)

    last_date_str = df.index.max().strftime("%Y-%m-%d")

    # Agrupar por ativo → lista de (data, pct)
    ativo_dates = defaultdict(list)
    for data_str, ativo, pct in rows:
        ativo_dates[ativo].append((data_str, pct))

    # Calcular períodos contíguos por ativo
    ativo_periods = {}
    for ativo, list_dp in ativo_dates.items():
        list_dp.sort()
        dates_only = [d for d, _ in list_dp]
        periods = _contiguous_periods(dates_only)
        period_data = []
        for start, end in periods:
            in_period = [(d, p) for d, p in list_dp if start <= d <= end]
            period_data.append((start, end, in_period))
        ativo_periods[ativo] = period_data

    # ── Fase 1: Remover dados antigos (transacional) ──
    with conn.cursor() as cur:
        cur.execute("DELETE FROM alocacoes WHERE produto_id = %s", (produto_id,))
        n_del_alloc = cur.rowcount
        cur.execute(
            "DELETE FROM stops WHERE posicao_id IN "
            "(SELECT id FROM posicoes WHERE produto_id = %s)",
            (produto_id,),
        )
        cur.execute("DELETE FROM posicoes WHERE produto_id = %s", (produto_id,))
        n_del_pos = cur.rowcount
    conn.commit()
    print(f"    Removidas {n_del_alloc} alocações e {n_del_pos} posições antigas.")

    # ── Fase 2: Montar linhas de posição e alocação ──
    seen_ids = set()
    pos_rows = []
    aloc_rows = []

    for ativo, periods in ativo_periods.items():
        for start, end, list_dp in periods:
            pos_id = _gerar_id()
            while pos_id in seen_ids:
                pos_id = _gerar_id()
            seen_ids.add(pos_id)

            status = "closed" if end < last_date_str else "open"
            data_saida = end if status == "closed" else None
            pos_rows.append(
                (pos_id, produto_id, ativo, "long", start, 0, data_saida, status)
            )

            for data_str, pct in list_dp:
                aloc_id = _gerar_id()
                while aloc_id in seen_ids:
                    aloc_id = _gerar_id()
                seen_ids.add(aloc_id)
                aloc_rows.append(
                    (aloc_id, produto_id, pos_id, round(pct * 100, 2), None, data_str, "active")
                )

    # ── Fase 3: Inserir em batch ──
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """INSERT INTO posicoes
                   (id, produto_id, ativo, side, data_entrada, preco_entrada, data_saida, status)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            pos_rows,
            page_size=500,
        )
    conn.commit()

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(
            cur,
            """INSERT INTO alocacoes
                   (id, produto_id, posicao_id, percentual, valor_usd, data, status)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            aloc_rows,
            page_size=1000,
        )
    conn.commit()

    print(f"    Criadas {len(pos_rows)} posições e {len(aloc_rows)} alocações.")
    return len(pos_rows), len(aloc_rows)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main():
    db_url = os.getenv("SUPABASE_DB_URL", "").strip()
    if not db_url:
        print("[ALLOC-MIGRATION] SUPABASE_DB_URL não configurado. Pulando migração.")
        return

    import psycopg2

    print()
    print("=" * 60)
    print("  MIGRAÇÃO DE ALOCAÇÕES → Supabase (PostgreSQL)")
    print("=" * 60)
    t0 = time.time()

    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"[ALLOC-MIGRATION] Erro de conexão: {e}")
        return

    try:
        _ensure_migration_table(conn)

        current_fp = _combined_fingerprint()
        stored_fp = _get_stored_fingerprint(conn)

        if current_fp == stored_fp:
            elapsed = time.time() - t0
            print(
                f"\n  CSVs inalterados desde a última migração — nada a fazer. ({elapsed:.1f}s)"
            )
            print("=" * 60)
            conn.close()
            return

        print(f"\n  Fingerprint anterior: {stored_fp or '(nenhum)'}")
        print(f"  Fingerprint atual:    {current_fp[:16]}...")

        total_pos = 0
        total_alloc = 0

        for key in ["EXC", "HB", "LC", "AC"]:
            config = PRODUTO_CONFIG[key]
            csv_path = _find_csv(config["csv"])

            if not csv_path:
                print(f"\n  [{key}] CSV não encontrado — pulando.")
                continue

            n_lines = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
            print(f"\n  [{key}] {config['nome']} (id={config['id']})")
            print(f"    CSV: {csv_path.name} — {n_lines} dias, fonte: {csv_path.parent.name}/")

            try:
                n_pos, n_alloc = _sync_product(conn, config["id"], csv_path)
                total_pos += n_pos
                total_alloc += n_alloc
            except Exception as e:
                print(f"    ERRO: {e}")
                conn.rollback()

        _store_fingerprint(conn, current_fp)

        elapsed = time.time() - t0
        print()
        print(f"  CONCLUÍDO: {total_pos} posições, {total_alloc} alocações — {elapsed:.1f}s")
        print("=" * 60)
        print()

    except Exception as e:
        print(f"[ALLOC-MIGRATION] Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
