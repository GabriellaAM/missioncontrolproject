"""
Cache de preços do BTC para o benchmark.
Funciona com SQLite (dev) ou Supabase/PostgreSQL (prod) conforme USE_SUPABASE.
Tabela Supabase: price_history_cache (asset, data, preco). Tabela SQLite: btc_price_history_cache (data, preco).
"""
import os
import logging
from datetime import date
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Carregar .env
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_root / ".env")
except ImportError:
    pass

ASSET_BTC = "BTC"


class BTCCacheService:
    """
    Cache de preços BTC. Detecta ambiente via USE_SUPABASE:
    - True: Supabase (tabela price_history_cache, asset=data, data=date, preco)
    - False: SQLite (tabela btc_price_history_cache, data, preco)
    """

    def __init__(self, repo=None, db_url: Optional[str] = None):
        if repo is not None:
            self._repo = repo
            self._db_url = getattr(repo, "db_url", None) or ""
        else:
            self._repo = None
            self._db_url = (db_url or os.getenv("SUPABASE_DB_URL") or "").strip()

        use_supabase_env = os.environ.get("USE_SUPABASE", "").strip().lower() in ("1", "true", "yes")
        self._use_supabase = use_supabase_env or (
            bool(self._db_url) and not self._db_url.strip().lower().startswith("sqlite://")
        )
        if self._repo is None and self._db_url:
            from storage.sqlite_repo import connect_pg
            self._connect = lambda: connect_pg(self._db_url)
        elif self._repo is not None:
            self._connect = None
        else:
            from storage.sqlite_repo import get_repo
            self._repo = get_repo()
            self._connect = None

    def _get_connection(self):
        if self._repo is not None:
            return self._repo.connection()
        from contextlib import contextmanager
        @contextmanager
        def _conn():
            conn = self._connect()
            try:
                yield conn
                if hasattr(conn, "commit"):
                    conn.commit()
            finally:
                if hasattr(conn, "close"):
                    conn.close()
        return _conn()

    def _ensure_table(self, conn) -> None:
        """
        Cria a tabela btc_price_history_cache no SQLite se não existir.
        Em Supabase (price_history_cache) a tabela é assumida existente; não cria.
        """
        if not getattr(conn, "_is_sqlite", False):
            return
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS btc_price_history_cache (
                   data TEXT PRIMARY KEY,
                   preco REAL NOT NULL
               )"""
        )
        if hasattr(conn, "commit"):
            conn.commit()

    def buscar_cache(self, data_inicio: date) -> List[Dict[str, Any]]:
        """
        Retorna lista de {data: str (YYYY-MM-DD), preco: float} do cache
        com data >= data_inicio, ordenada por data.
        """
        data_inicio_str = data_inicio.isoformat()
        result: List[Dict[str, Any]] = []
        with self._get_connection() as conn:
            self._ensure_table(conn)
            is_sqlite = getattr(conn, "_is_sqlite", False)
            cur = conn.cursor()
            if is_sqlite:
                cur.execute(
                    "SELECT data, preco FROM btc_price_history_cache WHERE data >= ? ORDER BY data",
                    (data_inicio_str,),
                )
                rows = cur.fetchall()
                if getattr(cur, "_as_dict", False) and rows:
                    result = [{"data": r["data"], "preco": float(r["preco"])} for r in rows]
                else:
                    result = [{"data": r[0], "preco": float(r[1])} for r in rows]
            else:
                cur.execute(
                    """SELECT data, preco FROM price_history_cache
                       WHERE asset = %s AND data >= %s ORDER BY data""",
                    (ASSET_BTC, data_inicio_str),
                )
                rows = cur.fetchall()
                for r in rows:
                    if isinstance(r, (list, tuple)) and len(r) >= 2:
                        result.append({"data": str(r[0])[:10], "preco": float(r[1])})
                    elif isinstance(r, dict):
                        result.append({"data": str(r["data"])[:10], "preco": float(r["preco"])})
        return result

    def salvar_cache(self, lista_precos: List[Dict[str, Any]]) -> None:
        """
        Salva em batch. Supabase: upsert em price_history_cache (asset, data, preco).
        SQLite: INSERT OR REPLACE em btc_price_history_cache. Nunca linha por linha.
        """
        if not lista_precos:
            return
        with self._get_connection() as conn:
            self._ensure_table(conn)
            is_sqlite = getattr(conn, "_is_sqlite", False)
            cur = conn.cursor()
            if is_sqlite:
                batch = [(p["data"], float(p["preco"])) for p in lista_precos if p.get("data") and p.get("preco") is not None]
                if batch:
                    cur.executemany(
                        "INSERT OR REPLACE INTO btc_price_history_cache (data, preco) VALUES (?, ?)",
                        batch,
                    )
            else:
                batch = [
                    (ASSET_BTC, p["data"], float(p["preco"]))
                    for p in lista_precos
                    if p.get("data") and p.get("preco") is not None
                ]
                if batch:
                    cur.executemany(
                        """INSERT INTO price_history_cache (asset, data, preco)
                           VALUES (%s, %s, %s)
                           ON CONFLICT (asset, data) DO UPDATE SET preco = EXCLUDED.preco""",
                        batch,
                    )
            if hasattr(conn, "commit"):
                conn.commit()
        logger.debug("[BTCCacheService] salvar_cache: %s registros", len(lista_precos))

    def cobre_periodo(self, dados_cache: List[Dict], data_inicio: date) -> bool:
        """
        True se o cache cobre o período desde data_inicio até hoje.
        Regras: vazio -> False; primeira_data > data_inicio -> False;
        última_data < hoje -> False; caso contrário -> True.
        """
        if not dados_cache:
            return False
        datas = [d.get("data") for d in dados_cache if d.get("data")]
        if not datas:
            return False
        primeira_str = min(datas)
        ultima_str = max(datas)
        hoje_str = date.today().isoformat()
        try:
            primeira = date.fromisoformat(primeira_str[:10])
            ultima = date.fromisoformat(ultima_str[:10])
        except (ValueError, TypeError):
            return False
        print("primeira:", primeira, "ultima:", ultima, "inicio:", data_inicio, flush=True)
        if primeira > data_inicio:
            return False
        if ultima < date.today():
            return False
        return True
