"""
Sincroniza os CSVs de alocação (EXC, HB, LC, AC) com o SQLite.
Substitui alocações e posições dos produtos de alocação livre pelos dados dos CSVs.
"""
import sys
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

from storage.sqlite_repo import get_repo
from domain.posicao import Posicao
from domain.alocacao import Alocacao
from services.portfolio_service import PORTFOLIO_CONFIG, PortfolioService

# Mapeamento portfolio key -> produto_id (alocação livre)
PRODUTO_IDS = {
    'EXC': 2150859854,
    'HB': 2000449260,
    'LC': 2394004756,
    'AC': 3476245316,
}

ALLOCATIONS_DIR = _ROOT / "data" / "allocations"


def _load_allocation_df(csv_path: Path) -> pd.DataFrame:
    """Carrega CSV de alocação com mesma lógica do PortfolioService (percentuais 0-1)."""
    df = pd.read_csv(csv_path, index_col='Data', parse_dates=['Data'], date_format='%Y-%m-%d')
    df.columns = [c.strip().upper() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].apply(_parse_pct)
    return df.fillna(0.0)


def _parse_pct(x):
    if isinstance(x, str) and '%' in x:
        return float(x.strip('%').replace(',', '.').strip()) / 100.0
    try:
        v = float(x)
        return v / 100.0 if v > 1.0 else v
    except (ValueError, TypeError):
        return 0.0


def _days_diff(d1: str, d2: str) -> int:
    """Diferença em dias entre duas datas YYYY-MM-DD."""
    from datetime import datetime
    a, b = datetime.strptime(d1[:10], '%Y-%m-%d'), datetime.strptime(d2[:10], '%Y-%m-%d')
    return (b - a).days


def _contiguous_periods(dates_sorted):
    """Agrupa datas consecutivas em períodos (inicio, fim)."""
    if not dates_sorted:
        return []
    periods = []
    start = end = dates_sorted[0]
    for d in dates_sorted[1:]:
        if _days_diff(end, d) <= 1:
            end = d
        else:
            periods.append((start, end))
            start = end = d
    periods.append((start, end))
    return periods


def sync_product(repo, produto_id: int, csv_path: Path, verbose: bool = True):
    """
    Sincroniza um produto: remove alocações e posições existentes e recria a partir do CSV.
    """
    if not csv_path.exists():
        if verbose:
            print(f"  CSV não encontrado: {csv_path}")
        return 0, 0

    df = _load_allocation_df(csv_path)
    if df.empty:
        if verbose:
            print(f"  CSV vazio: {csv_path}")
        return 0, 0

    # Lista (data, ativo, pct) com pct > 0
    rows = []
    for data in df.index:
        data_str = data.strftime('%Y-%m-%d') if hasattr(data, 'strftime') else str(data)[:10]
        for ativo in df.columns:
            pct = float(df.loc[data, ativo])
            if pct > 0:
                rows.append((data_str, ativo.strip().upper(), pct))

    if not rows:
        if verbose:
            print(f"  Nenhuma alocação > 0 no CSV: {csv_path}")
        return 0, 0

    # Última data do CSV (para saber se período está aberto)
    last_date_csv = df.index.max()
    last_date_str = last_date_csv.strftime('%Y-%m-%d') if hasattr(last_date_csv, 'strftime') else str(last_date_csv)[:10]

    # Agrupar por ativo -> lista de (data, pct)
    ativo_dates = defaultdict(list)
    for data_str, ativo, pct in rows:
        ativo_dates[ativo].append((data_str, pct))

    # Períodos contíguos por ativo
    ativo_periods = {}
    for ativo, list_data_pct in ativo_dates.items():
        list_data_pct.sort(key=lambda x: x[0])
        dates_only = [x[0] for x in list_data_pct]
        periods = _contiguous_periods(dates_only)
        period_data = []
        for start, end in periods:
            in_period = [(d, p) for d, p in list_data_pct if start <= d <= end]
            period_data.append((start, end, in_period))
        ativo_periods[ativo] = period_data

    with repo._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM alocacoes WHERE produto_id = %s", (produto_id,))
        n_del_alloc = cur.rowcount
        cur.execute("DELETE FROM stops WHERE posicao_id IN (SELECT id FROM posicoes WHERE produto_id = %s)", (produto_id,))
        cur.execute("DELETE FROM posicoes WHERE produto_id = %s", (produto_id,))
        n_del_pos = cur.rowcount

    if verbose:
        print(f"  Removidas {n_del_alloc} alocações e {n_del_pos} posições antigas.")

    n_pos = 0
    aloc_rows = []  # (posicao_id, data, percentual)
    for ativo, periods in ativo_periods.items():
        for start, end, list_data_pct in periods:
            pos = Posicao(ativo=ativo, side='long', data_entrada=start, preco_entrada=0)
            pos_id = repo.salvar_posicao(produto_id, pos)
            n_pos += 1
            if end < last_date_str:
                repo.atualizar_posicao(pos_id, data_saida=end, status='closed')
            for data_str, pct in list_data_pct:
                aloc_rows.append((pos_id, data_str, round(pct * 100, 2)))

    # Inserir alocações em lote (executemany) — IDs únicos para evitar colisão
    n_alloc = len(aloc_rows)
    if aloc_rows:
        seen = set()
        def next_id():
            while True:
                uid = repo._gerar_id()
                if uid not in seen:
                    seen.add(uid)
                    return uid
        with repo._get_connection() as conn:
            cur = conn.cursor()
            params_list = [
                (next_id(), produto_id, pos_id, pct, None, data_str, 'active')
                for pos_id, data_str, pct in aloc_rows
            ]
            cur.executemany("""
                INSERT INTO alocacoes (id, produto_id, posicao_id, percentual, valor_usd, data, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, params_list)
            conn.commit()

    if verbose:
        print(f"  Criadas {n_pos} posições e {n_alloc} alocações.")
    return n_pos, n_alloc


def main():
    repo = get_repo()
    _root = _ROOT
    total_pos = 0
    total_alloc = 0

    for key, produto_id in PRODUTO_IDS.items():
        cfg = PORTFOLIO_CONFIG.get(key)
        if not cfg:
            continue
        csv_path = _root / cfg['csv']
        if not csv_path.exists():
            csv_path = ALLOCATIONS_DIR / cfg['csv']
        print(f"Produto {key} (id={produto_id}): {csv_path.name}")
        n_pos, n_alloc = sync_product(repo, produto_id, csv_path, verbose=True)
        total_pos += n_pos
        total_alloc += n_alloc
        print()

    print(f"Total: {total_pos} posições e {total_alloc} alocações sincronizadas.")


if __name__ == '__main__':
    main()
