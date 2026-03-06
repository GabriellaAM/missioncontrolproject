"""
Baixa preços históricos do CoinGecko para todos os ativos
usados nos portfólios EXC, HB, LC e AC.
Grava na tabela precos_diarios (banco) e opcionalmente em CSVs locais.
"""
import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

from services.portfolio_service import TICKER_TO_COINGECKO, PORTFOLIO_CONFIG, PortfolioService
from storage.sqlite_repo import get_repo

PRICES_DIR = _ROOT / "data" / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)

try:
    from pycoingecko import CoinGeckoAPI
    CG_API_KEY = os.getenv('COINGECKO_API_KEY', 'CG-S24fNKNPYgDiQfRfPpxCPnTP')
    cg = CoinGeckoAPI(api_key=CG_API_KEY)
except ImportError:
    cg = None
    print("[WARN] pycoingecko não instalado. Instale com: pip install pycoingecko")


def download_prices(cg_id, days='max'):
    """Baixa preço histórico e retorna DataFrame."""
    if cg is None:
        return None
    try:
        data = cg.get_coin_market_chart_by_id(
            id=cg_id, vs_currency='usd', days=str(days), interval='daily')
        df = pd.DataFrame(data['prices'], columns=['data', 'close'])
        df['data'] = pd.to_datetime(df['data'], unit='ms').dt.normalize()
        df = df.iloc[:-1]
        df['data'] = df['data'] - timedelta(days=1)
        df.set_index('data', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        return df
    except Exception as e:
        print(f"  [ERRO] {cg_id}: {e}")
        return None


def save_to_db(repo, cg_id, df):
    """Grava DataFrame de preços na tabela precos_diarios."""
    if df is None or df.empty:
        return 0
    rows = [(cg_id, d.strftime('%Y-%m-%d'), float(r['close']), 'coingecko')
            for d, r in df.iterrows() if pd.notna(r['close'])]
    if not rows:
        return 0
    with repo.connection() as conn:
        cursor = conn.cursor()
        for row in rows:
            cursor.execute("""
                INSERT INTO precos_diarios (coingecko_id, data, preco, fonte)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (coingecko_id, data) DO UPDATE SET preco = EXCLUDED.preco
            """, row)
        conn.commit()
    return len(rows)


def get_ultima_data_db(repo, cg_id):
    """Retorna a última data disponível no banco para o ativo."""
    try:
        with repo.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT MAX(data) FROM precos_diarios WHERE coingecko_id = %s", (cg_id,))
            row = cursor.fetchone()
            if row and row[0]:
                return datetime.strptime(row[0], '%Y-%m-%d').date()
    except Exception:
        pass
    return None


def main():
    repo = get_repo()

    all_cg_ids = set()
    for key, cfg in PORTFOLIO_CONFIG.items():
        csv_path = _ROOT / cfg['csv']
        if not csv_path.exists():
            csv_path = _ROOT / "data" / "allocations" / cfg['csv']
        if csv_path.exists():
            svc = PortfolioService(nome=cfg['nome'], csv_path=str(csv_path))
            for ticker in svc.df_aloc.columns:
                cg_id = TICKER_TO_COINGECKO.get(ticker)
                if cg_id:
                    all_cg_ids.add(cg_id)

    all_cg_ids.add('bitcoin')

    print(f"Total de ativos para baixar: {len(all_cg_ids)}")
    downloaded = 0
    skipped = 0

    ontem = (datetime.now() - timedelta(days=1)).date()

    for cg_id in sorted(all_cg_ids):
        ultima_data = get_ultima_data_db(repo, cg_id)

        if ultima_data and ultima_data >= ontem:
            skipped += 1
            continue

        if ultima_data:
            dias_faltando = (datetime.now().date() - ultima_data).days
            print(f"  Atualizando {cg_id} ({dias_faltando} dias)...")
            df = download_prices(cg_id, days=dias_faltando + 2)
        else:
            print(f"  Baixando {cg_id} (max)...")
            df = download_prices(cg_id, days='max')

        if df is not None and not df.empty:
            n = save_to_db(repo, cg_id, df)
            downloaded += 1
            print(f"    -> {n} registros salvos no banco")

        time.sleep(1.5)

    print(f"\nConcluído: {downloaded} baixados, {skipped} já atualizados")


if __name__ == '__main__':
    main()
