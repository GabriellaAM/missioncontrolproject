"""
Migra preços históricos dos CSVs em data/prices/ para a tabela precos_diarios.
Funciona tanto com SQLite local quanto com Supabase (PostgreSQL).
"""
import sys
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

from storage.sqlite_repo import get_repo

PRICES_DIR = _ROOT / "data" / "prices"


def main():
    repo = get_repo()

    csv_files = sorted(PRICES_DIR.glob("*.csv"))
    if not csv_files:
        print("Nenhum CSV encontrado em", PRICES_DIR)
        return

    print(f"Encontrados {len(csv_files)} CSVs de preços")
    total_inserted = 0

    with repo.connection() as conn:
        cursor = conn.cursor()

        for csv_file in csv_files:
            cg_id = csv_file.stem  # ex: bitcoin, ethereum
            try:
                df = pd.read_csv(csv_file, index_col='data', parse_dates=True)
            except Exception as e:
                print(f"  [SKIP] {cg_id}: {e}")
                continue

            if df.empty or 'close' not in df.columns:
                continue

            df = df[~df.index.duplicated(keep='first')]
            rows = [(cg_id, d.strftime('%Y-%m-%d'), float(r['close']), 'coingecko')
                    for d, r in df.iterrows() if pd.notna(r['close'])]

            if not rows:
                continue

            batch_size = 500
            inserted = 0
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                for row in batch:
                    cursor.execute("""
                        INSERT INTO precos_diarios (coingecko_id, data, preco, fonte)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (coingecko_id, data) DO UPDATE SET preco = EXCLUDED.preco
                    """, row)
                inserted += len(batch)

            total_inserted += inserted
            print(f"  {cg_id}: {inserted} registros")

        conn.commit()

    print(f"\nTotal: {total_inserted} registros inseridos/atualizados")


if __name__ == '__main__':
    main()
