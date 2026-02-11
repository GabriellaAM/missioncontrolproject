"""
Atualiza alocações dos produtos AC, EXC, HB e LC com base nos CSVs de mesmo nome.
Cada CSV tem coluna Data e colunas de ativos com percentual (ex: 50.00%).
As alocações antigas do produto são removidas e substituídas pelas do CSV.
Requer que existam posições do produto para cada ativo (ativo = nome da coluna).

OTIMIZADO: usa INSERT batch direto em uma única conexão PostgreSQL para
evitar milhares de conexões ao Supabase (que causavam timeout/deadlock).
"""
import sys
import csv
import uuid
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import connect_pg

# Nome do CSV (arquivo) -> nome do produto no banco (se diferente)
PRODUTOS_CSV = ("AC", "EXC", "HB", "LC")
MAP_CSV_TO_PRODUTO = {"AC": "Alphacoins"}  # CSV AC.csv -> produto "Alphacoins"
# Caminho absoluto para funcionar no Render (CWD pode ser qualquer)
DIR_CSV = Path(__file__).resolve().parent.parent  # productsPositions

def _log(msg):
    print(msg, flush=True)

def _gerar_id():
    return int(uuid.uuid4().int % (10 ** 10))

def _parse_percentual(val):
    """Converte '50.00%' ou '50' para float 50.0."""
    if val is None or (isinstance(val, str) and not val.strip()):
        return 0.0
    s = str(val).strip().replace("%", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _posicao_aberta_em(data_entrada, data_saida, data):
    """True se a posição estava aberta na data."""
    if data_entrada is None:
        return False
    try:
        d = datetime.strptime(str(data)[:10], "%Y-%m-%d").date() if data else None
        e = datetime.strptime(str(data_entrada)[:10], "%Y-%m-%d").date() if data_entrada else None
        s = datetime.strptime(str(data_saida)[:10], "%Y-%m-%d").date() if data_saida else None
    except Exception:
        return False
    if e and d and e > d:
        return False
    if s and d and s < d:
        return False
    return True


def _escolher_posicao_id(posicoes_por_ativo, ativo, data):
    """
    Dado lista de dicts com id, ativo, data_entrada, data_saida para um produto,
    retorna o id de uma posição com esse ativo que estava aberta na data;
    se não houver, retorna qualquer posição com esse ativo (a de maior id).
    """
    candidatas = [p for p in posicoes_por_ativo if p["ativo"] == ativo]
    if not candidatas:
        return None
    abertas = [
        p for p in candidatas
        if _posicao_aberta_em(p.get("data_entrada"), p.get("data_saida"), data)
    ]
    if abertas:
        return max(abertas, key=lambda p: p["id"])["id"]
    return max(candidatas, key=lambda p: p["id"])["id"]


def processar_produto(conn, nome_produto, dry_run=False):
    """
    Carrega CSV {nome_produto}.csv, remove alocações do produto e insere as do CSV.
    Usa a conexão conn já aberta (sem abrir novas).
    Retorna (num_deletadas, num_inseridas, erros).
    """
    nome_no_banco = MAP_CSV_TO_PRODUTO.get(nome_produto, nome_produto)

    # Buscar produto por nome
    cur = conn.cursor()
    cur.execute("SELECT id FROM produtos WHERE nome = %s", (nome_no_banco,))
    row = cur.fetchone()
    if not row:
        return 0, 0, [f"Produto '{nome_produto}' (nome_banco='{nome_no_banco}') não encontrado."]
    produto_id = int(row[0])

    csv_path = DIR_CSV / f"{nome_produto}.csv"
    if not csv_path.exists():
        return 0, 0, [f"Arquivo não encontrado: {csv_path}"]

    # Carregar posições do produto
    cur.execute(
        "SELECT id, ativo, data_entrada, data_saida FROM posicoes WHERE produto_id = %s",
        (produto_id,),
    )
    posicoes = [
        {"id": r[0], "ativo": r[1], "data_entrada": r[2], "data_saida": r[3]}
        for r in cur.fetchall()
    ]
    if not posicoes:
        return 0, 0, [f"Produto {nome_produto} (id={produto_id}) não possui posições."]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or reader.fieldnames[0].strip().lower() != "data":
            return 0, 0, [f"CSV {nome_produto}.csv: primeira coluna deve ser 'Data'."]
        col_data = reader.fieldnames[0]
        colunas_ativos = [c.strip() for c in reader.fieldnames[1:] if c.strip()]

    # Montar todas as tuplas de INSERT em memória
    rows_to_insert = []
    erros = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = (row.get(col_data) or "").strip()
            if not data:
                continue
            for ativo in colunas_ativos:
                pct = _parse_percentual(row.get(ativo))
                if pct <= 0:
                    continue
                posicao_id = _escolher_posicao_id(posicoes, ativo, data)
                if posicao_id is None:
                    erros.append(f"{nome_produto} {data} ativo {ativo}: sem posição; ignorado.")
                    continue
                rows_to_insert.append((
                    _gerar_id(),        # id
                    produto_id,         # produto_id
                    posicao_id,         # posicao_id
                    pct,                # percentual
                    None,               # valor_usd
                    data,               # data
                    'active',           # status
                ))

    if dry_run:
        cur.execute("SELECT COUNT(*) FROM alocacoes WHERE produto_id = %s", (produto_id,))
        deletadas = cur.fetchone()[0]
        return deletadas, len(rows_to_insert), erros

    # DELETE antigos
    cur.execute("DELETE FROM alocacoes WHERE produto_id = %s", (produto_id,))
    deletadas = cur.rowcount

    # INSERT em batch (lotes de 500 para não estourar limites do PostgreSQL)
    BATCH = 500
    inseridas = 0
    for i in range(0, len(rows_to_insert), BATCH):
        lote = rows_to_insert[i:i + BATCH]
        placeholders = ",".join(
            cur.mogrify("(%s,%s,%s,%s,%s,%s,%s)", r).decode() for r in lote
        )
        cur.execute(
            "INSERT INTO alocacoes (id, produto_id, posicao_id, percentual, valor_usd, data, status) VALUES " + placeholders
        )
        inseridas += len(lote)

    return deletadas, inseridas, erros


def main(dry_run=False):
    import os
    _log("[ALOCAÇÕES] Conectando ao banco e carregando CSVs (AC, EXC, HB, LC)...")

    # Abrir conexão diretamente (sem context manager do repo que faz auto-commit)
    db_url = (os.getenv('SUPABASE_DB_URL') or '').strip()
    if not db_url:
        _log("[ALOCAÇÕES] ERRO: SUPABASE_DB_URL não definido.")
        return 0

    conn = connect_pg(db_url)
    conn.autocommit = False  # transação explícita

    total_del = 0
    total_ins = 0
    all_erros = []

    try:
        for nome in PRODUTOS_CSV:
            _log(f"  Processando {nome}...")
            del_n, ins_n, erros_prod = processar_produto(conn, nome, dry_run=dry_run)
            total_del += del_n
            total_ins += ins_n
            all_erros.extend(erros_prod)
            if dry_run:
                _log(f"  [dry-run] {nome}: deletaria {del_n} alocações, inseriria {ins_n}")
            else:
                _log(f"  {nome}: {del_n} alocações removidas, {ins_n} inseridas.")

        if not dry_run:
            conn.commit()
            _log("[ALOCAÇÕES] Commit realizado.")
    except Exception as e:
        conn.rollback()
        _log(f"[ALOCAÇÕES] ERRO — rollback: {e}")
        raise
    finally:
        conn.close()

    if all_erros:
        _log("\nAvisos/erros:")
        for e in all_erros[:50]:
            _log(f"  - {e}")
        if len(all_erros) > 50:
            _log(f"  ... e mais {len(all_erros) - 50}.")

    _log(f"\nTotal: {total_del} removidas, {total_ins} inseridas.")
    return total_ins


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Atualiza alocações a partir dos CSVs AC, EXC, HB, LC.")
    p.add_argument("--dry-run", action="store_true", help="Só simular, não gravar.")
    args = p.parse_args()
    main(dry_run=args.dry_run)
