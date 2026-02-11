"""
Atualiza alocações dos produtos AC, EXC, HB e LC com base nos CSVs de mesmo nome.
Cada CSV tem coluna Data e colunas de ativos com percentual (ex: 50.00%).
As alocações antigas do produto são removidas e substituídas pelas do CSV.
Requer que existam posições do produto para cada ativo (ativo = nome da coluna).
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from services.alocacao_service import AlocacaoService

# Nome do CSV (arquivo) -> nome do produto no banco (se diferente)
PRODUTOS_CSV = ("AC", "EXC", "HB", "LC")
MAP_CSV_TO_PRODUTO = {"AC": "Alphacoins"}  # CSV AC.csv -> produto "Alphacoins"
# Caminho absoluto para funcionar no Render (CWD pode ser qualquer)
DIR_CSV = Path(__file__).resolve().parent.parent  # productsPositions


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
        from datetime import datetime
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


def _carregar_posicoes_produto(repo, produto_id):
    """Retorna lista de dicts: id, ativo, data_entrada, data_saida."""
    with repo._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, ativo, data_entrada, data_saida FROM posicoes WHERE produto_id = %s",
            (produto_id,),
        )
        rows = cur.fetchall()
    return [
        {"id": r[0], "ativo": r[1], "data_entrada": r[2], "data_saida": r[3]}
        for r in rows
    ]


def _deletar_alocacoes_produto(repo, produto_id):
    with repo._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM alocacoes WHERE produto_id = %s", (produto_id,))
        return cur.rowcount


def processar_produto(repo, nome_produto, dry_run=False):
    """
    Carrega CSV {nome_produto}.csv, remove alocações do produto e insere as do CSV.
    Retorna (num_deletadas, num_inseridas, erros).
    """
    nome_no_banco = MAP_CSV_TO_PRODUTO.get(nome_produto, nome_produto)
    produto = repo.obter_produto_por_nome(nome_no_banco)
    if not produto:
        return 0, 0, [f"Produto '{nome_produto}' não encontrado."]

    produto_id = int(produto["id"])
    csv_path = DIR_CSV / f"{nome_produto}.csv"
    if not csv_path.exists():
        return 0, 0, [f"Arquivo não encontrado: {csv_path}"]

    posicoes = _carregar_posicoes_produto(repo, produto_id)
    if not posicoes:
        return 0, 0, [f"Produto {nome_produto} (id={produto_id}) não possui posições."]

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or reader.fieldnames[0].strip().lower() != "data":
            return 0, 0, [f"CSV {nome_produto}.csv: primeira coluna deve ser 'Data'."]
        col_data = reader.fieldnames[0]
        colunas_ativos = [c.strip() for c in reader.fieldnames[1:] if c.strip()]

    if dry_run:
        deletadas = 0
        inseridas = 0
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
                    if _escolher_posicao_id(posicoes, ativo, data) is None:
                        erros.append(f"{nome_produto} {data} ativo {ativo}: sem posição; ignorado.")
                    else:
                        inseridas += 1
        with repo._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM alocacoes WHERE produto_id = %s", (produto_id,))
            deletadas = cur.fetchone()[0]
        return deletadas, inseridas, erros

    BATCH = 2000
    deletadas = _deletar_alocacoes_produto(repo, produto_id)
    inseridas = 0
    erros = []
    batch = []

    def flush_batch():
        nonlocal inseridas
        if not batch:
            return
        try:
            repo.salvar_alocacoes(produto_id, batch)
            inseridas += len(batch)
        except Exception as e:
            if "UNIQUE" in str(e) or "unique" in str(e).lower():
                for aloc in batch:
                    try:
                        repo.salvar_alocacao(produto_id, aloc)
                        inseridas += 1
                    except Exception as e2:
                        erros.append(f"{nome_produto} id colisão fallback: {e2}")
            else:
                erros.append(f"{nome_produto} batch: {e}")
        batch.clear()

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
                try:
                    alocacao = AlocacaoService.criar_alocacao(
                        produto_id=produto_id,
                        posicao_id=posicao_id,
                        percentual=pct,
                        valor_usd=None,
                        data=data,
                    )
                    batch.append(alocacao)
                    if len(batch) >= BATCH:
                        flush_batch()
                except Exception as e:
                    erros.append(f"{nome_produto} {data} {ativo}: {e}")
    flush_batch()

    return deletadas, inseridas, erros


def main(dry_run=False):
    print("[ALOCAÇÕES] Conectando ao banco e carregando CSVs (AC, EXC, HB, LC)...")
    repo = SQLiteRepo()
    total_del = 0
    total_ins = 0
    all_erros = []

    for nome in PRODUTOS_CSV:
        del_n, ins_n, erros = processar_produto(repo, nome, dry_run=dry_run)
        total_del += del_n
        total_ins += ins_n
        all_erros.extend(erros)
        if dry_run:
            print(f"  [dry-run] {nome}: deletaria {del_n} alocações, inseriria {ins_n}")
        else:
            print(f"  {nome}: {del_n} alocações removidas, {ins_n} inseridas.")

    if all_erros:
        print("\nAvisos/erros:")
        for e in all_erros[:50]:
            print(f"  - {e}")
        if len(all_erros) > 50:
            print(f"  ... e mais {len(all_erros) - 50}.")

    print(f"\nTotal: {total_del} removidas, {total_ins} inseridas.")
    return total_ins


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Atualiza alocações a partir dos CSVs AC, EXC, HB, LC.")
    p.add_argument("--dry-run", action="store_true", help="Só simular, não gravar.")
    args = p.parse_args()
    main(dry_run=args.dry_run)
