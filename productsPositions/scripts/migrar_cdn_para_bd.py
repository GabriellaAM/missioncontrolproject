#!/usr/bin/env python3
"""
migrar_cdn_para_bd.py
=====================
Deleta todas as posições de Crypto Signals e ICOs do banco de dados
e reimporta a partir dos CSVs publicados no CDN da Empiricus.

Uso:
    python migrar_cdn_para_bd.py                # ambos (local + nuvem)
    python migrar_cdn_para_bd.py --local        # apenas banco local (SQLite)
    python migrar_cdn_para_bd.py --cloud        # apenas banco na nuvem (Supabase)
    python migrar_cdn_para_bd.py --both -y      # ambos, sem pedir confirmação
"""

import sys
import os
import csv
import io
import uuid
import argparse
from pathlib import Path
from datetime import date

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(1, str(_SCRIPT_DIR))

import requests
from storage.sqlite_repo import SQLiteRepo, _DEFAULT_SQLITE_PATH
from services.portfolio_service import TICKER_TO_COINGECKO

# ───────────────────────── CDN ──────────────────────────────────
CS_CDN = (
    "https://cdn-mediacenter-publicacoes-files.empiricus.com.br"
    "/crypto-integration/cryptosignals"
)
ICO_CDN = (
    "https://cdn-mediacenter-publicacoes-files.empiricus.com.br"
    "/crypto-integration/icos"
)

CS_PRODUTO_ID = 4970919917
ICOS_PRODUCT_NAMES = ["icos", "ico"]


# ═══════════════════════ Helpers ════════════════════════════════

def gerar_id():
    return uuid.uuid4().int % (10 ** 10)


def _normalize_key(k):
    """Normaliza chave para comparação (lowercase, remove acentos comuns)."""
    if not k:
        return ""
    s = str(k).strip().lower()
    for old, new in [("ç", "c"), ("ã", "a"), ("á", "a"), ("à", "a"), ("â", "a"), ("é", "e"), ("ê", "e"), ("í", "i"), ("ó", "o"), ("ô", "o"), ("ú", "u")]:
        s = s.replace(old, new)
    return s.replace(" ", "_")


def download_csv(url):
    """Baixa CSV do CDN, detecta encoding, delimitador e retorna lista de dicts."""
    print(f"  Baixando {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    raw = resp.content
    text = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        raise ValueError("Não foi possível decodificar o CSV (encoding)")

    text = text.strip().lstrip("\ufeff")
    if not text:
        return []
    first_line = text.split("\n")[0] if "\n" in text else text
    tab_c, comma_c, semi_c = first_line.count("\t"), first_line.count(","), first_line.count(";")
    delimiter = "\t" if tab_c >= comma_c and tab_c >= semi_c else (";" if semi_c >= comma_c else ",")
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = [{k.strip(): v for k, v in row.items()} for row in reader]
    print(f"  -> {len(rows)} linhas (delimitador: {repr(delimiter)})")
    return rows


def parse_price(value):
    """Converte string de preço para float. Aceita $0.08, 1.234,56, 1,234.56, etc."""
    if value is None:
        return None
    s = str(value).strip().replace("$", "").replace("%", "").replace(" ", "").replace("\xa0", "").replace("\u00a0", "")
    if not s or s in ("—", "-", "–", ""):
        return None
    s = s.replace("R$", "").strip()
    lc, ld = s.rfind(","), s.rfind(".")
    if lc > -1 and ld > -1:
        if lc > ld:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif lc > -1:
        s = s.replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_date_br(value):
    """dd/mm/yyyy ou dd-mm-yyyy -> yyyy-mm-dd. Aceita também yyyy-mm-dd direto."""
    if not value:
        return None
    s = str(value).strip()
    for sep in ("/", "-"):
        parts = s.split(sep)
        if len(parts) == 3:
            try:
                d, m, y = parts[0], parts[1], parts[2]
                if len(y) == 4 and len(d) <= 2 and len(m) <= 2:
                    return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
            except Exception:
                pass
    if "-" in s and len(s) == 10:
        return s
    return None


def _build_column_map(rows):
    """Constrói mapeamento de nomes canônicos para chaves reais do CSV (case-insensitive)."""
    if not rows:
        return {}
    header_keys = list(rows[0].keys())
    key_lower = {_normalize_key(k): k for k in header_keys}
    aliases = {
        "trade_id": ["Trade_ID", "trade_id"],
        "etapa": ["Etapa", "etapa"],
        "data": ["Data", "data"],
        "ativo": ["Ativo", "ativo"],
        "tipo": ["Tipo", "tipo"],
        "perfil": ["Perfil", "perfil"],
        "preco_entrada": ["Preço_Entrada", "Preco_Entrada", "preco_entrada"],
        "preco_atual": ["Preço_Atual", "Preco_Atual", "preco_atual"],
        "preco_saida": ["Preço_Saida", "Preco_Saida", "preco_saida"],
        "alvo1": ["Alvo1", "alvo1"],
        "alvo2": ["Alvo2", "alvo2"],
        "stop": ["Stop", "stop"],
        "projeto": ["Projeto", "projeto"],
        "data_ico": ["Data_ICO", "Data_Entrada", "data_ico"],
        "data_saida": ["Data_Saida", "Data_Saída", "data_saida"],
        "ticker": ["Ticker", "ticker"],
        "coingecko_id": ["coingecko_id"],
        "categoria": ["Categoria", "categoria"],
        "tipo_ico": ["Tipo", "tipo"],
        "resultado": ["Resultado", "resultado"],
        "pnl_pct": ["PnL (%)", "PnL", "pnl", "pnl_pct"],
        "local": ["Local", "local"],
        "ficha_tecnica": ["Ficha_Tecnica", "Ficha", "ficha_tecnica"],
        "disponivel_pos_ico": ["Disponivel_Pos_ICO", "disponivel_pos_ico"],
        "rank": ["Rank", "rank"],
        "tipo_janela": ["Tipo_Janela", "tipo_janela"],
        "tese": ["Tese", "tese"],
        "risco": ["Risco", "risco"],
        "atencao": ["Atencao", "atenção", "atencao"],
        "execucao": ["Execucao", "execucao"],
        "por_que": ["Por_que", "Por_que", "por_que"],
    }
    out = {}
    for canon, variants in aliases.items():
        for v in variants:
            n = _normalize_key(v)
            if n and n in key_lower:
                out[canon] = key_lower[n]
                break
    return out


def _get(row, *keys):
    """Retorna o primeiro valor não-vazio do row para as chaves dadas."""
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _get_mapped(row, col_map, *canon_keys):
    """Retorna o primeiro valor não-vazio usando col_map para resolver chaves."""
    for ck in canon_keys:
        real_key = col_map.get(ck)
        if real_key:
            v = row.get(real_key)
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


# ═══════════════════════ Limpeza ════════════════════════════════

def limpar_produto(repo, produto_id):
    """Remove todas as posições e dados relacionados de um produto."""
    with repo._get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM carteira_turma WHERE trade_id IN (
                SELECT tt.id FROM trades_turma tt
                JOIN turmas t ON tt.turma_id = t.id WHERE t.produto_id = %s)
        """, (produto_id,))
        cur.execute("""
            DELETE FROM trades_turma WHERE turma_id IN (
                SELECT id FROM turmas WHERE produto_id = %s)
        """, (produto_id,))
        cur.execute("""
            DELETE FROM stops WHERE posicao_id IN (
                SELECT id FROM posicoes WHERE produto_id = %s)
        """, (produto_id,))
        cur.execute(
            "DELETE FROM posicao_atributos_produto WHERE produto_id = %s",
            (produto_id,),
        )
        cur.execute("DELETE FROM posicoes WHERE produto_id = %s", (produto_id,))
    print(f"  Dados antigos removidos para produto {produto_id}")


def get_or_create_turma(repo, produto_id, nome):
    with repo._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM turmas WHERE produto_id = %s LIMIT 1", (produto_id,)
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            """INSERT INTO turmas (produto_id, nome, data_inicio, capital_base, data_criacao)
               VALUES (%s, %s, %s, %s, %s)""",
            (produto_id, nome, date.today().isoformat(), 1500.0,
             date.today().isoformat()),
        )
    with repo._get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM turmas WHERE produto_id = %s ORDER BY id DESC LIMIT 1",
            (produto_id,),
        )
        return cur.fetchone()[0]


def inserir_posicao_completa(
    repo, produto_id, turma_id, posicao_id, ativo,
    coingecko_id, side, data_entrada, preco_entrada,
    data_saida, preco_saida, status, stops_list,
    exchange_symbol=None,
):
    """Insere posição + stops + vínculo à turma numa única transação."""
    with repo._get_connection() as conn:
        cur = conn.cursor()

        cur.execute(
            """INSERT INTO posicoes
               (id, produto_id, ativo, coingecko_id, exchange_symbol,
                side, data_entrada, preco_entrada, data_saida, preco_saida, status)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (posicao_id, produto_id, ativo, coingecko_id, exchange_symbol,
             side, data_entrada, preco_entrada, data_saida, preco_saida, status),
        )

        for s_data, s_val in stops_list:
            cur.execute(
                "INSERT INTO stops (posicao_id, data, valor) VALUES (%s,%s,%s)",
                (posicao_id, s_data, s_val),
            )

        cur.execute(
            "INSERT INTO trades_turma (turma_id, posicao_id) VALUES (%s,%s)",
            (turma_id, posicao_id),
        )
        cur.execute(
            "SELECT id FROM trades_turma WHERE turma_id = %s AND posicao_id = %s",
            (turma_id, posicao_id),
        )
        tt_id = cur.fetchone()[0]

        is_closed = status == "closed"
        cur.execute(
            """INSERT INTO carteira_turma
               (turma_id, trade_id, origem, data_insercao,
                data_remocao, preco_entrada_turma, ativo_atual)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (turma_id, tt_id, "nativo", data_entrada,
             data_saida if is_closed else None,
             preco_entrada, 0 if is_closed else 1),
        )


# ═══════════════════ Crypto Signals ═════════════════════════════

def importar_crypto_signals(repo):
    print(f"\n{'=' * 60}")
    print("  CRYPTO SIGNALS")
    print("=" * 60)

    limpar_produto(repo, CS_PRODUTO_ID)

    rows = download_csv(f"{CS_CDN}/historico.csv")
    if not rows:
        print("  AVISO: historico.csv vazio!")
        return

    col_map = _build_column_map(rows)

    def _g(row, *keys):
        return _get_mapped(row, col_map, *keys)

    def _etapa(row):
        e = _g(row, "etapa")
        return (e or "").strip().lower() if e else ""

    trades: dict[str, list] = {}
    for row in rows:
        tid = _g(row, "trade_id")
        if tid:
            trades.setdefault(tid, []).append(row)

    print(f"  {len(trades)} trades encontrados")

    closed_ids = {
        tid
        for tid, entries in trades.items()
        if any(_etapa(e) == "encerramento" for e in entries)
    }

    turma_id = get_or_create_turma(repo, CS_PRODUTO_ID, "Crypto Signals")

    count = erros = 0
    for tid, entries in trades.items():
        try:
            abertura = next(
                (e for e in entries if _etapa(e) == "abertura"),
                entries[0],
            )
            encerramento = next(
                (e for e in entries if _etapa(e) == "encerramento"),
                None,
            )
            latest = entries[-1]

            ativo = _g(latest, "ativo")
            if not ativo:
                continue

            tipo = (_g(latest, "tipo") or "long").lower()
            side = "short" if "short" in tipo else "long"

            data_entrada = parse_date_br(_g(abertura, "data"))
            preco_entrada = parse_price(_g(abertura, "preco_entrada"))
            if not data_entrada or preco_entrada is None:
                print(f"  SKIP trade {tid}: dados de abertura incompletos")
                continue

            is_closed = tid in closed_ids
            data_saida = preco_saida = None
            if is_closed and encerramento:
                data_saida = parse_date_br(_g(encerramento, "data"))
                preco_saida = parse_price(
                    _g(encerramento, "preco_atual") or _g(encerramento, "preco_saida")
                )
                if preco_saida is None and preco_entrada:
                    pnl_raw = parse_price(_g(encerramento, "pnl_pct"))
                    if pnl_raw is not None:
                        mult = 1 + pnl_raw / 100
                        preco_saida = preco_entrada * mult if "short" not in tipo else preco_entrada * (2 - mult)

            ativo_upper = ativo.strip().upper()
            cg_id = TICKER_TO_COINGECKO.get(ativo_upper)
            ex_sym = f"{ativo_upper}USDT" if ativo_upper else None
            repo.registrar_ativo(ativo, cg_id)

            stops_list = []
            seen_stops: set[str] = set()
            for e in entries:
                sv = parse_price(_g(e, "stop"))
                sd = parse_date_br(_g(e, "data"))
                if sv is not None and sd:
                    key = f"{sd}|{sv}"
                    if key not in seen_stops:
                        stops_list.append((sd, sv))
                        seen_stops.add(key)

            posicao_id = gerar_id()
            inserir_posicao_completa(
                repo, CS_PRODUTO_ID, turma_id, posicao_id, ativo,
                cg_id, side, data_entrada, preco_entrada,
                data_saida, preco_saida,
                "closed" if is_closed else "open", stops_list,
                exchange_symbol=ex_sym,
            )

            attrs: dict = {}
            perfil = _g(latest, "perfil")
            alvo1 = parse_price(_g(latest, "alvo1"))
            alvo2 = parse_price(_g(latest, "alvo2"))
            if perfil:
                attrs["perfil"] = perfil
            if alvo1 is not None:
                attrs["alvo1"] = alvo1
            if alvo2 is not None:
                attrs["alvo2"] = alvo2
            if attrs:
                repo.salvar_atributos_posicao(posicao_id, CS_PRODUTO_ID, **attrs)

            count += 1
        except Exception as exc:
            erros += 1
            print(f"  ERRO trade {tid}: {exc}")

    n_open = sum(1 for t in trades if t not in closed_ids)
    print(f"\n  {count} posicoes importadas ({n_open} abertas, {len(closed_ids)} fechadas)")
    if erros:
        print(f"  {erros} erros")


# ═══════════════════════ ICOs ═══════════════════════════════════
#
# Estrutura alinhada ao dashboard (Possíveis ICOs | Abertas | Histórico):
#
#   possiveis_icos.csv  →  Tab "Possíveis ICOs"  (preco_entrada=0, status=open)
#   carteira.csv        →  Tab "Abertas"         (preco_entrada>0, status=open)
#   icos_participados   →  Tab "Histórico"       (status=closed)
#

ICO_CSV_URLS = {
    "possiveis": f"{ICO_CDN}/possiveis_icos.csv",
    "carteira": f"{ICO_CDN}/carteira.csv",
    "participados": f"{ICO_CDN}/icos_participados.csv",
}


def encontrar_produto_icos(repo):
    for p in repo.listar_produtos():
        if (p.get("nome") or "").strip().lower() in ICOS_PRODUCT_NAMES:
            return p["id"]
    return None


def _resolve_cg_ticker(row, _g, ativo):
    """Resolve coingecko_id e exchange_symbol a partir do row ou TICKER_TO_COINGECKO."""
    cg_id = _g(row, "coingecko_id") or TICKER_TO_COINGECKO.get(
        (ativo or "").strip().upper()
    ) or TICKER_TO_COINGECKO.get((_g(row, "ticker") or "").strip().upper())
    ticker = _g(row, "ticker") or (ativo if (ativo or "").upper() in TICKER_TO_COINGECKO else None)
    ex_sym = f"{ticker.strip().upper()}USDT" if ticker else None
    return cg_id, ex_sym


def _garantir_colunas_icos(repo, produto_id):
    """Cria colunas dinâmicas de ICOs em posicao_atributos_produto se necessário."""
    icos_attrs = {
        "categoria": "text", "tipo_ico": "text", "resultado": "text",
        "local": "text", "ficha_tecnica": "text", "disponivel_pos_ico": "text",
        "tese": "text", "risco": "text", "atencao": "text",
        "execucao": "text", "rank": "text", "tipo_janela": "text",
        "por_que": "text",
    }
    existentes = repo.listar_colunas_atributos()
    for nome, tipo in icos_attrs.items():
        if nome not in existentes:
            try:
                repo.adicionar_atributo_config(produto_id, nome, tipo)
                print(f"  Coluna '{nome}' criada em posicao_atributos_produto")
            except Exception as exc:
                print(f"  AVISO ao criar coluna '{nome}': {exc}")


def importar_icos(repo):
    print(f"\n{'=' * 60}")
    print("  ICOs (Possíveis | Abertas | Histórico)")
    print("=" * 60)

    produto_id = encontrar_produto_icos(repo)
    if not produto_id:
        print("  ERRO: Produto ICOs nao encontrado no banco!")
        print("  Produtos disponiveis:")
        for p in repo.listar_produtos():
            print(f"    - {p.get('id')}: {p.get('nome')}")
        return

    print(f"  Produto ICOs: id={produto_id}")
    limpar_produto(repo, produto_id)
    _garantir_colunas_icos(repo, produto_id)

    turma_id = get_or_create_turma(repo, produto_id, "ICOs")
    count = erros = 0

    # ── 1. Possíveis ICOs (tab "Possíveis ICOs": Rank, Tipo Janela, Tese, Risco, Atenção, Execução, Por quê)
    print("\n  --- 1. Possíveis ICOs (análise, preco_entrada=0) ---")
    try:
        possiveis_rows = download_csv(ICO_CSV_URLS["possiveis"])
        col_map = _build_column_map(possiveis_rows)

        def _g(row, *keys):
            return _get_mapped(row, col_map, *keys)

        for row in possiveis_rows:
            try:
                ativo = _g(row, "projeto")
                if not ativo:
                    continue
                cg_id, ex_sym = _resolve_cg_ticker(row, _g, ativo)
                data_ent = parse_date_br(_g(row, "data_ico")) or date.today().isoformat()

                repo.registrar_ativo(ativo, cg_id)
                pid = gerar_id()
                inserir_posicao_completa(
                    repo, produto_id, turma_id, pid, ativo,
                    cg_id, "long", data_ent, 0.0,
                    None, None, "open", [],
                    exchange_symbol=ex_sym,
                )

                attrs = {}
                for k in ("rank", "tipo_janela", "tese", "risco", "atencao", "execucao", "por_que"):
                    v = _g(row, k)
                    if v:
                        attrs[k] = v
                if attrs:
                    repo.salvar_atributos_posicao(pid, produto_id, **attrs)
                count += 1
            except Exception as exc:
                erros += 1
                print(f"  ERRO possivel '{_g(row, 'projeto') or '?'}': {exc}")
    except Exception as exc:
        print(f"  ERRO ao baixar possiveis_icos.csv: {exc}")

    # ── 2. Abertas (tab "Abertas": Categoria, Tipo, Data ICO, P. Entrada, P. Atual, PnL%)
    print("\n  --- 2. Abertas (carteira, preco_entrada>0) ---")
    try:
        carteira_rows = download_csv(ICO_CSV_URLS["carteira"])
        col_map = _build_column_map(carteira_rows)

        def _g(row, *keys):
            return _get_mapped(row, col_map, *keys)

        for row in carteira_rows:
            try:
                ativo = _g(row, "projeto")
                if not ativo:
                    continue
                data_ent = parse_date_br(_g(row, "data_ico")) or date.today().isoformat()
                preco_ent = parse_price(_g(row, "preco_entrada")) or 0.0
                cg_id, ex_sym = _resolve_cg_ticker(row, _g, ativo)

                repo.registrar_ativo(ativo, cg_id)
                pid = gerar_id()
                inserir_posicao_completa(
                    repo, produto_id, turma_id, pid, ativo,
                    cg_id, "long", data_ent, preco_ent,
                    None, None, "open", [],
                    exchange_symbol=ex_sym,
                )

                attrs = {}
                for k in ("categoria", "tipo_ico", "local", "ficha_tecnica", "disponivel_pos_ico"):
                    v = _g(row, k)
                    if v:
                        attrs[k] = v
                if attrs:
                    repo.salvar_atributos_posicao(pid, produto_id, **attrs)
                count += 1
            except Exception as exc:
                erros += 1
                print(f"  ERRO carteira '{_g(row, 'projeto') or '?'}': {exc}")
    except Exception as exc:
        print(f"  ERRO ao baixar carteira.csv: {exc}")

    # ── 3. Fechados (tab "Histórico": Categoria, Tipo, Resultado, P. Saída)
    print("\n  --- 3. Fechados (participados, status=closed) ---")
    try:
        participados_rows = download_csv(ICO_CSV_URLS["participados"])
        col_map = _build_column_map(participados_rows)

        def _g(row, *keys):
            return _get_mapped(row, col_map, *keys)

        for row in participados_rows:
            try:
                ativo = _g(row, "projeto")
                if not ativo:
                    continue
                data_ent = parse_date_br(_g(row, "data_ico")) or date.today().isoformat()
                preco_ent = parse_price(_g(row, "preco_entrada")) or 0.0
                preco_saida = parse_price(_g(row, "preco_saida") or _g(row, "preco_atual"))
                if preco_saida is None and preco_ent:
                    pnl_raw = parse_price(_g(row, "pnl_pct"))
                    if pnl_raw is not None:
                        preco_saida = preco_ent * (1 + pnl_raw / 100)
                data_saida = parse_date_br(_g(row, "data_saida") or _g(row, "data"))
                cg_id, ex_sym = _resolve_cg_ticker(row, _g, ativo)

                repo.registrar_ativo(ativo, cg_id)
                pid = gerar_id()
                inserir_posicao_completa(
                    repo, produto_id, turma_id, pid, ativo,
                    cg_id, "long", data_ent, preco_ent,
                    data_saida or data_ent, preco_saida, "closed", [],
                    exchange_symbol=ex_sym,
                )

                attrs = {}
                for k in ("categoria", "tipo_ico", "resultado"):
                    v = _g(row, k)
                    if v:
                        attrs[k] = v
                if attrs:
                    repo.salvar_atributos_posicao(pid, produto_id, **attrs)
                count += 1
            except Exception as exc:
                erros += 1
                print(f"  ERRO participado '{_g(row, 'projeto') or '?'}': {exc}")
    except Exception as exc:
        print(f"  ERRO ao baixar icos_participados.csv: {exc}")

    print(f"\n  {count} posicoes importadas (Possíveis + Abertas + Fechados)")
    if erros:
        print(f"  {erros} erros")


# ═══════════════════════ Main ═══════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Migra posicoes de CS e ICOs do CDN para o banco de dados",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_true",
                       help="Apenas banco local (SQLite)")
    group.add_argument("--cloud", action="store_true",
                       help="Apenas banco na nuvem (Supabase)")
    group.add_argument("--both", action="store_true",
                       help="Ambos (padrao se SUPABASE_DB_URL estiver configurado)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Pular confirmacao")
    args = parser.parse_args()

    targets: list[tuple[str, bool]] = []

    if args.local:
        targets.append(("LOCAL (SQLite)", True))
    elif args.cloud:
        targets.append(("NUVEM (Supabase)", False))
    else:
        targets.append(("LOCAL (SQLite)", True))
        db_url = (os.getenv("SUPABASE_DB_URL") or "").strip()
        if db_url:
            targets.append(("NUVEM (Supabase)", False))
        else:
            print(
                "AVISO: SUPABASE_DB_URL nao configurado; "
                "apenas banco local sera atualizado.\n"
            )

    print("Este script ira:")
    print("  1. DELETAR todas as posicoes de Crypto Signals e ICOs")
    print("  2. Reimportar posicoes a partir dos CSVs do CDN")
    print(f"  Alvos: {', '.join(t[0] for t in targets)}\n")

    if not args.yes:
        resp = input("Deseja continuar? (s/N) ").strip().lower()
        if resp != "s":
            print("Cancelado.")
            sys.exit(0)

    for label, is_local in targets:
        print(f"\n{'#' * 60}")
        print(f"  Alvo: {label}")
        print("#" * 60)

        if is_local:
            repo = SQLiteRepo(db_path=str(_DEFAULT_SQLITE_PATH))
        else:
            repo = SQLiteRepo()

        importar_crypto_signals(repo)
        importar_icos(repo)

        print(f"\n  Migracao {label} concluida!")

    print(f"\n{'#' * 60}")
    print("  MIGRACAO COMPLETA")
    print("#" * 60)


if __name__ == "__main__":
    main()
