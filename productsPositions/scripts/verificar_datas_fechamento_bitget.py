"""
Verifica se as datas de fechamento (data_saida) das posições fechadas dos produtos
Soros Perpétuos e Memebot Perpétuos no banco batem com as datas retornadas pela API Bitget.

Uso (na raiz do projeto ou em productsPositions):
  python -m productsPositions.scripts.verificar_datas_fechamento_bitget
  # ou
  cd productsPositions && python scripts/verificar_datas_fechamento_bitget.py
"""
import sys
from pathlib import Path

# Raiz do pacote productsPositions (contém storage/ e services/)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
from storage.sqlite_repo import get_repo
from services.bitget_service import (
    get_bitget_credentials,
    fetch_perpetual_history,
    _timestamp_to_date_str,
)


def _normalize_exchange_symbol(s: str) -> str:
    """Garante formato ATIVOUSDT para comparação com a API."""
    if not s or pd.isna(s):
        return ""
    s = str(s).strip().upper()
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s


def main():
    repo = get_repo()
    produtos = repo.listar_produtos()
    if not produtos:
        print("Nenhum produto encontrado no banco.")
        return

    # Filtrar Soros Perpétuos e Memebot Perpétuos (nome contém e é tipo perpétuo)
    def is_perp(nome, tipo):
        nome = (nome or "").lower()
        tipo = (tipo or "").lower()
        return "perp" in tipo or "perpetuo" in tipo or "pérpetuo" in nome or "perpétuo" in nome

    candidatos = []
    for row in produtos:
        nome = row.get("nome") or row.get("name")
        tipo = row.get("tipo") or row.get("type")
        if not is_perp(nome or "", tipo or ""):
            continue
        if "soros" in (nome or "").lower() or "memebot" in (nome or "").lower():
            candidatos.append({"id": row.get("id"), "nome": nome, "tipo": tipo})

    if not candidatos:
        print("Nenhum produto 'Soros Perpétuos' ou 'Memebot Perpétuos' encontrado.")
        return

    print("=" * 80)
    print("VERIFICAÇÃO: datas de fechamento (SQLite) x API Bitget (history-position)")
    print("=" * 80)

    totais_ok = 0
    totais_divergente = 0
    totais_sem_na_api = 0
    totais_sem_data_no_bd = 0

    for prod in candidatos:
        produto_id = prod["id"]
        nome = prod["nome"]
        print(f"\n--- Produto: {nome} (id={produto_id}) ---")

        creds = get_bitget_credentials(nome)
        if not creds:
            print("  [AVISO] Sem credenciais Bitget no .env — pulando.")
            continue

        df_fechadas = repo.carregar_posicoes_fechadas(produto_id)
        if df_fechadas is None or df_fechadas.empty:
            print("  Nenhuma posição fechada no banco.")
            continue

        try:
            history = fetch_perpetual_history(creds, limit=500, max_pages=15)
        except Exception as e:
            print(f"  [ERRO] Falha ao buscar histórico na API: {e}")
            continue

        # Índice (exchange_symbol_normalized, side) -> lista de entradas (mais recente primeiro na lista)
        from collections import defaultdict
        api_by_key = defaultdict(list)
        for h in history:
            key = (_normalize_exchange_symbol(h.get("exchange_symbol", "")), (h.get("side") or "").lower())
            api_by_key[key].append(h)

        # Para cada posição fechada no BD, pegar a correspondente na API (mesmo par ativo+side)
        # A API pode ter várias entradas (aberturas/fechamentos); queremos a que fecha esta posição.
        # Heurística: mesma data_entrada ou a entrada mais recente no histórico que tenha close_time.
        results = []
        for _, pos in df_fechadas.iterrows():
            ativo = pos.get("ativo") or ""
            side = (pos.get("side") or "long").lower()
            data_saida_bd = pos.get("data_saida")
            if pd.isna(data_saida_bd) or not str(data_saida_bd).strip():
                data_saida_bd_str = ""
                totais_sem_data_no_bd += 1
            else:
                data_saida_bd_str = str(data_saida_bd).strip()[:10]

            exchange_symbol = pos.get("exchange_symbol") or (ativo + "USDT" if ativo else "")
            key = (_normalize_exchange_symbol(exchange_symbol), side)
            candidatos_api = api_by_key.get(key, [])

            if not candidatos_api:
                results.append({
                    "ativo": ativo,
                    "side": side,
                    "data_saida_bd": data_saida_bd_str or "(vazio)",
                    "data_saida_api": "—",
                    "status": "NÃO ENCONTRADO NA API",
                })
                totais_sem_na_api += 1
                continue

            # Para cada entrada no histórico (mesmo ativo+side), obter data de fechamento da API
            datas_api = []
            for h in candidatos_api:
                close_ts = h.get("close_time") or h.get("uTime") or h.get("utime")
                if close_ts is not None:
                    data_api = _timestamp_to_date_str(close_ts)
                    if data_api:
                        datas_api.append(data_api)

            if not datas_api:
                results.append({
                    "ativo": ativo,
                    "side": side,
                    "data_saida_bd": data_saida_bd_str or "(vazio)",
                    "data_saida_api": "—",
                    "status": "NA API SEM close_time",
                })
                totais_sem_na_api += 1
                continue

            # Verificar se alguma data de fechamento na API bate com a do BD
            if data_saida_bd_str and data_saida_bd_str in datas_api:
                data_saida_api = data_saida_bd_str
                status = "OK"
                totais_ok += 1
            else:
                data_saida_api = max(datas_api)  # mostrar a mais recente da API para comparação
                status = "DIVERGENTE"
                totais_divergente += 1

            results.append({
                "ativo": ativo,
                "side": side,
                "data_saida_bd": data_saida_bd_str or "(vazio)",
                "data_saida_api": data_saida_api,
                "status": status,
            })

        # Ordenar: divergentes primeiro, depois não encontrados, depois OK
        def _ord(r):
            if r["status"] == "DIVERGENTE":
                return 0
            if "NÃO ENCONTRADO" in r["status"] or "SEM close_time" in r["status"]:
                return 1
            return 2
        results.sort(key=_ord)

        for r in results:
            print(f"  {r['ativo']:12} {r['side']:6}  BD: {r['data_saida_bd']:12}  API: {r['data_saida_api']:12}  {r['status']}")
        print(f"  Total fechadas no BD: {len(df_fechadas)}  |  Histórico API: {len(history)} posições")

    print("\n" + "=" * 80)
    print("RESUMO")
    print("  OK (datas batem):        ", totais_ok)
    print("  DIVERGENTE:             ", totais_divergente)
    print("  Não encontrado na API:   ", totais_sem_na_api)
    print("  Sem data_saida no BD:    ", totais_sem_data_no_bd)
    print("=" * 80)


if __name__ == "__main__":
    main()
