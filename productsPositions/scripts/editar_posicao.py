"""
Script para editar valores de posições existentes em produtos.

Permite escolher o produto, listar posições e editar campos específicos
de acordo com o tipo do produto (ex.: Spot pode editar quantidade).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo  # noqa: E402
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao  # noqa: E402


def _listar_produtos(repo: SQLiteRepo):
    produtos = repo.listar_produtos()
    if not produtos:
        print("❌ Nenhum produto encontrado.")
        return []

    print("Produtos disponíveis:")
    for p in produtos:
        print(f"  ID: {p['id']} - {p['nome']} ({p['tipo']})")
    return produtos


def _listar_posicoes_do_produto(repo: SQLiteRepo, produto_id: int):
    import sqlite3
    import pandas as pd

    conn = sqlite3.connect(repo.db_path)
    try:
        df = pd.read_sql_query(
            """
            SELECT 
                p.id,
                p.ativo,
                p.side,
                p.data_entrada,
                p.preco_entrada,
                p.data_saida,
                p.preco_saida,
                p.status
            FROM posicoes p
            WHERE p.produto_id = ?
            ORDER BY p.data_entrada
            """,
            conn,
            params=(produto_id,),
        )
    finally:
        conn.close()

    if df.empty:
        print("❌ Nenhuma posição encontrada para este produto.")
        return df

    imprimir_secao("POSIÇÕES DO PRODUTO")
    print(df.to_string(index=False))
    print()
    return df


def _editar_posicao_spot(repo: SQLiteRepo, produto_id: int, posicao_id: int):
    """
    Edição específica para produtos Spot (ex.: Soros Spot 1).

    Campos suportados:
      - ativo
      - coingecko_id
      - data_entrada
      - preco_entrada
      - quantidade
      - data_saida
      - preco_saida
      - status
    """
    imprimir_secao("EDIÇÃO DE POSIÇÃO (SPOT)")

    while True:
        print("Campos disponíveis para edição:")
        print("1. Ativo")
        print("2. CoinGecko ID")
        print("3. Data de entrada")
        print("4. Preço de entrada")
        print("5. Quantidade")
        print("6. Data de saída")
        print("7. Preço de saída")
        print("8. Status (open/closed)")
        print("0. Voltar")

        op = obter_input("\nEscolha o campo para editar: ", opcoes=[str(i) for i in range(0, 9)], obrigatorio=True)

        if op == "0":
            break

        if op == "1":
            novo_ativo = obter_input("Novo ativo (ex: BTC, ETH): ", obrigatorio=True).upper()
            repo.atualizar_posicao(posicao_id, ativo=novo_ativo)
            print("✅ Ativo atualizado com sucesso.")

        elif op == "2":
            novo_cgid = obter_input("Novo CoinGecko ID (ex: bitcoin, ethereum): ", obrigatorio=False)
            repo.atualizar_posicao(posicao_id, coingecko_id=novo_cgid if novo_cgid else None)
            print("✅ CoinGecko ID atualizado com sucesso.")

        elif op == "3":
            from utils.cli_utils import validar_data

            data = obter_input("Nova data de entrada (YYYY-MM-DD): ", obrigatorio=True)
            data = validar_data(data, "Data de entrada")
            repo.atualizar_posicao(posicao_id, data_entrada=data)
            print("✅ Data de entrada atualizada com sucesso.")

        elif op == "4":
            novo_preco = obter_input("Novo preço de entrada: ", tipo=float, obrigatorio=True)
            repo.atualizar_posicao(posicao_id, preco_entrada=novo_preco)

            # Recalcular preco_entrada_total se já existir quantidade
            attrs = repo.carregar_atributos_posicao(posicao_id)
            if attrs and attrs.get("quantidade") is not None:
                quantidade = attrs["quantidade"]
                preco_total = quantidade * novo_preco
                repo.salvar_atributos_posicao(
                    posicao_id,
                    produto_id,
                    quantidade=quantidade,
                    preco_entrada_total=preco_total,
                )
                print(f"✅ Preço de entrada atualizado e preço total recalculado: ${preco_total:,.2f}")
            else:
                print("ℹ️ Preço de entrada atualizado. (Quantidade não encontrada para recalcular total.)")

        elif op == "5":
            nova_qtd = obter_input("Nova quantidade: ", tipo=float, obrigatorio=True)
            # Obter preço de entrada atual para recalcular total
            pos = repo.carregar_posicao(posicao_id)
            preco_ent = pos.get("preco_entrada") if pos else None
            preco_total = None
            if preco_ent is not None:
                preco_total = nova_qtd * preco_ent

            repo.salvar_atributos_posicao(
                posicao_id,
                produto_id,
                quantidade=nova_qtd,
                preco_entrada_total=preco_total,
            )
            if preco_total is not None:
                print(f"✅ Quantidade atualizada. Preço total: ${preco_total:,.2f}")
            else:
                print("✅ Quantidade atualizada. (Preço de entrada não encontrado para calcular total.)")

        elif op == "6":
            from utils.cli_utils import validar_data

            data_saida = obter_input("Nova data de saída (YYYY-MM-DD): ", obrigatorio=True)
            data_saida = validar_data(data_saida, "Data de saída")
            repo.atualizar_posicao(posicao_id, data_saida=data_saida)
            print("✅ Data de saída atualizada com sucesso.")

        elif op == "7":
            novo_preco_saida = obter_input("Novo preço de saída: ", tipo=float, obrigatorio=True)
            repo.atualizar_posicao(posicao_id, preco_saida=novo_preco_saida)
            print("✅ Preço de saída atualizado com sucesso.")

        elif op == "8":
            novo_status = obter_input("Novo status (open/closed): ", opcoes=["open", "closed"], obrigatorio=True)
            repo.atualizar_posicao(posicao_id, status=novo_status)
            print("✅ Status atualizado com sucesso.")

        print()


def _editar_posicao_generica(repo: SQLiteRepo, produto_id: int, posicao_id: int):
    """
    Edição genérica para produtos não-Spot (ex.: Perpétuos, como Crypto Signals).

    Campos suportados:
      - ativo
      - side
      - coingecko_id
      - data_entrada
      - preco_entrada
      - data_saida
      - preco_saida
      - status
    """
    imprimir_secao("EDIÇÃO DE POSIÇÃO (GENÉRICA)")

    while True:
        print("Campos disponíveis para edição:")
        print("1. Ativo")
        print("2. Side (long/short)")
        print("3. CoinGecko ID")
        print("4. Data de entrada")
        print("5. Preço de entrada")
        print("6. Data de saída")
        print("7. Preço de saída")
        print("8. Status (open/closed)")
        print("0. Voltar")

        op = obter_input("\nEscolha o campo para editar: ", opcoes=[str(i) for i in range(0, 9)], obrigatorio=True)

        if op == "0":
            break

        if op == "1":
            novo_ativo = obter_input("Novo ativo (ex: BTC, ETH): ", obrigatorio=True).upper()
            repo.atualizar_posicao(posicao_id, ativo=novo_ativo)
            print("✅ Ativo atualizado com sucesso.")

        elif op == "2":
            novo_side = obter_input("Novo side (long/short): ", opcoes=["long", "short"], obrigatorio=True)
            repo.atualizar_posicao(posicao_id, side=novo_side)
            print("✅ Side atualizado com sucesso.")

        elif op == "3":
            novo_cgid = obter_input("Novo CoinGecko ID (ex: bitcoin, ethereum): ", obrigatorio=False)
            repo.atualizar_posicao(posicao_id, coingecko_id=novo_cgid if novo_cgid else None)
            print("✅ CoinGecko ID atualizado com sucesso.")

        elif op == "4":
            from utils.cli_utils import validar_data

            data = obter_input("Nova data de entrada (YYYY-MM-DD): ", obrigatorio=True)
            data = validar_data(data, "Data de entrada")
            repo.atualizar_posicao(posicao_id, data_entrada=data)
            print("✅ Data de entrada atualizada com sucesso.")

        elif op == "5":
            novo_preco = obter_input("Novo preço de entrada: ", tipo=float, obrigatorio=True)
            repo.atualizar_posicao(posicao_id, preco_entrada=novo_preco)
            print("✅ Preço de entrada atualizado com sucesso.")

        elif op == "6":
            from utils.cli_utils import validar_data

            data_saida = obter_input("Nova data de saída (YYYY-MM-DD): ", obrigatorio=True)
            data_saida = validar_data(data_saida, "Data de saída")
            repo.atualizar_posicao(posicao_id, data_saida=data_saida)
            print("✅ Data de saída atualizada com sucesso.")

        elif op == "7":
            novo_preco_saida = obter_input("Novo preço de saída: ", tipo=float, obrigatorio=True)
            repo.atualizar_posicao(posicao_id, preco_saida=novo_preco_saida)
            print("✅ Preço de saída atualizado com sucesso.")

        elif op == "8":
            novo_status = obter_input("Novo status (open/closed): ", opcoes=["open", "closed"], obrigatorio=True)
            repo.atualizar_posicao(posicao_id, status=novo_status)
            print("✅ Status atualizado com sucesso.")

        print()


def main():
    imprimir_titulo("EDITAR POSIÇÃO")

    repo = SQLiteRepo()

    produtos = _listar_produtos(repo)
    if not produtos:
        return

    produto_id = obter_input("\nID do produto: ", tipo=int, obrigatorio=True)
    produto_info = repo.carregar_produto(produto_id)
    if not produto_info:
        print(f"❌ Produto com ID {produto_id} não encontrado.")
        return

    tipo_nome = str(produto_info.get("tipo", "") or "")
    tipo_spot = "spot" in tipo_nome.lower()

    df_pos = _listar_posicoes_do_produto(repo, produto_id)
    if df_pos is None or df_pos.empty:
        return

    posicao_id = obter_input("ID da posição a editar: ", tipo=int, obrigatorio=True)

    # Verificar se posição pertence ao produto
    if posicao_id not in df_pos["id"].astype(int).tolist():
        print(f"❌ Posição {posicao_id} não pertence ao produto {produto_id}.")
        return

    if tipo_spot:
        _editar_posicao_spot(repo, produto_id, posicao_id)
    else:
        _editar_posicao_generica(repo, produto_id, posicao_id)

    print("\n✅ Edição concluída.")


if __name__ == "__main__":
    main()


