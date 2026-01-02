"""
Script para editar valores de posições existentes em produtos.

Permite escolher o produto, listar posições e editar campos específicos
de acordo com o tipo do produto (ex.: Spot pode editar quantidade).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo  # noqa: E402
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao, validar_data  # noqa: E402
from utils.produto_utils import obter_campos_editaveis_produto  # noqa: E402


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


def _editar_posicao_dinamica(repo: SQLiteRepo, produto_id: int, posicao_id: int):
    """
    Edição dinâmica que se adapta aos campos necessários de cada produto.
    """
    imprimir_secao("EDIÇÃO DE POSIÇÃO")

    # Obter campos editáveis para este produto específico
    campos = obter_campos_editaveis_produto(produto_id)

    while True:
        print("\nCampos disponíveis para edição:")
        for idx, campo in enumerate(campos, start=1):
            obrigatorio_str = " (obrigatório)" if campo.get('obrigatorio') else " (opcional)"
            print(f"{idx}. {campo['label']}{obrigatorio_str}")
        print("0. Voltar")

        opcoes_validas = [str(i) for i in range(0, len(campos) + 1)]
        op = obter_input("\nEscolha o campo para editar: ", opcoes=opcoes_validas, obrigatorio=True)

        if op == "0":
            break

        try:
            idx = int(op) - 1
            if idx < 0 or idx >= len(campos):
                print("❌ Opção inválida.")
                continue

            campo = campos[idx]
            nome_campo = campo['nome']
            is_atributo = campo.get('atributo', False)

            # Editar campo básico da posição (não é atributo dinâmico)
            if not is_atributo:
                if nome_campo == 'ativo':
                    novo_valor = obter_input(f"Novo {campo['label']}: ", obrigatorio=campo.get('obrigatorio', True)).upper()
                    repo.atualizar_posicao(posicao_id, ativo=novo_valor)

                elif nome_campo == 'side':
                    novo_valor = obter_input(f"Novo {campo['label']} ({'/'.join(campo.get('opcoes', []))}): ",
                                            opcoes=campo.get('opcoes', []), obrigatorio=campo.get('obrigatorio', True))
                    repo.atualizar_posicao(posicao_id, side=novo_valor)

                elif nome_campo == 'coingecko_id':
                    novo_valor = obter_input(f"Novo {campo['label']}: ", obrigatorio=False)
                    repo.atualizar_posicao(posicao_id, coingecko_id=novo_valor if novo_valor else None)

                elif nome_campo in ['data_entrada', 'data_saida']:
                    novo_valor = obter_input(f"Nova {campo['label']} (YYYY-MM-DD): ", obrigatorio=campo.get('obrigatorio', True))
                    novo_valor = validar_data(novo_valor, campo['label'])
                    repo.atualizar_posicao(posicao_id, **{nome_campo: novo_valor})

                elif nome_campo in ['preco_entrada', 'preco_saida']:
                    novo_valor = obter_input(f"Novo {campo['label']}: ", tipo=float, obrigatorio=campo.get('obrigatorio', True))
                    repo.atualizar_posicao(posicao_id, **{nome_campo: novo_valor})

                    # Se for preco_entrada e houver quantidade, recalcular preco_entrada_total
                    if nome_campo == 'preco_entrada':
                        attrs = repo.carregar_atributos_posicao(posicao_id)
                        if attrs and attrs.get("quantidade") is not None:
                            quantidade = attrs["quantidade"]
                            preco_total = quantidade * novo_valor
                            repo.salvar_atributos_posicao(
                                posicao_id,
                                produto_id,
                                quantidade=quantidade,
                                preco_entrada_total=preco_total
                            )
                            print(f"   Preço total recalculado: ${preco_total:,.2f}")

                elif nome_campo == 'status':
                    novo_valor = obter_input(f"Novo {campo['label']} ({'/'.join(campo.get('opcoes', []))}): ",
                                            opcoes=campo.get('opcoes', []), obrigatorio=campo.get('obrigatorio', True))
                    repo.atualizar_posicao(posicao_id, status=novo_valor)

                print(f"✅ {campo['label']} atualizado com sucesso.")

            # Editar atributos específicos do produto (dinâmicos)
            else:
                if nome_campo == 'quantidade':
                    nova_qtd = obter_input(f"Nova {campo['label']}: ", tipo=float, obrigatorio=campo.get('obrigatorio', True))
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
                        preco_entrada_total=preco_total
                    )
                    if preco_total is not None:
                        print(f"✅ {campo['label']} atualizada. Preço total: ${preco_total:,.2f}")
                    else:
                        print(f"✅ {campo['label']} atualizada.")

                elif nome_campo == 'preco_entrada_total':
                    # Preço total é calculado automaticamente, não deve ser editado diretamente
                    print("ℹ️  Preço de entrada total é calculado automaticamente (quantidade × preço de entrada).")
                    print("   Edite a quantidade ou o preço de entrada para alterar o total.")

                else:
                    # Outros atributos dinâmicos
                    if campo['tipo'] == float:
                        novo_valor = obter_input(f"Novo {campo['label']}: ", tipo=float, obrigatorio=campo.get('obrigatorio', False))
                    else:
                        novo_valor = obter_input(f"Novo {campo['label']}: ", obrigatorio=campo.get('obrigatorio', False))

                    # Salvar apenas o atributo específico (o repo já faz UPSERT com COALESCE)
                    repo.salvar_atributos_posicao(
                        posicao_id,
                        produto_id,
                        **{nome_campo: novo_valor if novo_valor else None}
                    )
                    print(f"✅ {campo['label']} atualizado com sucesso.")

        except ValueError as e:
            print(f"❌ Erro ao processar valor: {e}")
        except Exception as e:
            print(f"❌ Erro ao atualizar: {e}")

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

    df_pos = _listar_posicoes_do_produto(repo, produto_id)
    if df_pos is None or df_pos.empty:
        return

    posicao_id = obter_input("ID da posição a editar: ", tipo=int, obrigatorio=True)

    # Verificar se posição pertence ao produto
    if posicao_id not in df_pos["id"].astype(int).tolist():
        print(f"❌ Posição {posicao_id} não pertence ao produto {produto_id}.")
        return

    # Usar função dinâmica que se adapta ao produto
    _editar_posicao_dinamica(repo, produto_id, posicao_id)

    print("\n✅ Edição concluída.")


if __name__ == "__main__":
    main()


