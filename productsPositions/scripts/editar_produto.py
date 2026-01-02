"""
Script para editar um produto existente
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from utils.cli_utils import obter_input, validar_data, imprimir_titulo, imprimir_secao


def main():
    imprimir_titulo("EDITAR PRODUTO")

    repo = SQLiteRepo()

    # Listar produtos disponíveis
    produtos = repo.listar_produtos()
    if not produtos:
        print("Nenhum produto encontrado.")
        return

    print("Produtos disponíveis:")
    for p in produtos:
        capital = p.get('capital_inicial', 0) or 0
        print(f"  ID: {p['id']} - {p['nome']} ({p['tipo']}) - Capital: ${capital:,.2f}")

    produto_id = obter_input("\nID do produto a editar: ", tipo=int, obrigatorio=True)

    # Verificar se produto existe
    produto = repo.carregar_produto(produto_id)
    if not produto:
        print(f"Produto com ID {produto_id} nao encontrado!")
        return

    imprimir_secao("PRODUTO SELECIONADO")
    print(f"  Nome: {produto['nome']}")
    print(f"  Tipo: {produto['tipo']}")
    print(f"  Data inicio: {produto['data_inicio']}")
    capital = produto.get('capital_inicial', 0) or 0
    print(f"  Capital inicial: ${capital:,.2f}")

    # Menu de edição
    while True:
        imprimir_secao("CAMPOS EDITAVEIS")
        print("1. Nome")
        print("2. Tipo")
        print("3. Data de inicio")
        print("4. Capital inicial")
        print("0. Voltar")

        opcao = obter_input("\nCampo a editar: ", opcoes=["0", "1", "2", "3", "4"], obrigatorio=True)

        if opcao == "0":
            break

        elif opcao == "1":
            novo_nome = obter_input(f"Novo nome [{produto['nome']}]: ", obrigatorio=False)
            if novo_nome:
                repo.atualizar_produto(produto_id, nome=novo_nome)
                produto['nome'] = novo_nome
                print(f"Nome atualizado para: {novo_nome}")

        elif opcao == "2":
            tipos_disponiveis = repo.listar_tipos()
            print(f"Tipos disponiveis: {', '.join(tipos_disponiveis)}")
            novo_tipo = obter_input(
                f"Novo tipo [{produto['tipo']}]: ",
                opcoes=tipos_disponiveis,
                obrigatorio=False
            )
            if novo_tipo:
                repo.atualizar_produto(produto_id, tipo=novo_tipo)
                produto['tipo'] = novo_tipo
                print(f"Tipo atualizado para: {novo_tipo}")

        elif opcao == "3":
            nova_data = obter_input(
                f"Nova data de inicio [{produto['data_inicio']}] (YYYY-MM-DD): ",
                obrigatorio=False
            )
            if nova_data:
                nova_data = validar_data(nova_data, "Data de inicio")
                repo.atualizar_produto(produto_id, data_inicio=nova_data)
                produto['data_inicio'] = nova_data
                print(f"Data de inicio atualizada para: {nova_data}")

        elif opcao == "4":
            novo_capital = obter_input(
                f"Novo capital inicial [{capital:,.2f}]: ",
                tipo=float,
                obrigatorio=False
            )
            if novo_capital is not None:
                repo.atualizar_produto(produto_id, capital_inicial=novo_capital)
                produto['capital_inicial'] = novo_capital
                capital = novo_capital
                print(f"Capital inicial atualizado para: ${novo_capital:,.2f}")

    print("\nEdicao concluida.")


if __name__ == "__main__":
    main()
