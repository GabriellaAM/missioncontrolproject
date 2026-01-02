"""
Script para gerenciar atributos (colunas) de posições.
Permite listar, deletar colunas órfãs e editar configurações de atributos.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao


def listar_colunas():
    """Lista todas as colunas de atributos e seu status de uso"""
    repo = SQLiteRepo()

    imprimir_secao("COLUNAS DE ATRIBUTOS")

    colunas = repo.listar_colunas_atributos()
    orfas = repo.listar_colunas_orfas()

    print(f"Total de colunas: {len(colunas)}")
    print(f"Colunas órfãs (não usadas): {len(orfas)}\n")

    for coluna in colunas:
        status = " [ORFA]" if coluna in orfas else ""
        print(f"  - {coluna}{status}")

    if orfas:
        print(f"\nColunas órfãs podem ser removidas com a opção 'Limpar colunas órfãs'")


def listar_uso_colunas():
    """Lista quais produtos usam cada coluna"""
    repo = SQLiteRepo()

    imprimir_secao("USO DE COLUNAS POR PRODUTO")

    colunas = repo.listar_colunas_atributos()
    produtos = repo.listar_produtos()

    for coluna in colunas:
        produtos_usando = []
        for p in produtos:
            configs = repo.carregar_atributos_config(p['id'])
            if any(c['atributo_nome'] == coluna for c in configs):
                produtos_usando.append(p['nome'])

        if produtos_usando:
            print(f"  {coluna}: {', '.join(produtos_usando)}")
        else:
            print(f"  {coluna}: [NENHUM PRODUTO - ORFA]")


def limpar_orfas():
    """Remove colunas órfãs"""
    repo = SQLiteRepo()

    orfas = repo.listar_colunas_orfas()

    if not orfas:
        print("Nenhuma coluna órfã encontrada.")
        return

    imprimir_secao("COLUNAS ORFAS")
    print(f"Colunas que serão removidas: {', '.join(orfas)}\n")

    confirmar = obter_input("Confirmar remoção? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True)

    if confirmar.lower() == "s":
        removidas = repo.limpar_colunas_orfas()
        if removidas:
            print(f"\n Removidas {len(removidas)} coluna(s): {', '.join(removidas)}")
        else:
            print("\nNenhuma coluna foi removida.")
    else:
        print("\nOperação cancelada.")


def deletar_coluna():
    """Deleta uma coluna específica"""
    repo = SQLiteRepo()

    colunas = repo.listar_colunas_atributos()
    orfas = repo.listar_colunas_orfas()

    if not colunas:
        print("Nenhuma coluna de atributo encontrada.")
        return

    imprimir_secao("DELETAR COLUNA")

    print("Colunas disponíveis:")
    for i, coluna in enumerate(colunas, 1):
        status = " [ORFA - pode deletar]" if coluna in orfas else " [EM USO]"
        print(f"  {i}. {coluna}{status}")

    print("  0. Cancelar")

    opcao = obter_input("\nEscolha a coluna: ", tipo=int, obrigatorio=True)

    if opcao == 0:
        return

    if opcao < 1 or opcao > len(colunas):
        print("Opção inválida.")
        return

    coluna = colunas[opcao - 1]

    if coluna not in orfas:
        print(f"\nColuna '{coluna}' está em uso por produtos.")
        print("Remova primeiro a configuração dos produtos que usam essa coluna.")
        return

    confirmar = obter_input(f"\nConfirmar exclusão de '{coluna}'? (s/n): ",
                           opcoes=["s", "n", "S", "N"], obrigatorio=True)

    if confirmar.lower() == "s":
        try:
            repo.deletar_coluna_atributo(coluna)
            print(f"\nColuna '{coluna}' removida com sucesso.")
        except ValueError as e:
            print(f"\nErro: {e}")
    else:
        print("\nOperação cancelada.")


def editar_config_produto():
    """Edita configuração de atributos de um produto"""
    repo = SQLiteRepo()

    produtos = repo.listar_produtos()
    if not produtos:
        print("Nenhum produto encontrado.")
        return

    imprimir_secao("EDITAR CONFIG DE ATRIBUTOS")

    print("Produtos disponíveis:")
    for p in produtos:
        print(f"  ID: {p['id']} - {p['nome']}")

    produto_id = obter_input("\nID do produto: ", tipo=int, obrigatorio=True)

    produto = repo.carregar_produto(produto_id)
    if not produto:
        print(f"Produto {produto_id} não encontrado.")
        return

    configs = repo.carregar_atributos_config(produto_id)

    if not configs:
        print(f"\nProduto '{produto['nome']}' não possui atributos configurados.")
        adicionar = obter_input("Deseja adicionar atributos? (s/n): ",
                               opcoes=["s", "n", "S", "N"], obrigatorio=True)
        if adicionar.lower() == "s":
            _adicionar_atributo(repo, produto_id)
        return

    while True:
        imprimir_secao(f"ATRIBUTOS DE {produto['nome'].upper()}")

        for i, c in enumerate(configs, 1):
            obrig = "[obrig]" if c['obrigatorio'] else ""
            print(f"  {i}. {c['atributo_label'] or c['atributo_nome']} "
                  f"({c['atributo_tipo']}) {obrig}")

        print(f"\n  A. Adicionar atributo")
        print(f"  R. Remover atributo")
        print(f"  0. Voltar")

        opcao = obter_input("\nEscolha (número para editar): ", obrigatorio=True)

        if opcao == "0":
            break
        elif opcao.upper() == "A":
            _adicionar_atributo(repo, produto_id)
            configs = repo.carregar_atributos_config(produto_id)
        elif opcao.upper() == "R":
            _remover_atributo(repo, produto_id, configs)
            configs = repo.carregar_atributos_config(produto_id)
        else:
            try:
                idx = int(opcao) - 1
                if 0 <= idx < len(configs):
                    _editar_atributo(repo, produto_id, configs[idx])
                    configs = repo.carregar_atributos_config(produto_id)
                else:
                    print("Opção inválida.")
            except ValueError:
                print("Opção inválida.")


def _adicionar_atributo(repo, produto_id):
    """Adiciona novo atributo ao produto"""
    print("\nColunas existentes:", repo.listar_colunas_atributos())

    nome = obter_input("\nNome do atributo: ", obrigatorio=True)
    nome = nome.lower().strip().replace(' ', '_')

    tipo = obter_input("Tipo (text/float/int/date) [text]: ",
                      opcoes=["text", "float", "int", "date", ""], obrigatorio=False)
    if not tipo:
        tipo = "text"

    label = obter_input(f"Label [{nome.replace('_', ' ').title()}]: ", obrigatorio=False)

    obrigatorio = obter_input("Obrigatório? (s/n) [n]: ",
                             opcoes=["s", "n", "S", "N", ""], obrigatorio=False)
    obrigatorio = obrigatorio.lower() == "s" if obrigatorio else False

    repo.adicionar_atributo_config(produto_id, nome, tipo, label if label else None, obrigatorio)
    print(f"\nAtributo '{nome}' adicionado.")


def _remover_atributo(repo, produto_id, configs):
    """Remove atributo do produto"""
    print("\nQual atributo remover?")
    for i, c in enumerate(configs, 1):
        print(f"  {i}. {c['atributo_nome']}")

    opcao = obter_input("\nEscolha: ", tipo=int, obrigatorio=True)

    if opcao < 1 or opcao > len(configs):
        print("Opção inválida.")
        return

    config = configs[opcao - 1]
    confirmar = obter_input(f"Confirmar remoção de '{config['atributo_nome']}'? (s/n): ",
                           opcoes=["s", "n", "S", "N"], obrigatorio=True)

    if confirmar.lower() == "s":
        repo.remover_atributo_config(produto_id, config['atributo_nome'])
        print(f"\nAtributo '{config['atributo_nome']}' removido do produto.")
        print("Nota: A coluna permanece na tabela (pode ser usada por outros produtos).")


def _editar_atributo(repo, produto_id, config):
    """Edita um atributo específico"""
    print(f"\nEditando: {config['atributo_nome']}")
    print(f"  Label atual: {config['atributo_label'] or '(nenhum)'}")
    print(f"  Tipo atual: {config['atributo_tipo']}")
    print(f"  Obrigatório: {'Sim' if config['obrigatorio'] else 'Não'}")

    novo_label = obter_input(f"\nNovo label [{config['atributo_label'] or config['atributo_nome']}]: ",
                            obrigatorio=False)
    novo_tipo = obter_input(f"Novo tipo [{config['atributo_tipo']}]: ",
                           opcoes=["text", "float", "int", "date", ""], obrigatorio=False)
    novo_obrig = obter_input(f"Obrigatório? (s/n) [{'s' if config['obrigatorio'] else 'n'}]: ",
                            opcoes=["s", "n", "S", "N", ""], obrigatorio=False)

    # Preparar valores para atualização
    kwargs = {}
    if novo_label:
        kwargs['novo_label'] = novo_label
    if novo_tipo:
        kwargs['novo_tipo'] = novo_tipo
    if novo_obrig:
        kwargs['novo_obrigatorio'] = novo_obrig.lower() == "s"

    if kwargs:
        repo.editar_atributo_config(produto_id, config['atributo_nome'], **kwargs)
        print("\nAtributo atualizado.")
    else:
        print("\nNenhuma alteração feita.")


def main():
    while True:
        imprimir_titulo("GERENCIAR ATRIBUTOS")

        print("1. Listar colunas de atributos")
        print("2. Ver uso de colunas por produto")
        print("3. Limpar colunas órfãs")
        print("4. Deletar coluna específica")
        print("5. Editar atributos de um produto")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3", "4", "5"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            listar_colunas()
        elif opcao == "2":
            listar_uso_colunas()
        elif opcao == "3":
            limpar_orfas()
        elif opcao == "4":
            deletar_coluna()
        elif opcao == "5":
            editar_config_produto()

        print()


if __name__ == "__main__":
    main()
