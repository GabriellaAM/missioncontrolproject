"""
Menu principal do sistema de gestão de posições e produtos
Oferece acesso a todas as funcionalidades do sistema através de submenus organizados
"""
import sys
from pathlib import Path

# Adicionar pasta pai ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Adicionar pasta scripts ao path para imports locais
sys.path.insert(0, str(Path(__file__).parent))

from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao


def menu_produtos():
    """Submenu para gerenciamento de produtos"""
    while True:
        imprimir_secao("PRODUTOS")
        print("1. Criar produto")
        print("2. Editar produto")
        print("3. Deletar produto")
        print("4. Gerenciar atributos")
        print("5. Gerenciar visualizações")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3", "4", "5"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            from criar_produto import main as criar_produto_main
            criar_produto_main()
        elif opcao == "2":
            from editar_produto import main as editar_produto_main
            editar_produto_main()
        elif opcao == "3":
            from deletar_produto import main as deletar_produto_main
            deletar_produto_main()
        elif opcao == "4":
            from gerenciar_atributos import main as gerenciar_atributos_main
            gerenciar_atributos_main()
        elif opcao == "5":
            from gerenciar_visualizacoes import main as gerenciar_visualizacoes_main
            gerenciar_visualizacoes_main()


def menu_posicoes():
    """Submenu para gerenciamento de posições"""
    while True:
        imprimir_secao("POSIÇÕES")
        print("1. Criar posição")
        print("2. Editar posição")
        print("3. Deletar posição")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            from criar_posicao import main as criar_posicao_main
            criar_posicao_main()
        elif opcao == "2":
            from editar_posicao import main as editar_posicao_main
            editar_posicao_main()
        elif opcao == "3":
            from deletar_posicao import main as deletar_posicao_main
            deletar_posicao_main()


def menu_stops():
    """Submenu para gerenciamento de stops"""
    while True:
        imprimir_secao("STOPS")
        print("1. Adicionar stop (manual)")
        print("2. Configurar ATR Trailing Stop")
        print("3. Atualizar stops ATR (batch)")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            from adicionar_stop import main as adicionar_stop_main
            adicionar_stop_main()
        elif opcao == "2":
            from configurar_atr_stop import main as configurar_atr_stop_main
            configurar_atr_stop_main()
        elif opcao == "3":
            from atualizar_stops_atr import main as atualizar_stops_atr_main
            atualizar_stops_atr_main()


def menu_alocacoes_carteira():
    """Submenu para gerenciamento de alocações e carteira"""
    while True:
        imprimir_secao("ALOCAÇÕES E CARTEIRA")
        print("1. Criar alocação")
        print("2. Criar/atualizar carteira")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            from criar_alocacao import main as criar_alocacao_main
            criar_alocacao_main()
        elif opcao == "2":
            from criar_carteira import main as criar_carteira_main
            criar_carteira_main()


def menu_visualizacao():
    """Submenu para visualização de dados"""
    from visualizar_dados import main as visualizar_dados_main
    visualizar_dados_main()


def main():
    while True:
        imprimir_titulo("SISTEMA DE GESTÃO DE POSIÇÕES E PRODUTOS")

        print("1. Produtos")
        print("2. Posições")
        print("3. Stops")
        print("4. Alocações e Carteira")
        print("5. Visualização")
        print("0. Sair")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3", "4", "5"], obrigatorio=True)

        if opcao == "0":
            print("\n👋 Até logo!")
            break
        elif opcao == "1":
            menu_produtos()
        elif opcao == "2":
            menu_posicoes()
        elif opcao == "3":
            menu_stops()
        elif opcao == "4":
            menu_alocacoes_carteira()
        elif opcao == "5":
            menu_visualizacao()


if __name__ == "__main__":
    main()

