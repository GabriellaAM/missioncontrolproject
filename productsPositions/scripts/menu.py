"""
Menu principal do sistema de gestão de posições e produtos
Oferece acesso a todas as funcionalidades do sistema
"""
import sys
from pathlib import Path

# Adicionar pasta pai ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Adicionar pasta scripts ao path para imports locais
sys.path.insert(0, str(Path(__file__).parent))

from utils.cli_utils import obter_input, imprimir_titulo

def main():
    while True:
        imprimir_titulo("SISTEMA DE GESTÃO DE POSIÇÕES E PRODUTOS")
        
        print("Escolha uma opção:")
        print("1. Criar produto")
        print("2. Criar posição")
        print("3. Adicionar stop a posição (manual)")
        print("4. Configurar ATR Trailing Stop (automático)")
        print("5. Criar alocação")
        print("6. Criar/atualizar carteira")
        print("7. Visualizar dados (gráficos e tabelas)")
        print("0. Sair")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3", "4", "5", "6", "7"], obrigatorio=True)
        
        if opcao == "0":
            print("\n👋 Até logo!")
            break
        elif opcao == "1":
            from criar_produto import main as criar_produto_main
            criar_produto_main()
        elif opcao == "2":
            from criar_posicao import main as criar_posicao_main
            criar_posicao_main()
        elif opcao == "3":
            from adicionar_stop import main as adicionar_stop_main
            adicionar_stop_main()
        elif opcao == "4":
            from configurar_atr_stop import main as configurar_atr_stop_main
            configurar_atr_stop_main()
        elif opcao == "5":
            from criar_alocacao import main as criar_alocacao_main
            criar_alocacao_main()
        elif opcao == "6":
            from criar_carteira import main as criar_carteira_main
            criar_carteira_main()
        elif opcao == "7":
            from visualizar_dados import main as visualizar_dados_main
            visualizar_dados_main()
        
        continuar = obter_input("\nDeseja fazer outra operação? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True).lower()
        if continuar == "n":
            print("\n👋 Até logo!")
            break

if __name__ == "__main__":
    main()

