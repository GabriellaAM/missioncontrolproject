"""
Script para consultar e visualizar dados do sistema
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.parquet_repo import ParquetRepo
from analytics.queries import (
    posicoes_abertas, posicoes_fechadas, alocacoes_do_produto,
    resumo_alocacoes, carteira_do_produto, resumo_completo_produto,
    valores_da_posicao, valores_do_ativo
)
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao

def main():
    imprimir_titulo("CONSULTAR DADOS")
    
    repo = ParquetRepo()
    
    # Listar produtos disponíveis
    produtos = repo.listar_produtos()
    if not produtos:
        print("❌ Nenhum produto encontrado.")
        return
    
    print("Produtos disponíveis:")
    for produto_dict in produtos:
        print(f"  ID: {produto_dict['id']} - {produto_dict['nome']} ({produto_dict['tipo']})")
    
    produto_id = obter_input("\nID do produto: ", tipo=int, obrigatorio=True)
    
    # Verificar se produto existe
    produto = repo.carregar_produto(produto_id)
    if not produto:
        print(f"❌ Produto com ID {produto_id} não encontrado!")
        return
    
    imprimir_secao("OPÇÕES DE CONSULTA")
    print("1. Posições abertas")
    print("2. Posições fechadas (histórico)")
    print("3. Alocações do produto")
    print("4. Carteira do produto")
    print("5. Resumo completo do produto")
    print("6. Valores diários de uma posição")
    print("7. Valores diários de um ativo")
    print("8. Tudo")
    
    opcao = obter_input("\nEscolha uma opção (1-8): ", opcoes=["1", "2", "3", "4", "5", "6", "7", "8"], obrigatorio=True)
    
    if opcao == "1" or opcao == "8":
        imprimir_secao("POSIÇÕES ABERTAS")
        print(posicoes_abertas(produto_id))
    
    if opcao == "2" or opcao == "8":
        imprimir_secao("POSIÇÕES FECHADAS (HISTÓRICO)")
        print(posicoes_fechadas(produto_id))
    
    if opcao == "3" or opcao == "8":
        imprimir_secao("ALOCAÇÕES DO PRODUTO")
        print(alocacoes_do_produto(produto_id))
        print("\nRESUMO DE ALOCAÇÕES:")
        print(resumo_alocacoes(produto_id))
    
    if opcao == "4" or opcao == "8":
        imprimir_secao("CARTEIRA DO PRODUTO")
        print(carteira_do_produto(produto_id))
    
    if opcao == "5" or opcao == "8":
        imprimir_secao("RESUMO COMPLETO DO PRODUTO")
        print(resumo_completo_produto(produto_id))
    
    if opcao == "6":
        imprimir_secao("VALORES DIÁRIOS DA POSIÇÃO")
        posicao_id = obter_input("ID da posição: ", tipo=int, obrigatorio=True)
        print(valores_da_posicao(posicao_id))
    
    if opcao == "7":
        imprimir_secao("VALORES DIÁRIOS DO ATIVO")
        ativo = obter_input("Nome do ativo (ex: BTC): ", obrigatorio=True).upper()
        print(valores_do_ativo(ativo))
    
    print("\n" + "=" * 60)
    print("  CONSULTA CONCLUÍDA!")
    print("=" * 60)

if __name__ == "__main__":
    main()

