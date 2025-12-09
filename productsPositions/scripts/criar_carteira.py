"""
Script para criar ou atualizar a carteira de um produto
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.carteira_service import CarteiraService
from storage.parquet_repo import ParquetRepo
from analytics.queries import carteira_do_produto, resumo_completo_produto
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao

def main():
    imprimir_titulo("CRIAR/ATUALIZAR CARTEIRA")
    
    repo = ParquetRepo()
    
    # Listar produtos disponíveis
    produtos = repo.listar_produtos()
    if not produtos:
        print("❌ Nenhum produto encontrado. Crie um produto primeiro usando criar_produto.py")
        return None
    
    print("Produtos disponíveis:")
    for produto_dict in produtos:
        print(f"  ID: {produto_dict['id']} - {produto_dict['nome']} ({produto_dict['tipo']})")
    
    produto_id = obter_input("\nID do produto: ", tipo=int, obrigatorio=True)
    
    # Verificar se produto existe
    produto = repo.carregar_produto(produto_id)
    if not produto:
        print(f"❌ Produto com ID {produto_id} não encontrado!")
        return None
    
    imprimir_secao("OPÇÕES DE CRIAÇÃO")
    print("1. Criar carteira manualmente")
    print("2. Preencher carteira automaticamente (requer preço atual)")
    print("3. Criar carteira com capital inicial e recalcular tudo")
    
    opcao = obter_input("\nEscolha uma opção (1/2/3): ", opcoes=["1", "2", "3"], obrigatorio=True)
    
    if opcao == "1":
        imprimir_secao("CRIAR CARTEIRA MANUAL")
        valor_disponivel = obter_input("Valor disponível: ", tipo=float, obrigatorio=True)
        valor_investido = obter_input("Valor investido: ", tipo=float, obrigatorio=True)
        pnl_nao_realizado = obter_input("PnL não realizado: ", tipo=float, obrigatorio=True)
        
        carteira = CarteiraService.criar_carteira(
            produto_id=produto_id,
            valor_disponivel=valor_disponivel,
            valor_investido=valor_investido,
            pnl_nao_realizado=pnl_nao_realizado
        )
        repo.salvar_carteira(carteira)
        print("✅ Carteira criada manualmente")
    
    elif opcao == "2":
        imprimir_secao("PREENCHER CARTEIRA AUTOMATICAMENTE")
        
        # Listar posições abertas
        posicoes = repo.carregar_posicoes_abertas(produto_id)
        if posicoes.empty:
            print("❌ Nenhuma posição aberta encontrada!")
            return None
        
        print("Posições abertas:")
        preco_atual_por_posicao = {}
        for _, row in posicoes.iterrows():
            preco = obter_input(f"Preço atual do {row['ativo']} (ID {row['id']}): ", tipo=float, obrigatorio=True)
            preco_atual_por_posicao[row['id']] = preco
        
        carteira = CarteiraService.preencher_carteira(
            produto_id=produto_id,
            preco_atual_por_posicao=preco_atual_por_posicao
        )
        repo.salvar_carteira(carteira)
        print("✅ Carteira preenchida automaticamente")
    
    elif opcao == "3":
        imprimir_secao("CRIAR CARTEIRA COM CAPITAL INICIAL")
        
        capital_inicial = obter_input("Capital inicial: ", tipo=float, obrigatorio=True)
        
        # Listar posições abertas
        posicoes = repo.carregar_posicoes_abertas(produto_id)
        if posicoes.empty:
            print("❌ Nenhuma posição aberta encontrada!")
            return None
        
        print("\nPosições abertas:")
        preco_atual_por_posicao = {}
        for _, row in posicoes.iterrows():
            preco = obter_input(f"Preço atual do {row['ativo']} (ID {row['id']}): ", tipo=float, obrigatorio=True)
            preco_atual_por_posicao[row['id']] = preco
        
        carteira = CarteiraService.atualizar_carteira_com_capital_inicial(
            produto_id=produto_id,
            capital_inicial=capital_inicial,
            preco_atual_por_posicao=preco_atual_por_posicao
        )
        repo.salvar_carteira(carteira)
        print("✅ Carteira criada com capital inicial")
    
    # Mostrar resumo
    mostrar_resumo = obter_input("\nDeseja ver o resumo da carteira? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True).lower()
    if mostrar_resumo == "s":
        print("\nCARTEIRA DO PRODUTO:\n", carteira_do_produto(produto_id))
        print("\nRESUMO COMPLETO DO PRODUTO:\n", resumo_completo_produto(produto_id))
    
    return produto_id

if __name__ == "__main__":
    main()

