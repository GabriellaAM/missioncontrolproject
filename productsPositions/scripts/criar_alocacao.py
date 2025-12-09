"""
Script para criar uma alocação de capital em uma posição
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.alocacao_service import AlocacaoService
from storage.parquet_repo import ParquetRepo
from analytics.queries import alocacoes_do_produto, resumo_alocacoes
from utils.cli_utils import obter_input, validar_data, imprimir_titulo, imprimir_secao

def main():
    imprimir_titulo("CRIAR ALOCAÇÃO")
    
    repo = ParquetRepo()
    
    # Listar produtos disponíveis
    produtos = repo.listar_produtos()
    if produtos.empty:
        print("❌ Nenhum produto encontrado. Crie um produto primeiro usando criar_produto.py")
        return None
    
    print("Produtos disponíveis:")
    for _, row in produtos.iterrows():
        print(f"  ID: {row['id']} - {row['nome']} ({row['tipo']})")
    
    produto_id = obter_input("\nID do produto: ", tipo=int, obrigatorio=True)
    
    # Verificar se produto existe
    produto = repo.carregar_produto(produto_id)
    if not produto:
        print(f"❌ Produto com ID {produto_id} não encontrado!")
        return None
    
    # Listar posições do produto
    posicoes = repo.carregar_posicoes_abertas(produto_id)
    if posicoes.empty:
        print(f"❌ Nenhuma posição aberta encontrada para o produto {produto_id}")
        return None
    
    print("\nPosições abertas:")
    for _, row in posicoes.iterrows():
        print(f"  ID: {row['id']} - {row['ativo']} ({row['side']}) - Entrada: ${row['preco_entrada']:.2f}")
    
    posicao_id = obter_input("\nID da posição: ", tipo=int, obrigatorio=True)
    
    # Verificar se posição existe
    posicao = repo.carregar_posicao(posicao_id)
    if not posicao:
        print(f"❌ Posição com ID {posicao_id} não encontrada!")
        return None
    
    imprimir_secao("DADOS DA ALOCAÇÃO")
    
    percentual = obter_input("Percentual alocado (0-100): ", tipo=float, obrigatorio=True)
    while percentual < 0 or percentual > 100:
        print("⚠️  Percentual deve estar entre 0 e 100!")
        percentual = obter_input("Percentual alocado (0-100): ", tipo=float, obrigatorio=True)
    
    valor_usd = obter_input("Valor em USD [opcional]: ", tipo=float, obrigatorio=False)
    data_alocacao = obter_input("Data da alocação (YYYY-MM-DD) [opcional]: ", obrigatorio=False)
    if data_alocacao:
        data_alocacao = validar_data(data_alocacao, "Data da alocação")
    
    # Criar alocação
    alocacao = AlocacaoService.criar_alocacao(
        produto_id=produto_id,
        posicao_id=posicao_id,
        percentual=percentual,
        valor_usd=valor_usd if valor_usd else None,
        data=data_alocacao if data_alocacao else None
    )
    
    # Salvar alocação
    alocacao_id = repo.salvar_alocacao(produto_id, alocacao)
    
    print(f"\n✅ Alocação criada com ID: {alocacao_id}")
    print(f"   Posição: {posicao.ativo} ({posicao.side})")
    print(f"   Percentual: {percentual}%")
    if valor_usd:
        print(f"   Valor: ${valor_usd:.2f}")
    
    # Mostrar resumo
    mostrar_resumo = obter_input("\nDeseja ver o resumo de alocações? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True).lower()
    if mostrar_resumo == "s":
        print("\nALOCAÇÕES DO PRODUTO:\n", alocacoes_do_produto(produto_id))
        print("\nRESUMO DE ALOCAÇÕES:\n", resumo_alocacoes(produto_id))
    
    return alocacao_id

if __name__ == "__main__":
    main()

