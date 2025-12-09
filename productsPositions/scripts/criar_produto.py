"""
Script para criar um novo produto
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.tipo import Tipo
from domain.produto import Produto
from storage.parquet_repo import ParquetRepo
from utils.cli_utils import obter_input, validar_data, imprimir_titulo

def main():
    imprimir_titulo("CRIAR PRODUTO")
    
    nome_produto = obter_input("Nome do produto: ", obrigatorio=True)
    data_inicio = obter_input("Data de início (YYYY-MM-DD): ", obrigatorio=True)
    data_inicio = validar_data(data_inicio, "Data de início")
    
    # Listar tipos disponíveis
    repo = ParquetRepo()
    tipos_disponiveis = repo.listar_tipos()
    if tipos_disponiveis:
        print(f"\nTipos disponíveis: {', '.join(tipos_disponiveis)}")
    else:
        print("\nNenhum tipo cadastrado. Criando tipos padrão...")
        repo.registrar_tipo("Perpétuos", "Contratos perpétuos de criptomoedas")
        repo.registrar_tipo("Spot", "Trading spot de criptomoedas")
        tipos_disponiveis = repo.listar_tipos()
    
    tipo_nome = obter_input("Tipo do produto: ", opcoes=tipos_disponiveis, obrigatorio=True)
    repo.registrar_tipo(tipo_nome, f"Tipo {tipo_nome}")  # Garante que existe
    
    capital_inicial = obter_input("Capital inicial investido (USD) [opcional]: ", tipo=float, obrigatorio=False)
    
    # Usar 0.0 se não fornecido
    if capital_inicial is None:
        capital_inicial = 0.0
    
    tipo = Tipo(tipo_nome)
    produto = Produto(nome_produto, data_inicio, tipo, capital_inicial=capital_inicial)
    
    # Salvar produto
    produto_id = repo.salvar_produto(produto)
    
    print(f"\n✅ Produto criado com ID: {produto_id}")
    print(f"   Nome: {nome_produto}")
    print(f"   Tipo: {tipo_nome}")
    print(f"   Data de início: {data_inicio}")
    if capital_inicial > 0:
        print(f"   Capital inicial: ${capital_inicial:,.2f}")
    else:
        print(f"   Capital inicial: Não informado")
    
    return produto_id

if __name__ == "__main__":
    main()

