"""
Script para criar uma nova posição em um produto existente
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.posicao_service import PosicaoService
from services.valor_diario_service import ValorDiarioService
from storage.parquet_repo import ParquetRepo
from domain.produto import Produto
from utils.cli_utils import obter_input, validar_data, imprimir_titulo, imprimir_secao

def main():
    imprimir_titulo("CRIAR POSIÇÃO")
    
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
    
    # Verificar se produto existe e carregar como objeto
    produto = repo.carregar_produto_objeto(produto_id)
    if not produto:
        print(f"❌ Produto com ID {produto_id} não encontrado!")
        return None
    
    imprimir_secao("DADOS DA POSIÇÃO")
    
    ativo = obter_input("Ativo (ex: BTC, ETH): ", obrigatorio=True).upper()
    side = obter_input("Side (long/short): ", opcoes=["long", "short"], obrigatorio=True)
    data_entrada = obter_input("Data de entrada (YYYY-MM-DD): ", obrigatorio=True)
    data_entrada = validar_data(data_entrada, "Data de entrada")
    
    preco_entrada = obter_input("Preço de entrada: ", tipo=float, obrigatorio=True)
    coingecko_id = obter_input("CoinGecko ID (ex: bitcoin, ethereum) [opcional]: ", obrigatorio=False)
    
    # Abrir posição
    p = PosicaoService.abrir(
        produto,
        ativo=ativo,
        side=side,
        data=data_entrada,
        preco=preco_entrada,
        coingecko_id=coingecko_id if coingecko_id else None
    )
    
    # Adicionar stop inicial (opcional)
    adicionar_stop = obter_input("\nDeseja adicionar um stop inicial? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True).lower()
    
    if adicionar_stop == "s":
        data_stop = obter_input("Data do stop (YYYY-MM-DD): ", obrigatorio=True)
        data_stop = validar_data(data_stop, "Data do stop")
        valor_stop = obter_input("Valor do stop: ", tipo=float, obrigatorio=True)
        p.adicionar_stop(data_stop, valor_stop)
        print(f"✅ Stop adicionado: {data_stop} - ${valor_stop:.2f}")
    
    # Salvar posição
    posicao_id = repo.salvar_posicao(produto_id, p)
    
    print(f"\n✅ Posição criada com ID: {posicao_id}")
    print(f"   Ativo: {ativo}")
    print(f"   Side: {side}")
    print(f"   Preço de entrada: ${preco_entrada:.2f}")
    
    # Verificar se dados do CoinGecko estão disponíveis
    if coingecko_id:
        imprimir_secao("VERIFICANDO DADOS DO COINGECKO")
        print(f"📥 Verificando disponibilidade de dados do CoinGecko para {ativo}...")
        try:
            valores = ValorDiarioService.ler_valores_do_coingecko(coingecko_id)
            if valores:
                print(f"✅ {len(valores)} valores disponíveis no CoinGecko")
                print(f"   Os valores serão lidos diretamente da fonte quando necessário")
            else:
                print(f"⚠️  Nenhum dado encontrado para {coingecko_id}")
                print(f"   Verifique se o arquivo existe em: data_parquet/crypto_data/coingecko/{coingecko_id}/data.parquet")
        except Exception as e:
            print(f"⚠️  Erro ao verificar dados do CoinGecko: {str(e)}")
    
    return posicao_id

if __name__ == "__main__":
    main()

