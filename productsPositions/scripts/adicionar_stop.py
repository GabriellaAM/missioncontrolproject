"""
Script para adicionar stops a posições existentes
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from utils.cli_utils import obter_input, validar_data, imprimir_titulo, imprimir_secao, listar_posicoes_por_produto

def main():
    imprimir_titulo("ADICIONAR STOP A POSIÇÃO EXISTENTE")

    repo = SQLiteRepo()

    # Listar posições agrupadas por produto
    imprimir_secao("POSIÇÕES ABERTAS (por produto)")
    todas_posicoes = listar_posicoes_por_produto(status='open')

    print()
    imprimir_secao("POSIÇÕES FECHADAS (por produto)")
    todas_posicoes += listar_posicoes_por_produto(status='closed')

    if not todas_posicoes:
        print("Nenhuma posição encontrada.")
        return
    
    # Selecionar posição
    posicao_id = obter_input("\nID da posição: ", tipo=int, obrigatorio=True)
    
    # Verificar se posição existe
    posicao = repo.carregar_posicao(posicao_id)
    if not posicao:
        print(f"❌ Posição com ID {posicao_id} não encontrada!")
        return
    
    imprimir_secao("DADOS DA POSIÇÃO")
    print(f"✅ Posição encontrada:")
    print(f"   Ativo: {posicao['ativo']}")
    print(f"   Side: {posicao['side']}")
    print(f"   Data entrada: {posicao['data_entrada']}")
    print(f"   Preço entrada: ${posicao['preco_entrada']:.2f}")
    
    # Mostrar stops existentes
    stops = posicao.get('stops', [])
    if stops:
        print(f"\n📋 Stops existentes:")
        for stop in stops:
            print(f"   {stop['data']}: ${stop['valor']:.2f}")
    else:
        print("\n📋 Nenhum stop cadastrado ainda.")
    
    # Adicionar novo stop
    imprimir_secao("ADICIONAR NOVO STOP")
    data_stop = obter_input("Data do stop (YYYY-MM-DD): ", obrigatorio=True)
    data_stop = validar_data(data_stop, "Data do stop")
    
    valor_stop = obter_input("Valor do stop: ", tipo=float, obrigatorio=True)
    
    # Confirmar
    confirmar = obter_input(f"\nConfirmar adição de stop {data_stop} - ${valor_stop:.2f}? (s/n): ", 
                           opcoes=["s", "n", "S", "N"], obrigatorio=True).lower()
    
    if confirmar == "s":
        repo.adicionar_stop_posicao(posicao_id, data_stop, valor_stop)
        print(f"\n✅ Stop adicionado com sucesso!")
        
        # Mostrar stops atualizados
        posicao_atualizada = repo.carregar_posicao(posicao_id)
        if posicao_atualizada:
            stops = posicao_atualizada.get('stops', [])
            if stops:
                print(f"\n📋 Histórico completo de stops:")
                for stop in stops:
                    print(f"   {stop['data']}: ${stop['valor']:.2f}")
    else:
        print("\n❌ Operação cancelada.")

if __name__ == "__main__":
    main()

