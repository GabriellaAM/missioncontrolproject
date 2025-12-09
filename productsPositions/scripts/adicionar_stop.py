"""
Script para adicionar stops a posições existentes
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.parquet_repo import ParquetRepo
from analytics.queries import posicoes_abertas, posicoes_fechadas
from utils.cli_utils import obter_input, validar_data, imprimir_titulo, imprimir_secao

def main():
    imprimir_titulo("ADICIONAR STOP A POSIÇÃO EXISTENTE")
    
    repo = ParquetRepo()
    
    # Listar posições (abertas e fechadas)
    imprimir_secao("POSIÇÕES DISPONÍVEIS")
    
    posicoes_abertas_df = posicoes_abertas()
    posicoes_fechadas_df = posicoes_fechadas()
    
    todas_posicoes = []
    
    if not posicoes_abertas_df.empty:
        print("🟢 POSIÇÕES ABERTAS:")
        for idx, row in posicoes_abertas_df.iterrows():
            print(f"  ID: {row['id']} | Ativo: {row['ativo']} | Side: {row['side']} | "
                  f"Entrada: {row['data_entrada']} | Preço: ${row['preco_entrada']:.2f}")
            todas_posicoes.append(row['id'])
    
    if not posicoes_fechadas_df.empty:
        print("\n🔴 POSIÇÕES FECHADAS:")
        for idx, row in posicoes_fechadas_df.iterrows():
            print(f"  ID: {row['id']} | Ativo: {row['ativo']} | Side: {row['side']} | "
                  f"Entrada: {row['data_entrada']} | Saída: {row['data_saida']} | "
                  f"Preço Entrada: ${row['preco_entrada']:.2f} | Preço Saída: ${row['preco_saida']:.2f}")
            todas_posicoes.append(row['id'])
    
    if not todas_posicoes:
        print("❌ Nenhuma posição encontrada.")
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
    print(f"   Ativo: {posicao.ativo}")
    print(f"   Side: {posicao.side}")
    print(f"   Data entrada: {posicao.data_entrada}")
    print(f"   Preço entrada: ${posicao.preco_entrada:.2f}")
    
    # Mostrar stops existentes
    if posicao.stops:
        print(f"\n📋 Stops existentes:")
        for stop in posicao.stops:
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
        if posicao_atualizada and posicao_atualizada.stops:
            print(f"\n📋 Histórico completo de stops:")
            for stop in posicao_atualizada.stops:
                print(f"   {stop['data']}: ${stop['valor']:.2f}")
    else:
        print("\n❌ Operação cancelada.")

if __name__ == "__main__":
    main()

