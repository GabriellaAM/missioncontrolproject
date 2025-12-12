"""
Script para deletar uma posição do banco de dados
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from analytics.queries import posicoes_abertas, posicoes_fechadas
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao

def main():
    imprimir_titulo("DELETAR POSIÇÃO")
    
    repo = SQLiteRepo()
    
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
    posicao_id = obter_input("\nID da posição a deletar: ", tipo=int, obrigatorio=True)
    
    # Verificar se posição existe
    posicao = repo.carregar_posicao(posicao_id)
    if not posicao:
        print(f"❌ Posição com ID {posicao_id} não encontrada!")
        return
    
    imprimir_secao("DADOS DA POSIÇÃO")
    print(f"⚠️  ATENÇÃO: Você está prestes a deletar:")
    print(f"   Ativo: {posicao['ativo']}")
    print(f"   Side: {posicao['side']}")
    print(f"   Data entrada: {posicao['data_entrada']}")
    print(f"   Preço entrada: ${posicao['preco_entrada']:.2f}")
    print(f"   Status: {posicao['status']}")
    
    # Verificar alocações
    from analytics.queries import alocacoes_da_posicao
    alocacoes = alocacoes_da_posicao(posicao_id)
    if not alocacoes.empty:
        print(f"\n⚠️  Esta posição possui {len(alocacoes)} alocação(ões) ativa(s):")
        for _, aloc in alocacoes.iterrows():
            print(f"   - Alocação ID {aloc['id']}: {aloc['percentual']}% (${aloc['valor_usd']:.2f})")
        print("\n⚠️  As alocações serão deletadas junto com a posição!")
    
    # Verificar stops
    stops = posicao.get('stops', [])
    if stops:
        print(f"\n⚠️  Esta posição possui {len(stops)} stop(s):")
        for stop in stops:
            print(f"   - {stop['data']}: ${stop['valor']:.2f}")
        print("\n⚠️  Os stops serão deletados junto com a posição!")
    
    # Confirmar
    confirmar = obter_input(f"\n⚠️  CONFIRMAR DELETAR POSIÇÃO {posicao_id}? (digite 'DELETAR' para confirmar): ", obrigatorio=True)
    
    if confirmar.upper() != "DELETAR":
        print("\n❌ Operação cancelada.")
        return
    
    try:
        # Tentar deletar
        sucesso = repo.deletar_posicao(posicao_id, forcar=True)
        if sucesso:
            print(f"\n✅ Posição {posicao_id} deletada com sucesso!")
            if not alocacoes.empty:
                print(f"   {len(alocacoes)} alocação(ões) também foram deletada(s)")
            if stops:
                print(f"   {len(stops)} stop(s) também foram deletado(s)")
        else:
            print(f"\n❌ Erro ao deletar posição {posicao_id}")
    except ValueError as e:
        print(f"\n❌ Erro: {str(e)}")
        print("\n💡 Dica: Se a posição tem alocações ativas, você pode:")
        print("   1. Desativar as alocações primeiro")
        print("   2. Ou usar forcar=True no código")

if __name__ == "__main__":
    main()

