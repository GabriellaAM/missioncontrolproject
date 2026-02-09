"""
Script para deletar um produto e todos os dados associados

CUIDADO: Esta operacao deleta permanentemente:
- O produto
- Todas as posicoes do produto
- Todos os stops das posicoes
- Todas as alocacoes
- A carteira do produto
- Os atributos das posicoes
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao


def main():
    imprimir_titulo("DELETAR PRODUTO")

    repo = SQLiteRepo()

    # Listar produtos disponíveis
    produtos = repo.listar_produtos()
    if not produtos:
        print("Nenhum produto encontrado.")
        return

    print("Produtos disponiveis:")
    for p in produtos:
        capital = p.get('capital_inicial', 0) or 0
        print(f"  ID: {p['id']} - {p['nome']} ({p['tipo']}) - Capital: ${capital:,.2f}")

    produto_id = obter_input("\nID do produto a deletar: ", tipo=int, obrigatorio=True)

    # Verificar se produto existe
    produto = repo.carregar_produto(produto_id)
    if not produto:
        print(f"Produto com ID {produto_id} nao encontrado!")
        return

    imprimir_secao("PRODUTO SELECIONADO")
    print(f"  Nome: {produto['nome']}")
    print(f"  Tipo: {produto['tipo']}")
    print(f"  Data inicio: {produto['data_inicio']}")
    capital = produto.get('capital_inicial', 0) or 0
    print(f"  Capital inicial: ${capital:,.2f}")

    # Contar dados associados
    import psycopg2
    conn = psycopg2.connect(repo.db_url)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM posicoes WHERE produto_id = %s", (produto_id,))
    count_posicoes = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM alocacoes WHERE produto_id = %s", (produto_id,))
    count_alocacoes = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM carteiras WHERE produto_id = %s", (produto_id,))
    count_carteiras = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM stops s
        JOIN posicoes p ON s.posicao_id = p.id
        WHERE p.produto_id = %s
    """, (produto_id,))
    count_stops = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM posicao_atributos_produto
        WHERE produto_id = %s
    """, (produto_id,))
    count_atributos = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM visualizacoes_config WHERE produto_id = %s", (produto_id,))
    count_visualizacoes = cursor.fetchone()[0]

    conn.close()

    # Mostrar resumo do que será deletado
    imprimir_secao("DADOS QUE SERAO DELETADOS")
    print(f"  Posicoes: {count_posicoes}")
    print(f"  Stops: {count_stops}")
    print(f"  Alocacoes: {count_alocacoes}")
    print(f"  Atributos: {count_atributos}")
    print(f"  Visualizacoes: {count_visualizacoes}")
    print(f"  Carteira: {count_carteiras}")

    if count_posicoes > 0 or count_alocacoes > 0:
        print("\n  ATENCAO: Este produto possui dados importantes!")
        print("  Esta acao e IRREVERSIVEL!")

    # Confirmação dupla
    imprimir_secao("CONFIRMACAO")
    print(f"Para confirmar a exclusao, digite o nome do produto: {produto['nome']}")

    nome_confirmacao = obter_input("\nNome do produto: ", obrigatorio=True)
    if nome_confirmacao != produto['nome']:
        print("\nNome incorreto. Operacao cancelada.")
        return

    print("\nAgora digite 'DELETAR' para confirmar definitivamente:")
    confirmacao_final = obter_input("Confirmacao: ", obrigatorio=True)

    if confirmacao_final.upper() != "DELETAR":
        print("\nOperacao cancelada.")
        return

    # Executar deleção
    try:
        resumo = repo.deletar_produto(produto_id, forcar=True)
        print(f"\nProduto '{produto['nome']}' deletado com sucesso!")
        print(f"  Posicoes deletadas: {resumo['posicoes']}")
        print(f"  Stops deletados: {resumo['stops']}")
        print(f"  Alocacoes deletadas: {resumo['alocacoes']}")
        print(f"  Atributos deletados: {resumo['atributos']}")
        print(f"  Visualizacoes deletadas: {resumo['visualizacoes']}")
        print(f"  Carteiras deletadas: {resumo['carteiras']}")
    except Exception as e:
        print(f"\nErro ao deletar produto: {e}")


if __name__ == "__main__":
    main()
