"""
Script para gerenciar visualizações customizadas de posições.
Permite criar, editar e deletar visualizações com colunas, ordenação e filtros personalizados.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.sqlite_repo import SQLiteRepo
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao


def selecionar_produto(repo):
    """Seleciona um produto para gerenciar visualizações"""
    produtos = repo.listar_produtos()
    if not produtos:
        print("Nenhum produto encontrado.")
        return None

    print("Produtos disponíveis:")
    for p in produtos:
        print(f"  ID: {p['id']} - {p['nome']}")

    produto_id = obter_input("\nID do produto: ", tipo=int, obrigatorio=True)

    produto = repo.carregar_produto(produto_id)
    if not produto:
        print(f"Produto {produto_id} não encontrado.")
        return None

    return produto_id, produto['nome']


def listar_visualizacoes(repo, produto_id, produto_nome):
    """Lista visualizações de um produto"""
    imprimir_secao(f"VISUALIZAÇÕES - {produto_nome.upper()}")

    visualizacoes = repo.listar_visualizacoes(produto_id)

    if not visualizacoes:
        print("Nenhuma visualização cadastrada para este produto.")
        return []

    for v in visualizacoes:
        print(f"\n  [{v['id']}] {v['nome']}")
        print(f"      Colunas: {', '.join(v['colunas'][:5])}{'...' if len(v['colunas']) > 5 else ''}")
        if v.get('colunas_labels'):
            labels_count = len([l for l in v['colunas_labels'].values() if l])
            print(f"      Labels customizados: {labels_count}")
        if v['ordenacao']:
            print(f"      Ordenação: {v['ordenacao']['coluna']} ({v['ordenacao']['direcao']})")
        if v['filtros']:
            print(f"      Filtros: {len(v['filtros'])} filtro(s)")

    return visualizacoes


def criar_visualizacao(repo, produto_id):
    """Cria uma nova visualização"""
    imprimir_secao("CRIAR VISUALIZAÇÃO")

    # Nome
    nome = obter_input("Nome da visualização: ", obrigatorio=True)

    # Listar colunas disponíveis
    colunas_disponiveis = repo.obter_colunas_disponiveis(produto_id)

    print("\nColunas disponíveis:")
    for i, col in enumerate(colunas_disponiveis, 1):
        origem_str = f"[{col['origem']}]"
        print(f"  {i:2}. {col['label']} ({col['nome']}) {origem_str}")

    print("\nDigite os números das colunas separados por vírgula (ex: 1,2,5,8)")
    print("Ou 'todas' para incluir todas as colunas")
    selecao = obter_input("Colunas: ", obrigatorio=True)

    if selecao.lower() == 'todas':
        colunas_selecionadas = [c['nome'] for c in colunas_disponiveis]
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selecao.split(',')]
            colunas_selecionadas = [colunas_disponiveis[i]['nome'] for i in indices if 0 <= i < len(colunas_disponiveis)]
        except (ValueError, IndexError):
            print("Seleção inválida.")
            return

    if not colunas_selecionadas:
        print("Nenhuma coluna selecionada.")
        return

    print(f"\nColunas selecionadas: {', '.join(colunas_selecionadas)}")

    # Ordenação
    ordenacao = None
    add_ordenacao = obter_input("\nAdicionar ordenação? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True)
    if add_ordenacao.lower() == "s":
        ordenacao = _configurar_ordenacao(colunas_selecionadas, colunas_disponiveis)

    # Filtros
    filtros = None
    add_filtros = obter_input("\nAdicionar filtros? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True)
    if add_filtros.lower() == "s":
        filtros = _configurar_filtros(colunas_selecionadas, colunas_disponiveis)

    # Criar
    try:
        viz_id = repo.criar_visualizacao(produto_id, nome, colunas_selecionadas, ordenacao, filtros)
        print(f"\nVisualização '{nome}' criada com ID: {viz_id}")
    except Exception as e:
        print(f"\nErro ao criar visualização: {e}")


def _configurar_ordenacao(colunas_selecionadas, colunas_disponiveis):
    """Configura ordenação para a visualização"""
    print("\nColunas disponíveis para ordenação:")
    cols_map = {c['nome']: c for c in colunas_disponiveis}
    for i, nome in enumerate(colunas_selecionadas, 1):
        label = cols_map.get(nome, {}).get('label', nome)
        print(f"  {i}. {label} ({nome})")

    idx = obter_input("Coluna para ordenar (número): ", tipo=int, obrigatorio=True)
    if idx < 1 or idx > len(colunas_selecionadas):
        print("Opção inválida.")
        return None

    coluna = colunas_selecionadas[idx - 1]
    direcao = obter_input("Direção (asc/desc): ", opcoes=["asc", "desc"], obrigatorio=True)

    return {'coluna': coluna, 'direcao': direcao}


def _configurar_filtros(colunas_selecionadas, colunas_disponiveis):
    """Configura filtros para a visualização"""
    filtros = []
    cols_map = {c['nome']: c for c in colunas_disponiveis}

    print("\nOperadores disponíveis: =, !=, >, <, >=, <=, contém")
    print("Digite 'fim' para terminar\n")

    while True:
        print("Colunas disponíveis para filtro:")
        for i, nome in enumerate(colunas_selecionadas, 1):
            label = cols_map.get(nome, {}).get('label', nome)
            print(f"  {i}. {label} ({nome})")

        idx_str = obter_input("\nColuna para filtrar (número ou 'fim'): ", obrigatorio=True)
        if idx_str.lower() == 'fim':
            break

        try:
            idx = int(idx_str)
            if idx < 1 or idx > len(colunas_selecionadas):
                print("Opção inválida.")
                continue
        except ValueError:
            print("Número inválido.")
            continue

        coluna = colunas_selecionadas[idx - 1]
        operador = obter_input("Operador (=, !=, >, <, >=, <=, contém): ", obrigatorio=True)
        valor = obter_input("Valor: ", obrigatorio=True)

        # Tentar converter para número se possível
        try:
            if '.' in valor:
                valor = float(valor)
            else:
                valor = int(valor)
        except ValueError:
            pass  # Manter como string

        filtros.append({
            'coluna': coluna,
            'operador': operador,
            'valor': valor
        })
        print(f"  Filtro adicionado: {coluna} {operador} {valor}")

    return filtros if filtros else None


def _configurar_labels(colunas, labels_atuais=None):
    """Configura labels customizados para as colunas"""
    labels = labels_atuais.copy() if labels_atuais else {}

    while True:
        print("\nColunas e seus labels atuais:")
        for i, col in enumerate(colunas, 1):
            label_atual = labels.get(col, col)
            if label_atual != col:
                print(f"  {i}. {col} -> {label_atual}")
            else:
                print(f"  {i}. {col}")

        print("\nDigite o numero da coluna para alterar o label")
        print("Digite 0 para finalizar")

        opcao = obter_input("\nColuna: ", obrigatorio=True)

        if opcao == "0":
            break

        try:
            idx = int(opcao) - 1
            if 0 <= idx < len(colunas):
                col = colunas[idx]
                label_atual = labels.get(col, col)
                print(f"\nColuna: {col}")
                print(f"Label atual: {label_atual}")
                novo_label = obter_input("Novo label (Enter para resetar): ", obrigatorio=False)

                if novo_label:
                    labels[col] = novo_label
                    print(f"Label alterado para: {novo_label}")
                else:
                    # Resetar para nome original
                    if col in labels:
                        del labels[col]
                    print(f"Label resetado para: {col}")
            else:
                print("Numero invalido.")
        except ValueError:
            print("Digite um numero valido.")

    return labels if labels else None


def editar_visualizacao(repo, produto_id):
    """Edita uma visualização existente"""
    visualizacoes = repo.listar_visualizacoes(produto_id)
    if not visualizacoes:
        print("Nenhuma visualização para editar.")
        return

    viz_id = obter_input("\nID da visualização para editar: ", tipo=int, obrigatorio=True)

    viz = repo.carregar_visualizacao(viz_id)
    if not viz or viz['produto_id'] != produto_id:
        print("Visualização não encontrada.")
        return

    imprimir_secao(f"EDITAR: {viz['nome']}")

    print("\nO que deseja editar?")
    print("1. Nome")
    print("2. Colunas")
    print("3. Ordenação")
    print("4. Filtros")
    print("5. Nomes das colunas (labels)")
    print("0. Cancelar")

    opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3", "4", "5"], obrigatorio=True)

    if opcao == "0":
        return

    colunas_disponiveis = repo.obter_colunas_disponiveis(produto_id)

    if opcao == "1":
        novo_nome = obter_input(f"Novo nome [{viz['nome']}]: ", obrigatorio=False)
        if novo_nome:
            repo.atualizar_visualizacao(viz_id, nome=novo_nome)
            print(f"Nome atualizado para '{novo_nome}'")

    elif opcao == "2":
        print("\nColunas atuais:", ', '.join(viz['colunas']))
        print("\nColunas disponíveis:")
        for i, col in enumerate(colunas_disponiveis, 1):
            print(f"  {i:2}. {col['label']} ({col['nome']})")

        print("\nDigite os números das colunas separados por vírgula (ex: 1,2,5)")
        selecao = obter_input("Colunas: ", obrigatorio=True)

        try:
            # Filtrar strings vazias antes de converter para int
            partes = [x.strip() for x in selecao.split(',') if x.strip()]
            indices = [int(x) - 1 for x in partes]
            novas_colunas = [colunas_disponiveis[i]['nome'] for i in indices if 0 <= i < len(colunas_disponiveis)]

            if not novas_colunas:
                print("Nenhuma coluna válida selecionada.")
            else:
                # Verificar se a coluna de ordenação ainda existe nas novas colunas
                nova_ordenacao = viz.get('ordenacao')
                if nova_ordenacao and nova_ordenacao.get('coluna'):
                    col_ordenacao = nova_ordenacao['coluna']
                    if col_ordenacao not in novas_colunas:
                        print(f"\n⚠️  A coluna de ordenação '{col_ordenacao}' não está mais na visualização.")
                        acao_ord = obter_input("Deseja (r)emover ordenação ou (e)scolher nova coluna? ",
                                              opcoes=["r", "e", "R", "E"], obrigatorio=True)
                        if acao_ord.lower() == "r":
                            nova_ordenacao = None
                            print("Ordenação será removida.")
                        else:
                            nova_ordenacao = _configurar_ordenacao(novas_colunas, colunas_disponiveis)

                # Verificar se as colunas dos filtros ainda existem
                novos_filtros = viz.get('filtros')
                if novos_filtros:
                    filtros_removidos = []
                    filtros_validos = []
                    for f in novos_filtros:
                        if f['coluna'] in novas_colunas:
                            filtros_validos.append(f)
                        else:
                            filtros_removidos.append(f)

                    if filtros_removidos:
                        print(f"\n⚠️  {len(filtros_removidos)} filtro(s) usam colunas removidas:")
                        for f in filtros_removidos:
                            print(f"    - {f['coluna']} {f['operador']} {f['valor']}")
                        print("Esses filtros serão removidos automaticamente.")
                        novos_filtros = filtros_validos if filtros_validos else None

                # Atualizar visualização
                repo.atualizar_visualizacao(viz_id, colunas=novas_colunas,
                                           ordenacao=nova_ordenacao, filtros=novos_filtros)
                print(f"\nColunas atualizadas: {', '.join(novas_colunas)}")

        except (ValueError, IndexError):
            print("Seleção inválida. Use números separados por vírgula (ex: 1,2,3)")

    elif opcao == "3":
        if viz['ordenacao']:
            print(f"Ordenação atual: {viz['ordenacao']['coluna']} ({viz['ordenacao']['direcao']})")

        remover = obter_input("Remover ordenação? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True)
        if remover.lower() == "s":
            repo.atualizar_visualizacao(viz_id, ordenacao={})
            print("Ordenação removida.")
        else:
            nova_ordenacao = _configurar_ordenacao(viz['colunas'], colunas_disponiveis)
            if nova_ordenacao:
                repo.atualizar_visualizacao(viz_id, ordenacao=nova_ordenacao)
                print(f"Ordenação atualizada: {nova_ordenacao['coluna']} ({nova_ordenacao['direcao']})")

    elif opcao == "4":
        if viz['filtros']:
            print("Filtros atuais:")
            for f in viz['filtros']:
                print(f"  - {f['coluna']} {f['operador']} {f['valor']}")

        remover = obter_input("Remover todos os filtros? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True)
        if remover.lower() == "s":
            repo.atualizar_visualizacao(viz_id, filtros=[])
            print("Filtros removidos.")
        else:
            novos_filtros = _configurar_filtros(viz['colunas'], colunas_disponiveis)
            repo.atualizar_visualizacao(viz_id, filtros=novos_filtros)
            print("Filtros atualizados.")

    elif opcao == "5":
        novos_labels = _configurar_labels(viz['colunas'], viz.get('colunas_labels'))
        repo.atualizar_visualizacao(viz_id, colunas_labels=novos_labels)
        print("Labels atualizados.")


def deletar_visualizacao(repo, produto_id):
    """Deleta uma visualização"""
    visualizacoes = repo.listar_visualizacoes(produto_id)
    if not visualizacoes:
        print("Nenhuma visualização para deletar.")
        return

    viz_id = obter_input("\nID da visualização para deletar: ", tipo=int, obrigatorio=True)

    viz = repo.carregar_visualizacao(viz_id)
    if not viz or viz['produto_id'] != produto_id:
        print("Visualização não encontrada.")
        return

    confirmar = obter_input(f"Confirmar exclusão de '{viz['nome']}'? (s/n): ",
                           opcoes=["s", "n", "S", "N"], obrigatorio=True)

    if confirmar.lower() == "s":
        repo.deletar_visualizacao(viz_id)
        print(f"Visualização '{viz['nome']}' deletada.")
    else:
        print("Operação cancelada.")


def visualizar_colunas_disponiveis(repo, produto_id):
    """Mostra todas as colunas disponíveis para o produto"""
    imprimir_secao("COLUNAS DISPONÍVEIS")

    colunas = repo.obter_colunas_disponiveis(produto_id)

    print("\nColunas de Posição:")
    for c in colunas:
        if c['origem'] == 'posicao':
            print(f"  - {c['label']} ({c['nome']})")

    print("\nAtributos do Produto:")
    attrs = [c for c in colunas if c['origem'] == 'atributo']
    if attrs:
        for c in attrs:
            print(f"  - {c['label']} ({c['nome']})")
    else:
        print("  (nenhum atributo configurado)")

    print("\nCampos Calculados:")
    for c in colunas:
        if c['origem'] == 'calculado':
            print(f"  - {c['label']} ({c['nome']})")


def main():
    repo = SQLiteRepo()

    # Selecionar produto
    resultado = selecionar_produto(repo)
    if not resultado:
        return

    produto_id, produto_nome = resultado

    while True:
        imprimir_titulo(f"VISUALIZAÇÕES - {produto_nome}")

        print("1. Listar visualizações")
        print("2. Criar visualização")
        print("3. Editar visualização")
        print("4. Deletar visualização")
        print("5. Ver colunas disponíveis")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3", "4", "5"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            listar_visualizacoes(repo, produto_id, produto_nome)
        elif opcao == "2":
            criar_visualizacao(repo, produto_id)
        elif opcao == "3":
            listar_visualizacoes(repo, produto_id, produto_nome)
            editar_visualizacao(repo, produto_id)
        elif opcao == "4":
            listar_visualizacoes(repo, produto_id, produto_nome)
            deletar_visualizacao(repo, produto_id)
        elif opcao == "5":
            visualizar_colunas_disponiveis(repo, produto_id)

        print()


if __name__ == "__main__":
    main()
