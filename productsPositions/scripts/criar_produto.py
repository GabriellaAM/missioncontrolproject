"""
Script para criar um novo produto
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.tipo import Tipo
from domain.produto import Produto
from storage.sqlite_repo import SQLiteRepo
from utils.cli_utils import obter_input, validar_data, imprimir_titulo, imprimir_secao


# Templates de visualizações padrão
VISUALIZACOES_PADRAO = {
    'spot': {
        'Posicoes Abertas': {
            'colunas': [
                'data_entrada', 'ativo', 'quantidade', 'preco_entrada',
                'preco_entrada_total', 'preco_atual', 'preco_atual_total',
                'pnl', 'stop_atual'
            ],
            'ordenacao': {'coluna': 'data_entrada', 'direcao': 'asc'},
            'filtros': [{'coluna': 'status', 'operador': '=', 'valor': 'open'}]
        },
        'Posicoes Fechadas': {
            'colunas': [
                'data_entrada', 'data_saida', 'ativo', 'quantidade',
                'preco_entrada', 'preco_entrada_total', 'preco_saida',
                'preco_saida_total', 'pnl', 'stop_atual'
            ],
            'ordenacao': {'coluna': 'data_entrada', 'direcao': 'asc'},
            'filtros': [{'coluna': 'status', 'operador': '=', 'valor': 'closed'}]
        },
        'Historico': {
            'colunas': [
                'status', 'data_entrada', 'data_saida', 'ativo', 'quantidade',
                'preco_entrada', 'preco_entrada_total', 'preco_saida',
                'preco_saida_total', 'pnl', 'stop_atual'
            ],
            'ordenacao': {'coluna': 'data_entrada', 'direcao': 'asc'},
            'filtros': None
        }
    },
    'perpetuos': {
        'Posicoes Abertas': {
            'colunas': [
                'data_entrada', 'ativo', 'side', 'quantidade', 'preco_entrada',
                'preco_entrada_total', 'preco_saida', 'preco_saida_total',
                'pnl', 'stop_atual'
            ],
            'ordenacao': {'coluna': 'data_entrada', 'direcao': 'asc'},
            'filtros': [{'coluna': 'status', 'operador': '=', 'valor': 'open'}]
        },
        'Posicoes Fechadas': {
            'colunas': [
                'data_entrada', 'data_saida', 'ativo', 'side', 'quantidade',
                'preco_entrada', 'preco_entrada_total', 'preco_saida',
                'preco_saida_total', 'pnl', 'stop_atual'
            ],
            'ordenacao': {'coluna': 'data_entrada', 'direcao': 'asc'},
            'filtros': [{'coluna': 'status', 'operador': '=', 'valor': 'closed'}]
        },
        'Historico': {
            'colunas': [
                'status', 'data_entrada', 'data_saida', 'ativo', 'side',
                'quantidade', 'preco_entrada', 'preco_entrada_total',
                'preco_saida', 'preco_saida_total', 'pnl', 'stop_atual'
            ],
            'ordenacao': {'coluna': 'data_entrada', 'direcao': 'asc'},
            'filtros': None
        }
    },
    'generico': {
        'Posicoes Abertas': {
            'colunas': [
                'data_entrada', 'ativo', 'side', 'preco_entrada',
                'preco_atual', 'pnl', 'stop_atual'
            ],
            'ordenacao': {'coluna': 'data_entrada', 'direcao': 'asc'},
            'filtros': [{'coluna': 'status', 'operador': '=', 'valor': 'open'}]
        },
        'Posicoes Fechadas': {
            'colunas': [
                'data_entrada', 'data_saida', 'ativo', 'side',
                'preco_entrada', 'preco_saida', 'pnl'
            ],
            'ordenacao': {'coluna': 'data_entrada', 'direcao': 'asc'},
            'filtros': [{'coluna': 'status', 'operador': '=', 'valor': 'closed'}]
        },
        'Historico': {
            'colunas': [
                'status', 'data_entrada', 'data_saida', 'ativo', 'side',
                'preco_entrada', 'preco_saida', 'pnl'
            ],
            'ordenacao': {'coluna': 'data_entrada', 'direcao': 'asc'},
            'filtros': None
        }
    }
}


def configurar_visualizacoes(repo, produto_id, tipo_nome):
    """Configura visualizacoes padrao para o produto"""
    imprimir_secao("CONFIGURAR VISUALIZACOES")

    print("Visualizacoes padrao disponiveis:")
    print("  - Posicoes Abertas: posicoes com status 'open'")
    print("  - Posicoes Fechadas: posicoes com status 'closed'")
    print("  - Historico: todas as posicoes")

    adicionar_padrao = obter_input(
        "\nAdicionar visualizacoes padrao? (s/n): ",
        opcoes=["s", "n", "S", "N"],
        obrigatorio=True
    ).lower()

    if adicionar_padrao != "s":
        print("Nenhuma visualizacao adicionada.")
        print("Voce pode criar visualizacoes depois em: Produtos > Gerenciar visualizacoes")
        return

    # Determinar qual template usar baseado no tipo
    tipo_lower = tipo_nome.lower() if tipo_nome else ''

    if 'spot' in tipo_lower:
        template_key = 'spot'
    elif 'perpétuo' in tipo_lower or 'perpetuo' in tipo_lower:
        template_key = 'perpetuos'
    else:
        template_key = 'generico'

    template = VISUALIZACOES_PADRAO[template_key]

    print(f"\nCriando visualizacoes (template: {template_key})...")

    for nome, config in template.items():
        try:
            viz_id = repo.criar_visualizacao(
                produto_id=produto_id,
                nome=nome,
                colunas=config['colunas'],
                ordenacao=config.get('ordenacao'),
                filtros=config.get('filtros')
            )
            print(f"  - {nome}: criada (ID: {viz_id})")
        except Exception as e:
            print(f"  - {nome}: ERRO - {e}")

    print("\nVisualizacoes configuradas!")
    print("Voce pode edita-las em: Produtos > Gerenciar visualizacoes")


def configurar_atributos(repo, produto_id, tipo_nome):
    """Configura atributos personalizados para o produto"""
    imprimir_secao("CONFIGURAR ATRIBUTOS")

    # Mostrar colunas já existentes
    colunas_existentes = repo.listar_colunas_atributos()
    if colunas_existentes:
        print(f"Atributos disponiveis: {', '.join(colunas_existentes)}")

    # Perguntar se quer usar atributos padrão baseado no tipo
    tipo_lower = tipo_nome.lower()
    if 'spot' in tipo_lower or 'perpétuo' in tipo_lower or 'perpetuo' in tipo_lower:
        usar_padrao = obter_input(
            "\nUsar atributos padrao (quantidade, preco_entrada_total)? (s/n): ",
            opcoes=["s", "n", "S", "N"],
            obrigatorio=True
        ).lower()

        if usar_padrao == "s":
            repo.adicionar_atributo_config(produto_id, 'quantidade', 'float', 'Quantidade', True)
            repo.adicionar_atributo_config(produto_id, 'preco_entrada_total', 'float', 'Preco Entrada Total', False)
            print("  Adicionados: quantidade, preco_entrada_total")

    # Perguntar se quer adicionar atributos personalizados
    adicionar_custom = obter_input(
        "\nAdicionar atributos personalizados? (s/n): ",
        opcoes=["s", "n", "S", "N"],
        obrigatorio=True
    ).lower()

    if adicionar_custom == "s":
        print("\nTipos disponiveis: text, float, int, date")
        print("Digite 'fim' para terminar\n")

        while True:
            nome = obter_input("Nome do atributo (ou 'fim'): ", obrigatorio=True)
            if nome.lower() == 'fim':
                break

            # Verificar se já existe nas colunas
            nome_normalizado = nome.lower().strip().replace(' ', '_')
            if nome_normalizado in colunas_existentes:
                print(f"  Atributo '{nome_normalizado}' ja existe, sera vinculado ao produto.")
                usar_existente = obter_input("  Usar este atributo? (s/n): ", opcoes=["s", "n"], obrigatorio=True).lower()
                if usar_existente == "s":
                    obrigatorio = obter_input("  Obrigatorio? (s/n): ", opcoes=["s", "n"], obrigatorio=True).lower() == "s"
                    # Descobrir tipo da coluna existente
                    repo.adicionar_atributo_config(produto_id, nome_normalizado, 'text', None, obrigatorio)
                    print(f"  Atributo '{nome_normalizado}' vinculado ao produto.")
                continue

            tipo = obter_input("Tipo (text/float/int/date) [text]: ", opcoes=["text", "float", "int", "date", ""], obrigatorio=False)
            if not tipo:
                tipo = "text"

            label = obter_input(f"Label [{nome.replace('_', ' ').title()}]: ", obrigatorio=False)

            obrigatorio = obter_input("Obrigatorio? (s/n) [n]: ", opcoes=["s", "n", "S", "N", ""], obrigatorio=False)
            obrigatorio = obrigatorio.lower() == "s" if obrigatorio else False

            repo.adicionar_atributo_config(produto_id, nome, tipo, label if label else None, obrigatorio)
            print(f"  Atributo '{nome}' adicionado!")

            # Atualizar lista de colunas existentes
            colunas_existentes = repo.listar_colunas_atributos()

    # Mostrar atributos configurados
    configs = repo.carregar_atributos_config(produto_id)
    if configs:
        print("\nAtributos configurados para este produto:")
        for c in configs:
            obrig_str = " (obrigatorio)" if c['obrigatorio'] else ""
            print(f"  - {c['atributo_label'] or c['atributo_nome']}: {c['atributo_tipo']}{obrig_str}")


def main():
    imprimir_titulo("CRIAR PRODUTO")

    nome_produto = obter_input("Nome do produto: ", obrigatorio=True)
    data_inicio = obter_input("Data de inicio (YYYY-MM-DD): ", obrigatorio=True)
    data_inicio = validar_data(data_inicio, "Data de inicio")

    # Listar tipos disponíveis
    repo = SQLiteRepo()
    tipos_disponiveis = repo.listar_tipos()
    if tipos_disponiveis:
        print(f"\nTipos disponiveis: {', '.join(tipos_disponiveis)}")
    else:
        print("\nNenhum tipo cadastrado. Criando tipos padrao...")
        repo.registrar_tipo("Perpétuos", "Contratos perpetuos de criptomoedas")
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

    print(f"\nProduto criado com ID: {produto_id}")
    print(f"   Nome: {nome_produto}")
    print(f"   Tipo: {tipo_nome}")
    print(f"   Data de inicio: {data_inicio}")
    if capital_inicial > 0:
        print(f"   Capital inicial: ${capital_inicial:,.2f}")
    else:
        print(f"   Capital inicial: Nao informado")

    # Configurar atributos
    configurar_atributos(repo, produto_id, tipo_nome)

    # Configurar visualizacoes
    configurar_visualizacoes(repo, produto_id, tipo_nome)

    print(f"\nProduto '{nome_produto}' configurado com sucesso!")

    return produto_id


if __name__ == "__main__":
    main()

