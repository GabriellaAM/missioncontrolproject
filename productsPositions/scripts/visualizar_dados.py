"""
Script para visualizar dados de forma interativa usando notebook_utils
Exibe dados formatados e gráficos diretamente no console ou navegador
"""
import sys
import webbrowser
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.notebook_utils import *
from analytics.charts import *
from storage.sqlite_repo import SQLiteRepo
from utils.cli_utils import obter_input, imprimir_titulo, imprimir_secao
import pandas as pd

# Tentar importar rich para tabelas bonitas
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Tentar importar tabulate como fallback
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

def exibir_dataframe_bonito(df, titulo="DataFrame", abrir_html=False):
    """
    Exibe DataFrame de forma bonita usando rich, tabulate ou formato simples
    
    Args:
        df: DataFrame para exibir
        titulo: Título da tabela
        abrir_html: Se True, abre HTML no navegador (só se não tiver Rich/Tabulate)
    """
    if df is None or df.empty:
        return False
    
    # Opção 1: Rich (mais bonito)
    if RICH_AVAILABLE:
        console = Console()
        table = Table(title=titulo, box=box.ROUNDED, show_header=True, header_style="bold magenta")
        
        # Adicionar colunas
        for col in df.columns:
            table.add_column(col, style="cyan", no_wrap=False)
        
        # Adicionar linhas (limitar a 100 para não sobrecarregar)
        max_rows = min(100, len(df))
        for idx, row in df.head(max_rows).iterrows():
            table.add_row(*[str(val) if pd.notna(val) else "" for val in row])
        
        if len(df) > max_rows:
            table.add_row(*["..." for _ in df.columns])
            table.add_row(*[f"({len(df) - max_rows} mais linhas...)" for _ in df.columns])
        
        console.print(table)
        if len(df) > max_rows:
            print(f"\n⚠️  Mostrando apenas as primeiras {max_rows} de {len(df)} linhas")
        return True
    
    # Opção 2: Tabulate (fallback)
    elif TABULATE_AVAILABLE:
        print(f"\n{'='*80}")
        print(f"  {titulo}")
        print(f"{'='*80}\n")
        print(tabulate(df.head(100), headers='keys', tablefmt='grid', showindex=False))
        if len(df) > 100:
            print(f"\n⚠️  Mostrando apenas as primeiras 100 de {len(df)} linhas")
        return True
    
    # Opção 3: Formato simples no console (sempre funciona)
    else:
        print(f"\n{'='*80}")
        print(f"  {titulo}")
        print(f"{'='*80}\n")
        max_rows = min(100, len(df))
        print(df.head(max_rows).to_string(index=False))
        if len(df) > max_rows:
            print(f"\n⚠️  Mostrando apenas as primeiras {max_rows} de {len(df)} linhas")
            print("   💡 Instale 'rich' ou 'tabulate' para visualização melhorada")
            print("      pip install rich  ou  pip install tabulate")
        
        # Só abre HTML se explicitamente solicitado
        if abrir_html:
            return exibir_dataframe_html(df, titulo, produto_id=None, tipo_dado=None)
        
        return True

def mostrar_opcoes_exportacao(df, titulo, produto_id, tipo_dado):
    """
    Mostra opções de exportação após exibir um DataFrame
    """
    if df is None or df.empty:
        return
    
    print("\n" + "="*80)
    print("OPÇÕES ADICIONAIS")
    print("="*80)
    print("1. Exportar para CSV")
    print("2. Exportar para Excel")
    print("3. Abrir em HTML (navegador)")
    print("0. Continuar (sem exportar)")
    
    opcao_export = obter_input("\nEscolha uma opção: ", opcoes=["0", "1", "2", "3"], obrigatorio=True)
    
    if opcao_export == "1":
        if produto_id:
            nome_arquivo = f"{tipo_dado}_produto_{produto_id}.csv"
        else:
            nome_arquivo = f"{tipo_dado}.csv"
        caminho = Path.cwd() / nome_arquivo
        df.to_csv(caminho, index=False)
        print(f"\n✅ DataFrame exportado para CSV!")
        print(f"   Arquivo: {caminho.absolute()}")
    
    elif opcao_export == "2":
        try:
            import openpyxl
            if produto_id:
                nome_arquivo = f"{tipo_dado}_produto_{produto_id}.xlsx"
            else:
                nome_arquivo = f"{tipo_dado}.xlsx"
            caminho = Path.cwd() / nome_arquivo
            df.to_excel(caminho, index=False, engine='openpyxl')
            print(f"\n✅ DataFrame exportado para Excel!")
            print(f"   Arquivo: {caminho.absolute()}")
        except ImportError:
            print("❌ Biblioteca openpyxl não encontrada!")
            print("   Instale com: pip install openpyxl")
    
    elif opcao_export == "3":
        exibir_dataframe_html(df, titulo, produto_id=produto_id, tipo_dado=tipo_dado)

def exibir_dataframe_html(df, titulo="DataFrame", produto_id=None, tipo_dado=None):
    """
    Cria um arquivo HTML temporário e abre no navegador
    Garante que todas as colunas sejam exibidas, incluindo atributos do produto
    
    Args:
        df: DataFrame para exibir
        titulo: Título da visualização
        produto_id: ID do produto (para atualização)
        tipo_dado: Tipo de dado ('posicoes_abertas', 'posicoes_fechadas', 'historico', etc.)
    """
    if df is None or df.empty:
        return False
    
    try:
        from datetime import datetime
        
        # Criar cópia do DataFrame para não modificar o original
        df_html = df.copy()
        
        # Substituir valores None/NaN por strings vazias para melhor visualização
        df_html = df_html.fillna('')
        
        # Não reordenar colunas aqui: respeitar a ordem já definida
        # nas funções de display (display_posicoes_abertas, etc.)
        colunas_atributos = ['perfil', 'motivo', 'pnl', 'rr', 'alvo1', 'alvo2', 'stop_atual']  # usado apenas para destaque visual
        
        # Timestamp da última atualização
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Criar HTML estilizado
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{titulo}</title>
            <meta charset="UTF-8">
            <meta http-equiv="refresh" content="300"> <!-- Auto-refresh a cada 5 minutos -->
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 3px solid #4CAF50;
                    padding-bottom: 10px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }}
                .header-controls {{
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }}
                .btn-atualizar {{
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }}
                .btn-atualizar:hover {{
                    background-color: #45a049;
                }}
                .btn-atualizar:active {{
                    background-color: #3d8b40;
                }}
                .btn-atualizar:disabled {{
                    background-color: #cccccc;
                    cursor: not-allowed;
                }}
                .timestamp {{
                    font-size: 12px;
                    color: #666;
                    font-weight: normal;
                }}
                .container {{
                    overflow-x: auto;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-top: 20px;
                    min-width: 100%;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: bold;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .info {{
                    background-color: #e7f3ff;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .atributos {{
                    background-color: #fff3cd;
                    font-weight: bold;
                }}
                .loading {{
                    display: none;
                    text-align: center;
                    padding: 20px;
                    background-color: #fff3cd;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .loading.show {{
                    display: block;
                }}
            </style>
        </head>
        <body>
            <h1>
                <span>{titulo}</span>
                <div class="header-controls">
                    <span class="timestamp" id="timestamp">Última atualização: {timestamp}</span>
                    <button class="btn-atualizar" id="btnAtualizar" onclick="atualizarDados()">
                        🔄 Atualizar
                    </button>
                </div>
            </h1>
            <div class="loading" id="loading">
                Carregando...
            </div>
            <div class="sucesso" id="sucesso" style="display: none; text-align: center; padding: 15px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; margin: 20px 0; color: #155724;">
                <strong>✅ Dados atualizados com sucesso!</strong>
            </div>
            <div class="erro" id="erro" style="display: none; text-align: center; padding: 15px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; margin: 20px 0; color: #721c24;">
                <strong>❌ Erro ao atualizar dados. Certifique-se de que o servidor de atualização está rodando.</strong><br>
                <small>Execute: <code>python scripts/servidor_atualizacao.py</code></small>
            </div>
            <div class="info">
                <strong>Total de registros:</strong> <span id="totalRegistros">{len(df_html)}</span><br>
                <strong>Total de colunas:</strong> {len(df_html.columns)}<br>
                <strong>Colunas:</strong> {', '.join(df_html.columns.tolist())}
            </div>
            <div class="container" id="tableContainer">
                {df_html.to_html(index=False, classes='dataframe', escape=False, table_id='dataframe')}
            </div>
            <script>
                // Dados para atualização
                const produtoId = {produto_id if produto_id else 'null'};
                const tipoDado = {repr(tipo_dado) if tipo_dado else 'null'};
                const servidorUrl = 'http://localhost:8765';
                
                // Função para atualizar dados
                async function atualizarDados() {{
                    const btn = document.getElementById('btnAtualizar');
                    const loading = document.getElementById('loading');
                    const timestamp = document.getElementById('timestamp');
                    
                    // Desabilitar botão e mostrar loading
                    btn.disabled = true;
                    btn.textContent = '🔄 Atualizando...';
                    loading.classList.add('show');
                    
                    try {{
                        // Fazer requisição ao servidor
                        const url = `${{servidorUrl}}/atualizar?produto_id=${{produtoId}}&tipo_dado=${{tipoDado}}`;
                        console.log('Fazendo requisição para:', url);
                        const response = await fetch(url);
                        
                        if (!response.ok) {{
                            const errorText = await response.text();
                            console.error('Erro HTTP:', response.status, errorText);
                            throw new Error(`Erro HTTP ${{response.status}}: ${{errorText}}`);
                        }}
                        
                        const data = await response.json();
                        console.log('Resposta recebida:', data);
                        
                        if (data.sucesso) {{
                            // Atualizar timestamp
                            timestamp.textContent = 'Última atualização: ' + data.timestamp;
                            
                            // Recarregar a página para mostrar dados atualizados
                            setTimeout(() => {{
                                window.location.href = data.url;
                            }}, 1000);
                        }} else {{
                            throw new Error(data.erro || 'Erro desconhecido');
                        }}
                    }} catch (error) {{
                        console.error('Erro ao atualizar:', error);
                        const erroDiv = document.getElementById('erro');
                        if (erroDiv) {{
                            erroDiv.style.display = 'block';
                            erroDiv.innerHTML = `<strong>❌ Erro ao atualizar dados:</strong><br><small>${{error.message}}</small><br><small>Certifique-se de que o servidor está rodando: <code>python scripts/servidor_atualizacao.py</code></small>`;
                        }}
                        alert('Erro ao atualizar dados: ' + error.message + '\\n\\nCertifique-se de que o servidor de atualização está rodando.\\n\\nExecute: python scripts/servidor_atualizacao.py');
                        btn.disabled = false;
                        btn.textContent = '🔄 Atualizar';
                        loading.classList.remove('show');
                    }}
                }}
                
                // Destacar colunas de atributos (perfil, motivo, PnL, RR, alvos e stop atual)
                const table = document.getElementById('dataframe');
                if (table) {{
                    const headers = table.querySelectorAll('th');
                    headers.forEach((th, index) => {{
                        const colName = th.textContent.trim();
                        if (['perfil', 'motivo', 'pnl', 'rr', 'alvo1', 'alvo2', 'stop_atual'].includes(colName.toLowerCase())) {{
                            th.classList.add('atributos');
                            th.style.backgroundColor = '#ffc107';
                            // Aplicar também nas células da coluna
                            const rows = table.querySelectorAll('tr');
                            rows.forEach(row => {{
                                const cell = row.cells[index];
                                if (cell) {{
                                    cell.classList.add('atributos');
                                }}
                            }});
                        }}
                    }});
                }}
                
                // Atalho de teclado: F5 ou Ctrl+R para atualizar
                document.addEventListener('keydown', function(e) {{
                    if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {{
                        e.preventDefault();
                        atualizarDados();
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        # Salvar em arquivo temporário com nome baseado em produto_id e tipo_dado para permitir atualização
        if produto_id and tipo_dado:
            # Criar nome de arquivo baseado em produto_id e tipo_dado para permitir atualização
            temp_dir = Path(tempfile.gettempdir())
            temp_filename = f"visualizacao_produto_{produto_id}_{tipo_dado}.html"
            temp_path = temp_dir / temp_filename
        else:
            # Se não tiver produto_id e tipo_dado, usar arquivo temporário padrão
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
                temp_path = Path(f.name)
        
        # Salvar HTML
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Abrir no navegador
        webbrowser.open(f'file://{temp_path}')
        print(f"\n✅ DataFrame aberto no navegador!")
        print(f"   Arquivo: {temp_path}")
        if produto_id and tipo_dado:
            print(f"\n💡 Para usar o botão 'Atualizar' no navegador:")
            print(f"   1. Inicie o servidor de atualização em outro terminal:")
            print(f"      python scripts/servidor_atualizacao.py")
            print(f"   2. Clique no botão '🔄 Atualizar' no navegador")
            print(f"   3. Os dados serão atualizados automaticamente com os preços do CoinGecko")
        print(f"   Total de colunas exibidas: {len(df_html.columns)}")
        print(f"   Colunas de atributos: {', '.join([col for col in colunas_atributos if col in df_html.columns])}")
        return True
        
    except Exception as e:
        print(f"⚠️  Erro ao abrir HTML: {e}")
        import traceback
        traceback.print_exc()
        # Fallback para exibição simples
        print(f"\n{'='*80}")
        print(f"  {titulo}")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        return True

def _exibir_e_exportar(df, titulo, produto_id, tipo_dado):
    """Helper para exibir DataFrame e mostrar opções de exportação"""
    if df is not None and not df.empty:
        print("\n📊 DataFrame criado com sucesso!")
        print(f"   Total de registros: {len(df)}")
        if len(df.columns) <= 10:
            print(f"   Colunas: {', '.join(df.columns.tolist())}")
        print(f"   Forma: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print("\n" + "="*80)
        exibir_dataframe_bonito(df, titulo)
        print("="*80)
        mostrar_opcoes_exportacao(df, titulo, produto_id, tipo_dado)
        return True
    return False


def aplicar_visualizacao(df, visualizacao):
    """
    Aplica uma visualização salva a um DataFrame.
    Seleciona colunas, aplica ordenação, filtros e renomeia colunas.

    Args:
        df: DataFrame original
        visualizacao: Dict com configuração da visualização

    Returns:
        DataFrame filtrado, ordenado e com colunas renomeadas
    """
    if df is None or df.empty:
        return df

    # Selecionar apenas colunas que existem no DataFrame
    colunas_config = visualizacao.get('colunas', [])
    if colunas_config:
        colunas_disponiveis = [c for c in colunas_config if c in df.columns]
        if colunas_disponiveis:
            df = df[colunas_disponiveis]
        else:
            # Nenhuma coluna da visualizacao existe nos dados - retornar vazio
            print(f"Aviso: nenhuma coluna da visualizacao encontrada nos dados.")
            print(f"  Colunas configuradas: {colunas_config}")
            print(f"  Colunas disponiveis: {list(df.columns)}")
            return df.head(0)  # DataFrame vazio com mesma estrutura

    # Aplicar filtros
    if visualizacao.get('filtros'):
        for filtro in visualizacao['filtros']:
            coluna = filtro['coluna']
            operador = filtro['operador']
            valor = filtro['valor']

            if coluna not in df.columns:
                continue

            try:
                if operador == '=':
                    df = df[df[coluna] == valor]
                elif operador == '!=':
                    df = df[df[coluna] != valor]
                elif operador == '>':
                    df = df[df[coluna] > valor]
                elif operador == '<':
                    df = df[df[coluna] < valor]
                elif operador == '>=':
                    df = df[df[coluna] >= valor]
                elif operador == '<=':
                    df = df[df[coluna] <= valor]
                elif operador == 'contém':
                    df = df[df[coluna].astype(str).str.contains(str(valor), case=False, na=False)]
            except Exception:
                pass  # Ignorar erros de filtro

    # Aplicar ordenação
    if visualizacao.get('ordenacao') and visualizacao['ordenacao'].get('coluna'):
        coluna = visualizacao['ordenacao']['coluna']
        direcao = visualizacao['ordenacao'].get('direcao', 'asc')
        if coluna in df.columns:
            df = df.sort_values(by=coluna, ascending=(direcao == 'asc'))

    # Renomear colunas com labels customizados
    if visualizacao.get('colunas_labels'):
        rename_map = {}
        for col_nome, col_label in visualizacao['colunas_labels'].items():
            if col_nome in df.columns and col_label:
                rename_map[col_nome] = col_label
        if rename_map:
            df = df.rename(columns=rename_map)

    return df


def usar_visualizacao_salva(repo, produto_id):
    """Permite escolher e usar uma visualização salva"""
    visualizacoes = repo.listar_visualizacoes(produto_id)

    if not visualizacoes:
        print("Nenhuma visualização salva para este produto.")
        print("Crie visualizações em: Produtos > Gerenciar visualizações")
        return

    print("\nVisualizações disponíveis:")
    for v in visualizacoes:
        print(f"  [{v['id']}] {v['nome']}")

    viz_id = obter_input("\nID da visualização: ", tipo=int, obrigatorio=True)

    viz = repo.carregar_visualizacao(viz_id)
    if not viz or viz['produto_id'] != produto_id:
        print("Visualização não encontrada.")
        return

    # Escolher fonte de dados
    print("\nAplicar a:")
    print("1. Posições abertas")
    print("2. Posições fechadas")
    print("3. Histórico completo")

    fonte = obter_input("Opção: ", opcoes=["1", "2", "3"], obrigatorio=True)

    # filtrar_colunas=False permite que aplicar_visualizacao() selecione as colunas
    # conforme configurado no banco de dados
    if fonte == "1":
        df = display_posicoes_abertas(produto_id, formatar=True, filtrar_colunas=False)
        titulo = f"{viz['nome']} - Posições Abertas"
        tipo_dado = "viz_abertas"
    elif fonte == "2":
        df = display_posicoes_fechadas(produto_id, formatar=True, filtrar_colunas=False)
        titulo = f"{viz['nome']} - Posições Fechadas"
        tipo_dado = "viz_fechadas"
    else:
        df = display_historico_posicoes(produto_id, formatar=True, filtrar_colunas=False)
        titulo = f"{viz['nome']} - Histórico"
        tipo_dado = "viz_historico"

    if df is None or df.empty:
        print("Nenhum dado encontrado.")
        return

    # Aplicar visualização
    df_viz = aplicar_visualizacao(df, viz)

    if not _exibir_e_exportar(df_viz, titulo, produto_id, tipo_dado):
        print("Nenhum dado após aplicar filtros.")


def _obter_dados_para_visualizacao(produto_id, visualizacao):
    """
    Obtém os dados apropriados para uma visualização baseado nos filtros.
    Determina se deve buscar posições abertas, fechadas ou todas.

    IMPORTANTE: Passa filtrar_colunas=False para que as funções de display
    não apliquem filtros hard-coded de colunas. A seleção de colunas será
    feita pelo aplicar_visualizacao() usando a configuração do banco de dados.

    Args:
        produto_id: ID do produto
        visualizacao: Dicionário com configuração da visualização

    Returns:
        tuple: (DataFrame, titulo, tipo_dado)
    """
    # Verificar filtros para determinar fonte de dados
    filtros = visualizacao.get('filtros', []) or []

    # Procurar filtro de status
    status_filtro = None
    for f in filtros:
        if f.get('coluna') == 'status':
            status_filtro = f.get('valor')
            break

    nome_viz = visualizacao.get('nome', 'Visualização')

    # filtrar_colunas=False permite que aplicar_visualizacao() selecione as colunas
    # conforme configurado no banco de dados, sem interferência dos filtros hard-coded
    if status_filtro == 'open':
        # Posições abertas
        df = display_posicoes_abertas(produto_id, formatar=True, filtrar_colunas=False)
        titulo = nome_viz
        tipo_dado = f"viz_{visualizacao.get('id', 'custom')}_abertas"
    elif status_filtro == 'closed':
        # Posições fechadas
        df = display_posicoes_fechadas(produto_id, formatar=True, filtrar_colunas=False)
        titulo = nome_viz
        tipo_dado = f"viz_{visualizacao.get('id', 'custom')}_fechadas"
    else:
        # Histórico (todas)
        df = display_historico_posicoes(produto_id, formatar=True, filtrar_colunas=False)
        titulo = nome_viz
        tipo_dado = f"viz_{visualizacao.get('id', 'custom')}_historico"

    return df, titulo, tipo_dado


def submenu_posicoes(produto_id):
    """Submenu para visualização de posições usando visualizações do banco de dados"""
    repo = SQLiteRepo()

    while True:
        imprimir_secao("VISUALIZAR POSIÇÕES")

        # Listar visualizações disponíveis do banco de dados
        visualizacoes = repo.listar_visualizacoes(produto_id)

        if not visualizacoes:
            print("Nenhuma visualização configurada para este produto.")
            print("\nPara criar visualizações:")
            print("  Acesse: Menu > Produtos > Gerenciar visualizações")
            print("\n0. Voltar")
            opcao = obter_input("\nOpção: ", opcoes=["0"], obrigatorio=True)
            if opcao == "0":
                break
            continue

        # Mostrar visualizações disponíveis
        opcoes_validas = ["0"]
        for i, viz in enumerate(visualizacoes, 1):
            # Mostrar info resumida da visualização
            filtro_info = ""
            if viz.get('filtros'):
                for f in viz['filtros']:
                    if f.get('coluna') == 'status':
                        filtro_info = f" [{f.get('valor', '')}]"
                        break
            print(f"{i}. {viz['nome']}{filtro_info}")
            opcoes_validas.append(str(i))

        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=opcoes_validas, obrigatorio=True)

        if opcao == "0":
            break

        try:
            idx = int(opcao) - 1
            if 0 <= idx < len(visualizacoes):
                viz = visualizacoes[idx]

                # Obter dados apropriados
                df, titulo, tipo_dado = _obter_dados_para_visualizacao(produto_id, viz)

                if df is None or df.empty:
                    print(f"Nenhum dado encontrado para '{viz['nome']}'")
                    continue

                # Aplicar visualização (colunas, filtros adicionais, ordenação)
                df_viz = aplicar_visualizacao(df, viz)

                if not _exibir_e_exportar(df_viz, titulo, produto_id, tipo_dado):
                    print(f"Nenhum dado após aplicar filtros de '{viz['nome']}'")
        except (ValueError, IndexError):
            print("Opção inválida")


def submenu_carteira_alocacoes(produto_id):
    """Submenu para visualização de carteira e alocações"""
    while True:
        imprimir_secao("VISUALIZAR CARTEIRA E ALOCAÇÕES")
        print("1. Carteira")
        print("2. Alocações")
        print("3. Resumo completo")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            df = display_carteira(produto_id, formatar=False)
            if not _exibir_e_exportar(df, "Carteira", produto_id, "carteira"):
                print("❌ Carteira não encontrada")
        elif opcao == "2":
            df = display_alocacoes(produto_id, formatar=False)
            if not _exibir_e_exportar(df, "Alocações", produto_id, "alocacoes"):
                print("❌ Nenhuma alocação encontrada")
        elif opcao == "3":
            df = display_resumo_completo(produto_id, formatar=False)
            if not _exibir_e_exportar(df, "Resumo Completo", produto_id, "resumo"):
                print("❌ Produto não encontrado")


def submenu_valores_diarios(produto_id):
    """Submenu para visualização de valores diários"""
    while True:
        imprimir_secao("VISUALIZAR VALORES DIÁRIOS")
        print("1. Valores diários de uma posição")
        print("2. Valores diários de um ativo")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            from analytics.queries import valores_da_posicao
            posicao_id = obter_input("ID da posição: ", tipo=int, obrigatorio=True)
            df = valores_da_posicao(posicao_id)
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print(f"   Total de valores: {len(df)}")
                print("\n" + "=" * 80)
                print(df.to_string(index=False))
                print("=" * 80)
            else:
                print("❌ Nenhum valor encontrado para esta posição")
        elif opcao == "2":
            ativo = obter_input("Nome do ativo (ex: BTC): ", obrigatorio=True).upper()
            df = get_valores_ativo(ativo)
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print(f"   Total de valores: {len(df)}")
                print("\n" + "=" * 80)
                print(df.to_string(index=False))
                print("=" * 80)
            else:
                print(f"❌ Nenhum valor encontrado para {ativo}")


def submenu_graficos(produto_id):
    """Submenu para visualização de gráficos"""
    while True:
        imprimir_secao("GRÁFICOS")
        print("1. Evolução de preço (ativo)")
        print("2. Composição da carteira")
        print("3. Distribuição de alocações")
        print("4. Comparativo de ativos")
        print("5. Dashboard completo")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3", "4", "5"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            ativo = obter_input("Nome do ativo (ex: BTC): ", obrigatorio=True).upper()
            df_valores = get_valores_ativo(ativo)
            if not df_valores.empty:
                print(f"\n📊 Gerando gráfico para {ativo}...")
                fig = grafico_valores(df_valores, ativo=ativo)
                fig.show()
                print("✅ Gráfico aberto no navegador!")
            else:
                print(f"❌ Nenhum dado encontrado para {ativo}")
        elif opcao == "2":
            print("\n📊 Gerando gráfico da carteira...")
            fig = grafico_carteira(produto_id)
            if fig:
                fig.show()
                print("✅ Gráfico aberto no navegador!")
            else:
                print("❌ Erro ao gerar gráfico")
        elif opcao == "3":
            print("\n📊 Gerando gráfico de alocações...")
            fig = grafico_alocacoes(produto_id)
            if fig:
                fig.show()
                print("✅ Gráfico aberto no navegador!")
            else:
                print("❌ Erro ao gerar gráfico")
        elif opcao == "4":
            print("Digite os nomes dos ativos separados por vírgula (ex: BTC,ETH,SYRUP)")
            ativos_str = obter_input("Ativos: ", obrigatorio=True)
            ativos = [a.strip().upper() for a in ativos_str.split(",")]
            print(f"\n📊 Gerando gráfico comparativo para {', '.join(ativos)}...")
            fig = grafico_comparativo_ativos(ativos)
            if fig:
                fig.show()
                print("✅ Gráfico aberto no navegador!")
            else:
                print("❌ Erro ao gerar gráfico")
        elif opcao == "5":
            print("\n📊 Gerando dashboard completo...")
            fig = dashboard_produto(produto_id)
            if fig:
                fig.show()
                print("✅ Dashboard aberto no navegador!")
            else:
                print("❌ Erro ao gerar dashboard")


def main():
    imprimir_titulo("VISUALIZAÇÃO DE DADOS")

    repo = SQLiteRepo()

    # Listar produtos disponíveis
    produtos = repo.listar_produtos()
    if not produtos:
        print("❌ Nenhum produto encontrado.")
        return

    print("Produtos disponíveis:")
    for produto_dict in produtos:
        print(f"  ID: {produto_dict['id']} - {produto_dict['nome']} ({produto_dict['tipo']})")

    produto_id = obter_input("\nID do produto: ", tipo=int, obrigatorio=True)

    # Verificar se produto existe
    produto = repo.carregar_produto(produto_id)
    if not produto:
        print(f"❌ Produto com ID {produto_id} não encontrado!")
        return

    while True:
        imprimir_secao(f"VISUALIZAÇÃO - {produto.get('nome', f'Produto {produto_id}')}")
        print("1. Posições")
        print("2. Carteira e Alocações")
        print("3. Valores Diários")
        print("4. Gráficos")
        print("5. Todos os Produtos")
        print("0. Voltar")

        opcao = obter_input("\nOpção: ", opcoes=["0", "1", "2", "3", "4", "5"], obrigatorio=True)

        if opcao == "0":
            break
        elif opcao == "1":
            submenu_posicoes(produto_id)
        elif opcao == "2":
            submenu_carteira_alocacoes(produto_id)
        elif opcao == "3":
            submenu_valores_diarios(produto_id)
        elif opcao == "4":
            submenu_graficos(produto_id)
        elif opcao == "5":
            df = display_produtos()
            if not _exibir_e_exportar(df, "Produtos", None, "produtos"):
                print("❌ Nenhum produto encontrado")

if __name__ == "__main__":
    main()

