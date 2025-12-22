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
                <strong>🔄 Atualizando dados...</strong><br>
                <small>Buscando preços atualizados do CoinGecko e recalculando PnL...</small>
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
                        const response = await fetch(url);
                        const data = await response.json();
                        
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
                        alert('Erro ao atualizar dados. Certifique-se de que o servidor de atualização está rodando.\\n\\nExecute: python scripts/servidor_atualizacao.py');
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
        imprimir_secao("OPÇÕES DE VISUALIZAÇÃO")
        print("=== DADOS ===")
        print("1. Exibir produtos")
        print("2. Exibir posições abertas")
        print("3. Exibir posições fechadas (histórico)")
        print("4. Exibir histórico (abertas + fechadas)")
        print("5. Exibir carteira")
        print("6. Exibir alocações")
        print("7. Exibir resumo completo")
        print("8. Valores diários de uma posição")
        print("9. Valores diários de um ativo")
        print("\n=== GRÁFICOS ===")
        print("10. Gráfico de evolução de preço (ativo)")
        print("11. Gráfico de composição da carteira")
        print("12. Gráfico de distribuição de alocações")
        print("13. Comparativo de ativos")
        print("14. Dashboard completo")
        print("15. Exibir manutenções (Crypto Signals)")
        print("\n0. Voltar")
        
        opcao = obter_input(
            "\nEscolha uma opção: ",
            opcoes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
            obrigatorio=True
        )
        
        if opcao == "0":
            break
        elif opcao == "1":
            imprimir_secao("PRODUTOS")
            df = display_produtos()
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print(f"   Total de produtos: {len(df)}")
                print("\n" + "="*80)
                exibir_dataframe_bonito(df, "Produtos")
                print("="*80)
                mostrar_opcoes_exportacao(df, "Produtos", None, "produtos")
            else:
                print("❌ Nenhum produto encontrado")
        
        elif opcao == "2":
            imprimir_secao("POSIÇÕES ABERTAS")
            # Usar formatar=True para incluir preco_atual formatado
            df = display_posicoes_abertas(produto_id, formatar=True)
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print(f"   Total de posições: {len(df)}")
                print(f"   Colunas: {', '.join(df.columns.tolist())}")
                print(f"   Forma: {df.shape[0]} linhas x {df.shape[1]} colunas")
                
                # Exibir de forma bonita
                print("\n" + "="*80)
                exibir_dataframe_bonito(df, "Posições Abertas")
                print("="*80)
                
                # Opções adicionais após exibir
                mostrar_opcoes_exportacao(df, "Posições Abertas", produto_id, "posicoes_abertas")
            else:
                print("❌ Nenhuma posição aberta encontrada")
        
        elif opcao == "3":
            imprimir_secao("POSIÇÕES FECHADAS (HISTÓRICO)")
            df = display_posicoes_fechadas(produto_id, formatar=True)
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print(f"   Total de posições fechadas: {len(df)}")
                print(f"   Colunas: {', '.join(df.columns.tolist())}")
                print(f"   Forma: {df.shape[0]} linhas x {df.shape[1]} colunas")
                print("\n" + "="*80)
                exibir_dataframe_bonito(df, "Posições Fechadas")
                print("="*80)
                mostrar_opcoes_exportacao(df, "Posições Fechadas", produto_id, "posicoes_fechadas")
            else:
                print("❌ Nenhuma posição fechada encontrada")
        
        elif opcao == "4":
            imprimir_secao("HISTÓRICO (ABERTAS + FECHADAS)")
            df = display_historico_posicoes(produto_id, formatar=True)
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print(f"   Total de posições: {len(df)}")
                print(f"   Colunas: {', '.join(df.columns.tolist())}")
                print(f"   Forma: {df.shape[0]} linhas x {df.shape[1]} colunas")
                print("\n" + "="*80)
                exibir_dataframe_bonito(df, "Histórico (Abertas + Fechadas)")
                print("="*80)
                mostrar_opcoes_exportacao(df, "Histórico (Abertas + Fechadas)", produto_id, "historico_posicoes")
            else:
                print("❌ Nenhuma posição encontrada para o histórico")

        elif opcao == "5":
            imprimir_secao("CARTEIRA")
            df = display_carteira(produto_id, formatar=False)
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print("\n" + "="*80)
                exibir_dataframe_bonito(df, "Carteira")
                print("="*80)
                mostrar_opcoes_exportacao(df, "Carteira", produto_id, "carteira")
            else:
                print("❌ Carteira não encontrada")
        
        elif opcao == "6":
            imprimir_secao("ALOCAÇÕES")
            df = display_alocacoes(produto_id, formatar=False)
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print(f"   Total de alocações: {len(df)}")
                print("\n" + "="*80)
                exibir_dataframe_bonito(df, "Alocações")
                print("="*80)
                mostrar_opcoes_exportacao(df, "Alocações", produto_id, "alocacoes")
            else:
                print("❌ Nenhuma alocação encontrada")
        
        elif opcao == "7":
            imprimir_secao("RESUMO COMPLETO")
            df = display_resumo_completo(produto_id, formatar=False)
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print("\n" + "="*80)
                exibir_dataframe_bonito(df, "Resumo Completo")
                print("="*80)
                mostrar_opcoes_exportacao(df, "Resumo Completo", produto_id, "resumo")
            else:
                print("❌ Produto não encontrado")
        
        elif opcao == "8":
            imprimir_secao("VALORES DIÁRIOS DA POSIÇÃO")
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
        
        elif opcao == "9":
            imprimir_secao("VALORES DIÁRIOS DO ATIVO")
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
        
        elif opcao == "10":
            imprimir_secao("GRÁFICO DE EVOLUÇÃO DE PREÇO")
            ativo = obter_input("Nome do ativo (ex: BTC): ", obrigatorio=True).upper()
            df_valores = get_valores_ativo(ativo)
            if not df_valores.empty:
                print(f"\n📊 Gerando gráfico para {ativo}...")
                fig = grafico_valores(df_valores, ativo=ativo)
                fig.show()
                print("✅ Gráfico aberto no navegador!")
            else:
                print(f"❌ Nenhum dado encontrado para {ativo}")
        
        elif opcao == "11":
            imprimir_secao("GRÁFICO DE COMPOSIÇÃO DA CARTEIRA")
            print(f"\n📊 Gerando gráfico da carteira...")
            fig = grafico_carteira(produto_id)
            if fig:
                fig.show()
                print("✅ Gráfico aberto no navegador!")
            else:
                print("❌ Erro ao gerar gráfico")
        
        elif opcao == "12":
            imprimir_secao("GRÁFICO DE DISTRIBUIÇÃO DE ALOCAÇÕES")
            print(f"\n📊 Gerando gráfico de alocações...")
            fig = grafico_alocacoes(produto_id)
            if fig:
                fig.show()
                print("✅ Gráfico aberto no navegador!")
            else:
                print("❌ Erro ao gerar gráfico")
        
        elif opcao == "13":
            imprimir_secao("COMPARATIVO DE ATIVOS")
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
        
        elif opcao == "14":
            imprimir_secao("DASHBOARD COMPLETO")
            print(f"\n📊 Gerando dashboard completo...")
            fig = dashboard_produto(produto_id)
            if fig:
                fig.show()
                print("✅ Dashboard aberto no navegador!")
            else:
                print("❌ Erro ao gerar dashboard")

        elif opcao == "15":
            imprimir_secao("MANUTENÇÕES - CRYPTO SIGNALS")
            if produto_id != 4970919917:
                print("\n⚠️ Esta visualização está disponível apenas para o produto Crypto Signals (ID 4970919917).")
            else:
                df = display_manutencoes_signals(produto_id, formatar=True)
                if df is not None and not df.empty:
                    print("\n📊 DataFrame criado com sucesso!")
                    print(f"   Total de manutenções: {len(df)}")
                    print(f"   Colunas: {', '.join(df.columns.tolist())}")
                    print(f"   Forma: {df.shape[0]} linhas x {df.shape[1]} colunas")
                    print("\n" + "="*80)
                    exibir_dataframe_bonito(df, "Manutenções - Crypto Signals")
                    print("="*80)
                    mostrar_opcoes_exportacao(df, "Manutenções - Crypto Signals", produto_id, "manutencoes_signals")
                else:
                    print("❌ Nenhuma manutenção encontrada para o produto Crypto Signals")

        continuar = obter_input("\nDeseja visualizar outra coisa? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True).lower()
        if continuar == "n":
            break

if __name__ == "__main__":
    main()

