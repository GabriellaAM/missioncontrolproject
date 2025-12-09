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
from storage.parquet_repo import ParquetRepo
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
            return exibir_dataframe_html(df, titulo)
        
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
        exibir_dataframe_html(df, titulo)

def exibir_dataframe_html(df, titulo="DataFrame"):
    """
    Cria um arquivo HTML temporário e abre no navegador
    """
    if df is None or df.empty:
        return False
    
    try:
        # Criar HTML estilizado
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{titulo}</title>
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
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-top: 20px;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: bold;
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
            </style>
        </head>
        <body>
            <h1>{titulo}</h1>
            <div class="info">
                <strong>Total de registros:</strong> {len(df)}<br>
                <strong>Colunas:</strong> {', '.join(df.columns.tolist())}
            </div>
            {df.to_html(index=False, classes='dataframe', escape=False)}
        </body>
        </html>
        """
        
        # Salvar em arquivo temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_path = f.name
        
        # Abrir no navegador
        webbrowser.open(f'file://{temp_path}')
        print(f"\n✅ DataFrame aberto no navegador!")
        print(f"   Arquivo temporário: {temp_path}")
        return True
        
    except Exception as e:
        print(f"⚠️  Erro ao abrir HTML: {e}")
        # Fallback para exibição simples
        print(f"\n{'='*80}")
        print(f"  {titulo}")
        print(f"{'='*80}\n")
        print(df.to_string(index=False))
        return True

def main():
    imprimir_titulo("VISUALIZAÇÃO DE DADOS")
    
    repo = ParquetRepo()
    
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
        print("4. Exibir carteira")
        print("5. Exibir alocações")
        print("6. Exibir resumo completo")
        print("7. Valores diários de uma posição")
        print("8. Valores diários de um ativo")
        print("\n=== GRÁFICOS ===")
        print("9. Gráfico de evolução de preço (ativo)")
        print("10. Gráfico de composição da carteira")
        print("11. Gráfico de distribuição de alocações")
        print("12. Comparativo de ativos")
        print("13. Dashboard completo")
        print("\n0. Voltar")
        
        opcao = obter_input("\nEscolha uma opção: ", opcoes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"], obrigatorio=True)
        
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
            df = display_posicoes_abertas(produto_id, formatar=False)
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
            from analytics.queries import posicoes_fechadas
            df = posicoes_fechadas(produto_id)
            if df is not None and not df.empty:
                print("\n📊 DataFrame criado com sucesso!")
                print(f"   Total de posições fechadas: {len(df)}")
                print("\n" + "="*80)
                exibir_dataframe_bonito(df, "Posições Fechadas")
                print("="*80)
                mostrar_opcoes_exportacao(df, "Posições Fechadas", produto_id, "posicoes_fechadas")
            else:
                print("❌ Nenhuma posição fechada encontrada")
        
        elif opcao == "4":
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
        
        elif opcao == "5":
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
        
        elif opcao == "6":
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
        
        elif opcao == "7":
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
        
        elif opcao == "8":
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
        
        elif opcao == "9":
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
        
        elif opcao == "10":
            imprimir_secao("GRÁFICO DE COMPOSIÇÃO DA CARTEIRA")
            print(f"\n📊 Gerando gráfico da carteira...")
            fig = grafico_carteira(produto_id)
            if fig:
                fig.show()
                print("✅ Gráfico aberto no navegador!")
            else:
                print("❌ Erro ao gerar gráfico")
        
        elif opcao == "11":
            imprimir_secao("GRÁFICO DE DISTRIBUIÇÃO DE ALOCAÇÕES")
            print(f"\n📊 Gerando gráfico de alocações...")
            fig = grafico_alocacoes(produto_id)
            if fig:
                fig.show()
                print("✅ Gráfico aberto no navegador!")
            else:
                print("❌ Erro ao gerar gráfico")
        
        elif opcao == "12":
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
        
        elif opcao == "13":
            imprimir_secao("DASHBOARD COMPLETO")
            print(f"\n📊 Gerando dashboard completo...")
            fig = dashboard_produto(produto_id)
            if fig:
                fig.show()
                print("✅ Dashboard aberto no navegador!")
            else:
                print("❌ Erro ao gerar dashboard")
        
        continuar = obter_input("\nDeseja visualizar outra coisa? (s/n): ", opcoes=["s", "n", "S", "N"], obrigatorio=True).lower()
        if continuar == "n":
            break

if __name__ == "__main__":
    main()

