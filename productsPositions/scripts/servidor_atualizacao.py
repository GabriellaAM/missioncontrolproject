"""
Servidor HTTP simples para atualizar visualizações HTML
Permite que o botão "Atualizar" no HTML atualize os dados automaticamente
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from analytics.notebook_utils import (
    display_posicoes_abertas,
    display_posicoes_fechadas,
    display_historico_posicoes,
    display_carteira,
    display_alocacoes,
    display_resumo_completo,
    display_manutencoes_signals
)
from services.atr_stop_service import atualizar_stops_posicoes_abertas
from storage.sqlite_repo import SQLiteRepo as ATRRepo
import tempfile
import webbrowser
from datetime import datetime
import pandas as pd

class AtualizacaoHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        """Define headers CORS para permitir requisições do navegador"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests (preflight CORS)"""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b'<h1>Servidor de Atualizacao Ativo</h1><p>Use o botao Atualizar no HTML para atualizar os dados.</p>')
            return
        
        if self.path.startswith('/atualizar'):
            # Parse query parameters
            parsed_path = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed_path.query)
            
            produto_id = params.get('produto_id', [None])[0]
            tipo_dado = params.get('tipo_dado', [None])[0]
            
            if not produto_id or not tipo_dado:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'erro': 'produto_id e tipo_dado sao obrigatorios'}).encode())
                return
            
            try:
                produto_id = int(produto_id)
            except ValueError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'erro': 'produto_id deve ser um numero'}).encode())
                return
            
            # Atualizar stops ATR antes de exibir dados
            # (apenas para posições com atr_multiplier configurado)
            try:
                atr_repo = ATRRepo()
                atr_result = atualizar_stops_posicoes_abertas(
                    repo=atr_repo,
                    produto_id=produto_id,
                    verbose=False
                )
                print(f"[ATR] Produto {produto_id}: {atr_result['updated']} atualizados, "
                      f"{atr_result.get('unchanged', 0)} inalterados, "
                      f"{atr_result['skipped']} ignorados, {atr_result['breached']} breached")
            except Exception as e:
                print(f"[ATR] Erro ao atualizar stops: {e}")
                # Continua mesmo se ATR falhar - não bloqueia a visualização

            # Mapear tipo_dado para função de display
            funcoes_display = {
                'posicoes_abertas': lambda: display_posicoes_abertas(produto_id, formatar=True),
                'posicoes_fechadas': lambda: display_posicoes_fechadas(produto_id, formatar=True),
                'historico': lambda: display_historico_posicoes(produto_id, formatar=True),
                'carteira': lambda: display_carteira(produto_id, formatar=False),
                'alocacoes': lambda: display_alocacoes(produto_id, formatar=False),
                'resumo': lambda: display_resumo_completo(produto_id, formatar=False),
                'manutencoes_signals': lambda: display_manutencoes_signals(produto_id, formatar=True),
            }
            
            if tipo_dado not in funcoes_display:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'erro': f'tipo_dado "{tipo_dado}" nao suportado'}).encode())
                return
            
            # Obter dados atualizados
            try:
                df = funcoes_display[tipo_dado]()
            except Exception as e:
                import traceback
                print(f"Erro ao obter dados: {e}")
                print(traceback.format_exc())
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'erro': f'Erro ao processar dados: {str(e)}'}).encode())
                return
            
            if df is None or df.empty:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'erro': 'Nenhum dado encontrado'}).encode())
                return
            
            # Obter título
            from storage.sqlite_repo import SQLiteRepo
            repo = SQLiteRepo()
            produto = repo.carregar_produto(produto_id)
            nome_produto = produto.get('nome', f'Produto {produto_id}') if produto else f'Produto {produto_id}'
            
            titulos = {
                'posicoes_abertas': 'Posições Abertas',
                'posicoes_fechadas': 'Posições Fechadas',
                'historico': 'Histórico (Abertas + Fechadas)',
                'carteira': 'Carteira',
                'alocacoes': 'Alocações',
                'resumo': 'Resumo Completo',
                'manutencoes_signals': 'Manutenções - Crypto Signals',
            }
            
            titulo = f"{titulos.get(tipo_dado, 'Visualização')} - {nome_produto}"
            
            # Gerar HTML atualizado
            temp_dir = Path(tempfile.gettempdir())
            temp_filename = f"visualizacao_produto_{produto_id}_{tipo_dado}.html"
            temp_path = temp_dir / temp_filename
            
            # Criar HTML
            df_html = df.copy()
            # Preservar coluna pnl antes do fillna
            if 'pnl' in df_html.columns:
                # Garantir que pnl seja string e preserve valores formatados
                df_html['pnl'] = df_html['pnl'].apply(
                    lambda x: str(x) if pd.notna(x) and x is not None and str(x).strip() != '' else '—'
                )
            # Preencher outros valores NaN com string vazia
            df_html = df_html.fillna('')
            # Garantir que pnl não seja string vazia após fillna
            if 'pnl' in df_html.columns:
                df_html['pnl'] = df_html['pnl'].apply(
                    lambda x: '—' if (x == '' or str(x).strip() == '' or (isinstance(x, float) and pd.isna(x))) else str(x)
                )
            
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            colunas_atributos = ['perfil', 'motivo', 'pnl', 'rr', 'alvo1', 'alvo2', 'stop_atual']
            
            html_content = self._gerar_html(df_html, titulo, timestamp, produto_id, tipo_dado, colunas_atributos)
            
            # Salvar HTML
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Retornar sucesso com caminho do arquivo
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                'sucesso': True,
                'arquivo': str(temp_path),
                'url': f'file://{temp_path}',
                'timestamp': timestamp,
                'total_registros': len(df_html)
            }).encode())
            return
        
        # 404 para outras rotas
        self.send_response(404)
        self.send_header('Content-type', 'text/plain')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(b'Not Found')
    
    def _gerar_html(self, df_html, titulo, timestamp, produto_id, tipo_dado, colunas_atributos):
        """Gera o conteúdo HTML (reutiliza código de visualizar_dados.py)"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{titulo}</title>
            <meta charset="UTF-8">
            <meta http-equiv="refresh" content="300">
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
                .sucesso {{
                    display: none;
                    text-align: center;
                    padding: 15px;
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 5px;
                    margin: 20px 0;
                    color: #155724;
                }}
                .sucesso.show {{
                    display: block;
                }}
                .erro {{
                    display: none;
                    text-align: center;
                    padding: 15px;
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                    border-radius: 5px;
                    margin: 20px 0;
                    color: #721c24;
                }}
                .erro.show {{
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
            <div class="sucesso" id="sucesso">
                <strong>✅ Dados atualizados com sucesso!</strong>
            </div>
            <div class="erro" id="erro">
                <strong>❌ Erro ao atualizar dados. Tente novamente.</strong>
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
                const produtoId = {produto_id};
                const tipoDado = {repr(tipo_dado)};
                const servidorUrl = 'http://localhost:8765';
                
                // Função para atualizar dados
                async function atualizarDados() {{
                    const btn = document.getElementById('btnAtualizar');
                    const loading = document.getElementById('loading');
                    const sucesso = document.getElementById('sucesso');
                    const erro = document.getElementById('erro');
                    const timestamp = document.getElementById('timestamp');
                    
                    // Desabilitar botão e mostrar loading
                    btn.disabled = true;
                    btn.textContent = '🔄 Atualizando...';
                    loading.classList.add('show');
                    sucesso.classList.remove('show');
                    erro.classList.remove('show');
                    
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
                            
                            sucesso.classList.add('show');
                        }} else {{
                            throw new Error(data.erro || 'Erro desconhecido');
                        }}
                    }} catch (error) {{
                        console.error('Erro ao atualizar:', error);
                        erro.classList.add('show');
                        btn.disabled = false;
                        btn.textContent = '🔄 Atualizar';
                        loading.classList.remove('show');
                    }}
                }}
                
                // Destacar colunas de atributos
                const table = document.getElementById('dataframe');
                if (table) {{
                    const headers = table.querySelectorAll('th');
                    headers.forEach((th, index) => {{
                        const colName = th.textContent.trim();
                        if (['perfil', 'motivo', 'pnl', 'rr', 'alvo1', 'alvo2', 'stop_atual'].includes(colName.toLowerCase())) {{
                            th.classList.add('atributos');
                            th.style.backgroundColor = '#ffc107';
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
        return html_content
    
    def log_message(self, format, *args):
        """Override para suprimir logs"""
        pass

def iniciar_servidor(porta=8765):
    """Inicia o servidor HTTP"""
    servidor = HTTPServer(('localhost', porta), AtualizacaoHandler)
    print(f"✅ Servidor de atualização iniciado em http://localhost:{porta}")
    print(f"   O servidor está rodando em background.")
    print(f"   Pressione Ctrl+C para parar o servidor.")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Servidor parado.")
        servidor.shutdown()

if __name__ == "__main__":
    import sys
    porta = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    iniciar_servidor(porta)

