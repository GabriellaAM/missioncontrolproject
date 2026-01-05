"""
Dashboard Web para visualizacao de todos os produtos e posicoes
Servidor HTTP que fornece interface web completa para gerenciar o sistema
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
from datetime import datetime, date
import pandas as pd

from storage.sqlite_repo import SQLiteRepo
from analytics.notebook_utils import (
    display_posicoes_abertas,
    display_posicoes_fechadas,
    display_historico_posicoes,
    display_carteira,
    display_alocacoes,
)
from services.atr_stop_service import atualizar_stops_posicoes_abertas


# ============================================================
# FUNCOES DE VISUALIZACAO (copiadas de visualizar_dados.py)
# ============================================================

def aplicar_visualizacao(df, visualizacao):
    """Aplica uma visualizacao salva a um DataFrame."""
    if df is None or df.empty:
        return df

    # Selecionar apenas colunas que existem no DataFrame
    colunas_config = visualizacao.get('colunas', [])
    if colunas_config:
        colunas_disponiveis = [c for c in colunas_config if c in df.columns]
        if colunas_disponiveis:
            df = df[colunas_disponiveis]

    # Aplicar filtros (exceto status que ja foi aplicado)
    if visualizacao.get('filtros'):
        for filtro in visualizacao['filtros']:
            coluna = filtro['coluna']
            operador = filtro['operador']
            valor = filtro['valor']

            if coluna not in df.columns or coluna == 'status':
                continue

            try:
                if operador == '=':
                    df = df[df[coluna] == valor]
                elif operador == '!=':
                    df = df[df[coluna] != valor]
                elif operador == '>':
                    df = df[pd.to_numeric(df[coluna], errors='coerce') > float(valor)]
                elif operador == '<':
                    df = df[pd.to_numeric(df[coluna], errors='coerce') < float(valor)]
                elif operador == '>=':
                    df = df[pd.to_numeric(df[coluna], errors='coerce') >= float(valor)]
                elif operador == '<=':
                    df = df[pd.to_numeric(df[coluna], errors='coerce') <= float(valor)]
                elif operador == 'contem':
                    df = df[df[coluna].astype(str).str.contains(str(valor), case=False, na=False)]
            except Exception:
                pass

    # Aplicar ordenacao
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


def obter_dados_para_visualizacao(produto_id, visualizacao):
    """Obtem os dados apropriados para uma visualizacao baseado nos filtros."""
    filtros = visualizacao.get('filtros', []) or []

    # Procurar filtro de status
    status_filtro = None
    for f in filtros:
        if f.get('coluna') == 'status':
            status_filtro = f.get('valor')
            break

    if status_filtro == 'open':
        df = display_posicoes_abertas(produto_id, formatar=True, filtrar_colunas=False)
    elif status_filtro == 'closed':
        df = display_posicoes_fechadas(produto_id, formatar=True, filtrar_colunas=False)
    else:
        df = display_historico_posicoes(produto_id, formatar=True, filtrar_colunas=False)

    return df


# ============================================================
# HTML TEMPLATES
# ============================================================

def get_base_styles():
    """Estilos CSS compartilhados"""
    return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1a1a2e;
            color: #eee;
            min-height: 100vh;
        }
        .navbar {
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar h1 { font-size: 1.5em; color: #4ecca3; }
        .navbar a, .navbar-links a {
            color: #4ecca3;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 5px;
            transition: all 0.3s;
            margin-left: 5px;
        }
        .navbar a:hover, .navbar-links a:hover {
            background-color: #4ecca3;
            color: #1a1a2e;
        }
        .navbar-links { display: flex; gap: 5px; flex-wrap: wrap; }
        .container { max-width: 1400px; margin: 0 auto; padding: 30px; }
        .card {
            background: linear-gradient(135deg, #16213e 0%, #1f2833 100%);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .card:hover {
            box-shadow: 0 8px 25px rgba(78, 204, 163, 0.15);
        }
        .card h2 { color: #4ecca3; margin-bottom: 15px; font-size: 1.3em; }
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .badge-spot { background-color: #4ecca3; color: #1a1a2e; }
        .badge-perpetuos { background-color: #ff6b6b; color: white; }
        .badge-outro { background-color: #ffd93d; color: #1a1a2e; }
        .badge-success { background-color: #4ecca3; color: #1a1a2e; }
        .badge-danger { background-color: #ff6b6b; color: white; }
        .stats { display: flex; gap: 15px; margin-top: 15px; flex-wrap: wrap; }
        .stat {
            background: rgba(78, 204, 163, 0.1);
            padding: 10px 15px;
            border-radius: 8px;
            border-left: 3px solid #4ecca3;
            flex: 1;
            min-width: 100px;
        }
        .stat-label { font-size: 0.75em; color: #888; }
        .stat-value { font-size: 1.1em; font-weight: bold; color: #4ecca3; }
        .btn {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s;
            cursor: pointer;
            border: none;
            font-size: 0.9em;
        }
        .btn-primary { background-color: #4ecca3; color: #1a1a2e; }
        .btn-primary:hover { background-color: #3db892; }
        .btn-secondary {
            background-color: transparent;
            color: #4ecca3;
            border: 2px solid #4ecca3;
        }
        .btn-secondary:hover { background-color: #4ecca3; color: #1a1a2e; }
        .btn-danger { background-color: #ff6b6b; color: white; }
        .btn-danger:hover { background-color: #ff5252; }
        .btn-warning { background-color: #ffa726; color: #1a1a2e; }
        .btn-warning:hover { background-color: #ff9800; }
        .btn-sm { padding: 6px 12px; font-size: 0.8em; }
        .actions { margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: #16213e;
            border-radius: 10px;
            overflow: hidden;
            font-size: 0.9em;
        }
        th {
            background: linear-gradient(135deg, #4ecca3 0%, #3db892 100%);
            color: #1a1a2e;
            padding: 12px 10px;
            text-align: left;
            font-weight: bold;
            white-space: nowrap;
        }
        td { padding: 10px; border-bottom: 1px solid #2a2a4a; }
        tr:hover { background-color: rgba(78, 204, 163, 0.1); }
        .loading {
            text-align: center;
            padding: 40px;
            color: #4ecca3;
            display: none;
        }
        .loading.show { display: block; }
        .loading-spinner {
            border: 4px solid #2a2a4a;
            border-top: 4px solid #4ecca3;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .tab {
            padding: 10px 20px;
            background: #16213e;
            border: 2px solid #2a2a4a;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            color: #888;
            text-decoration: none;
        }
        .tab:hover, .tab.active { border-color: #4ecca3; color: #4ecca3; }
        .tab.active { background: rgba(78, 204, 163, 0.2); }
        .timestamp { color: #666; font-size: 0.9em; margin-bottom: 20px; }
        .empty-state { text-align: center; padding: 40px; color: #666; }
        .empty-state h3 { color: #888; margin-bottom: 10px; }
        .alert {
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        .alert.show { display: block; }
        .alert-success { background: rgba(78, 204, 163, 0.2); border: 1px solid #4ecca3; color: #4ecca3; }
        .alert-error { background: rgba(255, 107, 107, 0.2); border: 1px solid #ff6b6b; color: #ff6b6b; }
        /* Forms */
        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #4ecca3;
            font-weight: bold;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #2a2a4a;
            border-radius: 8px;
            background: #16213e;
            color: #eee;
            font-size: 1em;
        }
        .form-group input:focus, .form-group select:focus {
            border-color: #4ecca3;
            outline: none;
        }
        .form-row { display: flex; gap: 20px; }
        .form-row .form-group { flex: 1; }
        .viz-list { list-style: none; }
        .viz-item {
            background: rgba(78, 204, 163, 0.1);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .viz-item:hover { background: rgba(78, 204, 163, 0.2); }
        .menu-section { margin-bottom: 30px; }
        .menu-section h3 {
            color: #4ecca3;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #2a2a4a;
        }
        .menu-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        .menu-item {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            text-decoration: none;
            color: #eee;
            border: 2px solid #2a2a4a;
            transition: all 0.3s;
        }
        .menu-item:hover {
            border-color: #4ecca3;
            transform: translateY(-3px);
        }
        .menu-item-icon { font-size: 2em; margin-bottom: 10px; }
        .menu-item-label { font-weight: bold; }
        .table-container { overflow-x: auto; }
    """


def get_navbar(current_page=""):
    """Gera a barra de navegacao"""
    return f"""
    <nav class="navbar">
        <a href="/" style="text-decoration: none;"><h1>Products & Positions</h1></a>
        <div class="navbar-links">
            <a href="/" class="{'active' if current_page == 'home' else ''}">Dashboard</a>
            <a href="/menu" class="{'active' if current_page == 'menu' else ''}">Menu</a>
        </div>
    </nav>
    """


def get_dashboard_html(produtos, stats, repo):
    """Gera HTML da pagina principal do dashboard"""
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    cards_html = ""
    for p in produtos:
        prod_stats = stats.get(p['id'], {})
        posicoes_abertas = prod_stats.get('posicoes_abertas', 0)

        cards_html += f"""
        <div class="card">
            <h2>{p['nome']}</h2>
            <p style="color: #888; margin: 10px 0;">{posicoes_abertas} posicoes abertas</p>
            <div class="actions">
                <a href="/produto/{p['id']}" class="btn btn-primary">Ver Detalhes</a>
            </div>
        </div>
        """

    if not cards_html:
        cards_html = """
        <div class="empty-state">
            <h3>Nenhum produto encontrado</h3>
            <p>Crie um produto para comecar.</p>
            <a href="/produto/novo" class="btn btn-primary" style="margin-top: 20px;">Criar Produto</a>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard - Products & Positions</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar('home')}
        <div class="container">
            <p class="timestamp">Ultima atualizacao: {timestamp}</p>
            <div class="actions" style="margin-bottom: 20px;">
                <a href="/produto/novo" class="btn btn-primary">+ Novo Produto</a>
                <a href="/menu" class="btn btn-secondary">Menu Completo</a>
            </div>
            <div class="card-grid">{cards_html}</div>
        </div>
        <script>setTimeout(() => location.reload(), 300000);</script>
    </body>
    </html>
    """


def get_menu_html():
    """Gera HTML da pagina de menu completo"""
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Menu - Products & Positions</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar('menu')}
        <div class="container">
            <h2 style="color: #4ecca3; margin-bottom: 30px;">Menu de Operacoes</h2>

            <div class="menu-section">
                <h3>Produtos</h3>
                <div class="menu-grid">
                    <a href="/produto/novo" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Criar Produto</div>
                    </a>
                    <a href="/produtos/editar" class="menu-item">
                        <div class="menu-item-icon">E</div>
                        <div class="menu-item-label">Editar Produto</div>
                    </a>
                    <a href="/produtos/deletar" class="menu-item">
                        <div class="menu-item-icon">X</div>
                        <div class="menu-item-label">Deletar Produto</div>
                    </a>
                </div>
            </div>

            <div class="menu-section">
                <h3>Posicoes</h3>
                <div class="menu-grid">
                    <a href="/posicao/nova" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Criar Posicao</div>
                    </a>
                    <a href="/posicoes/editar" class="menu-item">
                        <div class="menu-item-icon">E</div>
                        <div class="menu-item-label">Editar Posicao</div>
                    </a>
                    <a href="/posicoes/deletar" class="menu-item">
                        <div class="menu-item-icon">X</div>
                        <div class="menu-item-label">Deletar Posicao</div>
                    </a>
                </div>
            </div>

            <div class="menu-section">
                <h3>Stops</h3>
                <div class="menu-grid">
                    <a href="/stop/novo" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Adicionar Stop</div>
                    </a>
                    <a href="/stops/atualizar-atr" class="menu-item">
                        <div class="menu-item-icon">A</div>
                        <div class="menu-item-label">Atualizar ATR</div>
                    </a>
                </div>
            </div>

            <div class="menu-section">
                <h3>Visualizacao</h3>
                <div class="menu-grid">
                    <a href="/" class="menu-item">
                        <div class="menu-item-icon">D</div>
                        <div class="menu-item-label">Dashboard</div>
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_produto_html(produto, visualizacoes, repo):
    """Gera HTML da pagina de detalhes de um produto com visualizacoes salvas"""
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    produto_id = produto['id']
    nome = produto['nome']
    tipo = produto.get('tipo', 'Outro')

    # Se tiver visualizacoes, mostrar a primeira por padrao
    tabs_html = ""
    content_html = ""
    n_posicoes = 0
    viz_nome = ""

    if visualizacoes:
        for i, viz in enumerate(visualizacoes):
            active_class = 'active' if i == 0 else ''
            tabs_html += f'<a href="/produto/{produto_id}/viz/{viz["id"]}" class="tab {active_class}">{viz["nome"]}</a>'

        # Mostrar primeira visualizacao
        first_viz = visualizacoes[0]
        viz_nome = first_viz['nome']
        df = obter_dados_para_visualizacao(produto_id, first_viz)
        df_viz = aplicar_visualizacao(df, first_viz)

        if df_viz is not None and not df_viz.empty:
            content_html = f'<div class="table-container">{df_viz.to_html(index=False, classes="dataframe", escape=False)}</div>'
            n_posicoes = len(df_viz)
        else:
            content_html = '<div class="empty-state"><p>Nenhum dado encontrado</p></div>'
    else:
        # Sem visualizacoes - mostrar posicoes abertas padrao
        tabs_html = f'''
            <a href="/produto/{produto_id}/abertas" class="tab active">Posicoes Abertas</a>
            <a href="/produto/{produto_id}/fechadas" class="tab">Posicoes Fechadas</a>
        '''
        viz_nome = "Posições Abertas"
        df = display_posicoes_abertas(produto_id, formatar=True, filtrar_colunas=False)
        if df is not None and not df.empty:
            content_html = f'<div class="table-container">{df.to_html(index=False, classes="dataframe", escape=False)}</div>'
            n_posicoes = len(df)
        else:
            content_html = '<div class="empty-state"><p>Nenhuma posicao aberta</p></div>'

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{nome} - Dashboard</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <p class="timestamp">Ultima atualizacao: {timestamp}</p>

            <div class="card" style="margin-bottom: 30px;">
                <h2>{nome}</h2>
                <span class="badge badge-{'spot' if 'spot' in tipo.lower() else 'perpetuos' if 'perp' in tipo.lower() else 'outro'}">{tipo}</span>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-label">Data Inicio</div>
                        <div class="stat-value">{produto.get('data_inicio', 'N/A')}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Posicoes</div>
                        <div class="stat-value">{n_posicoes}</div>
                    </div>
                </div>
                <div class="actions">
                    <button class="btn btn-primary" onclick="atualizarDados()">Atualizar Dados</button>
                    <a href="/produto/{produto_id}/editar" class="btn btn-secondary">Editar Produto</a>
                </div>
            </div>

            <div id="loading" class="loading">
                <div class="loading-spinner"></div>
                <p>Atualizando dados...</p>
            </div>
            <div id="alert" class="alert"></div>

            <div class="tabs">{tabs_html}</div>

            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h2 style="margin: 0;">{viz_nome}</h2>
                    <div class="actions" style="margin: 0;">
                        <a href="/posicao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Posicao</a>
                        <a href="/posicoes/editar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Editar Posicao</a>
                        <a href="/stop/novo?produto_id={produto_id}" class="btn btn-sm btn-secondary">+ Stop</a>
                        <a href="/atr/config?produto_id={produto_id}" class="btn btn-sm btn-secondary">ATR Stop</a>
                        <a href="/posicao/fechar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Fechar Posicao</a>
                    </div>
                </div>
                {content_html}
            </div>
        </div>

        <script>
            const produtoId = {produto_id};
            async function atualizarDados() {{
                const loading = document.getElementById('loading');
                const alert = document.getElementById('alert');
                loading.classList.add('show');
                alert.classList.remove('show');
                try {{
                    const response = await fetch('/api/atualizar/' + produtoId);
                    const data = await response.json();
                    if (data.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Dados atualizados!';
                        setTimeout(() => location.reload(), 1000);
                    }} else {{
                        throw new Error(data.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }} finally {{
                    loading.classList.remove('show');
                }}
            }}
        </script>
    </body>
    </html>
    """


def get_visualizacao_html(produto, visualizacao, df_viz):
    """Gera HTML para uma visualizacao especifica"""
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    produto_id = produto['id']
    nome_produto = produto['nome']
    nome_viz = visualizacao['nome']
    tipo = produto.get('tipo', 'Outro')

    repo = SQLiteRepo()
    visualizacoes = repo.listar_visualizacoes(produto_id)

    tabs_html = ""
    for viz in visualizacoes:
        active_class = 'active' if viz['id'] == visualizacao['id'] else ''
        tabs_html += f'<a href="/produto/{produto_id}/viz/{viz["id"]}" class="tab {active_class}">{viz["nome"]}</a>'

    if df_viz is not None and not df_viz.empty:
        content_html = f'<div class="table-container">{df_viz.to_html(index=False, classes="dataframe", escape=False)}</div>'
        n_posicoes = len(df_viz)
    else:
        content_html = '<div class="empty-state"><p>Nenhum dado encontrado para esta visualizacao</p></div>'
        n_posicoes = 0

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{nome_viz} - {nome_produto}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <p class="timestamp">Ultima atualizacao: {timestamp}</p>

            <div class="card" style="margin-bottom: 20px;">
                <h2>{nome_produto}</h2>
                <span class="badge badge-{'spot' if 'spot' in tipo.lower() else 'perpetuos' if 'perp' in tipo.lower() else 'outro'}">{tipo}</span>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-label">Data Inicio</div>
                        <div class="stat-value">{produto.get('data_inicio', 'N/A')}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Posicoes</div>
                        <div class="stat-value">{n_posicoes}</div>
                    </div>
                </div>
                <div class="actions">
                    <button class="btn btn-primary" onclick="atualizarDados()">Atualizar Dados</button>
                    <a href="/produto/{produto_id}/editar" class="btn btn-secondary">Editar Produto</a>
                </div>
            </div>

            <div id="loading" class="loading">
                <div class="loading-spinner"></div>
                <p>Atualizando dados...</p>
            </div>
            <div id="alert" class="alert"></div>

            <div class="tabs">{tabs_html}</div>

            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h2 style="margin: 0;">{nome_viz}</h2>
                    <div class="actions" style="margin: 0;">
                        <a href="/posicao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Posicao</a>
                        <a href="/posicoes/editar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Editar Posicao</a>
                        <a href="/stop/novo?produto_id={produto_id}" class="btn btn-sm btn-secondary">+ Stop</a>
                        <a href="/atr/config?produto_id={produto_id}" class="btn btn-sm btn-secondary">ATR Stop</a>
                        <a href="/posicao/fechar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Fechar Posicao</a>
                    </div>
                </div>
                {content_html}
            </div>
        </div>

        <script>
            const produtoId = {produto_id};
            async function atualizarDados() {{
                const loading = document.getElementById('loading');
                const alert = document.getElementById('alert');
                loading.classList.add('show');
                alert.classList.remove('show');
                try {{
                    const response = await fetch('/api/atualizar/' + produtoId);
                    const data = await response.json();
                    if (data.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Dados atualizados!';
                        setTimeout(() => location.reload(), 1000);
                    }} else {{
                        throw new Error(data.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }} finally {{
                    loading.classList.remove('show');
                }}
            }}
        </script>
    </body>
    </html>
    """


def get_form_produto_html(produto=None):
    """Gera formulario para criar/editar produto"""
    is_edit = produto is not None
    titulo = "Editar Produto" if is_edit else "Novo Produto"
    action = f"/api/produto/{produto['id']}/editar" if is_edit else "/api/produto/criar"

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{titulo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 600px; margin: 0 auto;">
                <h2>{titulo}</h2>
                <div id="alert" class="alert"></div>
                <form id="produtoForm">
                    <div class="form-group">
                        <label>Nome do Produto</label>
                        <input type="text" name="nome" required value="{produto.get('nome', '') if is_edit else ''}">
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Tipo</label>
                            <select name="tipo">
                                <option value="Spot" {'selected' if is_edit and produto.get('tipo') == 'Spot' else ''}>Spot</option>
                                <option value="Perpetuos" {'selected' if is_edit and produto.get('tipo') == 'Perpetuos' else ''}>Perpetuos</option>
                                <option value="Outro" {'selected' if is_edit and produto.get('tipo') not in ['Spot', 'Perpetuos'] else ''}>Outro</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Capital Inicial (USD)</label>
                            <input type="number" name="capital_inicial" step="0.01" value="{produto.get('capital_inicial', '') if is_edit else ''}">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Data de Inicio</label>
                        <input type="date" name="data_inicio" value="{produto.get('data_inicio', '') if is_edit else date.today().isoformat()}">
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">{'Salvar' if is_edit else 'Criar'}</button>
                        <a href="/" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            document.getElementById('produtoForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('{action}', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Produto salvo com sucesso!';
                        setTimeout(() => window.location.href = '/', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    </body>
    </html>
    """


def get_form_posicao_html(produto_id=None, produtos=None):
    """Gera formulario para criar posicao"""
    produtos_options = ""
    if produtos:
        for p in produtos:
            selected = 'selected' if produto_id and p['id'] == produto_id else ''
            produtos_options += f'<option value="{p["id"]}" {selected}>{p["nome"]}</option>'

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nova Posicao</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 700px; margin: 0 auto;">
                <h2>Nova Posicao</h2>
                <div id="alert" class="alert"></div>
                <form id="posicaoForm">
                    <div class="form-group">
                        <label>Produto</label>
                        <select name="produto_id" required>{produtos_options}</select>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Ativo (ex: BTC, ETH)</label>
                            <input type="text" name="ativo" required placeholder="BTC">
                        </div>
                        <div class="form-group">
                            <label>CoinGecko ID</label>
                            <input type="text" name="coingecko_id" placeholder="bitcoin">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Side</label>
                            <select name="side">
                                <option value="long">Long</option>
                                <option value="short">Short</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Data de Entrada</label>
                            <input type="date" name="data_entrada" value="{date.today().isoformat()}">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Preco de Entrada (USD)</label>
                            <input type="number" name="preco_entrada" step="0.00000001" required>
                        </div>
                        <div class="form-group">
                            <label>Quantidade (opcional)</label>
                            <input type="number" name="quantidade" step="0.00000001">
                        </div>
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Criar Posicao</button>
                        <a href="/" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            document.getElementById('posicaoForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/posicao/criar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Posicao criada!';
                        setTimeout(() => window.location.href = '/produto/' + obj.produto_id, 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    </body>
    </html>
    """


def get_lista_produtos_html(produtos, acao="editar"):
    """Lista produtos para selecao (editar/deletar)"""
    titulo = "Editar Produto" if acao == "editar" else "Deletar Produto"

    items_html = ""
    for p in produtos:
        if acao == "editar":
            link = f"/produto/{p['id']}/editar"
            btn_class = "btn-primary"
            btn_text = "Editar"
        else:
            link = f"/produto/{p['id']}/deletar"
            btn_class = "btn-danger"
            btn_text = "Deletar"

        items_html += f"""
        <div class="viz-item">
            <div>
                <strong>{p['nome']}</strong>
                <span class="badge badge-outro" style="margin-left: 10px;">{p.get('tipo', 'Outro')}</span>
            </div>
            <a href="{link}" class="btn btn-sm {btn_class}">{btn_text}</a>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{titulo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 600px; margin: 0 auto;">
                <h2>{titulo}</h2>
                <p style="color: #888; margin-bottom: 20px;">Selecione um produto:</p>
                <div class="viz-list">{items_html}</div>
                <div class="actions" style="margin-top: 20px;">
                    <a href="/" class="btn btn-secondary">Voltar</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_confirmar_delete_html(produto):
    """Pagina de confirmacao de exclusao"""
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Confirmar Exclusao</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 500px; margin: 0 auto; text-align: center;">
                <h2 style="color: #ff6b6b;">Confirmar Exclusao</h2>
                <p style="margin: 20px 0;">Tem certeza que deseja deletar o produto:</p>
                <p style="font-size: 1.3em; color: #4ecca3; font-weight: bold;">{produto['nome']}</p>
                <p style="color: #ff6b6b; margin: 20px 0;">Esta acao ira deletar todas as posicoes, stops e visualizacoes associadas!</p>
                <div id="alert" class="alert"></div>
                <div class="actions" style="justify-content: center;">
                    <button class="btn btn-danger" onclick="deletarProduto()">Sim, Deletar</button>
                    <a href="/" class="btn btn-secondary">Cancelar</a>
                </div>
            </div>
        </div>
        <script>
            async function deletarProduto() {{
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/produto/{produto["id"]}/deletar', {{
                        method: 'POST'
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Produto deletado!';
                        setTimeout(() => window.location.href = '/', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }}
        </script>
    </body>
    </html>
    """


def get_lista_posicoes_html(produto, posicoes, acao="editar"):
    """Lista posicoes de um produto para selecao (editar/stop/fechar/atr)"""
    titulo_map = {
        "editar": "Editar Posicao",
        "stop": "Adicionar Stop",
        "fechar": "Fechar Posicao",
        "atr": "Configurar ATR Stop"
    }
    titulo = titulo_map.get(acao, "Selecionar Posicao")

    items_html = ""
    is_spot = 'spot' in produto.get('tipo', '').lower()

    if posicoes is not None and not posicoes.empty:
        for _, pos in posicoes.iterrows():
            pos_id = pos.get('id') or pos.get('ID')
            ativo = pos.get('ativo') or pos.get('Ativo', 'N/A')
            side = pos.get('side') or pos.get('tipo') or pos.get('Tipo', 'N/A')
            preco = pos.get('preco_entrada') or pos.get('Preço Entrada', 0)

            if acao == "editar":
                link = f"/posicao/{pos_id}/editar?produto_id={produto['id']}"
                btn_class = "btn-primary"
                btn_text = "Editar"
            elif acao == "stop":
                link = f"/posicao/{pos_id}/stop?produto_id={produto['id']}"
                btn_class = "btn-warning"
                btn_text = "+ Stop"
            elif acao == "atr":
                link = f"/posicao/{pos_id}/atr?produto_id={produto['id']}"
                btn_class = "btn-secondary"
                btn_text = "Configurar"
            else:  # fechar
                link = f"/posicao/{pos_id}/fechar?produto_id={produto['id']}"
                btn_class = "btn-danger"
                btn_text = "Fechar"

            # Só mostra badge de side se não for spot
            if is_spot:
                side_badge_html = ""
            else:
                tipo_badge = 'success' if side.lower() == 'long' else 'danger'
                side_badge_html = f'<span class="badge badge-{tipo_badge}" style="margin-left: 10px;">{side}</span>'

            items_html += f"""
            <div class="viz-item">
                <div>
                    <strong>{ativo}</strong>
                    {side_badge_html}
                    <span style="margin-left: 10px; color: #888;">${preco:,.4f}</span>
                </div>
                <a href="{link}" class="btn btn-sm {btn_class}">{btn_text}</a>
            </div>
            """
    else:
        items_html = '<p style="text-align: center; color: #888;">Nenhuma posicao aberta</p>'

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{titulo} - {produto['nome']}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 600px; margin: 0 auto;">
                <h2>{titulo}</h2>
                <p style="color: #4ecca3; margin-bottom: 20px;">{produto['nome']}</p>
                <div class="viz-list">{items_html}</div>
                <div class="actions" style="margin-top: 20px;">
                    <a href="/produto/{produto['id']}" class="btn btn-secondary">Voltar</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_form_adicionar_stop_html(produto, posicao):
    """Formulario para adicionar stop a uma posicao"""
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', 'N/A')
    side = posicao.get('side') or posicao.get('tipo') or posicao.get('Tipo', 'N/A')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    is_spot = 'spot' in produto.get('tipo', '').lower()
    side_info = "" if is_spot else f" ({side})"

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Adicionar Stop - {ativo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 600px; margin: 0 auto;">
                <h2>Adicionar Stop</h2>
                <p style="color: #4ecca3; margin-bottom: 20px;">
                    {ativo}{side_info} - Entrada: ${preco_entrada:,.4f}
                </p>
                <div id="alert" class="alert"></div>
                <form id="stopForm">
                    <input type="hidden" name="posicao_id" value="{pos_id}">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Preco do Stop</label>
                            <input type="number" name="preco" step="0.00000001" required>
                        </div>
                        <div class="form-group">
                            <label>Data do Stop</label>
                            <input type="date" name="data_stop" value="{date.today().isoformat()}">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Motivo (opcional)</label>
                        <input type="text" name="motivo" placeholder="Ex: Stop loss tecnico">
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Adicionar Stop</button>
                        <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            document.getElementById('stopForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/stop/criar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Stop adicionado!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    </body>
    </html>
    """


def get_form_fechar_posicao_html(produto, posicao):
    """Formulario para fechar uma posicao"""
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', 'N/A')
    side = posicao.get('side') or posicao.get('tipo') or posicao.get('Tipo', 'N/A')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    is_spot = 'spot' in produto.get('tipo', '').lower()
    side_info = "" if is_spot else f" ({side})"

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fechar Posicao - {ativo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 600px; margin: 0 auto;">
                <h2>Fechar Posicao</h2>
                <p style="color: #4ecca3; margin-bottom: 20px;">
                    {ativo}{side_info} - Entrada: ${preco_entrada:,.4f}
                </p>
                <div id="alert" class="alert"></div>
                <form id="fecharForm">
                    <input type="hidden" name="posicao_id" value="{pos_id}">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Preco de Saida</label>
                            <input type="number" name="preco_saida" step="0.00000001" required>
                        </div>
                        <div class="form-group">
                            <label>Data de Saida</label>
                            <input type="date" name="data_saida" value="{date.today().isoformat()}">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Motivo (opcional)</label>
                        <input type="text" name="motivo" placeholder="Ex: Stop atingido, Take profit">
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-danger">Fechar Posicao</button>
                        <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            document.getElementById('fecharForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/posicao/fechar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Posicao fechada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    </body>
    </html>
    """


def get_form_editar_posicao_html(produto, posicao):
    """Formulario para editar uma posicao existente"""
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', '')
    coingecko_id = posicao.get('coingecko_id') or posicao.get('CoinGecko ID', '')
    side = posicao.get('side') or posicao.get('tipo') or posicao.get('Tipo', 'long')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    quantidade = posicao.get('quantidade') or posicao.get('Quantidade', '')
    data_entrada = posicao.get('data_entrada') or posicao.get('Data Entrada', date.today().isoformat())
    is_spot = 'spot' in produto.get('tipo', '').lower()

    # Para spot, não mostra seletor de tipo
    if is_spot:
        tipo_field_html = f"""
                        <div class="form-group">
                            <label>Data de Entrada</label>
                            <input type="date" name="data_entrada" value="{data_entrada}">
                        </div>
                        <input type="hidden" name="tipo" value="long">"""
    else:
        tipo_field_html = f"""
                        <div class="form-group">
                            <label>Tipo</label>
                            <select name="tipo" required>
                                <option value="long" {'selected' if side.lower() == 'long' else ''}>Long</option>
                                <option value="short" {'selected' if side.lower() == 'short' else ''}>Short</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Data de Entrada</label>
                            <input type="date" name="data_entrada" value="{data_entrada}">
                        </div>"""

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Editar Posicao - {ativo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 600px; margin: 0 auto;">
                <h2>Editar Posicao</h2>
                <p style="color: #4ecca3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <form id="editarPosicaoForm">
                    <input type="hidden" name="posicao_id" value="{pos_id}">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Ativo</label>
                            <input type="text" name="ativo" value="{ativo}" required>
                        </div>
                        <div class="form-group">
                            <label>CoinGecko ID</label>
                            <input type="text" name="coingecko_id" value="{coingecko_id or ''}">
                        </div>
                    </div>
                    <div class="form-row">
                        {tipo_field_html}
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Preco de Entrada</label>
                            <input type="number" name="preco_entrada" step="0.00000001" value="{preco_entrada}" required>
                        </div>
                        <div class="form-group">
                            <label>Quantidade</label>
                            <input type="number" name="quantidade" step="0.00000001" value="{quantidade or ''}">
                        </div>
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Salvar</button>
                        <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            document.getElementById('editarPosicaoForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/posicao/editar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Posicao atualizada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});
        </script>
    </body>
    </html>
    """


def get_form_atr_stop_html(produto, posicao):
    """Formulario para configurar ATR Trailing Stop"""
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', 'N/A')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    current_period = posicao.get('atr_period') or 14
    current_mult = posicao.get('atr_multiplier') or 3.0

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ATR Stop - {ativo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 600px; margin: 0 auto;">
                <h2>Configurar ATR Trailing Stop</h2>
                <p style="color: #4ecca3; margin-bottom: 20px;">
                    {ativo} - Entrada: ${preco_entrada:,.4f}
                </p>
                <p style="color: #888; font-size: 0.9em; margin-bottom: 20px;">
                    O ATR Trailing Stop calcula automaticamente o stop baseado na volatilidade do ativo.
                    O stop é atualizado quando você clica em "Atualizar Dados".
                </p>
                <div id="alert" class="alert"></div>
                <form id="atrForm">
                    <input type="hidden" name="posicao_id" value="{pos_id}">
                    <div class="form-row">
                        <div class="form-group">
                            <label>ATR Period (dias)</label>
                            <input type="number" name="atr_period" value="{current_period}" min="1" max="100" required>
                            <small style="color: #888;">Periodo para calculo do ATR (padrao: 14)</small>
                        </div>
                        <div class="form-group">
                            <label>ATR Multiplier</label>
                            <input type="number" name="atr_multiplier" value="{current_mult}" step="0.1" min="0.5" max="10" required>
                            <small style="color: #888;">Multiplicador do ATR (padrao: 3.0)</small>
                        </div>
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Salvar Configuracao</button>
                        <button type="button" class="btn btn-danger" onclick="removerATR()">Remover ATR</button>
                        <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            document.getElementById('atrForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const data = new FormData(form);
                const obj = Object.fromEntries(data.entries());
                const alert = document.getElementById('alert');

                try {{
                    const response = await fetch('/api/posicao/atr', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(obj)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'ATR configurado! O stop sera calculado na proxima atualizacao.';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1500);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});

            async function removerATR() {{
                const alert = document.getElementById('alert');
                if (!confirm('Remover configuracao de ATR desta posicao?')) return;

                try {{
                    const response = await fetch('/api/posicao/atr', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            posicao_id: {pos_id},
                            atr_period: null,
                            atr_multiplier: null
                        }})
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'ATR removido!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }}
        </script>
    </body>
    </html>
    """


# ============================================================
# HTTP SERVER
# ============================================================

class DashboardHandler(BaseHTTPRequestHandler):
    """Handler para requisicoes HTTP do dashboard"""

    def _set_headers(self, status=200, content_type='text/html'):
        self.send_response(status)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, data, status=200):
        self._set_headers(status, 'application/json')
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode('utf-8'))

    def _send_html(self, html, status=200):
        self._set_headers(status, 'text/html; charset=utf-8')
        self.wfile.write(html.encode('utf-8'))

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(post_data) if post_data else {}
        except json.JSONDecodeError:
            data = {}

        repo = SQLiteRepo()
        path = self.path

        # Criar produto
        if path == '/api/produto/criar':
            try:
                produto_id = repo.salvar_produto(
                    nome=data.get('nome'),
                    tipo=data.get('tipo', 'Outro'),
                    data_inicio=data.get('data_inicio'),
                    capital_inicial=float(data.get('capital_inicial', 0)) if data.get('capital_inicial') else None
                )
                self._send_json({'sucesso': True, 'produto_id': produto_id})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Editar produto
        if path.startswith('/api/produto/') and path.endswith('/editar'):
            try:
                produto_id = int(path.split('/')[3])
                repo.atualizar_produto(
                    produto_id=produto_id,
                    nome=data.get('nome'),
                    tipo=data.get('tipo'),
                    data_inicio=data.get('data_inicio'),
                    capital_inicial=float(data.get('capital_inicial')) if data.get('capital_inicial') else None
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Deletar produto
        if path.startswith('/api/produto/') and path.endswith('/deletar'):
            try:
                produto_id = int(path.split('/')[3])
                repo.deletar_produto(produto_id, forcar=True)
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar posicao
        if path == '/api/posicao/criar':
            try:
                posicao_id = repo.salvar_posicao(
                    produto_id=int(data.get('produto_id')),
                    ativo=data.get('ativo'),
                    side=data.get('side', 'long'),
                    data_entrada=data.get('data_entrada'),
                    preco_entrada=float(data.get('preco_entrada')),
                    coingecko_id=data.get('coingecko_id') or None
                )
                # Salvar quantidade como atributo se fornecida
                if data.get('quantidade'):
                    repo.salvar_atributos_posicao(
                        posicao_id=posicao_id,
                        produto_id=int(data.get('produto_id')),
                        quantidade=float(data.get('quantidade'))
                    )
                self._send_json({'sucesso': True, 'posicao_id': posicao_id})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Editar posicao
        if path == '/api/posicao/editar':
            try:
                posicao_id = int(data.get('posicao_id'))
                repo.atualizar_posicao(
                    posicao_id=posicao_id,
                    ativo=data.get('ativo'),
                    side=data.get('tipo'),
                    data_entrada=data.get('data_entrada'),
                    preco_entrada=float(data.get('preco_entrada')),
                    coingecko_id=data.get('coingecko_id') or None
                )
                # Atualizar quantidade como atributo se fornecida
                if data.get('quantidade'):
                    posicao = repo.carregar_posicao(posicao_id)
                    if posicao:
                        repo.salvar_atributos_posicao(
                            posicao_id=posicao_id,
                            produto_id=posicao['produto_id'],
                            quantidade=float(data.get('quantidade'))
                        )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Fechar posicao
        if path == '/api/posicao/fechar':
            try:
                posicao_id = int(data.get('posicao_id'))
                repo.atualizar_posicao(
                    posicao_id=posicao_id,
                    data_saida=data.get('data_saida'),
                    preco_saida=float(data.get('preco_saida')),
                    status='fechada'
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar stop
        if path == '/api/stop/criar':
            try:
                repo.adicionar_stop_posicao(
                    posicao_id=int(data.get('posicao_id')),
                    data=data.get('data_stop'),
                    valor=float(data.get('preco'))
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Configurar ATR Stop
        if path == '/api/posicao/atr':
            try:
                posicao_id = int(data.get('posicao_id'))
                atr_period = data.get('atr_period')
                atr_multiplier = data.get('atr_multiplier')

                # Se valores são None ou null string, remove a configuração
                if atr_period in [None, 'null', '']:
                    atr_period = None
                else:
                    atr_period = int(atr_period)

                if atr_multiplier in [None, 'null', '']:
                    atr_multiplier = None
                else:
                    atr_multiplier = float(atr_multiplier)

                repo.atualizar_posicao(
                    posicao_id=posicao_id,
                    atr_period=atr_period,
                    atr_multiplier=atr_multiplier
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        self._send_json({'erro': 'Rota nao encontrada'}, 404)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        repo = SQLiteRepo()

        # Dashboard principal
        if path == '/' or path == '':
            produtos = repo.listar_produtos()
            contagem = repo.contar_posicoes_abertas_por_produto()
            stats = {p['id']: {'posicoes_abertas': contagem.get(p['id'], 0)} for p in produtos}
            html = get_dashboard_html(produtos, stats, repo)
            self._send_html(html)
            return

        # Menu completo
        if path == '/menu':
            self._send_html(get_menu_html())
            return

        # Formulario novo produto
        if path == '/produto/novo':
            self._send_html(get_form_produto_html())
            return

        # Lista para editar produtos
        if path == '/produtos/editar':
            produtos = repo.listar_produtos()
            self._send_html(get_lista_produtos_html(produtos, "editar"))
            return

        # Lista para deletar produtos
        if path == '/produtos/deletar':
            produtos = repo.listar_produtos()
            self._send_html(get_lista_produtos_html(produtos, "deletar"))
            return

        # Formulario nova posicao
        if path == '/posicao/nova':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            produtos = repo.listar_produtos()
            self._send_html(get_form_posicao_html(produto_id, produtos))
            return

        # Lista posicoes para editar
        if path == '/posicoes/editar':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    self._send_html(get_lista_posicoes_html(produto, posicoes, "editar"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Lista posicoes para adicionar stop
        if path == '/stop/novo':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    self._send_html(get_lista_posicoes_html(produto, posicoes, "stop"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Lista posicoes para fechar
        if path == '/posicao/fechar':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    self._send_html(get_lista_posicoes_html(produto, posicoes, "fechar"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Lista posicoes para configurar ATR
        if path == '/atr/config':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    self._send_html(get_lista_posicoes_html(produto, posicoes, "atr"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Rotas de posicao especifica
        if path.startswith('/posicao/') and not path.startswith('/posicao/nova'):
            parts = path.split('/')
            if len(parts) >= 4:
                try:
                    posicao_id = int(parts[2])
                    acao = parts[3]
                    produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None

                    if produto_id:
                        produto = repo.carregar_produto(produto_id)
                        posicao = repo.carregar_posicao(posicao_id)

                        if produto and posicao:
                            if acao == 'editar':
                                self._send_html(get_form_editar_posicao_html(produto, posicao))
                                return
                            elif acao == 'stop':
                                self._send_html(get_form_adicionar_stop_html(produto, posicao))
                                return
                            elif acao == 'atr':
                                self._send_html(get_form_atr_stop_html(produto, posicao))
                                return
                            elif acao == 'fechar':
                                self._send_html(get_form_fechar_posicao_html(produto, posicao))
                                return
                except ValueError:
                    pass
            self._send_html("<h1>Posicao nao encontrada</h1>", 404)
            return

        # Rotas de produto
        if path.startswith('/produto/'):
            parts = path.split('/')

            if len(parts) >= 3:
                try:
                    produto_id = int(parts[2])
                except ValueError:
                    # Pode ser /produto/novo que ja foi tratado acima
                    self._send_html("<h1>Rota invalida</h1>", 404)
                    return

                produto = repo.carregar_produto(produto_id)
                if not produto:
                    self._send_html("<h1>Produto nao encontrado</h1>", 404)
                    return

                # Editar produto
                if len(parts) >= 4 and parts[3] == 'editar':
                    self._send_html(get_form_produto_html(produto))
                    return

                # Confirmar delete
                if len(parts) >= 4 and parts[3] == 'deletar':
                    self._send_html(get_confirmar_delete_html(produto))
                    return

                # Visualizacao especifica
                if len(parts) >= 5 and parts[3] == 'viz':
                    try:
                        viz_id = int(parts[4])
                        viz = repo.carregar_visualizacao(viz_id)
                        if viz and viz['produto_id'] == produto_id:
                            df = obter_dados_para_visualizacao(produto_id, viz)
                            df_viz = aplicar_visualizacao(df, viz)
                            self._send_html(get_visualizacao_html(produto, viz, df_viz))
                            return
                    except:
                        pass
                    self._send_html("<h1>Visualizacao nao encontrada</h1>", 404)
                    return

                # Posicoes abertas/fechadas (fallback sem visualizacoes)
                if len(parts) >= 4 and parts[3] in ['abertas', 'fechadas']:
                    tipo = parts[3]
                    visualizacoes = repo.listar_visualizacoes(produto_id)

                    if tipo == 'abertas':
                        df = display_posicoes_abertas(produto_id, formatar=True, filtrar_colunas=False)
                    else:
                        df = display_posicoes_fechadas(produto_id, formatar=True, filtrar_colunas=False)

                    # Criar visualizacao fake para reutilizar template
                    fake_viz = {'id': 0, 'nome': f'Posicoes {"Abertas" if tipo == "abertas" else "Fechadas"}', 'produto_id': produto_id}
                    self._send_html(get_visualizacao_html(produto, fake_viz, df))
                    return

                # Pagina do produto (default)
                visualizacoes = repo.listar_visualizacoes(produto_id)
                self._send_html(get_produto_html(produto, visualizacoes, repo))
                return

        # API: Lista produtos
        if path == '/api/produtos':
            produtos = repo.listar_produtos()
            self._send_json({'produtos': produtos})
            return

        # API: Atualizar dados
        if path.startswith('/api/atualizar/'):
            try:
                produto_id = int(path.split('/')[-1])
                resultado = atualizar_stops_posicoes_abertas(repo=repo, produto_id=produto_id, verbose=False)
                print(f"[ATR] Produto {produto_id}: {resultado['updated']} atualizados")
                self._send_json({'sucesso': True, 'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 500)
            return

        # 404
        self._send_html("<h1>404 - Pagina nao encontrada</h1>", 404)

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def iniciar_servidor(porta=8080):
    """Inicia o servidor do dashboard"""
    servidor = HTTPServer(('localhost', porta), DashboardHandler)

    print("=" * 60)
    print("  DASHBOARD WEB - Products & Positions")
    print("=" * 60)
    print(f"\n  Servidor iniciado em: http://localhost:{porta}")
    print(f"\n  Rotas principais:")
    print(f"    /                  - Dashboard")
    print(f"    /menu              - Menu completo")
    print(f"    /produto/novo      - Criar produto")
    print(f"    /posicao/nova      - Criar posicao")
    print(f"    /produto/ID        - Ver produto")
    print(f"    /produto/ID/viz/X  - Visualizacao salva")
    print(f"\n  Pressione Ctrl+C para parar.")
    print("=" * 60)

    try:
        import webbrowser
        webbrowser.open(f'http://localhost:{porta}')
    except:
        pass

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServidor encerrado.")
        servidor.shutdown()


if __name__ == "__main__":
    import sys
    porta = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    iniciar_servidor(porta)
