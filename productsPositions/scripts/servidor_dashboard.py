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
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from storage.sqlite_repo import get_repo
from domain.posicao import Posicao
from domain.produto import Produto
from domain.tipo import Tipo
from analytics.notebook_utils import (
    display_posicoes_abertas,
    display_posicoes_fechadas,
    display_historico_posicoes,
    display_carteira,
    display_alocacoes,
)
from services.atr_stop_service import atualizar_stops_posicoes_abertas, calcular_stop_para_posicao
from services.bitget_service import sync_positions_with_exchange
from services.notificacao_service import notificar_stop_atingido
from services.turmas_service import TurmasService
from services.rentabilidade_service import RentabilidadeService
from services.cotacoes_service import CotacoesService
from turmas_dashboard import (
    get_lista_turmas_html,
    get_form_nova_turma_html,
    get_turma_detalhes_html,
    get_rentabilidade_chart_html,
    get_comparar_turmas_html,
    get_rentabilidade_historica_html
)


def _rodar_migracao_alocacoes_em_background():
    """Roda migração de alocações (CSVs -> Supabase) em thread para não bloquear o servidor. Logs aparecem no Render."""
    import threading
    _scripts_dir = Path(__file__).resolve().parent
    def _run():
        try:
            if str(_scripts_dir) not in sys.path:
                sys.path.insert(0, str(_scripts_dir))
            from atualizar_alocacoes_csv import main
            print("[MIGRAÇÃO ALOCAÇÕES] Iniciando em background (mesmo banco do dashboard)...", flush=True)
            n = main(dry_run=False)
            print(f"[MIGRAÇÃO ALOCAÇÕES] Concluída. Total de alocações inseridas: {n}", flush=True)
        except Exception as e:
            print(f"[MIGRAÇÃO ALOCAÇÕES] ERRO: {e}", flush=True)
    t = threading.Thread(target=_run, daemon=True)
    t.start()


def atualizar_dados_produto(repo, produto_id: int) -> dict:
    """
    Atualiza dados do produto em paralelo (Bitget sync + ATR stops).
    Executado automaticamente ao carregar a pagina do produto.

    Returns:
        dict com resultados: {bitget: {...}, atr: {...}}
    """
    resultado = {'bitget': None, 'atr': None}

    def sync_bitget():
        try:
            return sync_positions_with_exchange(repo, produto_id, verbose=False)
        except Exception as e:
            print(f"[BITGET] Erro ao sincronizar produto {produto_id}: {e}")
            return {'synced': 0, 'errors': [str(e)]}

    def update_atr():
        try:
            return atualizar_stops_posicoes_abertas(repo=repo, produto_id=produto_id, verbose=False)
        except Exception as e:
            print(f"[ATR] Erro ao atualizar stops produto {produto_id}: {e}")
            return {'updated': 0, 'errors': [str(e)]}

    # Executa Bitget sync e ATR update em paralelo
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_bitget = executor.submit(sync_bitget)
        future_atr = executor.submit(update_atr)

        resultado['bitget'] = future_bitget.result()
        resultado['atr'] = future_atr.result()

    # Log resumido
    if resultado['bitget'] and resultado['bitget'].get('synced', 0) > 0:
        print(f"[BITGET] Produto {produto_id}: {resultado['bitget']['synced']} posicoes sincronizadas")
    if resultado['atr'] and resultado['atr'].get('updated', 0) > 0:
        print(f"[ATR] Produto {produto_id}: {resultado['atr']['updated']} stops atualizados")

    return resultado


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

    # Substituir valores None por travessão
    df = df.fillna("—")

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
        .navbar-links a.active { background: rgba(78, 204, 163, 0.25); }
        .turmas-subnav {
            display: flex;
            gap: 0;
            padding: 0 30px;
            background: rgba(0,0,0,0.15);
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .turmas-subnav a {
            color: #888;
            text-decoration: none;
            padding: 10px 20px;
            font-size: 0.9em;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .turmas-subnav a:hover { color: #ccc; }
        .turmas-subnav a.active {
            color: #4ecca3;
            border-bottom-color: #4ecca3;
        }
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
    """Gera a barra de navegacao principal (padronizada para todas as paginas)"""
    return f"""
    <nav class="navbar">
        <a href="/" style="text-decoration: none;"><h1>Products & Positions</h1></a>
        <div class="navbar-links">
            <a href="/" class="{'active' if current_page == 'home' else ''}">Dashboard</a>
            <a href="/menu" class="{'active' if current_page == 'menu' else ''}">Menu</a>
            <a href="/turmas" class="{'active' if current_page in ('turmas', 'historico', 'comparar') else ''}">Turmas</a>
        </div>
    </nav>
    """


def get_turmas_subnav(current_sub=""):
    """Gera sub-navegacao interna das paginas de turmas"""
    return f"""
    <div class="turmas-subnav">
        <a href="/turmas" class="{'active' if current_sub == 'turmas' else ''}">Turmas</a>
        <a href="/turmas/historico" class="{'active' if current_sub == 'historico' else ''}">Histórico</a>
        <a href="/turmas/comparar" class="{'active' if current_sub == 'comparar' else ''}">Comparar</a>
    </div>
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
                <h3>Alocacoes</h3>
                <div class="menu-grid">
                    <a href="/alocacao/nova" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Criar Alocacao</div>
                    </a>
                </div>
            </div>

            <div class="menu-section">
                <h3>Turmas & Rentabilidade</h3>
                <div class="menu-grid">
                    <a href="/turmas" class="menu-item">
                        <div class="menu-item-icon">T</div>
                        <div class="menu-item-label">Ver Turmas</div>
                    </a>
                    <a href="/turmas/historico" class="menu-item">
                        <div class="menu-item-icon">📊</div>
                        <div class="menu-item-label">Rentab. Histórica</div>
                    </a>
                    <a href="/turmas/nova" class="menu-item">
                        <div class="menu-item-icon">+</div>
                        <div class="menu-item-label">Nova Turma</div>
                    </a>
                    <a href="/turmas/comparar" class="menu-item">
                        <div class="menu-item-icon">C</div>
                        <div class="menu-item-label">Comparar</div>
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
            df_viz = df_viz.fillna("—")
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
            df = df.fillna("—")
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
                    <a href="/produto/{produto_id}/editar" class="btn btn-secondary">Editar Produto</a>
                    <a href="/produto/{produto_id}/visualizacoes" class="btn btn-secondary">Gerenciar Visualizacoes</a>
                    <a href="/produto/{produto_id}/atributos" class="btn btn-secondary">Gerenciar Atributos</a>
                </div>
            </div>

            <div class="tabs">{tabs_html}</div>

            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h2 style="margin: 0;">{viz_nome}</h2>
                    <div class="actions" style="margin: 0;">
                        <a href="/posicao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Posicao</a>
                        <a href="/alocacao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Alocacao</a>
                        <a href="/posicoes/editar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Editar Posicao</a>
                        <a href="/stop/novo?produto_id={produto_id}" class="btn btn-sm btn-secondary">+ Stop</a>
                        <a href="/atr/config?produto_id={produto_id}" class="btn btn-sm btn-secondary">ATR Stop</a>
                        <a href="/posicao/fechar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Fechar Posicao</a>
                        <a href="/posicoes/deletar?produto_id={produto_id}" class="btn btn-sm btn-danger">Deletar Posicao</a>
                    </div>
                </div>
                {content_html}
            </div>
        </div>

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

    repo = get_repo()
    visualizacoes = repo.listar_visualizacoes(produto_id)

    tabs_html = ""
    for viz in visualizacoes:
        active_class = 'active' if viz['id'] == visualizacao['id'] else ''
        tabs_html += f'<a href="/produto/{produto_id}/viz/{viz["id"]}" class="tab {active_class}">{viz["nome"]}</a>'

    if df_viz is not None and not df_viz.empty:
        df_viz = df_viz.fillna("—")
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
                    <a href="/produto/{produto_id}/editar" class="btn btn-secondary">Editar Produto</a>
                    <a href="/produto/{produto_id}/visualizacoes" class="btn btn-secondary">Gerenciar Visualizacoes</a>
                    <a href="/produto/{produto_id}/atributos" class="btn btn-secondary">Gerenciar Atributos</a>
                </div>
            </div>

            <div class="tabs">{tabs_html}</div>

            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h2 style="margin: 0;">{nome_viz}</h2>
                    <div class="actions" style="margin: 0;">
                        <a href="/posicao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Posicao</a>
                        <a href="/alocacao/nova?produto_id={produto_id}" class="btn btn-sm btn-primary">+ Nova Alocacao</a>
                        <a href="/posicoes/editar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Editar Posicao</a>
                        <a href="/stop/novo?produto_id={produto_id}" class="btn btn-sm btn-secondary">+ Stop</a>
                        <a href="/atr/config?produto_id={produto_id}" class="btn btn-sm btn-secondary">ATR Stop</a>
                        <a href="/posicao/fechar?produto_id={produto_id}" class="btn btn-sm btn-secondary">Fechar Posicao</a>
                        <a href="/posicoes/deletar?produto_id={produto_id}" class="btn btn-sm btn-danger">Deletar Posicao</a>
                    </div>
                </div>
                {content_html}
            </div>
        </div>

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
                            <label>Exchange Symbol (para sync)</label>
                            <input type="text" name="exchange_symbol" placeholder="BTCUSDT">
                        </div>
                        <div class="form-group">
                            <label>Side</label>
                            <select name="side">
                                <option value="long">Long</option>
                                <option value="short">Short</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Data de Entrada</label>
                            <input type="date" name="data_entrada" value="{date.today().isoformat()}">
                        </div>
                        <div class="form-group">
                            <label>Preco de Entrada (USD)</label>
                            <input type="number" name="preco_entrada" step="0.00000001" required>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Quantidade (opcional)</label>
                        <input type="number" name="quantidade" step="0.00000001">
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
    """Lista posicoes de um produto para selecao (editar/stop/fechar/atr/deletar)"""
    titulo_map = {
        "editar": "Editar Posicao",
        "stop": "Adicionar Stop",
        "fechar": "Fechar Posicao",
        "atr": "Configurar ATR Stop",
        "deletar": "Deletar Posicao"
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
            elif acao == "deletar":
                link = f"/posicao/{pos_id}/deletar?produto_id={produto['id']}"
                btn_class = "btn-danger"
                btn_text = "Deletar"
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
    exchange_symbol = posicao.get('exchange_symbol') or posicao.get('Exchange Symbol', '')
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
                        <div class="form-group">
                            <label>Exchange Symbol (para sync)</label>
                            <input type="text" name="exchange_symbol" value="{exchange_symbol or ''}" placeholder="BTCUSDT">
                        </div>
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
    from datetime import date as dt_date
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', 'N/A')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    data_entrada = posicao.get('data_entrada') or ''
    current_period = posicao.get('atr_period') or 14
    current_mult = posicao.get('atr_multiplier') or 3.0
    current_data_inicio = posicao.get('atr_data_inicio') or ''
    # Se não tiver data de início, usar ontem como padrão
    if not current_data_inicio:
        from datetime import timedelta
        current_data_inicio = (dt_date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

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
                    {ativo} - Entrada: ${preco_entrada:,.4f} (em {data_entrada})
                </p>
                <p style="color: #888; font-size: 0.9em; margin-bottom: 20px;">
                    O ATR Trailing Stop calcula automaticamente o stop baseado na volatilidade do ativo.
                    Use a "Data de Início" para definir a partir de quando o cálculo deve considerar.
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
                    <div class="form-group">
                        <label>Data de Início do Cálculo</label>
                        <input type="date" name="atr_data_inicio" value="{current_data_inicio}" required>
                        <small style="color: #888;">A partir de quando calcular o trailing stop (use data recente para posições antigas)</small>
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
                        alert.textContent = result.mensagem || 'ATR configurado!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}', 1000);
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
# VISUALIZACOES
# ============================================================

def get_lista_visualizacoes_html(produto, visualizacoes):
    """Lista visualizacoes de um produto para gerenciamento"""
    items_html = ""

    if visualizacoes:
        for viz in visualizacoes:
            colunas_preview = ', '.join(viz.get('colunas', [])[:4])
            if len(viz.get('colunas', [])) > 4:
                colunas_preview += '...'

            ordenacao_info = ""
            if viz.get('ordenacao') and viz['ordenacao'].get('coluna'):
                ordenacao_info = f" | Ordenado por: {viz['ordenacao']['coluna']}"

            items_html += f"""
            <div class="viz-item">
                <div style="flex: 1;">
                    <strong>{viz['nome']}</strong>
                    <p style="color: #888; font-size: 0.85em; margin-top: 5px;">
                        Colunas: {colunas_preview}{ordenacao_info}
                    </p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <a href="/produto/{produto['id']}/viz/{viz['id']}/editar" class="btn btn-sm btn-primary">Editar</a>
                    <a href="/produto/{produto['id']}/viz/{viz['id']}/deletar" class="btn btn-sm btn-danger">Deletar</a>
                </div>
            </div>
            """
    else:
        items_html = '<p style="text-align: center; color: #888; padding: 20px;">Nenhuma visualizacao cadastrada</p>'

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Visualizacoes - {produto['nome']}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 700px; margin: 0 auto;">
                <h2>Gerenciar Visualizacoes</h2>
                <p style="color: #4ecca3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <div class="viz-list">{items_html}</div>
                <div class="actions" style="margin-top: 20px;">
                    <a href="/produto/{produto['id']}/viz/nova" class="btn btn-primary">+ Nova Visualizacao</a>
                    <a href="/produto/{produto['id']}" class="btn btn-secondary">Voltar</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def get_form_nova_visualizacao_html(produto, colunas_disponiveis):
    """Formulario para criar nova visualizacao"""
    colunas_html = ""
    for i, col in enumerate(colunas_disponiveis):
        origem_badge = f'<span class="badge badge-outro" style="font-size: 0.7em;">{col["origem"]}</span>'
        colunas_html += f"""
        <label style="display: flex; align-items: center; gap: 10px; padding: 8px; background: #16213e; border-radius: 5px; cursor: pointer;">
            <input type="checkbox" name="colunas" value="{col['nome']}" style="width: 18px; height: 18px;">
            <span>{col['label']}</span>
            <span style="color: #666; font-size: 0.85em;">({col['nome']})</span>
            {origem_badge}
        </label>
        """

    colunas_ordenacao = ""
    for col in colunas_disponiveis:
        colunas_ordenacao += f'<option value="{col["nome"]}">{col["label"]}</option>'

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nova Visualizacao - {produto['nome']}</title>
        <style>
            {get_base_styles()}
            .colunas-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 8px;
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
                background: #1a1a2e;
                border-radius: 8px;
                border: 1px solid #2a2a4a;
            }}
            .filtros-container {{ margin-top: 15px; }}
            .filtro-item {{
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
                padding: 10px;
                background: #16213e;
                border-radius: 8px;
            }}
            .filtro-item select, .filtro-item input {{
                padding: 8px;
                border: 1px solid #2a2a4a;
                border-radius: 5px;
                background: #1a1a2e;
                color: #eee;
            }}
        </style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 800px; margin: 0 auto;">
                <h2>Nova Visualizacao</h2>
                <p style="color: #4ecca3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <form id="vizForm">
                    <div class="form-group">
                        <label>Nome da Visualizacao</label>
                        <input type="text" name="nome" required placeholder="Ex: Posicoes Abertas Long">
                    </div>

                    <div class="form-group">
                        <label>Colunas (selecione na ordem desejada)</label>
                        <div style="margin-bottom: 10px;">
                            <button type="button" class="btn btn-sm btn-secondary" onclick="selecionarTodas()">Selecionar Todas</button>
                            <button type="button" class="btn btn-sm btn-secondary" onclick="limparSelecao()">Limpar</button>
                        </div>
                        <div class="colunas-grid">
                            {colunas_html}
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Ordenar por (opcional)</label>
                            <select name="ordenacao_coluna">
                                <option value="">Sem ordenacao</option>
                                {colunas_ordenacao}
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Direcao</label>
                            <select name="ordenacao_direcao">
                                <option value="asc">Crescente</option>
                                <option value="desc">Decrescente</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Filtro de Status (opcional)</label>
                        <select name="filtro_status">
                            <option value="">Todas as posicoes</option>
                            <option value="open">Apenas abertas</option>
                            <option value="closed">Apenas fechadas</option>
                        </select>
                    </div>

                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Criar Visualizacao</button>
                        <a href="/produto/{produto['id']}/visualizacoes" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            function selecionarTodas() {{
                document.querySelectorAll('input[name="colunas"]').forEach(cb => cb.checked = true);
            }}
            function limparSelecao() {{
                document.querySelectorAll('input[name="colunas"]').forEach(cb => cb.checked = false);
            }}

            document.getElementById('vizForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const alert = document.getElementById('alert');

                // Coletar colunas selecionadas na ordem
                const colunas = Array.from(form.querySelectorAll('input[name="colunas"]:checked'))
                    .map(cb => cb.value);

                if (colunas.length === 0) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Selecione pelo menos uma coluna';
                    return;
                }}

                const dados = {{
                    produto_id: {produto['id']},
                    nome: form.nome.value,
                    colunas: colunas
                }};

                // Ordenacao
                if (form.ordenacao_coluna.value) {{
                    dados.ordenacao = {{
                        coluna: form.ordenacao_coluna.value,
                        direcao: form.ordenacao_direcao.value
                    }};
                }}

                // Filtro de status
                if (form.filtro_status.value) {{
                    dados.filtros = [{{
                        coluna: 'status',
                        operador: '=',
                        valor: form.filtro_status.value
                    }}];
                }}

                try {{
                    const response = await fetch('/api/visualizacao/criar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(dados)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Visualizacao criada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/visualizacoes', 1000);
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


def get_form_editar_visualizacao_html(produto, visualizacao, colunas_disponiveis):
    """Formulario para editar visualizacao existente"""
    colunas_selecionadas = visualizacao.get('colunas', [])

    colunas_html = ""
    for col in colunas_disponiveis:
        checked = 'checked' if col['nome'] in colunas_selecionadas else ''
        origem_badge = f'<span class="badge badge-outro" style="font-size: 0.7em;">{col["origem"]}</span>'
        colunas_html += f"""
        <label style="display: flex; align-items: center; gap: 10px; padding: 8px; background: #16213e; border-radius: 5px; cursor: pointer;">
            <input type="checkbox" name="colunas" value="{col['nome']}" {checked} style="width: 18px; height: 18px;">
            <span>{col['label']}</span>
            <span style="color: #666; font-size: 0.85em;">({col['nome']})</span>
            {origem_badge}
        </label>
        """

    ordenacao = visualizacao.get('ordenacao', {}) or {}
    ordenacao_coluna = ordenacao.get('coluna', '')
    ordenacao_direcao = ordenacao.get('direcao', 'asc')

    colunas_ordenacao = '<option value="">Sem ordenacao</option>'
    for col in colunas_disponiveis:
        selected = 'selected' if col['nome'] == ordenacao_coluna else ''
        colunas_ordenacao += f'<option value="{col["nome"]}" {selected}>{col["label"]}</option>'

    # Determinar filtro de status atual
    filtros = visualizacao.get('filtros', []) or []
    filtro_status = ''
    for f in filtros:
        if f.get('coluna') == 'status':
            filtro_status = f.get('valor', '')
            break

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Editar Visualizacao - {visualizacao['nome']}</title>
        <style>
            {get_base_styles()}
            .colunas-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 8px;
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
                background: #1a1a2e;
                border-radius: 8px;
                border: 1px solid #2a2a4a;
            }}
        </style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 800px; margin: 0 auto;">
                <h2>Editar Visualizacao</h2>
                <p style="color: #4ecca3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <form id="vizForm">
                    <div class="form-group">
                        <label>Nome da Visualizacao</label>
                        <input type="text" name="nome" required value="{visualizacao['nome']}">
                    </div>

                    <div class="form-group">
                        <label>Colunas</label>
                        <div style="margin-bottom: 10px;">
                            <button type="button" class="btn btn-sm btn-secondary" onclick="selecionarTodas()">Selecionar Todas</button>
                            <button type="button" class="btn btn-sm btn-secondary" onclick="limparSelecao()">Limpar</button>
                        </div>
                        <div class="colunas-grid">
                            {colunas_html}
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Ordenar por</label>
                            <select name="ordenacao_coluna">
                                {colunas_ordenacao}
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Direcao</label>
                            <select name="ordenacao_direcao">
                                <option value="asc" {'selected' if ordenacao_direcao == 'asc' else ''}>Crescente</option>
                                <option value="desc" {'selected' if ordenacao_direcao == 'desc' else ''}>Decrescente</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Filtro de Status</label>
                        <select name="filtro_status">
                            <option value="" {'selected' if filtro_status == '' else ''}>Todas as posicoes</option>
                            <option value="open" {'selected' if filtro_status == 'open' else ''}>Apenas abertas</option>
                            <option value="closed" {'selected' if filtro_status == 'closed' else ''}>Apenas fechadas</option>
                        </select>
                    </div>

                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Salvar</button>
                        <a href="/produto/{produto['id']}/visualizacoes" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            function selecionarTodas() {{
                document.querySelectorAll('input[name="colunas"]').forEach(cb => cb.checked = true);
            }}
            function limparSelecao() {{
                document.querySelectorAll('input[name="colunas"]').forEach(cb => cb.checked = false);
            }}

            document.getElementById('vizForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const alert = document.getElementById('alert');

                const colunas = Array.from(form.querySelectorAll('input[name="colunas"]:checked'))
                    .map(cb => cb.value);

                if (colunas.length === 0) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Selecione pelo menos uma coluna';
                    return;
                }}

                const dados = {{
                    visualizacao_id: {visualizacao['id']},
                    nome: form.nome.value,
                    colunas: colunas
                }};

                if (form.ordenacao_coluna.value) {{
                    dados.ordenacao = {{
                        coluna: form.ordenacao_coluna.value,
                        direcao: form.ordenacao_direcao.value
                    }};
                }} else {{
                    dados.ordenacao = null;
                }}

                if (form.filtro_status.value) {{
                    dados.filtros = [{{
                        coluna: 'status',
                        operador: '=',
                        valor: form.filtro_status.value
                    }}];
                }} else {{
                    dados.filtros = null;
                }}

                try {{
                    const response = await fetch('/api/visualizacao/editar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(dados)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Visualizacao atualizada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/visualizacoes', 1000);
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


def get_confirmar_delete_posicao_html(produto, posicao):
    """Pagina de confirmacao para deletar posicao"""
    pos_id = posicao.get('id') or posicao.get('ID')
    ativo = posicao.get('ativo') or posicao.get('Ativo', 'N/A')
    side = posicao.get('side') or posicao.get('tipo', 'N/A')
    preco_entrada = posicao.get('preco_entrada') or posicao.get('Preço Entrada', 0)
    status = posicao.get('status', 'open')
    is_spot = 'spot' in produto.get('tipo', '').lower()
    side_info = "" if is_spot else f" ({side})"

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deletar Posicao - {ativo}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 500px; margin: 0 auto; text-align: center;">
                <h2 style="color: #ff6b6b;">Deletar Posicao</h2>
                <p style="margin: 20px 0;">Tem certeza que deseja deletar:</p>
                <p style="font-size: 1.3em; color: #4ecca3; font-weight: bold;">{ativo}{side_info}</p>
                <p style="color: #888;">Entrada: ${preco_entrada:,.4f} | Status: {status}</p>
                <p style="color: #ff6b6b; margin: 20px 0; font-size: 0.9em;">
                    Esta acao ira deletar a posicao e todos os stops e alocacoes associados!
                </p>
                <div id="alert" class="alert"></div>
                <div class="actions" style="justify-content: center;">
                    <button class="btn btn-danger" onclick="deletarPosicao()">Sim, Deletar</button>
                    <a href="/produto/{produto['id']}" class="btn btn-secondary">Cancelar</a>
                </div>
            </div>
        </div>
        <script>
            async function deletarPosicao() {{
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/posicao/deletar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{posicao_id: {pos_id}}})
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Posicao deletada!';
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


def get_lista_atributos_html(produto, configs, colunas_orfas):
    """Lista atributos de um produto para gerenciamento"""
    items_html = ""

    if configs:
        for config in configs:
            obrig_badge = '<span class="badge badge-danger" style="margin-left: 8px;">obrigatorio</span>' if config['obrigatorio'] else ''
            items_html += f"""
            <div class="viz-item">
                <div style="flex: 1;">
                    <strong>{config['atributo_label'] or config['atributo_nome']}</strong>
                    <span style="color: #888; margin-left: 8px;">({config['atributo_nome']})</span>
                    <span class="badge badge-outro" style="margin-left: 8px;">{config['atributo_tipo']}</span>
                    {obrig_badge}
                </div>
                <div style="display: flex; gap: 8px;">
                    <a href="/produto/{produto['id']}/atributos/{config['atributo_nome']}/editar" class="btn btn-sm btn-primary">Editar</a>
                    <a href="/produto/{produto['id']}/atributos/{config['atributo_nome']}/remover" class="btn btn-sm btn-danger">Remover</a>
                </div>
            </div>
            """
    else:
        items_html = '<p style="text-align: center; color: #888; padding: 20px;">Nenhum atributo configurado para este produto</p>'

    # Mostrar colunas orfas se existirem
    orfas_html = ""
    if colunas_orfas:
        orfas_html = f"""
        <div style="margin-top: 30px; padding: 15px; background: rgba(255, 107, 107, 0.1); border-radius: 8px; border: 1px solid #ff6b6b;">
            <h3 style="color: #ff6b6b; margin-bottom: 10px;">Colunas Orfas ({len(colunas_orfas)})</h3>
            <p style="color: #888; font-size: 0.9em; margin-bottom: 10px;">Colunas que nao estao sendo usadas por nenhum produto:</p>
            <p style="color: #888;">{', '.join(colunas_orfas)}</p>
            <button class="btn btn-sm btn-danger" style="margin-top: 10px;" onclick="limparOrfas()">Limpar Colunas Orfas</button>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Atributos - {produto['nome']}</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 700px; margin: 0 auto;">
                <h2>Gerenciar Atributos</h2>
                <p style="color: #4ecca3; margin-bottom: 20px;">{produto['nome']}</p>
                <div id="alert" class="alert"></div>
                <div class="viz-list">{items_html}</div>
                {orfas_html}
                <div class="actions" style="margin-top: 20px;">
                    <a href="/produto/{produto['id']}/atributos/novo" class="btn btn-primary">+ Novo Atributo</a>
                    <a href="/produto/{produto['id']}" class="btn btn-secondary">Voltar</a>
                </div>
            </div>
        </div>
        <script>
            async function limparOrfas() {{
                if (!confirm('Remover todas as colunas orfas?')) return;
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/atributos/limpar-orfas', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}}
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Colunas orfas removidas: ' + result.removidas;
                        setTimeout(() => location.reload(), 1000);
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


def get_form_atributo_html(produto, config=None, colunas_existentes=None):
    """Formulario para criar/editar atributo"""
    is_edit = config is not None
    titulo = "Editar Atributo" if is_edit else "Novo Atributo"

    nome_value = config['atributo_nome'] if is_edit else ''
    label_value = config.get('atributo_label', '') or '' if is_edit else ''
    tipo_value = config['atributo_tipo'] if is_edit else 'text'
    obrig_checked = 'checked' if is_edit and config.get('obrigatorio') else ''

    # Lista de colunas existentes para sugestao
    colunas_datalist = ""
    if colunas_existentes and not is_edit:
        for col in colunas_existentes:
            colunas_datalist += f'<option value="{col}">'

    nome_field = f"""
        <input type="text" name="nome" value="{nome_value}" list="colunas_existentes" required
               placeholder="Ex: quantidade, risco, setor" {'readonly' if is_edit else ''}>
        <datalist id="colunas_existentes">{colunas_datalist}</datalist>
        <small style="color: #888;">Use colunas existentes ou crie uma nova</small>
    """ if not is_edit else f'<input type="text" name="nome" value="{nome_value}" readonly>'

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
                <div id="alert" class="alert"></div>
                <form id="atributoForm">
                    <div class="form-group">
                        <label>Nome do Atributo (coluna)</label>
                        {nome_field}
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Label (exibicao)</label>
                            <input type="text" name="label" value="{label_value}" placeholder="Ex: Quantidade, Nivel de Risco">
                        </div>
                        <div class="form-group">
                            <label>Tipo</label>
                            <select name="tipo">
                                <option value="text" {'selected' if tipo_value == 'text' else ''}>Texto</option>
                                <option value="float" {'selected' if tipo_value == 'float' else ''}>Numero decimal</option>
                                <option value="int" {'selected' if tipo_value == 'int' else ''}>Numero inteiro</option>
                                <option value="date" {'selected' if tipo_value == 'date' else ''}>Data</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-group">
                        <label style="display: flex; align-items: center; gap: 10px; cursor: pointer;">
                            <input type="checkbox" name="obrigatorio" {obrig_checked} style="width: 20px; height: 20px;">
                            <span>Campo obrigatorio</span>
                        </label>
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">{'Salvar' if is_edit else 'Criar'}</button>
                        <a href="/produto/{produto['id']}/atributos" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            document.getElementById('atributoForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const alert = document.getElementById('alert');

                const dados = {{
                    produto_id: {produto['id']},
                    nome: form.nome.value.toLowerCase().trim().replace(/ /g, '_'),
                    label: form.label.value || null,
                    tipo: form.tipo.value,
                    obrigatorio: form.obrigatorio.checked
                }};

                const endpoint = '{"editar" if is_edit else "criar"}';

                try {{
                    const response = await fetch('/api/atributo/' + endpoint, {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(dados)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Atributo {"atualizado" if is_edit else "criado"}!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/atributos', 1000);
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


def get_confirmar_remover_atributo_html(produto, config):
    """Pagina de confirmacao para remover atributo do produto"""
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Remover Atributo</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 500px; margin: 0 auto; text-align: center;">
                <h2 style="color: #ff6b6b;">Remover Atributo</h2>
                <p style="margin: 20px 0;">Remover atributo do produto {produto['nome']}:</p>
                <p style="font-size: 1.3em; color: #4ecca3; font-weight: bold;">{config['atributo_label'] or config['atributo_nome']}</p>
                <p style="color: #888; margin: 20px 0; font-size: 0.9em;">
                    A coluna permanecera no banco de dados e podera ser usada por outros produtos.
                </p>
                <div id="alert" class="alert"></div>
                <div class="actions" style="justify-content: center;">
                    <button class="btn btn-danger" onclick="removerAtributo()">Sim, Remover</button>
                    <a href="/produto/{produto['id']}/atributos" class="btn btn-secondary">Cancelar</a>
                </div>
            </div>
        </div>
        <script>
            async function removerAtributo() {{
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/atributo/remover', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{
                            produto_id: {produto['id']},
                            nome: '{config["atributo_nome"]}'
                        }})
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Atributo removido!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/atributos', 1000);
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


def get_form_alocacao_html(produto_id=None, produtos=None, posicoes=None):
    """Formulario para criar alocacao"""
    produtos_options = ""
    if produtos:
        for p in produtos:
            selected = 'selected' if produto_id and p['id'] == produto_id else ''
            produtos_options += f'<option value="{p["id"]}" {selected}>{p["nome"]}</option>'

    posicoes_options = ""
    if posicoes is not None and not posicoes.empty:
        for _, pos in posicoes.iterrows():
            pos_id = pos.get('id') or pos.get('ID')
            ativo = pos.get('ativo') or pos.get('Ativo', 'N/A')
            side = pos.get('side') or pos.get('tipo', '')
            preco = pos.get('preco_entrada') or pos.get('Preço Entrada', 0)
            posicoes_options += f'<option value="{pos_id}">{ativo} ({side}) - ${preco:,.2f}</option>'

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nova Alocacao</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 600px; margin: 0 auto;">
                <h2>Nova Alocacao</h2>
                <div id="alert" class="alert"></div>
                <form id="alocacaoForm">
                    <div class="form-group">
                        <label>Produto</label>
                        <select name="produto_id" id="produtoSelect" required onchange="carregarPosicoes()">
                            <option value="">Selecione um produto</option>
                            {produtos_options}
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Posicao</label>
                        <select name="posicao_id" id="posicaoSelect" required>
                            <option value="">Selecione uma posicao</option>
                            {posicoes_options}
                        </select>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Percentual Alocado (%)</label>
                            <input type="number" name="percentual" step="0.01" min="0" max="100" required placeholder="Ex: 10.5">
                        </div>
                        <div class="form-group">
                            <label>Valor em USD (opcional)</label>
                            <input type="number" name="valor_usd" step="0.01" placeholder="Ex: 1000.00">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Data da Alocacao (opcional)</label>
                        <input type="date" name="data_alocacao" value="{date.today().isoformat()}">
                    </div>
                    <div class="actions">
                        <button type="submit" class="btn btn-primary">Criar Alocacao</button>
                        <a href="/" class="btn btn-secondary">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
        <script>
            async function carregarPosicoes() {{
                const produtoId = document.getElementById('produtoSelect').value;
                const posicaoSelect = document.getElementById('posicaoSelect');

                if (!produtoId) {{
                    posicaoSelect.innerHTML = '<option value="">Selecione uma posicao</option>';
                    return;
                }}

                try {{
                    const response = await fetch('/api/posicoes/abertas/' + produtoId);
                    const data = await response.json();

                    posicaoSelect.innerHTML = '<option value="">Selecione uma posicao</option>';
                    if (data.posicoes) {{
                        data.posicoes.forEach(pos => {{
                            const option = document.createElement('option');
                            option.value = pos.id;
                            option.textContent = pos.ativo + ' (' + pos.side + ') - $' + pos.preco_entrada.toFixed(2);
                            posicaoSelect.appendChild(option);
                        }});
                    }}
                }} catch (error) {{
                    console.error('Erro ao carregar posicoes:', error);
                }}
            }}

            document.getElementById('alocacaoForm').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const alert = document.getElementById('alert');

                const dados = {{
                    produto_id: parseInt(form.produto_id.value),
                    posicao_id: parseInt(form.posicao_id.value),
                    percentual: parseFloat(form.percentual.value),
                    valor_usd: form.valor_usd.value ? parseFloat(form.valor_usd.value) : null,
                    data_alocacao: form.data_alocacao.value || null
                }};

                try {{
                    const response = await fetch('/api/alocacao/criar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(dados)
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Alocacao criada!';
                        setTimeout(() => window.location.href = '/produto/' + dados.produto_id, 1000);
                    }} else {{
                        throw new Error(result.erro);
                    }}
                }} catch (error) {{
                    alert.className = 'alert alert-error show';
                    alert.textContent = 'Erro: ' + error.message;
                }}
            }});

            // Carregar posicoes se produto ja selecionado
            if (document.getElementById('produtoSelect').value) {{
                carregarPosicoes();
            }}
        </script>
    </body>
    </html>
    """


def get_confirmar_delete_visualizacao_html(produto, visualizacao):
    """Pagina de confirmacao para deletar visualizacao"""
    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deletar Visualizacao</title>
        <style>{get_base_styles()}</style>
    </head>
    <body>
        {get_navbar()}
        <div class="container">
            <div class="card" style="max-width: 500px; margin: 0 auto; text-align: center;">
                <h2 style="color: #ff6b6b;">Deletar Visualizacao</h2>
                <p style="margin: 20px 0;">Tem certeza que deseja deletar:</p>
                <p style="font-size: 1.3em; color: #4ecca3; font-weight: bold;">{visualizacao['nome']}</p>
                <p style="color: #888; margin: 20px 0;">Esta acao nao pode ser desfeita.</p>
                <div id="alert" class="alert"></div>
                <div class="actions" style="justify-content: center;">
                    <button class="btn btn-danger" onclick="deletarVisualizacao()">Sim, Deletar</button>
                    <a href="/produto/{produto['id']}/visualizacoes" class="btn btn-secondary">Cancelar</a>
                </div>
            </div>
        </div>
        <script>
            async function deletarVisualizacao() {{
                const alert = document.getElementById('alert');
                try {{
                    const response = await fetch('/api/visualizacao/deletar', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{visualizacao_id: {visualizacao['id']}}})
                    }});
                    const result = await response.json();
                    if (result.sucesso) {{
                        alert.className = 'alert alert-success show';
                        alert.textContent = 'Visualizacao deletada!';
                        setTimeout(() => window.location.href = '/produto/{produto["id"]}/visualizacoes', 1000);
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

    def _send_excel(self, buffer, filename):
        """Envia arquivo Excel para download"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(buffer.getvalue())

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

        try:
            repo = get_repo()
        except Exception as e:
            self._send_json({'erro': f'Falha na conexao com banco: {str(e)}'}, 500)
            return

        path = self.path

        # Criar produto
        if path == '/api/produto/criar':
            try:
                tipo_nome = data.get('tipo', 'Outro')
                produto = Produto(
                    nome=data.get('nome'),
                    data_inicio=data.get('data_inicio'),
                    tipo=Tipo(tipo_nome),
                    capital_inicial=float(data.get('capital_inicial', 0)) if data.get('capital_inicial') else 0.0
                )
                produto_id = repo.salvar_produto(produto)
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
                produto_id = int(data.get('produto_id'))
                posicao = Posicao(
                    ativo=data.get('ativo'),
                    side=data.get('side', 'long'),
                    data_entrada=data.get('data_entrada'),
                    preco_entrada=float(data.get('preco_entrada')),
                    coingecko_id=data.get('coingecko_id') or None,
                    exchange_symbol=data.get('exchange_symbol') or None
                )
                posicao_id = repo.salvar_posicao(produto_id, posicao)
                # Salvar quantidade como atributo se fornecida
                if data.get('quantidade'):
                    repo.salvar_atributos_posicao(
                        posicao_id=posicao_id,
                        produto_id=produto_id,
                        quantidade=float(data.get('quantidade'))
                    )

                # Sincronizar com turmas existentes
                try:
                    turmas_service = TurmasService()
                    turmas_service.sync_nova_posicao(
                        posicao_id=posicao_id,
                        produto_id=produto_id,
                        data_entrada=data.get('data_entrada'),
                        preco_entrada=float(data.get('preco_entrada'))
                    )
                except Exception as sync_err:
                    print(f"Aviso: erro ao sincronizar posição com turmas: {sync_err}")

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
                    coingecko_id=data.get('coingecko_id') or None,
                    exchange_symbol=data.get('exchange_symbol') or None
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
                data_saida = data.get('data_saida')
                repo.atualizar_posicao(
                    posicao_id=posicao_id,
                    data_saida=data_saida,
                    preco_saida=float(data.get('preco_saida')),
                    status='closed'
                )

                # Sincronizar fechamento com turmas
                try:
                    turmas_service = TurmasService()
                    turmas_service.sync_posicao_fechada(
                        posicao_id=posicao_id,
                        data_saida=data_saida
                    )
                except Exception as sync_err:
                    print(f"Aviso: erro ao sincronizar fechamento com turmas: {sync_err}")

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
                from datetime import date as dt_date
                posicao_id = int(data.get('posicao_id'))
                atr_period = data.get('atr_period')
                atr_multiplier = data.get('atr_multiplier')
                atr_data_inicio = data.get('atr_data_inicio')

                # Se valores são None ou null string, remove a configuração
                if atr_period in [None, 'null', '']:
                    atr_period = None
                else:
                    atr_period = int(atr_period)

                if atr_multiplier in [None, 'null', '']:
                    atr_multiplier = None
                else:
                    atr_multiplier = float(atr_multiplier)

                if atr_data_inicio in [None, 'null', '']:
                    atr_data_inicio = None

                repo.atualizar_posicao(
                    posicao_id=posicao_id,
                    atr_period=atr_period,
                    atr_multiplier=atr_multiplier,
                    atr_data_inicio=atr_data_inicio
                )

                # Se ATR foi configurado, calcular e salvar o stop imediatamente
                mensagem = 'ATR configurado e stop calculado!'
                if atr_multiplier is not None:
                    # Buscar dados da posição
                    posicao = repo.carregar_posicao(posicao_id)
                    if posicao and posicao.get('coingecko_id'):
                        # Usar atr_data_inicio se disponível, senão data_entrada
                        data_calculo = atr_data_inicio or posicao['data_entrada']

                        # Obter exchange_symbol para usar dados da Bitget (CoinGecko pode não ter OHLC)
                        exchange_symbol = posicao.get('exchange_symbol')
                        if exchange_symbol and str(exchange_symbol).lower() in ('none', 'nan', ''):
                            exchange_symbol = None

                        # Detectar tipo de produto (spot vs perpetuos)
                        product_type = 'perpetuos'  # Default
                        produto_info = repo.carregar_produto(posicao.get('produto_id'))
                        if produto_info and produto_info.get('tipo'):
                            tipo_str = str(produto_info['tipo']).lower()
                            if 'spot' in tipo_str:
                                product_type = 'spot'

                        stop, breached, erro = calcular_stop_para_posicao(
                            coingecko_id=posicao['coingecko_id'],
                            side=posicao['side'],
                            data_entrada=data_calculo,
                            atr_period=atr_period,
                            atr_multiplier=atr_multiplier,
                            exchange_symbol=exchange_symbol,
                            product_type=product_type
                        )
                        if breached:
                            # Salvar um valor especial para indicar que foi breached
                            hoje = dt_date.today().strftime('%Y-%m-%d')
                            repo.adicionar_stop_posicao(posicao_id, hoje, -1)  # -1 indica breached
                            mensagem = 'ATR configurado - STOP ATINGIDO!'
                            # Enviar notificação por e-mail
                            try:
                                nome_produto = produto_info['nome'] if produto_info else "Desconhecido"
                                notificar_stop_atingido(
                                    ativo=posicao['ativo'],
                                    side=posicao['side'],
                                    preco_entrada=posicao.get('preco_entrada'),
                                    produto_nome=nome_produto,
                                    data_entrada=posicao.get('data_entrada')
                                )
                            except Exception as e:
                                print(f"[Dashboard] Erro ao enviar e-mail de stop: {e}")
                        elif stop is not None:
                            hoje = dt_date.today().strftime('%Y-%m-%d')
                            repo.adicionar_stop_posicao(posicao_id, hoje, stop)

                self._send_json({'sucesso': True, 'mensagem': mensagem})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar visualizacao
        if path == '/api/visualizacao/criar':
            try:
                viz_id = repo.criar_visualizacao(
                    produto_id=int(data.get('produto_id')),
                    nome=data.get('nome'),
                    colunas=data.get('colunas', []),
                    ordenacao=data.get('ordenacao'),
                    filtros=data.get('filtros')
                )
                self._send_json({'sucesso': True, 'visualizacao_id': viz_id})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Editar visualizacao
        if path == '/api/visualizacao/editar':
            try:
                viz_id = int(data.get('visualizacao_id'))
                repo.atualizar_visualizacao(
                    visualizacao_id=viz_id,
                    nome=data.get('nome'),
                    colunas=data.get('colunas'),
                    ordenacao=data.get('ordenacao'),
                    filtros=data.get('filtros')
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Deletar visualizacao
        if path == '/api/visualizacao/deletar':
            try:
                viz_id = int(data.get('visualizacao_id'))
                repo.deletar_visualizacao(viz_id)
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Deletar posicao
        if path == '/api/posicao/deletar':
            try:
                posicao_id = int(data.get('posicao_id'))
                repo.deletar_posicao(posicao_id, forcar=True)
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar atributo
        if path == '/api/atributo/criar':
            try:
                repo.adicionar_atributo_config(
                    produto_id=int(data.get('produto_id')),
                    atributo_nome=data.get('nome'),
                    atributo_tipo=data.get('tipo', 'text'),
                    atributo_label=data.get('label'),
                    obrigatorio=data.get('obrigatorio', False)
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Editar atributo
        if path == '/api/atributo/editar':
            try:
                repo.editar_atributo_config(
                    produto_id=int(data.get('produto_id')),
                    atributo_nome=data.get('nome'),
                    novo_label=data.get('label'),
                    novo_tipo=data.get('tipo'),
                    novo_obrigatorio=data.get('obrigatorio')
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Remover atributo do produto
        if path == '/api/atributo/remover':
            try:
                repo.remover_atributo_config(
                    produto_id=int(data.get('produto_id')),
                    atributo_nome=data.get('nome')
                )
                self._send_json({'sucesso': True})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Limpar colunas orfas
        if path == '/api/atributos/limpar-orfas':
            try:
                removidas = repo.limpar_colunas_orfas()
                self._send_json({'sucesso': True, 'removidas': len(removidas)})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # Criar alocacao
        if path == '/api/alocacao/criar':
            try:
                from services.alocacao_service import AlocacaoService
                alocacao = AlocacaoService.criar_alocacao(
                    produto_id=int(data.get('produto_id')),
                    posicao_id=int(data.get('posicao_id')),
                    percentual=float(data.get('percentual')),
                    valor_usd=float(data.get('valor_usd')) if data.get('valor_usd') else None,
                    data=data.get('data_alocacao')
                )
                alocacao_id = repo.salvar_alocacao(int(data.get('produto_id')), alocacao)
                self._send_json({'sucesso': True, 'alocacao_id': alocacao_id})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            return

        # ============================================================
        # ROTAS DE TURMAS (POST)
        # ============================================================

        # Criar turma com busca automática de preços
        # posicoes_config: [{"posicao_id": 1, "data_insercao": "2026-01-15"}, ...]
        # preco_entrada_turma é OPCIONAL se auto_fetch_prices=true (default)
        # Use /api/turma/posicoes-elegiveis para listar posições elegíveis
        if path == '/api/turma/criar':
            try:
                turmas_service = TurmasService()
                produto_id = int(data.get('produto_id'))
                data_inicio = data.get('data_inicio')

                # Retrocompatibilidade: se posicoes_config ausente ou vazio, preencher com posições elegíveis
                posicoes_config = data.get('posicoes_config')
                if not posicoes_config:
                    posicoes = turmas_service.listar_posicoes_elegiveis(produto_id, data_inicio)
                    posicoes_config = [
                        {'posicao_id': p['id'], 'data_insercao': data_inicio}
                        for p in posicoes
                    ]

                # auto_fetch_prices: se True (default), busca preços via CoinGecko
                auto_fetch_prices = data.get('auto_fetch_prices', True)

                resultado = turmas_service.criar_turma(
                    produto_id=produto_id,
                    nome=data.get('nome'),
                    data_inicio=data_inicio,
                    capital_base=float(data.get('capital_base', 1500)),
                    descricao=data.get('descricao'),
                    posicoes_config=posicoes_config,
                    auto_fetch_prices=auto_fetch_prices
                )

                self._send_json({
                    'sucesso': True,
                    'turma_id': resultado['turma_id'],
                    'precos_resolvidos': resultado['precos_resolvidos'],
                    'avisos': resultado['avisos']
                })
            except ValueError as e:
                # Validation error (missing posicoes_config, missing fields, etc.)
                self._send_json({'sucesso': False, 'erro': str(e)}, 400)
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 500)
            return

        self._send_json({'erro': 'Rota nao encontrada'}, 404)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # Health check (sem conexao ao banco - para Render/cloud)
        if path == '/health':
            self._send_json({'status': 'ok', 'timestamp': datetime.now().isoformat()})
            return

        try:
            repo = get_repo()
        except Exception as e:
            self._send_json({'erro': f'Falha na conexao com banco: {str(e)}'}, 500)
            return

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

        # Lista posicoes para deletar
        if path == '/posicoes/deletar':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            if produto_id:
                produto = repo.carregar_produto(produto_id)
                if produto:
                    posicoes = repo.carregar_posicoes_abertas(produto_id)
                    self._send_html(get_lista_posicoes_html(produto, posicoes, "deletar"))
                    return
            self._send_html("<h1>Produto nao encontrado</h1>", 404)
            return

        # Formulario nova alocacao
        if path == '/alocacao/nova':
            produto_id = int(query.get('produto_id', [0])[0]) if query.get('produto_id') else None
            produtos = repo.listar_produtos()
            posicoes = None
            if produto_id:
                posicoes = repo.carregar_posicoes_abertas(produto_id)
            self._send_html(get_form_alocacao_html(produto_id, produtos, posicoes))
            return

        # API: Listar posicoes abertas de um produto (para AJAX)
        if path.startswith('/api/posicoes/abertas/'):
            try:
                produto_id = int(path.split('/')[-1])
                posicoes_df = repo.carregar_posicoes_abertas(produto_id)
                posicoes_list = []
                if posicoes_df is not None and not posicoes_df.empty:
                    for _, row in posicoes_df.iterrows():
                        posicoes_list.append({
                            'id': int(row.get('id') or row.get('ID')),
                            'ativo': row.get('ativo') or row.get('Ativo', 'N/A'),
                            'side': row.get('side') or row.get('tipo', 'N/A'),
                            'preco_entrada': float(row.get('preco_entrada') or row.get('Preço Entrada', 0))
                        })
                self._send_json({'posicoes': posicoes_list})
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
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
                            elif acao == 'deletar':
                                self._send_html(get_confirmar_delete_posicao_html(produto, posicao))
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

                # Gerenciar visualizacoes - lista
                if len(parts) >= 4 and parts[3] == 'visualizacoes':
                    visualizacoes = repo.listar_visualizacoes(produto_id)
                    self._send_html(get_lista_visualizacoes_html(produto, visualizacoes))
                    return

                # Gerenciar atributos
                if len(parts) >= 4 and parts[3] == 'atributos':
                    # Novo atributo
                    if len(parts) >= 5 and parts[4] == 'novo':
                        colunas_existentes = repo.listar_colunas_atributos()
                        self._send_html(get_form_atributo_html(produto, None, colunas_existentes))
                        return

                    # Atributo especifico
                    if len(parts) >= 6:
                        atributo_nome = parts[4]
                        acao = parts[5]
                        configs = repo.carregar_atributos_config(produto_id)
                        config = next((c for c in configs if c['atributo_nome'] == atributo_nome), None)

                        if config:
                            if acao == 'editar':
                                self._send_html(get_form_atributo_html(produto, config))
                                return
                            elif acao == 'remover':
                                self._send_html(get_confirmar_remover_atributo_html(produto, config))
                                return

                    # Lista de atributos (default)
                    configs = repo.carregar_atributos_config(produto_id)
                    colunas_orfas = repo.listar_colunas_orfas()
                    self._send_html(get_lista_atributos_html(produto, configs, colunas_orfas))
                    return

                # Visualizacoes - criar/editar/deletar/ver
                if len(parts) >= 5 and parts[3] == 'viz':
                    # Nova visualizacao
                    if parts[4] == 'nova':
                        colunas_disponiveis = repo.obter_colunas_disponiveis(produto_id)
                        self._send_html(get_form_nova_visualizacao_html(produto, colunas_disponiveis))
                        return

                    # Visualizacao especifica
                    try:
                        viz_id = int(parts[4])
                        viz = repo.carregar_visualizacao(viz_id)
                        if viz and viz['produto_id'] == produto_id:
                            # Editar visualizacao
                            if len(parts) >= 6 and parts[5] == 'editar':
                                colunas_disponiveis = repo.obter_colunas_disponiveis(produto_id)
                                self._send_html(get_form_editar_visualizacao_html(produto, viz, colunas_disponiveis))
                                return

                            # Deletar visualizacao
                            if len(parts) >= 6 and parts[5] == 'deletar':
                                self._send_html(get_confirmar_delete_visualizacao_html(produto, viz))
                                return

                            # Ver visualizacao
                            # Atualizar dados automaticamente em paralelo
                            atualizar_dados_produto(repo, produto_id)

                            df = obter_dados_para_visualizacao(produto_id, viz)
                            df_viz = aplicar_visualizacao(df, viz)
                            self._send_html(get_visualizacao_html(produto, viz, df_viz))
                            return
                    except ValueError:
                        pass
                    self._send_html("<h1>Visualizacao nao encontrada</h1>", 404)
                    return

                # Posicoes abertas/fechadas (fallback sem visualizacoes)
                if len(parts) >= 4 and parts[3] in ['abertas', 'fechadas']:
                    tipo = parts[3]

                    # Atualizar dados automaticamente em paralelo (apenas para abertas)
                    if tipo == 'abertas':
                        atualizar_dados_produto(repo, produto_id)

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
                # Atualizar dados automaticamente (Bitget sync + ATR stops) em paralelo
                atualizar_dados_produto(repo, produto_id)

                visualizacoes = repo.listar_visualizacoes(produto_id)
                self._send_html(get_produto_html(produto, visualizacoes, repo))
                return

        # API: Lista produtos
        if path == '/api/produtos':
            produtos = repo.listar_produtos()
            self._send_json({'produtos': produtos})
            return

        # ============================================================
        # ROTAS DE TURMAS
        # ============================================================

        # Lista de turmas
        if path == '/turmas':
            turmas_service = TurmasService()
            rentabilidade_service = RentabilidadeService()

            # OTIMIZAÇÃO: Não chama preencher_historico_faltante() no carregamento
            # Use /api/cotacoes/atualizar para atualizar preços quando necessário

            turmas = turmas_service.listar_turmas()

            # Batch otimizado: uma query SQL + uma chamada API para preços em tempo real
            resumos_lista = rentabilidade_service.obter_rentabilidade_resumida_todas_turmas()
            resumos = {r['turma_id']: r for r in resumos_lista}

            self._send_html(get_lista_turmas_html(turmas, resumos))
            return

        # Formulario nova turma
        if path == '/turmas/nova':
            produtos = repo.listar_produtos()
            self._send_html(get_form_nova_turma_html(produtos))
            return

        # Comparar turmas
        if path == '/turmas/comparar':
            turmas_service = TurmasService()
            turmas = turmas_service.listar_turmas()
            self._send_html(get_comparar_turmas_html(turmas, {}))
            return

        # Rentabilidade histórica de todas as turmas (página HTML)
        if path == '/turmas/historico':
            rentabilidade_service = RentabilidadeService()

            # OTIMIZAÇÃO: Não chama preencher_historico_faltante() no carregamento
            # Use /api/cotacoes/atualizar para atualizar preços quando necessário

            # Obter resumo de todas as turmas (inclui rentabilidade max/min)
            turmas_resumo = rentabilidade_service.obter_rentabilidade_resumida_todas_turmas()
            self._send_html(get_rentabilidade_historica_html(turmas_resumo))
            return

        # Rotas de turma especifica: /turmas/{id}, /turmas/{id}/abertas, /turmas/{id}/fechadas, /turmas/{id}/historico, /turmas/{id}/rentabilidade
        if path.startswith('/turmas/') and path != '/turmas/nova' and path != '/turmas/comparar' and path != '/turmas/historico':
            parts = path.split('/')
            if len(parts) >= 3:
                try:
                    turma_id = int(parts[2])
                    turmas_service = TurmasService()
                    rentabilidade_service = RentabilidadeService()

                    # OTIMIZAÇÃO: Não chama preencher_historico_faltante() no carregamento
                    # Use /api/cotacoes/atualizar para atualizar preços quando necessário

                    turma = turmas_service.obter_turma(turma_id)
                    if not turma:
                        self._send_html("<h1>Turma nao encontrada</h1>", 404)
                        return

                    # Grafico de rentabilidade
                    if len(parts) >= 4 and parts[3] == 'rentabilidade':
                        serie = rentabilidade_service.calcular_serie_rentabilidade(turma_id)
                        self._send_html(get_rentabilidade_chart_html(turma, serie))
                        return

                    # Determinar aba ativa
                    tab_ativa = 'abertas'
                    if len(parts) >= 4 and parts[3] in ('abertas', 'fechadas', 'historico'):
                        tab_ativa = parts[3]

                    # Detalhes da turma com abas
                    # Buscar carteira primeiro para popular o cache de preços ANTES de resumo_turma
                    carteira = turmas_service.listar_carteira_turma(turma_id)

                    from datetime import date as date_type
                    hoje = date_type.today().isoformat()

                    # Batch API: busca preços atuais e popula cache module-level
                    # resumo_turma() depois usa o cache em vez de fazer N chamadas individuais
                    coingecko_ids_ativos = list(set(
                        t['coingecko_id'] for t in carteira
                        if t.get('ativo_atual') and t.get('coingecko_id')
                    ))
                    precos_atuais = {}
                    if coingecko_ids_ativos:
                        try:
                            cotacoes_service = CotacoesService()
                            precos_atuais = cotacoes_service.obter_precos_batch_coingecko(coingecko_ids_ativos)
                        except Exception:
                            pass

                    resumo = rentabilidade_service.resumo_turma(turma_id)

                    for trade in carteira:
                        # Calcular dias na turma
                        data_insercao = trade.get('data_insercao', '')
                        if trade.get('ativo_atual'):
                            try:
                                from datetime import datetime as dt
                                d_ins = dt.strptime(data_insercao, '%Y-%m-%d').date()
                                trade['dias'] = (date_type.today() - d_ins).days
                            except Exception:
                                trade['dias'] = 0
                        else:
                            try:
                                from datetime import datetime as dt
                                d_ins = dt.strptime(data_insercao, '%Y-%m-%d').date()
                                d_rem = dt.strptime(trade.get('data_remocao', hoje), '%Y-%m-%d').date()
                                trade['dias'] = (d_rem - d_ins).days
                            except Exception:
                                trade['dias'] = 0

                        preco_entrada = trade.get('preco_entrada_turma', 0) or 0
                        side = (trade.get('side') or '').upper()

                        if trade.get('ativo_atual'):
                            # Trade ativo: usar preço atual da API
                            cg_id = trade.get('coingecko_id')
                            preco_atual = precos_atuais.get(cg_id) if cg_id else None
                            trade['preco_atual'] = preco_atual if preco_atual else preco_entrada
                            # Calcular PnL
                            if preco_entrada and preco_entrada != 0:
                                if side == 'SHORT':
                                    trade['pnl_pct'] = (preco_entrada - trade['preco_atual']) / preco_entrada * 100
                                else:
                                    trade['pnl_pct'] = (trade['preco_atual'] - preco_entrada) / preco_entrada * 100
                            else:
                                trade['pnl_pct'] = 0.0
                        else:
                            # Trade fechado: usar preço de saída
                            preco_saida = trade.get('preco_saida') or preco_entrada
                            trade['preco_atual'] = preco_saida
                            if preco_entrada and preco_entrada != 0:
                                if side == 'SHORT':
                                    trade['pnl_pct'] = (preco_entrada - preco_saida) / preco_entrada * 100
                                else:
                                    trade['pnl_pct'] = (preco_saida - preco_entrada) / preco_entrada * 100
                            else:
                                trade['pnl_pct'] = 0.0

                    self._send_html(get_turma_detalhes_html(turma, resumo, carteira, tab_ativa))
                    return
                except ValueError:
                    pass
            self._send_html("<h1>Turma nao encontrada</h1>", 404)
            return

        # API: Rentabilidade de uma turma (para grafico comparativo)
        if path.startswith('/api/turma/') and path.endswith('/rentabilidade'):
            try:
                turma_id = int(path.split('/')[3])
                rentabilidade_service = RentabilidadeService()
                serie = rentabilidade_service.calcular_serie_rentabilidade(turma_id)
                data = [
                    {
                        'dia': p.dia,
                        'valor_total': p.valor_total,
                        'rentabilidade_acumulada_pct': p.rentabilidade_acumulada_pct
                    }
                    for p in serie
                ]
                self._send_json(data)
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Rentabilidade histórica de uma turma
        if path.startswith('/api/turma/') and path.endswith('/historico'):
            try:
                turma_id = int(path.split('/')[3])
                rentabilidade_service = RentabilidadeService()
                resultado = rentabilidade_service.obter_rentabilidade_historica(turma_id)
                if 'erro' in resultado:
                    self._send_json({'erro': resultado['erro']}, 404)
                else:
                    self._send_json(resultado)
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Listar posições elegíveis para replicação em uma turma
        # GET /api/turma/posicoes-elegiveis?produto_id=1&data_inicio=2026-01-15
        if path == '/api/turma/posicoes-elegiveis':
            try:
                produto_id = int(query.get('produto_id', [0])[0])
                data_inicio = query.get('data_inicio', [None])[0]

                if not produto_id or not data_inicio:
                    self._send_json({'erro': 'produto_id e data_inicio são obrigatórios'}, 400)
                    return

                turmas_service = TurmasService()
                posicoes = turmas_service.listar_posicoes_elegiveis(produto_id, data_inicio)

                self._send_json({
                    'posicoes': posicoes,
                    'total': len(posicoes),
                    'produto_id': produto_id,
                    'data_inicio': data_inicio
                })
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Obter cotação histórica de um ativo
        # GET /api/cotacao/historica?coingecko_id=bitcoin&data=2026-01-15
        # Usa o mesmo método que criar_turma para consistência de preços
        if path == '/api/cotacao/historica':
            try:
                coingecko_id = query.get('coingecko_id', [None])[0]
                data = query.get('data', [None])[0]

                if not coingecko_id or not data:
                    self._send_json({'erro': 'coingecko_id e data são obrigatórios'}, 400)
                    return

                cotacoes_service = CotacoesService()
                resultado = cotacoes_service.obter_preco_historico_exato(coingecko_id, data)

                if resultado.get('status') in ('ok', 'fallback'):
                    self._send_json({
                        'coingecko_id': coingecko_id,
                        'data_solicitada': data,
                        'data_encontrada': resultado.get('data_referencia'),
                        'preco': resultado['preco'],
                        'fonte': resultado.get('fonte', 'coingecko'),
                        'status': resultado['status'],
                        'aviso': resultado.get('aviso')
                    })
                else:
                    self._send_json({
                        'erro': resultado.get('erro', f'Preço não encontrado para {coingecko_id} na data {data}'),
                        'coingecko_id': coingecko_id,
                        'data': data
                    }, 404)
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Rentabilidade de todas as turmas
        if path == '/api/rentabilidade/todas':
            try:
                rentabilidade_service = RentabilidadeService()
                # Parse query params
                produto_id = None
                if '?' in self.path:
                    query = self.path.split('?')[1]
                    params = dict(p.split('=') for p in query.split('&') if '=' in p)
                    if 'produto_id' in params:
                        produto_id = int(params['produto_id'])
                resultado = rentabilidade_service.obter_rentabilidade_todas_turmas(produto_id=produto_id)
                self._send_json(resultado)
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Resumo de rentabilidade de todas as turmas
        if path == '/api/rentabilidade/resumo':
            try:
                rentabilidade_service = RentabilidadeService()
                # Parse query params
                produto_id = None
                if '?' in self.path:
                    query = self.path.split('?')[1]
                    params = dict(p.split('=') for p in query.split('&') if '=' in p)
                    if 'produto_id' in params:
                        produto_id = int(params['produto_id'])
                resultado = rentabilidade_service.obter_rentabilidade_resumida_todas_turmas(produto_id=produto_id)
                self._send_json(resultado)
            except Exception as e:
                self._send_json({'erro': str(e)}, 400)
            return

        # API: Atualizar cotações (preencher histórico faltante)
        if path == '/api/cotacoes/atualizar':
            try:
                cotacoes_service = CotacoesService()
                # Parse query params
                turma_id = None
                if '?' in self.path:
                    query = self.path.split('?')[1]
                    params = dict(p.split('=') for p in query.split('&') if '=' in p)
                    if 'turma_id' in params:
                        turma_id = int(params['turma_id'])

                resultado = cotacoes_service.preencher_historico_faltante(turma_id)
                self._send_json({
                    'sucesso': True,
                    'trades_processados': resultado.get('trades_processados', 0),
                    'dias_preenchidos': resultado.get('dias_preenchidos', 0),
                    'erros': resultado.get('erros', 0)
                })
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 500)
            return

        # API: Atualizar dados (mantido para compatibilidade, usa funcao centralizada)
        if path.startswith('/api/atualizar/'):
            try:
                produto_id = int(path.split('/')[-1])
                atualizar_dados_produto(repo, produto_id)
                self._send_json({'sucesso': True, 'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S")})
            except Exception as e:
                self._send_json({'sucesso': False, 'erro': str(e)}, 500)
            return

        # API: Download Excel com rentabilidade histórica
        if path == '/api/rentabilidade/excel':
            try:
                rentabilidade_service = RentabilidadeService()

                # Parse query params
                produto_id = None
                if '?' in self.path:
                    query = self.path.split('?')[1]
                    params = dict(p.split('=') for p in query.split('&') if '=' in p)
                    if 'produto_id' in params:
                        produto_id = int(params['produto_id'])

                # Buscar todas as turmas
                turmas_service = TurmasService()
                turmas = turmas_service.listar_turmas(produto_id)

                if not turmas:
                    self._send_json({'erro': 'Nenhuma turma encontrada'}, 404)
                    return

                # Pré-carregar preços de todos os ativos uma única vez
                turma_ids = [t['id'] for t in turmas]
                precos_cache = rentabilidade_service.construir_precos_cache(turma_ids)

                # Calcular rentabilidade histórica de cada turma
                all_data = []
                for turma in turmas:
                    turma_id = turma['id']
                    turma_nome = turma['nome']

                    # Obter série de rentabilidade (usa cache compartilhado)
                    serie = rentabilidade_service.calcular_serie_rentabilidade(
                        turma_id, precos_cache=precos_cache
                    )

                    for portfolio in serie:
                        all_data.append({
                            'Data': portfolio.dia,
                            'Turma': turma_nome,
                            'Rentabilidade Acumulada (%)': round(portfolio.rentabilidade_acumulada_pct, 4),
                            'Valor Total (R$)': round(portfolio.valor_total, 2),
                            'Capital Alocado (R$)': round(portfolio.capital_alocado, 2),
                            'Capital em Caixa (R$)': round(portfolio.capital_em_caixa, 2)
                        })

                # Criar DataFrame
                df = pd.DataFrame(all_data)

                # Pivotar para ter turmas como colunas (formato mais útil)
                if not df.empty:
                    # Criar uma aba com dados detalhados
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        # Aba 1: Dados detalhados (long format)
                        df.to_excel(writer, sheet_name='Detalhado', index=False)

                        # Aba 2: Rentabilidade pivotada (wide format)
                        df_pivot = df.pivot_table(
                            index='Data',
                            columns='Turma',
                            values='Rentabilidade Acumulada (%)',
                            aggfunc='first'
                        ).reset_index()
                        df_pivot.to_excel(writer, sheet_name='Rentabilidade por Turma', index=False)

                        # Aba 3: Valor Total pivotado
                        df_valor = df.pivot_table(
                            index='Data',
                            columns='Turma',
                            values='Valor Total (R$)',
                            aggfunc='first'
                        ).reset_index()
                        df_valor.to_excel(writer, sheet_name='Valor por Turma', index=False)

                    buffer.seek(0)

                    # Enviar arquivo
                    filename = f"rentabilidade_turmas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    self._send_excel(buffer, filename)
                else:
                    self._send_json({'erro': 'Nenhum dado de rentabilidade encontrado'}, 404)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({'erro': str(e)}, 500)
            return

        # API: Check ATR Stops (endpoint para cron externo)
        if path == '/api/check_stops':
            try:
                from services.atr_stop_service import atualizar_stops_posicoes_abertas
                resultado = atualizar_stops_posicoes_abertas(repo, verbose=False)
                self._send_json({
                    'status': 'ok',
                    'timestamp': datetime.now().isoformat(),
                    'resultado': {
                        'updated': resultado.get('updated', 0),
                        'skipped': resultado.get('skipped', 0),
                        'unchanged': resultado.get('unchanged', 0),
                        'breached': resultado.get('breached', 0),
                        'errors': resultado.get('errors', [])
                    }
                })
            except Exception as e:
                self._send_json({'status': 'error', 'erro': str(e)}, 500)
            return

        # 404
        self._send_html("<h1>404 - Pagina nao encontrada</h1>", 404)

    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def iniciar_servidor(porta=8080, host='localhost'):
    """Inicia o servidor do dashboard"""
    servidor = HTTPServer((host, porta), DashboardHandler)

    # Detectar qual banco está em uso (PostgreSQL ou SQLite fallback)
    try:
        repo = get_repo()
        if getattr(repo, '_use_sqlite', False):
            _db_info = "SQLite local (fallback)"
        else:
            import re as _re
            _url = getattr(repo, 'db_url', '') or ''
            _host_match = _re.search(r'@([^:/@]+)', _url)
            _db_info = f"PostgreSQL ({_host_match.group(1)})" if _host_match else "PostgreSQL (configurado)"
    except Exception as e:
        _db_info = f"Erro: {e}"

    print("=" * 60)
    print("  DASHBOARD WEB - Products & Positions")
    print("=" * 60)
    print(f"\n  Banco de dados: {_db_info}")
    print(f"\n  Servidor iniciado em: http://localhost:{porta}")
    print(f"\n  Rotas principais:")
    print(f"    /                  - Dashboard")
    print(f"    /menu              - Menu completo")
    print(f"    /produto/novo      - Criar produto")
    print(f"    /posicao/nova      - Criar posicao")
    print(f"    /produto/ID        - Ver produto")
    print(f"    /produto/ID/viz/X  - Visualizacao salva")
    print(f"\n  Turmas & Rentabilidade:")
    print(f"    /turmas            - Lista de turmas")
    print(f"    /turmas/nova       - Criar turma")
    print(f"    /turmas/ID         - Detalhes da turma")
    print(f"    /turmas/comparar   - Comparar rentabilidade")
    print(f"\n  Pressione Ctrl+C para parar.")
    print("=" * 60)

    try:
        import webbrowser
        webbrowser.open(f'http://localhost:{porta}')
    except:
        pass

    # Migração de alocações (CSVs -> Supabase) em background; logs aparecem no Render
    _rodar_migracao_alocacoes_em_background()

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServidor encerrado.")
        servidor.shutdown()


if __name__ == "__main__":
    import sys
    import os
    porta = int(os.environ.get('PORT', sys.argv[1] if len(sys.argv) > 1 else 8080))
    host = os.environ.get('HOST', '0.0.0.0' if os.environ.get('PORT') else 'localhost')
    iniciar_servidor(porta, host)
