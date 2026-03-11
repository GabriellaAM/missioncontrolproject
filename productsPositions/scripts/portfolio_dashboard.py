"""
Dashboard HTML para produtos baseados em portfolio (EXC/HB/LC/Alphacoins).

Reutiliza o mesmo layout e CSS do dashboard de turmas (get_produto_dashboard_html),
mas alimentado por dados do PortfolioService.
"""

import json
from datetime import datetime
from decimal import Decimal


def _get_shared_components():
    """Lazy import to avoid circular dependency with servidor_dashboard"""
    from servidor_dashboard import get_navbar, get_base_styles
    return get_navbar, get_base_styles


def _json_serializer(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)


def get_portfolio_dashboard_html(produto_nome, sub_portfolios, active_data,
                                  active_key, produto_id=0):
    """Dashboard rico para produtos portfolio com abas por sub-portfolio.

    Parameters
    ----------
    produto_nome : str
        "Exponential Coins" or "Alphacoins"
    sub_portfolios : list[dict]
        Each dict has keys: key, nome, data_inicio
    active_data : dict
        From PortfolioService.dados_para_dashboard() with keys:
        resumo, rentabilidade_serie, posicoes_abertas, posicoes_fechadas,
        alocacao_historica
    active_key : str
        Which sub-portfolio is currently selected (e.g. 'EXC')
    produto_id : int
        Product id for future compatibility
    """
    get_navbar, get_base_styles = _get_shared_components()

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    resumo = active_data.get('resumo', {})
    rentab_serie = active_data.get('rentabilidade_serie', [])
    posicoes_abertas = active_data.get('posicoes_abertas', [])
    posicoes_fechadas = active_data.get('posicoes_fechadas', [])
    alocacao_hist = active_data.get('alocacao_historica', [])

    rentab = resumo.get('rentabilidade_acumulada_pct', 0) or 0
    valor_total = resumo.get('valor_total', 0) or 0
    capital_alocado = resumo.get('capital_alocado', 0) or 0
    capital_em_caixa = resumo.get('capital_em_caixa', 0) or 0
    capital_base = resumo.get('capital_base', 0) or 0
    n_abertas = len(posicoes_abertas)
    n_fechadas = len(posicoes_fechadas)

    rentab_class = 'positive' if rentab >= 0 else 'negative'
    rentab_str = f"+{rentab:.2f}%" if rentab >= 0 else f"{rentab:.2f}%"

    active_info = next((sp for sp in sub_portfolios if sp['key'] == active_key), {})
    data_inicio = str(active_info.get('data_inicio', ''))[:10]

    tab_html = ""
    for sp in sub_portfolios:
        active_cls = 'active' if sp['key'] == active_key else ''
        tab_html += (
            f'<button class="turma-tab {active_cls}" '
            f'data-portfolio-key="{sp["key"]}" '
            f'onclick="switchPortfolio(\'{sp["key"]}\', this)">'
            f'{sp.get("nome", sp["key"])}</button>'
        )

    sub_portfolios_json = json.dumps(sub_portfolios, default=_json_serializer,
                                      ensure_ascii=False)
    serie_json = json.dumps(rentab_serie, default=_json_serializer,
                            ensure_ascii=False)
    abertas_json = json.dumps(posicoes_abertas, default=_json_serializer,
                              ensure_ascii=False)
    fechadas_json = json.dumps(posicoes_fechadas, default=_json_serializer,
                               ensure_ascii=False)
    alloc_json = json.dumps(alocacao_hist, default=_json_serializer,
                            ensure_ascii=False)
    resumo_safe = {
        'rentabilidade_acumulada_pct': rentab,
        'valor_total': valor_total,
        'capital_alocado': capital_alocado,
        'capital_em_caixa': capital_em_caixa,
        'capital_base': capital_base,
        'n_abertas': n_abertas,
        'n_fechadas': n_fechadas,
    }
    resumo_json = json.dumps(resumo_safe, default=_json_serializer)

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{produto_nome} - Dashboard</title>
    <script src="/assets/chart.umd.min.js"></script>
    <style>
        {get_base_styles()}

        .dash-header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px 30px;
            border-bottom: 2px solid #4ecca3;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .dash-header h1 {{ color: #4ecca3; font-size: 1.8em; letter-spacing: 1px; margin: 0; }}
        .dash-header .tipo-badge {{
            padding: 4px 14px; border-radius: 20px; font-size: 0.8em; font-weight: bold;
        }}
        .tipo-spot {{ background: #4ecca3; color: #1a1a2e; }}
        .dash-header .date-badge {{
            margin-left: auto;
            background: rgba(78, 204, 163, 0.15);
            color: #4ecca3;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            border: 1px solid rgba(78, 204, 163, 0.3);
        }}

        .turma-tabs-bar {{
            display: flex; gap: 0; padding: 0 30px;
            background: rgba(0,0,0,0.15);
            border-bottom: 1px solid rgba(255,255,255,0.05);
            flex-wrap: wrap;
        }}
        .turma-tab {{
            padding: 12px 24px; background: transparent; color: #888;
            border: none; border-bottom: 3px solid transparent;
            font-size: 1em; cursor: pointer; transition: all 0.3s; font-family: inherit;
        }}
        .turma-tab:hover {{ color: #ccc; background: rgba(78, 204, 163, 0.05); }}
        .turma-tab.active {{ color: #4ecca3; border-bottom-color: #4ecca3; font-weight: 600; }}

        .dash-content {{ padding: 25px 30px; }}

        .summary-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px; padding: 20px; text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .summary-card .s-value {{ font-size: 1.5em; font-weight: bold; color: #fff; margin-bottom: 5px; }}
        .summary-card .s-value.positive {{ color: #4ecca3; }}
        .summary-card .s-value.negative {{ color: #e74c3c; }}
        .summary-card .s-label {{ color: #888; font-size: 0.85em; }}

        .dash-section {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px; padding: 25px; margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .dash-section h3 {{ color: #4ecca3; margin-bottom: 18px; font-size: 1.2em; }}

        .chart-box {{ position: relative; height: 350px; width: 100%; }}

        .sub-tabs {{ display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; }}
        .sub-tab {{
            padding: 8px 18px; background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1); border-radius: 6px;
            color: #888; cursor: pointer; transition: all 0.3s; font-size: 0.9em;
            font-family: inherit;
        }}
        .sub-tab:hover {{ border-color: #4ecca3; color: #4ecca3; }}
        .sub-tab.active {{ background: rgba(78, 204, 163, 0.2); border-color: #4ecca3; color: #4ecca3; }}
        .sub-content {{ display: none; }}
        .sub-content.active {{ display: block; }}

        .dtable {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
        .dtable th, .dtable td {{
            padding: 10px 12px; text-align: right;
            border-bottom: 1px solid rgba(255,255,255,0.07); white-space: nowrap;
        }}
        .dtable th {{
            background: #0d1025; color: #4ecca3; font-weight: 600;
            position: sticky; top: 0; z-index: 2;
            cursor: pointer; user-select: none; transition: color 0.2s;
        }}
        .dtable th:hover {{ color: #fff; }}
        .dtable th .sort-arrow {{ display: inline-block; margin-left: 4px; font-size: 0.7em; opacity: 0.4; }}
        .dtable th.sorted .sort-arrow {{ opacity: 1; }}
        .dtable th:first-child, .dtable td:first-child {{ text-align: left; }}
        .dtable tr:hover {{ background: rgba(78, 204, 163, 0.06); }}

        .tbl-scroll {{ max-height: 400px; overflow: auto; border-radius: 8px; }}
        .tbl-scroll::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        .tbl-scroll::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.2); }}
        .tbl-scroll::-webkit-scrollbar-thumb {{ background: rgba(78, 204, 163, 0.35); border-radius: 3px; }}

        .bdg {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }}
        .bdg-long {{ background: rgba(78, 204, 163, 0.2); color: #4ecca3; }}
        .bdg-short {{ background: rgba(231, 76, 60, 0.2); color: #e74c3c; }}
        .bdg-open {{ background: rgba(52, 152, 219, 0.2); color: #3498db; }}
        .bdg-closed {{ background: rgba(149, 165, 166, 0.2); color: #95a5a6; }}
        .bdg-stop {{ background: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 1px solid rgba(231, 76, 60, 0.3); }}
        .pnl-pos {{ color: #4ecca3; font-weight: 600; }}
        .pnl-neg {{ color: #e74c3c; font-weight: 600; }}

        /* --- Gear menu (config) --- */
        .gear-menu-wrapper {{ position: relative; display: inline-flex; align-items: center; }}
        .gear-btn {{
            background: none; border: none; cursor: pointer; padding: 6px;
            border-radius: 8px; transition: background 0.2s, transform 0.3s; display:flex; align-items:center;
        }}
        .gear-btn:hover {{ background: rgba(78,204,163,0.12); }}
        .gear-btn.open {{ transform: rotate(90deg); }}
        .gear-btn svg {{ width: 22px; height: 22px; fill: #888; transition: fill 0.2s; }}
        .gear-btn:hover svg, .gear-btn.open svg {{ fill: #4ecca3; }}
        .gear-panel {{
            display: none; position: absolute; right: 0; top: calc(100% + 8px);
            background: #16213e; border: 1px solid rgba(78,204,163,0.35); border-radius: 12px;
            min-width: 260px; z-index: 200; box-shadow: 0 8px 28px rgba(0,0,0,.55);
            padding: 6px 0; max-height: 80vh; overflow-y: auto;
        }}
        .gear-panel.open {{ display: block; }}
        .gear-panel-group {{ padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .gear-panel-group:last-child {{ border-bottom: none; }}
        .gear-panel-title {{
            font-size: 0.7em; text-transform: uppercase; letter-spacing: 1.2px;
            color: #555; padding: 6px 18px 4px; font-weight: 700;
        }}
        .gear-panel a, .gear-panel button.gear-item {{
            display: flex; align-items: center; gap: 10px; width: 100%; padding: 10px 18px;
            color: #ccc; text-decoration: none; font-size: 0.85em; border: none;
            background: none; cursor: pointer; text-align: left; transition: background 0.15s, color 0.15s;
        }}
        .gear-panel a:hover, .gear-panel button.gear-item:hover {{ background: rgba(78,204,163,0.1); color: #4ecca3; }}
        .gear-panel .gear-icon {{ width: 16px; text-align: center; font-size: 1em; flex-shrink: 0; }}
        .gear-panel .gear-accent {{ color: #4ecca3; }}

        .period-btn {{
            padding: 4px 12px; border-radius: 4px; border: 1px solid #333;
            background: transparent; color: #888; font-size: 12px; cursor: pointer;
            transition: all 0.2s; font-family: inherit;
        }}
        .period-btn:hover {{ color: #e0e0e0; border-color: #4ecca3; }}
        .period-btn.active {{ background: #4ecca3; color: #1a1a2e; border-color: #4ecca3; font-weight: 600; }}

        .loading-overlay {{ text-align: center; padding: 60px 20px; color: #888; }}
        .loading-overlay .spinner {{
            border: 3px solid #2a2a4a; border-top: 3px solid #4ecca3;
            border-radius: 50%; width: 30px; height: 30px;
            animation: spn 0.8s linear infinite; margin: 0 auto 15px;
        }}
        @keyframes spn {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .empty-msg {{ text-align: center; color: #666; padding: 40px; font-style: italic; }}
    </style>
</head>
<body>
    {get_navbar()}

    <div class="dash-header">
        <div><h1>{produto_nome}</h1></div>
        <span class="tipo-badge tipo-spot">Spot</span>
        <div style="margin-left:auto; display:flex; align-items:center; gap:10px;">
            <span id="statusMsg" style="color:#888; font-size:0.85em;"></span>
            <div class="date-badge">Atualizado: {timestamp}</div>
            <div class="gear-menu-wrapper" id="gearWrapper">
                <button class="gear-btn" id="gearBtn" onclick="toggleGearMenu()" title="Configura&ccedil;&otilde;es">
                    <svg viewBox="0 0 24 24"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 00.12-.61l-1.92-3.32a.49.49 0 00-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.484.484 0 00-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.49.49 0 00-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.07.63-.07.94s.02.64.07.94l-2.03 1.58a.49.49 0 00-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6A3.6 3.6 0 1115.6 12 3.611 3.611 0 0112 15.6z"/></svg>
                </button>
                <div class="gear-panel" id="gearPanel">
                    <div class="gear-panel-group">
                        <div class="gear-panel-title">Pre&ccedil;os</div>
                        <button class="gear-item" onclick="atualizarCotacoes(); toggleGearMenu();">
                            <span class="gear-icon">&#8635;</span> Atualizar Pre&ccedil;os
                        </button>
                    </div>
                    <div class="gear-panel-group">
                        <div class="gear-panel-title">Produto</div>
                        <a href="/produto/{produto_id}/editar"><span class="gear-icon">&#9998;</span> Editar Produto</a>
                        <a href="/produto/{produto_id}/visualizacoes"><span class="gear-icon">&#128065;</span> Visualiza&ccedil;&otilde;es</a>
                        <a href="/produto/{produto_id}/atributos"><span class="gear-icon">&#9776;</span> Atributos</a>
                        <a href="/atr/config?produto_id={produto_id}"><span class="gear-icon">&#9632;</span> ATR Stop</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="turma-tabs-bar" id="portfolioTabs">
        {tab_html if tab_html else '<span style="padding:12px;color:#666;">Nenhum sub-portfolio</span>'}
    </div>

    <div class="dash-content">
        <div class="summary-row" id="summaryCards">
            <div class="summary-card">
                <div class="s-value {rentab_class}" id="card-rentab">{rentab_str}</div>
                <div class="s-label">Rentabilidade Acumulada</div>
            </div>
            <div class="summary-card">
                <div class="s-value" id="card-trades">{n_abertas} / {n_fechadas}</div>
                <div class="s-label">Ativos / Fechados</div>
            </div>
        </div>

        <div class="dash-section" id="sectionChart">
            <h3 id="chartTitle">Rentabilidade Acumulada</h3>
            <div class="chart-toggles" style="display:flex; gap:16px; margin-bottom:10px; align-items:center; flex-wrap:wrap;">
                <label style="display:flex; align-items:center; gap:5px; cursor:pointer; color:#e0e0e0; font-size:13px;">
                    <input type="checkbox" id="togglePortfolio" checked style="accent-color:#4ecca3; width:15px; height:15px; cursor:pointer;">
                    <span style="display:inline-block; width:14px; height:3px; background:#4ecca3; border-radius:2px;"></span>
                    <span id="togglePortfolioLabel">{active_info.get('nome', active_key)}</span>
                </label>
                <label style="display:flex; align-items:center; gap:5px; cursor:pointer; color:#e0e0e0; font-size:13px;">
                    <input type="checkbox" id="toggleBTC" checked style="accent-color:#f7931a; width:15px; height:15px; cursor:pointer;">
                    <span style="display:inline-block; width:14px; height:0; border-top:2px dashed #f7931a;"></span>
                    Bitcoin (BTC)
                </label>
                <span style="color:#555; font-size:13px;">|</span>
                <span style="color:#888; font-size:13px;">Comparar com:</span>
                <div id="compareCheckboxes" style="display:inline-flex; flex-wrap:wrap; gap:10px 14px; align-items:center;"></div>
            </div>
            <div style="display:flex; gap:6px; margin-bottom:12px; align-items:center; flex-wrap:wrap;">
                <button class="period-btn active" data-period="MAX" onclick="setPeriod('MAX',this)">MAX</button>
                <button class="period-btn" data-period="YTD" onclick="setPeriod('YTD',this)">YTD</button>
                <button class="period-btn" data-period="1Y" onclick="setPeriod('1Y',this)">1A</button>
                <button class="period-btn" data-period="6M" onclick="setPeriod('6M',this)">6M</button>
                <button class="period-btn" data-period="3M" onclick="setPeriod('3M',this)">3M</button>
                <button class="period-btn" data-period="1M" onclick="setPeriod('1M',this)">1M</button>
                <span style="color:#444; margin:0 4px;">|</span>
                <input type="date" id="periodStart" style="background:#1a1a2e; color:#e0e0e0; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:12px; cursor:pointer;" onchange="setCustomPeriod()">
                <span style="color:#666; font-size:12px;">a</span>
                <input type="date" id="periodEnd" style="background:#1a1a2e; color:#e0e0e0; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:12px; cursor:pointer;" onchange="setCustomPeriod()">
            </div>
            <div id="chartLoading" class="loading-overlay"><div class="spinner"></div>Carregando...</div>
            <div class="chart-box" id="chartWrapper" style="display:none;">
                <canvas id="chartRent"></canvas>
            </div>
        </div>

        <div class="dash-section">
            <h3>PnL dos ativos no per&iacute;odo</h3>
            <div style="display:flex; gap:6px; margin-bottom:12px; align-items:center; flex-wrap:wrap;">
                <button class="period-btn pnl-period-btn active" onclick="setPnlPeriod('MAX',this)">MAX</button>
                <button class="period-btn pnl-period-btn" onclick="setPnlPeriod('YTD',this)">YTD</button>
                <button class="period-btn pnl-period-btn" onclick="setPnlPeriod('1Y',this)">1A</button>
                <button class="period-btn pnl-period-btn" onclick="setPnlPeriod('6M',this)">6M</button>
                <button class="period-btn pnl-period-btn" onclick="setPnlPeriod('3M',this)">3M</button>
                <button class="period-btn pnl-period-btn" onclick="setPnlPeriod('1M',this)">1M</button>
                <span style="color:#444; margin:0 4px;">|</span>
                <input type="date" id="pnlPeriodStart" style="background:#1a1a2e; color:#e0e0e0; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:12px; cursor:pointer;" onchange="setPnlCustomPeriod()">
                <span style="color:#666; font-size:12px;">a</span>
                <input type="date" id="pnlPeriodEnd" style="background:#1a1a2e; color:#e0e0e0; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:12px; cursor:pointer;" onchange="setPnlCustomPeriod()">
            </div>
            <div id="pnlChartLoading" class="loading-overlay"><div class="spinner"></div>Carregando...</div>
            <div class="chart-box" id="pnlChartWrapper" style="height:280px; display:none;">
                <canvas id="chartPnlAbertas"></canvas>
            </div>
        </div>

        <div class="dash-section" id="sectionPositions">
            <h3>Posi&ccedil;&otilde;es</h3>
            <div class="sub-tabs" id="posTabs">
                <button class="sub-tab active" onclick="switchPosTab('abertas', this)">Abertas (<span id="countAbertas">{n_abertas}</span>)</button>
                <button class="sub-tab" onclick="switchPosTab('fechadas', this)">Fechadas (<span id="countFechadas">{n_fechadas}</span>)</button>
                <button class="sub-tab" onclick="switchPosTab('historico', this)">Hist&oacute;rico (<span id="countHistorico">{n_abertas + n_fechadas}</span>)</button>
            </div>

            <div id="tab-abertas" class="sub-content active">
                <div class="tbl-scroll"><table class="dtable" id="tblAbertas">
                    <thead><tr><th>Ativo</th><th>Aloca&ccedil;&atilde;o%</th><th>Data Entrada</th><th>Dias</th><th>Pre&ccedil;o Entrada</th><th>Pre&ccedil;o Atual</th><th>Stop</th><th>PnL%</th></tr></thead>
                    <tbody id="tbAbertas"></tbody>
                </table></div>
            </div>
            <div id="tab-fechadas" class="sub-content">
                <div class="tbl-scroll"><table class="dtable" id="tblFechadas">
                    <thead><tr><th>Ativo</th><th>Data Entrada</th><th>Data Sa&iacute;da</th><th>Dias</th><th>Pre&ccedil;o Entrada</th><th>Pre&ccedil;o Sa&iacute;da</th><th>Stop</th><th>PnL%</th></tr></thead>
                    <tbody id="tbFechadas"></tbody>
                </table></div>
            </div>
            <div id="tab-historico" class="sub-content">
                <div class="tbl-scroll"><table class="dtable" id="tblHistorico">
                    <thead><tr><th>Ativo</th><th>Status</th><th>Data Entrada</th><th>Data Sa&iacute;da</th><th>Dias</th><th>Pre&ccedil;o Entrada</th><th>Pre&ccedil;o Sa&iacute;da/Atual</th><th>Stop</th><th>PnL%</th></tr></thead>
                    <tbody id="tbHistorico"></tbody>
                </table></div>
            </div>
        </div>

        <div class="dash-section">
            <h3>Evolu&ccedil;&atilde;o da Aloca&ccedil;&atilde;o</h3>
            <div class="chart-box" style="height:320px;">
                <canvas id="chartAllocTimeline"></canvas>
            </div>
        </div>

    </div>

<script>
var PORTFOLIO_KEY = '{active_key}';
var PRODUTO_ID = {produto_id};
var RESUMO = {resumo_json};
var SERIE = {serie_json};
var ABERTAS = {abertas_json};
var FECHADAS = {fechadas_json};
var ALLOC_HIST = {alloc_json};
var SUB_PORTFOLIOS = {sub_portfolios_json};
var DATA_INICIO = '{data_inicio}';
var rentChart = null;
var pnlAbertasChart = null;
var allocTimelineChart = null;
var btcSerie = null;
var compareSeries = {{}};
var COMPARE_COLORS = ['#e056fd','#3498db','#f39c12','#1abc9c','#9b59b6'];
var currentSerie = null;
var fullSerie = null;
var activePeriod = 'MAX';
var periodStartDate = null;
var periodEndDate = null;

function filterSerieByDate(serie, startDate, endDate) {{
    if (!serie || !serie.length) return serie;
    return serie.filter(function(s) {{
        var d = s.dia;
        if (startDate && d < startDate) return false;
        if (endDate && d > endDate) return false;
        return true;
    }});
}}

function rebaseRent(serie) {{
    if (!serie || serie.length < 2) return serie;
    var base = serie[0].rentabilidade_acumulada_pct;
    if (base === 0) return serie;
    var baseFactor = 1 + base / 100;
    return serie.map(function(s) {{
        return {{
            dia: s.dia,
            rentabilidade_acumulada_pct: ((1 + s.rentabilidade_acumulada_pct / 100) / baseFactor - 1) * 100
        }};
    }});
}}

function getDateForPeriod(period) {{
    var now = new Date();
    var y = now.getFullYear(), m = now.getMonth(), d = now.getDate();
    switch(period) {{
        case '1M': return new Date(y, m - 1, d).toISOString().slice(0, 10);
        case '3M': return new Date(y, m - 3, d).toISOString().slice(0, 10);
        case '6M': return new Date(y, m - 6, d).toISOString().slice(0, 10);
        case '1Y': return new Date(y - 1, m, d).toISOString().slice(0, 10);
        case 'YTD': return y + '-01-01';
        default: return null;
    }}
}}

function setPeriod(period, btn) {{
    activePeriod = period;
    document.querySelectorAll('.period-btn').forEach(function(b){{ b.classList.remove('active'); }});
    if (btn) btn.classList.add('active');
    var startDate = getDateForPeriod(period);
    periodStartDate = startDate;
    periodEndDate = null;
    document.getElementById('periodStart').value = startDate || '';
    document.getElementById('periodEnd').value = '';
    applyPeriodFilter();
}}

function setCustomPeriod() {{
    var s = document.getElementById('periodStart').value;
    var e = document.getElementById('periodEnd').value;
    periodStartDate = s || null;
    periodEndDate = e || null;
    activePeriod = 'CUSTOM';
    document.querySelectorAll('.period-btn').forEach(function(b){{ b.classList.remove('active'); }});
    applyPeriodFilter();
}}

function applyPeriodFilter() {{
    if (!fullSerie) return;
    if (activePeriod === 'MAX' && !periodStartDate && !periodEndDate) {{
        currentSerie = fullSerie;
        renderRentChartFiltered(fullSerie, btcSerie, compareSeries);
        return;
    }}
    var inicio = periodStartDate || null;
    var fim = periodEndDate || null;
    var urlMain = '/api/portfolio/' + PORTFOLIO_KEY + '/rentabilidade_serie?';
    if (inicio) urlMain += 'inicio=' + encodeURIComponent(inicio) + '&';
    if (fim) urlMain += 'fim=' + encodeURIComponent(fim);
    var compareKeys = Object.keys(compareSeries);
    var query = (inicio ? 'inicio=' + encodeURIComponent(inicio) + '&' : '') + (fim ? 'fim=' + encodeURIComponent(fim) : '');
    var promises = [fetch(urlMain).then(function(r){{ return r.json(); }})];
    compareKeys.forEach(function(k) {{
        promises.push(fetch('/api/portfolio/' + k + '/rentabilidade_serie?' + query).then(function(r){{ return r.json(); }}));
    }});
    Promise.all(promises).then(function(results) {{
        var data = results[0];
        if (data.erro) {{ console.error(data.erro); return; }}
        var serieFullRes = data.rentabilidade_serie || [];
        if (!serieFullRes.length) {{ renderRentChartFiltered([], null, {{}}); return; }}
        var rebased = rebaseRent(serieFullRes);
        currentSerie = rebased;
        var filteredBtc = btcSerie ? rebaseRent(filterSerieByDate(btcSerie, inicio, fim)) : null;
        var filteredCompareSeries = {{}};
        for (var i = 0; i < compareKeys.length; i++) {{
            var d = results[i + 1];
            if (d && !d.erro && d.rentabilidade_serie && d.rentabilidade_serie.length) {{
                filteredCompareSeries[compareKeys[i]] = rebaseRent(d.rentabilidade_serie);
            }}
        }}
        renderRentChartFiltered(rebased, filteredBtc, filteredCompareSeries);
    }}).catch(function(e) {{ console.error('Erro ao buscar s\\u00e9rie do per\\u00edodo:', e); }});
}}

function toggleGearMenu() {{
    var btn = document.getElementById('gearBtn');
    var panel = document.getElementById('gearPanel');
    btn.classList.toggle('open');
    panel.classList.toggle('open');
}}
document.addEventListener('click', function(e) {{
    var wrap = document.getElementById('gearWrapper');
    if (wrap && !wrap.contains(e.target)) {{
        document.getElementById('gearBtn').classList.remove('open');
        document.getElementById('gearPanel').classList.remove('open');
    }}
}});

function fmtPrice(v) {{
    if (v == null) return '\\u2014';
    if (v === 0) return '0.0000';
    if (Math.abs(v) >= 1) return v.toLocaleString('en-US', {{minimumFractionDigits:4, maximumFractionDigits:4}});
    var digits = Math.min(20, Math.max(4, -Math.floor(Math.log10(Math.abs(v))) + 3));
    return v.toLocaleString('en-US', {{minimumFractionDigits:digits, maximumFractionDigits:digits}});
}}
function fmtPnl(p) {{
    if (p == null) return '\\u2014';
    var cls = p >= 0 ? 'pnl-pos' : 'pnl-neg';
    return '<span class="'+cls+'">'+(p>=0?'+':'')+p.toFixed(2)+'%</span>';
}}
function sideBdg(s) {{
    if (!s) return '\\u2014';
    return s.toLowerCase()==='long' ? '<span class="bdg bdg-long">LONG</span>' : '<span class="bdg bdg-short">SHORT</span>';
}}
function statusBdg(isOpen) {{
    return isOpen ? '<span class="bdg bdg-open">Aberto</span>' : '<span class="bdg bdg-closed">Fechado</span>';
}}
function fmtDate(d) {{ return d || '\\u2014'; }}
function fmtPct(v) {{
    if (v == null) return '\\u2014';
    return v.toFixed(1) + '%';
}}

function updateCards(r) {{
    var rentab = r.rentabilidade_acumulada_pct||0;
    var el = document.getElementById('card-rentab');
    el.textContent = (rentab>=0?'+':'')+rentab.toFixed(2)+'%';
    el.className = 's-value '+(rentab>=0?'positive':'negative');
    document.getElementById('card-trades').textContent = (r.n_abertas||0)+' / '+(r.n_fechadas||0);
    document.getElementById('countAbertas').textContent = r.n_abertas||0;
    document.getElementById('countFechadas').textContent = r.n_fechadas||0;
    document.getElementById('countHistorico').textContent = (r.n_abertas||0)+(r.n_fechadas||0);
}}

function fmtStop(v) {{ if(v==null||v==undefined) return '\\u2014'; if(v===-1) return '<span class="bdg bdg-stop">Stop Atingido</span>'; return '$ '+fmtPrice(v); }}
function rowOpen(t) {{
    return '<tr><td>'+t.ativo+'</td><td>'+fmtPct(t.alocacao_pct)+'</td><td>'+fmtDate(t.data_entrada)+'</td><td>'+(t.dias!=null?t.dias:'\\u2014')+'</td><td>$ '+fmtPrice(t.preco_entrada)+'</td><td>$ '+fmtPrice(t.preco_atual)+'</td><td>'+fmtStop(t.stop_atual)+'</td><td>'+fmtPnl(t.pnl_pct)+'</td></tr>';
}}
function rowClosed(t) {{
    return '<tr><td>'+t.ativo+'</td><td>'+fmtDate(t.data_entrada)+'</td><td>'+fmtDate(t.data_saida)+'</td><td>'+(t.dias!=null?t.dias:'\\u2014')+'</td><td>$ '+fmtPrice(t.preco_entrada)+'</td><td>$ '+fmtPrice(t.preco_saida)+'</td><td>'+fmtStop(t.stop_atual)+'</td><td>'+fmtPnl(t.pnl_pct)+'</td></tr>';
}}
function rowHist(t) {{
    var isOpen = !t.data_saida;
    var precoFim = isOpen ? t.preco_atual : t.preco_saida;
    return '<tr><td>'+t.ativo+'</td><td>'+statusBdg(isOpen)+'</td><td>'+fmtDate(t.data_entrada)+'</td><td>'+fmtDate(t.data_saida)+'</td><td>'+(t.dias!=null?t.dias:'\\u2014')+'</td><td>$ '+fmtPrice(t.preco_entrada)+'</td><td>$ '+fmtPrice(precoFim)+'</td><td>'+fmtStop(t.stop_atual)+'</td><td>'+fmtPnl(t.pnl_pct)+'</td></tr>';
}}

var sortSt = {{}};
function setupSort(tbl, data, cols, renderRow) {{
    var key = tbl.id;
    var ths = tbl.querySelectorAll('thead th');
    ths.forEach(function(th, idx) {{
        if (!th.querySelector('.sort-arrow')) th.innerHTML = th.textContent+' <span class="sort-arrow">\\u25B2</span>';
        th.onclick = function() {{
            var prev = sortSt[key]; var dir = 'asc';
            if (prev && prev.col===idx) dir = prev.dir==='asc'?'desc':'asc';
            sortSt[key] = {{col:idx, dir:dir}};
            ths.forEach(function(h){{h.classList.remove('sorted');h.querySelector('.sort-arrow').textContent='\\u25B2';}});
            th.classList.add('sorted');
            th.querySelector('.sort-arrow').textContent = dir==='asc'?'\\u25B2':'\\u25BC';
            var c=cols[idx];
            var sorted=[...data].sort(function(a,b){{
                var va=c.v(a),vb=c.v(b);
                if(c.t==='text'){{va=(va||'').toLowerCase();vb=(vb||'').toLowerCase();}}
                else if(c.t==='date'){{va=va||'0';vb=vb||'0';}}
                else{{va=typeof va==='number'&&!isNaN(va)?va:-Infinity;vb=typeof vb==='number'&&!isNaN(vb)?vb:-Infinity;}}
                var cmp=va<vb?-1:va>vb?1:0;
                return dir==='asc'?cmp:-cmp;
            }});
            tbl.querySelector('tbody').innerHTML=sorted.map(renderRow).join('');
        }};
    }});
}}

function renderAbertas(data) {{
    var tb = document.getElementById('tbAbertas');
    if (!tb) return;
    document.getElementById('countAbertas').textContent = data.length;
    if(!data.length) {{
        tb.innerHTML='<tr><td colspan="8" class="empty-msg">Nenhuma posi\\u00e7\\u00e3o aberta</td></tr>';
        return;
    }}
    tb.innerHTML = data.map(rowOpen).join('');
    setupSort(document.getElementById('tblAbertas'), data, [
        {{t:'text',v:function(r){{return r.ativo;}}}},
        {{t:'num',v:function(r){{return r.alocacao_pct;}}}},
        {{t:'date',v:function(r){{return r.data_entrada;}}}},
        {{t:'num',v:function(r){{return r.dias;}}}},
        {{t:'num',v:function(r){{return r.preco_entrada;}}}},
        {{t:'num',v:function(r){{return r.preco_atual;}}}},
        {{t:'num',v:function(r){{return r.stop_atual;}}}},
        {{t:'num',v:function(r){{return r.pnl_pct;}}}}
    ], rowOpen);
}}

function renderFechadas(data) {{
    var tb = document.getElementById('tbFechadas');
    document.getElementById('countFechadas').textContent = data.length;
    document.getElementById('countHistorico').textContent = ABERTAS.length + data.length;
    if(!data.length) {{
        tb.innerHTML='<tr><td colspan="8" class="empty-msg">Nenhum trade fechado</td></tr>';
        return;
    }}
    tb.innerHTML = data.map(rowClosed).join('');
    setupSort(document.getElementById('tblFechadas'), data, [
        {{t:'text',v:function(r){{return r.ativo;}}}},
        {{t:'date',v:function(r){{return r.data_entrada;}}}},
        {{t:'date',v:function(r){{return r.data_saida;}}}},
        {{t:'num',v:function(r){{return r.dias;}}}},
        {{t:'num',v:function(r){{return r.preco_entrada;}}}},
        {{t:'num',v:function(r){{return r.preco_saida;}}}},
        {{t:'num',v:function(r){{return r.stop_atual;}}}},
        {{t:'num',v:function(r){{return r.pnl_pct;}}}}
    ], rowClosed);
}}

function renderHistorico(abertas, fechadas) {{
    var all = [];
    abertas.forEach(function(t) {{
        all.push({{ativo:t.ativo, data_entrada:t.data_entrada, data_saida:null, dias:t.dias, preco_entrada:t.preco_entrada, preco_atual:t.preco_atual, preco_saida:null, stop_atual:t.stop_atual, pnl_pct:t.pnl_pct}});
    }});
    fechadas.forEach(function(t) {{
        all.push({{ativo:t.ativo, data_entrada:t.data_entrada, data_saida:t.data_saida, dias:t.dias, preco_entrada:t.preco_entrada, preco_atual:null, preco_saida:t.preco_saida, stop_atual:t.stop_atual, pnl_pct:t.pnl_pct}});
    }});
    all.sort(function(a,b){{ return (b.data_entrada||'').localeCompare(a.data_entrada||''); }});
    var tb = document.getElementById('tbHistorico');
    document.getElementById('countHistorico').textContent = all.length;
    if(!all.length) {{
        tb.innerHTML='<tr><td colspan="9" class="empty-msg">Nenhum trade</td></tr>';
        return;
    }}
    tb.innerHTML = all.map(rowHist).join('');
    setupSort(document.getElementById('tblHistorico'), all, [
        {{t:'text',v:function(r){{return r.ativo;}}}},
        {{t:'text',v:function(r){{return r.data_saida?'z':'a';}}}},
        {{t:'date',v:function(r){{return r.data_entrada;}}}},
        {{t:'date',v:function(r){{return r.data_saida;}}}},
        {{t:'num',v:function(r){{return r.dias;}}}},
        {{t:'num',v:function(r){{return r.preco_entrada;}}}},
        {{t:'num',v:function(r){{return r.data_saida?r.preco_saida:r.preco_atual;}}}},
        {{t:'num',v:function(r){{return r.stop_atual;}}}},
        {{t:'num',v:function(r){{return r.pnl_pct;}}}}
    ], rowHist);
}}

function renderPnlChart(ativos) {{
    if(pnlAbertasChart) pnlAbertasChart.destroy();
    var ctx = document.getElementById('chartPnlAbertas');
    if(!ctx) return;
    ctx = ctx.getContext('2d');
    if(!ativos || !ativos.length) {{
        pnlAbertasChart = new Chart(ctx, {{ type:'bar', data:{{labels:[],datasets:[]}}, options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{enabled:false}}}},scales:{{x:{{display:false}},y:{{display:false}}}}}} }});
        return;
    }}
    var labels = ativos.map(function(t){{ return t.ativo; }});
    var values = ativos.map(function(t){{ return t.pnl_pct != null ? t.pnl_pct : 0; }});
    var colors = values.map(function(v){{ return v >= 0 ? '#4ecca3' : '#e74c3c'; }});
    pnlAbertasChart = new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: labels,
            datasets: [{{
                label: 'PnL %',
                data: values,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 4
            }}]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ display: false }},
                tooltip: {{
                    backgroundColor: 'rgba(26,26,46,0.95)',
                    titleColor: '#4ecca3',
                    bodyColor: '#e0e0e0',
                    callbacks: {{ label: function(c){{ return (c.raw >= 0 ? '+' : '') + c.raw.toFixed(2) + '%'; }} }}
                }}
            }},
            scales: {{
                x: {{
                    ticks: {{ color: '#e0e0e0', font: {{ size: 11 }}, maxRotation: 45 }},
                    grid: {{ color: 'rgba(255,255,255,0.05)' }},
                    title: {{ display: true, text: 'Ativos', color: '#888' }}
                }},
                y: {{
                    ticks: {{ color: '#666', callback: function(v){{ return v + '%'; }} }},
                    grid: {{ color: 'rgba(255,255,255,0.05)' }},
                    title: {{ display: true, text: 'Rentabilidade Acumulada (%)', color: '#888' }}
                }}
            }}
        }}
    }});
}}

var activePnlPeriod = 'MAX';
var pnlLoading = false;

function fetchPnlData(inicio, fim) {{
    if (pnlLoading) return;
    pnlLoading = true;
    var loadEl = document.getElementById('pnlChartLoading');
    var wrapEl = document.getElementById('pnlChartWrapper');
    if (loadEl) loadEl.style.display = 'block';
    if (wrapEl) wrapEl.style.display = 'none';
    var url = '/api/portfolio/' + PORTFOLIO_KEY + '/pnl';
    var params = [];
    if (inicio) params.push('inicio=' + inicio);
    if (fim) params.push('fim=' + fim);
    if (params.length) url += '?' + params.join('&');
    fetch(url).then(function(r){{ return r.json(); }}).then(function(data) {{
        pnlLoading = false;
        if (loadEl) loadEl.style.display = 'none';
        if (wrapEl) wrapEl.style.display = 'block';
        if (data.erro) {{ console.error(data.erro); return; }}
        renderPnlChart(data.ativos || []);
    }}).catch(function(e) {{
        pnlLoading = false;
        if (loadEl) loadEl.style.display = 'none';
        if (wrapEl) wrapEl.style.display = 'block';
        console.error('Erro ao buscar PnL:', e);
    }});
}}

function applyPnlFilter() {{
    var inicio = null, fim = null;
    if (activePnlPeriod !== 'MAX') {{
        inicio = getDateForPeriod(activePnlPeriod);
    }}
    if (activePnlPeriod === 'CUSTOM') {{
        inicio = document.getElementById('pnlPeriodStart').value || null;
        fim = document.getElementById('pnlPeriodEnd').value || null;
    }}
    fetchPnlData(inicio, fim);
}}

function setPnlPeriod(period, btn) {{
    activePnlPeriod = period;
    document.querySelectorAll('.pnl-period-btn').forEach(function(b){{ b.classList.remove('active'); }});
    if (btn) btn.classList.add('active');
    var startDate = getDateForPeriod(period);
    document.getElementById('pnlPeriodStart').value = startDate || '';
    document.getElementById('pnlPeriodEnd').value = '';
    applyPnlFilter();
}}

function setPnlCustomPeriod() {{
    activePnlPeriod = 'CUSTOM';
    document.querySelectorAll('.pnl-period-btn').forEach(function(b){{ b.classList.remove('active'); }});
    applyPnlFilter();
}}

var ALLOC_COLORS = ['#4ecca3','#e74c3c','#3498db','#f39c12','#9b59b6','#1abc9c','#e67e22','#2ecc71','#e84393','#00cec9','#fd79a8','#6c5ce7','#ffeaa7','#dfe6e9','#fab1a0','#a29bfe'];

function renderAllocTimeline(allocData) {{
    if(allocTimelineChart) allocTimelineChart.destroy();
    if(!allocData || !allocData.length) return;

    var dates = allocData.map(function(d){{ return d.data || d.date; }});
        var assetSet = {{}};
        allocData.forEach(function(row) {{
            Object.keys(row).forEach(function(k) {{
                if(k !== 'data' && k !== 'date') assetSet[k] = 1;
            }});
        }});
        var assetKeys = Object.keys(assetSet).sort();

        var datasets = [];
        assetKeys.forEach(function(asset, idx) {{
            var color = ALLOC_COLORS[idx % ALLOC_COLORS.length];
            var data = allocData.map(function(row) {{ return row[asset] != null ? row[asset] : 0; }});
            datasets.push({{
                label: asset,
                data: data,
                backgroundColor: color + 'AA',
                borderColor: color,
                borderWidth: 1,
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                pointHoverRadius: 3
            }});
        }});

        allocTimelineChart = new Chart(document.getElementById('chartAllocTimeline').getContext('2d'), {{
            type: 'line',
            data: {{ labels: dates, datasets: datasets }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                scales: {{
                    x: {{
                        stacked: true,
                        ticks: {{ color: '#666', maxTicksLimit: 12, maxRotation: 0 }},
                        grid: {{ color: 'rgba(255,255,255,0.05)' }}
                    }},
                    y: {{
                        stacked: true,
                        max: 100,
                        ticks: {{ color: '#666', callback: function(v){{ return v + '%'; }} }},
                        grid: {{ color: 'rgba(255,255,255,0.05)' }},
                        title: {{ display: true, text: 'Aloca\\u00e7\\u00e3o (%)', color: '#888' }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'bottom',
                        labels: {{ color: '#e0e0e0', padding: 10, usePointStyle: true, pointStyle: 'rectRounded', font: {{ size: 11 }},
                            generateLabels: function(chart) {{
                                return chart.data.datasets.map(function(ds, i) {{
                                    return {{
                                        text: ds.label,
                                        fillStyle: ds.hidden ? '#555' : ds.backgroundColor,
                                        strokeStyle: ds.borderColor,
                                        lineWidth: 1,
                                        hidden: false,
                                        datasetIndex: i,
                                        fontColor: ds.hidden ? '#666' : '#e0e0e0',
                                        pointStyle: 'rectRounded'
                                    }};
                                }});
                            }}
                        }},
                        onClick: function(e, item, legend) {{
                            var idx = item.datasetIndex;
                            var ds = legend.chart.data.datasets[idx];
                            ds.hidden = !ds.hidden;
                            legend.chart.update();
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(26,26,46,0.95)',
                        titleColor: '#4ecca3',
                        bodyColor: '#e0e0e0',
                        callbacks: {{
                            label: function(ctx) {{
                                if(ctx.raw === 0) return null;
                                return ctx.dataset.label + ': ' + ctx.raw.toFixed(1) + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});
}}

function buildRentDatasets(serie, btc, compare) {{
    var datasets = [];
    var chartEl = document.getElementById('chartRent');
    if (!chartEl || !serie || !serie.length) return {{labels: [], datasets: []}};
    var ctx = chartEl.getContext('2d');
    var gradient = ctx.createLinearGradient(0,0,0,350);
    gradient.addColorStop(0,'rgba(78,204,163,0.3)');
    gradient.addColorStop(1,'rgba(78,204,163,0.0)');
    var labels = serie.map(function(s){{return s.dia;}});
    var data = serie.map(function(s){{return s.rentabilidade_acumulada_pct;}});
    var lastVal = data[data.length-1]||0;
    var info = SUB_PORTFOLIOS.find(function(p){{return p.key === PORTFOLIO_KEY;}});
    var portfolioLabel = info ? info.nome : PORTFOLIO_KEY;
    datasets.push({{
        label: portfolioLabel,
        data: data,
        borderColor: lastVal>=0 ? '#4ecca3' : '#e74c3c',
        backgroundColor: gradient,
        fill: true, tension: 0.3, pointRadius: 0,
        pointHoverRadius: 5, borderWidth: 2,
        hidden: !document.getElementById('togglePortfolio').checked
    }});
    if (btc && btc.length > 0) {{
        var btcMap = {{}};
        btc.forEach(function(s){{btcMap[s.dia]=s.rentabilidade_acumulada_pct;}});
        var btcData = labels.map(function(d){{return btcMap[d] !== undefined ? btcMap[d] : null;}});
        datasets.push({{
            label: 'Bitcoin (BTC)',
            data: btcData,
            borderColor: '#f7931a',
            backgroundColor: 'transparent',
            fill: false, tension: 0.3, pointRadius: 0,
            pointHoverRadius: 5, borderWidth: 2,
            borderDash: [6, 3], spanGaps: true,
            hidden: !document.getElementById('toggleBTC').checked
        }});
    }}
    var cmpKeys = compare && typeof compare === 'object' && !Array.isArray(compare) ? Object.keys(compare) : [];
    cmpKeys.forEach(function(cmpKey, idx) {{
        var cmp = compare[cmpKey];
        if (!cmp || !cmp.length) return;
        var cmpMap = {{}};
        cmp.forEach(function(s){{cmpMap[s.dia]=s.rentabilidade_acumulada_pct;}});
        var cmpData = labels.map(function(d){{return cmpMap[d] !== undefined ? cmpMap[d] : null;}});
        var cmpInfo = SUB_PORTFOLIOS.find(function(p){{return p.key === cmpKey;}});
        var cmpLabel = cmpInfo ? cmpInfo.nome : cmpKey;
        var color = COMPARE_COLORS[idx % COMPARE_COLORS.length];
        datasets.push({{
            label: cmpLabel,
            data: cmpData,
            borderColor: color,
            backgroundColor: 'transparent',
            fill: false, tension: 0.3, pointRadius: 0,
            pointHoverRadius: 5, borderWidth: 2,
            borderDash: [3, 3], spanGaps: true
        }});
    }});
    return {{labels: labels, datasets: datasets}};
}}

function renderRentChart(serie) {{
    fullSerie = serie;
    currentSerie = serie;
    document.getElementById('chartLoading').style.display='none';
    document.getElementById('chartWrapper').style.display='block';
    if (activePeriod !== 'MAX') {{
        applyPeriodFilter();
    }} else {{
        renderRentChartFiltered(serie, btcSerie, compareSeries);
    }}
}}

function renderRentChartFiltered(serie, btc, compareSeriesObj) {{
    var built = buildRentDatasets(serie, btc, compareSeriesObj || {{}});
    if (!built || !built.labels || !built.labels.length) return;
    if(rentChart) rentChart.destroy();
    rentChart = new Chart(document.getElementById('chartRent').getContext('2d'), {{
        type:'line',
        data: built,
        options:{{
            responsive:true, maintainAspectRatio:false,
            interaction:{{mode:'index', intersect:false}},
            plugins:{{
                legend:{{display:false}},
                tooltip:{{
                    backgroundColor:'rgba(26,26,46,0.95)', titleColor:'#4ecca3',
                    bodyColor:'#e0e0e0', borderColor:'#4ecca3', borderWidth:1,
                    callbacks:{{label:function(c){{
                        if(c.raw==null)return null;
                        return c.dataset.label+': '+(c.raw>=0?'+':'')+c.raw.toFixed(2)+'%';
                    }}}}
                }}
            }},
            scales:{{
                x:{{ticks:{{color:'#666',maxTicksLimit:12,maxRotation:0}},grid:{{color:'rgba(255,255,255,0.05)'}}}},
                y:{{ticks:{{color:'#666',callback:function(v){{return v+'%';}}}},grid:{{color:'rgba(255,255,255,0.05)'}},
                   title:{{display:true,text:'Rentabilidade Acumulada (%)',color:'#888'}}}}
            }}
        }}
    }});
}}

function rebuildChart() {{
    if(!fullSerie) return;
    applyPeriodFilter();
}}

function fetchBtcBenchmark(dataInicio) {{
    var coingeckoOnly = (PORTFOLIO_KEY === 'EXC' || PORTFOLIO_KEY === 'HB' || PORTFOLIO_KEY === 'LC' || PORTFOLIO_KEY === 'AC') ? '&coingecko_only=1' : '';
    fetch('/api/benchmark/btc?data_inicio='+dataInicio+coingeckoOnly)
        .then(function(r){{return r.json();}})
        .then(function(data){{
            if(Array.isArray(data)) btcSerie = data;
            else btcSerie = null;
            applyPeriodFilter();
        }})
        .catch(function(){{ btcSerie = null; }});
}}

function populateCompareCheckboxes() {{
    var container = document.getElementById('compareCheckboxes');
    if (!container) return;
    container.innerHTML = '';
    SUB_PORTFOLIOS.forEach(function(sp) {{
        if (sp.key === PORTFOLIO_KEY) return;
        var label = document.createElement('label');
        label.style.cssText = 'display:inline-flex; align-items:center; gap:5px; cursor:pointer; color:#e0e0e0; font-size:13px;';
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.className = 'compare-cb';
        cb.dataset.key = sp.key;
        cb.style.cssText = 'accent-color:#e056fd; width:15px; height:15px; cursor:pointer;';
        if (compareSeries[sp.key]) cb.checked = true;
        cb.addEventListener('change', function() {{ onCompareCheckboxChange(sp.key, cb.checked); }});
        label.appendChild(cb);
        label.appendChild(document.createTextNode(sp.nome));
        container.appendChild(label);
    }});
}}

function onCompareCheckboxChange(key, checked) {{
    if (checked) {{
        if (activePeriod === 'MAX' && !periodStartDate && !periodEndDate) {{
            fetchCompareSerie(key, function(serie) {{
                if (serie) {{ compareSeries[key] = serie; rebuildChart(); }}
            }});
        }} else {{
            compareSeries[key] = [];
            rebuildChart();
        }}
    }} else {{
        delete compareSeries[key];
        rebuildChart();
    }}
}}

function fetchCompareSerie(key, done) {{
    fetch('/api/portfolio/' + key + '/dashboard-data')
        .then(function(r){{ return r.json(); }})
        .then(function(data){{
            if (data.rentabilidade_serie && data.rentabilidade_serie.length) done(data.rentabilidade_serie);
            else done(null);
        }})
        .catch(function(){{ done(null); }});
}}

function switchPosTab(tab, btn) {{
    document.querySelectorAll('#sectionPositions .sub-tab').forEach(function(b){{b.classList.remove('active');}});
    document.querySelectorAll('#sectionPositions .sub-content').forEach(function(c){{c.classList.remove('active');}});
    btn.classList.add('active');
    document.getElementById('tab-'+tab).classList.add('active');
}}

async function switchPortfolio(key, btn) {{
    if(key === PORTFOLIO_KEY) return;
    PORTFOLIO_KEY = key;
    compareSeries = {{}};
    document.querySelectorAll('.turma-tab').forEach(function(t){{t.classList.remove('active');}});
    btn.classList.add('active');
    document.getElementById('chartLoading').style.display='block';
    document.getElementById('chartLoading').innerHTML='<div class="spinner"></div>Carregando...';
    document.getElementById('chartWrapper').style.display='none';
    var info = SUB_PORTFOLIOS.find(function(p){{return p.key === key;}});
    if(info) {{
        DATA_INICIO = info.data_inicio || '';
        document.getElementById('togglePortfolioLabel').textContent = info.nome;
    }}
    populateCompareCheckboxes();
    try {{
        var resp = await fetch('/api/portfolio/'+key+'/dashboard-data');
        var dashData = await resp.json();
        if(dashData.erro) {{ console.error(dashData.erro); return; }}
        RESUMO = dashData.resumo || {{}};
        ABERTAS = dashData.posicoes_abertas || [];
        FECHADAS = dashData.posicoes_fechadas || [];
        ALLOC_HIST = dashData.alocacao_historica || [];
        var serie = dashData.rentabilidade_serie || [];

        updateCards({{
            rentabilidade_acumulada_pct: RESUMO.rentabilidade_acumulada_pct || 0,
            valor_total: RESUMO.valor_total || 0,
            capital_alocado: RESUMO.capital_alocado || 0,
            capital_em_caixa: RESUMO.capital_em_caixa || 0,
            capital_base: RESUMO.capital_base || 0,
            n_abertas: ABERTAS.length,
            n_fechadas: FECHADAS.length
        }});
        applyPnlFilter();
        renderAbertas(ABERTAS);
        renderFechadas(FECHADAS);
        renderHistorico(ABERTAS, FECHADAS);
        renderAllocTimeline(ALLOC_HIST);
        if(serie.length > 0) {{
            renderRentChart(serie);
            if(DATA_INICIO) fetchBtcBenchmark(DATA_INICIO);
        }} else {{
            document.getElementById('chartLoading').style.display='none';
            document.getElementById('chartWrapper').style.display='block';
            document.getElementById('chartWrapper').innerHTML='<div style="padding:40px; text-align:center; color:#666;">Sem dados de rentabilidade</div>';
        }}
    }} catch(err) {{
        console.error('Erro:', err);
        document.getElementById('chartLoading').style.display='none';
        document.getElementById('chartWrapper').style.display='block';
        document.getElementById('chartWrapper').innerHTML='<div style="padding:40px; text-align:center; color:#e74c3c;">Erro: '+err.message+'</div>';
    }}
}}

async function atualizarCotacoes() {{
    var status = document.getElementById('statusMsg');
    status.textContent='Buscando pre\\u00e7os...'; status.style.color='#f39c12';
    try {{
        var response = await fetch('/api/cotacoes/atualizar');
        var data = await response.json();
        if(data.sucesso) {{
            status.textContent='\\u2713 '+data.dias_preenchidos+' dias preenchidos';
            status.style.color='#4ecca3';
            setTimeout(function(){{window.location.reload();}}, 1500);
        }} else {{
            status.textContent='\\u2717 '+data.erro; status.style.color='#e74c3c';
        }}
    }} catch(err) {{
        status.textContent='\\u2717 '+err.message; status.style.color='#e74c3c';
    }}
}}

document.addEventListener('DOMContentLoaded', function() {{
    applyPnlFilter();
    renderAbertas(ABERTAS);
    renderFechadas(FECHADAS);
    renderHistorico(ABERTAS, FECHADAS);
    renderAllocTimeline(ALLOC_HIST);
    populateCompareCheckboxes();

    document.getElementById('togglePortfolio').addEventListener('change', function(){{ rebuildChart(); }});
    document.getElementById('toggleBTC').addEventListener('change', function(){{ rebuildChart(); }});

    var loadEl = document.getElementById('chartLoading');
    var wrapEl = document.getElementById('chartWrapper');
    if(SERIE && SERIE.length > 0) {{
        if (typeof Chart === 'undefined') {{
            loadEl.style.display = 'none';
            wrapEl.style.display = 'block';
            wrapEl.innerHTML = '<div style="padding:40px; text-align:center; color:#e74c3c;">Erro: biblioteca de gr&aacute;ficos n&atilde;o carregou. Recarregue a p&aacute;gina.</div>';
        }} else {{
            renderRentChart(SERIE);
            if(DATA_INICIO) fetchBtcBenchmark(DATA_INICIO);
        }}
    }} else {{
        loadEl.style.display = 'none';
        wrapEl.style.display = 'block';
        wrapEl.innerHTML = '<div style="padding:40px; text-align:center; color:#666;">Sem dados de rentabilidade</div>';
    }}
}});
</script>
</body>
</html>"""
