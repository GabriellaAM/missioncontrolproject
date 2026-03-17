"""
Dashboard HTML para o produto Crypto Signals.

Reutiliza o layout dark-theme do portfolio_dashboard, mas com colunas
específicas para Crypto Signals e sem seção de rentabilidade acumulada.
"""

import json
from datetime import datetime
from decimal import Decimal


def _get_shared_components():
    """Lazy import to avoid circular dependency with servidor_dashboard"""
    from servidor_dashboard import get_navbar, get_base_styles, _get_form_modal_overlay_script
    return get_navbar, get_base_styles, _get_form_modal_overlay_script


def _json_serializer(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)


def get_cryptosignals_dashboard_html(produto_nome, carteira, produto_id=0,
                                      product_tabs=None):
    """Dashboard para Crypto Signals com colunas específicas.

    Parameters
    ----------
    produto_nome : str
        Nome do produto (ex: "Crypto Signals")
    carteira : list[dict]
        Lista de trades enriquecidos (com campos CS: perfil, alvo1, alvo2, rr).
    produto_id : int
        ID do produto.
    product_tabs : dict | None
        Abas de produto agrupado (se aplicável).
    """
    get_navbar, get_base_styles, _get_form_modal_overlay_script = _get_shared_components()
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    carteira_json = json.dumps(carteira, default=_json_serializer,
                               ensure_ascii=False)

    n_abertas = sum(1 for t in carteira if _is_open(t))
    n_fechadas = len(carteira) - n_abertas

    tab_html = ""
    if product_tabs:
        primary_id = product_tabs['primary_id']
        for i, ptab in enumerate(product_tabs['tabs']):
            active_cls = 'active' if ptab['active'] else ''
            href = f"/produto/{primary_id}?tab={i}"
            tab_html += f'<a class="turma-tab {active_cls}" href="{href}">{ptab["nome"]}</a>'

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
            border-bottom: 2px solid #39fda3;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .dash-header h1 {{ color: #39fda3; font-size: 1.8em; letter-spacing: 1px; margin: 0; }}
        .dash-header .tipo-badge {{
            padding: 4px 14px; border-radius: 20px; font-size: 0.8em; font-weight: bold;
        }}
        .tipo-perp {{ background: #ff6b6b; color: #fff; }}
        .dash-header .date-badge {{
            margin-left: auto;
            background: rgba(78, 204, 163, 0.15);
            color: #39fda3;
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
        .turma-tab.active {{ color: #39fda3; border-bottom-color: #39fda3; font-weight: 600; }}
        a.turma-tab {{ text-decoration: none; }}

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
        .summary-card .s-value.positive {{ color: #39fda3; }}
        .summary-card .s-value.negative {{ color: #e74c3c; }}
        .summary-card .s-label {{ color: #888; font-size: 0.85em; }}

        .dash-section {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px; padding: 25px; margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.05);
        }}
        .dash-section h3 {{ color: #39fda3; margin-bottom: 18px; font-size: 1.2em; }}

        .chart-box {{ position: relative; height: 350px; width: 100%; }}

        .sub-tabs {{ display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; }}
        .sub-tab {{
            padding: 8px 18px; background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1); border-radius: 6px;
            color: #888; cursor: pointer; transition: all 0.3s; font-size: 0.9em;
            font-family: inherit;
        }}
        .sub-tab:hover {{ border-color: #39fda3; color: #39fda3; }}
        .sub-tab.active {{ background: rgba(78, 204, 163, 0.2); border-color: #39fda3; color: #39fda3; }}
        .sub-content {{ display: none; }}
        .sub-content.active {{ display: block; }}

        .dtable {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
        .dtable th, .dtable td {{
            padding: 10px 12px; text-align: right;
            border-bottom: 1px solid rgba(255,255,255,0.07); white-space: nowrap;
        }}
        .dtable th {{
            background: #0d1025; color: #39fda3; font-weight: 600;
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
        .bdg-long {{ background: rgba(78, 204, 163, 0.2); color: #39fda3; }}
        .bdg-short {{ background: rgba(231, 76, 60, 0.2); color: #e74c3c; }}
        .bdg-open {{ background: rgba(52, 152, 219, 0.2); color: #3498db; }}
        .bdg-closed {{ background: rgba(149, 165, 166, 0.2); color: #95a5a6; }}
        .bdg-stop {{ background: rgba(231, 76, 60, 0.15); color: #e74c3c; border: 1px solid rgba(231, 76, 60, 0.3); }}
        .pnl-pos {{ color: #39fda3; font-weight: 600; }}
        .pnl-neg {{ color: #e74c3c; font-weight: 600; }}

        .gear-menu-wrapper {{ position: relative; display: inline-flex; align-items: center; }}
        .gear-btn {{
            background: none; border: none; cursor: pointer; padding: 6px;
            border-radius: 8px; transition: background 0.2s, transform 0.3s; display:flex; align-items:center;
        }}
        .gear-btn:hover {{ background: rgba(78,204,163,0.12); }}
        .gear-btn.open {{ transform: rotate(90deg); }}
        .gear-btn svg {{ width: 22px; height: 22px; fill: #888; transition: fill 0.2s; }}
        .gear-btn:hover svg, .gear-btn.open svg {{ fill: #39fda3; }}
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
        .gear-panel a:hover, .gear-panel button.gear-item:hover {{ background: rgba(78,204,163,0.1); color: #39fda3; }}
        .gear-panel .gear-icon {{ width: 16px; text-align: center; font-size: 1em; flex-shrink: 0; }}
        .gear-panel .gear-accent {{ color: #39fda3; }}

        .loading-overlay {{ text-align: center; padding: 60px 20px; color: #888; }}
        .loading-overlay .spinner {{
            border: 3px solid #2a2a4a; border-top: 3px solid #39fda3;
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
        <span class="tipo-badge tipo-perp">Perpétuos</span>
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
                        <div class="gear-panel-title">Posi&ccedil;&otilde;es</div>
                        <a href="/posicao/nova?produto_id={produto_id}"><span class="gear-icon gear-accent">+</span> Nova Posi&ccedil;&atilde;o</a>
                        <a href="/posicoes/editar?produto_id={produto_id}"><span class="gear-icon">&#9998;</span> Editar Posi&ccedil;&atilde;o</a>
                        <a href="/posicao/fechar?produto_id={produto_id}"><span class="gear-icon">&#10006;</span> Fechar Posi&ccedil;&atilde;o</a>
                    </div>
                    <div class="gear-panel-group">
                        <div class="gear-panel-title">Exportar</div>
                        <a href="#" onclick="baixarExcelComToast('/api/export/excel?produto_id={produto_id}', 'rentabilidade_posicoes.xlsx'); toggleGearMenu(); return false;"><span class="gear-icon">&#8595;</span> Baixar Excel (Rentab. + Posi&ccedil;&otilde;es)</a>
                    </div>
                    <div class="gear-panel-group">
                        <div class="gear-panel-title">Produto</div>
                        <a href="/produto/{produto_id}/editar"><span class="gear-icon">&#9998;</span> Editar Produto</a>
                        <a href="/produto/{produto_id}/atributos"><span class="gear-icon">&#9776;</span> Atributos</a>
                        <a href="/atr/config?produto_id={produto_id}"><span class="gear-icon">&#9632;</span> ATR Stop</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    {f'<div class="turma-tabs-bar">{tab_html}</div>' if tab_html else ''}

    <div class="dash-content">
        <div class="summary-row" id="summaryCards">
            <div class="summary-card">
                <div class="s-value" id="card-trades">{n_abertas} / {n_fechadas}</div>
                <div class="s-label">Ativos / Fechados</div>
            </div>
        </div>

        <div class="dash-section">
            <h3>PnL dos ativos abertos</h3>
            <div class="chart-box" id="pnlChartWrapper" style="height:280px;">
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
                    <thead><tr>
                        <th>Ativo</th><th>Tipo</th><th>Perfil</th><th>Dt. Abertura</th><th>Dura&ccedil;&atilde;o</th>
                        <th>P. Entrada</th><th>P. Atual</th><th>Alvo 1</th><th>Alvo 2</th>
                        <th>Stop</th><th>RR</th><th>PnL%</th>
                    </tr></thead>
                    <tbody id="tbAbertas"></tbody>
                </table></div>
            </div>
            <div id="tab-fechadas" class="sub-content">
                <div class="tbl-scroll"><table class="dtable" id="tblFechadas">
                    <thead><tr>
                        <th>Ativo</th><th>Tipo</th><th>Perfil</th><th>Dt. Abertura</th><th>Dt. Encerram.</th>
                        <th>Dura&ccedil;&atilde;o</th><th>P. Entrada</th><th>P. Sa&iacute;da</th><th>Stop</th><th>PnL%</th>
                    </tr></thead>
                    <tbody id="tbFechadas"></tbody>
                </table></div>
            </div>
            <div id="tab-historico" class="sub-content">
                <div class="tbl-scroll"><table class="dtable" id="tblHistorico">
                    <thead><tr>
                        <th>Ativo</th><th>Status</th><th>Tipo</th><th>Perfil</th>
                        <th>Dt. Abertura</th><th>Dt. Sa&iacute;da</th><th>Dura&ccedil;&atilde;o</th>
                        <th>P. Entrada</th><th>P. Sa&iacute;da/Atual</th>
                        <th>Alvo 1</th><th>Alvo 2</th><th>Stop</th><th>PnL%</th>
                    </tr></thead>
                    <tbody id="tbHistorico"></tbody>
                </table></div>
            </div>
        </div>
    </div>

<script>
var PRODUTO_ID = {produto_id};
var CARTEIRA = {carteira_json};

function toggleGearMenu() {{
    document.getElementById('gearBtn').classList.toggle('open');
    document.getElementById('gearPanel').classList.toggle('open');
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
    return s.toLowerCase()==='long'
        ? '<span class="bdg bdg-long">LONG</span>'
        : '<span class="bdg bdg-short">SHORT</span>';
}}
function statusBdg(isOpen) {{
    return isOpen
        ? '<span class="bdg bdg-open">Aberto</span>'
        : '<span class="bdg bdg-closed">Fechado</span>';
}}
function fmtDate(d) {{ return d || '\\u2014'; }}
function fmtRR(v) {{ return v != null ? v.toFixed(2) : '\\u2014'; }}
function fmtStop(v) {{
    if (v == null || v === undefined) return '\\u2014';
    if (v === -1) return '<span class="bdg bdg-stop">Stop Atingido</span>';
    return '$ ' + fmtPrice(v);
}}

function isTradeOpen(t) {{
    if (t.ativo_atual === 0 || t.ativo_atual === false) return false;
    if (t.data_remocao || t.data_saida) return false;
    if (String(t.status_posicao || '').toLowerCase() === 'closed') return false;
    return true;
}}

function rowOpen(t) {{
    return '<tr>'
        + '<td>' + t.ativo + '</td>'
        + '<td>' + sideBdg(t.side) + '</td>'
        + '<td>' + (t.perfil || '\\u2014') + '</td>'
        + '<td>' + fmtDate(t.data_insercao) + '</td>'
        + '<td>' + (t.dias != null ? t.dias + ' d' : '\\u2014') + '</td>'
        + '<td>$ ' + fmtPrice(t.preco_entrada_turma) + '</td>'
        + '<td>$ ' + fmtPrice(t.preco_atual) + '</td>'
        + '<td>' + (t.alvo1 != null ? '$ ' + fmtPrice(t.alvo1) : '\\u2014') + '</td>'
        + '<td>' + (t.alvo2 != null ? '$ ' + fmtPrice(t.alvo2) : '\\u2014') + '</td>'
        + '<td>' + fmtStop(t.stop_atual) + '</td>'
        + '<td>' + fmtRR(t.rr) + '</td>'
        + '<td>' + fmtPnl(t.pnl_pct) + '</td>'
        + '</tr>';
}}

function rowClosed(t) {{
    return '<tr>'
        + '<td>' + t.ativo + '</td>'
        + '<td>' + sideBdg(t.side) + '</td>'
        + '<td>' + (t.perfil || '\\u2014') + '</td>'
        + '<td>' + fmtDate(t.data_insercao) + '</td>'
        + '<td>' + fmtDate(t.data_remocao || t.data_saida) + '</td>'
        + '<td>' + (t.dias != null ? t.dias + ' d' : '\\u2014') + '</td>'
        + '<td>$ ' + fmtPrice(t.preco_entrada_turma) + '</td>'
        + '<td>$ ' + fmtPrice(t.preco_atual) + '</td>'
        + '<td>' + fmtStop(t.stop_atual) + '</td>'
        + '<td>' + fmtPnl(t.pnl_pct) + '</td>'
        + '</tr>';
}}

function rowHist(t) {{
    var isOpen = isTradeOpen(t);
    return '<tr>'
        + '<td>' + t.ativo + '</td>'
        + '<td>' + statusBdg(isOpen) + '</td>'
        + '<td>' + sideBdg(t.side) + '</td>'
        + '<td>' + (t.perfil || '\\u2014') + '</td>'
        + '<td>' + fmtDate(t.data_insercao) + '</td>'
        + '<td>' + fmtDate(t.data_remocao || t.data_saida) + '</td>'
        + '<td>' + (t.dias != null ? t.dias + ' d' : '\\u2014') + '</td>'
        + '<td>$ ' + fmtPrice(t.preco_entrada_turma) + '</td>'
        + '<td>$ ' + fmtPrice(t.preco_atual) + '</td>'
        + '<td>' + (t.alvo1 != null ? '$ ' + fmtPrice(t.alvo1) : '\\u2014') + '</td>'
        + '<td>' + (t.alvo2 != null ? '$ ' + fmtPrice(t.alvo2) : '\\u2014') + '</td>'
        + '<td>' + fmtStop(t.stop_atual) + '</td>'
        + '<td>' + fmtPnl(t.pnl_pct) + '</td>'
        + '</tr>';
}}

var sortSt = {{}};
function setupSort(tbl, data, cols, renderRow) {{
    var key = tbl.id;
    var ths = tbl.querySelectorAll('thead th');
    ths.forEach(function(th, idx) {{
        if (!th.querySelector('.sort-arrow'))
            th.innerHTML = th.textContent + ' <span class="sort-arrow">\\u25B2</span>';
        th.onclick = function() {{
            var prev = sortSt[key]; var dir = 'asc';
            if (prev && prev.col === idx) dir = prev.dir === 'asc' ? 'desc' : 'asc';
            sortSt[key] = {{ col: idx, dir: dir }};
            ths.forEach(function(h) {{
                h.classList.remove('sorted');
                h.querySelector('.sort-arrow').textContent = '\\u25B2';
            }});
            th.classList.add('sorted');
            th.querySelector('.sort-arrow').textContent = dir === 'asc' ? '\\u25B2' : '\\u25BC';
            var c = cols[idx];
            var sorted = [...data].sort(function(a, b) {{
                var va = c.v(a), vb = c.v(b);
                if (c.t === 'text') {{ va = (va || '').toLowerCase(); vb = (vb || '').toLowerCase(); }}
                else if (c.t === 'date') {{ va = va || '0'; vb = vb || '0'; }}
                else {{ va = typeof va === 'number' && !isNaN(va) ? va : -Infinity; vb = typeof vb === 'number' && !isNaN(vb) ? vb : -Infinity; }}
                var cmp = va < vb ? -1 : va > vb ? 1 : 0;
                return dir === 'asc' ? cmp : -cmp;
            }});
            tbl.querySelector('tbody').innerHTML = sorted.map(renderRow).join('');
        }};
    }});
}}

function renderAbertas(cart) {{
    var open = cart.filter(isTradeOpen).sort(function(a, b) {{
        return (b.data_insercao || '').localeCompare(a.data_insercao || '');
    }});
    var tb = document.getElementById('tbAbertas');
    document.getElementById('countAbertas').textContent = open.length;
    if (!open.length) {{
        tb.innerHTML = '<tr><td colspan="12" class="empty-msg">Nenhuma posi\\u00e7\\u00e3o aberta</td></tr>';
        return;
    }}
    tb.innerHTML = open.map(rowOpen).join('');
    setupSort(document.getElementById('tblAbertas'), open, [
        {{t:'text',v:function(r){{return r.ativo;}}}},
        {{t:'text',v:function(r){{return r.side;}}}},
        {{t:'text',v:function(r){{return r.perfil;}}}},
        {{t:'date',v:function(r){{return r.data_insercao;}}}},
        {{t:'num',v:function(r){{return r.dias;}}}},
        {{t:'num',v:function(r){{return r.preco_entrada_turma;}}}},
        {{t:'num',v:function(r){{return r.preco_atual;}}}},
        {{t:'num',v:function(r){{return r.alvo1;}}}},
        {{t:'num',v:function(r){{return r.alvo2;}}}},
        {{t:'num',v:function(r){{return r.stop_atual;}}}},
        {{t:'num',v:function(r){{return r.rr;}}}},
        {{t:'num',v:function(r){{return r.pnl_pct;}}}}
    ], rowOpen);
}}

function renderFechadas(cart) {{
    var closed = cart.filter(function(t) {{ return !isTradeOpen(t); }}).sort(function(a, b) {{
        return (b.data_insercao || '').localeCompare(a.data_insercao || '');
    }});
    var tb = document.getElementById('tbFechadas');
    document.getElementById('countFechadas').textContent = closed.length;
    document.getElementById('countHistorico').textContent = cart.length;
    if (!closed.length) {{
        tb.innerHTML = '<tr><td colspan="10" class="empty-msg">Nenhum trade fechado</td></tr>';
        return;
    }}
    tb.innerHTML = closed.map(rowClosed).join('');
    setupSort(document.getElementById('tblFechadas'), closed, [
        {{t:'text',v:function(r){{return r.ativo;}}}},
        {{t:'text',v:function(r){{return r.side;}}}},
        {{t:'text',v:function(r){{return r.perfil;}}}},
        {{t:'date',v:function(r){{return r.data_insercao;}}}},
        {{t:'date',v:function(r){{return r.data_remocao || r.data_saida;}}}},
        {{t:'num',v:function(r){{return r.dias;}}}},
        {{t:'num',v:function(r){{return r.preco_entrada_turma;}}}},
        {{t:'num',v:function(r){{return r.preco_atual;}}}},
        {{t:'num',v:function(r){{return r.stop_atual;}}}},
        {{t:'num',v:function(r){{return r.pnl_pct;}}}}
    ], rowClosed);
}}

function renderHistorico(cart) {{
    var all = [...cart].sort(function(a, b) {{
        return (b.data_insercao || '').localeCompare(a.data_insercao || '');
    }});
    var tb = document.getElementById('tbHistorico');
    document.getElementById('countHistorico').textContent = all.length;
    if (!all.length) {{
        tb.innerHTML = '<tr><td colspan="13" class="empty-msg">Nenhum trade</td></tr>';
        return;
    }}
    tb.innerHTML = all.map(rowHist).join('');
    setupSort(document.getElementById('tblHistorico'), all, [
        {{t:'text',v:function(r){{return r.ativo;}}}},
        {{t:'text',v:function(r){{return isTradeOpen(r)?'a':'z';}}}},
        {{t:'text',v:function(r){{return r.side;}}}},
        {{t:'text',v:function(r){{return r.perfil;}}}},
        {{t:'date',v:function(r){{return r.data_insercao;}}}},
        {{t:'date',v:function(r){{return r.data_remocao || r.data_saida;}}}},
        {{t:'num',v:function(r){{return r.dias;}}}},
        {{t:'num',v:function(r){{return r.preco_entrada_turma;}}}},
        {{t:'num',v:function(r){{return r.preco_atual;}}}},
        {{t:'num',v:function(r){{return r.alvo1;}}}},
        {{t:'num',v:function(r){{return r.alvo2;}}}},
        {{t:'num',v:function(r){{return r.stop_atual;}}}},
        {{t:'num',v:function(r){{return r.pnl_pct;}}}}
    ], rowHist);
}}

function renderPnlChart(cart) {{
    var open = cart.filter(isTradeOpen);
    var ctx = document.getElementById('chartPnlAbertas');
    if (!ctx) return;
    ctx = ctx.getContext('2d');
    if (!open.length) {{
        new Chart(ctx, {{
            type:'bar', data:{{labels:[],datasets:[]}},
            options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{display:false}},y:{{display:false}}}}}}
        }});
        return;
    }}
    var labels = open.map(function(t) {{ return t.ativo + ' (' + (t.side||'LONG').toUpperCase() + ')'; }});
    var values = open.map(function(t) {{ return t.pnl_pct != null ? t.pnl_pct : 0; }});
    var colors = values.map(function(v) {{ return v >= 0 ? '#39fda3' : '#e74c3c'; }});
    new Chart(ctx, {{
        type: 'bar',
        data: {{
            labels: labels,
            datasets: [{{ label: 'PnL %', data: values, backgroundColor: colors, borderColor: colors, borderWidth: 1, borderRadius: 4 }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{
                legend: {{ display: false }},
                tooltip: {{
                    backgroundColor: 'rgba(26,26,46,0.95)', titleColor: '#39fda3', bodyColor: '#e0e0e0',
                    callbacks: {{ label: function(c) {{ return (c.raw >= 0 ? '+' : '') + c.raw.toFixed(2) + '%'; }} }}
                }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#e0e0e0', font: {{ size: 11 }}, maxRotation: 45 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                y: {{ ticks: {{ color: '#666', callback: function(v) {{ return (typeof v === 'number' ? v.toFixed(2) : v) + '%'; }} }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, title: {{ display: true, text: 'PnL %', color: '#888' }} }}
            }}
        }}
    }});
}}

function switchPosTab(tab, btn) {{
    document.querySelectorAll('#sectionPositions .sub-tab').forEach(function(b) {{ b.classList.remove('active'); }});
    document.querySelectorAll('#sectionPositions .sub-content').forEach(function(c) {{ c.classList.remove('active'); }});
    btn.classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
}}

async function atualizarCotacoes() {{
    var status = document.getElementById('statusMsg');
    status.textContent = 'Buscando pre\\u00e7os...'; status.style.color = '#f39c12';
    try {{
        var response = await fetch('/api/cotacoes/atualizar');
        var data = await response.json();
        if (data.sucesso) {{
            status.textContent = '\\u2713 ' + data.dias_preenchidos + ' dias preenchidos';
            status.style.color = '#39fda3';
            setTimeout(function() {{ window.location.reload(); }}, 1500);
        }} else {{
            status.textContent = '\\u2717 ' + data.erro; status.style.color = '#e74c3c';
        }}
    }} catch(err) {{
        status.textContent = '\\u2717 ' + err.message; status.style.color = '#e74c3c';
    }}
}}

document.addEventListener('DOMContentLoaded', function() {{
    renderAbertas(CARTEIRA);
    renderFechadas(CARTEIRA);
    renderHistorico(CARTEIRA);
    renderPnlChart(CARTEIRA);
}});
</script>
{_get_form_modal_overlay_script(produto_id)}
</body>
</html>"""


def _is_open(t):
    """Check if a trade/carteira item is open (Python-side)."""
    if t.get('ativo_atual') in (0, False):
        return False
    if t.get('data_remocao') or t.get('data_saida'):
        return False
    status = str(t.get('status_posicao', '')).strip().lower()
    if status == 'closed':
        return False
    return True
