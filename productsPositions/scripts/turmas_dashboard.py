"""
Templates e funções para as telas de Turmas no Dashboard.

Este módulo contém os templates HTML e funções de renderização
para as páginas de gestão de turmas e rentabilidade.
"""

import json
from datetime import datetime
from decimal import Decimal


def _get_shared_components():
    """Lazy import to avoid circular dependency with servidor_dashboard"""
    from servidor_dashboard import get_navbar, get_turmas_subnav, get_base_styles, _get_form_modal_overlay_script
    return get_navbar, get_turmas_subnav, get_base_styles, _get_form_modal_overlay_script


def get_turmas_styles():
    """Estilos CSS específicos para páginas de turmas"""
    return """
        .turmas-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            padding: 20px;
        }
        .turma-card {
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .turma-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(78, 204, 163, 0.2);
        }
        .turma-card h3 {
            color: #39fda3;
            margin-bottom: 10px;
            font-size: 1.3em;
        }
        .turma-meta {
            color: #888;
            font-size: 0.9em;
            margin-bottom: 15px;
        }
        .turma-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .stat-box {
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.4em;
            font-weight: bold;
            color: #fff;
        }
        .stat-value.positive { color: #39fda3; }
        .stat-value.negative { color: #e74c3c; }
        .stat-label {
            font-size: 0.8em;
            color: #888;
            margin-top: 4px;
        }
        .turma-actions {
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }
        .turma-actions a, .turma-actions button {
            flex: 1;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            text-decoration: none;
            font-size: 0.9em;
            cursor: pointer;
            border: none;
            transition: all 0.3s;
        }
        .btn-primary {
            background: #39fda3;
            color: #1a1a2e;
        }
        .btn-primary:hover { background: #3db892; }
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        .btn-secondary:hover { background: rgba(255,255,255,0.2); }

        /* Chart container */
        .chart-container {
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 20px;
            margin: 20px;
        }
        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .chart-title {
            color: #39fda3;
            font-size: 1.3em;
        }

        /* Compare checkboxes */
        .compare-selector {
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 20px;
            margin: 20px;
        }
        .compare-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .compare-item {
            display: flex;
            align-items: center;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 5px;
        }
        .compare-item input {
            margin-right: 10px;
            width: 18px;
            height: 18px;
        }

        /* Data table */
        .rentabilidade-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .rentabilidade-table th, .rentabilidade-table td {
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .rentabilidade-table th {
            background: rgba(0,0,0,0.3);
            color: #39fda3;
            font-weight: 600;
        }
        .rentabilidade-table tr:hover {
            background: rgba(78, 204, 163, 0.1);
        }
        .rentabilidade-table td:first-child {
            text-align: left;
        }
    """


def get_lista_turmas_html(turmas, resumos):
    """Gera HTML da lista de turmas com abas por produto"""
    get_navbar, get_turmas_subnav, get_base_styles, _get_form_modal_overlay_script = _get_shared_components()
    styles = get_turmas_styles()

    # Agrupar turmas por produto
    produtos = {}  # {produto_nome: [turmas]}
    for turma in turmas:
        produto_nome = turma.get('produto_nome', 'Sem Produto')
        if produto_nome not in produtos:
            produtos[produto_nome] = []
        produtos[produto_nome].append(turma)

    # Gerar abas e conteúdo
    tabs_html = ""
    panels_html = ""
    for i, (produto_nome, turmas_produto) in enumerate(produtos.items()):
        tab_id = f"tab-{i}"
        active_class = "active" if i == 0 else ""

        tabs_html += f"""
        <button class="product-tab {active_class}" data-tab="{tab_id}"
                onclick="switchTab('{tab_id}')">{produto_nome} ({len(turmas_produto)})</button>
        """

        cards_html = ""
        for turma in turmas_produto:
            turma_id = turma['id']
            resumo = resumos.get(turma_id, {})

            rentab = resumo.get('rentabilidade_acumulada_pct', 0)
            rentab_class = 'positive' if rentab >= 0 else 'negative'
            rentab_str = f"+{rentab:.2f}%" if rentab >= 0 else f"{rentab:.2f}%"

            valor_total = resumo.get('valor_total', 0)
            trades_ativos = resumo.get('trades_ativos', 0)
            trades_fechados = resumo.get('trades_fechados', 0)
            data_inicio = turma.get('data_inicio', '1970-01-01')

            cards_html += f"""
            <div class="turma-card" data-rentab="{rentab}" data-data="{data_inicio}">
                <h3>{turma.get('nome', 'N/A')}</h3>
                <div class="turma-meta">
                    <span>Início: {data_inicio}</span>
                </div>
                <div class="turma-stats">
                    <div class="stat-box">
                        <div class="stat-value {rentab_class}">{rentab_str}</div>
                        <div class="stat-label">Rentabilidade</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">R$ {valor_total:,.2f}</div>
                        <div class="stat-label">Valor Total</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{trades_ativos}</div>
                        <div class="stat-label">Trades Ativos</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{trades_fechados}</div>
                        <div class="stat-label">Trades Fechados</div>
                    </div>
                </div>
                <div class="turma-actions">
                    <a href="/turmas/{turma_id}" class="btn-primary">Ver Detalhes</a>
                    <a href="/turmas/{turma_id}/rentabilidade" class="btn-secondary">Gráfico</a>
                </div>
            </div>
            """

        display = "grid" if i == 0 else "none"
        panels_html += f"""
        <div class="tab-panel-wrapper" id="{tab_id}" style="display: {'block' if i == 0 else 'none'};">
            <div class="sort-bar">
                <span class="sort-label">Ordenar por:</span>
                <button class="sort-btn active" onclick="sortPanel('{tab_id}', 'data-desc', this)">Mais recente</button>
                <button class="sort-btn" onclick="sortPanel('{tab_id}', 'data-asc', this)">Mais antiga</button>
                <button class="sort-btn" onclick="sortPanel('{tab_id}', 'rentab-desc', this)">Maior rentab.</button>
                <button class="sort-btn" onclick="sortPanel('{tab_id}', 'rentab-asc', this)">Menor rentab.</button>
            </div>
            <div class="turmas-grid">
                {cards_html if cards_html else '<p style="padding: 20px; color: #888;">Nenhuma turma neste produto.</p>'}
            </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Turmas - Dashboard</title>
    <style>
        {get_base_styles()}
        .page-header {{
            padding: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .page-header h2 {{ color: #fff; font-size: 1.8em; }}
        .btn-criar {{
            background: #39fda3;
            color: #1a1a2e;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .btn-criar:hover {{ background: #3db892; }}
        .product-tabs {{
            display: flex;
            gap: 0;
            padding: 0 20px;
        }}
        .product-tab {{
            padding: 12px 28px;
            background: transparent;
            color: #888;
            border: none;
            border-bottom: 3px solid transparent;
            font-size: 1.05em;
            cursor: pointer;
            transition: all 0.3s;
            font-family: inherit;
        }}
        .product-tab:hover {{
            color: #ccc;
            background: rgba(78, 204, 163, 0.05);
        }}
        .product-tab.active {{
            color: #39fda3;
            border-bottom-color: #39fda3;
            font-weight: 600;
        }}
        .sort-bar {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 20px;
        }}
        .sort-label {{
            color: #888;
            font-size: 0.85em;
            margin-right: 4px;
        }}
        .sort-btn {{
            padding: 6px 14px;
            background: rgba(255,255,255,0.06);
            color: #aaa;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 5px;
            font-size: 0.82em;
            cursor: pointer;
            transition: all 0.2s;
            font-family: inherit;
        }}
        .sort-btn:hover {{
            background: rgba(78, 204, 163, 0.1);
            color: #ccc;
        }}
        .sort-btn.active {{
            background: rgba(78, 204, 163, 0.15);
            color: #39fda3;
            border-color: #39fda3;
        }}
        {styles}
    </style>
</head>
<body>
    {get_navbar('turmas')}
    {get_turmas_subnav('turmas')}

    <div class="page-header">
        <h2>Turmas ({len(turmas)})</h2>
    </div>

    <div class="product-tabs">
        {tabs_html if tabs_html else ''}
    </div>

    {panels_html if panels_html else '<div class="turmas-grid"><p style="padding: 20px; color: #888;">Nenhuma turma cadastrada.</p></div>'}

    <script>
    function switchTab(tabId) {{
        document.querySelectorAll('.tab-panel-wrapper').forEach(p => p.style.display = 'none');
        document.querySelectorAll('.product-tab').forEach(t => t.classList.remove('active'));
        document.getElementById(tabId).style.display = 'block';
        document.querySelector('[data-tab="' + tabId + '"]').classList.add('active');
    }}

    function sortPanel(tabId, mode, btn) {{
        var wrapper = document.getElementById(tabId);
        var grid = wrapper.querySelector('.turmas-grid');
        var cards = Array.from(grid.querySelectorAll('.turma-card'));

        wrapper.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        cards.sort(function(a, b) {{
            if (mode === 'rentab-desc') return parseFloat(b.dataset.rentab) - parseFloat(a.dataset.rentab);
            if (mode === 'rentab-asc') return parseFloat(a.dataset.rentab) - parseFloat(b.dataset.rentab);
            if (mode === 'data-desc') return b.dataset.data.localeCompare(a.dataset.data);
            if (mode === 'data-asc') return a.dataset.data.localeCompare(b.dataset.data);
            return 0;
        }});

        cards.forEach(function(card) {{ grid.appendChild(card); }});
    }}
    </script>
    {_get_form_modal_overlay_script(0)}
</body>
</html>"""


def get_form_nova_turma_html(produtos, as_inner=False, return_url='/'):
    """Gera HTML do formulário para criar nova turma. Se as_inner=True, retorna só o conteúdo para overlay modal. return_url: URL para onde o Cancelar leva."""
    get_navbar, get_turmas_subnav, get_base_styles, _ = _get_shared_components()

    options_html = ""
    for p in produtos:
        options_html += f'<option value="{p["id"]}">{p["nome"]}</option>'

    form_styles = """
        .form-nova-turma.form-container,
        .form-container.form-nova-turma {
            width: 100%;
            max-width: 600px;
            min-width: 0;
            margin: 0 auto;
            padding: 30px;
            box-sizing: border-box;
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
        }
        .form-nova-turma.form-container h2,
        .form-container.form-nova-turma h2 {
            color: #39fda3;
            margin-bottom: 25px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        /* Layout responsivo: 2-3 colunas conforme largura */
        .form-nova-turma.form-container form,
        .form-container.form-nova-turma form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px 20px;
            min-width: 0;
        }
        @media (min-width: 700px) {
            .form-nova-turma form,
            .form-container.form-nova-turma form { grid-template-columns: 1fr 1fr 1fr; gap: 12px 16px; }
        }
        .form-nova-turma form .form-group,
        .form-container.form-nova-turma form .form-group { margin-bottom: 0; }
        .form-nova-turma form .form-group.form-group-full,
        .form-container.form-nova-turma form .form-group.form-group-full { grid-column: 1 / -1; }
        .form-nova-turma form .btn-submit,
        .form-container.form-nova-turma form .btn-submit { grid-column: 1 / -1; }
        .form-nova-turma form .btn-cancel,
        .form-container.form-nova-turma form .btn-cancel { grid-column: 1 / -1; margin-top: 0; }
        .form-group label {
            display: block;
            color: #888;
            margin-bottom: 8px;
            font-size: 0.9em;
        }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 5px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 1em;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #39fda3;
        }
        .btn-submit {
            width: 100%;
            padding: 15px;
            background: #39fda3;
            color: #1a1a2e;
            border: none;
            border-radius: 5px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-submit:hover { background: #3db892; }
        .btn-cancel {
            display: block;
            text-align: center;
            margin-top: 15px;
            color: #888;
            text-decoration: none;
        }
        .btn-cancel:hover { color: #fff; }
        .alert {
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        .alert-error { background: rgba(231, 76, 60, 0.2); color: #e74c3c; }
        .alert-success { background: rgba(78, 204, 163, 0.2); color: #39fda3; }
        .posicoes-elegiveis {
            margin-top: 20px;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            border: 1px solid rgba(78, 204, 163, 0.3);
        }
        .posicoes-elegiveis h3 {
            color: #39fda3;
            font-size: 0.95em;
            margin-bottom: 8px;
        }
        .posicoes-elegiveis p.hint {
            color: #888;
            font-size: 0.85em;
            margin-bottom: 12px;
        }
        .posicoes-elegiveis table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        .posicoes-elegiveis th, .posicoes-elegiveis td {
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .posicoes-elegiveis th {
            color: #888;
            font-weight: 600;
        }
        .posicoes-elegiveis input[type="checkbox"] {
            accent-color: #39fda3;
            cursor: pointer;
        }
        .posicoes-elegiveis input[type="date"] {
            padding: 6px 8px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 4px;
            color: #fff;
            font-size: 0.85em;
            min-width: 130px;
        }
        .posicoes-elegiveis .btn-aplicar-data {
            margin-bottom: 10px;
            padding: 6px 12px;
            font-size: 0.85em;
            background: rgba(78, 204, 163, 0.25);
            color: #39fda3;
            border: 1px solid #39fda3;
            border-radius: 5px;
            cursor: pointer;
        }
        .posicoes-elegiveis .btn-aplicar-data:hover {
            background: rgba(78, 204, 163, 0.4);
        }
        .posicoes-elegiveis .loading {
            color: #888;
            padding: 10px 0;
        }
        .posicoes-elegiveis .empty {
            color: #888;
            padding: 10px 0;
        }
        .posicoes-elegiveis .select-first {
            color: #666;
            font-style: italic;
        }
        /* Responsivo: ajustes em telas menores */
        @media (max-width: 900px) {
            .form-nova-turma.form-container,
            .form-container.form-nova-turma {
                padding: 16px;
                margin: 0;
                max-width: 100%;
            }
            .form-nova-turma h2, .form-container.form-nova-turma h2 { font-size: 1.2em; margin-bottom: 16px; }
            .form-nova-turma form, .form-container.form-nova-turma form { gap: 12px 16px; }
            .form-group input, .form-group select, .form-group textarea { padding: 10px; font-size: 16px; }
            .btn-submit { padding: 12px; font-size: 1em; }
            .posicoes-elegiveis { padding: 12px; margin-top: 0; overflow-x: auto; -webkit-overflow-scrolling: touch; }
            .posicoes-elegiveis table { font-size: 0.8em; min-width: 480px; }
            .posicoes-elegiveis th, .posicoes-elegiveis td { padding: 6px 8px; }
            .posicoes-elegiveis input[type="date"] { min-width: 110px; }
        }
        /* Telas muito pequenas: 1 coluna para caber na tela */
        @media (max-width: 500px) {
            .form-nova-turma form, .form-container.form-nova-turma form { grid-template-columns: 1fr; }
        }
        @media (max-width: 768px) {
            .posicoes-elegiveis { overflow-x: auto; -webkit-overflow-scrolling: touch; }
            .posicoes-elegiveis table { min-width: 520px; }
        }
        .form-nova-turma-page { margin: 40px auto; }
        @media (max-width: 900px) {
            .form-nova-turma-page { margin: 16px auto; }
        }
        /* Scroll dentro do form-container quando no modal */
        #formModalContent .form-container.form-nova-turma {
            max-height: calc(100vh - 48px);
            overflow-y: auto;
            -webkit-overflow-scrolling: touch;
        }
    """

    form_card = f"""
    <div class="form-container form-nova-turma">
        <h2>Criar Nova Turma</h2>

        <div id="alert" class="alert"></div>

        <form id="formTurma" onsubmit="criarTurma(event)">
            <div class="form-group">
                <label>Produto *</label>
                <select name="produto_id" id="produto_id" required>
                    <option value="">Selecione...</option>
                    {options_html}
                </select>
            </div>

            <div class="form-group">
                <label>Nome da Turma *</label>
                <input type="text" name="nome" required placeholder="Ex: Memebot Turma 8">
            </div>

            <div class="form-group">
                <label>Data de Início *</label>
                <input type="date" name="data_inicio" id="data_inicio" required value="{datetime.now().strftime('%Y-%m-%d')}">
            </div>

            <div class="form-group form-group-full" id="posicoes-elegiveis-container">
                <div class="posicoes-elegiveis">
                    <h3>Posições abertas que serão replicadas</h3>
                    <p class="hint">Escolha a data em que cada posição entra na turma (replicação). Desmarque as que não quiser incluir.</p>
                    <div id="posicoes-elegiveis">
                        <span class="select-first">Selecione o produto e a data de início para carregar as posições elegíveis.</span>
                    </div>
                </div>
            </div>

            <div class="form-group">
                <label>Capital Base (R$)</label>
                <input type="number" name="capital_base" step="0.01" value="1500">
            </div>

            <div class="form-group form-group-full">
                <label>Descrição</label>
                <textarea name="descricao" rows="3" placeholder="Descrição opcional..."></textarea>
            </div>

            <button type="submit" class="btn-submit">Criar Turma</button>
            <a href="{return_url}" class="btn-cancel" onclick="var o=document.getElementById('formModalOverlay');if(o&&o.style.display==='block'){{if(typeof formModalCloseNow==='function')formModalCloseNow();event.preventDefault();return false;}}">Cancelar</a>
        </form>
    </div>
    """

    form_script = """
    <script>
        function showAlert(message, type) {
            const alert = document.getElementById('alert');
            alert.className = 'alert alert-' + type;
            alert.textContent = message;
            alert.style.display = 'block';
        }

        async function carregarPosicoesElegiveis() {
            const produtoId = document.getElementById('produto_id').value;
            const dataInicio = document.getElementById('data_inicio').value;
            const container = document.getElementById('posicoes-elegiveis');

            if (!produtoId || !dataInicio) {
                container.innerHTML = '<span class="select-first">Selecione o produto e a data de início para carregar as posições elegíveis.</span>';
                return;
            }

            container.innerHTML = '<span class="loading">Carregando posições...</span>';
            try {
                const response = await fetch(`/api/turma/posicoes-elegiveis?produto_id=${produtoId}&data_inicio=${dataInicio}`);
                const result = await response.json();
                if (!response.ok) {
                    container.innerHTML = '<span class="empty">Erro ao carregar: ' + (result.erro || response.status) + '</span>';
                    return;
                }
                const posicoes = result.posicoes || [];
                if (posicoes.length === 0) {
                    container.innerHTML = '<span class="empty">Nenhuma posição aberta elegível para esta data (posições com data de entrada anterior à data de início).</span>';
                    return;
                }
                let html = '<button type="button" class="btn-aplicar-data" onclick="aplicarDataInicioTodas()">Usar data de início em todas</button>';
                html += '<table><thead><tr><th></th><th>Ativo</th><th>Side</th><th>Data entrada</th><th>Data inserção na turma</th><th>Preço entrada</th><th>Qtd</th></tr></thead><tbody>';
                for (const p of posicoes) {
                    const qtd = p.quantidade != null ? Number(p.quantidade) : '—';
                    const preco = p.preco_entrada != null ? Number(p.preco_entrada).toFixed(4) : '—';
                    html += '<tr><td><input type="checkbox" name="posicao_sel" value="' + p.id + '" data-posicao-id="' + p.id + '" checked></td>';
                    html += '<td>' + (p.ativo || '—') + '</td><td>' + (p.side || '—') + '</td><td>' + (p.data_entrada || '—') + '</td>';
                    html += '<td><input type="date" class="data-insercao-input" data-posicao-id="' + p.id + '" value="' + dataInicio + '"></td>';
                    html += '<td>' + preco + '</td><td>' + qtd + '</td></tr>';
                }
                html += '</tbody></table>';
                container.innerHTML = html;
            } catch (err) {
                container.innerHTML = '<span class="empty">Erro de conexão: ' + err.message + '</span>';
            }
        }

        function aplicarDataInicioTodas() {
            const dataInicio = document.getElementById('data_inicio').value;
            if (!dataInicio) return;
            document.querySelectorAll('.data-insercao-input').forEach(function(inp) { inp.value = dataInicio; });
        }

        document.getElementById('produto_id').addEventListener('change', carregarPosicoesElegiveis);
        document.getElementById('data_inicio').addEventListener('change', carregarPosicoesElegiveis);

        async function criarTurma(e) {
            e.preventDefault();
            const form = e.target;
            const formData = new FormData(form);
            const dataInicio = formData.get('data_inicio');

            const checkboxes = form.querySelectorAll('input[name="posicao_sel"]:checked');
            const posicoesConfig = Array.from(checkboxes).map(function(cb) {
                const row = cb.closest('tr');
                const dateInput = row ? row.querySelector('.data-insercao-input') : null;
                const dataInsercao = dateInput && dateInput.value ? dateInput.value : dataInicio;
                return {
                    posicao_id: parseInt(cb.getAttribute('data-posicao-id'), 10),
                    data_insercao: dataInsercao
                };
            });

            if (posicoesConfig.length === 0) {
                const total = form.querySelectorAll('input[name="posicao_sel"]').length;
                if (total === 0) {
                    showAlert('Selecione produto e data de início e aguarde carregar as posições, ou inclua ao menos uma posição.', 'error');
                } else {
                    showAlert('Marque ao menos uma posição para replicar na turma.', 'error');
                }
                return;
            }

            const data = {
                produto_id: parseInt(formData.get('produto_id')),
                nome: formData.get('nome'),
                data_inicio: dataInicio,
                capital_base: parseFloat(formData.get('capital_base') || 1500),
                descricao: formData.get('descricao') || null,
                posicoes_config: posicoesConfig,
                auto_fetch_prices: true
            };

            try {
                const response = await fetch('/api/turma/criar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const result = await response.json();

                if (result.sucesso) {
                    showAlert('Turma criada com sucesso!', 'success');
                    setTimeout(() => window.location.href = '/turmas/' + result.turma_id, 1500);
                } else {
                    showAlert('Erro: ' + result.erro, 'error');
                }
            } catch (err) {
                showAlert('Erro de conexão: ' + err.message, 'error');
            }
        }
    </script>
    """

    if as_inner:
        return f"<style>{form_styles}</style>" + form_card + form_script

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nova Turma - Dashboard</title>
    <style>
        {get_base_styles()}
        {form_styles}
    </style>
</head>
<body>
    {get_navbar('turmas')}
    {get_turmas_subnav('turmas')}

    <div class="form-container form-nova-turma form-nova-turma-page">
        <h2>Criar Nova Turma</h2>

        <div id="alert" class="alert"></div>

        <form id="formTurma" onsubmit="criarTurma(event)">
            <div class="form-group">
                <label>Produto *</label>
                <select name="produto_id" id="produto_id" required>
                    <option value="">Selecione...</option>
                    {options_html}
                </select>
            </div>

            <div class="form-group">
                <label>Nome da Turma *</label>
                <input type="text" name="nome" required placeholder="Ex: Memebot Turma 8">
            </div>

            <div class="form-group">
                <label>Data de Início *</label>
                <input type="date" name="data_inicio" id="data_inicio" required value="{datetime.now().strftime('%Y-%m-%d')}">
            </div>

            <div class="form-group form-group-full" id="posicoes-elegiveis-container">
                <div class="posicoes-elegiveis">
                    <h3>Posições abertas que serão replicadas</h3>
                    <p class="hint">Escolha a data em que cada posição entra na turma (replicação). Desmarque as que não quiser incluir.</p>
                    <div id="posicoes-elegiveis">
                        <span class="select-first">Selecione o produto e a data de início para carregar as posições elegíveis.</span>
                    </div>
                </div>
            </div>

            <div class="form-group">
                <label>Capital Base (R$)</label>
                <input type="number" name="capital_base" step="0.01" value="1500">
            </div>

            <div class="form-group form-group-full">
                <label>Descrição</label>
                <textarea name="descricao" rows="3" placeholder="Descrição opcional..."></textarea>
            </div>

            <button type="submit" class="btn-submit">Criar Turma</button>
            <a href="{return_url}" class="btn-cancel">Cancelar</a>
        </form>
    </div>

    {form_script}
</body>
</html>"""


def _format_preco(preco):
    """Formata preço com casas decimais adequadas."""
    if preco is None:
        return '—'
    preco = float(preco)
    if preco >= 1000:
        return f"$ {preco:,.2f}"
    elif preco >= 1:
        return f"$ {preco:,.4f}"
    else:
        return f"$ {preco:,.6f}"


def _format_pnl(pnl_pct):
    """Formata PnL% com cor."""
    if pnl_pct is None:
        return '<span style="color: #888;">—</span>'
    color = '#39fda3' if pnl_pct >= 0 else '#e74c3c'
    sinal = '+' if pnl_pct >= 0 else ''
    return f'<span style="color: {color}; font-weight: bold;">{sinal}{pnl_pct:.2f}%</span>'


def _side_badge(side):
    """Gera badge de side."""
    side = (side or '').upper()
    if side == 'SHORT':
        return '<span style="background: rgba(231,76,60,0.2); color: #e74c3c; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">SHORT</span>'
    return '<span style="background: rgba(78,204,163,0.2); color: #39fda3; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">LONG</span>'


def _origem_badge(origem):
    """Gera badge de origem."""
    if origem == 'nativo':
        return '<span style="color: #39fda3;">Nativo</span>'
    return '<span style="color: #f39c12;">Replicado</span>'


def _status_badge(ativo_atual):
    """Gera badge de status."""
    if ativo_atual:
        return '<span style="color: #39fda3;">Ativo</span>'
    return '<span style="color: #888;">Fechado</span>'


def _render_tab_abertas(carteira, turma_id):
    """Renderiza tabela da aba Abertas."""
    trades = [t for t in carteira if t.get('ativo_atual')]
    if not trades:
        return '<tr><td colspan="9" style="text-align: center; color: #888;">Nenhum trade ativo</td></tr>'
    html = ""
    for t in trades:
        qtd = t.get('quantidade')
        qtd_str = f"{float(qtd):,.4f}" if qtd is not None else '—'
        stop = t.get('stop_atual')
        stop_str = _format_preco(stop) if stop is not None else '—'
        html += f"""
        <tr>
            <td style="text-align: left;">{t.get('ativo', 'N/A')}</td>
            <td>{_side_badge(t.get('side'))}</td>
            <td>{_origem_badge(t.get('origem'))}</td>
            <td>{t.get('data_insercao', 'N/A')}</td>
            <td>{_format_preco(t.get('preco_entrada_turma'))}</td>
            <td>{qtd_str}</td>
            <td>{_format_preco(t.get('preco_atual'))}</td>
            <td>{stop_str}</td>
            <td>{_format_pnl(t.get('pnl_pct'))}</td>
        </tr>
        """
    return html


def _render_tab_fechadas(carteira, turma_id):
    """Renderiza tabela da aba Fechadas."""
    trades = [t for t in carteira if not t.get('ativo_atual')]
    if not trades:
        return '<tr><td colspan="10" style="text-align: center; color: #888;">Nenhum trade fechado</td></tr>'
    html = ""
    for t in trades:
        stop = t.get('stop_atual')
        stop_str = _format_preco(stop) if stop is not None else '—'
        html += f"""
        <tr>
            <td style="text-align: left;">{t.get('ativo', 'N/A')}</td>
            <td>{_side_badge(t.get('side'))}</td>
            <td>{_origem_badge(t.get('origem'))}</td>
            <td>{t.get('data_insercao', 'N/A')}</td>
            <td>{t.get('data_remocao', '—')}</td>
            <td>{t.get('dias', 0)}</td>
            <td>{_format_preco(t.get('preco_entrada_turma'))}</td>
            <td>{_format_preco(t.get('preco_atual'))}</td>
            <td>{stop_str}</td>
            <td>{_format_pnl(t.get('pnl_pct'))}</td>
        </tr>
        """
    return html


def _render_tab_historico(carteira, turma_id):
    """Renderiza tabela da aba Histórico."""
    if not carteira:
        return '<tr><td colspan="11" style="text-align: center; color: #888;">Nenhum trade na carteira</td></tr>'
    html = ""
    for t in carteira:
        preco_saida_atual = _format_preco(t.get('preco_atual'))
        stop = t.get('stop_atual')
        stop_str = _format_preco(stop) if stop is not None else '—'
        html += f"""
        <tr>
            <td style="text-align: left;">{t.get('ativo', 'N/A')}</td>
            <td>{_side_badge(t.get('side'))}</td>
            <td>{_origem_badge(t.get('origem'))}</td>
            <td>{_status_badge(t.get('ativo_atual'))}</td>
            <td>{t.get('data_insercao', 'N/A')}</td>
            <td>{t.get('data_remocao', '—')}</td>
            <td>{t.get('dias', 0)}</td>
            <td>{_format_preco(t.get('preco_entrada_turma'))}</td>
            <td>{preco_saida_atual}</td>
            <td>{stop_str}</td>
            <td>{_format_pnl(t.get('pnl_pct'))}</td>
        </tr>
        """
    return html


def get_turma_detalhes_html(turma, resumo, carteira, tab_ativa='abertas'):
    """Gera HTML da página de detalhes de uma turma com abas Abertas/Fechadas/Histórico"""
    get_navbar, get_turmas_subnav, get_base_styles, _ = _get_shared_components()

    rentab = resumo.get('rentabilidade_acumulada_pct', 0)
    rentab_class = 'positive' if rentab >= 0 else 'negative'
    rentab_str = f"+{rentab:.2f}%" if rentab >= 0 else f"{rentab:.2f}%"

    turma_id = turma.get('id')

    # Contagens para as abas
    n_abertas = len([t for t in carteira if t.get('ativo_atual')])
    n_fechadas = len([t for t in carteira if not t.get('ativo_atual')])
    n_total = len(carteira)

    # Gerar conteúdo da aba ativa
    if tab_ativa == 'fechadas':
        tab_headers = """
                        <th style="text-align: left;">Ativo</th>
                        <th>Side</th>
                        <th>Origem</th>
                        <th>Data Inserção</th>
                        <th>Data Remoção</th>
                        <th>Dias</th>
                        <th>Preço Entrada</th>
                        <th>Preço Saída</th>
                        <th>Stop</th>
                        <th>PnL%</th>
        """
        tab_body = _render_tab_fechadas(carteira, turma_id)
    elif tab_ativa == 'historico':
        tab_headers = """
                        <th style="text-align: left;">Ativo</th>
                        <th>Side</th>
                        <th>Origem</th>
                        <th>Status</th>
                        <th>Data Inserção</th>
                        <th>Data Remoção</th>
                        <th>Dias</th>
                        <th>Preço Entrada</th>
                        <th>Preço Saída/Atual</th>
                        <th>Stop</th>
                        <th>PnL%</th>
        """
        tab_body = _render_tab_historico(carteira, turma_id)
    else:
        tab_headers = """
                        <th style="text-align: left;">Ativo</th>
                        <th>Side</th>
                        <th>Origem</th>
                        <th>Data Inserção</th>
                        <th>Preço Entrada</th>
                        <th>Qtd</th>
                        <th>Preço Atual</th>
                        <th>Stop</th>
                        <th>PnL%</th>
        """
        tab_body = _render_tab_abertas(carteira, turma_id)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{turma.get('nome', 'Turma')} - Dashboard</title>
    <style>
        {get_base_styles()}
        .content {{ padding: 30px; }}
        .turma-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 30px;
        }}
        .turma-info h2 {{
            color: #fff;
            font-size: 2em;
            margin-bottom: 10px;
        }}
        .turma-info .meta {{
            color: #888;
        }}
        .rentab-big {{
            text-align: right;
        }}
        .rentab-big .value {{
            font-size: 3em;
            font-weight: bold;
        }}
        .rentab-big .value.positive {{ color: #39fda3; }}
        .rentab-big .value.negative {{ color: #e74c3c; }}
        .rentab-big .label {{
            color: #888;
        }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #fff;
        }}
        .stat-card .label {{
            color: #888;
            margin-top: 5px;
        }}
        .section {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        .section h3 {{
            color: #39fda3;
            margin-bottom: 20px;
            font-size: 1.3em;
        }}
        .carteira-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .carteira-table th, .carteira-table td {{
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .carteira-table th {{
            background: rgba(0,0,0,0.3);
            color: #39fda3;
        }}
        .carteira-table tr:hover {{
            background: rgba(78, 204, 163, 0.1);
        }}
        .btn-chart {{
            display: inline-block;
            padding: 12px 24px;
            background: #39fda3;
            color: #1a1a2e;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
        }}
        .btn-chart:hover {{ background: #3db892; }}
        .tabs {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
        .tab {{
            padding: 10px 20px;
            background: #16213e;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: #888;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
        }}
        .tab:hover, .tab.active {{ border-color: #39fda3; color: #39fda3; }}
        .tab.active {{ background: rgba(78, 204, 163, 0.2); }}
    </style>
</head>
<body>
    {get_navbar('turmas')}
    {get_turmas_subnav('turmas')}

    <div class="content">
        <div class="turma-header">
            <div class="turma-info">
                <h2>{turma.get('nome', 'N/A')}</h2>
                <div class="meta">
                    <strong>{turma.get('produto_nome', 'N/A')}</strong> •
                    Início: {turma.get('data_inicio', 'N/A')} •
                    Capital Base: R$ {turma.get('capital_base', 1500):,.2f}
                </div>
            </div>
            <div class="rentab-big">
                <div class="value {rentab_class}">{rentab_str}</div>
                <div class="label">Rentabilidade Acumulada</div>
            </div>
        </div>

        <div class="stats-row">
            <div class="stat-card">
                <div class="value">R$ {resumo.get('valor_total', 0):,.2f}</div>
                <div class="label">Valor Total</div>
            </div>
            <div class="stat-card">
                <div class="value">R$ {resumo.get('capital_alocado', 0):,.2f}</div>
                <div class="label">Capital Alocado</div>
            </div>
            <div class="stat-card">
                <div class="value">R$ {resumo.get('capital_em_caixa', 0):,.2f}</div>
                <div class="label">Capital em Caixa</div>
            </div>
            <div class="stat-card">
                <div class="value">{resumo.get('trades_ativos', 0)}/{resumo.get('trades_fechados', 0)}</div>
                <div class="label">Ativos/Fechados</div>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <a href="/turmas/{turma_id}/rentabilidade" class="btn-chart">Ver Grafico de Rentabilidade</a>
        </div>

        <div class="section">
            <div class="tabs">
                <a href="/turmas/{turma_id}/abertas" class="tab {'active' if tab_ativa == 'abertas' else ''}">Abertas ({n_abertas})</a>
                <a href="/turmas/{turma_id}/fechadas" class="tab {'active' if tab_ativa == 'fechadas' else ''}">Fechadas ({n_fechadas})</a>
                <a href="/turmas/{turma_id}/historico" class="tab {'active' if tab_ativa == 'historico' else ''}">Historico ({n_total})</a>
            </div>
            <table class="carteira-table">
                <thead>
                    <tr>
                        {tab_headers}
                    </tr>
                </thead>
                <tbody>
                    {tab_body}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""


def get_rentabilidade_chart_html(turma, serie):
    """Gera HTML com gráfico de rentabilidade usando Chart.js"""
    get_navbar, get_turmas_subnav, get_base_styles, _ = _get_shared_components()

    labels = [p.dia for p in serie]
    valores = [p.valor_total for p in serie]
    rentab = [p.rentabilidade_acumulada_pct for p in serie]

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Rentabilidade - {turma.get('nome', 'Turma')}</title>
    <script src="/assets/chart.umd.min.js"></script>
    <style>
        {get_base_styles()}
        .content {{ padding: 30px; }}
        .chart-card {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        .chart-card h2 {{
            color: #39fda3;
            margin-bottom: 20px;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
        }}
    </style>
</head>
<body>
    {get_navbar('turmas')}
    {get_turmas_subnav('turmas')}

    <div class="content">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h2 style="color: #fff; font-size: 1.8em; margin-bottom: 5px;">{turma.get('nome', 'Turma')}</h2>
                <span style="color: #888;">{turma.get('produto_nome', '')} • Inicio: {turma.get('data_inicio', '')}</span>
            </div>
            <a href="/turmas/{turma.get('id')}" style="background: rgba(255,255,255,0.1); color: #fff; padding: 10px 20px; border-radius: 8px; text-decoration: none; transition: all 0.3s;">Voltar aos Detalhes</a>
        </div>

        <div class="chart-card">
            <h2>Valor do Portfolio</h2>
            <div class="chart-container">
                <canvas id="valorChart"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <h2>Rentabilidade Acumulada (%)</h2>
            <div class="chart-container">
                <canvas id="rentabChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const labels = {labels};
        const valores = {valores};
        const rentab = {rentab};

        // Gráfico de valor
        new Chart(document.getElementById('valorChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Valor Total (R$)',
                    data: valores,
                    borderColor: '#39fda3',
                    backgroundColor: 'rgba(78, 204, 163, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#eee' }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }},
                    y: {{
                        ticks: {{ color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }}
                }}
            }}
        }});

        // Gráfico de rentabilidade
        new Chart(document.getElementById('rentabChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Rentabilidade (%)',
                    data: rentab,
                    borderColor: rentab[rentab.length-1] >= 0 ? '#39fda3' : '#e74c3c',
                    backgroundColor: rentab[rentab.length-1] >= 0 ? 'rgba(78, 204, 163, 0.1)' : 'rgba(231, 76, 60, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#eee' }}
                    }}
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }},
                    y: {{
                        ticks: {{ color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""


def get_rentabilidade_historica_html(turmas_resumo):
    """Gera HTML com histórico de rentabilidade de todas as turmas"""
    get_navbar, get_turmas_subnav, get_base_styles, _ = _get_shared_components()

    # Gerar linhas da tabela
    tabela_html = ""
    for turma in turmas_resumo:
        rentab = turma.get('rentabilidade_atual_pct', 0)
        rentab_class = 'positive' if rentab >= 0 else 'negative'
        rentab_str = f"+{rentab:.2f}%" if rentab >= 0 else f"{rentab:.2f}%"

        rentab_max = turma.get('rentabilidade_max_pct', 0)
        rentab_min = turma.get('rentabilidade_min_pct', 0)

        tabela_html += f"""
        <tr onclick="selecionarTurma({turma.get('turma_id')})" style="cursor: pointer;">
            <td style="text-align: left;">{turma.get('nome', 'N/A')}</td>
            <td>{turma.get('produto', 'N/A')}</td>
            <td>{turma.get('data_inicio', 'N/A')}</td>
            <td>R$ {turma.get('capital_base', 0):,.2f}</td>
            <td>R$ {turma.get('valor_atual', 0):,.2f}</td>
            <td class="{rentab_class}">{rentab_str}</td>
            <td class="positive">+{rentab_max:.2f}%</td>
            <td class="negative">{rentab_min:.2f}%</td>
            <td>{turma.get('total_dias', 0)}</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Rentabilidade Histórica - Dashboard</title>
    <script src="/assets/chart.umd.min.js"></script>
    <style>
        {get_base_styles()}
        .content {{ padding: 30px; }}
        .page-header {{
            margin-bottom: 25px;
        }}
        .page-header h2 {{ color: #fff; font-size: 1.8em; }}
        .page-header p {{ color: #888; margin-top: 8px; }}

        .section {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        .section h3 {{
            color: #39fda3;
            margin-bottom: 20px;
            font-size: 1.3em;
        }}

        .turmas-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .turmas-table th, .turmas-table td {{
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .turmas-table th {{
            background: rgba(0,0,0,0.3);
            color: #39fda3;
            font-weight: 600;
        }}
        .turmas-table tr:hover {{
            background: rgba(78, 204, 163, 0.15);
        }}
        .turmas-table tr.selected {{
            background: rgba(78, 204, 163, 0.25);
        }}
        .turmas-table td:first-child, .turmas-table th:first-child {{
            text-align: left;
        }}
        .positive {{ color: #39fda3; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}

        .chart-card {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 25px;
            margin-top: 20px;
        }}
        .chart-card h3 {{
            color: #39fda3;
            margin-bottom: 20px;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
        }}

        .info-box {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .info-item {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .info-item .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #fff;
        }}
        .info-item .value.positive {{ color: #39fda3; }}
        .info-item .value.negative {{ color: #e74c3c; }}
        .info-item .label {{
            font-size: 0.85em;
            color: #888;
            margin-top: 5px;
        }}

        .loading {{
            text-align: center;
            padding: 40px;
            color: #888;
        }}
        .hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    {get_navbar('turmas')}
    {get_turmas_subnav('historico')}

    <div class="content">
        <div class="page-header" style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h2>Rentabilidade Histórica por Turma</h2>
                <p>Clique em uma turma para ver o gráfico detalhado da rentabilidade acumulada ao longo do tempo.</p>
            </div>
            <div style="display: flex; gap: 10px;">
                <button onclick="atualizarCotacoes()" id="btnAtualizar" style="background: #39fda3; color: #1a1a2e; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; white-space: nowrap;">
                    🔄 Atualizar Preços
                </button>
                <button onclick="baixarExcel()" id="btnExcel" style="background: #3498db; color: white; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; white-space: nowrap;">
                    📥 Baixar Excel
                </button>
            </div>
        </div>

        <div class="section">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="margin: 0;">Resumo de Todas as Turmas</h3>
                <span id="statusAtualizar" style="color: #888; font-size: 0.9em;"></span>
            </div>
            <table class="turmas-table">
                <thead>
                    <tr>
                        <th>Turma</th>
                        <th>Produto</th>
                        <th>Data Início</th>
                        <th>Capital Base</th>
                        <th>Valor Atual</th>
                        <th>Rentab. Atual</th>
                        <th>Máxima</th>
                        <th>Mínima</th>
                        <th>Dias</th>
                    </tr>
                </thead>
                <tbody>
                    {tabela_html if tabela_html else '<tr><td colspan="9" style="text-align: center; color: #888;">Nenhuma turma cadastrada.</td></tr>'}
                </tbody>
            </table>
        </div>

        <div id="chartSection" class="chart-card hidden">
            <h3 id="chartTitle">Selecione uma turma acima</h3>

            <div id="turmaInfo" class="info-box hidden">
                <div class="info-item">
                    <div id="infoValorAtual" class="value">-</div>
                    <div class="label">Valor Atual</div>
                </div>
                <div class="info-item">
                    <div id="infoRentabAtual" class="value">-</div>
                    <div class="label">Rentabilidade Atual</div>
                </div>
                <div class="info-item">
                    <div id="infoRentabMax" class="value positive">-</div>
                    <div class="label">Máxima Histórica</div>
                </div>
                <div class="info-item">
                    <div id="infoRentabMin" class="value negative">-</div>
                    <div class="label">Mínima Histórica</div>
                </div>
            </div>

            <div id="loadingChart" class="loading">Carregando dados...</div>
            <div class="chart-container">
                <canvas id="historicoChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        let chart = null;
        let turmaAtual = null;

        function baixarExcel() {{
            const btn = document.getElementById('btnExcel');
            const status = document.getElementById('statusAtualizar');

            btn.disabled = true;
            btn.textContent = '⏳ Gerando...';
            status.textContent = 'Gerando planilha Excel...';
            status.style.color = '#3498db';

            // Criar link temporário para download
            const link = document.createElement('a');
            link.href = '/api/rentabilidade/excel';
            link.download = 'rentabilidade_turmas.xlsx';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Restaurar botão após 2s
            setTimeout(() => {{
                btn.disabled = false;
                btn.textContent = '📥 Baixar Excel';
                status.textContent = '✓ Download iniciado';
                status.style.color = '#39fda3';
            }}, 2000);
        }}

        async function atualizarCotacoes() {{
            const btn = document.getElementById('btnAtualizar');
            const status = document.getElementById('statusAtualizar');

            btn.disabled = true;
            btn.textContent = '⏳ Atualizando...';
            status.textContent = 'Buscando preços...';
            status.style.color = '#f39c12';

            try {{
                const response = await fetch('/api/cotacoes/atualizar');
                const data = await response.json();

                if (data.sucesso) {{
                    status.textContent = `✓ Atualizado: ${{data.dias_preenchidos}} dias preenchidos`;
                    status.style.color = '#39fda3';
                    // Recarregar a página após 1.5s para mostrar dados atualizados
                    setTimeout(() => window.location.reload(), 1500);
                }} else {{
                    status.textContent = `✗ Erro: ${{data.erro}}`;
                    status.style.color = '#e74c3c';
                }}
            }} catch (err) {{
                status.textContent = `✗ Erro de conexão: ${{err.message}}`;
                status.style.color = '#e74c3c';
            }}

            btn.disabled = false;
            btn.textContent = '🔄 Atualizar Preços';
        }}

        async function selecionarTurma(turmaId) {{
            // Highlight row
            document.querySelectorAll('.turmas-table tbody tr').forEach(row => {{
                row.classList.remove('selected');
            }});
            event.currentTarget.classList.add('selected');

            // Show chart section
            document.getElementById('chartSection').classList.remove('hidden');
            document.getElementById('loadingChart').classList.remove('hidden');
            document.getElementById('turmaInfo').classList.add('hidden');

            // Fetch data
            try {{
                const response = await fetch('/api/turma/' + turmaId + '/historico');
                const data = await response.json();

                if (data.erro) {{
                    document.getElementById('loadingChart').innerHTML = 'Erro: ' + data.erro;
                    return;
                }}

                turmaAtual = data;
                renderizarGrafico(data);

            }} catch (err) {{
                document.getElementById('loadingChart').innerHTML = 'Erro ao carregar dados: ' + err.message;
            }}
        }}

        function renderizarGrafico(data) {{
            document.getElementById('loadingChart').classList.add('hidden');
            document.getElementById('turmaInfo').classList.remove('hidden');

            // Update title
            document.getElementById('chartTitle').textContent = data.nome + ' - Rentabilidade Histórica';

            // Update info boxes
            const rentabAtual = data.rentabilidade_atual_pct || 0;
            const rentabMax = data.rentabilidade_max_pct || 0;
            const rentabMin = data.rentabilidade_min_pct || 0;

            document.getElementById('infoValorAtual').textContent = 'R$ ' + (data.valor_atual || 0).toLocaleString('pt-BR', {{minimumFractionDigits: 2}});

            const rentabAtualEl = document.getElementById('infoRentabAtual');
            rentabAtualEl.textContent = (rentabAtual >= 0 ? '+' : '') + rentabAtual.toFixed(2) + '%';
            rentabAtualEl.className = 'value ' + (rentabAtual >= 0 ? 'positive' : 'negative');

            document.getElementById('infoRentabMax').textContent = '+' + rentabMax.toFixed(2) + '%';
            document.getElementById('infoRentabMin').textContent = rentabMin.toFixed(2) + '%';

            // Prepare chart data
            const historico = data.historico || [];
            const labels = historico.map(h => h.data);
            const rentabData = historico.map(h => h.rentabilidade_acumulada_pct);
            const valorData = historico.map(h => h.valor_total);

            // Destroy existing chart
            if (chart) chart.destroy();

            // Create new chart
            const ctx = document.getElementById('historicoChart').getContext('2d');
            const ultimaRentab = rentabData[rentabData.length - 1] || 0;

            chart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'Rentabilidade Acumulada (%)',
                        data: rentabData,
                        borderColor: ultimaRentab >= 0 ? '#39fda3' : '#e74c3c',
                        backgroundColor: ultimaRentab >= 0 ? 'rgba(78, 204, 163, 0.1)' : 'rgba(231, 76, 60, 0.1)',
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y'
                    }}, {{
                        label: 'Valor do Portfolio (R$)',
                        data: valorData,
                        borderColor: '#3498db',
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4,
                        yAxisID: 'y1'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false
                    }},
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#eee' }}
                        }},
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    if (context.datasetIndex === 0) {{
                                        return context.dataset.label + ': ' + context.raw.toFixed(2) + '%';
                                    }} else {{
                                        return context.dataset.label + ': R$ ' + context.raw.toLocaleString('pt-BR', {{minimumFractionDigits: 2}});
                                    }}
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            ticks: {{
                                color: '#888',
                                callback: function(value) {{ return value.toFixed(1) + '%'; }}
                            }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            ticks: {{
                                color: '#3498db',
                                callback: function(value) {{ return 'R$ ' + value.toFixed(0); }}
                            }},
                            grid: {{ drawOnChartArea: false }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>"""


def get_comparar_turmas_html(turmas, comparacao):
    """Gera HTML para comparar múltiplas turmas"""
    get_navbar, get_turmas_subnav, get_base_styles, _ = _get_shared_components()

    # Checkboxes
    checkboxes_html = ""
    for t in turmas:
        checkboxes_html += f"""
        <label class="compare-item">
            <input type="checkbox" name="turma" value="{t['id']}" onchange="atualizarGrafico()">
            {t['nome']}
        </label>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comparar Turmas - Dashboard</title>
    <script src="/assets/chart.umd.min.js"></script>
    <style>
        {get_base_styles()}
        .content {{ padding: 30px; }}
        .selector-card {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        .selector-card h3 {{
            color: #39fda3;
            margin-bottom: 15px;
        }}
        .compare-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }}
        .compare-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 5px;
            cursor: pointer;
        }}
        .compare-item:hover {{
            background: rgba(78, 204, 163, 0.1);
        }}
        .compare-item input {{
            margin-right: 10px;
            width: 18px;
            height: 18px;
        }}
        .chart-card {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 25px;
        }}
        .chart-card h3 {{
            color: #39fda3;
            margin-bottom: 20px;
        }}
        .chart-container {{
            position: relative;
            height: 500px;
        }}
    </style>
</head>
<body>
    {get_navbar('turmas')}
    {get_turmas_subnav('comparar')}

    <div class="content">
        <div class="selector-card">
            <h3>Selecione as turmas para comparar:</h3>
            <div class="compare-grid">
                {checkboxes_html}
            </div>
        </div>

        <div class="chart-card">
            <h3>Rentabilidade Comparativa (%)</h3>
            <div class="chart-container">
                <canvas id="compareChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const turmasData = {{}};
        let chart = null;

        const cores = [
            '#39fda3', '#e74c3c', '#3498db', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#2ecc71', '#e91e63', '#00bcd4'
        ];

        async function carregarDados(turmaId) {{
            if (turmasData[turmaId]) return turmasData[turmaId];

            try {{
                const response = await fetch('/api/turma/' + turmaId + '/rentabilidade');
                const data = await response.json();
                turmasData[turmaId] = data;
                return data;
            }} catch (err) {{
                console.error('Erro ao carregar dados:', err);
                return null;
            }}
        }}

        async function atualizarGrafico() {{
            const checkboxes = document.querySelectorAll('input[name="turma"]:checked');
            const turmaIds = Array.from(checkboxes).map(cb => cb.value);

            // Carregar dados de todas as turmas selecionadas
            const turmasCarregadas = [];
            for (let i = 0; i < turmaIds.length; i++) {{
                const turmaId = turmaIds[i];
                const data = await carregarDados(turmaId);
                if (data && data.length > 0) {{
                    const label = document.querySelector('input[value="' + turmaId + '"]').parentElement.textContent.trim();
                    turmasCarregadas.push({{ turmaId, data, label, cor: cores[i % cores.length] }});
                }}
            }}

            // Unificar todas as datas em um conjunto ordenado
            const allDatesSet = new Set();
            turmasCarregadas.forEach(function(t) {{
                t.data.forEach(function(d) {{ allDatesSet.add(d.dia); }});
            }});
            const labels = Array.from(allDatesSet).sort();

            // Para cada turma, mapear valores para as datas unificadas
            const datasets = turmasCarregadas.map(function(t) {{
                const porData = {{}};
                t.data.forEach(function(d) {{ porData[d.dia] = d.rentabilidade_acumulada_pct; }});
                return {{
                    label: t.label,
                    data: labels.map(function(dia) {{ return porData[dia] !== undefined ? porData[dia] : null; }}),
                    borderColor: t.cor,
                    fill: false,
                    tension: 0.4,
                    spanGaps: false
                }};
            }});

            if (chart) chart.destroy();

            chart = new Chart(document.getElementById('compareChart'), {{
                type: 'line',
                data: {{ labels, datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            labels: {{ color: '#eee' }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            ticks: {{ color: '#888' }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }},
                        y: {{
                            ticks: {{
                                color: '#888',
                                callback: function(value) {{ return value.toFixed(1) + '%'; }}
                            }},
                            grid: {{ color: 'rgba(255,255,255,0.1)' }}
                        }}
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>"""


def _json_serializer(obj):
    """Serializa tipos especiais para JSON (Decimal, date, etc)."""
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return str(obj)


def get_produto_dashboard_html(produto, turmas, resumo, carteira, mostrar_caixa_alocacao=False, product_tabs=None):
    """Dashboard rico do produto com abas por turma, gráficos e tabelas interativas.
    mostrar_caixa_alocacao: exibe Caixa no gráfico de evolução da alocação apenas para produtos com API key (Soros, Memebot).
    product_tabs: optional dict with 'tabs', 'group_name', 'primary_id' for grouped products (e.g. Soros Spot 1+2)."""
    get_navbar, _, get_base_styles, _get_form_modal_overlay_script = _get_shared_components()

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    produto_id = produto['id']
    nome = product_tabs['group_name'] if product_tabs else produto['nome']
    tipo = produto.get('tipo', 'Outro')

    turma_ativa = turmas[0] if turmas else {}
    turma_ativa_id = turma_ativa.get('id', 0)

    rentab = resumo.get('rentabilidade_acumulada_pct', 0) or 0
    valor_total = resumo.get('valor_total', 0) or 0
    capital_alocado = resumo.get('capital_alocado', 0) or 0
    capital_em_caixa = resumo.get('capital_em_caixa', 0) or 0
    capital_base = resumo.get('capital_base', 1500) or 1500
    trades_ativos = resumo.get('trades_ativos', 0) or 0
    trades_fechados = resumo.get('trades_fechados', 0) or 0

    rentab_class = 'positive' if rentab >= 0 else 'negative'
    rentab_str = f"+{rentab:.2f}%" if rentab >= 0 else f"{rentab:.2f}%"

    turma_tabs_html = ""
    if product_tabs:
        primary_id = product_tabs['primary_id']
        for i, ptab in enumerate(product_tabs['tabs']):
            active_cls = 'active' if ptab['active'] else ''
            href = f"/produto/{primary_id}?tab={i}"
            turma_tabs_html += f'<a class="turma-tab {active_cls}" href="{href}">{ptab["nome"]}</a>'
    else:
        for turma in turmas:
            active = 'active' if turma['id'] == turma_ativa_id else ''
            turma_tabs_html += f'<button class="turma-tab {active}" data-turma-id="{turma["id"]}" onclick="switchTurma({turma["id"]}, this)">{turma.get("nome", "N/A")}</button>'

    turmas_json = json.dumps([
        {'id': t['id'], 'nome': t.get('nome', ''), 'data_inicio': str(t.get('data_inicio', ''))[:10]}
        for t in turmas
    ], default=_json_serializer, ensure_ascii=False)
    data_inicio_turma = str(turma_ativa.get('data_inicio', ''))[:10]

    carteira_json = json.dumps(carteira, default=_json_serializer, ensure_ascii=False)
    resumo_safe = {
        'rentabilidade_acumulada_pct': rentab,
        'valor_total': valor_total,
        'capital_alocado': capital_alocado,
        'capital_em_caixa': capital_em_caixa,
        'capital_base': capital_base,
        'trades_ativos': trades_ativos,
        'trades_fechados': trades_fechados,
    }
    resumo_json = json.dumps(resumo_safe, default=_json_serializer)

    tipo_lower = tipo.lower()
    if 'spot' in tipo_lower:
        tipo_badge_class = 'tipo-spot'
    elif 'perp' in tipo_lower:
        tipo_badge_class = 'tipo-perp'
    else:
        tipo_badge_class = 'tipo-outro'

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{nome} - Dashboard</title>
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
        .tipo-spot {{ background: #39fda3; color: #1a1a2e; }}
        .tipo-perp {{ background: #ff6b6b; color: #fff; }}
        .tipo-outro {{ background: #ffd93d; color: #1a1a2e; }}
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

        .period-btn {{
            padding: 4px 12px; border-radius: 4px; border: 1px solid #333;
            background: transparent; color: #888; cursor: pointer; font-size: 12px; transition: all 0.2s;
        }}
        .period-btn:hover {{ color: #e0e0e0; border-color: #39fda3; }}
        .period-btn.active {{ background: #39fda3; color: #1a1a2e; border-color: #39fda3; font-weight: 600; }}

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


        /* --- Gear menu (config) --- */
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
        <div><h1>{nome}</h1></div>
        <span class="tipo-badge {tipo_badge_class}">{tipo}</span>
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
                        <button class="gear-item" onclick="preencherPrecos({produto_id},'entrada'); toggleGearMenu();">
                            <span class="gear-icon">&#8594;</span> Preencher <span class="gear-accent">Entrada</span>
                        </button>
                        <button class="gear-item" onclick="preencherPrecos({produto_id},'saida'); toggleGearMenu();">
                            <span class="gear-icon">&#8592;</span> Preencher <span class="gear-accent">Sa&iacute;da</span>
                        </button>
                        <button class="gear-item" onclick="preencherPrecos({produto_id},'ambos'); toggleGearMenu();">
                            <span class="gear-icon">&#8596;</span> Preencher <span class="gear-accent">Todos</span>
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
                        <a href="/api/export/excel?produto_id={produto_id}" download><span class="gear-icon">&#8595;</span> Baixar Excel (Rentab. + Posi&ccedil;&otilde;es)</a>
                    </div>
                    <div class="gear-panel-group">
                        <div class="gear-panel-title">Aloca&ccedil;&atilde;o &amp; Turmas</div>
                        <a href="/alocacao/nova?produto_id={produto_id}"><span class="gear-icon gear-accent">+</span> Nova Aloca&ccedil;&atilde;o</a>
                        <a href="/turmas/nova?produto_id={produto_id}"><span class="gear-icon gear-accent">+</span> Nova Turma</a>
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

    <div class="turma-tabs-bar" id="turmaTabs">
        {turma_tabs_html if turma_tabs_html else '<span style="padding:12px;color:#666;">Nenhuma turma cadastrada</span>'}
    </div>

    <div class="dash-content">
        <div class="summary-row" id="summaryCards">
            <div class="summary-card">
                <div class="s-value {rentab_class}" id="card-rentab">{rentab_str}</div>
                <div class="s-label">Rentabilidade Acumulada</div>
            </div>
            <div class="summary-card">
                <div class="s-value" id="card-valor">R$ {valor_total:,.2f}</div>
                <div class="s-label">Valor Total</div>
            </div>
            <div class="summary-card">
                <div class="s-value" id="card-alocado">R$ {capital_alocado:,.2f}</div>
                <div class="s-label">Capital Alocado</div>
            </div>
            <div class="summary-card">
                <div class="s-value" id="card-caixa">R$ {capital_em_caixa:,.2f}</div>
                <div class="s-label">Capital em Caixa</div>
            </div>
            <div class="summary-card">
                <div class="s-value" id="card-base">R$ {capital_base:,.2f}</div>
                <div class="s-label">Capital Base</div>
            </div>
            <div class="summary-card">
                <div class="s-value" id="card-trades">{trades_ativos} / {trades_fechados}</div>
                <div class="s-label">Ativos / Fechados</div>
            </div>
        </div>

        <div class="dash-section" id="sectionChart">
            <h3 id="chartTitle">Rentabilidade Acumulada</h3>
            <div class="chart-toggles" style="display:flex; gap:16px; margin-bottom:10px; align-items:center; flex-wrap:wrap;">
                <label style="display:flex; align-items:center; gap:5px; cursor:pointer; color:#e0e0e0; font-size:13px;">
                    <input type="checkbox" id="toggleTurma" checked style="accent-color:#39fda3; width:15px; height:15px; cursor:pointer;">
                    <span style="display:inline-block; width:14px; height:3px; background:#39fda3; border-radius:2px;"></span>
                    <span id="toggleTurmaLabel">{turma_ativa.get('nome', 'Turma')}</span>
                </label>
                <label style="display:flex; align-items:center; gap:5px; cursor:pointer; color:#e0e0e0; font-size:13px;">
                    <input type="checkbox" id="toggleBTC" checked style="accent-color:#f7931a; width:15px; height:15px; cursor:pointer;">
                    <span style="display:inline-block; width:14px; height:0; border-top:2px dashed #f7931a;"></span>
                    Bitcoin (BTC)
                </label>
                <span style="color:#888; font-size:13px;">Comparar com:</span>
                <div id="compareCheckboxes" style="display:inline-flex; flex-wrap:wrap; gap:10px 14px; align-items:center;"></div>
            </div>
            <div style="display:flex; gap:6px; margin-bottom:12px; align-items:center; flex-wrap:wrap;">
                <button class="period-btn active" data-period="MAX" onclick="setRentPeriod('MAX',this)">MAX</button>
                <button class="period-btn" data-period="YTD" onclick="setRentPeriod('YTD',this)">YTD</button>
                <button class="period-btn" data-period="1Y" onclick="setRentPeriod('1Y',this)">1A</button>
                <button class="period-btn" data-period="6M" onclick="setRentPeriod('6M',this)">6M</button>
                <button class="period-btn" data-period="3M" onclick="setRentPeriod('3M',this)">3M</button>
                <button class="period-btn" data-period="1M" onclick="setRentPeriod('1M',this)">1M</button>
                <span style="color:#444; margin:0 4px;">|</span>
                <input type="date" id="rentPeriodStart" style="background:#1a1a2e; color:#e0e0e0; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:12px;" onchange="setRentCustomPeriod()">
                <span style="color:#666; font-size:12px;">a</span>
                <input type="date" id="rentPeriodEnd" style="background:#1a1a2e; color:#e0e0e0; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:12px;" onchange="setRentCustomPeriod()">
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
                <input type="date" id="pnlPeriodStart" style="background:#1a1a2e; color:#e0e0e0; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:12px;" onchange="setPnlCustomPeriod()">
                <span style="color:#666; font-size:12px;">a</span>
                <input type="date" id="pnlPeriodEnd" style="background:#1a1a2e; color:#e0e0e0; border:1px solid #333; border-radius:4px; padding:3px 8px; font-size:12px;" onchange="setPnlCustomPeriod()">
            </div>
            <div id="pnlChartLoading" class="loading-overlay"><div class="spinner"></div>Carregando gr&aacute;fico...</div>
            <div class="chart-box" id="pnlChartWrapper" style="height:280px; display:none;">
                <canvas id="chartPnlAbertas"></canvas>
            </div>
        </div>

        <div class="dash-section" id="sectionPositions">
            <h3>Posi&ccedil;&otilde;es</h3>
            <div class="sub-tabs" id="posTabs">
                <button class="sub-tab active" onclick="switchPosTab('abertas', this)">Abertas (<span id="countAbertas">{trades_ativos}</span>)</button>
                <button class="sub-tab" onclick="switchPosTab('fechadas', this)">Fechadas (<span id="countFechadas">{trades_fechados}</span>)</button>
                <button class="sub-tab" onclick="switchPosTab('historico', this)">Hist&oacute;rico (<span id="countHistorico">{trades_ativos + trades_fechados}</span>)</button>
            </div>

            <div id="tab-abertas" class="sub-content active">
                <div class="tbl-scroll"><table class="dtable" id="tblAbertas">
                    <thead><tr><th>Ativo</th><th>Side</th><th>Origem</th><th>Data Entrada</th><th>Pre&ccedil;o Entrada</th><th>Qtd</th><th>Entrada Total</th><th>Pre&ccedil;o Atual</th><th>Atual Total</th><th>Stop</th><th>PnL%</th></tr></thead>
                    <tbody id="tbAbertas"></tbody>
                </table></div>
            </div>
            <div id="tab-fechadas" class="sub-content">
                <div class="tbl-scroll"><table class="dtable" id="tblFechadas">
                    <thead><tr><th>Ativo</th><th>Side</th><th>Origem</th><th>Data Entrada</th><th>Data Sa&iacute;da</th><th>Dias</th><th>Pre&ccedil;o Entrada</th><th>Entrada Total</th><th>Pre&ccedil;o Sa&iacute;da</th><th>Sa&iacute;da Total</th><th>Stop</th><th>PnL%</th></tr></thead>
                    <tbody id="tbFechadas"></tbody>
                </table></div>
            </div>
            <div id="tab-historico" class="sub-content">
                <div class="tbl-scroll"><table class="dtable" id="tblHistorico">
                    <thead><tr><th>Ativo</th><th>Side</th><th>Origem</th><th>Status</th><th>Data Entrada</th><th>Data Sa&iacute;da</th><th>Dias</th><th>Pre&ccedil;o Entrada</th><th>Entrada Total</th><th>Pre&ccedil;o Sa&iacute;da/Atual</th><th>Sa&iacute;da Total</th><th>Stop</th><th>PnL%</th></tr></thead>
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
var TURMA_ID = {turma_ativa_id};
var PRODUTO_ID = {produto_id};
var RESUMO = {resumo_json};
var CARTEIRA = {carteira_json};
var TURMAS = {turmas_json};
var DATA_INICIO_TURMA = '{data_inicio_turma}';
var MOSTRAR_CAIXA_ALOCACAO = {str(mostrar_caixa_alocacao).lower()};
var rentChart = null;
var pnlAbertasChart = null;
var allocTimelineChart = null;

// Gear menu toggle
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
var btcSerie = null;
var compareSeries = {{}};
var COMPARE_COLORS = ['#e056fd','#3498db','#f39c12','#1abc9c','#9b59b6'];
var currentTurmaSerie = null;
var fullRentSerie = null;
var activeRentPeriod = 'MAX';
var rentPeriodStartDate = null;
var rentPeriodEndDate = null;

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
function origemBdg(o) {{
    return o==='nativo' ? '<span style="color:#39fda3;">Nativo</span>' : '<span style="color:#f39c12;">Replicado</span>';
}}
function statusBdg(a) {{
    return a ? '<span class="bdg bdg-open">Aberto</span>' : '<span class="bdg bdg-closed">Fechado</span>';
}}
function fmtDate(d) {{ return d || '\\u2014'; }}
function fmtQty(q) {{ return q!=null ? Number(q).toLocaleString('en-US',{{maximumFractionDigits:4}}) : '\\u2014'; }}

function filterSerieByDate(serie, startDate, endDate) {{
    if (!serie || !serie.length) return serie;
    return serie.filter(function(s) {{
        var d = s.dia || s.data;
        if (!d) return true;
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
        return {{ dia: s.dia, valor_total: s.valor_total, rentabilidade_acumulada_pct: ((1 + s.rentabilidade_acumulada_pct / 100) / baseFactor - 1) * 100 }};
    }});
}}
function getDateForRentPeriod(period) {{
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
function setRentPeriod(period, btn) {{
    activeRentPeriod = period;
    document.querySelectorAll('.dash-section#sectionChart .period-btn').forEach(function(b){{ if(b.classList.contains('pnl-period-btn')) return; b.classList.remove('active'); }});
    if (btn) btn.classList.add('active');
    rentPeriodStartDate = getDateForRentPeriod(period);
    rentPeriodEndDate = null;
    var startEl = document.getElementById('rentPeriodStart');
    var endEl = document.getElementById('rentPeriodEnd');
    if (startEl) startEl.value = rentPeriodStartDate || '';
    if (endEl) endEl.value = '';
    applyRentPeriodFilter();
}}
function setRentCustomPeriod() {{
    rentPeriodStartDate = document.getElementById('rentPeriodStart').value || null;
    rentPeriodEndDate = document.getElementById('rentPeriodEnd').value || null;
    activeRentPeriod = 'CUSTOM';
    document.querySelectorAll('.dash-section#sectionChart .period-btn').forEach(function(b){{ if(b.classList.contains('pnl-period-btn')) return; b.classList.remove('active'); }});
    applyRentPeriodFilter();
}}
function applyRentPeriodFilter() {{
    if (!fullRentSerie) {{ if (currentTurmaSerie) renderRentChart(currentTurmaSerie); return; }}
    if (activeRentPeriod === 'MAX' && !rentPeriodStartDate && !rentPeriodEndDate) {{
        currentTurmaSerie = fullRentSerie;
        renderRentChartFiltered(fullRentSerie, btcSerie, compareSeries);
        return;
    }}
    var inicio = rentPeriodStartDate || null;
    var fim = rentPeriodEndDate || null;
    var urlMain = '/api/turma/' + TURMA_ID + '/rentabilidade?';
    if (inicio) urlMain += 'inicio=' + encodeURIComponent(inicio) + '&';
    if (fim) urlMain += 'fim=' + encodeURIComponent(fim);
    var compareIds = Object.keys(compareSeries);
    var query = (inicio ? 'inicio=' + encodeURIComponent(inicio) + '&' : '') + (fim ? 'fim=' + encodeURIComponent(fim) : '');
    var promises = [fetch(urlMain).then(function(r){{ return r.json(); }})];
    compareIds.forEach(function(tid) {{ promises.push(fetch('/api/turma/' + tid + '/rentabilidade?' + query).then(function(r){{ return r.json(); }})); }});
    document.getElementById('chartLoading').style.display = 'block';
    document.getElementById('chartWrapper').style.display = 'none';
    Promise.all(promises).then(function(results) {{
        document.getElementById('chartLoading').style.display = 'none';
        document.getElementById('chartWrapper').style.display = 'block';
        var data = results[0];
        if (data.erro || !Array.isArray(data) || !data.length) {{ if (currentTurmaSerie) renderRentChart(currentTurmaSerie); return; }}
        var rebased = rebaseRent(data);
        currentTurmaSerie = rebased;
        var filteredBtc = btcSerie ? rebaseRent(filterSerieByDate(btcSerie, inicio, fim)) : null;
        var filteredCompare = {{}};
        for (var i = 0; i < compareIds.length; i++) {{
            var d = results[i + 1];
            if (d && !d.erro && Array.isArray(d) && d.length) filteredCompare[compareIds[i]] = rebaseRent(d);
        }}
        renderRentChartFiltered(rebased, filteredBtc, filteredCompare);
    }}).catch(function(e) {{
        document.getElementById('chartLoading').style.display = 'none';
        document.getElementById('chartWrapper').style.display = 'block';
        console.error('Erro ao buscar rentabilidade:', e);
    }});
}}

function updateCards(r) {{
    var rentab = r.rentabilidade_acumulada_pct||0;
    var el = document.getElementById('card-rentab');
    el.textContent = (rentab>=0?'+':'')+rentab.toFixed(2)+'%';
    el.className = 's-value '+(rentab>=0?'positive':'negative');
    document.getElementById('card-valor').textContent = 'R$ '+(r.valor_total||0).toLocaleString('pt-BR',{{minimumFractionDigits:2,maximumFractionDigits:2}});
    document.getElementById('card-alocado').textContent = 'R$ '+(r.capital_alocado||0).toLocaleString('pt-BR',{{minimumFractionDigits:2,maximumFractionDigits:2}});
    document.getElementById('card-caixa').textContent = 'R$ '+(r.capital_em_caixa||0).toLocaleString('pt-BR',{{minimumFractionDigits:2,maximumFractionDigits:2}});
    document.getElementById('card-base').textContent = 'R$ '+(r.capital_base||1500).toLocaleString('pt-BR',{{minimumFractionDigits:2,maximumFractionDigits:2}});
    document.getElementById('card-trades').textContent = (r.trades_ativos||0)+' / '+(r.trades_fechados||0);
    document.getElementById('countAbertas').textContent = r.trades_ativos||0;
    document.getElementById('countFechadas').textContent = r.trades_fechados||0;
    document.getElementById('countHistorico').textContent = (r.trades_ativos||0)+(r.trades_fechados||0);
}}

function fmtStop(v) {{ if(v==null||v==undefined) return '\\u2014'; if(v===-1) return '<span class="bdg bdg-stop">Stop Atingido</span>'; return '$ '+fmtPrice(v); }}
function fmtTotal(v) {{
    if(v==null||v===undefined) return '\\u2014';
    return '$ '+v.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}});
}}
function rowOpen(t) {{
    return '<tr><td>'+t.ativo+'</td><td>'+sideBdg(t.side)+'</td><td>'+origemBdg(t.origem)+'</td><td>'+fmtDate(t.data_insercao)+'</td><td>$ '+fmtPrice(t.preco_entrada_turma)+'</td><td>'+fmtQty(t.quantidade)+'</td><td>'+fmtTotal(t.preco_entrada_total)+'</td><td>$ '+fmtPrice(t.preco_atual)+'</td><td>'+fmtTotal(t.preco_saida_total)+'</td><td>'+fmtStop(t.stop_atual)+'</td><td>'+fmtPnl(t.pnl_pct)+'</td></tr>';
}}
function rowClosed(t) {{
    return '<tr><td>'+t.ativo+'</td><td>'+sideBdg(t.side)+'</td><td>'+origemBdg(t.origem)+'</td><td>'+fmtDate(t.data_insercao)+'</td><td>'+fmtDate(t.data_remocao)+'</td><td>'+(t.dias!=null?t.dias:'\\u2014')+'</td><td>$ '+fmtPrice(t.preco_entrada_turma)+'</td><td>'+fmtTotal(t.preco_entrada_total)+'</td><td>$ '+fmtPrice(t.preco_atual)+'</td><td>'+fmtTotal(t.preco_saida_total)+'</td><td>'+fmtStop(t.stop_atual)+'</td><td>'+fmtPnl(t.pnl_pct)+'</td></tr>';
}}
function rowHist(t) {{
    return '<tr><td>'+t.ativo+'</td><td>'+sideBdg(t.side)+'</td><td>'+origemBdg(t.origem)+'</td><td>'+statusBdg(isTradeOpen(t))+'</td><td>'+fmtDate(t.data_insercao)+'</td><td>'+fmtDate(t.data_remocao)+'</td><td>'+(t.dias!=null?t.dias:'\\u2014')+'</td><td>$ '+fmtPrice(t.preco_entrada_turma)+'</td><td>'+fmtTotal(t.preco_entrada_total)+'</td><td>$ '+fmtPrice(t.preco_atual)+'</td><td>'+fmtTotal(t.preco_saida_total)+'</td><td>'+fmtStop(t.stop_atual)+'</td><td>'+fmtPnl(t.pnl_pct)+'</td></tr>';
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

// Fonte da verdade: backend (carteira_turma.ativo_atual) + datas e status da posição
function isTradeOpen(t) {{
    if(t.ativo_atual === 0 || t.ativo_atual === false) return false;
    if(t.data_remocao || t.data_saida) return false;
    if(String(t.status_posicao||'').toLowerCase() === 'closed') return false;
    return true;
}}
function renderAbertas(cart) {{
    var open=cart.filter(function(t){{return isTradeOpen(t);}}).sort(function(a,b){{return(b.data_insercao||'').localeCompare(a.data_insercao||'');}});
    var tb=document.getElementById('tbAbertas');
    document.getElementById('countAbertas').textContent = open.length;
    if(!open.length){{tb.innerHTML='<tr><td colspan="11" class="empty-msg">Nenhuma posi\\u00e7\\u00e3o aberta</td></tr>';return;}}
    tb.innerHTML=open.map(rowOpen).join('');
    setupSort(document.getElementById('tblAbertas'),open,[
        {{t:'text',v:function(r){{return r.ativo;}}}},{{t:'text',v:function(r){{return r.side;}}}},{{t:'text',v:function(r){{return r.origem;}}}},
        {{t:'date',v:function(r){{return r.data_insercao;}}}},{{t:'num',v:function(r){{return r.preco_entrada_turma;}}}},
        {{t:'num',v:function(r){{return r.quantidade;}}}},{{t:'num',v:function(r){{return r.preco_entrada_total;}}}},
        {{t:'num',v:function(r){{return r.preco_atual;}}}},{{t:'num',v:function(r){{return r.preco_saida_total;}}}},
        {{t:'num',v:function(r){{return r.stop_atual;}}}},{{t:'num',v:function(r){{return r.pnl_pct;}}}}
    ],rowOpen);
}}
function renderFechadas(cart) {{
    var closed=cart.filter(function(t){{return !isTradeOpen(t);}}).sort(function(a,b){{return(b.data_insercao||'').localeCompare(a.data_insercao||'');}});
    var tb=document.getElementById('tbFechadas');
    document.getElementById('countFechadas').textContent = closed.length;
    document.getElementById('countHistorico').textContent = cart.length;
    if(!closed.length){{tb.innerHTML='<tr><td colspan="12" class="empty-msg">Nenhum trade fechado</td></tr>';return;}}
    tb.innerHTML=closed.map(rowClosed).join('');
    setupSort(document.getElementById('tblFechadas'),closed,[
        {{t:'text',v:function(r){{return r.ativo;}}}},{{t:'text',v:function(r){{return r.side;}}}},{{t:'text',v:function(r){{return r.origem;}}}},
        {{t:'date',v:function(r){{return r.data_insercao;}}}},{{t:'date',v:function(r){{return r.data_remocao;}}}},{{t:'num',v:function(r){{return r.dias;}}}},
        {{t:'num',v:function(r){{return r.preco_entrada_turma;}}}},{{t:'num',v:function(r){{return r.preco_entrada_total;}}}},
        {{t:'num',v:function(r){{return r.preco_atual;}}}},{{t:'num',v:function(r){{return r.preco_saida_total;}}}},
        {{t:'num',v:function(r){{return r.stop_atual;}}}},{{t:'num',v:function(r){{return r.pnl_pct;}}}}
    ],rowClosed);
}}
function renderHistorico(cart) {{
    var all=[...cart].sort(function(a,b){{return(b.data_insercao||'').localeCompare(a.data_insercao||'');}});
    var tb=document.getElementById('tbHistorico');
    if(!all.length){{tb.innerHTML='<tr><td colspan="13" class="empty-msg">Nenhum trade</td></tr>';return;}}
    tb.innerHTML=all.map(rowHist).join('');
    setupSort(document.getElementById('tblHistorico'),all,[
        {{t:'text',v:function(r){{return r.ativo;}}}},{{t:'text',v:function(r){{return r.side;}}}},{{t:'text',v:function(r){{return r.origem;}}}},
        {{t:'text',v:function(r){{return isTradeOpen(r)?'a':'z';}}}},{{t:'date',v:function(r){{return r.data_insercao;}}}},
        {{t:'date',v:function(r){{return r.data_remocao;}}}},{{t:'num',v:function(r){{return r.dias;}}}},
        {{t:'num',v:function(r){{return r.preco_entrada_turma;}}}},{{t:'num',v:function(r){{return r.preco_entrada_total;}}}},
        {{t:'num',v:function(r){{return r.preco_atual;}}}},{{t:'num',v:function(r){{return r.preco_saida_total;}}}},
        {{t:'num',v:function(r){{return r.stop_atual;}}}},{{t:'num',v:function(r){{return r.pnl_pct;}}}}
    ],rowHist);
}}

function renderPnlAbertasChart(cart) {{
    var open = cart.filter(function(t){{ return isTradeOpen(t); }});
    if(pnlAbertasChart) pnlAbertasChart.destroy();
    var ctx = document.getElementById('chartPnlAbertas');
    if(!ctx) return;
    ctx = ctx.getContext('2d');
    if(!open.length) {{
        pnlAbertasChart = new Chart(ctx, {{ type: 'bar', data: {{ labels: [], datasets: [] }}, options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }}, scales: {{ x: {{ display: false }}, y: {{ display: false }} }} }} }});
        return;
    }}
    var labels = open.map(function(t){{ return t.ativo + ' (' + (t.side||'LONG').toUpperCase() + ')'; }});
    var values = open.map(function(t){{ return t.pnl_pct != null ? t.pnl_pct : 0; }});
    var colors = values.map(function(v){{ return v >= 0 ? '#39fda3' : '#e74c3c'; }});
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
                    titleColor: '#39fda3',
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
                    title: {{ display: true, text: 'PnL %', color: '#888' }}
                }}
            }}
        }}
    }});
}}

var activePnlPeriod = 'MAX';

function applyPnlFilter() {{
    var loadEl = document.getElementById('pnlChartLoading');
    var wrapEl = document.getElementById('pnlChartWrapper');
    if (loadEl) loadEl.style.display = 'none';
    if (wrapEl) wrapEl.style.display = 'block';
    renderPnlAbertasChart(CARTEIRA);
}}

function setPnlPeriod(period, btn) {{
    activePnlPeriod = period;
    document.querySelectorAll('.pnl-period-btn').forEach(function(b){{ b.classList.remove('active'); }});
    if (btn) btn.classList.add('active');
    applyPnlFilter();
}}

function setPnlCustomPeriod() {{
    activePnlPeriod = 'CUSTOM';
    document.querySelectorAll('.pnl-period-btn').forEach(function(b){{ b.classList.remove('active'); }});
    applyPnlFilter();
}}

var ALLOC_COLORS = ['#39fda3','#e74c3c','#3498db','#f39c12','#9b59b6','#1abc9c','#e67e22','#2ecc71','#e84393','#00cec9','#fd79a8','#6c5ce7','#ffeaa7','#dfe6e9','#fab1a0','#a29bfe'];

// Gráfico de evolução da alocação usa os mesmos dados e regras da tabela (CARTEIRA + isTradeOpen + ativo_atual do backend)
function renderAllocTimeline(cart, rentSerie) {{
    function toYMD(v) {{ return (v && String(v).slice) ? String(v).slice(0,10) : (v || ''); }}
    var today = new Date().toISOString().slice(0,10);
    var entries = (cart||[]).filter(function(t){{
        if(!t.data_insercao) return false;
        if(isTradeOpen(t)) return true;
        return !!(t.data_remocao || t.data_saida);
    }});
    if(!entries.length) return;

    var dateSet = {{}};
    entries.forEach(function(t){{
        var start = toYMD(t.data_insercao);
        var dataSaida = t.data_remocao || t.data_saida;
        var isOpen = isTradeOpen(t);
        var end = isOpen ? today : toYMD(dataSaida);
        if(!start || !end) return;
        dateSet[start] = 1;
        dateSet[end] = 1;
    }});
    if(DATA_INICIO_TURMA) dateSet[DATA_INICIO_TURMA] = 1;
    dateSet[today] = 1;
    var allDates = Object.keys(dateSet).sort();

    var minDate = allDates[0];
    var maxDate = allDates[allDates.length-1];
    var dates = [];
    var d = new Date(minDate + 'T00:00:00');
    var dMax = new Date(maxDate + 'T00:00:00');
    while(d <= dMax) {{
        dates.push(d.toISOString().slice(0,10));
        d.setDate(d.getDate() + 1);
    }}

    var assetMap = {{}};
    entries.forEach(function(t){{
        var key = t.ativo + '_' + (t.side||'long').toLowerCase();
        if(!assetMap[key]) assetMap[key] = [];
        assetMap[key].push(t);
    }});

    var assetKeys = Object.keys(assetMap).sort();

    var rawByAsset = {{}};
    assetKeys.forEach(function(key){{
        var trades = assetMap[key];
        rawByAsset[key] = dates.map(function(day){{
            var total = 0;
            trades.forEach(function(t){{
                var start = toYMD(t.data_insercao);
                var dataSaida = t.data_remocao || t.data_saida;
                var end = isTradeOpen(t) ? today : toYMD(dataSaida);
                if(!start || !end) return;
                if(day >= start && day <= end) {{
                    var qty = t.quantidade || 0;
                    var pe = t.preco_entrada_turma || 0;
                    total += qty * pe;
                }}
            }});
            return total;
        }});
    }});

    var dailyTotals = dates.map(function(_, i){{
        var s = 0;
        assetKeys.forEach(function(key){{ s += rawByAsset[key][i]; }});
        return s;
    }});

    var caixaByDate = {{}};
    if(rentSerie && Array.isArray(rentSerie) && rentSerie.length) {{
        rentSerie.forEach(function(p){{
            var dia = p.dia || (p.dia && p.dia.toISOString ? p.dia.toISOString().slice(0,10) : '');
            if(dia) caixaByDate[dia] = (p.capital_em_caixa != null) ? Number(p.capital_em_caixa) : 0;
        }});
    }}
    var caixaPerDay = dates.map(function(day){{ return caixaByDate[day] != null ? Math.max(0, caixaByDate[day]) : 0; }});
    var totalPerDay = dates.map(function(_, i){{ return dailyTotals[i] + caixaPerDay[i]; }});

    var datasets = [];
    var hasCaixa = (typeof MOSTRAR_CAIXA_ALOCACAO !== 'undefined' && MOSTRAR_CAIXA_ALOCACAO) && rentSerie && Array.isArray(rentSerie) && rentSerie.length && totalPerDay.some(function(t,i){{ return t > 0 && caixaPerDay[i] > 0; }});
    if(hasCaixa) {{
        var caixaColor = '#6c7a89';
        var caixaPct = dates.map(function(_, i){{
            var tot = totalPerDay[i];
            return tot > 0 ? Math.round((caixaPerDay[i] / tot) * 10000) / 100 : 0;
        }});
        datasets.push({{
            label: 'USDT',
            data: caixaPct,
            _rawData: caixaPerDay,
            backgroundColor: caixaColor + 'AA',
            borderColor: caixaColor,
            borderWidth: 1,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 3
        }});
    }}

    assetKeys.forEach(function(key, idx){{
        var parts = key.split('_');
        var ativo = parts[0];
        var side = parts[1] || 'long';
        var color = ALLOC_COLORS[idx % ALLOC_COLORS.length];
        var denom = hasCaixa ? totalPerDay : dailyTotals;
        var data = dates.map(function(_, i){{
            var t = denom[i];
            return t > 0 ? Math.round((rawByAsset[key][i] / t) * 10000) / 100 : 0;
        }});
        datasets.push({{
            label: ativo + ' (' + side.toUpperCase() + ')',
            data: data,
            _rawData: rawByAsset[key],
            backgroundColor: color + 'AA',
            borderColor: color,
            borderWidth: 1,
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 3
        }});
    }});

    if(allocTimelineChart) allocTimelineChart.destroy();
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
                    title: {{ display: true, text: 'Aloca\u00e7\u00e3o (%)', color: '#888' }}
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
                    titleColor: '#39fda3',
                    bodyColor: '#e0e0e0',
                    callbacks: {{
                        label: function(ctx) {{
                            if(ctx.raw === 0) return null;
                            var raw = ctx.dataset._rawData ? ctx.dataset._rawData[ctx.dataIndex] : 0;
                            var dollar = raw ? '$ ' + raw.toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:2}}) : '';
                            return ctx.dataset.label + ': ' + ctx.raw.toFixed(1) + '%' + (dollar ? ' (' + dollar + ')' : '');
                        }}
                    }}
                }}
            }}
        }}
    }});
}}

function buildRentDatasets(serie, btc, compare) {{
    var datasets = [];
    var ctx = document.getElementById('chartRent').getContext('2d');
    var gradient = ctx.createLinearGradient(0,0,0,350);
    gradient.addColorStop(0,'rgba(78,204,163,0.3)');
    gradient.addColorStop(1,'rgba(78,204,163,0.0)');
    var labels = serie.map(function(s){{return s.dia;}});
    var data = serie.map(function(s){{return s.rentabilidade_acumulada_pct;}});
    var lastVal = data[data.length-1]||0;
    var turmaInfo = TURMAS.find(function(t){{return t.id === TURMA_ID;}});
    var turmaLabel = turmaInfo ? turmaInfo.nome : 'Turma';
    datasets.push({{
        label: turmaLabel,
        data: data,
        borderColor: lastVal>=0 ? '#39fda3' : '#e74c3c',
        backgroundColor: gradient,
        fill: true, tension: 0.3, pointRadius: 0,
        pointHoverRadius: 5, borderWidth: 2,
        hidden: !document.getElementById('toggleTurma').checked
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
    cmpKeys.forEach(function(tid, idx) {{
        var cmp = compare[tid];
        if (!cmp || !cmp.length) return;
        var cmpMap = {{}};
        cmp.forEach(function(s){{cmpMap[s.dia]=s.rentabilidade_acumulada_pct;}});
        var cmpData = labels.map(function(d){{return cmpMap[d] !== undefined ? cmpMap[d] : null;}});
        var cmpInfo = TURMAS.find(function(t){{return String(t.id) === String(tid);}});
        var cmpLabel = cmpInfo ? cmpInfo.nome : 'Turma '+tid;
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

function renderRentChartFiltered(serie, btc, compareObj) {{
    var built = buildRentDatasets(serie, btc, compareObj || {{}});
    if (!built || !built.labels || !built.labels.length) return;
    if(rentChart) rentChart.destroy();
    rentChart=new Chart(document.getElementById('chartRent').getContext('2d'),{{
        type:'line',
        data: built,
        options:{{ responsive:true, maintainAspectRatio:false, interaction:{{mode:'index', intersect:false}}, plugins:{{ legend:{{display:false}}, tooltip:{{ backgroundColor:'rgba(26,26,46,0.95)', titleColor:'#39fda3', bodyColor:'#e0e0e0', borderColor:'#39fda3', borderWidth:1, callbacks:{{ label:function(c){{ if(c.raw==null)return null; return c.dataset.label+': '+(c.raw>=0?'+':'')+c.raw.toFixed(2)+'%'; }} }} }} }}, scales:{{ x:{{ ticks:{{ color:'#666', maxTicksLimit:20, maxRotation:0 }}, grid:{{ color:'rgba(255,255,255,0.05)' }} }}, y:{{ ticks:{{ color:'#666', callback:function(v){{ return v+'%'; }} }}, grid:{{ color:'rgba(255,255,255,0.05)' }}, title:{{ display:true, text:'Rentabilidade Acumulada (%)', color:'#888' }} }} }} }}
    }});
}}
function renderRentChart(serie) {{
    document.getElementById('chartLoading').style.display='none';
    document.getElementById('chartWrapper').style.display='block';
    fullRentSerie = serie;
    currentTurmaSerie = serie;
    if (activeRentPeriod !== 'MAX' || rentPeriodStartDate || rentPeriodEndDate) {{
        applyRentPeriodFilter();
        return;
    }}
    var built = buildRentDatasets(serie, btcSerie, compareSeries);
    if(rentChart) rentChart.destroy();
    rentChart=new Chart(document.getElementById('chartRent').getContext('2d'),{{
        type:'line',
        data: built,
        options:{{
            responsive:true, maintainAspectRatio:false,
            interaction:{{mode:'index', intersect:false}},
            plugins:{{
                legend:{{display:false}},
                tooltip:{{
                    backgroundColor:'rgba(26,26,46,0.95)', titleColor:'#39fda3',
                    bodyColor:'#e0e0e0', borderColor:'#39fda3', borderWidth:1,
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
    if(!currentTurmaSerie) return;
    if (activeRentPeriod !== 'MAX' || rentPeriodStartDate || rentPeriodEndDate) applyRentPeriodFilter();
    else renderRentChartFiltered(currentTurmaSerie, btcSerie, compareSeries);
}}

function fetchBtcBenchmark(dataInicio) {{
    fetch('/api/benchmark/btc?data_inicio='+dataInicio)
        .then(function(r){{return r.json();}})
        .then(function(data){{
            if(Array.isArray(data)) btcSerie = data;
            else btcSerie = null;
            rebuildChart();
        }})
        .catch(function(){{ btcSerie = null; }});
}}

function populateCompareCheckboxes() {{
    var container = document.getElementById('compareCheckboxes');
    if (!container) return;
    container.innerHTML = '';
    TURMAS.forEach(function(t) {{
        if (t.id === TURMA_ID) return;
        var label = document.createElement('label');
        label.style.cssText = 'display:inline-flex; align-items:center; gap:5px; cursor:pointer; color:#e0e0e0; font-size:13px;';
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.dataset.turmaId = t.id;
        cb.style.cssText = 'accent-color:#e056fd; width:15px; height:15px; cursor:pointer;';
        if (compareSeries[t.id]) cb.checked = true;
        cb.addEventListener('change', function() {{ onCompareCheckboxChange(t.id, cb.checked); }});
        label.appendChild(cb);
        label.appendChild(document.createTextNode(t.nome));
        container.appendChild(label);
    }});
}}
function onCompareCheckboxChange(turmaId, checked) {{
    if (checked) {{
        fetch('/api/turma/' + turmaId + '/rentabilidade').then(function(r){{ return r.json(); }}).then(function(data){{
            if (Array.isArray(data) && data.length) {{ compareSeries[turmaId] = data; rebuildChart(); }}
        }}).catch(function(){{ rebuildChart(); }});
    }} else {{
        delete compareSeries[turmaId];
        rebuildChart();
    }}
}}

function switchPosTab(tab, btn) {{
    document.querySelectorAll('#sectionPositions .sub-tab').forEach(function(b){{b.classList.remove('active');}});
    document.querySelectorAll('#sectionPositions .sub-content').forEach(function(c){{c.classList.remove('active');}});
    btn.classList.add('active');
    document.getElementById('tab-'+tab).classList.add('active');
}}

async function switchTurma(turmaId, btn) {{
    if(turmaId===TURMA_ID) return;
    TURMA_ID=turmaId;
    document.querySelectorAll('.turma-tab').forEach(function(t){{t.classList.remove('active');}});
    btn.classList.add('active');
    document.getElementById('chartLoading').style.display='block';
    document.getElementById('chartLoading').innerHTML='<div class="spinner"></div>Carregando...';
    document.getElementById('chartWrapper').style.display='none';
    var turmaInfo = TURMAS.find(function(t){{return t.id === turmaId;}});
    if(turmaInfo) {{
        DATA_INICIO_TURMA = turmaInfo.data_inicio;
        document.getElementById('toggleTurmaLabel').textContent = turmaInfo.nome;
    }}
    compareSeries = {{}};
    populateCompareCheckboxes();
    try {{
        var results=await Promise.all([
            fetch('/api/turma/'+turmaId+'/dashboard-data'),
            fetch('/api/turma/'+turmaId+'/rentabilidade')
        ]);
        var dashData=await results[0].json();
        var serieData=await results[1].json();
        document.getElementById('chartLoading').style.display='none';
        document.getElementById('chartWrapper').style.display='block';
        if(dashData.erro){{console.error(dashData.erro);document.getElementById('chartWrapper').innerHTML='<div style="padding:40px; text-align:center; color:#e74c3c;">'+dashData.erro+'</div>';return;}}
        RESUMO=dashData.resumo;
        CARTEIRA=dashData.carteira;
        updateCards(RESUMO);
        applyPnlFilter();
        renderAbertas(CARTEIRA);
        renderFechadas(CARTEIRA);
        renderHistorico(CARTEIRA);
        renderAllocTimeline(CARTEIRA, serieData);
        if(Array.isArray(serieData)&&serieData.length>0){{
            renderRentChart(serieData);
            fetchBtcBenchmark(DATA_INICIO_TURMA);
        }}
        else{{ document.getElementById('chartWrapper').innerHTML='<div style="padding:40px; text-align:center; color:#666;">Sem dados de rentabilidade</div>'; }}
    }} catch(err) {{
        console.error('Erro:',err);
        document.getElementById('chartLoading').style.display='none';
        document.getElementById('chartWrapper').style.display='block';
        document.getElementById('chartWrapper').innerHTML='<div style="padding:40px; text-align:center; color:#e74c3c;">Erro: '+err.message+'</div>';
    }}
}}

// Atualizar Cotações (turma/rentabilidade)
async function atualizarCotacoes() {{
    var btn=document.getElementById('btnAtualizar');
    var status=document.getElementById('statusMsg');
    btn.disabled=true; btn.textContent='Atualizando...';
    status.textContent='Buscando pre\\u00e7os...'; status.style.color='#f39c12';
    try {{
        var response=await fetch('/api/cotacoes/atualizar');
        var data=await response.json();
        if(data.sucesso){{
            status.textContent='\\u2713 '+data.dias_preenchidos+' dias preenchidos';
            status.style.color='#39fda3';
            setTimeout(function(){{window.location.reload();}},1500);
        }} else {{
            status.textContent='\\u2717 '+data.erro; status.style.color='#e74c3c';
        }}
    }} catch(err) {{
        status.textContent='\\u2717 '+err.message; status.style.color='#e74c3c';
    }}
    btn.disabled=false; btn.textContent='Atualizar Pre\\u00e7os';
}}

// Preencher Preços (produto/posições)
document.addEventListener('click',function(e){{
    document.querySelectorAll('.dropdown-precos-menu.show').forEach(function(m){{
        if(!m.parentElement.contains(e.target)) m.classList.remove('show');
    }});
}});
function preencherPrecos(produtoId, tipo) {{
    document.querySelectorAll('.dropdown-precos-menu.show').forEach(function(m){{m.classList.remove('show');}});
    var nomes={{entrada:'pre\\u00e7os de entrada',saida:'pre\\u00e7os de sa\\u00edda',ambos:'todos os pre\\u00e7os'}};
    if(!confirm('Preencher '+nomes[tipo]+' faltantes?')) return false;
    var status=document.getElementById('statusMsg');
    status.textContent='Preenchendo '+nomes[tipo]+'...'; status.style.color='#f39c12';
    fetch('/api/posicoes/preencher-precos?produto_id='+produtoId+'&tipo='+tipo)
        .then(function(r){{return r.json();}})
        .then(function(res){{
            if(res.sucesso){{status.textContent='\\u2713 '+res.mensagem;status.style.color='#39fda3';}}
            else{{status.textContent='\\u2717 '+(res.erro||'Falha');status.style.color='#e74c3c';}}
            setTimeout(function(){{status.textContent='';}},8000);
        }})
        .catch(function(e){{
            status.textContent='\\u2717 '+e.message;status.style.color='#e74c3c';
            setTimeout(function(){{status.textContent='';}},8000);
        }});
    return false;
}}

document.addEventListener('DOMContentLoaded', function() {{
    console.log('[Dashboard] TURMA_ID='+TURMA_ID+' CARTEIRA.length='+(Array.isArray(CARTEIRA)?CARTEIRA.length:'N/A'));
    try {{ applyPnlFilter(); }} catch(e) {{ console.error('applyPnlFilter:', e); }}
    try {{ renderAbertas(CARTEIRA); }} catch(e) {{ console.error('renderAbertas:', e); }}
    try {{ renderFechadas(CARTEIRA); }} catch(e) {{ console.error('renderFechadas:', e); }}
    try {{ renderHistorico(CARTEIRA); }} catch(e) {{ console.error('renderHistorico:', e); }}
    try {{ renderAllocTimeline(CARTEIRA); }} catch(e) {{ console.error('renderAllocTimeline:', e); }}
    try {{ populateCompareCheckboxes(); }} catch(e) {{ console.error('populateCompareCheckboxes:', e); }}

    try {{
        document.getElementById('toggleTurma').addEventListener('change', function(){{ rebuildChart(); }});
        document.getElementById('toggleBTC').addEventListener('change', function(){{ rebuildChart(); }});
    }} catch(e) {{ console.error('toggles:', e); }}

    if(TURMA_ID){{
        var loadEl = document.getElementById('chartLoading');
        var wrapEl = document.getElementById('chartWrapper');
        var timeoutMs = 60000;
        console.log('[Dashboard] Fetching /api/turma/'+TURMA_ID+'/rentabilidade ...');
        var timeoutPromise = new Promise(function(_, reject){{ setTimeout(function(){{ reject(new Error('Tempo esgotado (60s) ao carregar dados de rentabilidade.')); }}, timeoutMs); }});
        Promise.race([
            fetch('/api/turma/'+TURMA_ID+'/rentabilidade').then(function(r){{
                console.log('[Dashboard] Rentabilidade response status='+r.status);
                if(!r.ok) throw new Error('HTTP '+r.status);
                return r.json();
            }}),
            timeoutPromise
        ])
            .then(function(data){{
                loadEl.style.display = 'none';
                wrapEl.style.display = 'block';
                console.log('[Dashboard] Rentabilidade data tipo='+(Array.isArray(data)?'array['+data.length+']':typeof data));
                if(data && data.erro){{
                    wrapEl.innerHTML = '<div style="padding:40px; text-align:center; color:#e74c3c;">Erro: '+String(data.erro)+'</div>';
                    return;
                }}
                if(Array.isArray(data)&&data.length>0){{
                    if(typeof Chart === 'undefined'){{
                        wrapEl.innerHTML = '<div style="padding:40px; text-align:center; color:#e74c3c;">Erro: biblioteca de gr&aacute;ficos n&atilde;o carregou. Recarregue a p&aacute;gina.</div>';
                    }} else {{
                        try {{
                            renderRentChart(data);
                            renderAllocTimeline(CARTEIRA, data);
                            fetchBtcBenchmark(DATA_INICIO_TURMA);
                        }} catch(chartErr) {{
                            console.error('[Dashboard] Erro ao renderizar charts:', chartErr);
                            wrapEl.innerHTML = '<div style="padding:40px; text-align:center; color:#e74c3c;">Erro ao renderizar: '+chartErr.message+'</div>';
                        }}
                    }}
                }}
                else{{ wrapEl.innerHTML = '<div style="padding:40px; text-align:center; color:#666;">Sem dados de rentabilidade</div>'; }}
            }})
            .catch(function(err){{
                console.error('[Dashboard] Rentabilidade fetch error:', err);
                loadEl.style.display = 'none';
                wrapEl.style.display = 'block';
                wrapEl.innerHTML = '<div style="padding:40px; text-align:center; color:#e74c3c;">Erro: '+err.message+'</div>';
            }});
    }} else {{
        console.warn('[Dashboard] TURMA_ID is falsy, skipping rentabilidade fetch');
        document.getElementById('chartLoading').style.display = 'none';
        document.getElementById('chartWrapper').style.display = 'block';
        document.getElementById('chartWrapper').innerHTML = '<div style="padding:40px; text-align:center; color:#666;">Nenhuma turma selecionada</div>';
    }}
}});
</script>
{_get_form_modal_overlay_script(produto_id)}
</body>
</html>"""
