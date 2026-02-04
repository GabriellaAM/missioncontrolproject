"""
Templates e funções para as telas de Turmas no Dashboard.

Este módulo contém os templates HTML e funções de renderização
para as páginas de gestão de turmas e rentabilidade.
"""

from datetime import datetime


def _get_shared_components():
    """Lazy import to avoid circular dependency with servidor_dashboard"""
    from servidor_dashboard import get_navbar, get_turmas_subnav, get_base_styles
    return get_navbar, get_turmas_subnav, get_base_styles


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
            color: #4ecca3;
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
        .stat-value.positive { color: #4ecca3; }
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
            background: #4ecca3;
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
            color: #4ecca3;
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
            color: #4ecca3;
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
    get_navbar, get_turmas_subnav, get_base_styles = _get_shared_components()
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
            background: #4ecca3;
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
            color: #4ecca3;
            border-bottom-color: #4ecca3;
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
            color: #4ecca3;
            border-color: #4ecca3;
        }}
        {styles}
    </style>
</head>
<body>
    {get_navbar('turmas')}
    {get_turmas_subnav('turmas')}

    <div class="page-header">
        <h2>Turmas ({len(turmas)})</h2>
        <a href="/turmas/nova" class="btn-criar">+ Nova Turma</a>
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
</body>
</html>"""


def get_form_nova_turma_html(produtos):
    """Gera HTML do formulário para criar nova turma"""
    get_navbar, get_turmas_subnav, get_base_styles = _get_shared_components()

    options_html = ""
    for p in produtos:
        options_html += f'<option value="{p["id"]}">{p["nome"]}</option>'

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Nova Turma - Dashboard</title>
    <style>
        {get_base_styles()}
        .form-container {{
            max-width: 600px;
            margin: 40px auto;
            padding: 30px;
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
        }}
        .form-container h2 {{
            color: #4ecca3;
            margin-bottom: 25px;
        }}
        .form-group {{
            margin-bottom: 20px;
        }}
        .form-group label {{
            display: block;
            color: #888;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        .form-group input, .form-group select, .form-group textarea {{
            width: 100%;
            padding: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 5px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 1em;
        }}
        .form-group input:focus, .form-group select:focus {{
            outline: none;
            border-color: #4ecca3;
        }}
        .btn-submit {{
            width: 100%;
            padding: 15px;
            background: #4ecca3;
            color: #1a1a2e;
            border: none;
            border-radius: 5px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .btn-submit:hover {{ background: #3db892; }}
        .btn-cancel {{
            display: block;
            text-align: center;
            margin-top: 15px;
            color: #888;
            text-decoration: none;
        }}
        .btn-cancel:hover {{ color: #fff; }}
        .alert {{
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }}
        .alert-error {{ background: rgba(231, 76, 60, 0.2); color: #e74c3c; }}
        .alert-success {{ background: rgba(78, 204, 163, 0.2); color: #4ecca3; }}
        .posicoes-elegiveis {{
            margin-top: 20px;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            border: 1px solid rgba(78, 204, 163, 0.3);
        }}
        .posicoes-elegiveis h3 {{
            color: #4ecca3;
            font-size: 0.95em;
            margin-bottom: 8px;
        }}
        .posicoes-elegiveis p.hint {{
            color: #888;
            font-size: 0.85em;
            margin-bottom: 12px;
        }}
        .posicoes-elegiveis table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        .posicoes-elegiveis th, .posicoes-elegiveis td {{
            padding: 8px 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }}
        .posicoes-elegiveis th {{
            color: #888;
            font-weight: 600;
        }}
        .posicoes-elegiveis input[type="checkbox"] {{
            accent-color: #4ecca3;
            cursor: pointer;
        }}
        .posicoes-elegiveis input[type="date"] {{
            padding: 6px 8px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 4px;
            color: #fff;
            font-size: 0.85em;
            min-width: 130px;
        }}
        .posicoes-elegiveis .btn-aplicar-data {{
            margin-bottom: 10px;
            padding: 6px 12px;
            font-size: 0.85em;
            background: rgba(78, 204, 163, 0.25);
            color: #4ecca3;
            border: 1px solid #4ecca3;
            border-radius: 5px;
            cursor: pointer;
        }}
        .posicoes-elegiveis .btn-aplicar-data:hover {{
            background: rgba(78, 204, 163, 0.4);
        }}
        .posicoes-elegiveis .loading {{
            color: #888;
            padding: 10px 0;
        }}
        .posicoes-elegiveis .empty {{
            color: #888;
            padding: 10px 0;
        }}
        .posicoes-elegiveis .select-first {{
            color: #666;
            font-style: italic;
        }}
    </style>
</head>
<body>
    {get_navbar('turmas')}
    {get_turmas_subnav('turmas')}

    <div class="form-container">
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

            <div class="form-group" id="posicoes-elegiveis-container">
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

            <div class="form-group">
                <label>Descrição</label>
                <textarea name="descricao" rows="3" placeholder="Descrição opcional..."></textarea>
            </div>

            <button type="submit" class="btn-submit">Criar Turma</button>
            <a href="/turmas" class="btn-cancel">Cancelar</a>
        </form>
    </div>

    <script>
        function showAlert(message, type) {{
            const alert = document.getElementById('alert');
            alert.className = 'alert alert-' + type;
            alert.textContent = message;
            alert.style.display = 'block';
        }}

        async function carregarPosicoesElegiveis() {{
            const produtoId = document.getElementById('produto_id').value;
            const dataInicio = document.getElementById('data_inicio').value;
            const container = document.getElementById('posicoes-elegiveis');

            if (!produtoId || !dataInicio) {{
                container.innerHTML = '<span class="select-first">Selecione o produto e a data de início para carregar as posições elegíveis.</span>';
                return;
            }}

            container.innerHTML = '<span class="loading">Carregando posições...</span>';
            try {{
                const response = await fetch(`/api/turma/posicoes-elegiveis?produto_id=${{produtoId}}&data_inicio=${{dataInicio}}`);
                const result = await response.json();
                if (!response.ok) {{
                    container.innerHTML = '<span class="empty">Erro ao carregar: ' + (result.erro || response.status) + '</span>';
                    return;
                }}
                const posicoes = result.posicoes || [];
                if (posicoes.length === 0) {{
                    container.innerHTML = '<span class="empty">Nenhuma posição aberta elegível para esta data (posições com data de entrada anterior à data de início).</span>';
                    return;
                }}
                let html = '<button type="button" class="btn-aplicar-data" onclick="aplicarDataInicioTodas()">Usar data de início em todas</button>';
                html += '<table><thead><tr><th></th><th>Ativo</th><th>Side</th><th>Data entrada</th><th>Data inserção na turma</th><th>Preço entrada</th><th>Qtd</th></tr></thead><tbody>';
                for (const p of posicoes) {{
                    const qtd = p.quantidade != null ? Number(p.quantidade) : '—';
                    const preco = p.preco_entrada != null ? Number(p.preco_entrada).toFixed(4) : '—';
                    html += '<tr><td><input type="checkbox" name="posicao_sel" value="' + p.id + '" data-posicao-id="' + p.id + '" checked></td>';
                    html += '<td>' + (p.ativo || '—') + '</td><td>' + (p.side || '—') + '</td><td>' + (p.data_entrada || '—') + '</td>';
                    html += '<td><input type="date" class="data-insercao-input" data-posicao-id="' + p.id + '" value="' + dataInicio + '"></td>';
                    html += '<td>' + preco + '</td><td>' + qtd + '</td></tr>';
                }}
                html += '</tbody></table>';
                container.innerHTML = html;
            }} catch (err) {{
                container.innerHTML = '<span class="empty">Erro de conexão: ' + err.message + '</span>';
            }}
        }}

        function aplicarDataInicioTodas() {{
            const dataInicio = document.getElementById('data_inicio').value;
            if (!dataInicio) return;
            document.querySelectorAll('.data-insercao-input').forEach(function(inp) {{ inp.value = dataInicio; }});
        }}

        document.getElementById('produto_id').addEventListener('change', carregarPosicoesElegiveis);
        document.getElementById('data_inicio').addEventListener('change', carregarPosicoesElegiveis);

        async function criarTurma(e) {{
            e.preventDefault();
            const form = e.target;
            const formData = new FormData(form);
            const dataInicio = formData.get('data_inicio');

            const checkboxes = form.querySelectorAll('input[name="posicao_sel"]:checked');
            const posicoesConfig = Array.from(checkboxes).map(function(cb) {{
                const row = cb.closest('tr');
                const dateInput = row ? row.querySelector('.data-insercao-input') : null;
                const dataInsercao = dateInput && dateInput.value ? dateInput.value : dataInicio;
                return {{
                    posicao_id: parseInt(cb.getAttribute('data-posicao-id'), 10),
                    data_insercao: dataInsercao
                }};
            }});

            if (posicoesConfig.length === 0) {{
                const total = form.querySelectorAll('input[name="posicao_sel"]').length;
                if (total === 0) {{
                    showAlert('Selecione produto e data de início e aguarde carregar as posições, ou inclua ao menos uma posição.', 'error');
                }} else {{
                    showAlert('Marque ao menos uma posição para replicar na turma.', 'error');
                }}
                return;
            }}

            const data = {{
                produto_id: parseInt(formData.get('produto_id')),
                nome: formData.get('nome'),
                data_inicio: dataInicio,
                capital_base: parseFloat(formData.get('capital_base') || 1500),
                descricao: formData.get('descricao') || null,
                posicoes_config: posicoesConfig,
                auto_fetch_prices: true
            }};

            try {{
                const response = await fetch('/api/turma/criar', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(data)
                }});

                const result = await response.json();

                if (result.sucesso) {{
                    showAlert('Turma criada com sucesso!', 'success');
                    setTimeout(() => window.location.href = '/turmas/' + result.turma_id, 1500);
                }} else {{
                    showAlert('Erro: ' + result.erro, 'error');
                }}
            }} catch (err) {{
                showAlert('Erro de conexão: ' + err.message, 'error');
            }}
        }}
    </script>
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
    color = '#4ecca3' if pnl_pct >= 0 else '#e74c3c'
    sinal = '+' if pnl_pct >= 0 else ''
    return f'<span style="color: {color}; font-weight: bold;">{sinal}{pnl_pct:.2f}%</span>'


def _side_badge(side):
    """Gera badge de side."""
    side = (side or '').upper()
    if side == 'SHORT':
        return '<span style="background: rgba(231,76,60,0.2); color: #e74c3c; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">SHORT</span>'
    return '<span style="background: rgba(78,204,163,0.2); color: #4ecca3; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">LONG</span>'


def _origem_badge(origem):
    """Gera badge de origem."""
    if origem == 'nativo':
        return '<span style="color: #4ecca3;">Nativo</span>'
    return '<span style="color: #f39c12;">Replicado</span>'


def _status_badge(ativo_atual):
    """Gera badge de status."""
    if ativo_atual:
        return '<span style="color: #4ecca3;">Ativo</span>'
    return '<span style="color: #888;">Fechado</span>'


def _render_tab_abertas(carteira, turma_id):
    """Renderiza tabela da aba Abertas."""
    trades = [t for t in carteira if t.get('ativo_atual')]
    if not trades:
        return '<tr><td colspan="8" style="text-align: center; color: #888;">Nenhum trade ativo</td></tr>'
    html = ""
    for t in trades:
        qtd = t.get('quantidade')
        qtd_str = f"{float(qtd):,.4f}" if qtd is not None else '—'
        html += f"""
        <tr>
            <td style="text-align: left;">{t.get('ativo', 'N/A')}</td>
            <td>{_side_badge(t.get('side'))}</td>
            <td>{_origem_badge(t.get('origem'))}</td>
            <td>{t.get('data_insercao', 'N/A')}</td>
            <td>{_format_preco(t.get('preco_entrada_turma'))}</td>
            <td>{qtd_str}</td>
            <td>{_format_preco(t.get('preco_atual'))}</td>
            <td>{_format_pnl(t.get('pnl_pct'))}</td>
        </tr>
        """
    return html


def _render_tab_fechadas(carteira, turma_id):
    """Renderiza tabela da aba Fechadas."""
    trades = [t for t in carteira if not t.get('ativo_atual')]
    if not trades:
        return '<tr><td colspan="9" style="text-align: center; color: #888;">Nenhum trade fechado</td></tr>'
    html = ""
    for t in trades:
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
            <td>{_format_pnl(t.get('pnl_pct'))}</td>
        </tr>
        """
    return html


def _render_tab_historico(carteira, turma_id):
    """Renderiza tabela da aba Histórico."""
    if not carteira:
        return '<tr><td colspan="10" style="text-align: center; color: #888;">Nenhum trade na carteira</td></tr>'
    html = ""
    for t in carteira:
        preco_saida_atual = _format_preco(t.get('preco_atual'))
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
            <td>{_format_pnl(t.get('pnl_pct'))}</td>
        </tr>
        """
    return html


def get_turma_detalhes_html(turma, resumo, carteira, tab_ativa='abertas'):
    """Gera HTML da página de detalhes de uma turma com abas Abertas/Fechadas/Histórico"""
    get_navbar, get_turmas_subnav, get_base_styles = _get_shared_components()

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
        .rentab-big .value.positive {{ color: #4ecca3; }}
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
            color: #4ecca3;
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
            color: #4ecca3;
        }}
        .carteira-table tr:hover {{
            background: rgba(78, 204, 163, 0.1);
        }}
        .btn-chart {{
            display: inline-block;
            padding: 12px 24px;
            background: #4ecca3;
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
        .tab:hover, .tab.active {{ border-color: #4ecca3; color: #4ecca3; }}
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
    get_navbar, get_turmas_subnav, get_base_styles = _get_shared_components()

    labels = [p.dia for p in serie]
    valores = [p.valor_total for p in serie]
    rentab = [p.rentabilidade_acumulada_pct for p in serie]

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Rentabilidade - {turma.get('nome', 'Turma')}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            color: #4ecca3;
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
                    borderColor: '#4ecca3',
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
                    borderColor: rentab[rentab.length-1] >= 0 ? '#4ecca3' : '#e74c3c',
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
    get_navbar, get_turmas_subnav, get_base_styles = _get_shared_components()

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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            color: #4ecca3;
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
            color: #4ecca3;
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
        .positive {{ color: #4ecca3; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}

        .chart-card {{
            background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
            border-radius: 10px;
            padding: 25px;
            margin-top: 20px;
        }}
        .chart-card h3 {{
            color: #4ecca3;
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
        .info-item .value.positive {{ color: #4ecca3; }}
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
                <button onclick="atualizarCotacoes()" id="btnAtualizar" style="background: #4ecca3; color: #1a1a2e; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; white-space: nowrap;">
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
                status.style.color = '#4ecca3';
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
                    status.style.color = '#4ecca3';
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
                        borderColor: ultimaRentab >= 0 ? '#4ecca3' : '#e74c3c',
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
    get_navbar, get_turmas_subnav, get_base_styles = _get_shared_components()

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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            color: #4ecca3;
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
            color: #4ecca3;
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
            '#4ecca3', '#e74c3c', '#3498db', '#f39c12', '#9b59b6',
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
