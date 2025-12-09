"""
Funções de visualização para notebooks Jupyter
Usa Plotly para gráficos interativos
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def grafico_valores(df, titulo="Evolução do Preço", ativo=None):
    """
    Gera gráfico de linha dos valores diários
    
    Args:
        df: DataFrame com colunas 'data' e 'preco'
        titulo: Título do gráfico
        ativo: Nome do ativo (para título automático)
    
    Returns:
        plotly.graph_objects.Figure: Figura do Plotly
    """
    if df.empty:
        print("⚠️  Nenhum dado para exibir")
        return None
    
    if 'data' not in df.columns or 'preco' not in df.columns:
        print("⚠️  DataFrame deve conter colunas 'data' e 'preco'")
        return None
    
    # Converter data para datetime se necessário
    df_plot = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_plot['data']):
        df_plot['data'] = pd.to_datetime(df_plot['data'])
    
    df_plot = df_plot.sort_values('data')
    
    if ativo:
        titulo = f"Evolução do Preço - {ativo}"
    
    fig = px.line(
        df_plot, 
        x="data", 
        y="preco", 
        title=titulo,
        labels={'data': 'Data', 'preco': 'Preço (USD)'},
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def grafico_carteira(produto_id, mostrar_pnl=True):
    """
    Gera gráfico de barras mostrando composição da carteira
    
    Args:
        produto_id: ID do produto
        mostrar_pnl: Se True, inclui PnL não realizado
    
    Returns:
        plotly.graph_objects.Figure: Figura do Plotly
    """
    from analytics.queries import carteira_do_produto
    
    df = carteira_do_produto(produto_id)
    
    if df.empty:
        print("⚠️  Carteira não encontrada")
        return None
    
    carteira = df.iloc[0]
    
    # Preparar dados para o gráfico
    labels = []
    values = []
    colors = []
    
    if carteira['valor_disponivel'] > 0:
        labels.append('Disponível')
        values.append(carteira['valor_disponivel'])
        colors.append('#2ecc71')  # Verde
    
    if carteira['valor_investido'] > 0:
        labels.append('Investido')
        values.append(carteira['valor_investido'])
        colors.append('#3498db')  # Azul
    
    if mostrar_pnl and carteira['pnl_nao_realizado'] != 0:
        labels.append('PnL Não Realizado')
        values.append(abs(carteira['pnl_nao_realizado']))
        colors.append('#e74c3c' if carteira['pnl_nao_realizado'] < 0 else '#f39c12')  # Vermelho ou Laranja
    
    if not labels:
        print("⚠️  Nenhum valor para exibir")
        return None
    
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"${v:,.2f}" for v in values],
        textposition='auto',
    )])
    
    fig.update_layout(
        title=f"Composição da Carteira - Produto {produto_id}",
        xaxis_title="",
        yaxis_title="Valor (USD)",
        template='plotly_white'
    )
    
    return fig

def grafico_alocacoes(produto_id):
    """
    Gera gráfico de pizza mostrando distribuição de alocações
    
    Args:
        produto_id: ID do produto
    
    Returns:
        plotly.graph_objects.Figure: Figura do Plotly
    """
    from analytics.queries import resumo_alocacoes
    
    df = resumo_alocacoes(produto_id)
    
    if df.empty:
        print("⚠️  Nenhuma alocação encontrada")
        return None
    
    # Criar labels com ativo e side
    df['label'] = df.apply(
        lambda row: f"{row['ativo']} ({row['side']})", 
        axis=1
    )
    
    fig = px.pie(
        df,
        values='valor_usd',
        names='label',
        title=f"Distribuição de Alocações - Produto {produto_id}",
        hole=0.3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Valor: $%{value:,.2f}<br>Percentual: %{percent}<extra></extra>'
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig

def grafico_comparativo_ativos(ativos, data_inicio=None, data_fim=None):
    """
    Gera gráfico comparando evolução de múltiplos ativos
    
    Args:
        ativos: Lista de nomes de ativos (ex: ["BTC", "ETH"])
        data_inicio: Data inicial (YYYY-MM-DD) - opcional
        data_fim: Data final (YYYY-MM-DD) - opcional
    
    Returns:
        plotly.graph_objects.Figure: Figura do Plotly
    """
    from analytics.queries import valores_do_ativo
    
    fig = go.Figure()
    
    for ativo in ativos:
        df = valores_do_ativo(ativo, data_inicio, data_fim)
        
        if df.empty:
            print(f"⚠️  Nenhum dado encontrado para {ativo}")
            continue
        
        # Converter data para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(df['data']):
            df['data'] = pd.to_datetime(df['data'])
        
        df = df.sort_values('data')
        
        fig.add_trace(go.Scatter(
            x=df['data'],
            y=df['preco'],
            mode='lines+markers',
            name=ativo,
            hovertemplate=f'<b>{ativo}</b><br>Data: %{{x}}<br>Preço: $%{{y:,.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Comparativo de Ativos",
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def dashboard_produto(produto_id):
    """
    Cria um dashboard completo para um produto
    
    Args:
        produto_id: ID do produto
    
    Returns:
        plotly.graph_objects.Figure: Figura do Plotly com subplots
    """
    from analytics.queries import resumo_alocacoes, posicoes_abertas
    
    # Criar subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Composição da Carteira', 'Distribuição de Alocações', 
                       'Evolução de Preços', 'Posições Abertas'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "scatter", "colspan": 2}, None]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Composição da Carteira (barra)
    from analytics.queries import carteira_do_produto
    df_carteira = carteira_do_produto(produto_id)
    if not df_carteira.empty:
        carteira = df_carteira.iloc[0]
        labels = []
        values = []
        for label, value in [
            ('Disponível', carteira['valor_disponivel']),
            ('Investido', carteira['valor_investido']),
            ('PnL', carteira['pnl_nao_realizado'])
        ]:
            if value and value != 0:
                labels.append(label)
                values.append(abs(value))
        
        if labels:
            fig.add_trace(
                go.Bar(x=labels, y=values, name="Carteira", showlegend=False),
                row=1, col=1
            )
    
    # 2. Distribuição de Alocações (pizza)
    df_alocacoes = resumo_alocacoes(produto_id)
    if not df_alocacoes.empty:
        df_alocacoes['label'] = df_alocacoes.apply(
            lambda row: f"{row['ativo']} ({row['side']})", axis=1
        )
        fig.add_trace(
            go.Pie(
                labels=df_alocacoes['label'],
                values=df_alocacoes['valor_usd'],
                name="Alocações",
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Evolução de Preços (linha)
    df_posicoes = posicoes_abertas(produto_id)
    if not df_posicoes.empty:
        from analytics.queries import valores_do_ativo
        for _, pos in df_posicoes.iterrows():
            ativo = pos['ativo']
            df_valores = valores_do_ativo(ativo)
            if not df_valores.empty:
                if not pd.api.types.is_datetime64_any_dtype(df_valores['data']):
                    df_valores['data'] = pd.to_datetime(df_valores['data'])
                df_valores = df_valores.sort_values('data')
                fig.add_trace(
                    go.Scatter(
                        x=df_valores['data'],
                        y=df_valores['preco'],
                        mode='lines',
                        name=ativo,
                        showlegend=True
                    ),
                    row=2, col=1
                )
    
    # Atualizar layout
    fig.update_layout(
        title_text=f"Dashboard - Produto {produto_id}",
        height=800,
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Valor (USD)", row=1, col=1)
    fig.update_xaxes(title_text="Data", row=2, col=1)
    fig.update_yaxes(title_text="Preço (USD)", row=2, col=1)
    
    return fig
