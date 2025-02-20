import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_hmm_results(df, asset):
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    def calculate_metrics(returns, cum_returns, is_strategy=True, df=None):
        total_return = cum_returns.iloc[-1]
        days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
        ann_return = ((1 + total_return) ** (365/days)) - 1

        if is_strategy:
            state_changes = df['raw_position'].diff().fillna(0)
            trades = (state_changes != 0).sum()
            active_returns = returns[df['raw_position'] != 0]
            win_rate = (active_returns > 0).sum() / len(active_returns) if len(active_returns) > 0 else 0

        else:
            trades = 0
            win_rate = 0

        wealth_index = 1 + cum_returns
        previous_peaks = wealth_index.expanding(min_periods=1).max()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = drawdowns.min()
        risk_free_rate = 0.04
        daily_rf = (1 + risk_free_rate) ** (1/365) - 1
        excess_returns = returns - daily_rf
        sharpe = np.sqrt(365) * (excess_returns.mean() / returns.std())
        
        return {
            'Total Return': f'{total_return:.2%}', 'Annualized Return': f'{ann_return:.2%}',
            'Number of Trades': trades, 'Max Drawdown': f'{max_drawdown:.2%}',
            'Sharpe Ratio': f'{sharpe:.2f}', 'Win Rate': f'{win_rate:.2%}'
        }
    
    strategy_metrics = calculate_metrics(df['strategy_return'], df['cum_strategy_return'], True, df)
    bh_metrics = calculate_metrics(df['open_return'], df['cum_buy_hold_return'], False, df)
    metrics_table = pd.DataFrame({'Strategy': strategy_metrics, 'Buy & Hold': bh_metrics}).round(4)

    # Create figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                       row_heights=[0.5, 0.2, 0.3],
                       specs=[[{"secondary_y": True}], [{"type": "table"}], [{"type": "scatter"}]])
    
    # Add asset price (Close) on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Asset Price',
            line=dict(color='lightgray', width=1),
            opacity=0.9,
            hovertemplate="<br>".join([
                "Date: %{x}",
                "Price: $%{y:,.2f}",
                "<extra></extra>"
            ])
        ),
        row=1, col=1,
        secondary_y=True
    )
    
    # Add Buy & Hold with enhanced hover
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['cum_buy_hold_return'],
            mode='markers',
            name='Buy & Hold',
            marker=dict(
                size=4.5,
                color=df['hidden_state'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Hidden State')
            ),
            hovertemplate="<br>".join([
                "Date: %{x}",
                "Return: %{y:.2%}",
                "State: %{customdata}",
                "<extra></extra>"
            ]),
            customdata=df['state']
        ),
        row=1, col=1,
        secondary_y=False
    )
    
    # Add Strategy line with enhanced hover
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['cum_strategy_return'],
            mode='lines',
            name='HMM Strategy',
            line=dict(color='red', width=2),
            hovertemplate="<br>".join([
                "Date: %{x}",
                "Return: %{y:.2%}",
                "Position: %{customdata}",
                "<extra></extra>"
            ]),
            customdata=df['state']
        ),
        row=1, col=1,
        secondary_y=False
    )
    
    # Add metrics table
    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Strategy', 'Buy & Hold'],
                       font=dict(size=12),
                       align="left"),
            cells=dict(values=[metrics_table.index,
                             metrics_table['Strategy'],
                             metrics_table['Buy & Hold']],
                      font=dict(size=11),
                      align="left")
        ),
        row=2, col=1
    )
    
    # Add probability plot
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['unfav'],
            mode='lines',
            name='Unfavored Probability',
            line=dict(color='black', width=2)
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f'HMM-Regime Based Strategy Analysis for {asset}',
        template='plotly_white',
        height=1000,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Update axes titles
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Asset Price", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Unfavored Probability", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    fig.show()
