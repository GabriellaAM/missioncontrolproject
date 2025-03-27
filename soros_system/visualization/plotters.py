import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta

class PortfolioVisualizer:
    """
    Class for visualizing portfolio performance and market trends.
    """
    def __init__(self):
        """Initialize the PortfolioVisualizer."""
        self.logger = logging.getLogger(__name__)
    
    def plot_portfolio_with_volatility(self, backtest_results, btc_data=None, show_plot=True):
        """
        Plot portfolio performance with volatility overlay.
        
        Args:
            backtest_results (dict): Results from portfolio backtesting
            btc_data (pd.DataFrame, optional): Bitcoin price data
            show_plot (bool, optional): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if not backtest_results:
            self.logger.warning("No backtest results provided for plotting.")
            return None
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Extract data
        dates = backtest_results.get('dates', [])
        portfolio_values = backtest_results.get('portfolio_values', [])
        btc_prices = backtest_results.get('btc_prices', [])
        
        if not dates or not portfolio_values:
            self.logger.warning("Missing required data for portfolio plot.")
            return None
        
        # Convert dates to datetime if needed
        if isinstance(dates[0], str):
            dates = [pd.Timestamp(d) for d in dates]
        
        # Calculate portfolio performance
        initial_value = portfolio_values[0]
        normalized_values = [v / initial_value for v in portfolio_values]
        
        # Calculate BTC performance
        if btc_prices:
            initial_btc = btc_prices[0]
            normalized_btc = [p / initial_btc for p in btc_prices]
        
        # Add portfolio line
        fig.add_trace(
            go.Scatter(
                x=dates, 
                y=normalized_values,
                mode='lines',
                name=f'Portfolio ({backtest_results.get("portfolio_name", "Unknown")})',
                line=dict(color='green', width=2)
            ),
            secondary_y=False
        )
        
        # Add BTC line
        if btc_prices:
            fig.add_trace(
                go.Scatter(
                    x=dates, 
                    y=normalized_btc,
                    mode='lines',
                    name='BTC Benchmark',
                    line=dict(color='orange', width=2, dash='dash')
                ),
                secondary_y=False
            )
        
        # Add volatility overlay if available
        if 'volatility_states' in backtest_results:
            vol_states = backtest_results['volatility_states']
            
            # Create volatility bands
            high_vol_x = []
            high_vol_y = []
            
            for i, date in enumerate(dates):
                if i < len(vol_states) and vol_states[i] == 1:  # High volatility
                    high_vol_x.append(date)
                    high_vol_y.append(normalized_values[i])
            
            # Add high volatility markers
            if high_vol_x:
                fig.add_trace(
                    go.Scatter(
                        x=high_vol_x,
                        y=high_vol_y,
                        mode='markers',
                        name='High Volatility',
                        marker=dict(color='red', size=8, symbol='circle')
                    ),
                    secondary_y=False
                )
        
        # Add annotations for key metrics
        metrics = backtest_results.get('metrics', {})
        annotations = []
        
        # Portfolio return
        if 'total_return' in metrics:
            annotations.append(
                dict(
                    x=0.01, y=0.99, 
                    xref='paper', yref='paper',
                    text=f"Portfolio Return: {metrics['total_return']:.2%}",
                    showarrow=False,
                    font=dict(size=12, color='green'),
                    align='left'
                )
            )
        
        # BTC return
        if 'btc_total_return' in metrics:
            annotations.append(
                dict(
                    x=0.01, y=0.94, 
                    xref='paper', yref='paper',
                    text=f"BTC Return: {metrics['btc_total_return']:.2%}",
                    showarrow=False,
                    font=dict(size=12, color='orange'),
                    align='left'
                )
            )
        
        # Sharpe ratio
        if 'sharpe_ratio' in metrics:
            annotations.append(
                dict(
                    x=0.01, y=0.89, 
                    xref='paper', yref='paper',
                    text=f"Sharpe: {metrics['sharpe_ratio']:.2f}",
                    showarrow=False,
                    font=dict(size=12),
                    align='left'
                )
            )
        
        # Max drawdown
        if 'max_drawdown' in metrics:
            annotations.append(
                dict(
                    x=0.01, y=0.84, 
                    xref='paper', yref='paper',
                    text=f"Max DD: {metrics['max_drawdown']:.2%}",
                    showarrow=False,
                    font=dict(size=12, color='red'),
                    align='left'
                )
            )
        
        # Set up layout
        fig.update_layout(
            title=f"Portfolio Performance: {backtest_results.get('portfolio_name', 'Unknown')}",
            xaxis_title="Date",
            yaxis_title="Normalized Value",
            legend=dict(x=0.01, y=0.01, bgcolor='rgba(255,255,255,0.8)'),
            annotations=annotations,
            hovermode="x unified",
            template="plotly_white"
        )
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_individual_asset_performance(self, backtest_results, asset_filter=None, show_plot=True):
        """
        Plot performance of individual assets in a portfolio.
        
        Args:
            backtest_results (dict): Results from portfolio backtesting
            asset_filter (list, optional): List of asset IDs to include
            show_plot (bool, optional): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if not backtest_results or 'asset_performance' not in backtest_results:
            self.logger.warning("No asset performance data available for plotting.")
            return None
        
        asset_performance = backtest_results['asset_performance']
        
        # Filter assets if specified
        if asset_filter:
            asset_performance = {k: v for k, v in asset_performance.items() if k in asset_filter}
        
        if not asset_performance:
            self.logger.warning("No assets to plot after filtering.")
            return None
        
        # Create DataFrame for plotting
        performance_data = []
        for asset_id, metrics in asset_performance.items():
            performance_data.append({
                'Asset': asset_id,
                'ROI': metrics.get('roi', 0) * 100,  # Convert to percentage
                'Total P&L': metrics.get('total_pnl', 0),
                'Trades': metrics.get('num_trades', 0)
            })
        
        df = pd.DataFrame(performance_data)
        
        # Sort by ROI descending
        df = df.sort_values('ROI', ascending=False)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Asset ROI (%)", "Total P&L"),
            vertical_spacing=0.15,
            specs=[[{"type": "bar"}], [{"type": "bar"}]]
        )
        
        # Add ROI bars
        fig.add_trace(
            go.Bar(
                x=df['Asset'],
                y=df['ROI'],
                name='ROI (%)',
                marker_color=df['ROI'].apply(lambda x: 'green' if x > 0 else 'red'),
                text=df['ROI'].apply(lambda x: f"{x:.2f}%"),
                textposition='auto',
                hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add P&L bars
        fig.add_trace(
            go.Bar(
                x=df['Asset'],
                y=df['Total P&L'],
                name='Total P&L',
                marker_color=df['Total P&L'].apply(lambda x: 'green' if x > 0 else 'red'),
                text=df['Total P&L'].apply(lambda x: f"${x:.2f}"),
                textposition='auto',
                hovertemplate='%{x}: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Asset Performance in {backtest_results.get('portfolio_name', 'Unknown')} Portfolio",
            showlegend=False,
            height=800,
            template="plotly_white"
        )
        
        # Add number of trades as annotations
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row['Asset'],
                y=row['ROI'],
                text=f"{row['Trades']} trades",
                showarrow=False,
                yshift=10,
                font=dict(size=10),
                row=1, col=1
            )
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_trend_distribution(self, asset_data, trend_type='USD', show_plot=True):
        """
        Plot distribution of trend classifications.
        
        Args:
            asset_data (pd.DataFrame): DataFrame with trend classifications
            trend_type (str, optional): Type of trend to analyze ('USD' or 'BTC')
            show_plot (bool, optional): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if asset_data.empty:
            self.logger.warning("Empty data provided for trend distribution plot.")
            return None
        
        # Ensure we have trend column
        trend_col = f'Overall_Trend_{trend_type}'
        if trend_col not in asset_data.columns:
            self.logger.warning(f"Column {trend_col} not found in data.")
            return None
        
        # Create copy with date as index
        df = asset_data.copy()
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('date', inplace=True)
        
        # Get trend counts
        trend_counts = df[trend_col].value_counts().sort_index()
        
        # Map trend values to descriptions
        trend_descriptions = {
            -2: "Strong Bear",
            -1: "Weak Bear",
            0: "Neutral",
            1: "Weak Bull",
            2: "Strong Bull"
        }
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(
            go.Bar(
                x=[trend_descriptions.get(t, str(t)) for t in trend_counts.index],
                y=trend_counts.values,
                text=trend_counts.values,
                textposition='auto',
                marker_color=['darkred', 'lightcoral', 'lightgrey', 'lightgreen', 'darkgreen'][:len(trend_counts)],
                hovertemplate='%{x}: %{y} days<extra></extra>'
            )
        )
        
        # Calculate percentages
        total_days = trend_counts.sum()
        percentages = [(count / total_days) * 100 for count in trend_counts.values]
        
        # Add annotations for percentages
        for i, (trend, count) in enumerate(trend_counts.items()):
            fig.add_annotation(
                x=trend_descriptions.get(trend, str(trend)),
                y=count,
                text=f"{percentages[i]:.1f}%",
                showarrow=False,
                yshift=10,
                font=dict(size=10)
            )
        
        # Update layout
        fig.update_layout(
            title=f"Distribution of {trend_type} Trend Classifications",
            xaxis_title="Trend Classification",
            yaxis_title="Number of Days",
            template="plotly_white"
        )
        
        if show_plot:
            fig.show()
        
        return fig
    
    def plot_transitions_heatmap(self, transition_matrix, show_plot=True):
        """
        Plot a heatmap of transition probabilities between trend states.
        
        Args:
            transition_matrix (pd.DataFrame): Transition probability matrix
            show_plot (bool, optional): Whether to display the plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if transition_matrix.empty:
            self.logger.warning("Empty transition matrix provided for heatmap.")
            return None
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=transition_matrix.values,
                x=transition_matrix.columns,
                y=transition_matrix.index,
                colorscale='RdBu',
                zmin=0, zmax=1,
                hovertemplate='%{y} -> %{x}: %{z:.2f}<extra></extra>',
                showscale=True,
                colorbar=dict(title='Probability')
            )
        )
        
        # Add text annotations
        for i in range(len(transition_matrix.index)):
            for j in range(len(transition_matrix.columns)):
                fig.add_annotation(
                    x=transition_matrix.columns[j],
                    y=transition_matrix.index[i],
                    text=f"{transition_matrix.iloc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color='white' if transition_matrix.iloc[i, j] > 0.5 else 'black')
                )
        
        # Update layout
        fig.update_layout(
            title="Trend State Transition Probabilities",
            xaxis_title="To State",
            yaxis_title="From State",
            template="plotly_white"
        )
        
        if show_plot:
            fig.show()
        
        return fig 