import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.integrate import quad
#from geckoAPI import getAssetsData

class roroClassifier:
    def __init__(self, ticker, base_dir='/Users/valter.rebelo/MissionControl/micro/assetData'):
        # Construct the file path using the base directory and ticker
        self.file_path = os.path.join(base_dir, f'{ticker}.csv')
        self.df = pd.read_csv(self.file_path, parse_dates=['date'])
        self.df.set_index('date', inplace=True)

    def calculate_variance(self):
        # Calculate log returns
        self.df['log_return'] = np.log(self.df['close'] / self.df['close'].shift(1))

        # Calculate the 7-day standard deviation of log returns
        self.df['std_7'] = self.df['log_return'].rolling(window=7).std()

        # Calculate the 90-day exponential moving average of the standard deviation
        self.df['smooth_ret_std'] = self.df['std_7'].ewm(span=90, adjust=False).mean()

        # Calculate the rolling z-score normalization of the exponential moving average with a 30-day window
        self.df['smooth_std_roll_norm'] = (self.df['smooth_ret_std'] - self.df['smooth_ret_std'].rolling(window=30).mean()) / self.df['smooth_ret_std'].rolling(window=30).std()

        # Calculate the one-day difference of the smooth_std_roll_norm and double smooth it with an EMA of 60
        self.df['smooth_std_roll_norm_diff'] = self.df['smooth_std_roll_norm'].diff()
        self.df['variance_diff'] = self.df['smooth_std_roll_norm_diff'].ewm(span=30, adjust=False).mean().ewm(span=30, adjust=False).mean()

        # Create a boolean column based on the specified rule
        self.df['variance_signal'] = np.where(self.df['smooth_std_roll_norm'] > 0, 1, np.where(self.df['variance_diff'].diff() > 0, 1, 0))

    def calculate_momentum(self):
        # Calculate the 90-day EMA of the 90-day EMA of the close
        self.df['momentum'] = self.df['close'].ewm(span=90, adjust=False).mean().ewm(span=90, adjust=False).mean()

        # Calculate a rolling 30-day z-score normalization of (close - momentum)
        self.df['momentum_shift'] = (self.df['close'] - self.df['momentum'] - (self.df['close'] - self.df['momentum']).rolling(window=30).mean()) / (self.df['close'] - self.df['momentum']).rolling(window=30).std()

    def classify_regimes(self):
        # Create columns for each regime
        self.df['bull_high_var'] = np.where((self.df['momentum_shift'] > 0) & (self.df['variance_signal'] > 0), self.df['close'], 0)
        self.df['bull_low_var'] = np.where((self.df['momentum_shift'] > 0) & (self.df['variance_signal'] <= 0), self.df['close'], 0)
        self.df['bear_high_var'] = np.where((self.df['momentum_shift'] <= 0) & (self.df['variance_signal'] > 0), self.df['close'], 0)
        self.df['bear_low_var'] = np.where((self.df['momentum_shift'] <= 0) & (self.df['variance_signal'] <= 0), self.df['close'], 0)

    def evaluate(self):
        self.calculate_variance()
        self.calculate_momentum()
        self.classify_regimes()
        return self.df

    def save_to_csv(self, output_path):
        self.df.to_csv(output_path)

    def plot_regimes(self):

        df = self.evaluate()

        fig = go.Figure()

        # Add the regime columns as bar plots with increased opacity
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['bull_high_var'],
            name='Bull High Var',
            marker_color='#008000'  # Increase opacity for better visibility
        ))

        fig.add_trace(go.Bar(
            x=df.index,
            y=df['bull_low_var'],
            name='Bull Low Var',
            marker_color='#0000FF'  # Increase opacity for better visibility
        ))

        fig.add_trace(go.Bar(
            x=df.index,
            y=df['bear_low_var'],
            name='Bear Low Var',
            marker_color='#FFFF00'  # Increase opacity for better visibility
        ))

        fig.add_trace(go.Bar(
            x=df.index,
            y=df['bear_high_var'],
            name='Bear High Var',
            marker_color='#FF0000'  # Increase opacity for better visibility
        ))

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Close',
            line=dict(color='black')
        ))

    
        # Update layout with white background and relative bar mode
        fig.update_layout(
            title='Close Prices and Regimes',
            xaxis_title='',
            yaxis_title='Price',
            barmode='overlay',
            bargap=0,  # Stack bars on top of each other
            legend_title='Legend',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig.show()

    def plot_regime_returns(self, start_date=None):
        # Filter the DataFrame by start_date if provided
        if start_date:
            df = self.df[self.df.index >= pd.to_datetime(start_date)]
        else:
            df = self.df

        # Ensure log returns are calculated
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Create subplots: 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=['Bull High Var', 'Bull Low Var', 'Bear Low Var', 'Bear High Var']
        )

        # Define regimes and colors
        regimes = {
            'Bull High Var': ('bull_high_var', 'green', 1, 1),
            'Bull Low Var': ('bull_low_var', 'blue', 1, 2),
            'Bear Low Var': ('bear_low_var', 'yellow', 2, 1),
            'Bear High Var': ('bear_high_var', 'red', 2, 2),
        }

        # Define a function to add histogram and KDE
        def add_histogram_and_kde(data, row, col, name):
            # Add histogram
            fig.add_trace(go.Histogram(
                x=data,
                name=f'{name} Histogram',
                marker_color=regimes[name][1],
                opacity=0.6,
                histnorm='probability density',  # Normalize histogram
                showlegend=False
            ), row=row, col=col)

            # Calculate KDE
            kde = gaussian_kde(data)
            x_range = np.linspace(min(data), max(data), 100)
            kde_values = kde(x_range)

            # Function to calculate probability in a small range
            def calculate_probability(x, dx=0.01):
                lower_bound = x - dx / 2
                upper_bound = x + dx / 2
                prob, _ = quad(kde, lower_bound, upper_bound)  # Integrate KDE over the interval
                return prob

            # Add KDE line with probability as a percentage in hovertemplate
            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_values,
                mode='lines',
                line=dict(color='black', width=2),
                name=f'{name} KDE',
                hovertemplate=(
                    'Log Return: %{x:.4f}<br>'
                    'Probability: %{customdata:.2f}%'  # Display as a percentage
                ),
                customdata=[calculate_probability(x) * 100 for x in x_range],  # Convert to percentage
                showlegend=False
            ), row=row, col=col)

        # Add histograms and KDEs for each regime
        for name, (regime, color, row, col) in regimes.items():
            regime_data = df[df[regime] > 0]['log_return'].dropna()
            if not regime_data.empty:  # Ensure data is not empty
                add_histogram_and_kde(regime_data, row, col, name)

        # Update layout
        fig.update_layout(
            title='Log Return Histograms with Probabilities (Percentage)',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Add subplot-specific axis titles
        for i, name in enumerate(regimes.keys(), start=1):
            fig.update_xaxes(title_text='Log Return', row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
            fig.update_yaxes(title_text='Probability Density', row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)

        fig.show()



    # Example usage:
    # processor = roroClassifier('aave')
    # processed_df = processor.process()
    # processor.save_to_csv('ativos/aave_processed.csv')