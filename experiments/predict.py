# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mlflow
import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
from config import setup_mlflow
from utils.feature_loader import FeatureLoader

# %%
# Setup MLflow
setup_mlflow()
client = mlflow.MlflowClient()

# %%
# List available models
models = client.search_registered_models()
print("📋 Available models:")
for model in models:
    print(f"  - {model.name}")

# %%
# Load a model to test
model_name = models[0].name if models else 'bitcoin_topo_catboost'
model_uri = f"models:/{model_name}/latest"

print(f"🔄 Loading: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
print(f"✅ Model '{model_name}' exists and is loaded")

# %%
def get_latest_data(asset='bitcoin', days=30):
    """Fetch latest data for predictions."""
    loader = FeatureLoader()

    # Load crypto data
    df = loader._load_single_crypto(asset)
    df = loader._apply_date_filter(df)

    # Get last N days
    data = df.tail(days).copy()

    # Rename columns to match model expectations
    data = data.rename(columns={
        'open': f'{asset}_open',
        'high': f'{asset}_high',
        'low': f'{asset}_low',
        'close': f'{asset}_close',
        'total_volume': f'{asset}_total_volume'
    })

    data = data.set_index('timestamp')

    return data

# %%
# Get fresh data
data = get_latest_data('bitcoin', days=360)
print(f"📊 Data shape: {data.shape}")
print(data.head())

# %%
# Make predictions
predictions = model.predict(data)
print(f"✅ Predictions: {predictions}")

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_predictions(data, predictions):
    """Chart showing cumulative returns and model decisions over time."""
    # Convert predictions to numpy array and flatten if 2D
    signals = predictions.to_numpy().flatten() if isinstance(predictions, pd.DataFrame) else predictions

    # Ensure signals is 1-dimensional
    if hasattr(signals, 'shape') and len(signals.shape) > 1:
        signals = signals.flatten()

    # Adjust the length of data and signals to match
    min_length = min(len(data), len(signals))
    data = data.iloc[-min_length:]
    signals = signals[-min_length:]

    # Calculate returns and cumulative returns
    df = pd.DataFrame({
        'price': data['bitcoin_close'],
        'signal': signals
    }, index=data.index)
    df['returns'] = df['price'].pct_change().fillna(0)
    df['cumulative_return'] = (1 + df['returns']).cumprod() - 1
    df['strategy_return'] = df['returns'] * df['signal'].shift(1).fillna(0)
    df['cumulative_strategy_return'] = (1 + df['strategy_return']).cumprod() - 1

    # Create a figure with two subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Cumulative Returns", "Model Trading Signals"))

    # Cumulative returns chart
    fig.add_trace(go.Scatter(x=df.index, y=df['cumulative_return'], mode='lines', name='Cumulative Return', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['cumulative_strategy_return'], mode='lines', name='Cumulative Strategy Return', line=dict(color='orange')), row=1, col=1)

    # Signal chart
    fig.add_trace(go.Scatter(x=df.index, y=df['signal'], mode='lines+markers', name='Model Signal', line=dict(color='green'), marker=dict(symbol='circle')), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1200, title_text="Model Predictions", showlegend=True)
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Signal", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig

# %%
# Plot results
fig = plot_predictions(data, predictions)
fig.show()

# %%
import plotly.graph_objects as go

def plot_predictions(data, predictions):
    """Simple chart showing model decisions over time."""
    # Convert predictions to numpy array and flatten if 2D
    signals = predictions.to_numpy().flatten() if isinstance(predictions, pd.DataFrame) else predictions

    # Ensure signals is 1-dimensional
    if hasattr(signals, 'shape') and len(signals.shape) > 1:
        signals = signals.flatten()
# %%



# %%
