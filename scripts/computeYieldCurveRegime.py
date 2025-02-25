import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

def classify_regime(delta_2y, delta_10y, delta_spread):
    """Classify yield curve regime based on changes in rates.
    
    Uses a tolerance to treat near-zero changes as neutral.
    """
    tolerance = 1e-5
    if abs(delta_2y) < tolerance or abs(delta_10y) < tolerance or abs(delta_spread) < tolerance:
        return "neutral"
        
    # Determine overall direction if both rates move in the same way
    if delta_2y < 0 and delta_10y < 0:
        direction = "bull"
    elif delta_2y > 0 and delta_10y > 0:
        direction = "bear"
    # Twist scenarios when the rates move oppositely
    elif delta_2y > 0 and delta_10y < 0:
        return "steepener_twist"
    elif delta_2y < 0 and delta_10y > 0:
        return "flattener_twist"
    else:
        direction = "neutral"
    
    # Decide if the yield curve is steepening or flattening
    if delta_spread > 0:
        curve = "steepener"
    elif delta_spread < 0:
        curve = "flattener"
    else:
        curve = "neutral"
    
    # If we have a neutral component, classify as neutral overall.
    if direction == "neutral" or curve == "neutral":
        return "neutral"
    
    return f"{direction}_{curve}"

def compute_regime(window=20):
    """Compute yield curve regime based on treasury rates and save to CSV.
    
    This function loads the 2-year and 10-year treasury yield data, computes
    the 2s10s spread, calculates rolling (default 20-day) differences, and then
    classifies the regime based on these changes. The results are saved to
    "yieldCurveRegime.csv" in the macro/fredData directory.
    """
    try:
        # Load yield data
        data_folder = "data/macro/fredData"
        t2y = pd.read_csv(os.path.join(data_folder, "treasury2Y.csv"), 
                         parse_dates=['date'], index_col='date')
        t10y = pd.read_csv(os.path.join(data_folder, "treasury10Y.csv"), 
                          parse_dates=['date'], index_col='date')
        
        # Merge the data on date, ensuring both series are available
        df = pd.merge(t2y, t10y, left_index=True, right_index=True, how='inner')
        df.columns = ['t2y', 't10y']
        
        # Compute the 2s10s spread dynamically
        df['spread'] = df['t10y'] - df['t2y']
        
        # Ensure the data is sorted by date
        df = df.sort_index()
        
        # Compute rolling changes over the specified window (default: 20 days)
        df['delta_2y'] = df['t2y'].diff(window)
        df['delta_10y'] = df['t10y'].diff(window)
        df['delta_spread'] = df['spread'].diff(window)
        
        # Drop the initial rows where rolling differences are NaN
        df = df.dropna(subset=['delta_2y', 'delta_10y', 'delta_spread'])
        
        # Classify the regime for each available date
        df['regime'] = df.apply(lambda x: classify_regime(x['delta_2y'], 
                                                            x['delta_10y'], 
                                                            x['delta_spread']), axis=1)
        
        # Create color mapping
        color_map = {
            'bull_steepener': '#03C04A',  # green
            'bear_steepener': '#FF0000',  # red
            'bull_flattener': '#0492C2',  # Cerulean
            'bear_flattener': '#9867C5',  # light purple
            'steepener_twist': '#FFA500',  # orange
            'flattener_twist': '#FCE205',  # yellow
            'neutral': '#FFFFFF'  # white
        }
        
        df['color'] = df['regime'].map(color_map)
        
        # Save to CSV
        df.to_csv(os.path.join(data_folder, "yieldCurveRegime.csv"))
        print("Updated yield curve regime classification successfully.")
        
    except Exception as e:
        print(f"Error computing yield curve regime: {e}")

def compute_and_save_yield_curve_regime_plot():
    """
    Compute and save the precomputed yield curve regime plot as JSON.
    This version:
      - Filters data from 1990 onward
      - Uses Plotly's Scatter (not Scattergl as it's not needed for this amount of data)
      - No redundant regime legends (we use the markdown legend in Streamlit)
      - Optimized for faster loading
    """
    # Define the CSV path
    csv_path = os.path.join("data/macro", "fredData", "yieldCurveRegime.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file from {csv_path}: {e}")
        return

    # Convert the 'date' column to datetime and sort chronologically
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values("date")
    
    # Filter data: only include dates from January 1, 1990 onward
    filter_date = pd.to_datetime("2010-01-01")
    df = df[df['date'] >= filter_date]
    if df.empty:
        print("No data available on or after January 1, 1990.")
        return

    # Create the base figure
    fig = go.Figure()

    # Add regime blocks as filled areas (without legends)
    current_regime = None
    current_color = None
    block_start = None

    for idx, row in df.iterrows():
        if row['regime'] != current_regime:
            if block_start is not None:
                # Add the previous block
                block_data = df[(df['date'] >= block_start) & (df['date'] <= row['date'])]
                fig.add_trace(go.Scatter(
                    x=block_data['date'],
                    y=block_data['spread'],
                    fill='tozeroy',
                    mode='none',
                    showlegend=False,  # No legend entries for regime blocks
                    fillcolor=current_color,
                    opacity=0.3,
                    hoverinfo='skip'
                ))
            block_start = row['date']
            current_regime = row['regime']
            current_color = row['color']

    # Add the last block
    if block_start is not None:
        block_data = df[df['date'] >= block_start]
        fig.add_trace(go.Scatter(
            x=block_data['date'],
            y=block_data['spread'],
            fill='tozeroy',
            mode='none',
            showlegend=False,  # No legend entry
            fillcolor=current_color,
            opacity=0.3,
            hoverinfo='skip'
        ))

    # Add the spread line on top (this is the only legend entry we'll keep)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['spread'],
        mode='lines',
        name='2-10 Spread',
        line=dict(color="#FC6A03", width=2),
        hoverinfo='all'
    ))

    # Update layout settings
    fig.update_layout(
        title="2-10 Year Yield Curve Regime Overlay",
        xaxis_title="Date",
        yaxis_title="Spread",
        template="plotly_white",
        showlegend=True,
        autosize=True,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Save the precomputed plot as JSON
    output_file = os.path.join("data/macro", "fredData", "yieldCurveRegimePlot.json")
    pio.write_json(fig, output_file)
    print(f"Saved yield curve regime plot to {output_file}")

if __name__ == '__main__':
    compute_and_save_yield_curve_regime_plot()
