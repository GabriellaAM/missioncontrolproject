import streamlit as st
import os
import pandas as pd
import plotly.express as px

# Set up the overall page configuration
st.set_page_config(page_title="Macro Variables Dashboard", layout="wide")

# Optionally add some custom CSS for a nicer look
st.markdown(
    """
    <style>
    .css-18e3th9 {padding: 2rem 1rem 10rem;}
    body {
        background-color: #f5f5f5;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Path to the folder with FRED CSV files
DATA_FOLDER = "./macro/fredData"

# Utility function to load a series (macro variable) from its CSV file.
def load_series(series_name):
    file_path = os.path.join(DATA_FOLDER, f"{series_name}.csv")
    if os.path.exists(file_path):
        try:
            # Load the date column as datetime and use it as index
            df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
            return df
        except Exception as e:
            st.error(f"Error loading {series_name}: {e}")
            return None
    else:
        st.error(f"File for {series_name} not found.")
        return None

# Get a sorted list of available series names (derived from the CSV file names)
def get_available_series():
    if os.path.exists(DATA_FOLDER):
        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
        series_names = [os.path.splitext(f)[0] for f in files]
        return sorted(series_names)
    else:
        return []

# --- Sidebar Options ---
st.sidebar.title("Dashboard Options")
plot_type = st.sidebar.radio("Select Visualization", ["Line Plot", "Dot Plot Comparison"])

available_series = get_available_series()
if not available_series:
    st.error("No macro FRED data found. Please run getFredData.py first.")
    st.stop()

# --- Page Title and Description ---
st.title("Macro Variables Dashboard")
st.markdown(
    """
This dashboard visualizes macroeconomic data fetched from FRED.
Use the sidebar to select a visualization type, choose your variables, 
and explore both trends over time and relationships between different macro variables.
    """
)

# --- Visualization Sections ---
if plot_type == "Line Plot":
    st.header("Line Plot of a Macro Variable")
    series_choice = st.sidebar.selectbox("Select a Macro Variable", available_series)
    df = load_series(series_choice)
    if df is not None:
        # Reset index so that 'date' becomes a column (to plot using Plotly Express)
        df_reset = df.reset_index()
        fig = px.line(
            df_reset,
            x="date",
            y=series_choice,
            title=f"{series_choice} Over Time",
            markers=True,
        )
        fig.update_layout(xaxis_title="Date", yaxis_title=series_choice, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Dot Plot Comparison":
    st.header("Dot Plot Comparison of Two Macro Variables")
    col1, col2 = st.columns(2)
    with col1:
        x_series = st.selectbox("Select X Variable", available_series, key="x_var")
    with col2:
        y_series = st.selectbox("Select Y Variable", available_series, key="y_var")
    
    if x_series == y_series:
        st.error("Please select two different macro variables for a comparison.")
    else:
        df_x = load_series(x_series)
        df_y = load_series(y_series)
        
        if df_x is not None and df_y is not None:
            # Merge the two series on the date index (only include overlapping dates)
            df_merged = pd.merge(df_x, df_y, left_index=True, right_index=True, how="inner")
            df_merged = df_merged.sort_index()
            
            if df_merged.empty:
                st.error("No overlapping dates found between the selected series.")
            else:
                # Identify the latest observation date (by the common dates)
                latest_date = df_merged.index.max()
                # Reset index so 'date' becomes a column and flag the latest data point
                df_merged = df_merged.reset_index()
                df_merged["Highlight"] = df_merged["date"].apply(
                    lambda x: "Latest" if x == latest_date else "Other"
                )
                
                fig = px.scatter(
                    df_merged,
                    x=x_series,
                    y=y_series,
                    color="Highlight",
                    hover_data=["date"],
                    title=f"{y_series} vs {x_series}",
                    color_discrete_map={"Latest": "red", "Other": "blue"},
                )
                fig.update_traces(marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")))
                fig.update_layout(xaxis_title=x_series, yaxis_title=y_series, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
