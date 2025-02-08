import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import ast
from macro.computeFredChanges import compute_changes, detect_frequency

# 🚀 MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Macro Overview", layout="wide")

# Folder where the CSV files are stored
DATA_FOLDER = "./macro/fredData"
 
# Function to load CSV data
def load_series(series_name):
    file_path = os.path.join(DATA_FOLDER, f"{series_name}.csv")
    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path, dtype={"date": str})
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data = data.sort_values(by="date")
            data.set_index("date", inplace=True)
            return data
        except Exception:
            return None
    else:
        return None

# Get global date range
def get_global_date_range():
    all_dates = []
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(DATA_FOLDER, f), parse_dates=["date"])
                if not df.empty:
                    all_dates.extend(df["date"].dropna().tolist())
            except Exception:
                continue
    return (min(all_dates), max(all_dates)) if all_dates else (None, None)

def plot_yield_curve_regime():
    """
    Load the precomputed yield curve regime plot from a JSON file and return it.
    The JSON file is generated during the FRED data ingestion pipeline.
    """
    import os
    import plotly.io as pio
    import streamlit as st

    json_path = os.path.join("macro", "fredData", "yieldCurveRegimePlot.json")
    if not os.path.exists(json_path):
        st.error("Precomputed yield curve regime plot not found. Ensure the ingestion pipeline has been run.")
        return None

    try:
        fig = pio.read_json(json_path)
        return fig
    except Exception as e:
        st.error(f"Error loading the precomputed yield curve regime plot: {e}")
        return None

# Macro categories
categories = {
    "Growth": {
        "Key Metrics": ["realGDP", "nonfarmPayrolls", "unemploymentRate", "financialConditionsIndex"]
    },
    "Inflation": {
        "Price Metrics": ["consumerPriceIndex", "pceTrimmedMean12M", "coreStickiness", "treasury5YInflationExpectation"]
    },
    "Liquidity": {
        "Global CB Liquidity": ["globalCbLiquidity", "usNetLiquidity", "bojAssetsUSD", "ecbAssetsUSD"],
        "Fed Balance Sheet": ["fedTotalAssets", "repoAgreements", "tga"],
        "Money Supply": ["m2", "reserveBalances"]
    },
    "Yields & Curve": {
        "Yield Curve Regime": ["yieldCurveRegime"],
        "Key Rates": ["fedFundsRate", "securedOvernightFinancingRate", "treasury2Y", "treasury10Y", "interestOnReserves"]
    },
    "Markets": {
        "Indexes": ["dxy", "nasdaq", "sp500", "vix"]
    }
}

# Page Title
st.title("📊 Macro Overview")
st.markdown("This dashboard presents macroeconomic indicators using the GIP framework.")

# Date Range Selection
global_min, global_max = get_global_date_range()

if global_min and global_max:
    st.sidebar.markdown("### 📅 Date Range")
    
    from_date = st.sidebar.date_input(
        "From",
        value=global_min,
        min_value=global_min,
        max_value=global_max
    )
    
    to_date = st.sidebar.date_input(
        "To",
        value=global_max,
        min_value=global_min,
        max_value=global_max
    )
    
    apply_changes = st.sidebar.button("Apply Changes", type="primary")
    
    # Store the dates in session state when Apply Changes is clicked
    if apply_changes:
        st.session_state['from_date'] = from_date
        st.session_state['to_date'] = to_date
    
    # Initialize session state if not already set
    if 'from_date' not in st.session_state:
        st.session_state['from_date'] = from_date
    if 'to_date' not in st.session_state:
        st.session_state['to_date'] = to_date
        
    # Use the dates from session state for filtering
    from_date = st.session_state['from_date']
    to_date = st.session_state['to_date']
else:
    st.error("No data available to determine date range")

# Add navigation index in sidebar
st.sidebar.markdown("---")  # Separator

# Style for the navigation header and links
st.markdown("""
    <style>
    .nav-header {
        font-size: 1.3em !important;
        font-weight: bold !important;
        margin-bottom: 0.8em !important;
    }
    .nav-link {
        color: white !important;
        text-decoration: none !important;
    }
    .nav-link:hover {
        color: #e6e6e6 !important;
        text-decoration: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Create single navigation expander
with st.sidebar.expander("🔍 Navigation", expanded=True):
    st.markdown('<p class="nav-header">Navigation</p>', unsafe_allow_html=True)
    
    for category_name, subcategories in categories.items():
        # Create category anchor
        category_id = category_name.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("&", "and")
        st.markdown(f'<a class="nav-link" href="#{category_id}"><strong>{category_name}</strong></a>', unsafe_allow_html=True)
        
        # Add subcategory links
        for sub_category_name, variables in subcategories.items():
            sub_category_id = sub_category_name.lower().replace(" ", "-").replace("&", "and").replace("/", "-")
            st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;• <a class="nav-link" href="#{sub_category_id}">{sub_category_name}</a>', unsafe_allow_html=True)

# In the main content area, make sure anchors match exactly
for category_name, subcategories in categories.items():
    # Create category anchor
    category_id = category_name.lower().replace(" ", "-").replace("(", "").replace(")", "").replace("&", "and")
    st.markdown(f"<div id='{category_id}'></div>", unsafe_allow_html=True)
    st.header(category_name)
    
    for sub_category_name, variables in subcategories.items():
        # Create subcategory anchor
        sub_category_id = sub_category_name.lower().replace(" ", "-").replace("&", "and").replace("/", "-")
        st.markdown(f"<div id='{sub_category_id}'></div>", unsafe_allow_html=True)
        st.subheader(sub_category_name)
        with st.expander(f"{sub_category_name} Charts", expanded=True):
            # Special handling for yield curve regime - full width
            if "yieldCurveRegime" in variables:
                fig = plot_yield_curve_regime()
                if fig:
                    st.plotly_chart(fig, use_container_width=True, height=600)
                    st.markdown("""
                    **Regime Legend:**
                    - 🟢 Bull Steepener: Both yields falling, spread widening
                    - 🔴 Bear Steepener: Both yields rising, spread widening
                    - 🔵 Bull Flattener: Both yields falling, spread narrowing
                    - 🟣 Bear Flattener: Both yields rising, spread narrowing
                    - 🟡 Flattener Twist: 2Y falling, 10Y rising
                    - 🟠 Steepener Twist: 2Y rising, 10Y falling
                    """)
                else:
                    st.info("Yield curve regime data not available.")
                
                # Remove yieldCurveRegime from variables to skip it in the regular loop
                variables = [v for v in variables if v != "yieldCurveRegime"]
            
            # Regular two-column layout for other charts
            if variables:  # Only create columns if there are other variables
                cols = st.columns(2)
                for idx, var in enumerate(variables):
                    with cols[idx % 2]:
                        # Create title row with help icon and variation toggles
                        title_col, help_col = st.columns([0.9, 0.1])
                        with title_col:
                            st.markdown(f"**{var}**")
                            # Add variation toggles only for non-yield curve regime charts
                            show_mom = st.checkbox(f"Show MoM for {var}", key=f"mom_{var}")
                            show_yoy = st.checkbox(f"Show YoY for {var}", key=f"yoy_{var}")
                        with help_col:
                            with open('pages/chart_descriptions.txt', 'r') as f:
                                file_content = f.read()
                                dict_start = file_content.find('descriptions = ')
                                descriptions = ast.literal_eval(file_content[dict_start + 14:])
                            if var in descriptions:
                                help_text = f"**What is it?**\n{descriptions[var]['what']}\n\n**Why it matters:**\n{descriptions[var]['why']}"
                                st.markdown("ℹ️", help=help_text)
                        
                        data = load_series(var)

                        if data is not None:
                            # Display metrics (always show latest value, optionally show changes)
                            metric_cols = st.columns(3)
                            metric_cols[0].metric(
                                "Latest Value",
                                f"{data[var].iloc[-1]:.2f}"
                            )
                            
                            # Only create MoM/YoY metrics if requested
                            if show_mom or show_yoy:
                                df_changes = compute_changes(data, 
                                                      detect_frequency(data.index.to_pydatetime()), 
                                                      var)
                                if show_mom:
                                    mom_value = df_changes['MoM'].iloc[-1]
                                    metric_cols[1].metric(
                                        "MoM Change",
                                        f"{mom_value:.1f}%",
                                        delta=mom_value
                                    )
                                if show_yoy:
                                    yoy_value = df_changes['YoY'].iloc[-1]
                                    metric_cols[2].metric(
                                        "YoY Change",
                                        f"{yoy_value:.1f}%",
                                        delta=yoy_value
                                    )

                            # Create the main chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data[var],
                                mode="lines",
                                line=dict(color="#FC6A03", width=3.05),
                                name=var
                            ))

                            # Add variation traces if selected
                            if show_mom or show_yoy:
                                df_changes = df_changes.reset_index()
                                if show_mom:
                                    fig.add_trace(go.Scatter(
                                        x=df_changes["date"],
                                        y=df_changes["MoM"],
                                        mode="lines",
                                        line=dict(color="#2E86C1", width=1.5, dash='dot'),
                                        name="MoM %",
                                        yaxis="y2"
                                    ))
                                if show_yoy:
                                    fig.add_trace(go.Scatter(
                                        x=df_changes["date"],
                                        y=df_changes["YoY"],
                                        mode="lines",
                                        line=dict(color="#28B463", width=1.5, dash='dot'),
                                        name="YoY %",
                                        yaxis="y2"
                                    ))

                            # Update layout for dual axis if showing variations
                            if show_mom or show_yoy:
                                fig.update_layout(
                                    title=var,
                                    xaxis=dict(title="", type="date"),
                                    yaxis=dict(title=var, 
                                             range=[data[var].min(), data[var].max()]),
                                    yaxis2=dict(title="% Change",
                                              overlaying="y",
                                              side="right",
                                              showgrid=False),
                                    template="plotly_white"
                                )
                            else:
                                fig.update_layout(
                                    title=var,
                                    xaxis=dict(title="", type="date"),
                                    yaxis=dict(title=var, 
                                             range=[data[var].min(), data[var].max()]),
                                    template="plotly_white"
                                )

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Data not available.")
    