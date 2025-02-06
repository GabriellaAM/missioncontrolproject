import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import ast

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

# Macro categories
categories = {
    "Growth - GDP Measures": {
        "Output Measures": ["realGDP", "nominalGDP"]
    },
    "Growth - Labor & Production": {
        "Employment & Job Market": ["nonfarmPayrolls", "retailEmployment", "jobOpenings", "jobQuits", "jobLayoffs", "unemploymentRate", "initialClaims"],
        "Industrial Activity": ["durableGoodsOrders", "durableGoods"]
    },
    "Growth - Sentiment & Consumption": {
        "Consumer Metrics": ["consumerSentiment", "personalConsumptionExpenditures"]
    },
    "Inflation": {
        "Price Indices": ["consumerPriceIndex", "pceTrimmedMean12M"],
        "Expectations & Dynamics": ["treasury5YInflationExpectation", "treasury5YInflationForwardRate", "coreStickiness"]
    },
    "Liquidity (and Monetary Conditions)": {
        "Interest Rates & Yields": ["fedFundsRate", "treasury3M", "treasury6M", "treasury1Y", "treasury2Y", "treasury10Y"],
        "Yield Spreads": ["treasury10Y2YSpread", "treasury10Y3MSpread"],
        "Central Bank & Money Supply": ["fedTotalAssets", "m2", "reserveBalances", "repoAgreements"],
        "Other Risk/Liquidity Metrics": ["securedOvernightFinancingRate", "interestOnReserves", "creditSpreads", "financialConditionsIndex", "dollarIndex", "usTotalDebt", "tga", "nasdaq", "sp500", "vix"]
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
            cols = st.columns(2)  # Two charts per row

            for idx, var in enumerate(variables):
                with cols[idx % 2]:
                    # Create title row with help icon
                    title_col, help_col = st.columns([0.9, 0.1])
                    with title_col:
                        st.markdown(f"")
                    with help_col:
                        with open('pages/chart_descriptions.txt', 'r') as f:
                            file_content = f.read()
                            # Find the dictionary definition starting point
                            dict_start = file_content.find('descriptions = ')
                            # Extract and evaluate the dictionary
                            descriptions = ast.literal_eval(file_content[dict_start + 14:])
                        if var in descriptions:
                            help_text = f"**What is it?**\n{descriptions[var]['what']}\n\n**Why it matters:**\n{descriptions[var]['why']}"
                            st.markdown("ℹ️", help=help_text)
                    
                    data = load_series(var)

                    if data is not None:
                        df_plot = data.reset_index()
                        col_name = next((col for col in df_plot.columns if col.lower() == var.lower()), None)

                        if not col_name:
                            continue

                        df_plot = df_plot[(df_plot["date"].dt.date >= from_date) & 
                                        (df_plot["date"].dt.date <= to_date)]

                        if df_plot.empty:
                            st.info("No data in selected date range.")
                        else:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df_plot["date"],
                                y=df_plot[col_name],
                                mode="lines",
                                line=dict(color="#FC6A03", width=3.05),
                                name=col_name
                            ))

                            fig.update_layout(
                                title=var,
                                xaxis=dict(title="", type="date"),
                                yaxis=dict(title=col_name, range=[df_plot[col_name].min(), df_plot[col_name].max()]),
                                template="plotly_white"
                            )

                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Data not available.")
