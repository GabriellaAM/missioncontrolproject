import pandas as pd
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()

load_dotenv()

# Connect to FRED API
fred = Fred(api_key=os.getenv('FRED_API_KEY'))

# Define the data series and their FRED codes
series_mapping = {
    'FF': 'fedFundsRate', # weekly  
    'UNRATE': 'unemploymentRate', # monthly
    'CORESTICKM159SFRBATL': 'coreStickiness', # monthly
    'DGS2': 'treasury2Y', # daily
    'DGS1': 'treasury1Y', # daily
    'DTB3': 'treasury3M', # daily
    'DTB6': 'treasury6M', # daily
    'T10Y2Y': 'treasury10Y2YSpread', # daily
    'T10Y3M': 'treasury10Y3MSpread', # daily        
    'CPIAUCSL': 'consumerPriceIndex', # monthly
    'T5YIE': 'treasury5YInflationExpectation', # monthly
    'DGORDER': 'durableGoodsOrders', # monthly
    'JTSJOL': 'jobOpenings', # monthly
    'PAYEMS': 'nonfarmPayrolls', # monthly
    'CES0500000003': 'retailEmployment', # monthly
    'DGS10': 'treasury10Y', # daily
    'JTSQUL': 'jobQuits', # monthly
    'JTSLDL': 'jobLayoffs', # monthly
    'ADXTNO': 'durableGoods', # monthly
    'ICSA': 'initialClaims', # weekly
    'PCEPI': 'personalConsumptionExpenditures', # monthly
    'T5YIFR': 'treasury5YInflationForwardRate', # monthly
    'PCETRIM12M159SFRBDAL': 'pceTrimmedMean12M', # monthly
    'MICH': 'consumerSentiment', # monthly
    'IORB': 'interestOnReserves', # daily
    'SOFR': 'securedOvernightFinancingRate', # daily
    'M2SL': 'm2', # monthly
    'RESPPANWW': 'fedTotalAssets', # monthly
    'WRESBAL': 'reserveBalances', # monthly
    'RRPONTSYD': 'repoAgreements', # daily
    'GDPC1': 'realGDP', # quarterly
    'GDP': 'nominalGDP', # quarterly
    'NFCI': 'financialConditionsIndex', # monthly
    'DTWEXBGS': 'dollarIndex', # daily
    'BAMLH0A0HYM2': 'creditSpreads', # monthly
    'BAMLC0A0CM': 'creditSpreadsHighGrade', # monthly
    'SP500': 'sp500', # daily
    'NASDAQCOM': 'nasdaq', # daily
    'GFDEBTN': 'usTotalDebt', # monthly
    'WTREGEN': 'tga', # monthly
    'VIXCLS': 'vix' # daily

}

# Folder for saving the data
output_folder = "macro/fredData"
os.makedirs(output_folder, exist_ok=True)

output_folder = "macro/fredData"
os.makedirs(output_folder, exist_ok=True)

def fetch_and_save_series(series_mapping, start_date=None, end_date=None):
    """Fetch series from FRED and save to CSV files."""
    for series_id, column_name in series_mapping.items():
        file_path = os.path.join(output_folder, f"{column_name}.csv")

        # Check if the file already exists
        if os.path.exists(file_path):
            print(f"Updating {column_name}...")
            existing_data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            last_date = existing_data.index.max()

            # Fetch new data if needed
            new_data = fred.get_series(series_id, observation_start=last_date)
            new_data = new_data.iloc[1:]  # Skip overlapping last row
            if not new_data.empty:
                new_data = new_data.to_frame(name=column_name)
                new_data.index.name = 'date'
                existing_data = pd.concat([existing_data, new_data])
                existing_data.to_csv(file_path)
                print(f"{column_name} - Tail of 10:")
                print(existing_data.tail(10))
        else:
            print(f"Fetching {column_name} for the first time...")
            data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            data = data.to_frame(name=column_name)
            data.index.name = 'date'
            data.to_csv(file_path)
            print(f"{column_name} - Tail of 10:")
            print(data.tail(10))

# Specify date range if needed (or leave as None for full range)
fetch_and_save_series(series_mapping, start_date=None, end_date=None)
