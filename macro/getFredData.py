import pandas as pd
from fredapi import Fred
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import certifi
import yfinance as yf
import warnings
from macro.computeYieldCurveRegime import compute_regime

warnings.filterwarnings("ignore")

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
    #'T10Y2Y': 'treasury10Y2YSpread', # daily
    #'T10Y3M': 'treasury10Y3MSpread', # daily        
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
    'VIXCLS': 'vix', # daily
    'JPNASSETS': 'bojAssets', # monthly
    'ECBASSETSW': 'ecbAssets', # monthly
}

# Folder for saving the data
output_folder = "macro/fredData"
os.makedirs(output_folder, exist_ok=True)

output_folder = "macro/fredData"
os.makedirs(output_folder, exist_ok=True)

def fetch_and_save_series(series_mapping, start_date=None, end_date=None):
    """Fetch series from FRED and save to CSV files."""
    for series_id, column_name in series_mapping.items():
        try:
            file_path = os.path.join(output_folder, f"{column_name}.csv")

            # Check if the file already exists
            if os.path.exists(file_path):
                print(f"Updating {column_name}...")
                existing_data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
                last_date = existing_data.index.max()

                # Add retry mechanism with delay
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        new_data = fred.get_series(series_id, observation_start=last_date)
                        if not new_data.empty:
                            new_data = new_data.iloc[1:]  # Skip overlapping last row
                            new_data = new_data.to_frame(name=column_name)
                            new_data.index.name = 'date'
                            existing_data = pd.concat([existing_data, new_data])
                            existing_data.to_csv(file_path)
                            print(f"{column_name} - Updated successfully")
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"Failed to update {column_name} after {max_retries} attempts: {e}")
                        else:
                            import time
                            time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Fetching {column_name} for the first time...")
                data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                if not data.empty:
                    data = data.to_frame(name=column_name)
                    data.index.name = 'date'
                    data.to_csv(file_path)
                    print(f"{column_name} - Initial fetch successful")

        except Exception as e:
            print(f"Error processing {column_name}: {e}")
            continue

def calculate_net_liquidity():
    """Calculate and save US Net Liquidity metric."""
    try:
        # Load required components
        fed_assets = pd.read_csv(os.path.join(output_folder, "fedTotalAssets.csv"), 
                               parse_dates=['date'], index_col='date')
        repo = pd.read_csv(os.path.join(output_folder, "repoAgreements.csv"), 
                          parse_dates=['date'], index_col='date')
        tga = pd.read_csv(os.path.join(output_folder, "tga.csv"), 
                         parse_dates=['date'], index_col='date')
        
        # Convert Fed Assets from millions to billions
        fed_assets = fed_assets / 1000  
        
        # Combine all series and forward fill missing values
        combined = pd.concat([fed_assets, repo, tga], axis=1).ffill()
        
        # Calculate Net Liquidity (all in billions now)
        combined['usNetLiquidity'] = (combined['fedTotalAssets'] - 
                                    combined['repoAgreements'] - 
                                    combined['tga'])
        
        # Save the net liquidity series
        net_liquidity = combined[['usNetLiquidity']]
        net_liquidity.to_csv(os.path.join(output_folder, "usNetLiquidity.csv"))
        print("Updated usNetLiquidity calculation (in billions of USD)")
        
    except Exception as e:
        print(f"Error calculating Net Liquidity: {e}")

def fetch_yfinance_data():
    """Fetch data from Yahoo Finance and save to CSV."""
    try:
        tickers = {
            "DX-Y.NYB": "dxy",
            "JPY=X": "usdjpy",
            "EUR=X": "usdeur",
            "^MOVE": "move"  # Adding MOVE index
        }
        
        for symbol, name in tickers.items():
            file_path = os.path.join(output_folder, f"{name}.csv")
            
            if os.path.exists(file_path):
                print(f"Updating {name}...")
                existing_data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
                last_date = existing_data.index.max()
                
                ticker = yf.Ticker(symbol)
                new_data = ticker.history(start=last_date)['Close']
                new_data.index = new_data.index.tz_localize(None)
                new_data = new_data.iloc[1:]
                
                if not new_data.empty:
                    new_data = new_data.to_frame(name=name)
                    new_data.index.name = 'date'
                    existing_data = pd.concat([existing_data, new_data])
                    existing_data.to_csv(file_path)
                    print(f"{name} - Tail of 10:")
                    print(existing_data.tail(10))
            else:
                print(f"Fetching {name} for the first time...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="max")['Close']
                data.index = data.index.tz_localize(None)
                data = data.to_frame(name=name)
                data.index.name = 'date'
                data.to_csv(file_path)
                print(f"{name} - Tail of 10:")
                print(data.tail(10))
                
    except Exception as e:
        print(f"Error fetching Yahoo Finance data: {e}")

def calculate_global_cb_liquidity():
    """Calculate global central bank liquidity in USD billions."""
    try:
        # Load all required components
        us_liq = pd.read_csv(os.path.join(output_folder, "usNetLiquidity.csv"), 
                            parse_dates=['date'], index_col='date')
        boj = pd.read_csv(os.path.join(output_folder, "bojAssets.csv"), 
                         parse_dates=['date'], index_col='date')
        ecb = pd.read_csv(os.path.join(output_folder, "ecbAssets.csv"), 
                         parse_dates=['date'], index_col='date')
        usdjpy = pd.read_csv(os.path.join(output_folder, "usdjpy.csv"), 
                            parse_dates=['date'], index_col='date')
        usdeur = pd.read_csv(os.path.join(output_folder, "usdeur.csv"), 
                            parse_dates=['date'], index_col='date')
        
        # Combine all series and forward fill missing values
        combined = pd.concat([boj, ecb, usdjpy, usdeur], axis=1).ffill()
        
        # Convert BOJ assets (100M yen -> USD billions)
        combined['bojAssetsUSD'] = (combined['bojAssets'] * 100) / combined['usdjpy'] / 1000
        
        # Convert ECB assets (M EUR -> USD billions)
        combined['ecbAssetsUSD'] = combined['ecbAssets'] * combined['usdeur'] / 1000
        
        # Calculate total global liquidity
        combined = pd.concat([combined, us_liq], axis=1).ffill()
        combined['globalCbLiquidity'] = (combined['usNetLiquidity'] + 
                                       combined['bojAssetsUSD'] + 
                                       combined['ecbAssetsUSD'])
        
        # Save the converted series
        for col in ['bojAssetsUSD', 'ecbAssetsUSD', 'globalCbLiquidity']:
            df = combined[[col]]
            df.to_csv(os.path.join(output_folder, f"{col}.csv"))
            print(f"Updated {col}")
            
    except Exception as e:
        print(f"Error calculating global CB liquidity: {e}")

# Add at the end of the file
if __name__ == "__main__":
    # Verify API key is present
    if not os.getenv('FRED_API_KEY'):
        raise ValueError("FRED API key not found in environment variables")
    
    try:
        # 1. Fetch all raw data first
        fetch_and_save_series(series_mapping)
        fetch_yfinance_data()
        
        # 2. Calculate derived metrics
        calculate_net_liquidity()
        calculate_global_cb_liquidity()
        
        # 3. Compute yield curve regime and changes for all series
        from macro.computeFredChanges import main as compute_changes_main
        from macro.computeYieldCurveRegime import compute_regime, compute_and_save_yield_curve_regime_plot
        compute_regime()
        compute_and_save_yield_curve_regime_plot()  # uses the revised logic (spread computed from t10y & t2y)
        compute_changes_main()
    except Exception as e:
        print(f"Error in main execution: {e}")
