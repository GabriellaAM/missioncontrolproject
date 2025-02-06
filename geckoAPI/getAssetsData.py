import os
import pandas as pd
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
from geckoAPI.assetsRoster import ROSTER
from geckoAPI.utils import parse_gecko_prices

load_dotenv()

# Initialize the API
cg = CoinGeckoAPI(api_key=os.getenv('GECKO_API_KEY'))

# Output folder
output_folder = "./micro/assetData"

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Implementar check com ativos presentes na cas carteiras para salvar em calls


        
for coin in ROSTER:
    print(f"Checking data for {coin}...")

    csv_file_path = os.path.join(output_folder, f"{coin}.csv")
    days_needed = 100000  # Default to max if no file exists

    if os.path.exists(csv_file_path):
        # Read the existing CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check the latest date in the CSV
        if not df.empty:
            latest_date_str = df['date'].max()
            latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
            yesterday = datetime.now() - timedelta(days=1)

            # If the latest date is not yesterday, calculate days needed
            if latest_date < yesterday:
                days_needed = (yesterday - latest_date).days
            else:
                print(f"Data for {coin} is up to date.")
                continue  # Skip to the next coin if data is up to date

    print(f"Fetching data for {coin} for the last {days_needed} days...")

    try:
        # Fetch data from the API
        data = cg.get_coin_market_chart_by_id(
            id=coin, 
            vs_currency='usd',
            days=days_needed,
            interval='daily'
        )

        # Parse and append new data
        new_data_df = parse_gecko_prices(data)
        
        if os.path.exists(csv_file_path):
            # Append new data to the existing CSV
            new_data_df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            # Save new data to a new CSV file
            new_data_df.to_csv(csv_file_path, index=False)

    except Exception as e:
        print(f"Error fetching data for {coin}: {e}")