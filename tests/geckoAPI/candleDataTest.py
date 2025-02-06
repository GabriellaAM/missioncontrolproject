import unittest
import pandas as pd
import os
from datetime import datetime, timedelta
from geckoAPI.getAssetsCandleData import fetch_daily, initialize_historical_data, get_asset_launch_date
from geckoAPI.assetsRoster import ROSTER
import time

def print_separator(message=""):
    print("\n" + "="*80)
    if message:
        print(message)
        print("-"*80)

class candleDataTest(unittest.TestCase):
    def setUp(self):
        """Setup test environment"""
        print_separator("Setting up test environment")
        self.output_folder = "./micro/candleData"
        self.asset_folder = "./micro/assetData"
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print("Created candleData directory")
            
        if not os.path.exists(self.asset_folder):
            os.makedirs(self.asset_folder)
            print("Created assetData directory")
            print("\nUpdating asset price histories...")
            import geckoAPI.getAssetsData
        
        # Force full update of candle data
        print("\nForcing full update of historical data...")
        fetch_daily(force_full_update=True)
        
        print(f"Found {len(ROSTER)} assets to process")
    
    def test_1_initialize_historical_data(self):
        """Test if historical data is properly initialized"""
        print_separator("Testing Historical Data Initialization")
        
        for coin in ROSTER:
            print(f"\nProcessing {coin}")
            print("-"*40)
            
            # Check asset data first
            asset_file = os.path.join(self.asset_folder, f"{coin}.csv")
            if not os.path.exists(asset_file):
                print(f"❌ No asset data found for {coin}, skipping...")
                continue
                
            candle_file = os.path.join(self.output_folder, f"{coin}_candles.csv")
            
            if not os.path.exists(candle_file):
                print(f"📊 Initializing new candle data for {coin}")
                initialize_historical_data(coin)
            else:
                df = pd.read_csv(candle_file)
                asset_df = pd.read_csv(asset_file)
                
                candle_latest = datetime.strptime(df['date'].max(), '%Y-%m-%d')
                asset_latest = datetime.strptime(asset_df['date'].max(), '%Y-%m-%d')
                
                print(f"Current candle data until: {candle_latest.date()}")
                print(f"Asset data available until: {asset_latest.date()}")
                
                if candle_latest < asset_latest:
                    days_behind = (asset_latest - candle_latest).days
                    print(f"⚠️  Candle data is {days_behind} days behind asset data")
                    print(f"📈 Updating historical data for {coin}")
                    initialize_historical_data(coin)
                else:
                    print(f"✅ Candle data is up to date")
            
            # Verify the data
            self.assertTrue(os.path.exists(candle_file), 
                          f"Historical data file for {coin} does not exist")
            
            df = pd.read_csv(candle_file)
            launch_date = get_asset_launch_date(coin)
            
            if launch_date:
                oldest_date = datetime.strptime(df['date'].min(), '%Y-%m-%d')
                days_difference = (oldest_date - launch_date).days
                print(f"Launch date: {launch_date.date()}")
                print(f"Oldest data: {oldest_date.date()}")
                print(f"Difference: {abs(days_difference)} days")
                
                self.assertLessEqual(abs(days_difference), 5,
                    f"Historical data for {coin} doesn't go back far enough")
    
    def test_2_data_structure(self):
        """Test if data has the correct structure"""
        for coin in ROSTER:
            file_path = os.path.join(self.output_folder, f"{coin}_candles.csv")
            self.assertTrue(os.path.exists(file_path), 
                          f"Candle data file for {coin} does not exist")
            
            df = pd.read_csv(file_path)
            expected_columns = ['date', 'open', 'high', 'low', 'close']
            self.assertListEqual(list(df.columns), expected_columns,
                f"Incorrect columns in {coin}_candles.csv")
            
            # Check data types
            self.assertEqual(df['date'].dtype, 'object')
            for col in ['open', 'high', 'low', 'close']:
                self.assertTrue(pd.api.types.is_numeric_dtype(df[col]))
    
    def test_3_data_freshness(self):
        """Test if data is up to date"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        for coin in ROSTER:
            file_path = os.path.join(self.output_folder, f"{coin}_candles.csv")
            df = pd.read_csv(file_path)
            latest_date = df['date'].max()
            self.assertGreaterEqual(latest_date, yesterday,
                f"Data for {coin} is not up to date. Latest: {latest_date}, Expected: {yesterday}")
    
    def test_4_data_continuity(self):
        """Test if there are no gaps in the data"""
        print_separator("Testing Data Continuity")
        
        # Dictionary to collect gap patterns
        gap_patterns = {}
        
        for coin in ROSTER:
            print(f"\nChecking {coin} for gaps")
            print("-"*40)
            
            file_path = os.path.join(self.output_folder, f"{coin}_candles.csv")
            df = pd.read_csv(file_path)
            
            # Convert dates to datetime and sort
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Check for gaps
            date_diff = df['date'].diff().dt.days
            gaps_mask = date_diff > 1
            
            if gaps_mask.any():
                gap_indices = df.index[gaps_mask]
                print(f"\n⚠️  Gaps found in {coin} data:")
                
                # Collect gaps for analysis
                for idx in gap_indices:
                    gap_start = df.iloc[idx-1]['date'].strftime('%Y-%m-%d')
                    gap_end = df.iloc[idx]['date'].strftime('%Y-%m-%d')
                    gap_size = (df.iloc[idx]['date'] - df.iloc[idx-1]['date']).days - 1
                    gap_key = f"{gap_start} to {gap_end}"
                    
                    if gap_key not in gap_patterns:
                        gap_patterns[gap_key] = {'count': 0, 'assets': []}
                    gap_patterns[gap_key]['count'] += 1
                    gap_patterns[gap_key]['assets'].append(coin)
                    
                    print(f"   Gap of {gap_size} days between {gap_start} and {gap_end}")
                
                print(f"⚠️  Gaps found in {coin}. Running initialize_historical_data to fix...")
                initialize_historical_data(coin)
                
                # Verify gaps were fixed
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                date_diff = df['date'].diff().dt.days
                if (date_diff > 1).any():
                    print(f"❌ Gaps still present after resampling in {coin}")
                else:
                    print(f"✅ Gaps successfully filled in {coin}")
            else:
                print(f"✅ No gaps found in {coin} data")
        
        # Print gap pattern analysis
        print("\n" + "="*80)
        print("Gap Pattern Analysis")
        print("-"*80)
        for gap_period, stats in gap_patterns.items():
            print(f"\nGap period: {gap_period}")
            print(f"Occurred in {stats['count']} assets:")
            print(f"Affected assets: {', '.join(stats['assets'])}")

if __name__ == '__main__':
    unittest.main()
