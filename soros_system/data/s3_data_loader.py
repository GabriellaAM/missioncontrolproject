"""
S3 Data Loader for Soros System
Reads data directly from S3 buckets
"""
import pandas as pd
import boto3
from io import StringIO
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class S3DataLoader:
    """Data loader that reads from S3 instead of local files"""
    
    def __init__(self, bucket_name, aws_access_key_id=None, aws_secret_access_key=None, region='us-east-1'):
        """
        Initialize S3 data loader
        
        Args:
            bucket_name: Name of your S3 bucket
            aws_access_key_id: AWS access key (optional)
            aws_secret_access_key: AWS secret key (optional)
            region: AWS region
        """
        self.bucket_name = bucket_name
        
        # PLACEHOLDER CONFIGURATION
        # Same credential options as upload script:
        # 1. Environment variables (recommended)
        # 2. AWS CLI configuration
        # 3. IAM roles
        # 4. Pass credentials directly (not recommended)
        
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region
            )
        else:
            self.s3_client = boto3.client('s3', region_name=region)
    
    @lru_cache(maxsize=128)
    def _read_csv_from_s3(self, s3_key):
        """Read a CSV file from S3 with caching"""
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        except Exception as e:
            logger.error(f"Failed to read {s3_key} from S3: {e}")
            return None
    
    def load_asset_data(self, asset_id):
        """Load asset price data from S3"""
        s3_key = f"data/micro/assetData/{asset_id}.csv"
        df = self._read_csv_from_s3(s3_key)
        if df is not None:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df
    
    def load_candle_data(self, asset_id):
        """Load OHLC candle data from S3"""
        s3_key = f"data/micro/candleData/{asset_id}_candles.csv"
        df = self._read_csv_from_s3(s3_key)
        if df is not None:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        return df
    
    def load_fred_data(self, indicator):
        """Load FRED economic data from S3"""
        s3_key = f"data/macro/fredData/{indicator}.csv"
        df = self._read_csv_from_s3(s3_key)
        if df is not None and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df
    
    def load_onchain_data(self, metric):
        """Load on-chain data from S3"""
        s3_key = f"data/onchainData/{metric}.csv"
        df = self._read_csv_from_s3(s3_key)
        if df is not None and 't' in df.columns:
            df['t'] = pd.to_datetime(df['t'])
            df.set_index('t', inplace=True)
        return df
    
    def list_available_assets(self):
        """List all available assets in S3"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='data/micro/assetData/',
                Delimiter='/'
            )
            
            assets = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    filename = obj['Key'].split('/')[-1]
                    if filename.endswith('.csv'):
                        asset_id = filename.replace('.csv', '')
                        assets.append(asset_id)
            
            return sorted(assets)
        except Exception as e:
            logger.error(f"Failed to list assets from S3: {e}")
            return []


# Convenience function for pandas integration
def read_csv_from_s3(s3_path, bucket_name=None, **kwargs):
    """
    Read CSV directly from S3 using pandas-like interface
    
    Args:
        s3_path: S3 path like 's3://bucket/path/file.csv' or just 'path/file.csv'
        bucket_name: Bucket name if not included in s3_path
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame
    """
    # PLACEHOLDER: Set default bucket name
    DEFAULT_BUCKET = "your-company-soros-data"
    
    if s3_path.startswith('s3://'):
        # Full S3 URL provided
        return pd.read_csv(s3_path, **kwargs)
    else:
        # Just the key provided
        bucket = bucket_name or DEFAULT_BUCKET
        return pd.read_csv(f's3://{bucket}/{s3_path}', **kwargs)


# Example usage for notebooks
if __name__ == "__main__":
    # PLACEHOLDER: Replace with your bucket name
    BUCKET_NAME = "your-company-soros-data"
    
    # Initialize loader
    loader = S3DataLoader(bucket_name=BUCKET_NAME)
    
    # Load Bitcoin price data
    btc_data = loader.load_asset_data('bitcoin')
    print(f"Bitcoin data shape: {btc_data.shape}")
    
    # Or use direct pandas integration
    # btc_data = pd.read_csv('s3://your-company-soros-data/data/micro/assetData/bitcoin.csv')
    
    # List all available assets
    assets = loader.list_available_assets()
    print(f"Available assets: {len(assets)}")