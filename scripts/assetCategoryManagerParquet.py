import os
import time
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
from datetime import datetime, timezone
from assetsRoster import ROSTER

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv('GECKO_API_KEY')
if api_key:
    cg = CoinGeckoAPI(api_key=api_key)
else:
    cg = CoinGeckoAPI()

class AssetCategoryManagerParquet:
    """
    Asset Category Manager for Parquet storage with one-hot encoding.
    
    Stores asset categories in a parquet file with boolean columns for each category,
    optimized for ML feature engineering and Feast integration.
    """
    
    def __init__(self, parquet_path=None, rate_limit_delay=0.2):
        if parquet_path is None:
            # Get the project root directory (assuming scripts is a subdirectory)
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            parquet_path = project_root / "data_parquet" / "asset_categories" / "data.parquet"
        self.parquet_path = Path(parquet_path)
        self.rate_limit_delay = rate_limit_delay
        
        # Define priority categories for primary_category selection
        self.category_priorities = [
            'Meme',
            'Decentralized Finance (DeFi)', 'DeFi',
            'Layer 1 (L1)', 'Layer 1', 'L1',
            'Layer 2 (L2)', 'Layer 2', 'L2', 
            'Gaming', 'GameFi',
            'NFT', 'Non-fungible tokens',
            'Oracle',
            'Stablecoin',
            'Infrastructure',
            'Decentralized Exchange (DEX)', 'DEX',
            'Smart Contract Platform',
            'Artificial Intelligence (AI)', 'AI'
        ]
        
        # Key categories to create boolean columns for (most important for ML)
        self.key_categories = [
            'Meme',
            'Decentralized Finance (DeFi)',
            'Layer 1 (L1)',
            'Layer 2 (L2)',
            'Gaming',
            'NFT',
            'Oracle',
            'Stablecoin',
            'Infrastructure',
            'Decentralized Exchange (DEX)',
            'Smart Contract Platform',
            'Artificial Intelligence (AI)',
            'Ethereum Ecosystem',
            'Solana Ecosystem',
            'BNB Chain Ecosystem',
            'Base Ecosystem',
            'Arbitrum Ecosystem',
            'Polygon Ecosystem'
        ]
        
        # Ensure data directory exists
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize parquet if it doesn't exist
        if not self.parquet_path.exists():
            self._create_empty_parquet()
    
    def _normalize_category_name(self, category):
        """Normalize category name for column naming"""
        return category.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
    
    def _create_empty_parquet(self):
        """Create empty parquet with schema"""
        # Base columns
        schema_fields = [
            pa.field('asset_id', pa.string()),
            pa.field('timestamp', pa.timestamp('us', tz='UTC')),
            pa.field('primary_category', pa.string()),
            pa.field('category_count', pa.int32()),
            pa.field('all_categories', pa.string())  # Still keep for reference
        ]
        
        # Add boolean columns for key categories
        for category in self.key_categories:
            col_name = f"is_{self._normalize_category_name(category)}"
            schema_fields.append(pa.field(col_name, pa.bool_()))
        
        schema = pa.schema(schema_fields)
        
        # Create empty arrays for each field
        empty_arrays = []
        for field in schema_fields:
            if field.type == pa.string():
                empty_arrays.append(pa.array([], type=pa.string()))
            elif field.type == pa.timestamp('us', tz='UTC'):
                empty_arrays.append(pa.array([], type=pa.timestamp('us', tz='UTC')))
            elif field.type == pa.int32():
                empty_arrays.append(pa.array([], type=pa.int32()))
            elif field.type == pa.bool_():
                empty_arrays.append(pa.array([], type=pa.bool_()))
        
        # Create empty table and write to parquet
        empty_table = pa.table(empty_arrays, schema=schema)
        pq.write_table(empty_table, self.parquet_path)
        logger.info(f"Created empty parquet file at {self.parquet_path}")
    
    def _load_existing_data(self):
        """Load existing parquet data"""
        try:
            if self.parquet_path.exists():
                return pd.read_parquet(self.parquet_path)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            return pd.DataFrame()
    
    def _get_existing_assets(self):
        """Get set of assets that already have categories"""
        try:
            df = self._load_existing_data()
            if not df.empty:
                return set(df['asset_id'].tolist())
            return set()
        except Exception as e:
            logger.error(f"Error reading existing categories: {e}")
            return set()
    
    def _fetch_asset_categories(self, asset_id, max_retries=3):
        """Fetch categories for a single asset from CoinGecko with retry logic"""
        for attempt in range(max_retries):
            try:
                time.sleep(self.rate_limit_delay)
                
                data = cg.get_coin_by_id(
                    id=asset_id,
                    localization=False,
                    tickers=False,
                    market_data=False,
                    community_data=False,
                    developer_data=False,
                    sparkline=False
                )
                
                categories = data.get('categories', [])
                # Filter out None values
                categories = [cat for cat in categories if cat is not None]
                return categories
                
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limit hit for {asset_id}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed all retries for {asset_id}: {e}")
                        raise
                else:
                    logger.error(f"Error fetching categories for {asset_id}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Brief pause before retry
                    else:
                        raise
    
    def _determine_primary_category(self, categories):
        """Determine the primary category based on priority rules"""
        if not categories:
            return ''
        
        # Look for priority categories first
        for priority_keyword in self.category_priorities:
            for cat in categories:
                if priority_keyword.lower() in cat.lower():
                    return cat
        
        # If no priority match, use the first category
        return categories[0]
    
    def _create_category_row(self, asset_id, categories):
        """Create a row with one-hot encoded categories"""
        timestamp = datetime.now(timezone.utc)
        primary_category = self._determine_primary_category(categories)
        all_categories = ';'.join(categories) if categories else ''
        
        # Base row data
        row_data = {
            'asset_id': asset_id,
            'timestamp': timestamp,
            'primary_category': primary_category,
            'category_count': len(categories),
            'all_categories': all_categories
        }
        
        # Add one-hot encoded columns
        for category in self.key_categories:
            col_name = f"is_{self._normalize_category_name(category)}"
            # Check if any of the asset's categories match this key category
            has_category = any(
                category.lower() in cat.lower() or cat.lower() in category.lower()
                for cat in categories
            )
            row_data[col_name] = has_category
        
        return row_data
    
    def _save_data(self, df):
        """Save dataframe to parquet"""
        try:
            # Ensure timestamp column is properly formatted
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Write to parquet
            df.to_parquet(self.parquet_path, index=False)
            logger.info(f"Saved {len(df)} records to {self.parquet_path}")
        except Exception as e:
            logger.error(f"Error saving to parquet: {e}")
            raise
    
    def update_missing_categories(self, asset_list=None):
        """Update categories for assets that don't have them yet"""
        if asset_list is None:
            asset_list = ROSTER
        
        existing_assets = self._get_existing_assets()
        missing_assets = [asset for asset in asset_list if asset not in existing_assets]
        
        if not missing_assets:
            logger.info("All assets already have categories")
            return 0
        
        logger.info(f"Updating categories for {len(missing_assets)} missing assets")
        
        # Load existing data
        df = self._load_existing_data()
        
        updated_count = 0
        skipped_count = 0
        failed_assets = []
        new_rows = []
        
        for i, asset_id in enumerate(missing_assets):
            try:
                logger.info(f"Processing {i+1}/{len(missing_assets)}: {asset_id}")
                
                categories = self._fetch_asset_categories(asset_id)
                
                # Create row with one-hot encoding
                row_data = self._create_category_row(asset_id, categories)
                new_rows.append(row_data)
                
                if categories:
                    updated_count += 1
                    logger.info(f"Added {len(categories)} categories for {asset_id}")
                else:
                    logger.warning(f"{asset_id} has no categories on CoinGecko")
                    skipped_count += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(missing_assets)} assets")
                    
            except Exception as e:
                logger.error(f"Failed to process {asset_id}: {e}")
                failed_assets.append(asset_id)
                continue
        
        # Add new rows to dataframe and save
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            if not df.empty:
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = new_df
            
            self._save_data(df)
        
        logger.info(f"Updated categories for {updated_count}/{len(missing_assets)} assets")
        logger.info(f"Skipped {skipped_count} assets with no categories")
        if failed_assets:
            logger.error(f"Failed to fetch categories for {len(failed_assets)} assets: {failed_assets}")
        
        return updated_count
    
    def get_asset_categories(self, asset_id):
        """Get categories for a specific asset"""
        try:
            df = self._load_existing_data()
            if df.empty:
                return None
                
            row = df[df['asset_id'] == asset_id]
            if not row.empty:
                categories_str = row.iloc[0]['all_categories']
                if pd.notna(categories_str) and categories_str:
                    return categories_str.split(';')
                return []
            return None
        except Exception as e:
            logger.error(f"Error getting categories for {asset_id}: {e}")
            return None
    
    def get_category_statistics(self):
        """Get statistics about asset categories"""
        try:
            df = self._load_existing_data()
            if df.empty:
                return {}
            
            # Count by category count
            category_count_dist = df['category_count'].value_counts().sort_index()
            
            # Count by primary category
            primary_category_counts = df['primary_category'].value_counts()
            
            # Count boolean columns
            boolean_cols = [col for col in df.columns if col.startswith('is_')]
            boolean_stats = {}
            for col in boolean_cols:
                true_count = df[col].sum()
                boolean_stats[col.replace('is_', '')] = int(true_count)
            
            return {
                'total_assets': len(df),
                'category_count_distribution': category_count_dist.to_dict(),
                'top_primary_categories': primary_category_counts.head(10).to_dict(),
                'boolean_category_counts': boolean_stats
            }
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {}

# Convenience functions
def update_asset_categories():
    """Update categories for all missing assets"""
    manager = AssetCategoryManagerParquet()
    return manager.update_missing_categories()

def get_asset_categories(asset_id):
    """Get categories for a specific asset"""
    manager = AssetCategoryManagerParquet()
    return manager.get_asset_categories(asset_id)

def check_new_assets_need_categories():
    """Check if any assets in ROSTER need categories"""
    manager = AssetCategoryManagerParquet()
    existing = manager._get_existing_assets()
    missing = [asset for asset in ROSTER if asset not in existing]
    return missing

if __name__ == "__main__":
    # Update missing categories when run directly
    logging.basicConfig(level=logging.INFO)
    manager = AssetCategoryManagerParquet()
    updated = manager.update_missing_categories()
    
    # Print statistics
    stats = manager.get_category_statistics()
    print(f"\n=== Asset Category Statistics ===")
    print(f"Total assets: {stats.get('total_assets', 0)}")
    print(f"Updated: {updated} assets")
    
    if 'top_primary_categories' in stats:
        print(f"\nTop Primary Categories:")
        for cat, count in list(stats['top_primary_categories'].items())[:5]:
            print(f"  {cat}: {count} assets")
    
    if 'boolean_category_counts' in stats:
        print(f"\nBoolean Category Counts:")
        for cat, count in sorted(stats['boolean_category_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cat}: {count} assets")