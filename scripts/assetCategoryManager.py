import csv
import os
import time
import logging
import pandas as pd
from pathlib import Path
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
from .assetsRoster import ROSTER

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv('GECKO_API_KEY')
if api_key:
    cg = CoinGeckoAPI(api_key=api_key)
else:
    cg = CoinGeckoAPI()

class AssetCategoryManager:
    def __init__(self, csv_path="./data/asset_categories.csv", rate_limit_delay=0.2):
        self.csv_path = Path(csv_path)
        self.rate_limit_delay = rate_limit_delay
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
            'Decentralized Exchange (DEX)', 'DEX'
        ]
        
        # Ensure data directory exists
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not self.csv_path.exists():
            self._create_empty_csv()
    
    def _create_empty_csv(self):
        """Create empty CSV with headers"""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['asset_id', 'categories', 'category_count', 'primary_category'])
    
    def _get_existing_assets(self):
        """Get set of assets that already have categories"""
        try:
            if self.csv_path.exists():
                df = pd.read_csv(self.csv_path)
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
    
    def _append_to_csv(self, asset_id, categories):
        """Append a single asset's categories to the CSV"""
        categories_str = ';'.join(categories) if categories else ''
        category_count = len(categories)
        primary_category = self._determine_primary_category(categories)
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([asset_id, categories_str, category_count, primary_category])
    
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
        
        updated_count = 0
        skipped_count = 0
        failed_assets = []
        
        for i, asset_id in enumerate(missing_assets):
            try:
                logger.info(f"Processing {i+1}/{len(missing_assets)}: {asset_id}")
                
                categories = self._fetch_asset_categories(asset_id)
                
                # Only append if we successfully fetched categories or confirmed they're empty
                # If _fetch_asset_categories raises an exception, we won't append
                if categories:
                    self._append_to_csv(asset_id, categories)
                    updated_count += 1
                    logger.info(f"Added {len(categories)} categories for {asset_id}")
                else:
                    # Asset genuinely has no categories on CoinGecko
                    logger.warning(f"{asset_id} has no categories on CoinGecko")
                    # Still write to CSV to mark as processed, but with empty categories
                    self._append_to_csv(asset_id, categories)
                    skipped_count += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(missing_assets)} assets")
                    
            except Exception as e:
                logger.error(f"Failed to process {asset_id}: {e}")
                failed_assets.append(asset_id)
                continue
        
        logger.info(f"Updated categories for {updated_count}/{len(missing_assets)} assets")
        logger.info(f"Skipped {skipped_count} assets with no categories")
        if failed_assets:
            logger.error(f"Failed to fetch categories for {len(failed_assets)} assets: {failed_assets}")
        
        return updated_count
    
    def get_asset_categories(self, asset_id):
        """Get categories for a specific asset"""
        try:
            df = pd.read_csv(self.csv_path)
            row = df[df['asset_id'] == asset_id]
            if not row.empty:
                categories_str = row.iloc[0]['categories']
                if pd.notna(categories_str) and categories_str:
                    return categories_str.split(';')
                return []
            return None
        except Exception as e:
            logger.error(f"Error getting categories for {asset_id}: {e}")
            return None
    
    def get_assets_by_category(self, category_filter):
        """Get all assets that have a specific category"""
        try:
            df = pd.read_csv(self.csv_path)
            matching_assets = []
            
            for _, row in df.iterrows():
                categories_str = row['categories']
                if pd.notna(categories_str) and categories_str:
                    categories = categories_str.split(';')
                    if any(category_filter.lower() in cat.lower() for cat in categories):
                        matching_assets.append({
                            'asset_id': row['asset_id'],
                            'primary_category': row['primary_category'],
                            'all_categories': categories
                        })
            
            return matching_assets
        except Exception as e:
            logger.error(f"Error filtering by category {category_filter}: {e}")
            return []
    
    def get_category_statistics(self):
        """Get statistics about asset categories"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Count by category count
            category_count_dist = df['category_count'].value_counts().sort_index()
            
            # Count by primary category
            primary_category_counts = df['primary_category'].value_counts()
            
            # All categories frequency
            all_categories = []
            for categories_str in df['categories'].dropna():
                if categories_str:
                    all_categories.extend(categories_str.split(';'))
            
            category_frequency = pd.Series(all_categories).value_counts()
            
            return {
                'total_assets': len(df),
                'category_count_distribution': category_count_dist.to_dict(),
                'top_primary_categories': primary_category_counts.head(10).to_dict(),
                'top_all_categories': category_frequency.head(20).to_dict()
            }
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return {}
    
    def update_single_asset(self, asset_id):
        """Update categories for a single asset"""
        try:
            categories = self._fetch_asset_categories(asset_id)
            
            # Check if asset already exists in CSV
            existing_assets = self._get_existing_assets()
            
            if asset_id in existing_assets:
                # Update existing entry
                df = pd.read_csv(self.csv_path)
                
                categories_str = ';'.join(categories) if categories else ''
                category_count = len(categories)
                primary_category = self._determine_primary_category(categories)
                
                df.loc[df['asset_id'] == asset_id, 'categories'] = categories_str
                df.loc[df['asset_id'] == asset_id, 'category_count'] = category_count
                df.loc[df['asset_id'] == asset_id, 'primary_category'] = primary_category
                
                df.to_csv(self.csv_path, index=False)
            else:
                # Add new entry
                self._append_to_csv(asset_id, categories)
            
            logger.info(f"Updated categories for {asset_id}: {len(categories)} categories")
            return True
            
        except Exception as e:
            logger.error(f"Error updating single asset {asset_id}: {e}")
            return False

# Convenience functions
def update_asset_categories():
    """Update categories for all missing assets"""
    manager = AssetCategoryManager()
    return manager.update_missing_categories()

def get_asset_categories(asset_id):
    """Get categories for a specific asset"""
    manager = AssetCategoryManager()
    return manager.get_asset_categories(asset_id)

def check_new_assets_need_categories():
    """Check if any assets in ROSTER need categories"""
    manager = AssetCategoryManager()
    existing = manager._get_existing_assets()
    missing = [asset for asset in ROSTER if asset not in existing]
    return missing

if __name__ == "__main__":
    # Update missing categories when run directly
    logging.basicConfig(level=logging.INFO)
    manager = AssetCategoryManager()
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