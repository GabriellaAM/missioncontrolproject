#!/usr/bin/env python3
"""
Migration script to add 'source' column to existing crypto parquet files.

This script adds 'source': 'COINGECKO' to all existing crypto data files
that don't already have this column.
"""

import pandas as pd
import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_crypto_source_column():
    """Add source column to all existing crypto parquet files."""
    
    # Path to crypto data
    crypto_data_path = Path("data_parquet/crypto_data")
    
    if not crypto_data_path.exists():
        logger.error(f"Crypto data path does not exist: {crypto_data_path}")
        return False
    
    # Get all asset folders
    asset_folders = [f for f in crypto_data_path.iterdir() if f.is_dir()]
    
    logger.info(f"Found {len(asset_folders)} crypto assets to migrate")
    
    migrated_count = 0
    already_migrated_count = 0
    error_count = 0
    
    for asset_folder in asset_folders:
        asset_id = asset_folder.name
        parquet_file = asset_folder / "data.parquet"
        
        if not parquet_file.exists():
            logger.warning(f"{asset_id}: No data.parquet file found")
            continue
        
        try:
            # Read existing data
            df = pd.read_parquet(parquet_file)
            
            # Check if source column already exists
            if 'source' in df.columns:
                logger.info(f"{asset_id}: Already has source column")
                already_migrated_count += 1
                continue
            
            # Add source column
            df['source'] = 'COINGECKO'
            
            # Save back to parquet
            df.to_parquet(parquet_file, index=False, compression='snappy')
            
            logger.info(f"✅ {asset_id}: Added source column ({len(df)} records)")
            migrated_count += 1
            
        except Exception as e:
            logger.error(f"❌ {asset_id}: Error during migration - {e}")
            error_count += 1
    
    # Summary
    logger.info("=== MIGRATION SUMMARY ===")
    logger.info(f"Successfully migrated: {migrated_count}")
    logger.info(f"Already migrated: {already_migrated_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Total processed: {migrated_count + already_migrated_count + error_count}")
    
    return error_count == 0

if __name__ == "__main__":
    success = migrate_crypto_source_column()
    exit(0 if success else 1)