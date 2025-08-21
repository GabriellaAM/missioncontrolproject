import pandas as pd
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to sys.path to ensure correct module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Import our parquet managers
from scripts.macroDataParquet import (
    YahooDataManager, 
    FredDataManager, 
    MacroCalculationsManager,
    update_all_macro_data
)

def main():
    """
    Main function to update all macro economic data in parquet format.
    
    This replaces the old CSV-based getFredData.py approach with a modern
    parquet-based system that provides:
    - Better performance
    - Consistent partitioned storage  
    - OHLC data from Yahoo Finance
    - Calculated metrics (RTY/YM ratio, net liquidity, etc.)
    """
    start_time = datetime.now()
    logger.info(f"Starting macro data update to Parquet at {start_time}")
    
    try:
        # Verify API keys are present
        if not os.getenv('FRED_API_KEY'):
            raise ValueError("FRED API key not found in environment variables")
        
        # Run complete macro data update
        results = update_all_macro_data()
        
        # Process results
        yahoo_results = results['yahoo']
        fred_results = results['fred'] 
        calc_results = results['calculations']
        
        # Count successes and failures
        yahoo_success = len([r for r in yahoo_results if r[1]])
        yahoo_total = len(yahoo_results)
        
        fred_success = len([r for r in fred_results if r[1]])
        fred_total = len(fred_results)
        
        calc_success = len([r for r in calc_results if r[1]])
        calc_total = len(calc_results)
        
        # Log detailed results
        logger.info("=== UPDATE SUMMARY ===")
        logger.info(f"Yahoo Finance: {yahoo_success}/{yahoo_total} successful")
        if yahoo_success < yahoo_total:
            failed_yahoo = [r[0] for r in yahoo_results if not r[1]]
            logger.warning(f"Failed Yahoo assets: {failed_yahoo}")
        
        logger.info(f"FRED Data: {fred_success}/{fred_total} successful")  
        if fred_success < fred_total:
            failed_fred = [r[0] for r in fred_results if not r[1]]
            logger.warning(f"Failed FRED series: {failed_fred}")
        
        logger.info(f"Calculations: {calc_success}/{calc_total} successful")
        if calc_success < calc_total:
            failed_calc = [r[0] for r in calc_results if not r[1]]
            logger.warning(f"Failed calculations: {failed_calc}")
        
        # Overall success rate
        total_success = yahoo_success + fred_success + calc_success
        total_operations = yahoo_total + fred_total + calc_total
        success_rate = (total_success / total_operations) * 100 if total_operations > 0 else 0
        
        logger.info(f"Overall success rate: {success_rate:.1f}% ({total_success}/{total_operations})")
        
        # Run dependent scripts if macro data update was successful
        if success_rate >= 80:  # At least 80% success rate
            logger.info("=== RUNNING DEPENDENT CALCULATIONS ===")
            
            try:
                # Import and run dependent scripts
                # Note: These scripts may need to be updated to read from parquet instead of CSV
                from scripts.computeFredChanges import main as compute_changes_main
                from scripts.computeYieldCurveRegime import compute_regime, compute_and_save_yield_curve_regime_plot
                
                logger.info("Computing yield curve regime...")
                compute_regime()
                compute_and_save_yield_curve_regime_plot()
                logger.info("✅ Yield curve regime computed successfully")
                
                logger.info("Computing FRED changes...")
                compute_changes_main()
                logger.info("✅ FRED changes computed successfully")
                
            except ImportError as e:
                logger.warning(f"Could not import dependent scripts: {e}")
                logger.warning("Dependent scripts may need to be updated to read parquet data")
            except Exception as e:
                logger.error(f"Error running dependent calculations: {e}")
        else:
            logger.warning(f"Skipping dependent calculations due to low success rate ({success_rate:.1f}%)")
        
        # Final timing
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Macro data update completed in {duration}")
        
        # Show storage summary
        logger.info("=== STORAGE SUMMARY ===")
        try:
            from pathlib import Path
            macro_data_path = Path(project_root) / "data_parquet" / "macro_data"
            
            if macro_data_path.exists():
                asset_folders = [f for f in macro_data_path.iterdir() if f.is_dir()]
                total_assets = len(asset_folders)
                
                logger.info(f"Total macro assets stored: {total_assets}")
                
                # Sample some asset info
                sample_assets = asset_folders[:5]
                for asset_folder in sample_assets:
                    parquet_file = asset_folder / "data.parquet"
                    if parquet_file.exists():
                        try:
                            df = pd.read_parquet(parquet_file)
                            latest_date = df['timestamp'].max() if 'timestamp' in df.columns else 'Unknown'
                            logger.info(f"  {asset_folder.name}: {len(df)} records, latest: {latest_date}")
                        except Exception as e:
                            logger.warning(f"  {asset_folder.name}: Error reading parquet - {e}")
                
                if total_assets > 5:
                    logger.info(f"  ... and {total_assets - 5} more assets")
            
        except Exception as e:
            logger.warning(f"Could not generate storage summary: {e}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    results = main()