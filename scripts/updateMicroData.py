import subprocess
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('micro_data_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_path, script_name):
    """
    Run a Python script and allow its logging to flow through
    """
    logger.info(f"Starting {script_name}")
    try:
        # Run script without capturing output so logging flows through
        result = subprocess.run(
            ['python', script_path],
            check=True
        )
        logger.info(f"Successfully completed {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {str(e)}")
        return False
    
def update_micro_data():
    """
    Run all micro data update scripts in sequence
    """
    base_path = "/Users/valter.rebelo/MissionControl/scripts"
    scripts = [
        ("getAssetsData.py", "Asset Data Update"),
        ("getAssetsCandleData.py", "Asset Candle Data Update"),
        ("getMarketData.py", "Market Data Update")
    ]
    
    logger.info("Starting micro data update sequence")
    
    for script_file, script_name in scripts:
        script_path = os.path.join(base_path, script_file)
        if not run_script(script_path, script_name):
            logger.error(f"Failed to complete {script_name}. Stopping sequence.")
            return False
    
    logger.info("Successfully completed all updates")
    return True

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting update process at {start_time}")
    
    try:
        success = update_micro_data()
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            logger.info(f"Update process completed successfully in {duration}")
        else:
            logger.error(f"Update process failed after {duration}")
    except Exception as e:
        logger.error(f"Unexpected error in update process: {str(e)}")
