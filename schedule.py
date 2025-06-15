import schedule
import time
import subprocess
import logging

# Setup logging
logging.basicConfig(filename='scheduler.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name):
    """Run a Python script and log its output or errors."""
    try:
        result = subprocess.run(["python3", script_name], check=True, capture_output=True, text=True)
        logger.info(f"Ran {script_name} successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run {script_name}: {e.stderr}")
    except FileNotFoundError:
        logger.error(f"Script {script_name} not found. Check the file path.")
    except Exception as e:
        logger.error(f"Unexpected error running {script_name}: {str(e)}")

# Schedule once daily at 8 AM IST
schedule.every().day.at("08:00").do(run_script, "data_fetch_update.py")
schedule.every().day.at("08:00").do(run_script, "injest.py")

logger.info("Scheduler started at %s", time.strftime("%Y-%m-%d %H:%M:%S %Z"))
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute