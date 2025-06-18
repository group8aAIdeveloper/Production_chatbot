import os
import json
import logging
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
MONGODB_CONNECTION_URI = os.getenv("MONGODB_CONNECTION_URI")
MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "ad_performance_db")
API_URL = os.getenv("API_URL", "https://reports.group8a.com/api/meta-data-records")
API_KEY = os.getenv("API_KEY", None)
JSON_DATA_DIR = os.getenv("JSON_DATA_DIR", r"C:\Users\katya\OneDrive\Desktop\final_bot_mongodbRAg\data")

# Validate environment variables
if not MONGODB_CONNECTION_URI:
    logger.error("MONGODB_CONNECTION_URI is not set in .env")
    exit(1)
if not os.path.exists(JSON_DATA_DIR):
    os.makedirs(JSON_DATA_DIR, exist_ok=True)
    logger.info(f"Created JSON_DATA_DIR: {JSON_DATA_DIR}")

# Initialize MongoDB client
try:
    client = MongoClient(MONGODB_CONNECTION_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client[MONGODB_DATABASE_NAME]
    ad_collection = db["ad_records_7"]
except Exception as e:
    logger.error(f"Initialization error: {e}")
    exit(1)

def score_result(result):
    """Score results: Pass=1, Optimize=0, Fail=-1, Incomplete=0."""
    scoring = {"Pass": 1, "Optimize": 0, "Fail": -1, "Incomplete": 0}
    return scoring.get(result, 0)

def parse_adinfo(adinfo_str):
    """Parse adinfo JSON string."""
    if not adinfo_str:
        return {
            "ad_name": "Unknown",
            "adset_name": "Unknown",
            "campaign_name": "Unknown"
        }
    try:
        adinfo = json.loads(adinfo_str)
        return {
            "ad_name": adinfo.get("ad_name", "Unknown"),
            "adset_name": adinfo.get("adset_name", "Unknown"),
            "campaign_name": adinfo.get("campaign_name", "Unknown")
        }
    except Exception as e:
        logger.error(f"Failed to parse adinfo: {e}")
        return {
            "ad_name": "Unknown",
            "adset_name": "Unknown",
            "campaign_name": "Unknown"
        }

def safe_float(value):
    """Convert value to float, remove commas, return 0 if invalid."""
    try:
        if isinstance(value, str):
            value = value.replace(",", "")
        return float(value) if value not in (None, "") else 0.0
    except (ValueError, TypeError):
        logger.warning(f"Invalid float value: {value}")
        return 0.0

def fetch_api_data():
    """Fetch data from API."""
    try:
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        response = requests.post(API_URL, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        ads = data.get("data", [])
        if not ads:
            logger.warning("No ads returned from API")
            return []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(JSON_DATA_DIR, f"api_data_{timestamp}.json")
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(ads, f)
        logger.info(f"Saved {len(ads)} ads to {file_path}")
        return ads
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {e}")
        return []

def update_ad_records(ads):
    """Remove old data and update ad_records_7 with API data, skipping ads with non-null store_deleted_at."""
    try:
        ad_collection.delete_many({})
        logger.info("Cleared previous data in ad_records_7 collection")
        skipped_count = 0
        for ad in ads:
            if "store_deleted_at" in ad and ad["store_deleted_at"] is not None:
                logger.debug(f"Skipping ad_id {ad.get('ad_id', 'unknown')} with store_deleted_at: {ad.get('store_deleted_at')}")
                skipped_count += 1
                continue
            adinfo = parse_adinfo(ad.get("adinfo", "{}"))
            spend = safe_float(ad.get("spend"))
            ad_type = ad.get("ad_type", "Unknown")
            if spend >= 1000:
                ad_type = "Evergreen"
            if spend < 200 and ad_type != "Test":
                ad["result"] = "INCOMPLETE"
                ad["result_30"] = "INCOMPLETE"
                ad["result_90"] = "INCOMPLETE"
            ad["ad_id"] = str(ad.get("ad_id", "Unknown"))
            ad["ad_type"] = ad_type
            ad["campaign_name"] = adinfo["campaign_name"]
            ad["ad_name"] = adinfo["ad_name"]
            ad["adset_name"] = adinfo["adset_name"]
            ad["store"] = ad.get("store_name", "Unknown")
            ad["spend"] = spend
            # Ensure result fields are strings
            ad["result"] = str(ad.get("result", "UNKNOWN")) if ad.get("result") is not None else "UNKNOWN"
            ad["result_30"] = str(ad.get("result_30", "UNKNOWN")) if ad.get("result_30") is not None else "UNKNOWN"
            ad["result_90"] = str(ad.get("result_90", "UNKNOWN")) if ad.get("result_90") is not None else "UNKNOWN"
            ad["score"] = score_result(ad["result"])
            ad["score_30"] = score_result(ad["result_30"])
            ad["score_90"] = score_result(ad["result_90"])
            ad["total_score"] = ad["score"] + ad["score_30"] + ad["score_90"]
            ad_collection.update_one(
                {"ad_id": ad["ad_id"]},
                {"$set": ad},
                upsert=True
            )
        logger.info(f"Updated {len(ads) - skipped_count} documents in ad_records_7, skipped {skipped_count} ads with non-null store_deleted_at")
    except Exception as e:
        logger.error(f"Update error: {e}")

def main():
    """Orchestrate API data fetching and updates."""
    logger.info(f"Starting update at {datetime.now()}")
    ads = fetch_api_data()
    if ads:
        update_ad_records(ads)
    else:
        logger.warning("No ads loaded from API")
    client.close()

if __name__ == "__main__":
    main()