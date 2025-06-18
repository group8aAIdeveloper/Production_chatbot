import os
import json
import re
import logging
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
MONGODB_CONNECTION_URI = os.getenv("MONGODB_CONNECTION_URI")
MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "ad_performance_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BATCH_SIZE = 50
CHECKPOINT_FILE = "ingest_checkpoint.json"

# Validate environment variables
if not MONGODB_CONNECTION_URI:
    logger.error("MONGODB_CONNECTION_URI is not set in .env")
    exit(1)
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in .env")
    exit(1)

# Initialize clients
try:
    client = MongoClient(MONGODB_CONNECTION_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client[MONGODB_DATABASE_NAME]
    ad_collection = db["ad_records_7"]
    embedded_content_collection = db["embedded_content"]
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error(f"Initialization error: {e}")
    exit(1)

def save_checkpoint(processed_count, last_ad_id):
    """Save checkpoint to file."""
    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump({"processed_count": processed_count, "last_ad_id": last_ad_id}, f)
        logger.info(f"Checkpoint saved: {processed_count} ads, last ad_id: {last_ad_id}")
    except Exception as e:
        logger.error(f"Checkpoint save error: {e}")

def load_checkpoint():
    """Load checkpoint from file."""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        return {"processed_count": 0, "last_ad_id": None}
    except Exception as e:
        logger.error(f"Checkpoint load error: {e}")
        return {"processed_count": 0, "last_ad_id": None}

def get_embedding(text):
    """Generate embedding using OpenAI API."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

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
            value = re.sub(r'[^\d.]', '', value)  # Remove commas and non-numeric chars
        return float(value) if value not in (None, "") else 0.0
    except (ValueError, TypeError):
        logger.warning(f"Invalid float value: {value}")
        return 0.0

def get_score(result):
    """Score results: Pass=1, Optimize=0, Fail=-1, Incomplete=0."""
    scoring = {"Pass": 1, "Optimize": 0, "Fail": -1, "Incomplete": 0}
    return scoring.get(result.capitalize(), 0)

def process_batch(batch, processed, total_ads):
    """Process a batch of ads, skipping those with non-null store_deleted_at."""
    skipped_count = 0
    for ad in batch:
        ad_id = str(ad.get("ad_id", "Unknown"))
        logger.info(f"Processing ad_id: {ad_id} ({processed + 1}/{total_ads})")
        if "store_deleted_at" in ad and ad["store_deleted_at"] is not None:
            logger.debug(f"Skipping ad_id {ad_id} due to store_deleted_at: {ad.get('store_deleted_at')}")
            skipped_count += 1
            continue
        try:
            adinfo = parse_adinfo(ad.get("adinfo", "{}"))
            spend = safe_float(ad.get("spend", 0))
            ad_type = ad.get("ad_type", "Unknown")
            # Handle None values for result fields
            result = str(ad.get("result", "UNKNOWN")) if ad.get("result") is not None else "UNKNOWN"
            result_30 = str(ad.get("result_30", "UNKNOWN")) if ad.get("result_30") is not None else "UNKNOWN"
            result_90 = str(ad.get("result_90", "UNKNOWN")) if ad.get("result_90") is not None else "UNKNOWN"
            result = result.capitalize()
            result_30 = result_30.capitalize()
            result_90 = result_90.capitalize()
            # Set ad_type to Evergreen if spend >= 1000
            if spend >= 1000:
                ad_type = "Evergreen"
            # Apply spend rule for non-Test ads
            if spend < 200 and ad_type != "Test":
                result = "Incomplete"
                result_30 = "Incomplete"
                result_90 = "Incomplete"
            total_score = get_score(result) + get_score(result_30) + get_score(result_90)
            content = {
                "ad_id": ad_id,
                "type": ad_type,
                "campaign_name": adinfo["campaign_name"],
                "ad_name": adinfo["ad_name"],
                "adset_name": adinfo["adset_name"],
                "store": ad.get("store_name", "Unknown"),
                "spend": spend,
                "revenue": safe_float(ad.get("revenue", 0)),
                "cpa": safe_float(ad.get("cpa", 0)),
                "conversions": safe_float(ad.get("purchases_ct", 0)),
                "cpc": safe_float(ad.get("cpc", 0)),
                "ctr": safe_float(ad.get("ctr", 0)),
                "roas": safe_float(ad.get("roas", 0)),
                "cpc_30": safe_float(ad.get("cpc_30", 0)),
                "ctr_30": safe_float(ad.get("ctr_30", 0)),
                "roas_30": safe_float(ad.get("roas_30", 0)),
                "cpc_90": safe_float(ad.get("cpc_90", 0)),
                "ctr_90": safe_float(ad.get("ctr_90", 0)),
                "roas_90": safe_float(ad.get("roas_90", 0)),
                "result": result,
                "result_30": result_30,
                "result_90": result_90,
                "benchmark_cpc": {
                    "current": safe_float(ad.get("benchmark_cpc", 0)),
                    "30": safe_float(ad.get("benchmark_cpc_30", 0)),
                    "90": safe_float(ad.get("benchmark_cpc_90", 0))
                },
                "benchmark_ctr": {
                    "current": safe_float(ad.get("benchmark_ctr", 0)),
                    "30": safe_float(ad.get("benchmark_ctr_30", 0)),
                    "90": safe_float(ad.get("benchmark_ctr_90", 0))
                },
                "benchmark_roas": {
                    "current": safe_float(ad.get("benchmark_roas", 0)),
                    "30": safe_float(ad.get("benchmark_roas_30", 0)),
                    "90": safe_float(ad.get("benchmark_roas_90", 0))
                },
                "total_score": total_score
            }
            try:
                content_json = json.dumps(content)
            except Exception as e:
                logger.error(f"JSON serialization error for ad_id {ad_id}: {e}")
                continue
            embedding = get_embedding(content_json)
            if embedding:
                try:
                    embedded_content_collection.insert_one({
                        "content": content_json,
                        "embedding": embedding,
                        "metadata": {"ad_id": content["ad_id"], "campaign_name": content["campaign_name"]}
                    })
                except Exception as e:
                    logger.error(f"MongoDB insert error for ad_id {ad_id}: {e}")
                    continue
            else:
                logger.warning(f"Skipping ad_id {ad_id} due to embedding failure")
        except Exception as e:
            logger.error(f"Error processing ad_id {ad_id}: {e}")
        processed += 1
    logger.info(f"Processed {len(batch) - skipped_count} ads in batch, skipped {skipped_count} ads with non-null store_deleted_at")

def generate_embeddings():
    """Generate embeddings for ad_records_7 after clearing embedded_content."""
    try:
        # Clear embedded_content collection
        embedded_content_collection.delete_many({})
        logger.info("Cleared embedded_content collection")

        # Reset checkpoint
        save_checkpoint(0, None)
        logger.info("Checkpoint reset")

        total_ads = ad_collection.count_documents({})
        logger.info(f"Processing {total_ads} ads for embeddings")

        processed = 0
        batch = []
        cursor = ad_collection.find({})
        for ad in cursor:
            ad_id = str(ad.get("ad_id", "Unknown"))
            batch.append(ad)
            if len(batch) >= BATCH_SIZE:
                process_batch(batch, processed, total_ads)
                processed += len(batch)
                batch = []
                if processed % 100 == 0:
                    save_checkpoint(processed, ad_id)
            if processed >= total_ads:
                break
        if batch:
            process_batch(batch, processed, total_ads)
            processed += len(batch)
            save_checkpoint(processed, ad_id)
        logger.info(f"Embedding generation complete, processed {processed} ads")
    except Exception as e:
        logger.error(f"Embedding update error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    generate_embeddings()