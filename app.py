# import os
# import json
# import logging
# import re
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from pymongo import MongoClient
# from dotenv import load_dotenv
# from typing import List, Dict, Any
# import uvicorn
# from pymongo.errors import ConnectionFailure
# from openai import OpenAI
# from contextlib import asynccontextmanager

# # Setup logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
#     logging.FileHandler('app.log'),
#     logging.StreamHandler()
# ])
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# MONGODB_CONNECTION_URI = os.getenv("MONGODB_CONNECTION_URI")
# MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "ad_performance_db")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# API_PORT = int(os.getenv("API_PORT", 8000))

# # Validate environment variables
# if not MONGODB_CONNECTION_URI:
#     logger.error("MONGODB_CONNECTION_URI is not set in .env")
#     exit(1)
# if not OPENAI_API_KEY:
#     logger.warning("OPENAI_API_KEY is not set in .env; some features may be limited")

# # MongoDB connection
# try:
#     mongo_client = MongoClient(MONGODB_CONNECTION_URI, serverSelectionTimeoutMS=5000)
#     mongo_client.admin.command('ping')
#     db = mongo_client[MONGODB_DATABASE_NAME]
#     embedded_content_collection = db["embedded_content"]
#     logger.info(f"Connected to MongoDB: {MONGODB_CONNECTION_URI}, database: {MONGODB_DATABASE_NAME}, collection: embedded_content")
#     count = embedded_content_collection.count_documents({"content": {"$exists": True}})
#     logger.info(f"Found {count} documents with content field")
#     sample_doc = embedded_content_collection.find_one({"content": {"$exists": True}})
#     logger.info(f"Sample document: {str(sample_doc)[:500] if sample_doc else 'None'}")
# except ConnectionFailure as e:
#     logger.error(f"MongoDB connection failure: {e}")
#     exit(1)
# except Exception as e:
#     logger.error(f"Unexpected MongoDB error: {e}")
#     exit(1)

# # FastAPI app with lifespan
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("Application startup")
#     yield
#     logger.info("Application shutdown")
#     mongo_client.close()
#     logger.info("MongoDB connection closed")

# app = FastAPI(title="Ad Performance API", lifespan=lifespan)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8501", "http://localhost:8502"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Request model
# class QueryRequest(BaseModel):
#     query: str

# # Response models
# class AdResponse(BaseModel):
#     ad_id: str
#     type: str
#     campaign_name: str
#     ad_name: str
#     adset_name: str
#     store: str
#     spend: float
#     revenue: float
#     cpa: float
#     conversions: float
#     cpc: float
#     ctr: float
#     roas: float
#     total_score: int
#     result: str
#     result_30: str
#     result_90: str

# class AnalysisResponse(BaseModel):
#     trends: Dict[str, Any]
#     wins: List[str]
#     concerns: List[str]
#     insights: List[str]
#     best_ads: List[AdResponse]

# class SummaryResponse(BaseModel):
#     total_ads: int
#     evergreen_count: int
#     test_count: int
#     stores: List[str]
#     metrics: Dict[str, float]
#     top_ad: AdResponse
#     trends: Dict[str, Any]
#     wins: List[str]
#     concerns: List[str]
#     insights: List[str]
#     message: str

# class SingleAdSummaryResponse(BaseModel):
#     ad: AdResponse
#     performance: Dict[str, str]
#     wins: List[str]
#     concerns: List[str]
#     insights: List[str]
#     message: str

# def parse_content(content_str: str) -> Dict:
#     """Parse content JSON string."""
#     try:
#         return json.loads(content_str) if isinstance(content_str, str) else content_str
#     except Exception as e:
#         logger.error(f"Failed to parse content: {e}")
#         return {}

# def prioritize_ads(ad_type: str = None, min_spend: float = 0, max_spend: float = float('inf')) -> List[Dict]:
#     """Prioritize ads based on type, spend, and metrics with robust debugging."""
#     try:
#         query = {"content": {"$exists": True}}
#         ads = []
#         doc_count = 0
#         parse_fail_count = 0
#         type_mismatch_count = 0
#         spend_invalid_count = 0
#         logger.info(f"Executing query: {query} for ad_type: {ad_type}, min_spend: {min_spend}, max_spend: {max_spend}")
        
#         for doc in embedded_content_collection.find(query):
#             doc_count += 1
#             content_raw = doc.get("content")
#             if content_raw is None:
#                 parse_fail_count += 1
#                 logger.debug(f"Skipped doc due to None content: doc_id {doc.get('_id', 'unknown')}")
#                 continue
                
#             try:
#                 content = parse_content(content_raw)
#                 if not content or not isinstance(content, dict):
#                     parse_fail_count += 1
#                     logger.debug(f"Parse failed for doc_id {doc.get('_id', 'unknown')}, content: {str(content_raw)[:100]}")
#                     continue
                
#                 content_type = content.get("type", None)
#                 if content_type is None or not isinstance(content_type, str):
#                     type_mismatch_count += 1
#                     logger.debug(f"Type mismatch (None or non-string) for ad_id {content.get('ad_id', 'unknown')}: {content_type}")
#                     continue
#                 content_type = content_type.strip()
                
#                 if ad_type and content_type.lower() != ad_type.lower():
#                     type_mismatch_count += 1
#                     logger.debug(f"Type mismatch for ad_id {content.get('ad_id', 'unknown')}: {content_type} != {ad_type}")
#                     continue
                
#                 try:
#                     spend = float(content.get("spend", 0))
#                 except (ValueError, TypeError):
#                     spend_invalid_count += 1
#                     logger.debug(f"Invalid spend for ad_id {content.get('ad_id', 'unknown')}: {content.get('spend', 'None')}")
#                     continue
                
#                 if not (min_spend <= spend <= max_spend):
#                     spend_invalid_count += 1
#                     logger.debug(f"Spend {spend} out of range [{min_spend}, {max_spend}] for ad_id {content.get('ad_id', 'unknown')}")
#                     continue
                
#                 ads.append({
#                     "ad_id": content.get("ad_id", ""),
#                     "ad_name": content.get("ad_name", ""),
#                     "spend": spend,
#                     "total_score": float(content.get("total_score", 0)),
#                     "roas": float(content.get("roas", 0)),
#                     "cpc": float(content.get("cpc", float('inf'))),
#                     "ctr": float(content.get("ctr", 0)),
#                     "type": content.get("type", ""),
#                     "campaign_name": content.get("campaign_name", ""),
#                     "adset_name": content.get("adset_name", ""),
#                     "store": content.get("store", ""),
#                     "revenue": float(content.get("revenue", 0)),
#                     "cpa": float(content.get("cpa", 0)),
#                     "conversions": float(content.get("conversions", 0)),
#                     "result": content.get("result", ""),
#                     "result_30": content.get("result_30", ""),
#                     "result_90": content.get("result_90", "")
#                 })
#                 logger.debug(f"Added ad_id {content.get('ad_id', 'unknown')} with type {content_type}, spend {spend}")
                
#             except Exception as e:
#                 parse_fail_count += 1
#                 logger.debug(f"Error processing doc_id {doc.get('_id', 'unknown')}: {e}, content: {str(content_raw)[:100]}")
#                 continue
        
#         sorted_ads = sorted(
#             ads,
#             key=lambda x: (
#                 x["total_score"],  # Ascending, as scores are negative
#                 x["roas"],
#                 -x["cpc"],
#                 x["ctr"]
#             ),
#             reverse=True
#         )
        
#         logger.info(f"Results: {doc_count} docs queried, {parse_fail_count} parse failures, "
#                     f"{type_mismatch_count} type mismatches, {spend_invalid_count} spend issues, "
#                     f"{len(ads)} valid {ad_type} ads found, returning {len(sorted_ads[:10])}")
#         return sorted_ads[:10]
#     except Exception as e:
#         logger.error(f"Critical error in prioritize_ads: {e}")
#         return []

# def analyze_trends(ads: List[Dict]) -> Dict:
#     """Analyze trends based on current, 30-day, and 90-day metrics."""
#     try:
#         if not ads:
#             return {}
#         cpc_trends = []
#         ctr_trends = []
#         roas_trends = []
#         for ad in ads[:10]:
#             cpc_current = ad.get("cpc", 0)
#             cpc_30 = ad.get("cpc_30", 0)
#             cpc_90 = ad.get("cpc_90", 0)
#             ctr_current = ad.get("ctr", 0)
#             ctr_30 = ad.get("ctr_30", 0)
#             ctr_90 = ad.get("ctr_90", 0)
#             roas_current = ad.get("roas", 0)
#             roas_30 = ad.get("roas_30", 0)
#             roas_90 = ad.get("roas_90", 0)
#             cpc_trend = (f"Ad {ad['ad_id']}: CPC ${cpc_current:.2f} (current) vs ${cpc_30:.2f} "
#                         f"(30-day, {'↑' if cpc_current > cpc_30 else '↓'}{(cpc_current - cpc_30)/cpc_30*100:.1f}%) vs "
#                         f"${cpc_90:.2f} (90-day, {'↑' if cpc_current > cpc_90 else '↓'}{(cpc_current - cpc_90)/cpc_90*100:.1f}%)"
#                         if cpc_30 and cpc_90 else f"Ad {ad['ad_id']}: CPC ${cpc_current:.2f}")
#             ctr_trend = (f"Ad {ad['ad_id']}: CTR {ctr_current:.2f}% (current) vs {ctr_30:.2f}% "
#                         f"(30-day, {'↑' if ctr_current > ctr_30 else '↓'}{(ctr_current - ctr_30)/ctr_30*100:.1f}%) vs "
#                         f"{ctr_90:.2f}% (90-day, {'↑' if ctr_current > ctr_90 else '↓'}{(ctr_current - ctr_90)/ctr_90*100:.1f}%)"
#                         if ctr_30 and ctr_90 else f"Ad {ad['ad_id']}: CTR {ctr_current:.2f}%")
#             roas_trend = (f"Ad {ad['ad_id']}: ROAS {roas_current:.2f} (current) vs {roas_30:.2f} "
#                          f"(30-day, {'↑' if roas_current > roas_30 else '↓'}{(roas_current - roas_30)/roas_30*100:.1f}%) vs "
#                          f"{roas_90:.2f} (90-day, {'↑' if roas_current > roas_90 else '↓'}{(roas_current - roas_90)/roas_90*100:.1f}%)"
#                          if roas_30 and roas_90 else f"Ad {ad['ad_id']}: ROAS {roas_current:.2f}")
#             cpc_trends.append(cpc_trend)
#             ctr_trends.append(ctr_trend)
#             roas_trends.append(roas_trend)
#         return {"cpc": cpc_trends, "ctr": ctr_trends, "roas": roas_trends}
#     except Exception as e:
#         logger.error(f"Error analyzing trends: {e}")
#         return {}

# def get_wins_concerns_insights(ad: Dict) -> tuple:
#     """Generate wins, concerns, and insights for a single ad."""
#     try:
#         wins = []
#         concerns = []
#         insights = []
#         ad_id = ad.get("ad_id", "Unknown")
#         roas = ad.get("roas", 0)
#         cpc = ad.get("cpc", 0)
#         ctr = ad.get("ctr", 0)
#         conversions = ad.get("conversions", 0)
#         spend = ad.get("spend", 0)
#         total_score = ad.get("total_score", 0)
#         benchmark_roas = ad.get("benchmark_roas", {}).get("current", 1)
#         benchmark_cpc = ad.get("benchmark_cpc", {}).get("current", 1)
#         benchmark_ctr = ad.get("benchmark_ctr", {}).get("current", 1)
#         # Wins
#         if roas > benchmark_roas * 5:
#             wins.append(f"High ROAS ({roas:.2f}, {roas/benchmark_roas:.1f}x benchmark)")
#         if ctr > benchmark_ctr * 5:
#             wins.append(f"Strong CTR ({ctr:.2f}%, {ctr/benchmark_ctr:.1f}x benchmark)")
#         if total_score >= 2:
#             wins.append(f"High total_score ({total_score})")
#         # Concerns
#         if cpc > benchmark_cpc * 10:
#             concerns.append(f"High CPC (${cpc:.2f}, {cpc/benchmark_cpc:.1f}x benchmark)")
#         if conversions < 5 and spend >= 200:
#             concerns.append(f"Low conversions ({conversions} for ${spend:.2f} spend)")
#         if total_score <= -2:
#             concerns.append(f"Low total_score ({total_score})")
#         # Insights
#         if roas > benchmark_roas * 5 and cpc > benchmark_cpc * 10:
#             insights.append("Optimize targeting to reduce CPC while maintaining high ROAS")
#         if ctr > benchmark_ctr * 5 and conversions < 5:
#             insights.append("Improve landing page to convert high CTR into purchases")
#         if total_score <= -2:
#             insights.append("Review ad performance metrics; consider pausing or revising creative")
#         return wins, concerns, insights
#     except Exception as e:
#         logger.error(f"Error generating wins/concerns/insights: {e}")
#         return [], [], []

# def get_ads_summary() -> Dict:
#     """Generate summary of all ads."""
#     try:
#         ads = []
#         stores = set()
#         total_spend = 0
#         total_revenue = 0
#         total_cpc = 0
#         total_ctr = 0
#         total_roas = 0
#         valid_ads = 0
#         evergreen_count = 0
#         test_count = 0
#         for doc in embedded_content_collection.find({"content": {"$exists": True}}):
#             content = parse_content(doc.get("content", "{}"))
#             if not content:
#                 continue
#             ads.append(content)
#             stores.add(content.get("store", "Unknown"))
#             total_spend += content.get("spend", 0)
#             total_revenue += content.get("revenue", 0)
#             cpc = content.get("cpc", 0)
#             ctr = content.get("ctr", 0)
#             roas = content.get("roas", 0)
#             if cpc > 0:
#                 total_cpc += cpc
#                 valid_ads += 1
#             total_ctr += ctr
#             total_roas += roas
#             if content.get("type") == "Evergreen":
#                 evergreen_count += 1
#             elif content.get("type") == "Test":
#                 test_count += 1
#         total_ads = len(ads)
#         avg_cpc = total_cpc / valid_ads if valid_ads > 0 else 0
#         avg_ctr = total_ctr / total_ads if total_ads > 0 else 0
#         avg_roas = total_roas / total_ads if total_ads > 0 else 0
#         top_ad = max(ads, key=lambda x: x.get("total_score", 0), default={})
#         trends = analyze_trends(ads)
#         wins, concerns, insights = get_wins_concerns_insights(top_ad)
#         return {
#             "total_ads": total_ads,
#             "evergreen_count": evergreen_count,
#             "test_count": test_count,
#             "stores": list(stores),
#             "metrics": {
#                 "total_spend": total_spend,
#                 "total_revenue": total_revenue,
#                 "avg_cpc": avg_cpc,
#                 "avg_ctr": avg_ctr * 100,
#                 "avg_roas": avg_roas
#             },
#             "top_ad": top_ad,
#             "trends": trends,
#             "wins": wins,
#             "concerns": concerns,
#             "insights": insights,
#             "message": f"Summarized {total_ads} ads"
#         }
#     except Exception as e:
#         logger.error(f"Error generating summary: {e}")
#         return {}

# def get_single_ad_summary(ad_id: str) -> Dict:
#     """Generate summary for a specific ad."""
#     try:
#         doc = embedded_content_collection.find_one({"content": {"$regex": f'"ad_id":\\s*"{ad_id}"'}})
#         if not doc:
#             raise HTTPException(status_code=404, detail=f"Ad {ad_id} not found")
#         content = parse_content(doc.get("content", "{}"))
#         if not content:
#             raise HTTPException(status_code=500, detail="Failed to parse ad content")
#         wins, concerns, insights = get_wins_concerns_insights(content)
#         performance = {
#             "cpc_vs_benchmark": f"${content.get('cpc', 0):.2f} ({content.get('cpc', 0)/content.get('benchmark_cpc', {}).get('current', 1):.1f}x benchmark)",
#             "ctr_vs_benchmark": f"{content.get('ctr', 0):.2f}% ({content.get('ctr', 0)/content.get('benchmark_ctr', {}).get('current', 1):.1f}x benchmark)",
#             "roas_vs_benchmark": f"{content.get('roas', 0):.2f} ({content.get('roas', 0)/content.get('benchmark_roas', {}).get('current', 1):.1f}x benchmark)",
#             "trend_cpc": (f"CPC ${content.get('cpc', 0):.2f} (current) vs ${content.get('cpc_30', 0):.2f} "
#                          f"(30-day, {'↑' if content.get('cpc', 0) > content.get('cpc_30', 0) else '↓'}{(content.get('cpc', 0) - content.get('cpc_30', 0))/content.get('cpc_30', 0)*100:.1f}%)"
#                          if content.get('cpc_30', 0) else f"CPC ${content.get('cpc', 0):.2f}"),
#             "trend_ctr": (f"CTR {content.get('ctr', 0):.2f}% (current) vs {content.get('ctr_30', 0):.2f}% "
#                          f"(30-day, {'↑' if content.get('ctr', 0) > content.get('ctr_30', 0) else '↓'}{(content.get('ctr', 0) - content.get('ctr_30', 0))/content.get('ctr_30', 0)*100:.1f}%)"
#                          if content.get('ctr_30', 0) else f"CTR {content.get('ctr', 0):.2f}%"),
#             "trend_roas": (f"ROAS {content.get('roas', 0):.2f} (current) vs {content.get('roas_30', 0):.2f} "
#                           f"(30-day, {'↑' if content.get('roas', 0) > content.get('roas_30', 0) else '↓'}{(content.get('roas', 0) - content.get('roas_30', 0))/content.get('roas_30', 0)*100:.1f}%)"
#                           if content.get('roas_30', 0) else f"ROAS {content.get('roas', 0):.2f}")
#         }
#         return {
#             "ad": content,
#             "performance": performance,
#             "wins": wins,
#             "concerns": concerns,
#             "insights": insights,
#             "message": f"Summary for ad {ad_id}"
#         }
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         logger.error(f"Error generating summary for ad {ad_id}: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# def get_intent_from_openai(query: str) -> dict:
#     """Use OpenAI to extract intent and parameters from the query."""
#     client = OpenAI(api_key=OPENAI_API_KEY)
#     prompt = f"""
# You are an intent extraction assistant for an ad analytics tool.
# Given a user query, return a JSON with 'intent' and any relevant parameters.
# Possible intents (choose the closest match, always use one of these exactly): 
# - summary_of_ads
# - single_ad_summary
# - prioritize_evergreen
# - prioritize_microdata
# - trends

# If the query is about a specific ad, use 'single_ad_summary' and include 'ad_id'.
# If the query is about evergreen ads, use 'prioritize_evergreen'.
# If the query is about microdata/test ads, use 'prioritize_microdata'.
# If the query is about trends, use 'trends'.
# If the query is about a general summary, use 'summary_of_ads'.

# Example:
# User: summary of ads
# {{"intent": "summary_of_ads"}}
# User: summary of ad_id 12345
# {{"intent": "single_ad_summary", "ad_id": "12345"}}
# User: prioritize evergreen
# {{"intent": "prioritize_evergreen"}}
# User: prioritize microdata
# {{"intent": "prioritize_microdata"}}
# User: show trends
# {{"intent": "trends"}}
# User: {query}
# Return only the JSON.
# """
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=100,
#             temperature=0
#         )
#         content = response.choices[0].message.content
#         match = re.search(r"\{.*\}", content, re.DOTALL)
#         if match:
#             intent_data = json.loads(match.group(0))
#             logger.info(f"OpenAI intent_data: {intent_data}")
#             return intent_data
#         logger.warning(f"Invalid OpenAI response format: {content}")
#         return {"intent": "unknown"}
#     except Exception as e:
#         logger.error(f"OpenAI error: {e}")
#         return {"intent": "unknown"}

# @app.post("/api/message", response_model=Dict[str, Any])
# async def process_query(request: QueryRequest):
#     try:
#         query = request.query.strip()
#         logger.info(f"Processing query: {query}")

#         intent_data = get_intent_from_openai(query)
#         intent_aliases = {
#             "top_evergreen_ads": "prioritize_evergreen",
#             "top 10 evergreen ads": "prioritize_evergreen",
#             "evergreen_ads": "prioritize_evergreen",
#         }
#         intent = intent_data.get("intent", "unknown")
#         intent = intent_aliases.get(intent, intent)

#         if intent == "summary_of_ads":
#             summary = get_ads_summary()
#             return summary

#         if intent == "single_ad_summary":
#             ad_id = intent_data.get("ad_id")
#             if not ad_id:
#                 raise HTTPException(status_code=400, detail="ad_id not specified")
#             return get_single_ad_summary(ad_id)

#         if intent == "prioritize_evergreen":
#             ads = prioritize_ads(ad_type="Evergreen", min_spend=1000)
#             return {"ads": ads, "message": f"Prioritized {len(ads)} Evergreen ads"}

#         if intent == "prioritize_microdata":
#             ads = prioritize_ads(ad_type="Test", min_spend=200, max_spend=999.99)
#             return {"ads": ads, "message": f"Prioritized {len(ads)} Test ads"}

#         if intent == "trends":
#             evergreen_ads = prioritize_ads(ad_type="Evergreen", min_spend=1000)
#             test_ads = prioritize_ads(ad_type="Test", min_spend=200, max_spend=999.99)
#             all_ads = evergreen_ads + test_ads
#             trends = analyze_trends(all_ads)
#             wins, concerns, insights = get_wins_concerns_insights(all_ads[0] if all_ads else {})
#             best_ads = sorted(all_ads, key=lambda x: x.get("total_score", 0), reverse=True)[:5]
#             return {
#                 "trends": trends,
#                 "wins": wins,
#                 "concerns": concerns,
#                 "insights": insights,
#                 "best_ads": best_ads,
#                 "message": f"Analyzed {len(all_ads)} ads"
#             }

#         raise HTTPException(status_code=400, detail=f"Invalid query or intent: {intent}")
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         logger.error(f"Error processing query: {query}: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=API_PORT)




import os
import json
import logging
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from typing import List, Dict, Any
import uvicorn
from pymongo.errors import ConnectionFailure
from openai import OpenAI
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler('app.log'),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGODB_CONNECTION_URI = os.getenv("MONGODB_CONNECTION_URI")
MONGODB_DATABASE_NAME = os.getenv("MONGODB_DATABASE_NAME", "ad_performance_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_PORT = int(os.getenv("API_PORT", 8000))

# Validate environment variables
if not MONGODB_CONNECTION_URI:
    logger.error("MONGODB_CONNECTION_URI is not set in .env")
    exit(1)
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set in .env; required for summarization")
    exit(1)

# MongoDB connection
try:
    mongo_client = MongoClient(MONGODB_CONNECTION_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    db = mongo_client[MONGODB_DATABASE_NAME]
    embedded_content_collection = db["embedded_content"]
    logger.info(f"Connected to MongoDB: {MONGODB_CONNECTION_URI}, database: {MONGODB_DATABASE_NAME}, collection: embedded_content")
    count = embedded_content_collection.count_documents({"content": {"$exists": True}})
    logger.info(f"Found {count} documents with content field")
    sample_doc = embedded_content_collection.find_one({"content": {"$exists": True}})
    logger.info(f"Sample document: {str(sample_doc)[:500] if sample_doc else 'None'}")
except ConnectionFailure as e:
    logger.error(f"MongoDB connection failure: {e}")
    exit(1)
except Exception as e:
    logger.error(f"Unexpected MongoDB error: {e}")
    exit(1)

# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup")
    yield
    logger.info("Application shutdown")
    mongo_client.close()
    logger.info("MongoDB connection closed")

app = FastAPI(title="Ad Performance API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:8502"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    query: str

# Response models
class AdResponse(BaseModel):
    ad_id: str
    type: str
    campaign_name: str
    ad_name: str
    adset_name: str
    store: str
    spend: float
    revenue: float
    cpa: float
    conversions: float
    cpc: float
    ctr: float
    roas: float
    total_score: int
    result: str
    result_30: str
    result_90: str

class AnalysisResponse(BaseModel):
    trends: Dict[str, Any]
    wins: List[str]
    concerns: List[str]
    insights: List[str]
    best_ads: List[AdResponse]

class SummaryResponse(BaseModel):
    total_ads: int
    evergreen_count: int
    test_count: int
    stores: List[str]
    metrics: Dict[str, float]
    top_ad: AdResponse
    trends: Dict[str, Any]
    wins: List[str]
    concerns: List[str]
    insights: List[str]
    message: str

class SingleAdSummaryResponse(BaseModel):
    ad: AdResponse
    performance: Dict[str, str]
    wins: List[str]
    concerns: List[str]
    insights: List[str]
    message: str

class LLMResponse(BaseModel):
    text: str
    data: Dict[str, Any]

def parse_content(content_str: str) -> Dict:
    """Parse content JSON string."""
    try:
        return json.loads(content_str) if isinstance(content_str, str) else content_str
    except Exception as e:
        logger.error(f"Failed to parse content: {e}")
        return {}

def prioritize_ads(ad_type: str = None, min_spend: float = 0, max_spend: float = float('inf')) -> List[Dict]:
    """Prioritize ads based on type, spend, and metrics with robust debugging."""
    try:
        query = {"content": {"$exists": True}}
        ads = []
        doc_count = 0
        parse_fail_count = 0
        type_mismatch_count = 0
        spend_invalid_count = 0
        logger.info(f"Executing query: {query} for ad_type: {ad_type}, min_spend: {min_spend}, max_spend: {max_spend}")
        
        for doc in embedded_content_collection.find(query):
            doc_count += 1
            content_raw = doc.get("content")
            if content_raw is None:
                parse_fail_count += 1
                logger.debug(f"Skipped doc due to None content: doc_id {doc.get('_id', 'unknown')}")
                continue
                
            try:
                content = parse_content(content_raw)
                if not content or not isinstance(content, dict):
                    parse_fail_count += 1
                    logger.debug(f"Parse failed for doc_id {doc.get('_id', 'unknown')}, content: {str(content_raw)[:100]}")
                    continue
                
                content_type = content.get("type", None)
                if content_type is None or not isinstance(content_type, str):
                    type_mismatch_count += 1
                    logger.debug(f"Type mismatch (None or non-string) for ad_id {content.get('ad_id', 'unknown')}: {content_type}")
                    continue
                content_type = content_type.strip()
                
                if ad_type and content_type.lower() != ad_type.lower():
                    type_mismatch_count += 1
                    logger.debug(f"Type mismatch for ad_id {content.get('ad_id', 'unknown')}: {content_type} != {ad_type}")
                    continue
                
                try:
                    spend = float(content.get("spend", 0))
                except (ValueError, TypeError):
                    spend_invalid_count += 1
                    logger.debug(f"Invalid spend for ad_id {content.get('ad_id', 'unknown')}: {content.get('spend', 'None')}")
                    continue
                
                if not (min_spend <= spend <= max_spend):
                    spend_invalid_count += 1
                    logger.debug(f"Spend {spend} out of range [{min_spend}, {max_spend}] for ad_id {content.get('ad_id', 'unknown')}")
                    continue
                
                ads.append({
                    "ad_id": content.get("ad_id", ""),
                    "ad_name": content.get("ad_name", ""),
                    "spend": spend,
                    "total_score": float(content.get("total_score", 0)),
                    "roas": float(content.get("roas", 0)),
                    "cpc": float(content.get("cpc", float('inf'))),
                    "ctr": float(content.get("ctr", 0)),
                    "type": content.get("type", ""),
                    "campaign_name": content.get("campaign_name", ""),
                    "adset_name": content.get("adset_name", ""),
                    "store": content.get("store", ""),
                    "revenue": float(content.get("revenue", 0)),
                    "cpa": float(content.get("cpa", 0)),
                    "conversions": float(content.get("conversions", 0)),
                    "result": content.get("result", ""),
                    "result_30": content.get("result_30", ""),
                    "result_90": content.get("result_90", "")
                })
                logger.debug(f"Added ad_id {content.get('ad_id', 'unknown')} with type {content_type}, spend {spend}")
                
            except Exception as e:
                parse_fail_count += 1
                logger.debug(f"Error processing doc_id {doc.get('_id', 'unknown')}: {e}, content: {str(content_raw)[:100]}")
                continue
        
        sorted_ads = sorted(
            ads,
            key=lambda x: (
                x["total_score"],  # Ascending, as scores are negative
                x["roas"],
                -x["cpc"],
                x["ctr"]
            ),
            reverse=True
        )
        
        logger.info(f"Results: {doc_count} docs queried, {parse_fail_count} parse failures, "
                    f"{type_mismatch_count} type mismatches, {spend_invalid_count} spend issues, "
                    f"{len(ads)} valid {ad_type} ads found, returning {len(sorted_ads[:10])}")
        return sorted_ads[:10]
    except Exception as e:
        logger.error(f"Critical error in prioritize_ads: {e}")
        return []

def analyze_trends(ads: List[Dict]) -> Dict:
    """Analyze trends based on current, 30-day, and 90-day metrics."""
    try:
        if not ads:
            return {}
        cpc_trends = []
        ctr_trends = []
        roas_trends = []
        for ad in ads[:10]:
            cpc_current = ad.get("cpc", 0)
            cpc_30 = ad.get("cpc_30", 0)
            cpc_90 = ad.get("cpc_90", 0)
            ctr_current = ad.get("ctr", 0)
            ctr_30 = ad.get("ctr_30", 0)
            ctr_90 = ad.get("ctr_90", 0)
            roas_current = ad.get("roas", 0)
            roas_30 = ad.get("roas_30", 0)
            roas_90 = ad.get("roas_90", 0)
            cpc_trend = (f"Ad {ad['ad_id']}: CPC ${cpc_current:.2f} (current) vs ${cpc_30:.2f} "
                        f"(30-day, {'↑' if cpc_current > cpc_30 else '↓'}{(cpc_current - cpc_30)/cpc_30*100:.1f}%) vs "
                        f"${cpc_90:.2f} (90-day, {'↑' if cpc_current > cpc_90 else '↓'}{(cpc_current - cpc_90)/cpc_90*100:.1f}%)"
                        if cpc_30 and cpc_90 else f"Ad {ad['ad_id']}: CPC ${cpc_current:.2f}")
            ctr_trend = (f"Ad {ad['ad_id']}: CTR {ctr_current:.2f}% (current) vs {ctr_30:.2f}% "
                        f"(30-day, {'↑' if ctr_current > ctr_30 else '↓'}{(ctr_current - ctr_30)/ctr_30*100:.1f}%) vs "
                        f"{ctr_90:.2f}% (90-day, {'↑' if ctr_current > ctr_90 else '↓'}{(ctr_current - cpc_90)/cpc_90*100:.1f}%)"
                        if ctr_30 and ctr_90 else f"Ad {ad['ad_id']}: CTR {ctr_current:.2f}%")
            roas_trend = (f"Ad {ad['ad_id']}: ROAS {roas_current:.2f} (current) vs {roas_30:.2f} "
                         f"(30-day, {'↑' if roas_current > roas_30 else '↓'}{(roas_current - roas_30)/roas_30*100:.1f}%) vs "
                         f"{roas_90:.2f} (90-day, {'↑' if roas_current > roas_90 else '↓'}{(roas_current - roas_90)/roas_90*100:.1f}%)"
                         if roas_30 and roas_90 else f"Ad {ad['ad_id']}: ROAS {roas_current:.2f}")
            cpc_trends.append(cpc_trend)
            ctr_trends.append(ctr_trend)
            roas_trends.append(roas_trend)
        return {"cpc": cpc_trends, "ctr": ctr_trends, "roas": roas_trends}
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        return {}

def get_wins_concerns_insights(ad: Dict) -> tuple:
    """Generate wins, concerns, and insights for a single ad."""
    try:
        wins = []
        concerns = []
        insights = []
        ad_id = ad.get("ad_id", "Unknown")
        roas = ad.get("roas", 0)
        cpc = ad.get("cpc", 0)
        ctr = ad.get("ctr", 0)
        conversions = ad.get("conversions", 0)
        spend = ad.get("spend", 0)
        total_score = ad.get("total_score", 0)
        benchmark_roas = ad.get("benchmark_roas", {}).get("current", 1)
        benchmark_cpc = ad.get("benchmark_cpc", {}).get("current", 1)
        benchmark_ctr = ad.get("benchmark_ctr", {}).get("current", 1)
        # Wins
        if roas > benchmark_roas * 5:
            wins.append(f"High ROAS ({roas:.2f}, {roas/benchmark_roas:.1f}x benchmark)")
        if ctr > benchmark_ctr * 5:
            wins.append(f"Strong CTR ({ctr:.2f}%, {ctr/benchmark_ctr:.1f}x benchmark)")
        if total_score >= 2:
            wins.append(f"High total_score ({total_score})")
        # Concerns
        if cpc > benchmark_cpc * 10:
            concerns.append(f"High CPC (${cpc:.2f}, {cpc/benchmark_cpc:.1f}x benchmark)")
        if conversions < 5 and spend >= 200:
            concerns.append(f"Low conversions ({conversions} for ${spend:.2f} spend)")
        if total_score <= -2:
            concerns.append(f"Low total_score ({total_score})")
        # Insights
        if roas > benchmark_roas * 5 and cpc > benchmark_cpc * 10:
            insights.append("Optimize targeting to reduce CPC while maintaining high ROAS")
        if ctr > benchmark_ctr * 5 and conversions < 5:
            insights.append("Improve landing page to convert high CTR into purchases")
        if total_score <= -2:
            insights.append("Review ad performance metrics; consider pausing or revising creative")
        return wins, concerns, insights
    except Exception as e:
        logger.error(f"Error generating wins/concerns/insights: {e}")
        return [], [], []

def get_ads_summary() -> Dict:
    """Generate summary of all ads."""
    try:
        ads = []
        stores = set()
        total_spend = 0
        total_revenue = 0
        total_cpc = 0
        total_ctr = 0
        total_roas = 0
        valid_ads = 0
        evergreen_count = 0
        test_count = 0
        for doc in embedded_content_collection.find({"content": {"$exists": True}}):
            content = parse_content(doc.get("content", "{}"))
            if not content:
                continue
            ads.append(content)
            stores.add(content.get("store", "Unknown"))
            total_spend += content.get("spend", 0)
            total_revenue += content.get("revenue", 0)
            cpc = content.get("cpc", 0)
            ctr = content.get("ctr", 0)
            roas = content.get("roas", 0)
            if cpc > 0:
                total_cpc += cpc
                valid_ads += 1
            total_ctr += ctr
            total_roas += roas
            if content.get("type") == "Evergreen":
                evergreen_count += 1
            elif content.get("type") == "Test":
                test_count += 1
        total_ads = len(ads)
        avg_cpc = total_cpc / valid_ads if valid_ads > 0 else 0
        avg_ctr = total_ctr / total_ads if total_ads > 0 else 0
        avg_roas = total_roas / total_ads if total_ads > 0 else 0
        top_ad = max(ads, key=lambda x: x.get("total_score", 0), default={})
        trends = analyze_trends(ads)
        wins, concerns, insights = get_wins_concerns_insights(top_ad)
        return {
            "total_ads": total_ads,
            "evergreen_count": evergreen_count,
            "test_count": test_count,
            "stores": list(stores),
            "metrics": {
                "total_spend": total_spend,
                "total_revenue": total_revenue,
                "avg_cpc": avg_cpc,
                "avg_ctr": avg_ctr * 100,
                "avg_roas": avg_roas
            },
            "top_ad": top_ad,
            "trends": trends,
            "wins": wins,
            "concerns": concerns,
            "insights": insights,
            "message": f"Summarized {total_ads} ads"
        }
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return {}

def get_single_ad_summary(ad_id: str) -> Dict:
    """Generate summary for a specific ad."""
    try:
        doc = embedded_content_collection.find_one({"content": {"$regex": f'"ad_id":\\s*"{ad_id}"'}})
        if not doc:
            raise HTTPException(status_code=404, detail=f"Ad {ad_id} not found")
        content = parse_content(doc.get("content", "{}"))
        if not content:
            raise HTTPException(status_code=500, detail="Failed to parse ad content")
        wins, concerns, insights = get_wins_concerns_insights(content)
        performance = {
            "cpc_vs_benchmark": f"${content.get('cpc', 0):.2f} ({content.get('cpc', 0)/content.get('benchmark_cpc', {}).get('current', 1):.1f}x benchmark)",
            "ctr_vs_benchmark": f"{content.get('ctr', 0):.2f}% ({content.get('ctr', 0)/content.get('benchmark_ctr', {}).get('current', 1):.1f}x benchmark)",
            "roas_vs_benchmark": f"{content.get('roas', 0):.2f} ({content.get('roas', 0)/content.get('benchmark_roas', {}).get('current', 1):.1f}x benchmark)",
            "trend_cpc": (f"CPC ${content.get('cpc', 0):.2f} (current) vs ${content.get('cpc_30', 0):.2f} "
                         f"(30-day, {'↑' if content.get('cpc', 0) > content.get('cpc_30', 0) else '↓'}{(content.get('cpc', 0) - content.get('cpc_30', 0))/content.get('cpc_30', 0)*100:.1f}%)"
                         if content.get('cpc_30', 0) else f"CPC ${content.get('cpc', 0):.2f}"),
            "trend_ctr": (f"CTR {content.get('ctr', 0):.2f}% (current) vs {content.get('ctr_30', 0):.2f}% "
                         f"(30-day, {'↑' if content.get('ctr', 0) > content.get('ctr_30', 0) else '↓'}{(content.get('ctr', 0) - content.get('ctr_30', 0))/content.get('ctr_30', 0)*100:.1f}%)"
                         if content.get('ctr_30', 0) else f"CTR {content.get('ctr', 0):.2f}%"),
            "trend_roas": (f"ROAS {content.get('roas', 0):.2f} (current) vs {content.get('roas_30', 0):.2f} "
                          f"(30-day, {'↑' if content.get('roas', 0) > content.get('roas_30', 0) else '↓'}{(content.get('roas', 0) - content.get('roas_30', 0))/content.get('roas_30', 0)*100:.1f}%)"
                          if content.get('roas_30', 0) else f"ROAS {content.get('roas', 0):.2f}")
        }
        return {
            "ad": content,
            "performance": performance,
            "wins": wins,
            "concerns": concerns,
            "insights": insights,
            "message": f"Summary for ad {ad_id}"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating summary for ad {ad_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_intent_from_openai(query: str) -> dict:
    """Use OpenAI to extract intent and parameters from the query."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
You are an intent extraction assistant for an ad analytics tool.
Given a user query, return a JSON with 'intent' and any relevant parameters.
Possible intents (choose the closest match, always use one of these exactly): 
- summary_of_ads
- single_ad_summary
- prioritize_evergreen
- prioritize_microdata
- trends

If the query is about a specific ad, use 'single_ad_summary' and include 'ad_id'.
If the query is about evergreen ads, use 'prioritize_evergreen'.
If the query is about microdata/test ads, use 'prioritize_microdata'.
If the query is about trends, use 'trends'.
If the query is about a general summary, use 'summary_of_ads'.

Example:
User: summary of ads
{{"intent": "summary_of_ads"}}
User: summary of ad_id 12345
{{"intent": "single_ad_summary", "ad_id": "12345"}}
User: prioritize evergreen
{{"intent": "prioritize_evergreen"}}
User: prioritize microdata
{{"intent": "prioritize_microdata"}}
User: show trends
{{"intent": "trends"}}
User: {query}
Return only the JSON.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        content = response.choices[0].message.content
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            intent_data = json.loads(match.group(0))
            logger.info(f"OpenAI intent_data: {intent_data}")
            return intent_data
        logger.warning(f"Invalid OpenAI response format: {content}")
        return {"intent": "unknown"}
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return {"intent": "unknown"}

def summarize_with_openai(intent: str, data: Dict, query: str, intent_data: Dict) -> str:
    """Generate a conversational summary of JSON data using OpenAI."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    data_str = json.dumps(data, indent=2)
    prompt = f"""
You are a vibrant, engaging assistant for an ad analytics tool, inspired by ChatGPT's lively tone. Summarize the provided JSON data , enthusiastic, and professional manner, tailored to the user's intent. Highlight key metrics (e.g., ad count, ROAS, spend, trends) without repeating the raw JSON structure, and use an upbeat tone (e.g., "Let’s dive in!" or "Wow, check this out!"). End with a fun, actionable call-to-action (e.g., "Ready to explore? See the details below!") if data is present, or suggest next steps if no data is found.

Intent: {intent}
Query: {query}
JSON Data: {data_str}

Guidelines:
- For 'prioritize_evergreen': Highlight number of Evergreen ads, top ad’s ROAS/spend, and sorting by performance.
- For 'prioritize_microdata': Focus on Test ads, spend range ($200-$999.99), and top ad’s metrics.
- For 'summary_of_ads': Summarize total ads, Evergreen/Test counts, total spend, avg ROAS, and top ad.
- For 'single_ad_summary': Describe ad’s type, spend, ROAS, and wins/concerns count.
- For 'trends': Emphasize CPC/CTR/ROAS trends, number of ads, and insights/wins count.
- For unknown intent: Suggest rephrasing with supported intents (summary, single ad, evergreen, microdata, trends).

Return only the plain text summary (no JSON or code blocks).
"""
    try:
        logger.debug(f"OpenAI prompt: {prompt[:500]}...")  # Log prompt snippet
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            # max_tokens=100,  # Reduced for conciseness
            temperature=0.8  # Slightly higher for creative tone
        )
        summary = response.choices[0].message.content.strip()
        logger.info(f"OpenAI summary generated: {summary}")
        return summary
    except Exception as e:
        logger.error(f"OpenAI summarization error: {e}")
        # Fallback to hardcoded narrative
        if intent == "prioritize_evergreen":
            ads = data.get("ads", [])
            if ads:
                top_ad = ads[0]
                return (
                    f"Wow, check this out! I found {len(ads)} top Evergreen ads with spends over $1000, sorted by performance, ROAS, and more. "
                    f"The leader, '{top_ad['ad_name']}' (ID: {top_ad['ad_id']}), has a stellar ROAS of {top_ad['roas']:.2f}! Ready to explore? See the details below!"
                )
            return (
                f"Oops, no Evergreen ads with spends over $1000 found. Try another query or check your data! 😊"
            )
        if intent == "prioritize_microdata":
            ads = data.get("ads", [])
            if ads:
                top_ad = ads[0]
                return (
                    f"Let’s dive in! I’ve got {len(ads)} Test ads with spends between $200 and $999.99, ranked by performance. "
                    f"Top pick '{top_ad['ad_name']}' (ID: {top_ad['ad_id']}) shines with a ROAS of {top_ad['roas']:.2f}! Ready to explore? See the details below!"
                )
            return (
                f"Hmm, no Test ads in the $200-$999.99 range. Double-check your data or try another query! 😊"
            )
        if intent == "summary_of_ads":
            summary = data
            return (
                f"Big picture time! I’ve summarized {summary.get('total_ads', 0)} ads, with {summary.get('evergreen_count', 0)} Evergreen and "
                f"{summary.get('test_count', 0)} Test ads across {len(summary.get('stores', []))} stores, spending ${summary.get('metrics', {}).get('total_spend', 0):.2f}. "
                f"The top ad '{summary.get('top_ad', {}).get('ad_name', 'N/A')}' rocks a great ROAS! Ready to explore? See the details below!"
            )
        if intent == "single_ad_summary":
            ad_id = intent_data.get("ad_id", "unknown")
            ad = data.get("ad", {})
            if ad:
                return (
                    f"Zooming in on ad {ad_id}! This {ad.get('type', 'N/A')} ad, '{ad.get('ad_name', 'N/A')}', spent ${ad.get('spend', 0):.2f} "
                    f"with a ROAS of {ad.get('roas', 0):.2f} and {len(data.get('wins', []))} wins. Ready to explore? See the details below!"
                )
            return f"Sorry, ad {ad_id} wasn’t found. Check the ID or try another! 😊"
        if intent == "trends":
            trends = data.get("trends", {})
            return (
                f"Trend alert! I analyzed {data.get('message', '').split()[1]} ads, spotting {len(trends.get('cpc', []))} CPC, "
                f"{len(trends.get('ctr', []))} CTR, and {len(trends.get('roas', []))} ROAS trends. "
                f"With {len(data.get('insights', []))} insights, you’re set! Ready to explore? See the details below!"
            )
        return (
            f"Huh, I’m not sure what you meant by '{query}'. Try asking about ad summaries, single ads, Evergreen, Test ads, or trends! 😊"
        )

def format_llm_response(intent: str, data: Dict, query: str, intent_data: Dict) -> Dict:
    """Format response with a ChatGPT-generated summary."""
    try:
        summary = summarize_with_openai(intent, data, query, intent_data)
        return {"text": summary, "data": data}
    except Exception as e:
        logger.error(f"Error formatting LLM response: {e}")
        return {"text": f"Oops, something went wrong: {str(e)}. Try again? 😊", "data": data}

@app.post("/api/message", response_model=LLMResponse)
async def process_query(request: QueryRequest):
    try:
        query = request.query.strip()
        logger.info(f"Processing query: {query}")

        intent_data = get_intent_from_openai(query)
        intent_aliases = {
            "top_evergreen_ads": "prioritize_evergreen",
            "top 10 evergreen ads": "prioritize_evergreen",
            "evergreen_ads": "prioritize_evergreen",
        }
        intent = intent_data.get("intent", "unknown")
        intent = intent_aliases.get(intent, intent)
        logger.info(f"OpenAI intent_data: {intent_data}")

        data = {}
        if intent == "summary_of_ads":
            data = get_ads_summary()

        elif intent == "single_ad_summary":
            ad_id = intent_data.get("ad_id")
            if not ad_id:
                raise HTTPException(status_code=400, detail="ad_id not specified")
            data = get_single_ad_summary(ad_id)

        elif intent == "prioritize_evergreen":
            ads = prioritize_ads(ad_type="Evergreen", min_spend=1000)
            data = {"ads": ads, "message": f"Prioritized {len(ads)} Evergreen ads"}

        elif intent == "prioritize_microdata":
            ads = prioritize_ads(ad_type="Test", min_spend=200, max_spend=999.99)
            data = {"ads": ads, "message": f"Prioritized {len(ads)} Test ads"}

        elif intent == "trends":
            evergreen_ads = prioritize_ads(ad_type="Evergreen", min_spend=1000)
            test_ads = prioritize_ads(ad_type="Test", min_spend=200, max_spend=999.99)
            all_ads = evergreen_ads + test_ads
            trends = analyze_trends(all_ads)
            wins, concerns, insights = get_wins_concerns_insights(all_ads[0] if all_ads else {})
            best_ads = sorted(all_ads, key=lambda x: x.get("total_score", 0), reverse=True)[:5]
            data = {
                "trends": trends,
                "wins": wins,
                "concerns": concerns,
                "insights": insights,
                "best_ads": best_ads,
                "message": f"Analyzed {len(all_ads)} ads"
            }

        else:
            raise HTTPException(status_code=400, detail=f"Invalid query or intent: {intent}")

        return format_llm_response(intent, data, query, intent_data)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing query: {query}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=API_PORT)