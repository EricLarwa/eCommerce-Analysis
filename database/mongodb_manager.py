import pymongo
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ecommerce_analysis.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_mongodb():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["ecommerce_preferences"]

    if "user_preferences" not in db.list_collection_names():
        db.create_collection("user_preferences")

    if "recommendations" not in db.list_collection_names():
        db.create_collection("recommendations")

    if "user_segments" not in db.list_collection_names():
        db.create_collection("user_segments")

    logger.info("MongoDB collections created")
    return client, db