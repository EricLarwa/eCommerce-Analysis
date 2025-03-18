import pymongo
from utils import make_logger

logger = make_logger()

def setup_mongodb():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")

        client.server_info()

        db = client["ecommerce_preferences"]

        if "user_preferences" not in db.list_collection_names():
            db.create_collection("user_preferences")

        if "recommendations" not in db.list_collection_names():
            db.create_collection("recommendations")

        if "user_segments" not in db.list_collection_names():
            db.create_collection("user_segments")

        logger.info("MongoDB collections created")
        return client, db
    except pymongo.errors.ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection error: {e}")
        return None, None
    except Exception as e:
        logger.error(f"MongoDB setup error: {e}")
        raise

