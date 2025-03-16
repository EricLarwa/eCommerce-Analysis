import pandas as pd
import kagglehub
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
import logging
import pymongo

pg_engine = create_engine('postgresql://username:password@localhost:5432/ecommerce_db')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ecommerce_analysis.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")
Base = declarative_base()

def make_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("ecommerce_analysis.log"), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    return logger
def get_path():
    path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")
    return path

def setup_postgresql():
    engine = pg_engine
    Base.metadata.create_all(engine)
    logger.info("PostgreSQL tables created")
    return engine

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
def process_batch(path):

    df = pd.read_csv(path)

    df[['event_time', 'user_id', 'product_id', 'price', 'event_type']].to_sql(
        'transactions', pg_engine, if_exists='append', index=False)

    preferences = df.groupby('user_id')['category_id'].value_counts().reset_index(name='count')
    preferences_records = preferences.to_dict('records')

    preferences_collection.insert_many(preferences_records)
