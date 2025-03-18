import pandas as pd
from utils import get_path
from database.postgres_manager import get_engine
from database.mongodb_manager import setup_mongodb
from data_pipeline.data_loader import preprocess_data
from data_pipeline.feature_eng import create_all_features
from sqlalchemy.ext.declarative import declarative_base
from utils import make_logger

pg_engine = get_engine()
mongo_db = setup_mongodb()
preferences_collection = mongo_db['user_references']
logger = make_logger()

Base = declarative_base()


def process_batch(path):
    """Process a batch of data and store it in the databases."""
    try:
        df = pd.read_csv(path)

        # Store transaction data in PostgreSQL
        df[['event_time', 'user_id', 'product_id', 'price', 'event_type']].to_sql(
            'transactions', pg_engine, if_exists='append', index=False)

        # Prepare and store user preferences in MongoDB
        preferences = df.groupby('user_id')['category_id'].value_counts().reset_index(name='count')
        preferences_records = preferences.to_dict('records')
        preferences_collection.insert_many(preferences_records)

        make_logger().info(f"Processed batch from {path} successfully.")
    except Exception as e:
        logger.error(f"Error processing batch from {path}: {e}")

def main():
    path = get_path()
    preprocess_data(path, batch_size=1000)

    create_all_features(pg_engine, mongo_db, days_lookback=90)



if __name__ == "__main__":
    main()