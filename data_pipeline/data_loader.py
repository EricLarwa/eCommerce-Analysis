import pandas as pd
from sqlalchemy import text
from tqdm import tqdm
import time
from utils import make_logger
from database.postgres_manager import get_engine
from database.postgres_manager import get_session
from database.mongodb_manager import setup_mongodb
from utils import get_path

logger = make_logger()

file_path = get_path()

def preprocess_data(file_path, batch_size=1000):

    pg_engine = get_engine()
    session = get_session()

    try:
        mongo_client, mongo_db = setup_mongodb()
        if mongo_db is None:
            logger.error("Failed to connect to MongoDB")
            return

        preferences_collection = mongo_db['user_preferences']

        chunk_iter = pd.read_csv(file_path, chunksize=batch_size)

        total_processed = 0
        start_time = time.time()

        for chunk in tqdm(chunk_iter):
            chunk['event_time'] = pd.to_datetime(chunk['event_time'])

            chunk['category_id'].fillna('unkown', inplace=True)
            chunk['category_code'].fillna('unkown', inplace=True)
            chunk['brand'].fillna('unkown', inplace=True)

            products_df = chunk[['product_id', 'category_id', 'category_code', 'brand', 'price']].drop_duplicates()

            user_stats = chunk.groupby('user_id').agg(
                first_seen=('event_time', 'min'),
                last_seen=('event_time', 'max')
            ).reset_index()

            product_event_counts = chunk.groupby(['product_id', 'event_type']).size().unstack(fill_value=0).reset_index()
            if 'view' not in product_event_counts.columns:
                product_event_counts['view'] = 0
            if 'cart' not in product_event_counts.columns:
                product_event_counts['cart'] = 0
            if 'purchase' not in product_event_counts.columns:
                product_event_counts['purchase'] = 0

            # Put batch into Postgresql
            with pg_engine.begin() as conn:
                chunk[['event_time', 'event_type', 'user_id', 'product_id', 'price']].to_sql(
                    'transactions', conn, if_exists='append', index=False)

                product_stats = pd.merge(
                    products_df,
                    product_event_counts,
                    on='product_id',
                    how='left'
                )
                product_stats.fillna(0, inplace=True)
                product_stats.rename(
                    columns={'view': 'view_count', 'cart': 'cart_count', 'purchase': 'purchase_count'},
                    inplace=True
                )

                product_stats.to_sql('temp_products', conn, if_exists='replace', index=False)
                conn.execute(text("""
                    INSERT INTO products (product_id, category_id, category_code, brand, price, view_count, cart_count, purchase_count)
                    SELECT product_id, category_id, category_code, brand, price, view_count, cart_count, purchase_count
                    FROM temp_products
                    ON CONFLICT (product_id) DO UPDATE SET
                        view_count = products.view_count + EXCLUDED.view_count,
                        cart_count = products.cart_count + EXCLUDED.cart_count,
                        purchase_count = products.purchase_count + EXCLUDED.purchase_count   
                """))

                user_stats.to_sql('temp_users', conn, if_exists='replace', index=False)
                conn.execute(text("""
                    INSERT INTO users (user_id, first_seen, last_seen)
                    SELECT user_id, first_seen, last_seen
                    FROM temp_users
                    ON CONFLICT (user_id) DO UPDATE SET
                        first_seen = LEAST(users.first_seen, EXCLUDED.first_seen),
                        last_seen = GREATEST(users.last_seen, EXCLUDED.last_seen)
                """))

                purchase_prefs= chunk[chunk['event_type'] == 'purchase'].groupby('user_id')['category_id'].apply(list).reset_index()
                view_prefs = chunk[chunk['event_type'] == 'view'].groupby('user_id')['category_id'].apply(list).reset_index()

                for _, row in purchase_prefs.iterrows():
                    user_id = row['user_id']
                    categories = row['category_id']

                    preferences_collection.update_one(
                        {'user_id': user_id},
                        {
                            '$push': {'purchased_categories': {'$each': categories}},
                            '$currentDate': {'last_updated': True}
                        },
                        upsert=True
                    )

                for _, row in view_prefs.iterrows():
                    user_id = row['user_id']
                    categories = row['category_id']

                    preferences_collection.update_one(
                        {'user_id': user_id},
                        {
                            '$push': {'viewed_categories': {'$each': categories}},
                            '$currentDate': {'last_updated': True}
                        },
                        upsert=True
                    )

                    total_processed += len(chunk)
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {total_processed} records in {elapsed:.2f} seconds")

        session.close()
        mongo_client.close()
        logger.info(f"Preprocessing complete - total records: {total_processed}")

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise

