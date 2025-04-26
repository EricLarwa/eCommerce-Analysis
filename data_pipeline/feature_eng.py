import pandas as pd
import pymongo
from database.postgres_manager import get_engine
from database.mongodb_manager import setup_mongodb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from utils import make_logger

pg_engine = get_engine()
mongo_db = setup_mongodb()

logger = make_logger()

def plot_user_segmentation(df_rfm, output_dir='./Feature_output'):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    segmentation_counts= df_rfm['activity_status'].value_counts()
    plt.figure(figsize=(8,8))
    plt.pie(segmentation_counts, labels=segmentation_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('User Segmentation by Activity Status')
    plt.axis('equal')
    plt.savefig(f"{output_dir}/user_segmentation")
def create_all_features(pg_engine, mongo_db):
    """Generate all features for users, products, and interactions."""
    logger.info(f"Creating all features ")

    create_user_features(pg_engine, mongo_db)
    create_product_features(pg_engine, mongo_db)
    create_interaction_features(pg_engine, mongo_db)

    logger.info("All features created successfully")

def create_user_features(pg_engine, mongo_db):
    """Create and store user-related features."""
    logger.info("Creating user features")

    client = mongo_db[0]
    db = mongo_db[1]

    cutoff_date = datetime(2019, 10, 1)

    # RFM Analysis Query
    rfm_query = f"""
    WITH purchase_data AS (
        SELECT 
            user_id,
            MAX(event_time) as last_purchase_date,
            COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN CAST(event_time AS date) ELSE NULL END) as frequency,
            SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) as monetary
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}'
        GROUP BY user_id
    ),
    user_stats AS (
        SELECT 
            user_id,
            COUNT(*) as total_events,
            COUNT(CASE WHEN event_type = 'view' THEN 1 END) as view_count,
            COUNT(CASE WHEN event_type = 'cart' THEN 1 END) as cart_count,
            COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) as purchase_count,
            COUNT(DISTINCT product_id) as unique_products_interacted,
            COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN product_id ELSE NULL END) as unique_products_purchased
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}'
        GROUP BY user_id
    )
    SELECT 
        us.user_id,
        us.total_events,
        us.view_count,
        us.cart_count,
        us.purchase_count,
        COALESCE(pd.monetary, 0) as total_spent,
        us.unique_products_interacted,
        us.unique_products_purchased,
        CASE WHEN us.view_count > 0 THEN us.cart_count::float / us.view_count ELSE 0 END as cart_conversion_rate,
        CASE WHEN us.cart_count > 0 THEN us.purchase_count::float / us.cart_count ELSE 0 END as purchase_conversion_rate,
        CASE WHEN pd.last_purchase_date IS NOT NULL 
            THEN EXTRACT(DAY FROM NOW()::timestamp - pd.last_purchase_date::timestamp) 
            ELSE 999 
        END as days_since_last_purchase,
        COALESCE(pd.frequency, 0) as purchase_frequency,
        CASE 
            WHEN pd.last_purchase_date IS NULL THEN 'inactive'
            WHEN EXTRACT(DAY FROM NOW()::timestamp - pd.last_purchase_date::timestamp) <= 30 THEN 'active'
            ELSE 'lapsed'
        END as activity_status
    FROM user_stats us
    LEFT JOIN purchase_data pd ON us.user_id = pd.user_id
    """

    # Category Preferences Query
    category_query = f"""
    SELECT 
        e.user_id,
        e.category_id,
        COUNT(CASE WHEN e.event_type = 'view' THEN 1 END) as view_count,
        COUNT(CASE WHEN e.event_type = 'cart' THEN 1 END) as cart_count,
        COUNT(CASE WHEN e.event_type = 'purchase' THEN 1 END) as purchase_count
    FROM ecommerce_db e
    WHERE e.event_time >= '{cutoff_date.isoformat()}'
    GROUP BY e.user_id, e.category_id
    """

    df_category = pd.read_sql(category_query, pg_engine)

    # Time Pattern Analysis Query
    time_query = f"""
    SELECT 
        user_id,
        EXTRACT(HOUR FROM event_time::timestamp) as hour_of_day,
        COUNT(*) as event_count
    FROM ecommerce_db
    WHERE event_time >= '{cutoff_date.isoformat()}'
    GROUP BY user_id, EXTRACT(HOUR FROM event_time::timestamp)
    """

    df_rfm = pd.read_sql(rfm_query, pg_engine)
    df_time = pd.read_sql(time_query, pg_engine)

    bulk_operations = []
    user_features_collection = db["user_features"]

    for _, user_row in df_rfm.iterrows():
        user_id = user_row['user_id']

        # Get category preferences
        user_categories = df_category[df_category['user_id'] == user_id]
        category_prefs = {}

        if not user_categories.empty:
            for _, cat_row in user_categories.iterrows():
                cat_id = cat_row['category_id']
                score = (cat_row['view_count'] +
                         cat_row['cart_count'] * 3 +
                         cat_row['purchase_count'] * 5)
                category_prefs[cat_id] = score

            total_score = sum(category_prefs.values())
            if total_score > 0:
                category_prefs = {k: round(v / total_score * 100, 2) for k, v in category_prefs.items()}

        # Time pattern detection
        user_time = df_time[df_time['user_id'] == user_id]
        time_pattern = {}

        if not user_time.empty:
            time_buckets = {
                'morning': 0,
                'afternoon': 0,
                'evening': 0,
                'night': 0
            }

            for _, time_row in user_time.iterrows():
                hour = time_row['hour_of_day']
                count = time_row['event_count']

                if 5 <= hour < 12:
                    time_buckets['morning'] += count
                elif 12 <= hour < 17:
                    time_buckets['afternoon'] += count
                elif 17 <= hour < 22:
                    time_buckets['evening'] += count
                else:
                    time_buckets['night'] += count

            total_time_events = sum(time_buckets.values())
            if total_time_events > 0:
                time_pattern = {k: round(v / total_time_events * 100, 2) for k, v in time_buckets.items()}

        # RFM Segmentation
        rfm_score = calculate_rfm_score(
            user_row['days_since_last_purchase'],
            user_row['purchase_frequency'],
            user_row['total_spent']
        )

        # Create user feature document
        user_features = {
            'user_id': user_id,
            'basic_metrics': {
                'total_events': int(user_row['total_events']),
                'view_count': int(user_row['view_count']),
                'cart_count': int(user_row['cart_count']),
                'purchase_count': int(user_row['purchase_count']),
                'total_spent': float(user_row['total_spent']),
                'unique_products_interacted': int(user_row['unique_products_interacted']),
                'unique_products_purchased': int(user_row['unique_products_purchased'])
            },
            'conversion_metrics': {
                'cart_conversion_rate': float(user_row['cart_conversion_rate']),
                'purchase_conversion_rate': float(user_row['purchase_conversion_rate'])
            },
            'rfm_metrics': {
                'recency': int(user_row['days_since_last_purchase']),
                'frequency': int(user_row['purchase_frequency']),
                'monetary': float(user_row['total_spent']),
                'rfm_score': rfm_score,
                'activity_status': user_row['activity_status']
            },
            'category_preferences': category_prefs,
            'time_patterns': time_pattern,
            'last_updated': datetime.now()
        }

        bulk_operations.append(
            pymongo.UpdateOne(
                {'user_id': user_id},
                {'$set': user_features},
                upsert=True
            )
        )

        # Execute in batches of 1000
        if len(bulk_operations) >= 1000:
            user_features_collection.bulk_write(bulk_operations)
            bulk_operations = []

    # Insert any remaining operations
    if bulk_operations:
        user_features_collection.bulk_write(bulk_operations)

    logger.info(f"Created features for {len(df_rfm)} users")
    user_features_collection.create_index("user_id")
    user_features_collection.create_index("rfm_metrics.activity_status")

    plot_user_segmentation(df_rfm)

def calculate_rfm_score(recency: float, frequency: int, monetary: float) -> dict:
    """Calculate RFM scores and segment."""
    # Recency score (lower days = higher score)
    if recency <= 7:
        r_score = 5
    elif recency <= 14:
        r_score = 4
    elif recency <= 30:
        r_score = 3
    elif recency <= 90:
        r_score = 2
    else:
        r_score = 1

    # Frequency score
    if frequency >= 10:
        f_score = 5
    elif frequency >= 5:
        f_score = 4
    elif frequency >= 3:
        f_score = 3
    elif frequency >= 1:
        f_score = 2
    else:
        f_score = 1

    # Monetary score
    if monetary >= 1000:
        m_score = 5
    elif monetary >= 500:
        m_score = 4
    elif monetary >= 250:
        m_score = 3
    elif monetary >= 100:
        m_score = 2
    else:
        m_score = 1

    # Combined RFM score
    combined_score = r_score * 100 + f_score * 10 + m_score

    # Customer segment
    if r_score >= 4 and f_score >= 4 and m_score >= 4:
        segment = "Champions"
    elif r_score >= 3 and f_score >= 3 and m_score >= 3:
        segment = "Loyal Customers"
    elif r_score >= 4 and f_score >= 1 and m_score >= 1:
        segment = "Recent Customers"
    elif r_score >= 3 and f_score >= 1 and m_score >= 1:
        segment = "Promising"
    elif r_score <= 2 and f_score >= 3 and m_score >= 3:
        segment = "At Risk"
    elif r_score <= 2 and f_score >= 2 and m_score >= 2:
        segment = "Needs Attention"
    elif r_score <= 1 and f_score <= 1 and m_score <= 2:
        segment = "Lost"
    else:
        segment = "Others"

    return {
        "r_score": r_score,
        "f_score": f_score,
        "m_score": m_score,
        "combined": combined_score,
        "segment": segment
    }

def create_product_features(pg_engine, mongo_db):
    """Create and store product-related features."""
    logger.info("Creating product features")
    client = mongo_db[0]
    db = mongo_db[1]

    cutoff_date = datetime(2019,10,1)

    product_query = f"""
    WITH product_stats AS (
        SELECT 
            e.product_id,
            COUNT(DISTINCT e.user_id) as unique_users,
            COUNT(CASE WHEN e.event_type = 'view' THEN 1 END) as view_count,
            COUNT(CASE WHEN e.event_type = 'cart' THEN 1 END) as cart_count,
            COUNT(CASE WHEN e.event_type = 'purchase' THEN 1 END) as purchase_count
        FROM ecommerce_db e
        WHERE e.event_time >= '{cutoff_date.isoformat()}'
        GROUP BY e.product_id
    ),
    product_price AS (
        SELECT 
            product_id,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}'
        GROUP BY product_id
    ),
    product_info AS (
        SELECT DISTINCT
            product_id,
            category_id,
            category_code,
            brand,
            price
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}'
    )
    SELECT 
        ps.product_id,
        pi.category_id,
        pi.category_code,
        pi.brand,
        COALESCE(pp.median_price, pi.price) as price,
        COALESCE(ps.view_count, 0) as view_count,
        COALESCE(ps.cart_count, 0) as cart_count,
        COALESCE(ps.purchase_count, 0) as purchase_count,
        COALESCE(ps.unique_users, 0) as unique_users,
        CASE 
            WHEN ps.view_count > 0 THEN ps.cart_count::float / ps.view_count
            ELSE 0
        END as cart_rate,
        CASE 
            WHEN ps.purchase_count > 0 THEN ps.unique_users::float / ps.purchase_count
            ELSE 0
        END as purchase_rate
    FROM product_stats ps
    LEFT JOIN product_info pi ON ps.product_id = pi.product_id
    LEFT JOIN product_price pp ON ps.product_id = pp.product_id
    """

    df_product_features = pd.read_sql(product_query, pg_engine)

    bulk_operations = []
    product_features_collection = db["product_features"]

    for _, product_row in df_product_features.iterrows():
        product_id = product_row['product_id']

        product_features = {
            'product_id': product_id,
            'category_id': product_row['category_id'],
            'category_code': product_row['category_code'],
            'brand': product_row['brand'],
            'price': float(product_row['price']),
            'view_count': int(product_row['view_count']),
            'cart_count': int(product_row['cart_count']),
            'purchase_count': int(product_row['purchase_count']),
            'unique_users': int(product_row['unique_users']),
            'cart_rate': float(product_row['cart_rate']),
            'purchase_rate': float(product_row['purchase_rate']),
            'last_updated': datetime.now()
        }

        bulk_operations.append(
            pymongo.UpdateOne(
                {'product_id': product_id},
                {'$set': product_features},
                upsert=True
            )
        )

        if len(bulk_operations) >= 1000:
            product_features_collection.bulk_write(bulk_operations)
            bulk_operations = []

    if bulk_operations:
        product_features_collection.bulk_write(bulk_operations)

    logger.info(f"Created features for {len(df_product_features)} products")
    product_features_collection.create_index("product_id")

def create_interaction_features(pg_engine, mongo_db, days_lookback: int = 300):
    """Create and store interaction-related features."""
    logger.info("Creating interaction features")
    client = mongo_db[0]
    db = mongo_db[1]

    cutoff_date = datetime(2019,10,1)

    interaction_query = f"""
    SELECT 
        user_id,
        product_id,
        COUNT(CASE WHEN event_type = 'view' THEN 1 END) as view_count,
        COUNT(CASE WHEN event_type = 'cart' THEN 1 END) as cart_count,
        COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) as purchase_count
    FROM ecommerce_db
    WHERE event_time >= '{cutoff_date.isoformat()}'
    GROUP BY user_id, product_id
    """

    df_interaction_features = pd.read_sql(interaction_query, pg_engine)

    bulk_operations = []
    interaction_features_collection = db["interaction_features"]

    for _, interaction_row in df_interaction_features.iterrows():
        user_id = interaction_row['user_id']
        product_id = interaction_row['product_id']

        interaction_features = {
            'user_id': user_id,
            'product_id': product_id,
            'view_count': int(interaction_row['view_count']),
            'cart_count': int(interaction_row['cart_count']),
            'purchase_count': int(interaction_row['purchase_count']),
            'last_updated': datetime.now()
        }

        bulk_operations.append(
            pymongo.UpdateOne(
                {'user_id': user_id, 'product_id': product_id},
                {'$set': interaction_features},
                upsert=True
            )
        )

        if len(bulk_operations) >= 1000:
            interaction_features_collection.bulk_write(bulk_operations)
            bulk_operations = []

    if bulk_operations:
        interaction_features_collection.bulk_write(bulk_operations)

    logger.info(f"Created features for {len(df_interaction_features)} user-product interactions")
    interaction_features_collection.create_index(
        [("user_id", pymongo.ASCENDING), ("product_id", pymongo.ASCENDING)]
    )

    return True