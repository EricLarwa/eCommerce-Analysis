import pandas as pd
import pymongo
from database.postgres_manager import get_engine
from database.mongodb_manager import setup_mongodb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import make_logger

pg_engine = get_engine()
mongo_db = setup_mongodb()

logger = make_logger()


def plot_user_segmentation(df_rfm, output_dir='./Feature_output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    segmentation_counts = df_rfm['activity_status'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(segmentation_counts, labels=segmentation_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('User Segmentation by Activity Status')
    plt.axis('equal')
    plt.savefig(f"{output_dir}/user_segmentation.png")
    plt.close()

    # Additional user visualizations
    plt.figure(figsize=(10, 6))
    sns.histplot(df_rfm['total_spent'], bins=20, kde=True)
    plt.title('Distribution of Total Spending by Users')
    plt.xlabel('Total Spent ($)')
    plt.ylabel('Number of Users')
    plt.savefig(f"{output_dir}/user_spending_distribution.png")
    plt.close()

    # RFM segments visualization
    if 'rfm_segment' in df_rfm.columns:
        plt.figure(figsize=(10, 6))
        segment_counts = df_rfm['rfm_segment'].value_counts()
        sns.barplot(x=segment_counts.index, y=segment_counts.values)
        plt.title('User Segments Based on RFM Analysis')
        plt.xlabel('Segment')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/user_rfm_segments.png")
        plt.close()


def plot_product_analytics(df_product, output_dir='./Feature_output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Top 10 products by view
    top_viewed = df_product.sort_values('view_count', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='view_count', y='product_id', data=top_viewed)
    plt.title('Top 10 Most Viewed Products')
    plt.xlabel('View Count')
    plt.ylabel('Product ID')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_viewed_products.png")
    plt.close()

    # Product conversion rates
    plt.figure(figsize=(12, 6))
    df_conversion = df_product[df_product['view_count'] > 100].sort_values('cart_rate', ascending=False).head(10)
    sns.barplot(x='cart_rate', y='product_id', data=df_conversion)
    plt.title('Top 10 Products by Cart Conversion Rate (min 100 views)')
    plt.xlabel('Cart Conversion Rate')
    plt.ylabel('Product ID')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/product_conversion_rates.png")
    plt.close()

    # Price distribution by category
    if 'category_code' in df_product.columns and df_product['category_code'].notna().any():
        plt.figure(figsize=(14, 8))
        top_categories = df_product['category_code'].value_counts().head(8).index
        df_filtered = df_product[df_product['category_code'].isin(top_categories)]
        sns.boxplot(x='category_code', y='price', data=df_filtered)
        plt.title('Price Distribution by Top Categories')
        plt.xlabel('Category')
        plt.ylabel('Price ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/price_by_category.png")
        plt.close()


def plot_interaction_patterns(df_interaction, output_dir='./Feature_output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Distribution of interaction counts
    plt.figure(figsize=(12, 6))
    interaction_data = {
        'View': df_interaction['view_count'].sum(),
        'Cart': df_interaction['cart_count'].sum(),
        'Purchase': df_interaction['purchase_count'].sum()
    }
    sns.barplot(x=list(interaction_data.keys()), y=list(interaction_data.values()))
    plt.title('Distribution of Event Types')
    plt.xlabel('Event Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/event_type_distribution.png")
    plt.close()

    # User interaction frequency
    user_interaction_counts = df_interaction.groupby('user_id').size().reset_index(name='interaction_count')
    plt.figure(figsize=(12, 6))
    sns.histplot(user_interaction_counts['interaction_count'], bins=30, kde=True)
    plt.title('Distribution of User Interaction Frequency')
    plt.xlabel('Number of Product Interactions per User')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/user_interaction_frequency.png")
    plt.close()

    # Product interaction frequency
    product_interaction_counts = df_interaction.groupby('product_id').size().reset_index(name='interaction_count')
    plt.figure(figsize=(12, 6))
    sns.histplot(product_interaction_counts['interaction_count'], bins=30, kde=True)
    plt.title('Distribution of Product Interaction Frequency')
    plt.xlabel('Number of User Interactions per Product')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/product_interaction_frequency.png")
    plt.close()


def create_all_features(pg_engine, mongo_db, output_dir='./Feature_output'):
    logger.info(f"Creating all features in {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Starting user features generation")
    user_features = create_user_features(pg_engine, mongo_db)

    # Extract RFM data for visualization
    client = mongo_db[0]
    db = mongo_db[1]
    user_features_collection = db["user_features"]

    user_sample = list(user_features_collection.find(
        {},
        {"user_id": 1, "rfm_metrics.activity_status": 1, "rfm_metrics.rfm_score.segment": 1,
         "basic_metrics.total_spent": 1}
    ).limit(10000))

    if user_sample:
        df_user_sample = pd.DataFrame([
            {
                "user_id": user["user_id"],
                "activity_status": user["rfm_metrics"]["activity_status"],
                "total_spent": user["basic_metrics"]["total_spent"],
                "rfm_segment": user["rfm_metrics"]["rfm_score"]["segment"] if "rfm_score" in user[
                    "rfm_metrics"] else "Unknown"
            }
            for user in user_sample
        ])

        plot_user_segmentation(df_user_sample, output_dir)

    logger.info("Starting product features generation")
    create_product_features(pg_engine, mongo_db)

    logger.info("Starting interaction features generation")
    create_interaction_features(pg_engine, mongo_db)

    logger.info("All features created successfully")
    logger.info(f"Feature visualizations saved to {output_dir}")


def create_user_features(pg_engine, mongo_db):
    """Create and store user-related features with optimized queries."""
    logger.info("Creating user features")

    client = mongo_db[0]
    db = mongo_db[1]
    cutoff_date = datetime(2019, 10, 1)

    batch_size = 10000
    offset = 0

    max_batches = 10
    batch_count = 0

    while batch_count < max_batches:
        batch_count += 1
        logger.info(f"Processing batch #{batch_count} with offset {offset}")

        # Step 1: Get a batch of users to process
        user_batch_query = f"""
        SELECT DISTINCT user_id 
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}'
        ORDER BY user_id
        LIMIT {batch_size} OFFSET {offset}
        """

        user_batch = pd.read_sql(user_batch_query, pg_engine)

        if user_batch.empty:
            logger.info("No more users to process. Exiting.")
            break

        user_ids = tuple(user_batch['user_id'].tolist())
        if len(user_ids) == 1:
            user_filter = f"user_id = {user_ids[0]}"
        else:
            user_filter = f"user_id IN {user_ids}"

        # Step 2: Get purchase data for this batch
        purchase_query = f"""
        SELECT 
            user_id,
            MAX(event_time) as last_purchase_date,
            COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN CAST(event_time AS date) ELSE NULL END) as frequency,
            SUM(CASE WHEN event_type = 'purchase' THEN price ELSE 0 END) as monetary
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}' AND {user_filter}
        GROUP BY user_id
        """

        # Step 3: Get user stats for this batch
        stats_query = f"""
        SELECT 
            user_id,
            COUNT(*) as total_events,
            COUNT(CASE WHEN event_type = 'view' THEN 1 END) as view_count,
            COUNT(CASE WHEN event_type = 'cart' THEN 1 END) as cart_count,
            COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) as purchase_count,
            COUNT(DISTINCT product_id) as unique_products_interacted,
            COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN product_id ELSE NULL END) as unique_products_purchased
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}' AND {user_filter}
        GROUP BY user_id
        """

        df_purchase = pd.read_sql(purchase_query, pg_engine)
        df_stats = pd.read_sql(stats_query, pg_engine)

        # Step 4: Merge the results
        df_rfm = pd.merge(df_stats, df_purchase, on='user_id', how='left')

        # Add calculated fields
        df_rfm['total_spent'] = df_rfm['monetary'].fillna(0)
        df_rfm['cart_conversion_rate'] = df_rfm.apply(
            lambda x: x['cart_count'] / x['view_count'] if x['view_count'] > 0 else 0, axis=1)
        df_rfm['purchase_conversion_rate'] = df_rfm.apply(
            lambda x: x['purchase_count'] / x['cart_count'] if x['cart_count'] > 0 else 0, axis=1)

        reference_date = datetime(2019, 11, 30)

        df_rfm['days_since_last_purchase'] = df_rfm.apply(
            lambda x: (reference_date - pd.to_datetime(x['last_purchase_date']).replace(tzinfo=None)).days
            if pd.notna(x['last_purchase_date']) else 999, axis=1)

        df_rfm['activity_status'] = df_rfm.apply(
            lambda x: 'inactive' if pd.isna(x['last_purchase_date']) else
            ('active' if x['days_since_last_purchase'] <= 30 else 'lapsed'),
            axis=1)

        # Step 5: Get category preferences
        category_query = f"""
        SELECT 
            e.user_id,
            e.category_id,
            COUNT(CASE WHEN e.event_type = 'view' THEN 1 END) as view_count,
            COUNT(CASE WHEN e.event_type = 'cart' THEN 1 END) as cart_count,
            COUNT(CASE WHEN e.event_type = 'purchase' THEN 1 END) as purchase_count
        FROM ecommerce_db e
        WHERE e.event_time >= '{cutoff_date.isoformat()}' AND {user_filter}
        GROUP BY e.user_id, e.category_id
        """

        df_category = pd.read_sql(category_query, pg_engine)

        # Step 6: Get time patterns
        time_query = f"""
        SELECT 
            user_id,
            EXTRACT(HOUR FROM event_time::timestamp) as hour_of_day,
            COUNT(*) as event_count
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}' AND {user_filter}
        GROUP BY user_id, EXTRACT(HOUR FROM event_time::timestamp)
        """

        df_time = pd.read_sql(time_query, pg_engine)

        bulk_operations = []
        user_features_collection = db["user_features"]

        for _, user_row in df_rfm.iterrows():
            user_id = user_row['user_id']

            # Initialize category_prefs
            category_prefs = {}

            # Get category preferences
            user_categories = df_category[df_category['user_id'] == user_id]
            if not user_categories.empty:
                for _, cat_row in user_categories.iterrows():
                    cat_id = cat_row['category_id']
                    score = (cat_row['view_count'] +
                             cat_row['cart_count'] * 3 +
                             cat_row['purchase_count'] * 5)
                    category_prefs[str(cat_id)] = score

                total_score = sum(category_prefs.values())
                if total_score > 0:
                    category_prefs = {k: round(v / total_score * 100, 2) for k, v in category_prefs.items()}

            # Initialize time_pattern
            time_pattern = {}

            # Time pattern detection
            user_time = df_time[df_time['user_id'] == user_id]
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
                user_row['frequency'] if 'frequency' in user_row else 0,
                user_row['total_spent']
            )

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
                    'frequency': int(user_row['frequency']) if 'frequency' in user_row else 0,
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

            if len(bulk_operations) >= 1000:
                user_features_collection.bulk_write(bulk_operations)
                bulk_operations = []

        if bulk_operations:
            user_features_collection.bulk_write(bulk_operations)

        offset += batch_size
        logger.info(f"Processed batch {batch_count} with {len(df_rfm)} users")

    if batch_count >= max_batches:
        logger.warning(f"Reached maximum batch count of {max_batches}. Consider adjusting parameters.")

    logger.info("Finished creating user features")
    user_features_collection.create_index("user_id")
    user_features_collection.create_index("rfm_metrics.activity_status")

def calculate_rfm_score(recency: float, frequency: int, monetary: float) -> dict:
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
    logger.info("Creating product features")
    client = mongo_db[0]
    db = mongo_db[1]

    cutoff_date = datetime(2019, 10, 1)

    batch_size = 5000
    offset = 0
    max_batches = 3
    batch_count = 0

    product_features_collection = db["product_features"]
    all_product_features = []

    while batch_count < max_batches:
        batch_count += 1
        logger.info(f"Processing product batch #{batch_count} with offset {offset}")

        batch_query = f"""
        SELECT DISTINCT product_id 
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}'
        ORDER BY product_id
        LIMIT {batch_size} OFFSET {offset}
        """

        product_batch = pd.read_sql(batch_query, pg_engine)

        if product_batch.empty:
            logger.info("No more products to process. Exiting.")
            break

        product_ids = tuple(product_batch['product_id'].tolist())
        if len(product_ids) == 1:
            product_filter = f"product_id = {product_ids[0]}"
        else:
            product_filter = f"product_id IN {product_ids}"

        # Get product stats for this batch
        product_query = f"""
        WITH product_stats AS (
            SELECT 
                e.product_id,
                COUNT(DISTINCT e.user_id) as unique_users,
                COUNT(CASE WHEN e.event_type = 'view' THEN 1 END) as view_count,
                COUNT(CASE WHEN e.event_type = 'cart' THEN 1 END) as cart_count,
                COUNT(CASE WHEN e.event_type = 'purchase' THEN 1 END) as purchase_count
            FROM ecommerce_db e
            WHERE e.event_time >= '{cutoff_date.isoformat()}' AND {product_filter}
            GROUP BY e.product_id
        ),
        product_price AS (
            SELECT 
                product_id,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price
            FROM ecommerce_db
            WHERE event_time >= '{cutoff_date.isoformat()}' AND {product_filter}
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
            WHERE event_time >= '{cutoff_date.isoformat()}' AND {product_filter}
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
                WHEN ps.cart_count > 0 THEN ps.purchase_count::float / ps.cart_count
                ELSE 0
            END as purchase_rate
        FROM product_stats ps
        LEFT JOIN product_info pi ON ps.product_id = pi.product_id
        LEFT JOIN product_price pp ON ps.product_id = pp.product_id
        """

        df_product_features = pd.read_sql(product_query, pg_engine)
        all_product_features.append(df_product_features)

        bulk_operations = []

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

        offset += batch_size
        logger.info(f"Processed batch {batch_count} with {len(df_product_features)} products")

    if batch_count >= max_batches:
        logger.warning(f"Reached maximum batch count of {max_batches}. Consider adjusting parameters.")

    logger.info("Finished creating product features")
    product_features_collection.create_index("product_id")
    product_features_collection.create_index("category_code")

    if all_product_features:
        combined_df = pd.concat(all_product_features)
        plot_product_analytics(combined_df)

    return True


def create_interaction_features(pg_engine, mongo_db):
    logger.info("Creating interaction features")
    client = mongo_db[0]
    db = mongo_db[1]

    cutoff_date = datetime(2019, 10, 1)

    batch_size = 5000
    offset = 0
    max_batches = 5
    batch_count = 0

    interaction_features_collection = db["interaction_features"]
    all_interaction_features = []

    while batch_count < max_batches:
        batch_count += 1
        logger.info(f"Processing interaction batch #{batch_count} with offset {offset}")

        # Get a batch of users
        user_batch_query = f"""
        SELECT DISTINCT user_id 
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}'
        ORDER BY user_id
        LIMIT {batch_size} OFFSET {offset}
        """

        user_batch = pd.read_sql(user_batch_query, pg_engine)

        if user_batch.empty:
            logger.info("No more users to process. Exiting.")
            break

        user_ids = tuple(user_batch['user_id'].tolist())
        if len(user_ids) == 1:
            user_filter = f"user_id = {user_ids[0]}"
        else:
            user_filter = f"user_id IN {user_ids}"

        # Get interactions for these users
        interaction_query = f"""
        SELECT 
            user_id,
            product_id,
            COUNT(CASE WHEN event_type = 'view' THEN 1 END) as view_count,
            COUNT(CASE WHEN event_type = 'cart' THEN 1 END) as cart_count,
            COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) as purchase_count,
            MAX(event_time) as last_interaction
        FROM ecommerce_db
        WHERE event_time >= '{cutoff_date.isoformat()}' AND {user_filter}
        GROUP BY user_id, product_id
        """

        df_interaction_features = pd.read_sql(interaction_query, pg_engine)
        all_interaction_features.append(df_interaction_features)

        bulk_operations = []

        for _, interaction_row in df_interaction_features.iterrows():
            user_id = interaction_row['user_id']
            product_id = interaction_row['product_id']

            # Views = 1 point, Cart adds = 3 points, Purchases = 5 points
            interaction_score = (
                    interaction_row['view_count'] * 1 +
                    interaction_row['cart_count'] * 3 +
                    interaction_row['purchase_count'] * 5
            )

            interaction_features = {
                'user_id': user_id,
                'product_id': product_id,
                'view_count': int(interaction_row['view_count']),
                'cart_count': int(interaction_row['cart_count']),
                'purchase_count': int(interaction_row['purchase_count']),
                'interaction_score': int(interaction_score),
                'last_interaction': interaction_row['last_interaction'],
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

        offset += batch_size
        logger.info(f"Processed batch {batch_count} with {len(df_interaction_features)} user-product interactions")

    if batch_count >= max_batches:
        logger.warning(f"Reached maximum batch count of {max_batches}. Consider adjusting parameters.")

    logger.info("Finished creating interaction features")
    interaction_features_collection.create_index([("user_id", pymongo.ASCENDING), ("product_id", pymongo.ASCENDING)])
    interaction_features_collection.create_index("interaction_score")

    if all_interaction_features:
        combined_df = pd.concat(all_interaction_features)
        plot_interaction_patterns(combined_df)

    return True