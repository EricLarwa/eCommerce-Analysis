import pandas as pd
from database.postgres_manager import get_engine
from utils import make_logger
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logger = make_logger()
def perform_eda(output_dir="./eda_output"):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Starting Exploratory Data Analysis")

    pg_engine = get_engine()

    # First Analysis / Event Type Distribution
    event_counts = pd.read_sql("""
        SELECT event_type, COUNT(*) as count
        FROM transactions
        GROUP BY event_type
        ORDER BY count DESC
    """, pg_engine)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='event_type', y='count', data=event_counts)
    plt.title('Distribution of Event Types')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/event_distribution.png")

    # Second Analysis / Top Product Categories
    top_categories = pd.read_sql("""
        SELECT p.category_code, COUNT(*) as transaction_count
        FROM transactions t
        JOIN products p ON t.product_id = p.product_id
        WHERE p.category_code != 'unknown'
        GROUP BY p.category_code
        ORDER BY transaction_count DESC
        LIMIT 20
    """, pg_engine)

    plt.figure(figsize=(12, 8))
    sns.barplot(y='category_code', x='transaction_count', data=top_categories)
    plt.title('Top 20 Product Categories by Transaction Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_categories.png")

    # Third Analysis / Purchase Funnel
    funnel_data = pd.read_sql("""
            SELECT 
                COUNT(DISTINCT CASE WHEN event_type = 'view' THEN user_id END) as views,
                COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN user_id END) as carts,
                COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) as purchases
            FROM transactions
        """, pg_engine)

    funnel_long = pd.DataFrame({
        'Stage': ['View', 'Cart', 'Purchase'],
        'Users': [funnel_data['views'][0], funnel_data['carts'][0], funnel_data['purchases'][0]]
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Stage', y='Users', data=funnel_long)
    plt.title('Purchase Funnel')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/purchase_funnel.png")

    # Fourth Analysis / Time Based
    time_data = pd.read_sql("""
            SELECT 
                DATE_TRUNC('day', event_time) as day,
                COUNT(*) as events
            FROM transactions
            GROUP BY DATE_TRUNC('day', event_time)
            ORDER BY day
        """, pg_engine)

    plt.figure(figsize=(14, 6))
    plt.plot(time_data['day'], time_data['events'])
    plt.title('Event Volume by Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Events')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_analysis.png")

    # Fifth Analysis / Price Distribution
    price_data = pd.read_sql("""
            SELECT price
            FROM products
            WHERE price > 0 AND price < 1000  -- Filter out extreme values
        """, pg_engine)

    plt.figure(figsize=(12, 6))
    sns.histplot(price_data['price'], bins=50)
    plt.title('Product Price Distribution')
    plt.xlabel('Price')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/price_distribution.png")

    logger.info(f"EDA complete - visualizations saved to {output_dir}")
    return True
