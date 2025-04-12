import pandas as pd
from database.postgres_manager import get_engine
from utils import make_logger
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

logger = make_logger()


def perform_eda(output_dir="./eda_output"):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Starting Exploratory Data Analysis")

    pg_engine = get_engine()

    # List of analyses
    analyses = [
        ("Event Type Distribution", """
            SELECT event_type, COUNT(*) as count
            FROM ecommerce_db
            GROUP BY event_type
            ORDER BY count DESC
        """),
        ("Top Product Categories", """
            SELECT category_code, COUNT(*) as transaction_count
            FROM ecommerce_db
            WHERE category_code != 'unknown'
            GROUP BY category_code
            ORDER BY transaction_count DESC
            LIMIT 20
        """),
        ("Purchase Funnel", """
            SELECT
                COUNT(DISTINCT CASE WHEN event_type = 'view' THEN user_id END) as views,
                COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN user_id END) as carts,
                COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) as purchases
            FROM ecommerce_db
        """),
        ("Time Based Analysis", """
            SELECT
                DATE_TRUNC('day', event_time::timestamp) as day,
                COUNT(*) as events
            FROM ecommerce_db
            GROUP BY DATE_TRUNC('day', event_time::timestamp)
            ORDER BY day
        """),
        ("Price Distribution", """
            SELECT price
            FROM ecommerce_db
            WHERE price > 0 AND price < 1000 LIMIT 1000000  
        """)
    ]

    for analysis_name, query in tqdm(analyses, desc="Performing EDA", unit="analysis"):
        logger.info(f"Starting analysis: {analysis_name}")

        if analysis_name == "Event Type Distribution":
            event_counts = pd.read_sql(query, pg_engine)
            plt.figure(figsize=(10, 6))
            sns.barplot(x='event_type', y='count', data=event_counts)
            plt.title('Distribution of Event Types')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/event_distribution.png")

        elif analysis_name == "Top Product Categories":
            top_categories = pd.read_sql(query, pg_engine)
            plt.figure(figsize=(12, 8))
            sns.barplot(y='category_code', x='transaction_count', data=top_categories)
            plt.title('Top 20 Product Categories by Transaction Count')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/top_categories.png")

        elif analysis_name == "Purchase Funnel":
            funnel_data = pd.read_sql(query, pg_engine)
            funnel_long = pd.DataFrame({
                'Stage': ['View', 'Cart', 'Purchase'],
                'Users': [funnel_data['views'][0], funnel_data['carts'][0], funnel_data['purchases'][0]]
            })
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Stage', y='Users', data=funnel_long)
            plt.title('Purchase Funnel')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/purchase_funnel.png")

        elif analysis_name == "Time Based Analysis":
            time_data = pd.read_sql(query, pg_engine)
            plt.figure(figsize=(14, 6))
            plt.plot(time_data['day'], time_data['events'])
            plt.title('Event Volume by Day')
            plt.xlabel('Date')
            plt.ylabel('Number of Events')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/time_analysis.png")

        elif analysis_name == "Price Distribution":
            price_data = pd.read_sql(query, pg_engine)
            plt.figure(figsize=(12, 6))
            sns.histplot(price_data['price'], bins=50)
            plt.title('Product Price Distribution')
            plt.xlabel('Price')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/price_distribution.png")

        logger.info(f"Completed analysis: {analysis_name}")

    logger.info(f"EDA complete - visualizations saved to {output_dir}")
    return True