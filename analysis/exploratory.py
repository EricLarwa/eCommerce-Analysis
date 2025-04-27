import pandas as pd
from database.postgres_manager import get_engine
from utils import make_logger
import matplotlib.pyplot as plt
import pygal
from pygal.style import Style
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

    custom_style = Style(
        colors=('#4C72B0', '#DD8452', '#55A868', '#C44E52'),
        font_family='googlefont:Raleway',
        background='white',
        plot_background='white',
        foreground='black',
        foreground_strong='black',
        foreground_subtle='#555555',
        opacity='.6',
        opacity_hover='.9',
        transition='400ms ease-in',
        value_font_size=12,
        label_font_size=12,
        major_label_font_size=12,
        title_font_size=16,
        legend_font_size=14,
        tooltip_font_size=12
    )

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
        """),
        ("Top Brands by Purchase Count", """
            SELECT brand, COUNT(*) as purchase_count
            FROM ecommerce_db
            WHERE event_type = 'purchase' AND brand != 'unknown'
            GROUP BY brand
            ORDER BY purchase_count DESC
            LIMIT 15
        """),
        ("User Activity Distribution", """
            SELECT user_id, COUNT(*) as event_count
            FROM ecommerce_db
            GROUP BY user_id
            HAVING COUNT(*) < 500
        """),
        ("Hourly Event Trend", """
            SELECT EXTRACT(HOUR FROM event_time::timestamp) as hour, COUNT(*) as events
            FROM ecommerce_db
            GROUP BY hour
            ORDER BY hour
        """),
        ("Category Conversion Rate", """
            SELECT category_code,
                SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) as views,
                SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchases
            FROM ecommerce_db
            WHERE category_code != 'unknown'
            GROUP BY category_code
            HAVING SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) > 1000
        """),
        ("Cart Abandonment", """
            SELECT
                COUNT(DISTINCT CASE WHEN event_type = 'cart' THEN user_id END) as carts,
                COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN user_id END) as purchases
            FROM ecommerce_db
        """)
    ]

    for analysis_name, query in tqdm(analyses, desc="Performing EDA", unit="analysis"):
        logger.info(f"Starting analysis: {analysis_name}")

        try:

            if analysis_name == "Event Type Distribution":
                event_counts = pd.read_sql(query, pg_engine)
                chart = pygal.Bar(
                    style=custom_style,
                    title='Distribution of Event Types',
                    x_title='Event Type',
                    y_title='Count',
                    show_legend=False,
                    print_values=True,
                    print_values_position='top'
                )
                for _, row in event_counts.iterrows():
                    chart.add(row['event_type'], row['count'])
                chart.render_to_file(f"{output_dir}/event_distribution.svg")

            elif analysis_name == "Top Product Categories":

                top_categories = pd.read_sql(query, pg_engine)
                chart = pygal.HorizontalBar(
                    style=custom_style,
                    title='Top 20 Product Categories by Transaction Count',
                    x_title='Transaction Count',
                    show_legend=True,
                    print_values=True,
                    print_values_position='right',
                    margin_bottom=50
                )

                for _, row in top_categories.sort_values('transaction_count').iterrows():
                    chart.add(row['category_code'], row['transaction_count'])

                chart.render_to_file(f"{output_dir}/top_categories.svg")

            elif analysis_name == "Purchase Funnel":

                funnel_data = pd.read_sql(query, pg_engine)
                funnel_long = pd.DataFrame({
                    'Stage': ['View', 'Cart', 'Purchase'],
                    'Users': [funnel_data['views'][0], funnel_data['carts'][0], funnel_data['purchases'][0]]
                })

                chart = pygal.Funnel(
                    style=custom_style,
                    title='Purchase Funnel',
                    print_values=True,
                    human_readable=True
                )

                for _, row in funnel_long.iterrows():
                    chart.add(row['Stage'], row['Users'])

                chart.render_to_file(f"{output_dir}/purchase_funnel.svg")

            elif analysis_name == "Time Based Analysis":

                time_data = pd.read_sql(query, pg_engine)
                time_data['day'] = pd.to_datetime(time_data['day'])
                chart = pygal.Line(
                    style=custom_style,
                    title='Event Volume by Day',
                    x_title='Date',
                    y_title='Number of Events',
                    x_label_rotation=45,
                    show_dots=True,
                    dots_size=3,
                    stroke_style={'width': 3}
                )

                all_dates = [d.strftime('%Y-%m-%d') for d in time_data['day']]

                chart.x_labels = all_dates
                chart.x_labels_major = all_dates[::5]
                chart.show_minor_x_labels = False

                chart.add('Events', time_data['events'].tolist())
                chart.render_to_file(f"{output_dir}/time_analysis.svg")

            elif analysis_name == "Price Distribution":

                price_data = pd.read_sql(query, pg_engine)
                chart = pygal.Histogram(
                    style=custom_style,
                    title='Product Price Distribution',
                    x_title='Price',
                    y_title='Count',
                    show_legend=False,
                    bins=20

                )

                chart.add('Prices', [(v, 1) for v in price_data['price']])

                chart.render_to_file(f"{output_dir}/price_distribution.svg")

            elif analysis_name == "Top Brands by Purchase Count":
                brand_data = pd.read_sql(query, pg_engine)
                chart = pygal.HorizontalBar(
                    style=custom_style,
                    title='Top 15 Brands by Purchase Count',
                    x_title='Purchase Count',
                    print_values=True,
                    print_values_position='right',
                    margin_bottom=50
                )
                for _, row in brand_data.sort_values('purchase_count').iterrows():
                    chart.add(row['brand'], row['purchase_count'])
                chart.render_to_file(f"{output_dir}/top_brands.svg")

            elif analysis_name == "User Activity Distribution":
                activity_data = pd.read_sql(query, pg_engine)
                chart = pygal.Histogram(
                    style=custom_style,
                    title='Distribution of Events per User (filtered at 500)',
                    x_title='Number of Events',
                    y_title='User Count',
                    bins=20,
                    show_legend=False
                )
                # Pygal histogram expects a list of (value, frequency) tuples
                # We'll simulate this by counting occurrences ourselves
                hist_data = activity_data['event_count'].value_counts().reset_index()
                hist_data.columns = ['value', 'count']
                chart.add('Users', [(row['value'], row['count']) for _, row in hist_data.iterrows()])
                chart.render_to_file(f"{output_dir}/user_activity.svg")

            elif analysis_name == "Hourly Event Trend":
                hour_data = pd.read_sql(query, pg_engine)
                chart = pygal.Line(
                    style=custom_style,
                    title='Hourly Distribution of Events',
                    x_title='Hour of Day',
                    y_title='Event Count',
                    show_dots=True,
                    dots_size=3,
                    stroke_style={'width': 3},
                )
                chart.x_labels = [str(hour) for hour in range(24)]

                chart.add('Events', hour_data['events'].tolist())
                chart.render_to_file(f"{output_dir}/hourly_trend.svg")

            elif analysis_name == "Category Conversion Rate":
                conv_data = pd.read_sql(query, pg_engine)
                conv_data['conversion_rate'] = conv_data['purchases'] / conv_data['views']
                conv_data = conv_data.sort_values('conversion_rate', ascending=False).head(15)

                chart = pygal.HorizontalBar(
                    style=custom_style,
                    title='Top 15 Categories by Conversion Rate',
                    x_title='Conversion Rate',
                    print_values=True,
                    print_values_position='right',
                    print_values_format='{:.1%}',
                    margin_bottom=50
                )
                for _, row in conv_data.iterrows():
                    chart.add(row['category_code'], row['conversion_rate'])
                chart.render_to_file(f"{output_dir}/category_conversion_rate.svg")

            elif analysis_name == "Cart Abandonment":
                abandon_data = pd.read_sql(query, pg_engine)
                carted = abandon_data['carts'][0]
                purchased = abandon_data['purchases'][0]
                abandoned = carted - purchased

                chart = pygal.Pie(
                    style=custom_style,
                    title='Cart Abandonment Analysis',
                    inner_radius=0.4,
                    print_values=True,
                    human_readable=True
                )
                chart.add('Purchased', purchased)
                chart.add('Abandoned', abandoned)
                chart.render_to_file(f"{output_dir}/cart_abandonment.svg")

            logger.info(f"Completed analysis: {analysis_name}")

        except Exception as e:
            logger.error(f"Error in {analysis_name}: {str(e)}")
            continue

    logger.info(f"EDA complete - visualizations saved to {output_dir}")
    return True