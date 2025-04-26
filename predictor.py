# predictor.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from database.postgres_manager import get_engine
from database.mongodb_manager import setup_mongodb
from utils import make_logger
import os

logger = make_logger()

def run_predictions(output_dir="./predictions"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Connecting to databases...")
    pg_engine = get_engine()
    mongo_client, mongo_db = setup_mongodb()

    logger.info("Loading user features from MongoDB...")
    user_docs = list(mongo_db["user_features"].find({}))
    user_df = pd.json_normalize(user_docs)

    if user_df.empty:
        logger.error("No user features found.")
        return

    logger.info("Loading purchase transactions from PostgreSQL...")
    purchases = pd.read_sql("""
        SELECT user_id, event_time
        FROM transactions
        WHERE event_type = 'purchase'
    """, pg_engine)

    purchases['event_time'] = pd.to_datetime(purchases['event_time'])

    logger.info("Generating labels from timestamp comparisons...")

    user_df['last_event_time'] = pd.to_datetime("now") - pd.to_timedelta(user_df['rfm_metrics.recency'], unit='D')

    labels = []
    for i, row in user_df.iterrows():
        user_id = row['user_id']
        last_time = row['last_event_time']
        future_window = last_time + pd.Timedelta(days=7)

        user_purchases = purchases[(purchases['user_id'] == user_id)]
        has_future_purchase = any((user_purchases['event_time'] > last_time) & (user_purchases['event_time'] <= future_window))

        labels.append(1 if has_future_purchase else 0)

    user_df['label'] = labels

    logger.info("Preparing features and training model...")

    features = [
        'basic_metrics.total_events',
        'basic_metrics.view_count',
        'basic_metrics.cart_count',
        'basic_metrics.purchase_count',
        'basic_metrics.total_spent',
        'basic_metrics.unique_products_interacted',
        'conversion_metrics.cart_conversion_rate',
        'conversion_metrics.purchase_conversion_rate',
        'rfm_metrics.recency',
        'rfm_metrics.frequency',
        'rfm_metrics.monetary'
    ]

    X = user_df[features]
    y = user_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    logger.info("Saving confusion matrix...")
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title("7-Day Purchase Prediction")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # === Predict & rank users by probability ===
    logger.info("Ranking users by purchase probability...")
    full_probs = model.predict_proba(X)[:, 1]
    user_df['purchase_probability'] = full_probs

    top_users = user_df[['user_id', 'purchase_probability']].sort_values(by='purchase_probability', ascending=False).head(10)
    top_users.to_csv(f"{output_dir}/top_predicted_users.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='purchase_probability', y='user_id', data=top_users, palette='crest')
    plt.title("Top 10 Users by Predicted Purchase Probability")
    plt.xlabel("Probability")
    plt.ylabel("User ID")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_predicted_users.png")
    plt.close()

    logger.info("✅ Prediction complete. Charts and CSV saved to predictions/")
    mongo_client.close()
