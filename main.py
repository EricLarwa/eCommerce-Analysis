import pandas as pd
from utils import get_path
from database.postgres_manager import get_engine
from database.mongodb_manager import setup_mongodb
from data_pipeline.data_loader import preprocess_data
from data_pipeline.feature_eng import create_all_features
from analysis.exploratory import perform_eda
from utils import make_logger

pg_engine = get_engine()
mongo_db = setup_mongodb()
logger = make_logger()


def main():
    path = get_path()

    perform_eda()

    create_all_features(pg_engine, mongo_db, days_lookback=30)

    preprocess_data(path, batch_size=1000)




if __name__ == "__main__":
    main()