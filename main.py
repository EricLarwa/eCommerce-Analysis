import pandas as pd
from utils import get_path
from database.postgres_manager import get_engine
from database.mongodb_manager import setup_mongodb
from data_pipeline.feature_eng import create_all_features
from predictor import run_predictions
from analysis.exploratory import perform_eda
from utils import make_logger

pg_engine = get_engine()
mongo_db = setup_mongodb()
logger = make_logger()


def main():
    path = get_path()

    # perform_eda()

    create_all_features(pg_engine, mongo_db)


    run_predictions()




if __name__ == "__main__":
    main()