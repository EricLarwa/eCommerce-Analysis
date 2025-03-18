import kagglehub
import logging

def get_path():
    path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")
    return path

def make_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler("ecommerce_analysis.log"), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    return logger

logger = make_logger()