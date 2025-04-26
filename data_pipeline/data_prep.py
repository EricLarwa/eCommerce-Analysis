import pandas as pd
from sqlalchemy import create_engine
import kagglehub
import os
from tqdm import tqdm

pg_engine = create_engine('postgresql://postgres:Majoie@localhost:5432/ecommerce_db')

dataset_path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")

file_names = ['2019-Nov.csv', '2019-Oct.csv']
file_paths = [os.path.join(dataset_path, name) for name in file_names]

def write_csv_in_chunks(file_path, engine):
    try:
        total_lines = sum(1 for _ in open(file_path)) - 1
        chunk_size = 5000
        with tqdm(total=total_lines, desc=f"Uploading {os.path.basename(file_path)}") as pbar:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk.to_sql('ecommerce_data', engine, if_exists='append', index=False)
                pbar.update(len(chunk))
        print(f" Finished: {file_path}")
    except Exception as e:
        print(f" Error processing {file_path}: {e}")

for file_path in file_paths:
    write_csv_in_chunks(file_path, pg_engine)
