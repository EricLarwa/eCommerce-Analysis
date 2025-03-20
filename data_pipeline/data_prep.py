import pandas as pd
from sqlalchemy import create_engine
import os

pg_engine = create_engine('postgresql://postgres:password@localhost:5432/ecommerce_db')

# change to your specific path
directory_path = '/Users/ericlarwa/.cache/kagglehub/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store/versions/8'


file_names = ['2019-Nov.csv', '2019-Oct.csv']

file_paths = [os.path.join(directory_path, file_name) for file_name in file_names]

dataframes = []

def append_and_write_to_db(df, engine):
    dataframes.append(df)  # Append to the list
    df.to_sql('ecommerce_db', engine, if_exists='append', index=False, chunksize=5000)

for file_path in file_paths:
    try:
        for chunk in pd.read_csv(file_path, chunksize=5000):
            append_and_write_to_db(chunk, pg_engine)
        print(f"Processed {file_path} successfully.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

if dataframes:
    combined_data = pd.concat(dataframes, ignore_index=True)
    try:
        combined_data.to_sql('ecommerce_db', pg_engine, if_exists='append', index=False, chunksize=5000)
    except Exception as e:
        print(f"An error occurred while writing combined data: {e}")

print(combined_data.head())

with pg_engine.connect() as connection:
    result = connection.execute("SELECT * FROM ecommerce_db LIMIT 5;")
    for row in result:
        print(row)