import pandas as pd
from sqlalchemy import create_engine
import os


pg_engine = create_engine('postgresql://postgres:password@localhost:5432/ecommerce_db')

# Path to the directory containing the CSV files
directory_path = '/Users/ericlarwa/.cache/kagglehub/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store/versions/8'

# List of CSV file names
file_names = ['2019-Nov.csv', '2019-Oct.csv']

# Create full file paths
file_paths = [os.path.join(directory_path, file_name) for file_name in file_names]

# Create a list to hold DataFrames
dataframes = []

# Read each CSV file and append the DataFrame to the list
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_data = pd.concat(dataframes, ignore_index=True)

# Display the first few rows of the combined DataFrame
print(combined_data.head())

combined_data.to_sql('ecommerce_db', pg_engine, if_exists='append', index=False)

# Verify the data insertion
with pg_engine.connect() as connection:
    result = connection.execute("SELECT * FROM ecommerce_db LIMIT 5;")
    for row in result:
        print(row)
