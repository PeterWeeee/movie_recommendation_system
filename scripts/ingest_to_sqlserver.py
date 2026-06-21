import os
import pandas as pd
from sqlalchemy import create_engine, text
import urllib
import pyodbc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

DB_NAME = 'MovieRecommendationDB'
SERVER = 'localhost'

def get_driver():
    drivers = [driver for driver in pyodbc.drivers()]
    if 'ODBC Driver 17 for SQL Server' in drivers:
        return 'ODBC Driver 17 for SQL Server'
    elif 'ODBC Driver 18 for SQL Server' in drivers:
        return 'ODBC Driver 18 for SQL Server'
    elif 'SQL Server Native Client 11.0' in drivers:
        return 'SQL Server Native Client 11.0'
    else:
        return 'SQL Server' # Fallback default

def get_engine(db_name='master'):
    driver = get_driver()
    params = urllib.parse.quote_plus(
        f'DRIVER={{{driver}}};'
        f'SERVER={SERVER};'
        f'DATABASE={db_name};'
        f'Trusted_Connection=yes;'
        f'TrustServerCertificate=yes;'
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)

def create_database():
    engine = get_engine('master')
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        res = conn.execute(text(f"SELECT name FROM sys.databases WHERE name = '{DB_NAME}'"))
        if not res.fetchone():
            print(f"Creating database {DB_NAME}...")
            conn.execute(text(f"CREATE DATABASE {DB_NAME}"))
        else:
            print(f"Database {DB_NAME} already exists.")

def ingest_data():
    engine = get_engine(DB_NAME)
    
    # 1. Users
    print("Ingesting users...")
    user_file = os.path.join(RAW_DATA_DIR, 'u.user')
    user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    df_users = pd.read_csv(user_file, sep='|', names=user_cols)
    df_users.to_sql('users', engine, if_exists='replace', index=False)
    
    # 2. Movies
    print("Ingesting movies...")
    movie_file = os.path.join(RAW_DATA_DIR, 'u.item')
    movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)]
    df_movies = pd.read_csv(movie_file, sep='|', names=movie_cols, encoding='ISO-8859-1')
    df_movies = df_movies[['movie_id', 'title']]
    df_movies.to_sql('movies', engine, if_exists='replace', index=False)
    
    # 3. Ratings
    print("Ingesting ratings...")
    rating_file = os.path.join(RAW_DATA_DIR, 'u.data')
    rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    df_ratings = pd.read_csv(rating_file, sep='\t', names=rating_cols)
    df_ratings.to_sql('ratings', engine, if_exists='replace', index=False)
    
    print("Data Ingestion Completed!")

if __name__ == "__main__":
    try:
        print(f"Detected ODBC Driver: {get_driver()}")
        create_database()
        ingest_data()
    except Exception as e:
        print(f"Error during ingestion: {e}")
