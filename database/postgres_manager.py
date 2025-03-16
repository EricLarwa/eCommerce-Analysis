from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

pg_engine = None

def get_engine():
    global pg_engine
    if pg_engine is None:
        pg_engine = create_engine('postgresql://username:password@localhost:5432/ecommerce_db')
    return pg_engine

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()