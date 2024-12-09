from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.config import config

SQLALCHEMY_DATABASE_URL = config.db_url
engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
