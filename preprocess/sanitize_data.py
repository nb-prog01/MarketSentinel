from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import mplfinance as mpf
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL_psycopg2 = os.getenv("DATABASE_URL_PSYCOPG2","postgresql+psycopg2://postgres:arise#007@localhost:5432/marketsentinel")

engine=create_engine(DB_URL_psycopg2)

conn=engine.connect()

query="""
SELECT symbol, timestamp, open,high,low,close,volume,interval,source FROM marketsentinel.market_data
ORDER BY symbol, timestamp ASC
"""

df=pd.read_sql(query, conn)

print(df.head())

