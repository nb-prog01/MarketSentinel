import os
from sqlalchemy.ext.asyncio import create_async_engine,AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL","postgresql+asyncpg://postgres:arise#007@localhost:5432/marketsentinel")

engine=create_async_engine(DB_URL, pool_size=20, max_overflow=40, pool_timeout=60, pool_recycle=1800, future=True)
AsyncSessionLocal=sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

#helper to get session from async context
async def get_session():
    async with AsyncSessionLocal() as session:
        yield session
