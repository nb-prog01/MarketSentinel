import asyncio
from db.async_db import engine
from sqlalchemy import text

async def check_connection_and_schema():
    async with engine.begin() as conn:
        # 1️⃣ Print database + schema
        res = await conn.execute(text("SELECT current_database(), current_schema();"))
        print("Database & Schema:", res.fetchall())  # no await here ✅

        # 2️⃣ List columns for ingestion_log in marketsentinel schema
        q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='marketsentinel'
          AND table_name='ingestion_log'
        ORDER BY ordinal_position;
        """)
        cols = await conn.execute(q)
        print("Columns:", cols.fetchall())  # no await here ✅

asyncio.run(check_connection_and_schema())
