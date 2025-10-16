# scripts/schema_validation.py
"""
Schema validation utility for MarketSentinel database.
Ensures required tables, schemas, and hypertables exist and are correctly configured.
Run this before starting ingestion.
"""

import asyncio
from sqlalchemy import text
from db.async_db import engine

REQUIRED_TABLES = [
    "market_data",
    "news_data",
    "macro_data",
    "transaction_data",
    "ingestion_log"
]

SCHEMA = "marketsentinel"

async def check_schema_exists(conn):
    res = await conn.execute(
        text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = :schema"),
        {"schema": SCHEMA}
    )
    return res.scalar() is not None

async def list_tables(conn):
    res = await conn.execute(
        text("SELECT table_name FROM information_schema.tables WHERE table_schema = :schema"),
        {"schema": SCHEMA}
    )
    return [r[0] for r in res.fetchall()]

async def check_hypertable(conn, table_name):
    res = await conn.execute(
        text("""
        SELECT hypertable_name
        FROM timescaledb_information.hypertables
        WHERE hypertable_schema = :schema AND hypertable_name = :table;
        """),
        {"schema": SCHEMA, "table": table_name}
    )
    return res.scalar() is not None

# async def get_primary_keys(conn, table_name):
#     res = await conn.execute(
#         text("""
#         SELECT a.attname
#         FROM pg_index i
#         JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
#         WHERE i.indrelid = :schema || '.' || :table::regclass
#         AND i.indisprimary;
#         """),
#         {"schema": SCHEMA, "table": table_name}
#     )
#     return [r[0] for r in res.fetchall()]
async def get_primary_keys(conn, table_name):
    qualified = f"{SCHEMA}.{table_name}"
    query = text(f"""
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a
          ON a.attrelid = i.indrelid
         AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = '{qualified}'::regclass
          AND i.indisprimary;
    """)
    res = await conn.execute(query)
    return [r[0] for r in res.fetchall()]



async def validate_schema():
    async with engine.begin() as conn:
        print("🔍 Validating MarketSentinel database schema...\n")

        schema_exists = await check_schema_exists(conn)
        if not schema_exists:
            print(f"❌ Schema '{SCHEMA}' not found. Create it first:")
            print(f"    CREATE SCHEMA {SCHEMA};")
            return

        print(f"✅ Schema '{SCHEMA}' exists.")

        existing_tables = await list_tables(conn)
        missing_tables = [t for t in REQUIRED_TABLES if t not in existing_tables]

        if missing_tables:
            print(f"❌ Missing tables: {', '.join(missing_tables)}")
        else:
            print("✅ All required tables are present.")

        print("\n🧩 Checking hypertables:")
        for table in REQUIRED_TABLES:
            if table == "ingestion_log":
                print(f"   ⚠️  {table}: Not expected to be hypertable (log table).")
                continue
            is_hyper = await check_hypertable(conn, table)
            print(f"   {'✅' if is_hyper else '❌'} {table} -> {'Hypertable' if is_hyper else 'Not a hypertable'}")

        print("\n🔑 Checking primary keys:")
        for table in REQUIRED_TABLES:
            keys = await get_primary_keys(conn, table)
            if not keys:
                print(f"   ❌ {table}: No primary key set.")
            else:
                print(f"   ✅ {table}: Primary key(s) -> {', '.join(keys)}")

        print("\n🎯 Validation complete.")

async def main():
    try:
        await validate_schema()
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
