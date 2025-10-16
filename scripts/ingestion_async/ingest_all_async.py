import asyncio, datetime as dt, argparse
from sqlalchemy import text
from db.async_db import engine
from scripts.ingestion_async.fetch_market_async import ingest_market_symbol
from scripts.ingestion_async.fetch_news_async import ingest_news_for_symbol
from scripts.ingestion_async.fetch_macro_async import ingest_macro
from scripts.ingestion_async.fetch_transactions_async import ingest_transactions

SYMBOLS = ["AAPL", "MSFT", "GOOG"]
MAX_CONCURRENT = 3

# Flag that will be set by CLI argument
DRY_RUN = False

async def log_ingest(source, status, message=""):
    if DRY_RUN:
        print(f"[DRY] Log {source} | {status}: {message}")
        return
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO marketsentinel.ingestion_log (source, status, message, timestamp)
            VALUES (:src, :status, :msg, :ts);
            """),
            {"src": source, "status": status, "msg": message[:3000], "ts": dt.datetime.utcnow()}
        )

async def safe_run(name, coro, *args):
    try:
        count = await coro(*args)
        if DRY_RUN:
            print(f"[DRY] {name} -> {count} rows fetched (not inserted)")
        else:
            await log_ingest(name, "success", f"{count} records")
            print(f"[OK] {name} -> {count}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        if not DRY_RUN:
            await log_ingest(name, "failed", tb[:3000])
        print(f"[FAIL] {name}: {e}")

async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async def limited(fn, *a):
        async with sem:
            await fn(*a)

    tasks = []
    for s in SYMBOLS:
        tasks.append(asyncio.create_task(limited(safe_run, f"market:{s}", ingest_market_symbol, s, "1h")))
        tasks.append(asyncio.create_task(limited(safe_run, f"news:{s}", ingest_news_for_symbol, s)))

    await asyncio.gather(*tasks)
    await safe_run("macro", ingest_macro)
    await safe_run("transactions", ingest_transactions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MarketSentinel async data ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to DB")
    args = parser.parse_args()

    DRY_RUN = args.dry_run
    asyncio.run(main())
