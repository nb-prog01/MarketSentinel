import os
import argparse
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd

from db.async_db import engine
from scripts.ingestion_async.async_utils import get_client, http_get, LIMITERS
from scripts.tickers_list import SYMBOLS, FRED_INDICATORS
from sqlalchemy import text

# Env keys (assumed in .env)
TWELVE_KEY = os.getenv("TWELVE_KEY")
ALPHA_KEY = os.getenv("ALPHA_KEY")
MARKETAUX_KEY = os.getenv("MARKETAUX_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FRED_KEY = os.getenv("FRED_KEY")
WORLD_BASE = "https://api.worldbank.org/v2"
OBP_BASE = os.getenv("OBP_BASE", "https://apisandbox.openbankproject.com/obp/v4.0.0")

SCHEMA = "marketsentinel"

# Utility Normalizers

def normalize_keywords(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return []

def parse_timestamp(ts) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return pd.to_datetime(ts, utc=True).to_pydatetime()
    except Exception:
        # last resort
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

def now_utc():
    return datetime.utcnow()


# DB helpers: checkpoint + upsert

async def chunk_done(conn, job_type: str, symbol: Optional[str], start_iso: str, end_iso: str) -> bool:
    q = text(f"""
        SELECT 1 FROM {SCHEMA}.ingestion_log
        WHERE job_type = :job_type
          AND symbol = :symbol
          AND start_time = :start_iso
          AND end_time = :end_iso
          AND status = 'SUCCESS'
        LIMIT 1;
    """)
    r = await conn.execute(q, {"job_type": job_type, "symbol": symbol, "start_iso": start_iso, "end_iso": end_iso})
    return r.scalar() is not None

async def log_chunk(conn, job_type: str, symbol: Optional[str], start_iso: str, end_iso: str, status: str, message: str = ""):
    q = text(f"""
        INSERT INTO {SCHEMA}.ingestion_log
        (job_type, symbol, start_time, end_time, status, error_message, created_at)
        VALUES (:job_type, :symbol, :start_iso, :end_iso, :status, :msg, now());
    """)
    await conn.execute(q, {"job_type": job_type, "symbol": symbol, "start_iso": start_iso, "end_iso": end_iso, "status": status, "msg": message[:3000]})

async def upsert_market_rows(rows: List[Dict[str,Any]], dry_run: bool):
    if not rows:
        return 0
    if dry_run:
        return len(rows)
    async with engine.begin() as conn:
        await conn.execute(
            text(f"""
            INSERT INTO {SCHEMA}.market_data
            (symbol,timestamp,open,high,low,close,volume,source,retrieved_at)
            VALUES (:symbol,:timestamp,:open,:high,:low,:close,:volume,:source,:retrieved_at)
            ON CONFLICT (symbol,timestamp) DO NOTHING;
            """),
            rows
        )
    return len(rows)

async def upsert_news_rows(rows: List[Dict[str,Any]], dry_run: bool):
    if not rows:
        return 0
    if dry_run:
        return len(rows)
    async with engine.begin() as conn:
        await conn.execute(
            text(f"""
            INSERT INTO {SCHEMA}.news_data
            (news_id,symbol,timestamp,headline,summary,source,url,sentiment_score,keywords,retrieved_at)
            VALUES (:news_id,:symbol,:timestamp,:headline,:summary,:source,:url,:sentiment_score,:keywords,:retrieved_at)
            ON CONFLICT (news_id) DO UPDATE
              SET sentiment_score = EXCLUDED.sentiment_score,
                  retrieved_at = EXCLUDED.retrieved_at;
            """), rows
        )
    return len(rows)

async def upsert_macro_rows(rows: List[Dict[str,Any]], dry_run: bool):
    if not rows:
        return 0
    if dry_run:
        return len(rows)
    async with engine.begin() as conn:
        await conn.execute(
            text(f"""
            INSERT INTO {SCHEMA}.macro_data
            (country,indicator,timestamp,value,source,retrieved_at)
            VALUES (:country,:indicator,:timestamp,:value,:source,:retrieved_at)
            ON CONFLICT (country,indicator,timestamp)
            DO UPDATE SET value = EXCLUDED.value, retrieved_at = EXCLUDED.retrieved_at;
            """), rows
        )
    return len(rows)

async def upsert_tx_rows(rows: List[Dict[str,Any]], dry_run: bool):
    if not rows:
        return 0
    if dry_run:
        return len(rows)
    async with engine.begin() as conn:
        await conn.execute(
            text(f"""
            INSERT INTO {SCHEMA}.transaction_data
            (account_id,transaction_id,timestamp,amount,currency,description,category,balance_after,source,retrieved_at)
            VALUES (:account_id,:transaction_id,:timestamp,:amount,:currency,:description,:category,:balance_after,:source,:retrieved_at)
            ON CONFLICT (account_id,transaction_id,timestamp) DO NOTHING;
            """), rows
        )
    return len(rows)


# Fetchers for historical endpoints

async def fetch_market_history(session, symbol: str, start: datetime, end: datetime, interval: str="1day"):
    """Twelve Data primary historical time_series. Fallback to Alpha for daily."""
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    # Twelve Data
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "start_date": start_s, "end_date": end_s, "apikey": TWELVE_KEY, "format":"JSON", "outputsize":5000}
    try:
        j = await http_get(session, url, params=params, limiter=LIMITERS.get("twelve"))
        values = j.get("values", [])
        return values
    except Exception:
        # fallback to Alpha (daily range handled by outputsize/pagination; we do simple attempt)
        url2 = "https://www.alphavantage.co/query"
        params2 = {"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":symbol,"apikey":ALPHA_KEY,"outputsize":"full"}
        j2 = await http_get(session, url2, params=params2, limiter=LIMITERS.get("alpha"))
        key = "Time Series (Daily)"
        if key in j2:
            # filter by date range
            rows = []
            for d, rec in j2[key].items():
                if start_s <= d <= end_s:
                    rec2 = rec.copy()
                    rec2["timestamp"] = d
                    rows.append(rec2)
            return rows
        return []

async def fetch_news_history(session, symbol: str, start: datetime, end: datetime, per_page=100):
    """Marketaux primary; fallback to NewsAPI"""
    start_s = start.isoformat()
    end_s = end.isoformat()
    url = "https://api.marketaux.com/v1/news/all"
    params = {"symbols": symbol, "api_token": MARKETAUX_KEY, "language":"en", "published_after": start_s, "published_before": end_s, "per_page": per_page}
    try:
        j = await http_get(session, url, params=params, limiter=LIMITERS.get("marketaux"))
        return j.get("data", [])
    except Exception:
        url2 = "https://newsapi.org/v2/everything"
        params2 = {"q": symbol, "apiKey": NEWSAPI_KEY, "from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"), "pageSize": per_page}
        j2 = await http_get(session, url2, params=params2, limiter=LIMITERS.get("newsapi"))
        return j2.get("articles", [])

async def fetch_fred_series(session, series_id: str, start: datetime, end: datetime):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": FRED_KEY, "file_type": "json", "observation_start": start.strftime("%Y-%m-%d"), "observation_end": end.strftime("%Y-%m-%d")}
    j = await http_get(session, url, params=params, limiter=LIMITERS.get("fred"))
    return j.get("observations", [])

async def fetch_worldbank_indicator(session, indicator: str, country: str, start: datetime, end: datetime):
    url = f"{WORLD_BASE}/country/{country}/indicator/{indicator}"
    params = {"format":"json", "date": f"{start.year}:{end.year}", "per_page":1000}
    j = await http_get(session, url, params=params, limiter=LIMITERS.get("worldbank"))
    return j[1] if isinstance(j, list) and len(j) > 1 else []

async def fetch_obp_transactions(session, bank_id: str, account_id: str, start: datetime, end: datetime):
    url = f"{OBP_BASE}/banks/{bank_id}/accounts/{account_id}/transactions"
    params = {"date_from": start.strftime("%Y-%m-%d"), "date_to": end.strftime("%Y-%m-%d")}
    try:
        j = await http_get(session, url, params=params, limiter=None)
        return j.get("transactions", [])
    except Exception:
        return []


# Normalizers for fetched payloads

def normalize_market_values(values: List[Dict[str,Any]], symbol: str):
    out = []
    for v in values:
        ts = v.get("datetime") or v.get("timestamp") or v.get("date")
        parsed = parse_timestamp(ts)
        if parsed is None:
            continue
        row = {
            "symbol": symbol,
            "timestamp": parsed,
            "open": float(v.get("open") or v.get("1. open") or 0.0),
            "high": float(v.get("high") or v.get("2. high") or 0.0),
            "low": float(v.get("low") or v.get("3. low") or 0.0),
            "close": float(v.get("close") or v.get("4. close") or 0.0),
            "volume": float(v.get("volume") or v.get("6. volume") or v.get("5. volume") or 0.0),
            "source": v.get("source") or None,
            "retrieved_at": now_utc()
        }
        out.append(row)
    return out

def normalize_news_items(items: List[Dict[str,Any]], symbol: str):
    out = []
    for it in items:
        published_at = it.get("published_at") or it.get("publishedAt") or it.get("published") or it.get("date")
        ts = parse_timestamp(published_at) or now_utc()
        url = it.get("url") or it.get("link")
        title = it.get("title") or ""
        news_id = it.get("id") or f"{symbol}|{hash((url or '') + title + str(published_at))}"
        keywords = normalize_keywords(it.get("keywords") or it.get("tags") or "")
        row = {
            "news_id": str(news_id),
            "symbol": symbol,
            "timestamp": ts,
            "headline": title,
            "summary": it.get("description") or it.get("summary") or "",
            "source": (it.get("source") or {}).get("name") if isinstance(it.get("source"), dict) else it.get("source"),
            "url": url,
            "sentiment_score": None,
            "keywords": keywords,
            "retrieved_at": now_utc()
        }
        out.append(row)
    return out

def normalize_fred_obs(observations: List[Dict[str,Any]], indicator_name: str, country="US"):
    out = []
    for o in observations:
        date = o.get("date") or o.get("obs_date")
        ts = parse_timestamp(date)
        if ts is None:
            continue
        val = o.get("value")
        try:
            valf = float(val) if val not in (None, ".", "") else None
        except Exception:
            valf = None
        out.append({
            "country": country,
            "indicator": indicator_name,
            "timestamp": ts,
            "value": valf,
            "source": "FRED",
            "retrieved_at": now_utc()
        })
    return out

def normalize_worldbank_obs(items: List[Dict[str,Any]], indicator_name: str):
    out = []
    for o in items:
        date = o.get("date")
        ts = parse_timestamp(date)
        if ts is None:
            continue
        val = o.get("value")
        try:
            valf = float(val) if val not in (None, ".", "") else None
        except Exception:
            valf = None
        country = o.get("country", {}).get("id", "UNK")
        out.append({
            "country": country,
            "indicator": indicator_name,
            "timestamp": ts,
            "value": valf,
            "source": "WorldBank",
            "retrieved_at": now_utc()
        })
    return out

def normalize_obp_txs(txs: List[Dict[str,Any]]):
    out = []
    for t in txs:
        acct = t.get("this_account", {}).get("id") or "unknown"
        txid = t.get("id") or t.get("transaction_id") or str(hash(str(t)))
        posted = (t.get("details") or {}).get("value", {}).get("posted")
        ts = parse_timestamp(posted) or now_utc()
        amount = float((t.get("details") or {}).get("value", {}).get("amount") or 0.0)
        currency = (t.get("details") or {}).get("value", {}).get("currency") or "USD"
        out.append({
            "account_id": acct,
            "transaction_id": txid,
            "timestamp": ts,
            "amount": amount,
            "currency": currency,
            "description": (t.get("details") or {}).get("description") or "",
            "category": None,
            "balance_after": None,
            "source": "open_bank_project",
            "retrieved_at": now_utc()
        })
    return out


# Backfill orchestration

async def process_chunks_for_symbol(fetcher, normalizer, upserter, symbol: str,
                                    start: datetime, end: datetime, chunk_days: int,
                                    dry_run: bool, job_type: str, concurrency_semaphore: asyncio.Semaphore):
    cur = start
    async with engine.begin() as conn:
        while cur < end:
            chunk_end = min(cur + timedelta(days=chunk_days), end)
            start_iso = cur
            end_iso = chunk_end
            # skip if already done
            if await chunk_done(conn, job_type, symbol, start_iso, end_iso):
                print(f"[SKIP] {job_type}:{symbol} {start_iso} -> {end_iso} already SUCCESS")
                cur = chunk_end
                continue
            # fetch in guarded concurrency slot
            async with concurrency_semaphore:
                try:
                    async with get_client() as session:
                        raw = await fetcher(session, symbol, cur, chunk_end)
                        normalized = normalizer(raw, symbol) if job_type != "macro" else normalizer(raw, job_type)
                        # ensure sanitized types (keywords -> list, timestamp -> datetime)
                        for r in normalized:
                            if "keywords" in r:
                                r["keywords"] = normalize_keywords(r.get("keywords"))
                            if "timestamp" in r and isinstance(r["timestamp"], str):
                                r["timestamp"] = parse_timestamp(r["timestamp"])
                        count = await upserter(normalized, dry_run)
                        msg = f"{count} rows"
                        async with engine.begin() as conn2:
                            await log_chunk(conn2, job_type, symbol, start_iso, end_iso, "SUCCESS", msg)
                        print(f"[OK] {job_type}:{symbol} {start_iso} -> {end_iso} inserted {count}")
                except Exception as e:
                    # log failure for this chunk
                    async with engine.begin() as conn2:
                        await log_chunk(conn2, job_type, symbol, start_iso, end_iso, "FAILED", str(e))
                    print(f"[FAIL] {job_type}:{symbol} {start_iso}->{end_iso}: {e}")
            cur = chunk_end

async def backfill_markets(symbols: List[str], start: datetime, end: datetime, chunk_days: int, dry_run: bool, concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for s in symbols:
        tasks.append(asyncio.create_task(process_chunks_for_symbol(fetch_market_history, normalize_market_values, upsert_market_rows, s, start, end, chunk_days, dry_run, "market", sem)))
    await asyncio.gather(*tasks)

async def backfill_news(symbols: List[str], start: datetime, end: datetime, chunk_days: int, dry_run: bool, concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for s in symbols:
        # reuse chunking by days but news fetcher accepts start/end
        tasks.append(asyncio.create_task(process_chunks_for_symbol(fetch_news_history, normalize_news_items, upsert_news_rows, s, start, end, chunk_days, dry_run, "news", sem)))
    await asyncio.gather(*tasks)

async def backfill_macro(indicators: List[str], start: datetime, end: datetime, dry_run: bool):
    # simple sequential processing for macro; maps indicator -> FRED id must be provided externally
    async with get_client() as session:
        rows = []
        for ind in indicators:
            try:
                obs = await fetch_fred_series(session, ind, start, end)
                rows.extend(normalize_fred_obs(obs, ind))
            except Exception:
                wb = await fetch_worldbank_indicator(session, ind, "US", start, end)
                rows.extend(normalize_worldbank_obs(wb, ind))
    count = await upsert_macro_rows(rows, dry_run)
    print(f"[OK] macro -> {count}")

async def backfill_transactions(bank_id: str, account_id: str, start: datetime, end: datetime, dry_run: bool):
    async with get_client() as session:
        txs = await fetch_obp_transactions(session, bank_id, account_id, start, end)
        rows = normalize_obp_txs(txs) if txs else []
    count = await upsert_tx_rows(rows, dry_run)
    print(f"[OK] transactions -> {count}")


# CLI & main

def parse_args():
    p = argparse.ArgumentParser(description="MarketSentinel historical backfill")
    p.add_argument("--symbols", type=str, default="AAPL", help="Comma-separated symbols")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--chunk-size", type=int, default=30, help="Days per chunk")
    p.add_argument("--concurrency", type=int, default=4, help="Max concurrent symbol backfills")
    p.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    p.add_argument("--do-news", action="store_true", help="Backfill news as well")
    p.add_argument("--do-macro", action="store_true", help="Backfill macro (pass indicator IDs in FRED map)")
    p.add_argument("--do-tx", action="store_true", help="Backfill transactions")
    p.add_argument("--bank", type=str, default="rbs", help="OBP bank id")
    p.add_argument("--account", type=str, default="savings", help="OBP account id")
    return p.parse_args()

async def main():
    args = parse_args()
    # symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    symbols=SYMBOLS
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    chunk_days = args.chunk_size
    dry_run = args.dry_run
    concurrency = args.concurrency

    print(f"[START] backfill symbols={symbols} start={start} end={end} chunk_days={chunk_days} dry_run={dry_run}")

    # 1) market data
    await backfill_markets(symbols, start, end, chunk_days, dry_run, concurrency)

    # 2) news (optional)
    if args.do_news:
        await backfill_news(symbols, start, end, chunk_days, dry_run, concurrency)

    # 3) macro (optional) - supply FRED series ids inside fred_series_map or pass them directly
    if args.do_macro:
        # example indicators list; replace with real series ids mapping
        indicators = FRED_INDICATORS
        await backfill_macro(indicators, start, end, dry_run)

    # 4) transactions (optional)
    if args.do_tx:
        await backfill_transactions(args.bank, args.account, start, end, dry_run)

    await engine.dispose()
    print("[DONE] backfill complete")

if __name__ == "__main__":
    asyncio.run(main())
