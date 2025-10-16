import os, datetime as dt, pandas as pd
from sqlalchemy import text
from db.async_db import engine
from scripts.ingestion_async.async_utils import get_client, http_get, LIMITERS
from typing import List

TWELVE_KEY=os.getenv("TWELVE_KEY")
ALPHA_KEY=os.getenv("ALPHA_KEY")

async def fetch_twelve(session, symbol:str, interval:str="1h"):
    url="https://api.twelvedata.com/time_series"
    params={"symbol":symbol, "interval":interval, "apikey":TWELVE_KEY, "format":"JSON", "outputsize":5000}
    return await http_get(session, url, params=params, limiter=LIMITERS["twelve"])

async def fetch_alpha(session,symbol:str):
    url="https://www.alphavantage.co/query"
    params={"function":"TIME_SERIES_DAILY_ADJUSTED", "symbol":symbol, "apikey":ALPHA_KEY, "outputsize":"compact"}
    return await http_get(session,url,params=params,limiter=LIMITERS["alpha"])

# def normalize_market_payload(payload, symbol: str):
#     """
#     Normalizes market API payloads from TwelveData or Alpha Vantage into a consistent DataFrame.
#     Ensures a 'timestamp' column exists and numeric fields are coerced properly.
#     """

#     # Determine payload type
#     if isinstance(payload, dict) and "values" in payload:
#         # TwelveData format
#         rows = payload["values"]
#         df = pd.DataFrame(rows)
#     elif isinstance(payload, dict) and "Time Series (Daily)" in payload:
#         # Alpha Vantage format
#         df = pd.DataFrame.from_dict(payload["Time Series (Daily)"], orient="index").reset_index()
#         df = df.rename(columns={"index": "timestamp"})
#     else:
#         raise ValueError("Unknown market payload shape")

#     # Normalize timestamp
#     timestamp_keys = ["timestamp", "datetime", "date", "time"]
#     for key in timestamp_keys:
#         if key in df.columns:
#             df["timestamp"] = pd.to_datetime(df[key], utc=True)
#             # Make tz-aware
#             df["timestamp"] = df["timestamp"].dt.tz_localize("UTC", ambiguous="NaT", nonexistent="shift_forward")
#             break
#     else:
#         # fallback if no timestamp present
#         df["timestamp"] = pd.to_datetime(dt.datetime.utcnow()).tz_localize("UTC")
#     #ensure retreived_at is utc
#     df["retrieved_at"]=pd.to_datetime(dt.datetime.utcnow()).tz_localize("UTC")

#     # Helper to pick numeric columns safely
#     def pick(col_options):
#         for c in col_options:
#             if c in df.columns:
#                 return pd.to_numeric(df[c], errors="coerce")
#         return None

#     out = pd.DataFrame({
#         "symbol": symbol,
#         "timestamp": df["timestamp"],
#         "open": pick(["open", "1. open"]),
#         "high": pick(["high", "2. high"]),
#         "low": pick(["low", "3. low"]),
#         "close": pick(["close", "4. close", "adjusted close"]),
#         "volume": pick(["volume", "5. volume", "6. volume"]),
#         "source": payload.get("meta", {}).get("provider") if isinstance(payload, dict) and "meta" in payload else None,
#         "retrieved_at": pd.to_datetime(dt.datetime.utcnow()).tz_localize("UTC")
#     })

#     # Drop rows with NaT timestamps just in case
#     out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

#     return out
def normalize_market_payload(payload, symbol: str):
    """
    Normalizes market API payloads from TwelveData or Alpha Vantage into a consistent DataFrame.
    Ensures 'timestamp' and 'retrieved_at' are timezone-aware (UTC) for PostgreSQL insertion.
    """
    import pandas as pd
    import datetime as dt

    # Determine payload type
    if isinstance(payload, dict) and "values" in payload:
        # TwelveData format
        rows = payload["values"]
        df = pd.DataFrame(rows)
    elif isinstance(payload, dict) and "Time Series (Daily)" in payload:
        # Alpha Vantage format
        df = pd.DataFrame.from_dict(payload["Time Series (Daily)"], orient="index").reset_index()
        df = df.rename(columns={"index": "timestamp"})
    else:
        raise ValueError("Unknown market payload shape")

    # Normalize timestamp
    timestamp_keys = ["timestamp", "datetime", "date", "time"]
    for key in timestamp_keys:
        if key in df.columns:
            df["timestamp"] = pd.to_datetime(df[key], errors="coerce")
            # If tz-naive, localize to UTC; if already tz-aware, convert to UTC
            df["timestamp"] = df["timestamp"].apply(
                lambda x: x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")
            )
            break
    else:
        # fallback if no timestamp present
        df["timestamp"] = pd.to_datetime(dt.datetime.utcnow()).tz_localize("UTC")

    # Helper to pick numeric columns safely
    def pick(col_options):
        for c in col_options:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce")
        return None

    # Prepare output DataFrame
    out = pd.DataFrame({
        "symbol": symbol,
        "timestamp": df["timestamp"],
        "open": pick(["open", "1. open"]),
        "high": pick(["high", "2. high"]),
        "low": pick(["low", "3. low"]),
        "close": pick(["close", "4. close", "adjusted close"]),
        "volume": pick(["volume", "5. volume", "6. volume"]),
        "source": payload.get("meta", {}).get("provider") if isinstance(payload, dict) and "meta" in payload else None,
        "retrieved_at": pd.to_datetime(dt.datetime.utcnow()).tz_localize("UTC")
    })

    # Drop rows with invalid timestamps
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)

    return out


async def upsert_market_rows(df:pd.DataFrame):
    if df.empty:
        return 0
    records=df.to_dict(orient="records")
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO marketsentinel.market_data (symbol, timestamp,open,high,low,close,volume,source,retrieved_at)
            VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume, :source, :retrieved_at)
            ON CONFLICT (symbol, timestamp) DO NOTHING;
            """),records
        )
    return len(records)

async def ingest_market_symbol(symbol:str, interval:str="1h")->int:
    async with get_client() as session:
        try:
            payload=await fetch_twelve(session,symbol,interval)
        except Exception as e:
            #Fallback to Alpha if Twelve fails
            payload=await fetch_alpha(session,symbol)
        df=normalize_market_payload(payload, symbol)
        count=await upsert_market_rows(df)
        return count