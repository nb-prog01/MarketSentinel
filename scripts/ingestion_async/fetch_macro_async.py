import os, datetime as dt, pandas as pd
from sqlalchemy import text
from db.async_db import engine
from scripts.ingestion_async.async_utils import get_client, http_get, LIMITERS

FRED_KEY=os.getenv("FRED_KEY")
WORLD_BASE="https://api.worldbank.org/v2"

fred_series_map={
    "GDP":"GDP", #placeholder, replace with real series IDs
    "Inflation":"FPCPITOTLZGUSA",
    "Unemployment":"UNRATE",
    "InterestRate":"FEDFUNDS"
}

wb_map={
    "GDP":"NY.GDP.MKTP.CD",
    "Inflation":"FP.CPI.TOTL.ZG",
    "Unemployment":"SL.UEM.TOTL.ZS",
    "InterestRate":"FR.INR.RINR"
}

async def fetch_fred(series_id:str):
    url="https://api.stlouisfed.org/fred/series/observations"
    params={"series_id":series_id, "api_key":FRED_KEY, "file_type":"json"}
    return await http_get(None, url, params=params, limiter=LIMITERS["fred"])

async def fetch_worldbank(indicator:str, country:str="US"):
    url=f"{WORLD_BASE}/country/{country}/indicator/{indicator}"
    params={"format":"json", "per_page":5000}
    return await http_get(None, url, params=params, limiter=LIMITERS["worldbank"])

def normalize_fred(obs, indicator_name):
    date=obs.get("date")
    val=obs.get("value")
    return{
        "country":"US",
        "indicator":indicator_name,
        "timestamp": pd.to_datetime(date, utc=True),
        "value":float(val) if val not in (None, ".","") else None,
        "source":"FRED",
        "retrieved_at": dt.datetime.utcnow()
    }

def normalize_wb(obs, indicator_name, country_code):
    date=obs.get("date")
    val=obs.get("value")
    return{
        "country":country_code,
        "indicator":indicator_name,
        "timestamp": pd.to_datetime(date,utc=True),
        "value":float(val) if val not in (None, ".","") else None,
        "source":"WorldBank",
        "retrieved_at": dt.datetime.utcnow()
    }

async def upsert_macro_rows(rows):
    if not rows:
        return 0
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO marketsentinel.macro_data
            (country, indicator, timestamp, value, source, retrieved_at)
            VALUES (:country, :indicator, :timestamp, :value, :source, :retrieved_at)
            ON CONFLICT (country, indicator, timestamp)
            DO UPDATE SET value=EXCLUDED.value, retrieved_at=EXCLUDED.retrieved_at;
            """), rows
        )
    return len(rows)

async def ingest_macro():
    rows=[]
    #try FRED first for each indicator
    for name, s in fred_series_map.items():
        try:
            payload=await fetch_fred(s)
            observations=payload.get("observations", [])
            for obs in observations:
                rows.append(normalize_fred(obs, name))
        except Exception:
                #fallback to World Bank
            wb_id=wb_map.get(name)
            if wb_id:
                try:
                    payload=await fetch_worldbank(wb_id)
                    for obs in payload[1]:
                        rows.append(normalize_wb(obs, name, obs.get("country", {}).get("id", "US")))
                except Exception:
                        pass
    return await upsert_macro_rows(rows)