import os, datetime as dt, pandas as pd, hashlib
from sqlalchemy import text
from db.async_db import engine
from scripts.ingestion_async.async_utils import get_client, http_get, LIMITERS

MARKETAUX_KEY=os.getenv("MARKETAUX_KEY")
NEWSAPI_KEY=os.getenv("NEWSAPI_KEY")

def make_news_id(url, title, published_at):
    return hashlib.sha256(f"{url}|{title}|{published_at}".encode()).hexdigest()

async def fetch_marketaux(session, symbol:str):
    url="https://api.marketaux.com/v1/news/all"
    params={"symbols":symbol, "api_token":MARKETAUX_KEY, "language":"en", "filter_entities":"True"}
    data=await http_get(session, url, params=params, limiter=LIMITERS["marketaux"])
    return data.get("data",[])

async def fetch_newsapi(session, keyword:str):
    url="https://newsapi.org/v2/everything"
    params={"q":keyword, "apiKey":NEWSAPI_KEY, "language":"en", "pageSize":100}
    data=await http_get(session, url, params=params, limiter=LIMITERS["newsapi"])
    return data.get("articles", [])



def normalize_marketaux_item(it: dict, symbol: str):
    published_at=it.get("published_at") or it.get("publishedAt")
    url=it.get("url") or it.get("link")
    title=it.get("title") or ""
    nid=it.get("id") or make_news_id(url or "", title, published_at or "")
    raw_keywords=it.get("keywords","")
    if isinstance(raw_keywords, str):
        keywords=[kw.strip() for kw in raw_keywords.split(",") if kw.strip()]
    elif isinstance (raw_keywords, list):
        keywords=raw_keywords
    else:
        keywords=[]

    return {
        "news_id": nid,
        "symbol": symbol,
        "timestamp": pd.to_datetime(published_at, utc=True),
        "headline": title,
        "summary": it.get("description") or it.get("summary") or "",
        "source": it.get("source") if isinstance(it.get("source"), str) else (it.get("source")or {}).get("name"),
        "url":url,
        "sentiment_score":None,
        "keywords":keywords,
        "retrieved_at": dt.datetime.utcnow()
    }

def normalize_newsapi_item(it: dict, symbol: str):
    published_at=it.get("published_at") or it.get("publishedAt")
    url=it.get("url")
    title=it.get("title") or ""
    nid=make_news_id(url or "", title, published_at or "")
    raw_keywords=it.get("keywords","")
    if isinstance(raw_keywords, str):
        keywords=[kw.strip() for kw in raw_keywords.split(",") if kw.strip()]
    elif isinstance (raw_keywords, list):
        keywords=raw_keywords
    else:
        keywords=[]
    return {
        "news_id": nid,
        "symbol": symbol,
        "timestamp": pd.to_datetime(published_at, utc=True),
        "headline": title,
        "summary": it.get("description") or "",
        "source": (it.get("source")or {}).get("name"),
        "url":url,
        "sentiment_score":None,
        "keywords":keywords,
        "retrieved_at": dt.datetime.utcnow()
    }

async def upsert_news_rows(rows:list):
    if not rows:
        return 0
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO marketsentinel.news_data
            (news_id, symbol, timestamp, headline, summary, source, url, sentiment_score, keywords, retrieved_at)
            VALUES (:news_id, :symbol, :timestamp, :headline, :summary, :source, :url, :sentiment_score, :keywords, :retrieved_at)
            ON CONFLICT (symbol, news_id, timestamp) DO UPDATE
            SET sentiment_score =EXCLUDED.sentiment_score,
            retrieved_at = EXCLUDED.retrieved_at;
            """), rows
        )
    return len(rows)

async def ingest_news_for_symbol(symbol:str)->int:
    async with get_client() as session:
        try:
            items=await fetch_marketaux(session, symbol)
        except Exception:
            items=await fetch_newsapi(session, symbol)
    rows=[]
    for it in items:
        #choose normalizer by presence of certain keys
        if it.get("id") or it.get("published_at"):
            rows.append(normalize_marketaux_item(it, symbol))
        else:
            rows.append(normalize_newsapi_item(it, symbol))
    return await upsert_news_rows(rows)