import os, datetime as dt, uuid, pandas as pd
from sqlalchemy import text
from db.async_db import engine
from scripts.ingestion_async.async_utils import get_client, http_get

OBP_BASE = os.getenv("OBP_BASE", "https://apisandbox.openbankproject.com/obp/v4.0.0")

async def fetch_obp_transactions(session, bank_id: str, account_id: str):
    url = f"{OBP_BASE}/banks/{bank_id}/accounts/{account_id}/transactions"
    return await http_get(session, url, limiter=None)

def make_synthetic(n=10):
    rows = []
    for i in range(n):
        ts = dt.datetime.utcnow() - dt.timedelta(minutes=i*5)
        rows.append({
            "account_id": "synth",
            "transaction_id": str(uuid.uuid4()),
            "timestamp": ts,
            "amount": float(100 + i),
            "currency": "USD",
            "description": "synthetic",
            "category": None,
            "balance_after": None,
            "source": "synthetic",
            "retrieved_at": dt.datetime.utcnow()
        })
    return rows

async def upsert_tx_rows(rows):
    if not rows:
        return 0
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            INSERT INTO marketsentinel.transaction_data
            (account_id,transaction_id,timestamp,amount,currency,description,category,balance_after,source,retrieved_at)
            VALUES (:account_id,:transaction_id,:timestamp,:amount,:currency,:description,:category,:balance_after,:source,:retrieved_at)
            ON CONFLICT (account_id,transaction_id,timestamp) DO NOTHING;
            """), rows
        )
    return len(rows)

async def ingest_transactions(bank_id="rbs", account_id="savings"):
    async with get_client() as session:
        try:
            payload = await fetch_obp_transactions(session, bank_id, account_id)
            txs = payload.get("transactions", [])
            rows = []
            for t in txs:
                # adapt to OBP shape; fallback to best effort
                acct = t.get("this_account", {}).get("id", account_id)
                txid = t.get("id") or str(uuid.uuid4())
                posted = t.get("details", {}).get("value", {}).get("posted") if t.get("details") else None
                ts = pd.to_datetime(posted, utc=True) if posted else dt.datetime.utcnow()
                amt = float(t.get("details", {}).get("value", {}).get("amount", 0)) if t.get("details") else 0.0
                rows.append({
                    "account_id": acct,
                    "transaction_id": txid,
                    "timestamp": ts,
                    "amount": amt,
                    "currency": t.get("details", {}).get("value", {}).get("currency", "USD") if t.get("details") else "USD",
                    "description": t.get("details", {}).get("description", ""),
                    "category": None,
                    "balance_after": None,
                    "source": "open_bank_project",
                    "retrieved_at": dt.datetime.utcnow()
                })
        except Exception:
            rows = make_synthetic(10)
    return await upsert_tx_rows(rows)
