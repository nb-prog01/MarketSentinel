import os
import asyncio
import uuid
import datetime as dt
import pandas as pd
import numpy as np
import random
import json
from sqlalchemy import text
from db.async_db import engine
from functools import partial
from typing import Optional

# ---------------------------
# CONFIG DEFAULTS
# ---------------------------
DEFAULT_CONFIG = {
    "source_type": "both",                  # "synthetic" | "kaggle" | "both"
    "synthetic_txn_count": 1_000_000,
    "synthetic_user_count": 10_000,
    "synthetic_accounts_per_user": (1, 10), # lognormal sampled
    "country_mix": {"US":0.35,"EU":0.25,"IN":0.15,"UK":0.08,"JP":0.07,"CA":0.05,"AU":0.03,"SG":0.02},
    "currency_map": {"US":"USD","EU":"EUR","IN":"INR","UK":"GBP","JP":"JPY","CA":"CAD","AU":"AUD","SG":"SGD"},
    "fx_edge_rate": 0.03,                   # % of transactions that are cross-currency
    "kaggle_file_path": None,
    "kaggle_mapping_yaml": None,            # optional YAML mapping for column names
    "batch_size": 50_000,
    "max_concurrency": 4,
    "async_enabled": True,
    "enable_merge_sources": True,
    "dedup_keys": ["transaction_id","source_dataset"],
    "fail_quarantine_table": True,
    "logging_level": "INFO",
    "metrics_sink": None,
    "dry_run": False
}

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def generate_synthetic_transactions(cfg):
    """
    Generates synthetic global transactions based on user, account, country, FX and transaction presets.
    """
    txn_count = cfg["synthetic_txn_count"]
    user_count = cfg["synthetic_user_count"]
    min_acc, max_acc = cfg["synthetic_accounts_per_user"]
    country_mix = cfg["country_mix"]
    currency_map = cfg["currency_map"]
    fx_rate = cfg["fx_edge_rate"]

    # Generate users and assign countries
    users = [f"user_{i}" for i in range(user_count)]
    countries = np.random.choice(list(country_mix.keys()), size=user_count, p=list(country_mix.values()))
    user_country_map = dict(zip(users, countries))

    # Generate accounts per user (lognormal)
    accounts = []
    account_user_map = {}
    for u in users:
        acc_count = max(min_acc, int(np.random.lognormal(mean=1.5, sigma=0.5)))  # lognormal
        acc_count = min(acc_count, max_acc)
        for _ in range(acc_count):
            acc_id = str(uuid.uuid4())
            accounts.append(acc_id)
            account_user_map[acc_id] = u

    # Prepare transactions
    tx_rows = []
    for _ in range(txn_count):
        acct = random.choice(accounts)
        user = account_user_map[acct]
        country = user_country_map[user]
        currency = currency_map.get(country, "USD")

        # FX edge chance
        if random.random() < fx_rate:
            currency = random.choice(list(currency_map.values()))

        txn_id = str(uuid.uuid4())
        ts = dt.datetime.utcnow() - dt.timedelta(days=random.randint(0,30), minutes=random.randint(0,1440))
        amount = round(random.uniform(1, 5000),2)
        direction = random.choice(["debit","credit"])
        if direction=="debit": amount *= -1

        tx_rows.append({
            "transaction_id": txn_id,
            "account_id": acct,
            "user_id": user,
            "timestamp": ts,
            "amount": amount,
            "currency": currency,
            "description": f"Synthetic TX {txn_id[:8]}",
            "category": None,
            "balance_before": None,
            "balance_after": None,
            "direction": direction,
            "country": country,
            "status": "completed",
            "source_dataset": "synthetic",
            "session_id": str(uuid.uuid4()),
            "counterparty_id": None,
            "merchant": None,
            "channel": None,
            "location": None,
            "is_fraud": False,
            "retrieved_at": dt.datetime.utcnow()
        })
    return tx_rows

def load_kaggle_transactions(cfg):
    """
    Loads and maps a Kaggle CSV to canonical schema.
    Auto-infers columns with optional mapping YAML override.
    """
    if not cfg["kaggle_file_path"]:
        return []

    df = pd.read_csv(cfg["kaggle_file_path"])
    mapping = cfg.get("kaggle_mapping_yaml", {})

    # Auto-map common patterns
    col_map = {
        "oldbalanceOrg":"balance_before",
        "newbalanceOrig":"balance_after",
        "nameOrig":"account_id",
        "nameDest":"counterparty_id",
        "step":"timestamp",
        "time":"timestamp",
        "amount":"amount",
        "type":"category"
    }
    # Apply YAML override if present
    col_map.update(mapping)

    df.rename(columns=col_map, inplace=True)

    # Ensure required fields exist
    required_cols = ["transaction_id","account_id","timestamp","amount","currency","direction","country","status","source_dataset","session_id"]
    for col in required_cols:
        if col not in df.columns:
            # Fill missing with defaults
            if col=="transaction_id": df[col] = [str(uuid.uuid4()) for _ in range(len(df))]
            elif col=="timestamp": df[col] = pd.to_datetime(dt.datetime.utcnow())
            elif col=="currency": df[col] = "USD"
            elif col=="direction": df[col] = "debit"
            elif col=="country": df[col] = "US"
            elif col=="status": df[col] = "completed"
            elif col=="source_dataset": df[col] = os.path.basename(cfg["kaggle_file_path"])
            elif col=="session_id": df[col] = str(uuid.uuid4())
            else: df[col] = None

    # Type casting
    df["amount"] = df["amount"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df.to_dict(orient="records")

async def upsert_tx_rows(rows, batch_size=50_000):
    """
    Async batch insert into TimescaleDB table.
    Deduplicate on primary key enforced in DB.
    """
    if not rows:
        return 0

    total_rows = len(rows)
    inserted_rows = 0
    async with engine.begin() as conn:
        for i in range(0, total_rows, batch_size):
            batch = rows[i:i+batch_size]
            await conn.execute(
                text("""
                INSERT INTO marketsentinel.transaction_data
                (transaction_id,account_id,user_id,timestamp,amount,currency,description,
                 category,balance_before,balance_after,direction,country,status,
                 source_dataset,session_id,counterparty_id,merchant,channel,location,is_fraud,retrieved_at)
                VALUES (:transaction_id,:account_id,:user_id,:timestamp,:amount,:currency,:description,
                        :category,:balance_before,:balance_after,:direction,:country,:status,
                        :source_dataset,:session_id,:counterparty_id,:merchant,:channel,:location,:is_fraud,:retrieved_at)
                ON CONFLICT (transaction_id,source_dataset,timestamp) DO NOTHING;
                """), batch
            )
            inserted_rows += len(batch)
    return inserted_rows

# ---------------------------
# MAIN INGESTION FUNCTION
# ---------------------------

async def ingest_transactions(cfg: Optional[dict]=None):
    cfg = cfg or DEFAULT_CONFIG
    session_id = str(uuid.uuid4())

    synthetic_rows, kaggle_rows = [], []

    if cfg["source_type"] in ["synthetic","both"]:
        synthetic_rows = generate_synthetic_transactions(cfg)
        # assign same session_id to all synthetic rows
        for r in synthetic_rows: r["session_id"] = session_id

    if cfg["source_type"] in ["kaggle","both"]:
        kaggle_rows = load_kaggle_transactions(cfg)
        # assign session_id if missing
        for r in kaggle_rows: 
            if not r.get("session_id"): r["session_id"] = session_id

    # Merge sources if enabled
    if cfg["enable_merge_sources"]:
        all_rows = synthetic_rows + kaggle_rows
    elif cfg["source_type"]=="synthetic":
        all_rows = synthetic_rows
    else:
        all_rows = kaggle_rows

    print(f"[INFO] Starting ingestion: {len(all_rows)} rows, session_id={session_id}")
    start_time = dt.datetime.utcnow()
    inserted = await upsert_tx_rows(all_rows, batch_size=cfg["batch_size"])
    duration = (dt.datetime.utcnow()-start_time).total_seconds()
    print(f"[INFO] Ingestion complete: inserted {inserted} rows in {duration:.2f}s, throughput={inserted/duration:.2f} rows/sec")
    return inserted

# ---------------------------
# RUN SCRIPT
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_type", type=str, default=DEFAULT_CONFIG["source_type"])
    parser.add_argument("--synthetic_txn_count", type=int, default=DEFAULT_CONFIG["synthetic_txn_count"])
    parser.add_argument("--kaggle_file_path", type=str, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(vars(args))

    inserted = asyncio.run(ingest_transactions(cfg))
    print(f"Total rows inserted: {inserted}")
