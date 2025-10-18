# db/models.py
"""
Declarative SQLAlchemy models for the MarketSentinel TimescaleDB schema.
Compatible with async SQLAlchemy (used only for reflection and analytics queries).
"""

from sqlalchemy import Column, String, Float, TIMESTAMP, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()
SCHEMA = "marketsentinel"

# --- Market Data --------------------------------------------------------------

class MarketData(Base):
    __tablename__ = "market_data"
    __table_args__ = {"schema": SCHEMA}

    symbol = Column(String, primary_key=True)
    timestamp = Column(TIMESTAMP(timezone=True), primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    source = Column(String)
    retrieved_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

# --- Financial News -----------------------------------------------------------

class NewsData(Base):
    __tablename__ = "news_data"
    __table_args__ = {"schema": SCHEMA}

    news_id = Column(String, primary_key=True)
    symbol = Column(String)
    timestamp = Column(TIMESTAMP(timezone=True))
    headline = Column(Text)
    summary = Column(Text)
    source = Column(String)
    url = Column(String)
    sentiment_score = Column(Float)
    keywords = Column(JSON)
    retrieved_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

# --- Macroeconomic Indicators -------------------------------------------------

class MacroData(Base):
    __tablename__ = "macro_data"
    __table_args__ = {"schema": SCHEMA}

    country = Column(String, primary_key=True)
    indicator = Column(String, primary_key=True)
    timestamp = Column(TIMESTAMP(timezone=True), primary_key=True)
    value = Column(Float)
    source = Column(String)
    retrieved_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

# --- Transaction Data ---------------------------------------------------------

class TransactionData(Base):
    __tablename__ = "transaction_data"
    __table_args__ = {"schema": SCHEMA}

    account_id = Column(String, primary_key=True)
    transaction_id = Column(String, primary_key=True)
    timestamp = Column(TIMESTAMP(timezone=True), primary_key=True)
    amount = Column(Float)
    currency = Column(String(10))
    description = Column(Text)
    category = Column(String)
    balance_after = Column(Float)
    source = Column(String)
    retrieved_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

# --- Ingestion Log ------------------------------------------------------------

class IngestionLog(Base):
    __tablename__ = "ingestion_log"
    __table_args__ = {"schema": SCHEMA}

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    source = Column(String)
    job_type=Column(Text)
    error_message=Column(Text)
    status = Column(String)
    message = Column(Text)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now())
