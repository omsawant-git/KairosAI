# database/db.py
# ─────────────────────────────────────────────────────────────
# Database connection setup using SQLAlchemy
# This file is imported by every module that needs DB access
# ─────────────────────────────────────────────────────────────

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Read database URL from .env
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kairos:kairos123@localhost:5432/kairosai"
)

# Engine manages the actual connection pool to PostgreSQL
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# SessionLocal is a factory — call SessionLocal() to open a session
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Base class for future ORM models
Base = declarative_base()


def get_db():
    # Opens a session, yields it, closes it when done
    # Use this in every route that needs database access
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_query(sql: str, params: dict = {}):
    # Runs a raw SQL query and returns all rows as a list
    # Used for analytics queries that don't need the ORM
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        return result.fetchall()


def test_connection():
    # Call this on startup to verify the DB is reachable
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database connection successful.")
    except Exception as e:
        print(f"Database connection failed: {e}")